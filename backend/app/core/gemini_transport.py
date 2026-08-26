"""Transport implementations for :mod:`gemini_client`.

``GenaiTransport`` is the only place the ``google-genai`` SDK is imported.
``MockTransport`` is a deterministic, content-derived stand-in used when
``MOCK_MODE=true`` (and by the end-to-end flow test) so that the whole system
runs without credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Sequence

from .gemini_client import NonRetryableError, TransportError
from .taxonomy import DEFAULT_TAXONOMY, SEED_TAXONOMY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real SDK transport
# ---------------------------------------------------------------------------


class GenaiTransport:
    """Adapts the official ``google-genai`` SDK to the ``Transport`` protocol."""

    def __init__(self) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise NonRetryableError(
                "google-genai is not installed; install it or run with MOCK_MODE=true"
            ) from exc

    def _client(self, api_key: str):
        from google import genai

        return genai.Client(api_key=api_key)

    def generate(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        images: Sequence[bytes] | None = None,
    ) -> str:
        from google.genai import types

        config: dict[str, Any] = {"temperature": temperature}
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        contents: list[Any] = [prompt]
        for image in images or []:
            contents.append(types.Part.from_bytes(data=image, mime_type="image/png"))
        try:
            response = self._client(api_key).models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
        except Exception as exc:  # pragma: no cover - network path
            raise _translate(exc) from exc
        return response.text or ""

    def embed(self, *, api_key: str, model: str, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client(api_key).models.embed_content(
                model=model, contents=list(texts)
            )
        except Exception as exc:  # pragma: no cover - network path
            raise _translate(exc) from exc
        return [list(e.values or []) for e in (response.embeddings or [])]


def _translate(exc: Exception) -> Exception:  # pragma: no cover - network path
    """Map an SDK exception onto a status-carrying :class:`TransportError`."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status is None:
        text = str(exc)
        match = re.search(r"\b(4\d\d|5\d\d)\b", text)
        if match:
            status = int(match.group(1))
        elif "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
            status = 429
    if status is None:
        return exc
    return TransportError(int(status), str(exc))


# ---------------------------------------------------------------------------
# Deterministic mock transport
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs|سنوات|سنة)", re.IGNORECASE)
_RANGE_RE = re.compile(r"(\d{4})\s*(?:-|–|—|to|حتى)\s*(\d{4}|present|current|الآن|حتى الآن)", re.I)

# The fake shares the application's single skills vocabulary rather than keeping
# a second one of its own — which is also what makes it work in Arabic.
_SKILL_HINTS = [entry.canonical for entry in SEED_TAXONOMY]
_SKILL_SURFACES: dict[str, list[str]] = {
    entry.canonical: [entry.canonical, *entry.aliases] for entry in SEED_TAXONOMY
}

_DEGREE_HINTS = {
    "phd": ["ph.d", "phd", "doctorate", "دكتوراه"],
    "master": ["master", "m.sc", "msc", "ماجستير"],
    "bachelor": ["bachelor", "b.sc", "bsc", "بكالوريوس", "b.eng"],
    "diploma": ["diploma", "دبلوم"],
}


class MockTransport:
    """A content-derived fake that always returns schema-valid JSON.

    It is *not* random: the same input text always produces the same output, so
    the end-to-end flow test can assert exact routing buckets.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []
        # Page-image hash -> transcription, so the vision fallback is
        # deterministic without anything resembling real OCR.
        self._vision_texts: dict[str, str] = {}

    def register_vision(self, images: Sequence[bytes], text: str) -> None:
        """Teach the fake what a set of page images transcribes to."""
        for image in images:
            self._vision_texts[hashlib.sha256(image).hexdigest()] = text

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _payload_block(prompt: str) -> str:
        """Pull the untrusted-data block out of a prompt built by ``prompts.py``."""
        match = re.search(r"<<<BEGIN_UNTRUSTED_DATA>>>(.*?)<<<END_UNTRUSTED_DATA>>>", prompt, re.S)
        body = match.group(1) if match else prompt
        # Drop the "[CV_TEXT]" style label the wrapper prepends.
        return re.sub(r"^\s*\[[A-Z_]+\]\n", "", body.lstrip("\n"))

    @staticmethod
    def _quote_for(text: str, needle: str) -> str | None:
        """Return the first line of ``text`` containing ``needle`` (verbatim)."""
        low = needle.lower()
        for line in text.splitlines():
            if low in line.lower() and line.strip():
                return line.strip()
        return None

    # -- Transport protocol -------------------------------------------------
    def generate(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        images: Sequence[bytes] | None = None,
    ) -> str:
        self.calls.append({"model": model, "kind": _kind_of(prompt), "images": len(images or [])})
        kind = _kind_of(prompt)
        if kind == "jd":
            return json.dumps(self._structure_jd(prompt))
        if kind == "judge":
            return json.dumps(self._judge(prompt))
        if kind == "vision":
            return json.dumps(self._vision(prompt, images or []))
        return json.dumps(self._extract(prompt))

    def embed(self, *, api_key: str, model: str, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        """A deterministic bag-of-concepts hash embedding.

        Tokens fold through the skills taxonomy before hashing, so the Arabic
        and English wordings of the same concept land on the same dimension.
        Cosine between two of these therefore tracks *semantic* overlap across
        languages — the property the semantic scorer needs — while staying
        completely deterministic.
        """
        vec = [0.0] * self.dim
        low = text.lower()
        # Taxonomy concepts dominate the vector; they are the part of the
        # meaning that survives translation.
        for skill in _SKILL_HINTS:
            if _skill_present(skill, low):
                h = int(hashlib.sha256(skill.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 3.0
                vec[(h >> 16) % self.dim] += 1.5
        for raw in re.findall(r"[\w؀-ۿ]+", low):
            token = DEFAULT_TAXONOMY.normalise(raw) or raw
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
            vec[(h >> 16) % self.dim] += 0.5
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    # -- response builders --------------------------------------------------
    def _structure_jd(self, prompt: str) -> dict[str, Any]:
        """Split the JD into its labelled sections and read each one."""
        text = self._payload_block(prompt)
        sections = _split_sections(text)
        must = _skills_in(sections.get("must", ""))
        nice = [s for s in _skills_in(sections.get("nice", "")) if s not in must]

        must_block = sections.get("must", "")
        min_years = 0.0
        m = _YEARS_RE.search(must_block or text)
        if m:
            min_years = float(m.group(1))
        degree = _degree_in(must_block or text)

        responsibilities = [
            line.strip("-• \t")
            for line in (sections.get("responsibilities", "")).splitlines()
            if line.strip().startswith(("-", "•")) and len(line.strip()) > 12
        ][:12]
        return {
            "title": next((ln.strip() for ln in text.splitlines() if ln.strip()), "Untitled role")[:120],
            "must_have": [{"skill": s, "importance": "must"} for s in must[:12]],
            "nice_to_have": [{"skill": s, "importance": "nice"} for s in nice[:12]],
            "responsibilities": responsibilities,
            "weights": {
                "must_have_skills": 0.30,
                "preferred_skills": 0.15,
                "experience_years": 0.20,
                "similar_experience": 0.25,
                "education": 0.10,
            },
            "thresholds": {
                "min_years_experience": min_years,
                "required_degree": degree,
                "required_certifications": [],
            },
        }

    def _extract(self, prompt: str) -> dict[str, Any]:
        text = self._payload_block(prompt)
        low = text.lower()
        email = _EMAIL_RE.search(text)
        phone = _PHONE_RE.search(text)
        name = None
        for line in text.splitlines():
            s = line.strip()
            if s and not _EMAIL_RE.search(s) and len(s.split()) <= 5 and len(s) > 3:
                name = s
                break
        skills = self._skills(text, low)
        stated = None
        m = _YEARS_RE.search(text)
        if m:
            stated = float(m.group(1))
        work = self._work_history(text)
        education: list[dict[str, Any]] = []
        for canonical, hints in _DEGREE_HINTS.items():
            for hint in hints:
                if hint in low:
                    quote = self._quote_for(text, hint)
                    if quote:
                        education.append(
                            {
                                "degree": canonical,
                                "field": None,
                                "institution": None,
                                "graduation_year": None,
                                "evidence_quote": quote,
                                "confidence": 0.85,
                            }
                        )
                    break
            if education:
                break
        return {
            "full_name": name,
            "email": email.group(0) if email else None,
            "phone": phone.group(0).strip() if phone else None,
            "stated_years_experience": stated,
            "skills": skills,
            "education": education,
            "work_history": work,
            "languages": [],
        }

    def _skills(self, text: str, low: str) -> list[dict[str, Any]]:
        """Skills named anywhere in the document, plus everything the CV lists.

        A real extractor is not limited to a fixed vocabulary: whatever the
        candidate writes under "Skills" is a skill, known to the taxonomy or
        not.  Taking both sources keeps unusual disciplines (design, hospitality)
        from silently extracting to nothing.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(name: str, quote: str | None) -> None:
            key = name.strip().lower()
            if not key or key in seen or not quote:
                return
            seen.add(key)
            out.append(
                {
                    "name": name.strip(),
                    "level": "advanced" if f"advanced {key}" in low else None,
                    "evidence_quote": quote,
                    "confidence": 0.9,
                }
            )

        for skill in _SKILL_HINTS:
            surface = _find_surface(skill, low)
            if surface is not None:
                add(surface, self._quote_for(text, surface))

        for line in _skills_section(text).splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 240:
                continue
            for item in re.split(r"[,،;|/]| - ", stripped):
                item = item.strip(" .•-\t")
                if 1 < len(item) <= 40:
                    add(item, stripped)
        return out

    def _work_history(self, text: str) -> list[dict[str, Any]]:
        """Group bullet lines under the dated role heading that precedes them.

        This mirrors what a real extractor does: a role's achievements are its
        highlights, and those highlights are what the semantic scorer compares
        against the job description's responsibilities.
        """
        lines = text.splitlines()
        work: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        in_experience = False
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            if upper and upper == stripped and len(stripped.split()) <= 3 and stripped.isascii():
                in_experience = "EXPERIENCE" in upper
            if any(w in stripped for w in ("الخبرة", "التعليم", "المهارات", "الملخص")):
                in_experience = "الخبرة" in stripped

            rm = _RANGE_RE.search(stripped)
            if rm and not stripped.startswith(("-", "•")):
                end_raw = rm.group(2).lower()
                current = {
                    "title": re.split(r"[,|(]", stripped)[0].strip("-• \t")[:120] or None,
                    "company": _company_of(stripped),
                    "from_date": f"{rm.group(1)}-01-01",
                    "to_date": None
                    if end_raw in {"present", "current", "الآن", "حتى الآن"}
                    else f"{rm.group(2)}-01-01",
                    "highlights": [],
                    "evidence_quote": stripped,
                    "confidence": 0.85,
                }
                work.append(current)
                continue
            if stripped.startswith(("-", "•")) and in_experience:
                highlight = stripped.lstrip("-• \t")
                if current is not None:
                    current["highlights"].append(highlight)
                else:
                    # An undated role: still worth extracting, with no dates.
                    current = {
                        "title": None,
                        "company": None,
                        "from_date": None,
                        "to_date": None,
                        "highlights": [highlight],
                        "evidence_quote": stripped,
                        "confidence": 0.6,
                    }
                    work.append(current)
                continue
            if in_experience and stripped and not stripped.startswith(("-", "•")):
                # A role heading with no dates at all.
                current = {
                    "title": re.split(r"[,|(]", stripped)[0].strip()[:120],
                    "company": _company_of(stripped),
                    "from_date": None,
                    "to_date": None,
                    "highlights": [],
                    "evidence_quote": stripped,
                    "confidence": 0.6,
                }
                work.append(current)
        return [w for w in work if w["highlights"] or w["from_date"]]

    def _judge(self, prompt: str) -> dict[str, Any]:
        """Score the masked record against the JD dimensions, deterministically."""
        text = self._payload_block(prompt)
        jd_match = re.search(r"<<<BEGIN_JOB_SPEC>>>(.*?)<<<END_JOB_SPEC>>>", prompt, re.S)
        jd_text = jd_match.group(1) if jd_match else ""
        try:
            spec = json.loads(jd_text)
        except json.JSONDecodeError:
            spec = {}
        cand_low = text.lower()

        dimensions = {
            "preferred_skills": self._judge_skills(spec, text, cand_low),
            "similar_experience": self._judge_experience(spec, text, cand_low),
            "education": self._judge_education(spec, text, cand_low),
        }
        return {"dimensions": dimensions}

    def _judge_skills(self, spec: dict[str, Any], text: str, cand_low: str) -> dict[str, Any]:
        wanted = [
            str(s.get("canonical") or s.get("skill", "")).lower()
            for s in (spec.get("nice_to_have") or [])
        ]
        wanted = [w for w in wanted if w]
        if not wanted:
            return _dim(0, None, "the job spec lists no preferred skills")
        hits = [(w, _find_surface(w, cand_low)) for w in wanted]
        hits = [(w, s) for w, s in hits if s]
        return _dim(
            _ratio_to_rubric(len(hits) / len(wanted)),
            self._quote_for(text, hits[0][1]) if hits else None,
            f"{len(hits)} of {len(wanted)} preferred skills appear in the record.",
        )

    def _judge_experience(self, spec: dict[str, Any], text: str, cand_low: str) -> dict[str, Any]:
        responsibilities = [str(r) for r in (spec.get("responsibilities") or [])]
        if not responsibilities:
            return _dim(0, None, "the job spec lists no responsibilities")
        cand_concepts = _concepts(text)
        cand_skills = _skill_concepts(text)
        matched: list[str] = []
        for resp in responsibilities:
            # Skill concepts survive translation; generic vocabulary does not,
            # so a responsibility that names skills is judged on those alone.
            skills = _skill_concepts(resp)
            if skills:
                overlap = len(skills & cand_skills) / len(skills)
            else:
                concepts = _concepts(resp)
                if not concepts:
                    continue
                overlap = len(concepts & cand_concepts) / len(concepts)
            if overlap >= 0.5:
                matched.append(resp)
        quote = None
        if matched:
            shared = sorted(_concepts(matched[0]) & cand_concepts)
            for concept in shared:
                surface = _find_surface(concept, cand_low) or concept
                quote = self._quote_for(text, surface)
                if quote:
                    break
        return _dim(
            _ratio_to_rubric(len(matched) / len(responsibilities)),
            quote,
            f"{len(matched)} of {len(responsibilities)} responsibilities are evidenced in the record.",
        )

    def _judge_education(self, spec: dict[str, Any], text: str, cand_low: str) -> dict[str, Any]:
        required = (spec.get("thresholds") or {}).get("required_degree")
        actual = _degree_in(cand_low)
        rank = {"diploma": 1, "bachelor": 2, "master": 3, "phd": 4}
        quote = None
        if actual:
            hint = next(h for h in _DEGREE_HINTS[actual] if h in cand_low)
            quote = self._quote_for(text, hint)
        if actual is None:
            return _dim(0, None, "no education entry found in the record")
        if not required:
            return _dim(70, quote, f"holds a {actual}; the job spec states no degree requirement")
        if rank.get(actual, 0) >= rank.get(required, 0):
            return _dim(100, quote, f"holds a {actual}, meeting the {required} requirement")
        return _dim(40, quote, f"holds a {actual}, below the {required} requirement")

    def _vision(self, prompt: str, images: Sequence[bytes]) -> dict[str, Any]:
        """Vision fallback: return the registered transcription for these pages.

        Unregistered pages transcribe to nothing, which is the honest mock-mode
        answer — and correctly drives the CV into the review queue with a
        low-source-quality flag rather than inventing content.
        """
        seen: list[str] = []
        for image in images:
            text = self._vision_texts.get(hashlib.sha256(image).hexdigest())
            if text and text not in seen:
                seen.append(text)
        return {"text": "\n".join(seen), "page_count": len(images)}


_SECTION_HEADERS = (
    ("must", re.compile(r"^\s*(must[- ]?haves?|required|requirements|essential)\s*:?\s*$", re.I)),
    ("nice", re.compile(r"^\s*(nice[- ]to[- ]haves?|preferred|bonus|desirable)\s*:?\s*$", re.I)),
    ("responsibilities", re.compile(r"^\s*(responsibilities|duties|what you.ll do)\s*:?\s*$", re.I)),
)


def _split_sections(text: str) -> dict[str, str]:
    """Group a job description's lines under their section headers."""
    sections: dict[str, list[str]] = {}
    current = "preamble"
    for line in (text or "").splitlines():
        matched = next((name for name, rx in _SECTION_HEADERS if rx.match(line)), None)
        if matched:
            current = matched
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def _surface_present(surface: str, low: str) -> bool:
    """Whole-token match so 'r' does not match 'requirements'."""
    return bool(re.search(rf"(?<![\w+#]){re.escape(surface)}(?![\w+#])", low))


def _find_surface(skill: str, low: str) -> str | None:
    """Return the surface form of ``skill`` present in ``low``, in any language."""
    for surface in _SKILL_SURFACES.get(skill, [skill]):
        if _surface_present(surface.lower(), low):
            return surface
    return None


def _skill_present(skill: str, low: str) -> bool:
    return _find_surface(skill, low) is not None


def _skills_in(block: str) -> list[str]:
    low = (block or "").lower()
    return [s for s in _SKILL_HINTS if _skill_present(s, low)]


def _degree_in(block: str) -> str | None:
    low = (block or "").lower()
    for canonical, hints in _DEGREE_HINTS.items():
        if any(h in low for h in hints):
            return canonical
    return None


def _kind_of(prompt: str) -> str:
    head = prompt[:400].lower()
    if "job description" in head and "structure" in head:
        return "jd"
    if "rubric" in head or "evaluator" in head:
        return "judge"
    if "page images" in head or "scanned" in head:
        return "vision"
    return "extract"


def _rubric_level(score: float) -> str:
    return {0: "none", 40: "weak", 70: "solid", 100: "strong"}.get(int(score), "weak")


def _dim(score: float, quote: str | None, rationale: str) -> dict[str, Any]:
    return {
        "score": score,
        "rubric_level": _rubric_level(score),
        "evidence_quote": quote,
        "rationale": rationale,
    }


def _ratio_to_rubric(ratio: float) -> float:
    """Snap a coverage ratio onto the four written rubric levels."""
    if ratio <= 0:
        return 0.0
    if ratio < 0.25:
        return 40.0
    if ratio < 0.6:
        return 70.0
    return 100.0


def _company_of(line: str) -> str | None:
    parts = [p.strip() for p in re.split(r"[,|]", line) if p.strip()]
    return parts[1] if len(parts) >= 2 else None


_SKILLS_HEADING = re.compile(r"^\s*(skills?|المهارات)\b.*$", re.I)
_ANY_HEADING = re.compile(
    r"^\s*(summary|experience|education|projects|languages|certifications|"
    r"الملخص|الخبرة|التعليم|اللغات|الشهادات)\b.*$",
    re.I,
)


def _skills_section(text: str) -> str:
    """The lines under a 'Skills' heading, in either language."""
    collected: list[str] = []
    inside = False
    for line in (text or "").splitlines():
        if _SKILLS_HEADING.match(line):
            inside = True
            continue
        if inside and _ANY_HEADING.match(line):
            break
        if inside:
            collected.append(line)
    return "\n".join(collected)


def _skill_concepts(text: str) -> set[str]:
    """Just the taxonomy concepts named in ``text``, in any language."""
    low = (text or "").lower()
    return {skill for skill in _SKILL_HINTS if _skill_present(skill, low)}


def _concepts(text: str) -> set[str]:
    """Language-independent concept set for a piece of text.

    Words fold through the skills taxonomy first, so the English "Kubernetes"
    and the Arabic "كوبرنيتس" reduce to the same concept.  This is what lets the
    fake behave like a genuinely multilingual model without any translation.
    """
    low = (text or "").lower()
    found: set[str] = set()
    for skill in _SKILL_HINTS:
        if _skill_present(skill, low):
            found.add(skill)
    for word in re.findall(r"[a-z؀-ۿ]{4,}", low):
        if word not in _STOPWORDS:
            found.add(DEFAULT_TAXONOMY.normalise(word) or word)
    return found


_STOPWORDS = {
    "and", "the", "for", "with", "from", "into", "that", "this", "their", "them",
    "across", "every", "over", "under", "about", "against", "between", "which",
    "your", "will", "must", "have", "been", "were", "than", "then", "also",
    "على", "من", "في", "الى", "إلى", "مع", "عن", "التي", "الذي", "هذا", "هذه",
}
