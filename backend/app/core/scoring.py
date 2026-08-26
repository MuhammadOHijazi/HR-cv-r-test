"""The hybrid scoring engine: rules gate, semantic similarity, LLM judge, merge.

The three scorers are independent and their raw outputs are all stored.  The
merge is a weighted sum using the *approved* JD's weights.  Years of experience
contribute through a saturation curve, so the tenth year is worth far less than
the third.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from .evidence import verify_quote
from .gemini_client import GeminiClient
from .jd import DEGREE_RANK
from .masking import mask_record
from .prompts import JUDGE_PROMPT_VERSION, JUDGE_SCHEMA, build_judge_prompt
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy
from ..models import EmbeddingCache

logger = logging.getLogger(__name__)

DIMENSIONS = (
    "must_have_skills",
    "preferred_skills",
    "experience_years",
    "similar_experience",
    "education",
)

# k chosen so that years == required maps to 0.75 of the curve's range.
_SATURATION_K = -math.log(0.25)


# ---------------------------------------------------------------------------
# Years-of-experience saturation curve
# ---------------------------------------------------------------------------


def years_score(years: float | None, required: float) -> float:
    """Diminishing-returns score in [0,1] for years of experience.

    ``years == required`` -> 0.75, ``years == 2 * required`` -> ~0.94, and the
    curve keeps rising towards but never reaching 1.0.  With no requirement the
    curve is anchored at three years so more experience still scores higher.
    """
    if years is None:
        return 0.0
    y = max(0.0, float(years))
    anchor = float(required) if required and required > 0 else 3.0
    return round(1.0 - math.exp(-_SATURATION_K * y / anchor), 6)


# ---------------------------------------------------------------------------
# Scorer 1 — deterministic rules gate
# ---------------------------------------------------------------------------


@dataclass
class RulesResult:
    passed: bool
    hard_failures: list[dict[str, Any]] = field(default_factory=list)
    soft_failures: list[dict[str, Any]] = field(default_factory=list)
    must_have_coverage: float = 0.0
    matched_must_have: list[str] = field(default_factory=list)
    missing_must_have: list[str] = field(default_factory=list)
    years_component: float = 0.0
    education_component: float = 0.0

    @property
    def has_hard_failure(self) -> bool:
        return bool(self.hard_failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "hard_failures": self.hard_failures,
            "soft_failures": self.soft_failures,
            "must_have_coverage": round(self.must_have_coverage, 4),
            "matched_must_have": self.matched_must_have,
            "missing_must_have": self.missing_must_have,
            "years_component": round(self.years_component, 4),
            "education_component": round(self.education_component, 4),
        }


class RulesGate:
    """Pass/fail on must-haves, split by the confidence of the evidence behind them."""

    def __init__(
        self,
        *,
        high_confidence_threshold: float = 0.7,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.high_confidence_threshold = high_confidence_threshold
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

    def evaluate(
        self,
        extraction: dict[str, Any],
        jd: dict[str, Any],
        *,
        field_confidence: dict[str, float] | None = None,
    ) -> RulesResult:
        payload = extraction.get("payload", extraction)
        confidences = field_confidence or extraction.get("field_confidence") or {}
        thresholds = jd.get("thresholds", {})
        result = RulesResult(passed=True)

        # -- must-have skills -----------------------------------------------
        cand_skills = {
            s.get("canonical") or self.taxonomy.normalise(str(s.get("name", ""))): float(
                s.get("confidence") or 0.0
            )
            for s in payload.get("skills", []) or []
            if isinstance(s, dict) and s.get("name")
        }
        must = [m.get("canonical") or self.taxonomy.normalise(m.get("skill", "")) for m in jd.get("must_have", [])]
        must = [m for m in must if m]
        for skill in must:
            if skill in cand_skills:
                result.matched_must_have.append(skill)
                if cand_skills[skill] < self.high_confidence_threshold:
                    result.soft_failures.append(
                        {
                            "rule": "must_have_skill_low_confidence",
                            "field": skill,
                            "confidence": cand_skills[skill],
                        }
                    )
            else:
                result.missing_must_have.append(skill)
                bucket = (
                    result.hard_failures
                    if _extraction_is_reliable(extraction)
                    else result.soft_failures
                )
                bucket.append({"rule": "missing_must_have_skill", "field": skill})
        result.must_have_coverage = (
            len(result.matched_must_have) / len(must) if must else 1.0
        )

        # -- minimum years ---------------------------------------------------
        min_years = float(thresholds.get("min_years_experience") or 0.0)
        years = _effective_years(extraction)
        result.years_component = years_score(years, min_years)
        if min_years > 0:
            if years is None:
                result.soft_failures.append(
                    {"rule": "min_years_unknown", "required": min_years, "actual": None}
                )
            elif years + 1e-9 < min_years:
                entry = {"rule": "min_years_not_met", "required": min_years, "actual": years}
                if _years_are_reliable(extraction):
                    result.hard_failures.append(entry)
                else:
                    result.soft_failures.append(entry)

        # -- required degree -------------------------------------------------
        required_degree = thresholds.get("required_degree")
        best_degree, degree_conf = _best_degree(payload)
        result.education_component = _education_component(best_degree, required_degree)
        if required_degree:
            required_rank = DEGREE_RANK.get(required_degree, 0)
            actual_rank = DEGREE_RANK.get(best_degree or "", 0)
            if actual_rank < required_rank:
                entry = {
                    "rule": "required_degree_not_met",
                    "required": required_degree,
                    "actual": best_degree,
                }
                if best_degree is not None and degree_conf >= self.high_confidence_threshold:
                    result.hard_failures.append(entry)
                else:
                    result.soft_failures.append(entry)

        # -- required certifications ----------------------------------------
        certs_required = [c.lower() for c in thresholds.get("required_certifications", [])]
        if certs_required:
            blob = _text_blob(payload).lower()
            for cert in certs_required:
                if cert not in blob:
                    result.hard_failures.append({"rule": "missing_certification", "field": cert})

        result.passed = not result.hard_failures and not result.soft_failures
        return result


def _extraction_is_reliable(extraction: dict[str, Any]) -> bool:
    """Was the source good enough that an absence really means absence?"""
    quality = float(extraction.get("source_quality", 1.0) or 0.0)
    rate = float(extraction.get("evidence_verification_rate", 1.0) or 0.0)
    mean_conf = float(extraction.get("mean_confidence", 1.0) or 0.0)
    return quality >= 0.55 and rate >= 0.6 and mean_conf >= 0.5


def _years_are_reliable(extraction: dict[str, Any]) -> bool:
    if extraction.get("years_conflict"):
        return False
    return _extraction_is_reliable(extraction)


def _effective_years(extraction: dict[str, Any]) -> float | None:
    """Prefer computed years; fall back to stated; conflicts prefer the lower."""
    stated = extraction.get("stated_years")
    computed = extraction.get("computed_years")
    if computed is not None and stated is not None:
        return min(float(computed), float(stated)) if extraction.get("years_conflict") else float(computed)
    if computed is not None:
        return float(computed)
    if stated is not None:
        return float(stated)
    return None


def _best_degree(payload: dict[str, Any]) -> tuple[str | None, float]:
    best: str | None = None
    best_conf = 0.0
    for edu in payload.get("education", []) or []:
        if not isinstance(edu, dict):
            continue
        degree = (edu.get("degree") or "").strip().lower()
        if degree not in DEGREE_RANK:
            continue
        if best is None or DEGREE_RANK[degree] > DEGREE_RANK[best]:
            best = degree
            best_conf = float(edu.get("confidence") or 0.0)
    return best, best_conf


def _education_component(actual: str | None, required: str | None) -> float:
    if actual is None:
        return 0.0
    actual_rank = DEGREE_RANK.get(actual, 0)
    if not required:
        return min(1.0, actual_rank / 4.0 + 0.25)
    required_rank = DEGREE_RANK.get(required, 0)
    if actual_rank >= required_rank:
        return 1.0
    return max(0.0, actual_rank / max(required_rank, 1)) * 0.6


def _text_blob(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for skill in payload.get("skills", []) or []:
        if isinstance(skill, dict):
            parts.append(str(skill.get("name", "")))
            parts.append(str(skill.get("evidence_quote") or ""))
    for job in payload.get("work_history", []) or []:
        if isinstance(job, dict):
            parts.append(str(job.get("title") or ""))
            parts.append(str(job.get("company") or ""))
            parts.extend(str(h) for h in (job.get("highlights") or []))
    for edu in payload.get("education", []) or []:
        if isinstance(edu, dict):
            parts.extend(str(edu.get(k) or "") for k in ("degree", "field", "institution"))
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Scorer 2 — semantic similarity
# ---------------------------------------------------------------------------


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def content_hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingStore:
    """DB-backed embedding cache keyed by (model, content) hash."""

    def __init__(self, session: Session, gemini: GeminiClient, *, model: str | None = None):
        self.session = session
        self.gemini = gemini
        self.model = model or gemini.models.embedding
        self.api_calls = 0

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        hashes = [content_hash(t, self.model) for t in texts]
        cached: dict[str, list[float]] = {}
        if hashes:
            rows = self.session.scalars(
                select(EmbeddingCache).where(EmbeddingCache.content_hash.in_(set(hashes)))
            ).all()
            cached = {r.content_hash: unpack_vector(r.vector) for r in rows}

        # Deduplicate within the batch too: the same highlight can appear twice.
        missing: dict[str, int] = {}
        for i, h in enumerate(hashes):
            if h not in cached and h not in missing:
                missing[h] = i
        if missing:
            order = list(missing.items())
            fresh = self.gemini.embed([texts[i] for _, i in order], model=self.model)
            self.api_calls += 1
            for offset, (h, _) in enumerate(order):
                vector = list(fresh[offset]) if offset < len(fresh) else []
                cached[h] = vector
                self.session.add(
                    EmbeddingCache(
                        content_hash=h,
                        model=self.model,
                        dim=len(vector),
                        vector=pack_vector(vector),
                    )
                )
            self.session.commit()
        return [cached[h] for h in hashes]


@dataclass
class SemanticResult:
    score: float
    pairs: list[dict[str, Any]] = field(default_factory=list)
    responsibility_count: int = 0
    highlight_count: int = 0
    baseline: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "pairs": self.pairs,
            "responsibility_count": self.responsibility_count,
            "highlight_count": self.highlight_count,
            "baseline": round(self.baseline, 4),
        }


# A deliberately unrelated sentence. Embedding it alongside the job's
# responsibilities tells us what "no relationship at all" scores on whichever
# embedding model is configured, which is the floor the calibration needs.
CONTROL_TEXT = (
    "A recipe for lemon sorbet: freeze the syrup, churn it slowly, "
    "and serve the sorbet in chilled glasses."
)


def contrast(best: float, baseline: float) -> float:
    """Rescale a similarity against a known "unrelated" floor.

    Raw cosine is not comparable across embedding models — genuinely related
    short texts sit anywhere from 0.3 to 0.9 depending on the provider.  The
    baseline is measured, not assumed: it is how strongly the job's own
    responsibilities match an unrelated control sentence.  Anything at or below
    that floor is noise; the distance above it is signal.

    Calibrating against a fixed control rather than against the candidate's own
    similarity grid matters when a CV has only one or two highlights — there,
    the grid's own average is dominated by the good matches and would cancel out
    the very signal being measured.
    """
    ceiling = 1.0 - baseline
    if ceiling <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (best - baseline) / ceiling))


class SemanticScorer:
    """Cosine similarity between JD responsibilities and CV highlights.

    The score is the *coverage* of the job's responsibilities — the mean over
    every responsibility of how well the candidate's best highlight matches it —
    not the mean of the best few.  Averaging only the top matches lets a single
    lucky pairing carry an otherwise unrelated CV, which is exactly the failure
    mode this scorer exists to avoid.
    """

    def __init__(self, store: EmbeddingStore):
        self.store = store

    def score(self, extraction: dict[str, Any], jd: dict[str, Any]) -> SemanticResult:
        payload = extraction.get("payload", extraction)
        responsibilities = [r for r in (jd.get("responsibilities") or []) if str(r).strip()]
        highlights: list[str] = []
        for job in payload.get("work_history", []) or []:
            if not isinstance(job, dict):
                continue
            for h in job.get("highlights") or []:
                if str(h).strip():
                    highlights.append(str(h).strip())
            title = job.get("title")
            if title:
                highlights.append(str(title))
        if not responsibilities or not highlights:
            return SemanticResult(0.0, [], len(responsibilities), len(highlights))

        vectors = self.store.embed_many(responsibilities + highlights + [CONTROL_TEXT])
        r_vecs = vectors[: len(responsibilities)]
        h_vecs = vectors[len(responsibilities) : -1]
        control_vec = vectors[-1]

        pairs: list[dict[str, Any]] = []
        bests: list[tuple[float, str]] = []
        controls: list[float] = []
        for r_vec in r_vecs:
            row = [max(0.0, cosine(r_vec, h_vec)) for h_vec in h_vecs]
            best_index = max(range(len(row)), key=row.__getitem__) if row else 0
            bests.append((row[best_index] if row else 0.0, highlights[best_index] if row else ""))
            controls.append(max(0.0, cosine(r_vec, control_vec)))

        baseline = sum(controls) / len(controls) if controls else 0.0

        calibrated: list[float] = []
        for r_text, (best, best_text) in zip(responsibilities, bests):
            value = contrast(best, baseline)
            calibrated.append(value)
            pairs.append(
                {
                    "responsibility": r_text,
                    "best_highlight": best_text,
                    "similarity": round(best, 4),
                    "calibrated": round(value, 4),
                }
            )
        score = sum(calibrated) / len(calibrated) if calibrated else 0.0
        return SemanticResult(
            score=round(min(1.0, max(0.0, score)), 6),
            pairs=pairs,
            responsibility_count=len(responsibilities),
            highlight_count=len(highlights),
            baseline=baseline,
        )


# ---------------------------------------------------------------------------
# Scorer 3 — LLM judge
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    dimensions: dict[str, dict[str, Any]]
    evidence_verified_rate: float
    prompt_version: str = JUDGE_PROMPT_VERSION

    @property
    def mean_score(self) -> float:
        scores = [float(d.get("score") or 0.0) for d in self.dimensions.values()]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "evidence_verified_rate": round(self.evidence_verified_rate, 4),
            "mean_score": self.mean_score,
            "prompt_version": self.prompt_version,
        }


class LLMJudge:
    """Rubric-scored judgement of an IDENTITY-MASKED candidate record."""

    JUDGE_DIMENSIONS = ("preferred_skills", "similar_experience", "education")

    def __init__(self, gemini: GeminiClient, *, evidence_threshold: float = 0.8):
        self.gemini = gemini
        self.evidence_threshold = evidence_threshold

    def score(self, extraction: dict[str, Any], jd: dict[str, Any]) -> JudgeResult:
        payload = extraction.get("payload", extraction)
        masked = mask_record(payload)
        prompt = build_judge_prompt(masked, jd)
        raw = self.gemini.judge(prompt, JUDGE_SCHEMA)
        dimensions_in = raw.get("dimensions", {}) if isinstance(raw, dict) else {}

        # The judge may only quote the masked record it was shown.
        import json as _json

        masked_text = _json.dumps(masked, ensure_ascii=False)
        verified = 0
        total = 0
        dimensions: dict[str, dict[str, Any]] = {}
        for dim in self.JUDGE_DIMENSIONS:
            entry = dimensions_in.get(dim) or {}
            score = _clamp_rubric(entry.get("score"))
            check = verify_quote(
                entry.get("evidence_quote"), masked_text, threshold=self.evidence_threshold
            )
            total += 1
            if check.verified:
                verified += 1
            elif score > 0:
                # A score with no traceable evidence is not allowed to stand.
                score = 0.0
            dimensions[dim] = {
                "score": score,
                "rubric_level": entry.get("rubric_level"),
                "evidence_quote": entry.get("evidence_quote"),
                "evidence_verified": check.verified,
                "evidence_ratio": round(check.ratio, 4),
                "rationale": entry.get("rationale", ""),
            }
        return JudgeResult(dimensions, verified / total if total else 0.0)


def _clamp_rubric(value: Any) -> float:
    """Snap a judge score onto the nearest allowed rubric level."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    raw = max(0.0, min(100.0, raw))
    return float(min((0.0, 40.0, 70.0, 100.0), key=lambda level: abs(level - raw)))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


@dataclass
class MergedScore:
    total: float
    dimensions: dict[str, dict[str, float]]
    weights: dict[str, float]
    disagreement: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "dimensions": self.dimensions,
            "weights": self.weights,
            "disagreement": round(self.disagreement, 4),
        }


def merge_scores(
    rules: RulesResult,
    semantic: SemanticResult,
    judge: JudgeResult,
    jd: dict[str, Any],
) -> MergedScore:
    """Weighted 0-100 merge using the approved JD weights."""
    weights = {d: float((jd.get("weights") or {}).get(d, 0.0)) for d in DIMENSIONS}
    total_weight = sum(weights.values()) or 1.0
    weights = {k: v / total_weight for k, v in weights.items()}

    judge_dims = judge.dimensions
    components = {
        "must_have_skills": rules.must_have_coverage,
        "preferred_skills": float(judge_dims.get("preferred_skills", {}).get("score", 0.0)) / 100.0,
        "experience_years": rules.years_component,
        "similar_experience": _blend_similar(semantic.score, judge_dims),
        "education": _blend_education(rules.education_component, judge_dims),
    }

    dimensions = {
        name: {
            "raw": round(value, 4),
            "weight": round(weights[name], 4),
            "contribution": round(value * weights[name] * 100.0, 4),
        }
        for name, value in components.items()
    }
    total = sum(d["contribution"] for d in dimensions.values())
    if rules.has_hard_failure:
        # A hard must-have failure caps the total; the router decides the bucket.
        total = min(total, 40.0)
    return MergedScore(
        total=round(max(0.0, min(100.0, total)), 4),
        dimensions=dimensions,
        weights=weights,
        disagreement=scorer_disagreement(rules, semantic, judge),
    )


def _blend_similar(semantic_score: float, judge_dims: dict[str, dict[str, Any]]) -> float:
    judge_similar = float(judge_dims.get("similar_experience", {}).get("score", 0.0)) / 100.0
    return 0.5 * semantic_score + 0.5 * judge_similar


def _blend_education(rules_education: float, judge_dims: dict[str, dict[str, Any]]) -> float:
    judge_education = float(judge_dims.get("education", {}).get("score", 0.0)) / 100.0
    return 0.6 * rules_education + 0.4 * judge_education


def scorer_disagreement(
    rules: RulesResult, semantic: SemanticResult, judge: JudgeResult
) -> float:
    """How far apart the scorers are *where they measure the same thing*.

    Comparing the scorers' overall verdicts would be meaningless: the rules gate
    measures hard requirements, the judge also weighs preferred skills, and a
    candidate who clears every must-have but few nice-to-haves would look like a
    disagreement when the scorers simply answer different questions.

    Only two quantities are genuinely measured twice, by independent methods:

    * *similar experience* — embedding similarity vs. the judge's rubric score
    * *education fit* — the deterministic degree comparison vs. the judge's view

    A gap on either of those is real disagreement, and that is what caps
    confidence.
    """
    judge_similar = float(judge.dimensions.get("similar_experience", {}).get("score", 0.0))
    judge_education = float(judge.dimensions.get("education", {}).get("score", 0.0))
    gaps = [
        abs(semantic.score * 100.0 - judge_similar),
        abs(rules.education_component * 100.0 - judge_education),
    ]
    return round(max(gaps), 4)
