"""Two-layer CV extraction.

Layer 1 is :mod:`regexlayer` — deterministic, never wrong about an e-mail
address.  Layer 2 is Gemini under a strict JSON schema.  What comes back is then
put through three gates before anyone is allowed to score it:

* **validation** — shape, types, ISO dates, confidences in range.  A validation
  failure is re-prompted (at most ``max_retries`` times) with the validator's own
  error text appended, so the model is told exactly what it got wrong.
* **evidence verification** — every quote must be findable in the source text.
  An unverifiable quote zeroes that field's confidence.
* **cross-checks** — stated years vs. years computed from the work-history date
  ranges; a gap beyond tolerance raises a flag rather than picking a winner.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import regexlayer
from .evidence import EvidenceCheck, verify_quote
from .gemini_client import GeminiClient, MalformedResponse
from .prompts import (
    EXTRACTION_SCHEMA,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_extraction_prompt,
)
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_TOP_LEVEL = ("skills", "education", "work_history")


class ValidationError(ValueError):
    """The model's JSON did not satisfy the extraction contract."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_payload(payload: Any) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"top level must be a JSON object, got {type(payload).__name__}"]

    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            problems.append(f"missing required key '{key}'")
        elif not isinstance(payload[key], list):
            problems.append(f"'{key}' must be an array")

    for key in ("full_name", "email", "phone"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            problems.append(f"'{key}' must be a string or null")

    sy = payload.get("stated_years_experience")
    if sy is not None and not _is_number(sy):
        problems.append("'stated_years_experience' must be a number or null")
    elif _is_number(sy) and not (0 <= float(sy) <= 60):
        problems.append("'stated_years_experience' must be between 0 and 60")

    for i, skill in enumerate(payload.get("skills", []) or []):
        prefix = f"skills[{i}]"
        if not isinstance(skill, dict):
            problems.append(f"{prefix} must be an object")
            continue
        if not isinstance(skill.get("name"), str) or not skill.get("name", "").strip():
            problems.append(f"{prefix}.name must be a non-empty string")
        problems.extend(_check_confidence(skill.get("confidence"), prefix))

    for i, edu in enumerate(payload.get("education", []) or []):
        prefix = f"education[{i}]"
        if not isinstance(edu, dict):
            problems.append(f"{prefix} must be an object")
            continue
        year = edu.get("graduation_year")
        if year is not None and (not isinstance(year, int) or not (1900 <= year <= 2100)):
            problems.append(f"{prefix}.graduation_year must be an integer year or null")
        problems.extend(_check_confidence(edu.get("confidence"), prefix))

    for i, job in enumerate(payload.get("work_history", []) or []):
        prefix = f"work_history[{i}]"
        if not isinstance(job, dict):
            problems.append(f"{prefix} must be an object")
            continue
        for key in ("from_date", "to_date"):
            value = job.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not ISO_DATE_RE.match(value):
                problems.append(f"{prefix}.{key} must be an ISO date YYYY-MM-DD or null")
            else:
                try:
                    dt.date.fromisoformat(value)
                except ValueError:
                    problems.append(f"{prefix}.{key} is not a real calendar date")
        start, end = job.get("from_date"), job.get("to_date")
        if (
            isinstance(start, str)
            and isinstance(end, str)
            and ISO_DATE_RE.match(start)
            and ISO_DATE_RE.match(end)
        ):
            try:
                if dt.date.fromisoformat(end) < dt.date.fromisoformat(start):
                    problems.append(f"{prefix}.to_date is before from_date")
            except ValueError:
                # An impossible calendar date was already reported above; do not
                # report the same field twice.
                ...
        highlights = job.get("highlights")
        if highlights is not None and not isinstance(highlights, list):
            problems.append(f"{prefix}.highlights must be an array")
        problems.extend(_check_confidence(job.get("confidence"), prefix))

    return problems


def _check_confidence(value: Any, prefix: str) -> list[str]:
    if value is None:
        return [f"{prefix}.confidence is required"]
    if not _is_number(value):
        return [f"{prefix}.confidence must be a number"]
    if not (0.0 <= float(value) <= 1.0):
        return [f"{prefix}.confidence must be within [0,1]"]
    return []


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    payload: dict[str, Any]
    field_confidence: dict[str, float]
    evidence: dict[str, list[dict[str, Any]]]
    stated_years: float | None
    computed_years: float | None
    years_conflict: bool
    retries: int
    warnings: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model: str = ""

    @property
    def mean_confidence(self) -> float:
        values = list(self.field_confidence.values())
        return round(sum(values) / len(values), 4) if values else 0.0

    @property
    def evidence_verification_rate(self) -> float:
        checks = [c for group in self.evidence.values() for c in group]
        if not checks:
            return 0.0
        return round(sum(1.0 for c in checks if c["verified"]) / len(checks), 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "field_confidence": self.field_confidence,
            "evidence": self.evidence,
            "stated_years": self.stated_years,
            "computed_years": self.computed_years,
            "years_conflict": self.years_conflict,
            "retries": self.retries,
            "warnings": self.warnings,
            "mean_confidence": self.mean_confidence,
            "evidence_verification_rate": self.evidence_verification_rate,
        }


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------


class Extractor:
    def __init__(
        self,
        gemini: GeminiClient,
        *,
        taxonomy: Taxonomy | None = None,
        max_retries: int = 2,
        evidence_threshold: float = 0.8,
        years_tolerance: float = 1.5,
        model: str | None = None,
    ) -> None:
        self.gemini = gemini
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self.max_retries = max_retries
        self.evidence_threshold = evidence_threshold
        self.years_tolerance = years_tolerance
        self.model = model

    def extract(self, cv_text: str) -> ExtractionResult:
        findings = regexlayer.run(cv_text)
        payload, retries, warnings = self._call_with_retry(cv_text, findings)

        payload = self._merge_regex_layer(payload, findings)
        evidence, field_confidence = self._verify_and_score(payload, cv_text)
        payload = self._normalise_skills(payload)

        stated = payload.get("stated_years_experience")
        stated = float(stated) if _is_number(stated) else findings.stated_years
        computed = self._computed_years(payload, findings)
        conflict = (
            stated is not None
            and computed is not None
            and abs(float(stated) - float(computed)) > self.years_tolerance
        )
        if conflict:
            warnings.append("stated_vs_computed_years_conflict")
        if not payload.get("work_history"):
            warnings.append("no_work_history")
        elif all(not j.get("from_date") for j in payload["work_history"]):
            warnings.append("no_dates_in_work_history")

        return ExtractionResult(
            payload=payload,
            field_confidence=field_confidence,
            evidence=evidence,
            stated_years=stated,
            computed_years=computed,
            years_conflict=conflict,
            retries=retries,
            warnings=warnings,
            model=self.model or self.gemini.models.extraction,
        )

    # -- Gemini call + bounded retry ---------------------------------------
    def _call_with_retry(
        self, cv_text: str, findings: regexlayer.RegexFindings
    ) -> tuple[dict[str, Any], int, list[str]]:
        prompt = build_extraction_prompt(cv_text, regex_hints=findings.as_hints())
        warnings: list[str] = []
        last_problems: list[str] = []
        for attempt in range(self.max_retries + 1):
            try:
                payload = self.gemini.generate_structured(
                    prompt, EXTRACTION_SCHEMA, model=self.model
                )
            except MalformedResponse as exc:
                last_problems = [str(exc)]
                warnings.append(f"attempt_{attempt + 1}_malformed_json")
                prompt = _repair_prompt(prompt, last_problems)
                continue
            problems = validate_payload(payload)
            if not problems:
                return payload, attempt, warnings
            last_problems = problems
            warnings.append(f"attempt_{attempt + 1}_invalid: {problems[0]}")
            prompt = _repair_prompt(prompt, problems)

        # Every attempt failed. Fall back to the deterministic layer rather than
        # discarding the candidate; the empty payload will route to review.
        warnings.append("extraction_failed_after_retries")
        logger.warning("extraction failed after %d retries: %s", self.max_retries, last_problems)
        return _empty_payload(), self.max_retries, warnings

    # -- post-processing ----------------------------------------------------
    def _merge_regex_layer(
        self, payload: dict[str, Any], findings: regexlayer.RegexFindings
    ) -> dict[str, Any]:
        """The deterministic layer wins on contact details."""
        merged = dict(payload)
        if findings.emails:
            merged["email"] = findings.emails[0]
        if findings.phones:
            merged["phone"] = findings.phones[0]
        merged.setdefault("languages", [])
        merged["urls"] = findings.urls
        if merged.get("stated_years_experience") is None and findings.stated_years is not None:
            merged["stated_years_experience"] = findings.stated_years
        for key in REQUIRED_TOP_LEVEL:
            merged.setdefault(key, [])
        return merged

    def _verify_and_score(
        self, payload: dict[str, Any], cv_text: str
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
        evidence: dict[str, list[dict[str, Any]]] = {}
        confidences: dict[str, float] = {}
        for group in ("skills", "education", "work_history"):
            checks: list[dict[str, Any]] = []
            for index, item in enumerate(payload.get(group, []) or []):
                if not isinstance(item, dict):
                    continue
                check: EvidenceCheck = verify_quote(
                    item.get("evidence_quote"), cv_text, threshold=self.evidence_threshold
                )
                raw = float(item.get("confidence") or 0.0)
                effective = raw if check.verified else 0.0
                item["confidence"] = effective
                item["evidence_verified"] = check.verified
                item["evidence_ratio"] = round(check.ratio, 4)
                label = _item_label(group, index, item)
                confidences[label] = effective
                checks.append({"field": label, **check.as_dict()})
            evidence[group] = checks
        return evidence, confidences

    def _normalise_skills(self, payload: dict[str, Any]) -> dict[str, Any]:
        for skill in payload.get("skills", []) or []:
            if isinstance(skill, dict) and skill.get("name"):
                skill["canonical"] = self.taxonomy.normalise(str(skill["name"]))
                skill["category"] = self.taxonomy.category_of(skill["canonical"])
        return payload

    def _computed_years(
        self, payload: dict[str, Any], findings: regexlayer.RegexFindings
    ) -> float | None:
        ranges: list[regexlayer.DateRange] = []
        for job in payload.get("work_history", []) or []:
            if not isinstance(job, dict):
                continue
            start = _parse_iso(job.get("from_date"))
            if start is None:
                continue
            ranges.append(
                regexlayer.DateRange(start, _parse_iso(job.get("to_date")), job.get("evidence_quote") or "")
            )
        if not ranges:
            return findings.computed_years
        return regexlayer.computed_years_from_ranges(ranges)


def _parse_iso(value: Any) -> dt.date | None:
    if not isinstance(value, str) or not ISO_DATE_RE.match(value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _item_label(group: str, index: int, item: dict[str, Any]) -> str:
    if group == "skills":
        return f"skill:{item.get('name', index)}"
    if group == "education":
        return f"education:{item.get('degree', index)}:{index}"
    return f"work:{item.get('title', index)}:{index}"


def _empty_payload() -> dict[str, Any]:
    return {
        "full_name": None,
        "email": None,
        "phone": None,
        "stated_years_experience": None,
        "skills": [],
        "education": [],
        "work_history": [],
        "languages": [],
    }


def _repair_prompt(prompt: str, problems: list[str]) -> str:
    """Re-prompt with the validator's own complaint appended."""
    listing = "\n".join(f"- {p}" for p in problems[:8])
    return (
        f"{prompt}\n\n"
        "YOUR PREVIOUS RESPONSE WAS REJECTED BY THE SCHEMA VALIDATOR:\n"
        f"{listing}\n"
        "Return corrected JSON that fixes exactly these problems. Do not invent "
        "values to satisfy the validator — use null where the document is silent."
    )
