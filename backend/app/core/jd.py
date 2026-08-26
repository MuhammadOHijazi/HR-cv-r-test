"""Job-description structuring, versioning and approval.

Gemini turns free text into a structured spec; a human reviews, edits and
approves it.  Screening refuses to run against an unapproved version.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .audit import record as audit_record
from .gemini_client import GeminiClient
from .prompts import JD_PROMPT_VERSION, JD_SCHEMA, build_jd_prompt
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy
from ..models import JDVersion, Job

DIMENSIONS = (
    "must_have_skills",
    "preferred_skills",
    "experience_years",
    "similar_experience",
    "education",
)
VALID_DEGREES = ("diploma", "bachelor", "master", "phd")
DEGREE_RANK = {"diploma": 1, "bachelor": 2, "master": 3, "phd": 4}


class JDNotApproved(RuntimeError):
    """Screening was attempted against a JD version nobody has approved."""


def normalise_structured(
    raw: dict[str, Any], *, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Coerce a model-produced JD into the canonical shape the scorers expect."""
    tax = taxonomy or DEFAULT_TAXONOMY
    data = dict(raw or {})

    def _skills(key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in data.get(key, []) or []:
            name = item.get("skill") if isinstance(item, dict) else item
            if not name or not str(name).strip():
                continue
            canonical = tax.normalise(str(name))
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(
                {
                    "skill": str(name).strip(),
                    "canonical": canonical,
                    "importance": (item.get("importance") if isinstance(item, dict) else None)
                    or ("must" if key == "must_have" else "nice"),
                }
            )
        return out

    weights = {d: float((data.get("weights") or {}).get(d, 0.0) or 0.0) for d in DIMENSIONS}
    total = sum(weights.values())
    if total <= 0:
        weights = {
            "must_have_skills": 0.30,
            "preferred_skills": 0.15,
            "experience_years": 0.20,
            "similar_experience": 0.25,
            "education": 0.10,
        }
    else:
        weights = {k: round(v / total, 6) for k, v in weights.items()}

    thresholds = dict(data.get("thresholds") or {})
    min_years = thresholds.get("min_years_experience")
    try:
        min_years = max(0.0, float(min_years)) if min_years is not None else 0.0
    except (TypeError, ValueError):
        min_years = 0.0
    degree = thresholds.get("required_degree")
    degree = degree.strip().lower() if isinstance(degree, str) and degree.strip() else None
    if degree not in VALID_DEGREES:
        degree = None
    certs = [
        str(c).strip()
        for c in (thresholds.get("required_certifications") or [])
        if str(c).strip()
    ]

    responsibilities = [
        str(r).strip() for r in (data.get("responsibilities") or []) if str(r).strip()
    ]

    return {
        "title": str(data.get("title") or "Untitled role").strip()[:200],
        "must_have": _skills("must_have"),
        "nice_to_have": _skills("nice_to_have"),
        "responsibilities": responsibilities,
        "weights": weights,
        "thresholds": {
            "min_years_experience": min_years,
            "required_degree": degree,
            "required_certifications": certs,
        },
    }


class JDService:
    def __init__(self, session: Session, gemini: GeminiClient, *, taxonomy: Taxonomy | None = None):
        self.session = session
        self.gemini = gemini
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

    def structure(self, job: Job, *, actor: str = "system") -> JDVersion:
        """Ask Gemini to structure the raw JD text into a new, unapproved version."""
        raw = self.gemini.generate_structured(build_jd_prompt(job.raw_jd_text), JD_SCHEMA)
        structured = normalise_structured(raw, taxonomy=self.taxonomy)
        version = self._next_version(job.id)
        row = JDVersion(
            job_id=job.id,
            version=version,
            structured_json=json.dumps(structured, ensure_ascii=False),
            approved=False,
            source_model=self.gemini.models.extraction,
            prompt_version=JD_PROMPT_VERSION,
        )
        self.session.add(row)
        self.session.flush()
        audit_record(
            self.session,
            "jd_version",
            row.id,
            "structured",
            actor=actor,
            after={"version": version, "structured": structured},
        )
        self.session.commit()
        return row

    def edit(self, job: Job, base: JDVersion, structured: dict[str, Any], *, actor: str) -> JDVersion:
        """An edit always creates a NEW unapproved version — history is immutable."""
        cleaned = normalise_structured(structured, taxonomy=self.taxonomy)
        row = JDVersion(
            job_id=job.id,
            version=self._next_version(job.id),
            structured_json=json.dumps(cleaned, ensure_ascii=False),
            approved=False,
            source_model=base.source_model,
            prompt_version=base.prompt_version,
        )
        self.session.add(row)
        self.session.flush()
        audit_record(
            self.session,
            "jd_version",
            row.id,
            "edited",
            actor=actor,
            before=json.loads(base.structured_json or "{}"),
            after=cleaned,
        )
        self.session.commit()
        return row

    def approve(self, job: Job, version: JDVersion, *, actor: str) -> JDVersion:
        before = {"approved": version.approved}
        version.approved = True
        version.approved_by = actor
        version.approved_at = dt.datetime.now(dt.timezone.utc)
        job.active_jd_version_id = version.id
        job.status = "ready"
        audit_record(
            self.session,
            "jd_version",
            version.id,
            "approved",
            actor=actor,
            before=before,
            after={"approved": True},
        )
        self.session.commit()
        return version

    def active_version(self, job: Job) -> JDVersion:
        if job.active_jd_version_id is None:
            raise JDNotApproved(f"job {job.id} has no approved JD version")
        version = self.session.get(JDVersion, job.active_jd_version_id)
        if version is None or not version.approved:
            raise JDNotApproved(f"job {job.id} has no approved JD version")
        return version

    def _next_version(self, job_id: int) -> int:
        existing = self.session.scalars(
            select(JDVersion.version).where(JDVersion.job_id == job_id)
        ).all()
        return (max(existing) + 1) if existing else 1


def load_structured(version: JDVersion) -> dict[str, Any]:
    return json.loads(version.structured_json or "{}")
