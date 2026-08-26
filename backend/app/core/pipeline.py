"""The screening pipeline: one place that turns a CV file into a routed result.

This is also what the "correct a field" review action re-runs, so a correction
re-scores and re-routes through exactly the same code path as the original run.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import injection, routing
from .audit import record as audit_record
from .confidence import assemble as assemble_confidence
from .extraction import Extractor
from .gemini_client import GeminiClient
from .jd import JDNotApproved, load_structured
from .prompts import JUDGE_PROMPT_VERSION, PROMPT_VERSION, SCHEMA_VERSION
from .routing import RoutingInput, Thresholds
from .scoring import (
    EmbeddingStore,
    LLMJudge,
    RulesGate,
    SemanticScorer,
    merge_scores,
)
from ..models import (
    CVFile,
    Extraction,
    JDVersion,
    Job,
    JobConfig,
    ReviewQueueEntry,
    ScreeningResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ScreeningOutcome:
    candidate_id: int
    cv_file_id: int
    score: float
    confidence: float
    routing: str
    flags: list[str]
    reasons: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cv_file_id": self.cv_file_id,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 4),
            "routing": self.routing,
            "flags": self.flags,
            "reasons": self.reasons,
        }


class ScreeningPipeline:
    def __init__(self, session: Session, gemini: GeminiClient, *, settings: Any = None):
        self.session = session
        self.gemini = gemini
        self.settings = settings
        self.extractor = Extractor(
            gemini,
            max_retries=int(getattr(settings, "extraction_max_retries", 2)),
            evidence_threshold=float(getattr(settings, "evidence_match_threshold", 0.8)),
            years_tolerance=float(getattr(settings, "years_conflict_tolerance", 1.5)),
        )
        self.rules = RulesGate(
            high_confidence_threshold=float(getattr(settings, "high_confidence_threshold", 0.7))
        )
        self.judge = LLMJudge(
            gemini, evidence_threshold=float(getattr(settings, "evidence_match_threshold", 0.8))
        )
        self.embeddings = EmbeddingStore(session, gemini)
        self.semantic = SemanticScorer(self.embeddings)

    # -- text access --------------------------------------------------------
    def source_text(self, cv_file: CVFile) -> str:
        path = Path(cv_file.text_path)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    # -- extraction (cached per CV file) ------------------------------------
    def ensure_extraction(self, cv_file: CVFile, *, force: bool = False) -> Extraction:
        existing = self.session.scalar(
            select(Extraction).where(Extraction.cv_file_id == cv_file.id)
        )
        if existing is not None and not force:
            return existing

        text = self.source_text(cv_file)
        result = self.extractor.extract(text)
        row = existing or Extraction(cv_file_id=cv_file.id)
        row.schema_version = SCHEMA_VERSION
        row.prompt_version = PROMPT_VERSION
        row.model = result.model
        row.payload_json = json.dumps(result.payload, ensure_ascii=False, default=str)
        row.field_confidence_json = json.dumps(result.field_confidence)
        row.evidence_json = json.dumps(
            {"groups": result.evidence, "warnings": result.warnings}, ensure_ascii=False, default=str
        )
        row.stated_years = result.stated_years
        row.computed_years = result.computed_years
        row.years_conflict = result.years_conflict
        row.retries = result.retries
        if existing is None:
            self.session.add(row)
        self.session.commit()
        return row

    def extraction_view(self, extraction: Extraction, cv_file: CVFile) -> dict[str, Any]:
        """The dict shape the scorers consume."""
        payload = json.loads(extraction.payload_json or "{}")
        field_conf = json.loads(extraction.field_confidence_json or "{}")
        evidence = json.loads(extraction.evidence_json or "{}")
        checks = [c for group in (evidence.get("groups") or {}).values() for c in group]
        verified_rate = (
            sum(1.0 for c in checks if c.get("verified")) / len(checks) if checks else 0.0
        )
        values = list(field_conf.values())
        return {
            "payload": payload,
            "field_confidence": field_conf,
            "mean_confidence": round(sum(values) / len(values), 4) if values else 0.0,
            "evidence_verification_rate": round(verified_rate, 4),
            "evidence": evidence.get("groups", {}),
            "warnings": evidence.get("warnings", []),
            "stated_years": extraction.stated_years,
            "computed_years": extraction.computed_years,
            "years_conflict": extraction.years_conflict,
            "source_quality": cv_file.source_quality,
            "is_scanned": cv_file.is_scanned,
        }

    # -- scoring ------------------------------------------------------------
    def screen_one(
        self,
        job: Job,
        jd_version: JDVersion,
        cv_file: CVFile,
        *,
        force_extract: bool = False,
        actor: str = "system",
    ) -> ScreeningResult:
        if not jd_version.approved:
            raise JDNotApproved(f"JD version {jd_version.id} is not approved")

        extraction_row = self.ensure_extraction(cv_file, force=force_extract)
        view = self.extraction_view(extraction_row, cv_file)
        jd = load_structured(jd_version)
        thresholds = self._thresholds(job)

        rules = self.rules.evaluate(view, jd)
        semantic = self.semantic.score(view, jd)
        judge = self.judge.score(view, jd)
        merged = merge_scores(rules, semantic, judge, jd)

        text = self.source_text(cv_file)
        scan = injection.scan(text)
        missing = routing.missing_critical_fields(view["payload"])
        extraction_failed = "extraction_failed_after_retries" in view.get("warnings", [])
        has_dates = any(
            (j or {}).get("from_date") for j in view["payload"].get("work_history", []) or []
        )

        confidence = assemble_confidence(
            mean_field_confidence=view["mean_confidence"],
            source_quality=view["source_quality"],
            evidence_verification_rate=view["evidence_verification_rate"],
            disagreement=merged.disagreement,
            disagreement_cap=thresholds.disagreement_cap,
            disagreement_ceiling=float(
                getattr(self.settings, "disagreement_confidence_ceiling", 0.65)
            ),
            missing_critical_fields=bool(missing),
            extraction_failed=extraction_failed,
        )

        decision = routing.route(
            RoutingInput(
                score=merged.total,
                confidence=confidence.value,
                disagreement=merged.disagreement,
                source_quality=view["source_quality"],
                evidence_verification_rate=view["evidence_verification_rate"],
                years_conflict=bool(view["years_conflict"]),
                injection_suspected=scan.suspected,
                injection_matches=scan.matches,
                hard_rule_failures=rules.hard_failures,
                soft_rule_failures=rules.soft_failures,
                missing_fields=missing,
                extraction_failed=extraction_failed,
                has_dated_work_history=has_dates,
            ),
            thresholds,
        )

        return self._persist(
            job=job,
            jd_version=jd_version,
            cv_file=cv_file,
            rules=rules,
            semantic=semantic,
            judge=judge,
            merged=merged,
            confidence=confidence,
            decision=decision,
            thresholds=thresholds,
            injection_scan=scan,
            actor=actor,
        )

    def screen_job(self, job: Job, jd_version: JDVersion, *, actor: str = "system") -> list[ScreeningOutcome]:
        """Screen every ingested CV against the job. Idempotent by (job, candidate)."""
        outcomes: list[ScreeningOutcome] = []
        cv_files = self.session.scalars(
            select(CVFile).where(CVFile.candidate_id.is_not(None)).order_by(CVFile.id)
        ).all()
        seen: set[int] = set()
        for cv_file in cv_files:
            if cv_file.candidate_id in seen:
                continue
            seen.add(cv_file.candidate_id)
            try:
                result = self.screen_one(job, jd_version, cv_file, actor=actor)
            except Exception:
                logger.exception("screening failed for cv_file %s", cv_file.id)
                continue
            outcomes.append(
                ScreeningOutcome(
                    candidate_id=result.candidate_id,
                    cv_file_id=result.cv_file_id,
                    score=result.merged_score,
                    confidence=result.confidence,
                    routing=result.routing,
                    flags=json.loads(result.flags_json or "[]"),
                    reasons=json.loads(
                        self.session.scalar(
                            select(ReviewQueueEntry.reasons_json).where(
                                ReviewQueueEntry.screening_result_id == result.id
                            )
                        )
                        or "[]"
                    ),
                )
            )
        outcomes.sort(key=lambda o: o.score, reverse=True)
        return outcomes

    # -- persistence --------------------------------------------------------
    def _persist(
        self,
        *,
        job: Job,
        jd_version: JDVersion,
        cv_file: CVFile,
        rules,
        semantic,
        judge,
        merged,
        confidence,
        decision,
        thresholds: Thresholds,
        injection_scan,
        actor: str,
    ) -> ScreeningResult:
        existing = self.session.scalar(
            select(ScreeningResult).where(
                ScreeningResult.job_id == job.id,
                ScreeningResult.candidate_id == cv_file.candidate_id,
            )
        )
        before = (
            {"score": existing.merged_score, "routing": existing.routing}
            if existing is not None
            else None
        )
        row = existing or ScreeningResult(job_id=job.id, candidate_id=cv_file.candidate_id)
        row.jd_version_id = jd_version.id
        row.cv_file_id = cv_file.id
        row.rules_json = json.dumps(rules.as_dict(), ensure_ascii=False, default=str)
        row.semantic_json = json.dumps(semantic.as_dict(), ensure_ascii=False, default=str)
        row.judge_json = json.dumps(judge.as_dict(), ensure_ascii=False, default=str)
        row.merged_score = merged.total
        row.confidence = confidence.value
        row.routing = decision.routing
        row.flags_json = json.dumps(decision.flags)
        row.dimension_breakdown_json = json.dumps(
            {
                "dimensions": merged.dimensions,
                "weights": merged.weights,
                "disagreement": merged.disagreement,
                "confidence": confidence.as_dict(),
                "injection": injection_scan.as_dict(),
            },
            ensure_ascii=False,
            default=str,
        )
        row.prompt_version = f"extract={PROMPT_VERSION};judge={JUDGE_PROMPT_VERSION}"
        row.schema_version = SCHEMA_VERSION
        row.model_name = f"extract={self.gemini.models.extraction};judge={self.gemini.models.judge};embed={self.gemini.models.embedding}"
        row.thresholds_json = json.dumps(thresholds.as_dict())
        row.updated_at = dt.datetime.now(dt.timezone.utc)
        if existing is None:
            self.session.add(row)
        self.session.flush()

        self._sync_review_queue(row, decision)
        audit_record(
            self.session,
            "screening_result",
            row.id,
            "screened",
            actor=actor,
            before=before,
            after={"score": row.merged_score, "routing": row.routing, "flags": decision.flags},
        )
        self.session.commit()
        return row

    def _sync_review_queue(self, result: ScreeningResult, decision) -> None:
        entry = self.session.scalar(
            select(ReviewQueueEntry).where(ReviewQueueEntry.screening_result_id == result.id)
        )
        if decision.routing != routing.HUMAN_REVIEW:
            if entry is not None and entry.status == "open":
                self.session.execute(
                    delete(ReviewQueueEntry).where(ReviewQueueEntry.id == entry.id)
                )
            return
        if entry is None:
            entry = ReviewQueueEntry(
                screening_result_id=result.id,
                job_id=result.job_id,
                reasons_json=json.dumps(decision.reasons, ensure_ascii=False),
                status="open",
            )
            self.session.add(entry)
        elif entry.status == "open":
            entry.reasons_json = json.dumps(decision.reasons, ensure_ascii=False)

    def _thresholds(self, job: Job) -> Thresholds:
        config = self.session.scalar(select(JobConfig).where(JobConfig.job_id == job.id))
        defaults = Thresholds(
            shortlist_score_min=float(getattr(self.settings, "shortlist_score_min", 75.0)),
            reject_score_max=float(getattr(self.settings, "reject_score_max", 45.0)),
            confidence_min=float(getattr(self.settings, "confidence_min", 0.7)),
            disagreement_cap=float(getattr(self.settings, "disagreement_cap", 25.0)),
            years_conflict_tolerance=float(getattr(self.settings, "years_conflict_tolerance", 1.5)),
            min_source_quality=float(getattr(self.settings, "min_source_quality", 0.55)),
        )
        return Thresholds.from_config(config, defaults=defaults)

    # -- review actions -----------------------------------------------------
    def apply_correction(
        self,
        result: ScreeningResult,
        corrections: dict[str, Any],
        *,
        actor: str,
    ) -> ScreeningResult:
        """Apply a human field correction, then re-score and re-route."""
        extraction = self.session.scalar(
            select(Extraction).where(Extraction.cv_file_id == result.cv_file_id)
        )
        if extraction is None:
            raise ValueError(f"no extraction for cv_file {result.cv_file_id}")

        payload = json.loads(extraction.payload_json or "{}")
        field_conf = json.loads(extraction.field_confidence_json or "{}")
        before = {"payload": payload, "field_confidence": dict(field_conf)}

        payload, field_conf = apply_field_corrections(payload, field_conf, corrections)
        extraction.payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        extraction.field_confidence_json = json.dumps(field_conf)
        if "stated_years_experience" in corrections:
            value = corrections["stated_years_experience"]
            extraction.stated_years = float(value) if value is not None else None
        if "computed_years" in corrections:
            value = corrections["computed_years"]
            extraction.computed_years = float(value) if value is not None else None
        tolerance = self._thresholds(self.session.get(Job, result.job_id)).years_conflict_tolerance
        extraction.years_conflict = (
            extraction.stated_years is not None
            and extraction.computed_years is not None
            and abs(extraction.stated_years - extraction.computed_years) > tolerance
        )
        self.session.commit()

        audit_record(
            self.session,
            "extraction",
            extraction.id,
            "corrected",
            actor=actor,
            before=before,
            after={"payload": payload, "field_confidence": field_conf, "corrections": corrections},
        )

        job = self.session.get(Job, result.job_id)
        jd_version = self.session.get(JDVersion, result.jd_version_id)
        return self.screen_one(job, jd_version, self.session.get(CVFile, result.cv_file_id), actor=actor)


HUMAN_CONFIDENCE = 1.0


def apply_field_corrections(
    payload: dict[str, Any],
    field_confidence: dict[str, float],
    corrections: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Apply a closed set of correction operations to an extraction payload.

    A human-supplied value is authoritative: it gets confidence 1.0 and is
    marked as evidence-verified, because a person has just looked at the source.
    """
    payload = json.loads(json.dumps(payload, default=str))

    for key in ("full_name", "email", "phone", "stated_years_experience"):
        if key in corrections:
            payload[key] = corrections[key]

    for skill in corrections.get("add_skills", []) or []:
        name = skill["name"] if isinstance(skill, dict) else str(skill)
        entry = {
            "name": name,
            "canonical": (skill.get("canonical") if isinstance(skill, dict) else None) or name.lower(),
            "level": skill.get("level") if isinstance(skill, dict) else None,
            "evidence_quote": (skill.get("evidence_quote") if isinstance(skill, dict) else None)
            or f"[human correction] {name}",
            "confidence": HUMAN_CONFIDENCE,
            "evidence_verified": True,
            "evidence_ratio": 1.0,
            "source": "human_correction",
        }
        payload.setdefault("skills", []).append(entry)
        field_confidence[f"skill:{name}"] = HUMAN_CONFIDENCE

    removals = {str(s).lower() for s in (corrections.get("remove_skills", []) or [])}
    if removals:
        payload["skills"] = [
            s
            for s in payload.get("skills", []) or []
            if str(s.get("name", "")).lower() not in removals
            and str(s.get("canonical", "")).lower() not in removals
        ]
        for label in list(field_confidence):
            if label.startswith("skill:") and label.split(":", 1)[1].lower() in removals:
                field_confidence.pop(label)

    for edu in corrections.get("add_education", []) or []:
        entry = dict(edu)
        entry.setdefault("evidence_quote", f"[human correction] {edu.get('degree')}")
        entry["confidence"] = HUMAN_CONFIDENCE
        entry["evidence_verified"] = True
        entry["evidence_ratio"] = 1.0
        entry["source"] = "human_correction"
        payload.setdefault("education", []).append(entry)
        field_confidence[f"education:{edu.get('degree')}:{len(payload['education']) - 1}"] = HUMAN_CONFIDENCE

    for job_entry in corrections.get("add_work_history", []) or []:
        entry = dict(job_entry)
        entry.setdefault("highlights", [])
        entry.setdefault("evidence_quote", f"[human correction] {job_entry.get('title')}")
        entry["confidence"] = HUMAN_CONFIDENCE
        entry["evidence_verified"] = True
        entry["evidence_ratio"] = 1.0
        entry["source"] = "human_correction"
        payload.setdefault("work_history", []).append(entry)
        field_confidence[f"work:{job_entry.get('title')}:{len(payload['work_history']) - 1}"] = HUMAN_CONFIDENCE

    for update in corrections.get("update_skill_confidence", []) or []:
        name = str(update.get("name", "")).lower()
        for skill in payload.get("skills", []) or []:
            if str(skill.get("name", "")).lower() == name:
                skill["confidence"] = float(update.get("confidence", HUMAN_CONFIDENCE))
                skill["evidence_verified"] = True
                field_confidence[f"skill:{skill.get('name')}"] = skill["confidence"]

    return payload, field_confidence
