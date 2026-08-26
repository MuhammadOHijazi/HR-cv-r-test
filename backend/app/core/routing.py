"""The two-axis decision matrix: match score x confidence.

    score >= shortlist_min AND confidence >= conf_min AND no flags
        -> auto_shortlist

    score < reject_max AND confidence >= conf_min
        (or a high-confidence must-have failure)
        -> preliminary_reject   [queued for one-click human batch confirmation;
                                 NEVER final without a human]

    everything else
        -> human_review         [stored with machine-readable reasons]

Thresholds come from the per-job config, never from module constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AUTO_SHORTLIST = "auto_shortlist"
HUMAN_REVIEW = "human_review"
PRELIMINARY_REJECT = "preliminary_reject"

# Machine-readable review reasons.
FLAG_LOW_SOURCE_QUALITY = "low_ocr_quality"
FLAG_MISSING_CRITICAL_FIELD = "missing_critical_field"
FLAG_YEARS_CONFLICT = "stated_vs_computed_years_conflict"
FLAG_SCORER_DISAGREEMENT = "scorer_disagreement"
FLAG_INJECTION = "injection_suspicion"
FLAG_UNVERIFIED_EVIDENCE = "high_score_weak_evidence"
FLAG_SOFT_RULE_FAILURE = "must_have_failure_on_low_confidence_field"
FLAG_EXTRACTION_FAILED = "extraction_failed"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_BORDERLINE_SCORE = "borderline_score"
FLAG_NO_WORK_HISTORY = "no_dated_work_history"

CRITICAL_FIELDS = ("skills", "work_history")


@dataclass
class Thresholds:
    shortlist_score_min: float = 75.0
    reject_score_max: float = 45.0
    confidence_min: float = 0.7
    disagreement_cap: float = 35.0
    years_conflict_tolerance: float = 1.5
    min_source_quality: float = 0.55
    near_perfect_score: float = 90.0
    weak_evidence_rate: float = 0.5

    @classmethod
    def from_config(cls, config: Any, *, defaults: "Thresholds | None" = None) -> "Thresholds":
        base = defaults or cls()
        if config is None:
            return base
        return cls(
            shortlist_score_min=float(getattr(config, "shortlist_score_min", base.shortlist_score_min)),
            reject_score_max=float(getattr(config, "reject_score_max", base.reject_score_max)),
            confidence_min=float(getattr(config, "confidence_min", base.confidence_min)),
            disagreement_cap=float(getattr(config, "disagreement_cap", base.disagreement_cap)),
            years_conflict_tolerance=float(
                getattr(config, "years_conflict_tolerance", base.years_conflict_tolerance)
            ),
            min_source_quality=base.min_source_quality,
            near_perfect_score=base.near_perfect_score,
            weak_evidence_rate=base.weak_evidence_rate,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "shortlist_score_min": self.shortlist_score_min,
            "reject_score_max": self.reject_score_max,
            "confidence_min": self.confidence_min,
            "disagreement_cap": self.disagreement_cap,
            "years_conflict_tolerance": self.years_conflict_tolerance,
            "min_source_quality": self.min_source_quality,
        }


@dataclass
class RoutingDecision:
    routing: str
    flags: list[str] = field(default_factory=list)
    reasons: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"routing": self.routing, "flags": self.flags, "reasons": self.reasons}


@dataclass
class RoutingInput:
    """Everything the router is allowed to look at."""

    score: float
    confidence: float
    disagreement: float = 0.0
    source_quality: float = 1.0
    evidence_verification_rate: float = 1.0
    years_conflict: bool = False
    injection_suspected: bool = False
    injection_matches: list[str] = field(default_factory=list)
    hard_rule_failures: list[dict[str, Any]] = field(default_factory=list)
    soft_rule_failures: list[dict[str, Any]] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    extraction_failed: bool = False
    has_dated_work_history: bool = True


def collect_flags(data: RoutingInput, thresholds: Thresholds) -> list[dict[str, Any]]:
    """Every review reason that applies, machine-readable and ordered."""
    reasons: list[dict[str, Any]] = []

    if data.extraction_failed:
        reasons.append({"code": FLAG_EXTRACTION_FAILED, "detail": "structured extraction did not validate"})
    if data.source_quality < thresholds.min_source_quality:
        reasons.append(
            {
                "code": FLAG_LOW_SOURCE_QUALITY,
                "detail": f"source quality {data.source_quality:.2f} < {thresholds.min_source_quality:.2f}",
            }
        )
    if data.missing_fields:
        reasons.append(
            {"code": FLAG_MISSING_CRITICAL_FIELD, "detail": ", ".join(sorted(data.missing_fields))}
        )
    if not data.has_dated_work_history:
        reasons.append({"code": FLAG_NO_WORK_HISTORY, "detail": "no work history entry carries a date"})
    if data.years_conflict:
        reasons.append(
            {
                "code": FLAG_YEARS_CONFLICT,
                "detail": f"stated and computed years differ by more than {thresholds.years_conflict_tolerance}",
            }
        )
    if data.disagreement > thresholds.disagreement_cap:
        reasons.append(
            {
                "code": FLAG_SCORER_DISAGREEMENT,
                "detail": f"spread {data.disagreement:.1f} > {thresholds.disagreement_cap:.1f}",
            }
        )
    if data.injection_suspected:
        reasons.append(
            {"code": FLAG_INJECTION, "detail": ", ".join(data.injection_matches) or "instruction-like text in CV"}
        )
    if (
        data.score >= thresholds.near_perfect_score
        and data.evidence_verification_rate < thresholds.weak_evidence_rate
    ):
        reasons.append(
            {
                "code": FLAG_UNVERIFIED_EVIDENCE,
                "detail": f"score {data.score:.1f} with only {data.evidence_verification_rate:.0%} verified evidence",
            }
        )
    if data.soft_rule_failures:
        reasons.append(
            {
                "code": FLAG_SOFT_RULE_FAILURE,
                "detail": "; ".join(
                    f"{f.get('rule')}:{f.get('field', f.get('required', ''))}"
                    for f in data.soft_rule_failures
                ),
            }
        )
    return reasons


def route(data: RoutingInput, thresholds: Thresholds | None = None) -> RoutingDecision:
    """Apply the two-axis matrix."""
    th = thresholds or Thresholds()
    reasons = collect_flags(data, th)
    flags = [r["code"] for r in reasons]

    high_confidence_must_have_failure = bool(data.hard_rule_failures)
    confident = data.confidence >= th.confidence_min

    # 1. Auto-shortlist: strong, confident and completely unflagged.
    if (
        data.score >= th.shortlist_score_min
        and confident
        and not flags
        and not high_confidence_must_have_failure
    ):
        return RoutingDecision(AUTO_SHORTLIST, flags, reasons)

    # 2. Preliminary reject: a confident low score, or a high-confidence
    #    must-have failure. Still requires a human to confirm.
    #
    #    A review flag always outranks this. A flag means something about the
    #    record is untrustworthy — bad OCR, a years contradiction, an injection
    #    attempt — and we will not queue a rejection, however routine, on
    #    evidence we have already marked as suspect.
    if not flags and confident:
        if high_confidence_must_have_failure:
            reasons.append(
                {
                    "code": "high_confidence_must_have_failure",
                    "detail": "; ".join(
                        f"{f.get('rule')}:{f.get('field', f.get('required', ''))}"
                        for f in data.hard_rule_failures
                    ),
                }
            )
            return RoutingDecision(
                PRELIMINARY_REJECT, ["high_confidence_must_have_failure"], reasons
            )
        if data.score < th.reject_score_max:
            reasons.append(
                {
                    "code": "below_reject_threshold",
                    "detail": f"score {data.score:.1f} < {th.reject_score_max:.1f}",
                }
            )
            return RoutingDecision(PRELIMINARY_REJECT, flags, reasons)

    # 3. Everything else is a human decision.
    if not confident:
        reasons.insert(
            0,
            {
                "code": FLAG_LOW_CONFIDENCE,
                "detail": f"confidence {data.confidence:.2f} < {th.confidence_min:.2f}",
            },
        )
        flags = [FLAG_LOW_CONFIDENCE] + flags
    elif th.reject_score_max <= data.score < th.shortlist_score_min:
        reasons.append(
            {
                "code": FLAG_BORDERLINE_SCORE,
                "detail": f"score {data.score:.1f} between {th.reject_score_max:.1f} and {th.shortlist_score_min:.1f}",
            }
        )
        flags = flags + [FLAG_BORDERLINE_SCORE]
    return RoutingDecision(HUMAN_REVIEW, flags, reasons)


def missing_critical_fields(payload: dict[str, Any]) -> list[str]:
    return [f for f in CRITICAL_FIELDS if not (payload.get(f) or [])]
