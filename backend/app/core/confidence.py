"""Confidence assembly.

Confidence answers a different question from the score: not "is this candidate
good?" but "do we actually know?".  It mixes

* the mean per-field extraction confidence (0.40)
* the quality of the source text (0.20)
* the share of evidence quotes that verified (0.25)
* how much the three scorers agree with each other (0.15)

and is then hard-capped whenever the scorers disagree beyond the per-job
threshold.  A low confidence always beats a high score in the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WEIGHTS = {
    "field_confidence": 0.40,
    "source_quality": 0.20,
    "evidence_verification": 0.25,
    "scorer_agreement": 0.15,
}


@dataclass
class ConfidenceResult:
    value: float
    components: dict[str, float]
    caps_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "caps_applied": self.caps_applied,
        }


def agreement_from_disagreement(disagreement: float) -> float:
    """Map a 0-100 spread onto a 0-1 agreement score."""
    return max(0.0, min(1.0, 1.0 - (max(0.0, disagreement) / 100.0)))


def assemble(
    *,
    mean_field_confidence: float,
    source_quality: float,
    evidence_verification_rate: float,
    disagreement: float,
    disagreement_cap: float = 35.0,
    disagreement_ceiling: float = 0.65,
    missing_critical_fields: bool = False,
    extraction_failed: bool = False,
) -> ConfidenceResult:
    components = {
        "field_confidence": _clamp(mean_field_confidence),
        "source_quality": _clamp(source_quality),
        "evidence_verification": _clamp(evidence_verification_rate),
        "scorer_agreement": agreement_from_disagreement(disagreement),
    }
    value = sum(components[k] * w for k, w in WEIGHTS.items())
    caps: list[str] = []

    if disagreement > disagreement_cap:
        value = min(value, disagreement_ceiling)
        caps.append("scorer_disagreement")
    if missing_critical_fields:
        value = min(value, 0.6)
        caps.append("missing_critical_field")
    if extraction_failed:
        value = min(value, 0.3)
        caps.append("extraction_failed")

    return ConfidenceResult(round(_clamp(value), 6), components, caps)


def _clamp(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
