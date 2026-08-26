"""The human review queue and the three review actions.

No candidate leaves the system rejected without one of these actions being taken
by a person: even ``preliminary_reject`` needs a batch confirmation.
"""

from __future__ import annotations

import datetime as dt
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.audit import record as audit_record
from ..core.gemini_client import AllKeysExhausted, GeminiClient
from ..core.pipeline import ScreeningPipeline
from ..models import Job, ReviewQueueEntry, ScreeningResult
from ..schemas.api import (
    REJECT_REASONS,
    ApproveReviewRequest,
    ConfirmRejectsRequest,
    CorrectionRequest,
    RejectRequest,
)
from .deps import get_db, get_gemini, loads
from .screening import result_dict

router = APIRouter(prefix="/api", tags=["review"])


@router.get("/review/reasons")
def reject_reasons() -> list[str]:
    """The closed list a human must pick from when rejecting."""
    return list(REJECT_REASONS)


@router.get("/jobs/{job_id}/review")
def review_queue(
    job_id: int, session: Session = Depends(get_db), status: str = "open"
) -> list[dict]:
    if session.get(Job, job_id) is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    entries = session.scalars(
        select(ReviewQueueEntry)
        .where(ReviewQueueEntry.job_id == job_id, ReviewQueueEntry.status == status)
        .order_by(ReviewQueueEntry.id)
    ).all()
    out: list[dict] = []
    for entry in entries:
        result = session.get(ScreeningResult, entry.screening_result_id)
        if result is None:
            continue
        payload = result_dict(session, result)
        # The routing reason comes first: the reviewer sees WHY before the score.
        payload["review_entry_id"] = entry.id
        payload["reasons"] = loads(entry.reasons_json, [])
        payload["review_status"] = entry.status
        out.append(payload)
    return out


def _entry_or_404(session: Session, entry_id: int) -> ReviewQueueEntry:
    entry = session.get(ReviewQueueEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"review entry {entry_id} not found")
    if entry.status != "open":
        raise HTTPException(status_code=409, detail=f"review entry {entry_id} is already resolved")
    return entry


def _resolve(entry: ReviewQueueEntry, action: str, reason: str | None = None) -> None:
    entry.status = "resolved"
    entry.resolved_action = action
    entry.resolved_reason = reason
    entry.resolved_at = dt.datetime.now(dt.timezone.utc)


@router.post("/review/{entry_id}/approve")
def approve_candidate(
    entry_id: int, body: ApproveReviewRequest, session: Session = Depends(get_db)
) -> dict:
    entry = _entry_or_404(session, entry_id)
    result = session.get(ScreeningResult, entry.screening_result_id)
    before = {"routing": result.routing, "human_decision": result.human_decision}
    result.routing = "auto_shortlist"
    result.human_decision = "approved"
    _resolve(entry, "approve")
    audit_record(
        session,
        "screening_result",
        result.id,
        "human_approved",
        actor=body.actor,
        before=before,
        after={"routing": result.routing, "human_decision": "approved", "note": body.note},
    )
    session.commit()
    return result_dict(session, result)


@router.post("/review/{entry_id}/reject")
def reject_candidate(
    entry_id: int, body: RejectRequest, session: Session = Depends(get_db)
) -> dict:
    if body.reason not in REJECT_REASONS:
        raise HTTPException(
            status_code=400,
            detail=f"reason must be one of: {', '.join(REJECT_REASONS)}",
        )
    entry = _entry_or_404(session, entry_id)
    result = session.get(ScreeningResult, entry.screening_result_id)
    before = {"routing": result.routing, "human_decision": result.human_decision}
    result.routing = "rejected"
    result.human_decision = "rejected"
    _resolve(entry, "reject", body.reason)
    audit_record(
        session,
        "screening_result",
        result.id,
        "human_rejected",
        actor=body.actor,
        before=before,
        after={"routing": "rejected", "reason": body.reason, "note": body.note},
    )
    session.commit()
    return result_dict(session, result)


@router.post("/review/{entry_id}/correct")
def correct_field(
    entry_id: int,
    body: CorrectionRequest,
    session: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Correct an extracted field; the candidate is immediately re-scored."""
    entry = _entry_or_404(session, entry_id)
    result = session.get(ScreeningResult, entry.screening_result_id)
    if not body.corrections:
        raise HTTPException(status_code=400, detail="no corrections supplied")
    # Rounded the same way the results endpoint rounds, so the UI can compare
    # the before/after pair it is showing without spurious differences.
    before = {
        "score": round(result.merged_score, 2),
        "routing": result.routing,
        "confidence": round(result.confidence, 4),
    }
    pipeline = ScreeningPipeline(session, gemini, settings=settings)
    try:
        updated = pipeline.apply_correction(result, body.corrections, actor=body.actor)
    except AllKeysExhausted as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    audit_record(
        session,
        "screening_result",
        updated.id,
        "human_corrected",
        actor=body.actor,
        before=before,
        after={
            "score": updated.merged_score,
            "routing": updated.routing,
            "confidence": updated.confidence,
            "corrections": body.corrections,
            "note": body.note,
        },
    )
    # The re-run may have cleared the entry; resolve whatever survived.
    surviving = session.get(ReviewQueueEntry, entry_id)
    if surviving is not None and surviving.status == "open":
        if updated.routing == "human_review":
            surviving.reasons_json = json.dumps(
                loads(surviving.reasons_json, []), ensure_ascii=False
            )
        else:
            _resolve(surviving, "correct")
    session.commit()
    payload = result_dict(session, updated)
    payload["before"] = before
    return payload


@router.post("/jobs/{job_id}/confirm-rejects")
def confirm_rejects(
    job_id: int, body: ConfirmRejectsRequest, session: Session = Depends(get_db)
) -> dict:
    """One-click human batch confirmation of preliminary rejects."""
    if session.get(Job, job_id) is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    rows = session.scalars(
        select(ScreeningResult).where(
            ScreeningResult.job_id == job_id,
            ScreeningResult.id.in_(body.result_ids or []),
            ScreeningResult.routing == "preliminary_reject",
        )
    ).all()
    confirmed = []
    for row in rows:
        before = {"routing": row.routing, "human_decision": row.human_decision}
        row.routing = "rejected"
        row.human_decision = "rejected"
        audit_record(
            session,
            "screening_result",
            row.id,
            "human_confirmed_reject",
            actor=body.actor,
            before=before,
            after={"routing": "rejected"},
        )
        confirmed.append(row.id)
    session.commit()
    return {"confirmed": confirmed, "count": len(confirmed)}


@router.get("/jobs/{job_id}/preliminary-rejects")
def preliminary_rejects(job_id: int, session: Session = Depends(get_db)) -> list[dict]:
    if session.get(Job, job_id) is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    rows = session.scalars(
        select(ScreeningResult)
        .where(
            ScreeningResult.job_id == job_id,
            ScreeningResult.routing == "preliminary_reject",
        )
        .order_by(ScreeningResult.merged_score)
    ).all()
    return [result_dict(session, r) for r in rows]
