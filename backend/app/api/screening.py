"""Run screening and serve ranked results."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.gemini_client import AllKeysExhausted, GeminiClient
from ..core.jd import JDNotApproved, JDService, load_structured
from ..core.pipeline import ScreeningPipeline
from ..models import Candidate, CVFile, JDVersion, Job, ReviewQueueEntry, ScreeningResult
from .deps import get_db, get_gemini, loads

router = APIRouter(prefix="/api", tags=["screening"])

SORT_FIELDS = {
    "score": ScreeningResult.merged_score,
    "confidence": ScreeningResult.confidence,
    "candidate": ScreeningResult.candidate_id,
}


def _job_or_404(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


def result_dict(session: Session, row: ScreeningResult, *, include_reasons: bool = True) -> dict:
    candidate = session.get(Candidate, row.candidate_id)
    cv_file = session.get(CVFile, row.cv_file_id)
    breakdown = loads(row.dimension_breakdown_json, {})
    judge = loads(row.judge_json, {})
    rules = loads(row.rules_json, {})
    semantic = loads(row.semantic_json, {})
    payload = {
        "id": row.id,
        "job_id": row.job_id,
        "candidate_id": row.candidate_id,
        "candidate_name": candidate.full_name if candidate else None,
        "candidate_email": candidate.email if candidate else None,
        "cv_file_id": row.cv_file_id,
        "filename": cv_file.filename if cv_file else None,
        "source_quality": cv_file.source_quality if cv_file else None,
        "is_scanned": cv_file.is_scanned if cv_file else None,
        "score": round(row.merged_score, 2),
        "confidence": round(row.confidence, 4),
        "routing": row.routing,
        "flags": loads(row.flags_json, []),
        "human_decision": row.human_decision,
        "dimensions": breakdown.get("dimensions", {}),
        "weights": breakdown.get("weights", {}),
        "disagreement": breakdown.get("disagreement"),
        "confidence_detail": breakdown.get("confidence", {}),
        "injection": breakdown.get("injection", {}),
        "evidence": {
            dim: {
                "score": d.get("score"),
                "quote": d.get("evidence_quote"),
                "verified": d.get("evidence_verified"),
                "rationale": d.get("rationale"),
            }
            for dim, d in (judge.get("dimensions") or {}).items()
        },
        "rules": rules,
        "semantic": {
            "score": semantic.get("score"),
            "pairs": semantic.get("pairs", [])[:5],
        },
        "audit": {
            "prompt_version": row.prompt_version,
            "schema_version": row.schema_version,
            "model_name": row.model_name,
            "thresholds": loads(row.thresholds_json, {}),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        },
    }
    if include_reasons:
        entry = session.scalar(
            select(ReviewQueueEntry).where(ReviewQueueEntry.screening_result_id == row.id)
        )
        payload["review_reasons"] = loads(entry.reasons_json, []) if entry else []
        payload["review_entry_id"] = entry.id if entry else None
        payload["review_status"] = entry.status if entry else None
    return payload


@router.post("/jobs/{job_id}/screen")
def run_screening(
    job_id: int,
    session: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> dict:
    job = _job_or_404(session, job_id)
    try:
        version = JDService(session, gemini).active_version(job)
    except JDNotApproved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    pipeline = ScreeningPipeline(session, gemini, settings=settings)
    try:
        outcomes = pipeline.screen_job(job, version, actor="system")
    except AllKeysExhausted as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job.status = "screened"
    session.commit()
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.routing] = counts.get(outcome.routing, 0) + 1
    return {
        "job_id": job_id,
        "jd_version_id": version.id,
        "screened": len(outcomes),
        "counts": counts,
        "outcomes": [o.as_dict() for o in outcomes],
    }


@router.get("/jobs/{job_id}/results")
def list_results(
    job_id: int,
    session: Session = Depends(get_db),
    routing: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100),
    max_score: float | None = Query(default=None, ge=0, le=100),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    flag: str | None = Query(default=None),
    sort: str = Query(default="score"),
    order: str = Query(default="desc"),
) -> list[dict]:
    _job_or_404(session, job_id)
    if sort not in SORT_FIELDS:
        raise HTTPException(
            status_code=400, detail=f"sort must be one of {', '.join(SORT_FIELDS)}"
        )
    column = SORT_FIELDS[sort]
    stmt = select(ScreeningResult).where(ScreeningResult.job_id == job_id)
    if routing:
        stmt = stmt.where(ScreeningResult.routing == routing)
    if min_score is not None:
        stmt = stmt.where(ScreeningResult.merged_score >= min_score)
    if max_score is not None:
        stmt = stmt.where(ScreeningResult.merged_score <= max_score)
    if min_confidence is not None:
        stmt = stmt.where(ScreeningResult.confidence >= min_confidence)
    stmt = stmt.order_by(column.desc() if order == "desc" else column.asc())
    rows = session.scalars(stmt).all()
    results = [result_dict(session, r) for r in rows]
    if flag:
        results = [r for r in results if flag in r["flags"]]
    return results


@router.get("/candidates/{candidate_id}")
def get_candidate(
    candidate_id: int,
    session: Session = Depends(get_db),
    job_id: int | None = Query(default=None),
) -> dict:
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"candidate {candidate_id} not found")
    files = session.scalars(select(CVFile).where(CVFile.candidate_id == candidate_id)).all()
    source_text = ""
    if files:
        path = Path(files[0].text_path)
        if path.is_file():
            source_text = path.read_text(encoding="utf-8")
    stmt = select(ScreeningResult).where(ScreeningResult.candidate_id == candidate_id)
    if job_id is not None:
        stmt = stmt.where(ScreeningResult.job_id == job_id)
    results = session.scalars(stmt).all()
    return {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "email": candidate.email,
        "phone": candidate.phone,
        "files": [
            {
                "id": f.id,
                "filename": f.filename,
                "mime_type": f.mime_type,
                "source_quality": f.source_quality,
                "is_scanned": f.is_scanned,
                "page_count": f.page_count,
            }
            for f in files
        ],
        "source_text": source_text,
        "results": [result_dict(session, r) for r in results],
    }


@router.get("/jobs/{job_id}/jd")
def get_active_jd(job_id: int, session: Session = Depends(get_db)) -> dict:
    job = _job_or_404(session, job_id)
    if job.active_jd_version_id is None:
        raise HTTPException(status_code=409, detail="job has no approved JD version")
    version = session.get(JDVersion, job.active_jd_version_id)
    return {"version": version.version, "structured": load_structured(version)}
