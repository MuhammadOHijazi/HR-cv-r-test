"""Jobs and job-description structuring / editing / approval / versioning."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.gemini_client import AllKeysExhausted, GeminiClient
from ..core.jd import JDService, load_structured
from ..models import JDVersion, Job, JobConfig, ScreeningResult
from ..schemas.api import ApproveRequest, JDEdit, JobConfigUpdate, JobCreate
from .deps import get_db, get_gemini

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

BUCKETS = ("auto_shortlist", "human_review", "preliminary_reject")


def _job_or_404(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


def _bucket_counts(session: Session, job_id: int) -> dict[str, int]:
    rows = session.execute(
        select(ScreeningResult.routing, func.count(ScreeningResult.id))
        .where(ScreeningResult.job_id == job_id)
        .group_by(ScreeningResult.routing)
    ).all()
    counts = {bucket: 0 for bucket in BUCKETS}
    for routing, count in rows:
        counts[routing] = count
    counts["total"] = sum(counts[b] for b in BUCKETS)
    return counts


def _job_dict(session: Session, job: Job) -> dict:
    active = session.get(JDVersion, job.active_jd_version_id) if job.active_jd_version_id else None
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "raw_jd_text": job.raw_jd_text,
        "active_jd_version": active.version if active else None,
        "active_jd_version_id": job.active_jd_version_id,
        "approved": bool(active and active.approved),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "counts": _bucket_counts(session, job.id),
    }


def _version_dict(version: JDVersion) -> dict:
    return {
        "id": version.id,
        "job_id": version.job_id,
        "version": version.version,
        "structured": load_structured(version),
        "approved": version.approved,
        "approved_by": version.approved_by,
        "approved_at": version.approved_at.isoformat() if version.approved_at else None,
        "source_model": version.source_model,
        "prompt_version": version.prompt_version,
        "created_at": version.created_at.isoformat() if version.created_at else None,
    }


def _config_dict(config: JobConfig) -> dict:
    return {
        "job_id": config.job_id,
        "shortlist_score_min": config.shortlist_score_min,
        "reject_score_max": config.reject_score_max,
        "confidence_min": config.confidence_min,
        "disagreement_cap": config.disagreement_cap,
        "years_conflict_tolerance": config.years_conflict_tolerance,
    }


def ensure_config(session: Session, job_id: int, settings: Settings) -> JobConfig:
    config = session.scalar(select(JobConfig).where(JobConfig.job_id == job_id))
    if config is None:
        config = JobConfig(
            job_id=job_id,
            shortlist_score_min=settings.shortlist_score_min,
            reject_score_max=settings.reject_score_max,
            confidence_min=settings.confidence_min,
            disagreement_cap=settings.disagreement_cap,
            years_conflict_tolerance=settings.years_conflict_tolerance,
            weights_json="{}",
        )
        session.add(config)
        session.commit()
    return config


@router.get("")
def list_jobs(session: Session = Depends(get_db)) -> list[dict]:
    jobs = session.scalars(select(Job).order_by(Job.id)).all()
    return [_job_dict(session, job) for job in jobs]


@router.post("", status_code=201)
def create_job(
    body: JobCreate,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    job = Job(title=body.title.strip(), raw_jd_text=body.raw_jd_text, status="draft")
    session.add(job)
    session.commit()
    ensure_config(session, job.id, settings)
    return _job_dict(session, job)


@router.get("/{job_id}")
def get_job(job_id: int, session: Session = Depends(get_db)) -> dict:
    return _job_dict(session, _job_or_404(session, job_id))


@router.post("/{job_id}/structure")
def structure_jd(
    job_id: int,
    session: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
) -> dict:
    job = _job_or_404(session, job_id)
    if not job.raw_jd_text.strip():
        raise HTTPException(status_code=400, detail="job has no raw JD text to structure")
    try:
        version = JDService(session, gemini).structure(job)
    except AllKeysExhausted as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    job.status = "structured"
    session.commit()
    return _version_dict(version)


@router.get("/{job_id}/versions")
def list_versions(job_id: int, session: Session = Depends(get_db)) -> list[dict]:
    _job_or_404(session, job_id)
    versions = session.scalars(
        select(JDVersion).where(JDVersion.job_id == job_id).order_by(JDVersion.version)
    ).all()
    return [_version_dict(v) for v in versions]


def _version_or_404(session: Session, job_id: int, version: int) -> JDVersion:
    row = session.scalar(
        select(JDVersion).where(JDVersion.job_id == job_id, JDVersion.version == version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} has no version {version}")
    return row


@router.put("/{job_id}/versions/{version}")
def edit_version(
    job_id: int,
    version: int,
    body: JDEdit,
    session: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
) -> dict:
    """Editing never mutates history — it creates a new unapproved version."""
    job = _job_or_404(session, job_id)
    base = _version_or_404(session, job_id, version)
    new_version = JDService(session, gemini).edit(job, base, body.structured, actor="recruiter")
    return _version_dict(new_version)


@router.post("/{job_id}/versions/{version}/approve")
def approve_version(
    job_id: int,
    version: int,
    body: ApproveRequest,
    session: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
) -> dict:
    job = _job_or_404(session, job_id)
    row = _version_or_404(session, job_id, version)
    structured = load_structured(row)
    if not structured.get("must_have") and not structured.get("nice_to_have"):
        raise HTTPException(
            status_code=400,
            detail="cannot approve a JD version with no requirements — edit it first",
        )
    JDService(session, gemini).approve(job, row, actor=body.actor)
    return _version_dict(row)


@router.get("/{job_id}/config")
def get_config(
    job_id: int,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    _job_or_404(session, job_id)
    return _config_dict(ensure_config(session, job_id, settings))


@router.put("/{job_id}/config")
def update_config(
    job_id: int,
    body: JobConfigUpdate,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    _job_or_404(session, job_id)
    if body.reject_score_max >= body.shortlist_score_min:
        raise HTTPException(
            status_code=400, detail="reject_score_max must be below shortlist_score_min"
        )
    config = ensure_config(session, job_id, settings)
    config.shortlist_score_min = body.shortlist_score_min
    config.reject_score_max = body.reject_score_max
    config.confidence_min = body.confidence_min
    config.disagreement_cap = body.disagreement_cap
    config.years_conflict_tolerance = body.years_conflict_tolerance
    session.commit()
    return _config_dict(config)
