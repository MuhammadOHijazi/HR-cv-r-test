"""Drive connection, folder listing, folder assignment and on-demand sync."""

from __future__ import annotations

import datetime as dt
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.drive_client import DriveAuthError, DriveClient
from ..core.gemini_client import GeminiClient
from ..core.ingestion import IngestionService
from ..models import DriveFolder, Job, JobFolder, SyncRun
from ..schemas.api import FolderAssignment
from .deps import get_db, get_drive, get_gemini

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["drive"])


def _service(
    session: Session, drive: DriveClient, gemini: GeminiClient, settings: Settings
) -> IngestionService:
    return IngestionService(
        session, drive, gemini, raw_dir=settings.raw_dir, text_dir=settings.text_dir
    )


@router.get("/drive/status")
def drive_status(
    drive: DriveClient = Depends(get_drive), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        status = drive.status()
    except DriveAuthError as exc:
        status = {"connected": False, "error": str(exc), "service_account_email": ""}
    status.setdefault("service_account_email", settings.service_account_email)
    if not status.get("service_account_email"):
        status["service_account_email"] = settings.service_account_email
    return status


@router.get("/drive/folders")
def list_folders(session: Session = Depends(get_db)) -> list[dict]:
    rows = session.scalars(select(DriveFolder).order_by(DriveFolder.name)).all()
    return [
        {
            "folder_id": r.folder_id,
            "name": r.name,
            "last_synced_at": r.last_synced_at.isoformat() if r.last_synced_at else None,
        }
        for r in rows
    ]


@router.post("/drive/folders/refresh")
def refresh_folders(
    session: Session = Depends(get_db),
    drive: DriveClient = Depends(get_drive),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    try:
        rows = _service(session, drive, gemini, settings).refresh_folders()
    except DriveAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        {
            "folder_id": r.folder_id,
            "name": r.name,
            "last_synced_at": r.last_synced_at.isoformat() if r.last_synced_at else None,
        }
        for r in rows
    ]


def _job_or_404(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@router.get("/jobs/{job_id}/folders")
def get_job_folders(job_id: int, session: Session = Depends(get_db)) -> list[str]:
    _job_or_404(session, job_id)
    return list(
        session.scalars(select(JobFolder.folder_id).where(JobFolder.job_id == job_id)).all()
    )


@router.post("/jobs/{job_id}/folders")
def assign_folders(
    job_id: int, body: FolderAssignment, session: Session = Depends(get_db)
) -> list[str]:
    _job_or_404(session, job_id)
    known = {
        f.folder_id for f in session.scalars(select(DriveFolder)).all()
    }
    unknown = [f for f in body.folder_ids if f not in known]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown folder(s): {', '.join(unknown)} — refresh the folder list first",
        )
    session.execute(delete(JobFolder).where(JobFolder.job_id == job_id))
    for folder_id in dict.fromkeys(body.folder_ids):
        session.add(JobFolder(job_id=job_id, folder_id=folder_id))
    session.commit()
    return list(dict.fromkeys(body.folder_ids))


@router.post("/jobs/{job_id}/sync")
def sync_now(
    job_id: int,
    session: Session = Depends(get_db),
    drive: DriveClient = Depends(get_drive),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> dict:
    _job_or_404(session, job_id)
    folders = session.scalars(
        select(JobFolder.folder_id).where(JobFolder.job_id == job_id)
    ).all()
    if not folders:
        raise HTTPException(
            status_code=400, detail="assign at least one Drive folder to this job first"
        )

    run = SyncRun(job_id=job_id, status="running")
    session.add(run)
    session.commit()
    try:
        report = _service(session, drive, gemini, settings).sync_job(job_id, sync_run=run)
    except DriveAuthError as exc:
        run.status = "failed"
        run.errors_json = json.dumps([{"error": str(exc)}])
        run.finished_at = dt.datetime.now(dt.timezone.utc)
        session.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    run.status = "completed"
    run.finished_at = dt.datetime.now(dt.timezone.utc)
    session.commit()
    return {"sync_run_id": run.id, **report.as_dict()}


@router.get("/jobs/{job_id}/sync/status")
def sync_status(job_id: int, session: Session = Depends(get_db)) -> dict:
    _job_or_404(session, job_id)
    run = session.scalar(
        select(SyncRun).where(SyncRun.job_id == job_id).order_by(SyncRun.id.desc()).limit(1)
    )
    if run is None:
        return {"status": "never_run", "processed": 0, "total": 0}
    return {
        "sync_run_id": run.id,
        "status": run.status,
        "total": run.total,
        "processed": run.processed,
        "new_files": run.new_files,
        "duplicates": run.duplicates,
        "errors": json.loads(run.errors_json or "[]"),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
