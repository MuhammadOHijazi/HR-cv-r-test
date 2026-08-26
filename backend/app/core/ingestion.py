"""Drive sync orchestration: download, dedup, extract text, archive.

Deduplication happens on two axes:

* **File identity** — MD5 checksum.  The same bytes are never ingested twice,
  whichever folder they appear in.
* **Candidate identity** — normalised e-mail + phone.  Two different files
  belonging to the same person collapse onto one candidate record.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import regexlayer, textextract
from .drive_client import DriveClient, DriveFileInfo, compute_md5, iter_supported
from .gemini_client import GeminiClient
from .prompts import VISION_SCHEMA, build_vision_prompt
from ..models import Candidate, CVFile, DriveFolder, JobFolder, SyncRun

logger = logging.getLogger(__name__)


class GeminiVisionFallback:
    """Adapts the Gemini gateway to the ``textextract.VisionFallback`` seam."""

    def __init__(self, client: GeminiClient) -> None:
        self.client = client

    def transcribe(self, images: list[bytes]) -> str:
        result = self.client.vision_extract(
            build_vision_prompt(len(images)), images, VISION_SCHEMA
        )
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)


@dataclass
class IngestOutcome:
    file_id: str
    filename: str
    status: str  # ingested | duplicate_file | duplicate_candidate_file | error
    cv_file_id: int | None = None
    candidate_id: int | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "status": self.status,
            "cv_file_id": self.cv_file_id,
            "candidate_id": self.candidate_id,
            "detail": self.detail,
        }


@dataclass
class SyncReport:
    total: int = 0
    ingested: int = 0
    duplicates: int = 0
    errors: int = 0
    outcomes: list[IngestOutcome] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ingested": self.ingested,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


class IngestionService:
    def __init__(
        self,
        session: Session,
        drive: DriveClient,
        gemini: GeminiClient | None,
        *,
        raw_dir: Path,
        text_dir: Path,
    ) -> None:
        self.session = session
        self.drive = drive
        self.gemini = gemini
        self.raw_dir = Path(raw_dir)
        self.text_dir = Path(text_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    # -- folders ------------------------------------------------------------
    def refresh_folders(self) -> list[DriveFolder]:
        """Mirror the folders the service account can see into the DB."""
        remote = self.drive.list_folders()
        known = {f.folder_id: f for f in self.session.scalars(select(DriveFolder)).all()}
        for info in remote:
            row = known.get(info.id)
            if row is None:
                row = DriveFolder(folder_id=info.id, name=info.name)
                self.session.add(row)
            else:
                row.name = info.name
        self.session.commit()
        return list(self.session.scalars(select(DriveFolder)).all())

    def folders_for_job(self, job_id: int) -> list[str]:
        return list(
            self.session.scalars(select(JobFolder.folder_id).where(JobFolder.job_id == job_id)).all()
        )

    # -- sync ---------------------------------------------------------------
    def sync_job(
        self,
        job_id: int,
        *,
        progress: Callable[[int, int], None] | None = None,
        sync_run: SyncRun | None = None,
    ) -> SyncReport:
        folder_ids = self.folders_for_job(job_id)
        files: list[DriveFileInfo] = []
        for folder_id in folder_ids:
            files.extend(iter_supported(self.drive.list_files(folder_id)))
        return self.ingest_files(files, progress=progress, sync_run=sync_run)

    def ingest_files(
        self,
        files: Iterable[DriveFileInfo],
        *,
        progress: Callable[[int, int], None] | None = None,
        sync_run: SyncRun | None = None,
    ) -> SyncReport:
        files = list(files)
        report = SyncReport(total=len(files))
        if sync_run is not None:
            sync_run.total = len(files)
            self.session.commit()
        for index, info in enumerate(files, start=1):
            try:
                outcome = self.ingest_one(info)
            except Exception as exc:  # one bad file must not abort the sync
                logger.exception("ingestion failed for %s", info.name)
                outcome = IngestOutcome(info.id, info.name, "error", detail=str(exc))
            report.outcomes.append(outcome)
            if outcome.status == "ingested":
                report.ingested += 1
            elif outcome.status == "error":
                report.errors += 1
            else:
                report.duplicates += 1
            if sync_run is not None:
                sync_run.processed = index
                sync_run.new_files = report.ingested
                sync_run.duplicates = report.duplicates
                sync_run.errors_json = json.dumps(
                    [o.as_dict() for o in report.outcomes if o.status == "error"]
                )
                self.session.commit()
            if progress is not None:
                progress(index, len(files))
        self._stamp_folders({f.folder_id for f in files})
        return report

    def ingest_one(self, info: DriveFileInfo) -> IngestOutcome:
        existing_by_id = self.session.scalar(
            select(CVFile).where(CVFile.md5_checksum == (info.md5_checksum or ""))
        )
        if existing_by_id is not None:
            return IngestOutcome(
                info.id,
                info.name,
                "duplicate_file",
                cv_file_id=existing_by_id.id,
                candidate_id=existing_by_id.candidate_id,
                detail="md5 already ingested",
            )

        data = self.drive.download(info.id)
        checksum = compute_md5(data)
        existing = self.session.scalar(select(CVFile).where(CVFile.md5_checksum == checksum))
        if existing is not None:
            return IngestOutcome(
                info.id,
                info.name,
                "duplicate_file",
                cv_file_id=existing.id,
                candidate_id=existing.candidate_id,
                detail="md5 already ingested",
            )

        vision = GeminiVisionFallback(self.gemini) if self.gemini is not None else None
        extracted = textextract.extract(
            data, info.name, declared_mime=info.mime_type, vision=vision
        )

        raw_path = self.raw_dir / f"{checksum}{Path(info.name).suffix.lower()}"
        raw_path.write_bytes(data)
        text_path = self.text_dir / f"{checksum}.txt"
        text_path.write_text(extracted.text, encoding="utf-8")

        findings = regexlayer.run(extracted.text)
        email = findings.emails[0] if findings.emails else None
        phone = findings.phones[0] if findings.phones else None
        key = regexlayer.identity_key(email, phone, checksum)

        candidate = self.session.scalar(select(Candidate).where(Candidate.canonical_key == key))
        is_new_candidate = candidate is None
        if candidate is None:
            candidate = Candidate(
                canonical_key=key,
                full_name=_guess_name(extracted.text),
                email=email,
                phone=phone,
            )
            self.session.add(candidate)
            self.session.flush()

        cv_file = CVFile(
            candidate_id=candidate.id,
            drive_file_id=info.id,
            folder_id=info.folder_id,
            filename=info.name,
            mime_type=info.mime_type,
            md5_checksum=checksum,
            size=len(data),
            raw_path=str(raw_path),
            text_path=str(text_path),
            source_quality=extracted.source_quality,
            is_scanned=extracted.is_scanned,
            page_count=extracted.page_count,
        )
        self.session.add(cv_file)
        self.session.commit()

        status = "ingested" if is_new_candidate else "duplicate_candidate_file"
        return IngestOutcome(
            info.id,
            info.name,
            status,
            cv_file_id=cv_file.id,
            candidate_id=candidate.id,
            detail=extracted.method,
        )

    def _stamp_folders(self, folder_ids: set[str]) -> None:
        if not folder_ids:
            return
        now = dt.datetime.now(dt.timezone.utc)
        for row in self.session.scalars(
            select(DriveFolder).where(DriveFolder.folder_id.in_(folder_ids))
        ).all():
            row.last_synced_at = now
        self.session.commit()


# Section headings a CV opens with; never a candidate's name.
_SECTION_WORDS = {
    "summary", "profile", "objective", "about", "skills", "experience",
    "education", "projects", "languages", "certifications", "contact",
    "الملخص", "نبذة", "المهارات", "الخبرة", "الخبرات", "التعليم", "اللغات",
    "الشهادات", "المشاريع", "الاتصال",
}


def _guess_name(text: str) -> str | None:
    """First plausible name-looking line of the document."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            continue
        if regexlayer.EMAIL_RE.search(stripped) or regexlayer.PHONE_RE.search(stripped):
            continue
        if stripped.endswith(":"):
            continue
        words = [w for w in stripped.replace("/", " ").split() if w]
        if not 1 < len(words) <= 6:
            continue
        # A section heading, however it is capitalised or punctuated.
        if any(w.strip(".,:").casefold() in _SECTION_WORDS for w in words):
            continue
        return stripped
    return None
