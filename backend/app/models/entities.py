"""SQLAlchemy models for the CV screening pipeline."""

from __future__ import annotations

import datetime as dt


from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    raw_jd_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="draft")
    active_jd_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("jd_versions.id"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    versions: Mapped[list["JDVersion"]] = relationship(
        back_populates="job",
        foreign_keys="JDVersion.job_id",
        cascade="all, delete-orphan",
    )
    config: Mapped["JobConfig | None"] = relationship(
        back_populates="job", cascade="all, delete-orphan", uselist=False
    )


class JDVersion(Base):
    __tablename__ = "jd_versions"
    __table_args__ = (UniqueConstraint("job_id", "version", name="uq_jd_job_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    version: Mapped[int] = mapped_column(Integer)
    structured_json: Mapped[str] = mapped_column(Text, default="{}")
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    source_model: Mapped[str] = mapped_column(String(64), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="versions", foreign_keys=[job_id])


class JobConfig(Base):
    __tablename__ = "job_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)
    shortlist_score_min: Mapped[float] = mapped_column(Float, default=75.0)
    reject_score_max: Mapped[float] = mapped_column(Float, default=45.0)
    confidence_min: Mapped[float] = mapped_column(Float, default=0.7)
    disagreement_cap: Mapped[float] = mapped_column(Float, default=35.0)
    years_conflict_tolerance: Mapped[float] = mapped_column(Float, default=1.5)
    weights_json: Mapped[str] = mapped_column(Text, default="{}")

    job: Mapped[Job] = relationship(back_populates="config")


class DriveFolder(Base):
    __tablename__ = "drive_folders"

    id: Mapped[int] = mapped_column(primary_key=True)
    folder_id: Mapped[str] = mapped_column(String(128), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    connected_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    last_synced_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class JobFolder(Base):
    __tablename__ = "job_folders"
    __table_args__ = (UniqueConstraint("job_id", "folder_id", name="uq_job_folder"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    folder_id: Mapped[str] = mapped_column(String(128))


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_key: Mapped[str] = mapped_column(String(255), unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    files: Mapped[list["CVFile"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )


class CVFile(Base):
    __tablename__ = "cv_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("candidates.id"), nullable=True)
    drive_file_id: Mapped[str] = mapped_column(String(128), default="")
    folder_id: Mapped[str] = mapped_column(String(128), default="")
    filename: Mapped[str] = mapped_column(String(512), default="")
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    md5_checksum: Mapped[str] = mapped_column(String(64), unique=True)
    size: Mapped[int] = mapped_column(Integer, default=0)
    raw_path: Mapped[str] = mapped_column(String(1024), default="")
    text_path: Mapped[str] = mapped_column(String(1024), default="")
    source_quality: Mapped[float] = mapped_column(Float, default=1.0)
    is_scanned: Mapped[bool] = mapped_column(Boolean, default=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    candidate: Mapped[Candidate | None] = relationship(back_populates="files")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True)
    cv_file_id: Mapped[int] = mapped_column(ForeignKey("cv_files.id"), unique=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    field_confidence_json: Mapped[str] = mapped_column(Text, default="{}")
    evidence_json: Mapped[str] = mapped_column(Text, default="{}")
    stated_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    years_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class EmbeddingCache(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    model: Mapped[str] = mapped_column(String(64), default="")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ScreeningResult(Base):
    __tablename__ = "screening_results"
    __table_args__ = (UniqueConstraint("job_id", "candidate_id", name="uq_screen_job_cand"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    jd_version_id: Mapped[int] = mapped_column(ForeignKey("jd_versions.id"))
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    cv_file_id: Mapped[int] = mapped_column(ForeignKey("cv_files.id"))
    rules_json: Mapped[str] = mapped_column(Text, default="{}")
    semantic_json: Mapped[str] = mapped_column(Text, default="{}")
    judge_json: Mapped[str] = mapped_column(Text, default="{}")
    merged_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    routing: Mapped[str] = mapped_column(String(32), default="human_review")
    flags_json: Mapped[str] = mapped_column(Text, default="[]")
    dimension_breakdown_json: Mapped[str] = mapped_column(Text, default="{}")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    schema_version: Mapped[str] = mapped_column(String(32), default="")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    thresholds_json: Mapped[str] = mapped_column(Text, default="{}")
    human_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ReviewQueueEntry(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(primary_key=True)
    screening_result_id: Mapped[int] = mapped_column(
        ForeignKey("screening_results.id"), unique=True
    )
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(32), default="open")
    resolved_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(128), default="system")
    before_json: Mapped[str] = mapped_column(Text, default="null")
    after_json: Mapped[str] = mapped_column(Text, default="null")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ApiKeyUsage(Base):
    __tablename__ = "api_key_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_index: Mapped[int] = mapped_column(Integer, unique=True)
    key_last4: Mapped[str] = mapped_column(String(8), default="")
    requests: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    rate_limit_hits: Mapped[int] = mapped_column(Integer, default=0)
    cooldown_until: Mapped[float] = mapped_column(Float, default=0.0)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class SkillTaxonomy(Base):
    __tablename__ = "skills_taxonomy"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical: Mapped[str] = mapped_column(String(128), unique=True)
    aliases_json: Mapped[str] = mapped_column(Text, default="[]")
    category: Mapped[str] = mapped_column(String(64), default="general")


class SyncRun(Base):
    __tablename__ = "sync_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    status: Mapped[str] = mapped_column(String(32), default="running")
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    new_files: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    errors_json: Mapped[str] = mapped_column(Text, default="[]")
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
