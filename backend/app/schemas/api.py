"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

REJECT_REASONS = (
    "missing_must_have_skill",
    "insufficient_experience",
    "education_requirement_not_met",
    "irrelevant_background",
    "unreadable_document",
    "duplicate_application",
    "other_documented",
)


class JobCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    raw_jd_text: str = Field(min_length=1)


class JobConfigUpdate(BaseModel):
    shortlist_score_min: float = Field(ge=0, le=100)
    reject_score_max: float = Field(ge=0, le=100)
    confidence_min: float = Field(ge=0, le=1)
    disagreement_cap: float = Field(ge=0, le=100)
    years_conflict_tolerance: float = Field(ge=0, le=20)


class JDEdit(BaseModel):
    structured: dict[str, Any]


class ApproveRequest(BaseModel):
    actor: str = "recruiter"


class FolderAssignment(BaseModel):
    folder_ids: list[str]


class RejectRequest(BaseModel):
    reason: str
    actor: str = "recruiter"
    note: str = ""


class ApproveReviewRequest(BaseModel):
    actor: str = "recruiter"
    note: str = ""


class CorrectionRequest(BaseModel):
    corrections: dict[str, Any]
    actor: str = "recruiter"
    note: str = ""


class ConfirmRejectsRequest(BaseModel):
    result_ids: list[int]
    actor: str = "recruiter"
