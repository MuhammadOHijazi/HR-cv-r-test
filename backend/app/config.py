"""Application configuration.

Every value is overridable through the environment or a ``.env`` file; the
defaults here are what ``.env.example`` documents, and they are chosen so that
the backend boots and the whole test-suite runs *without any real credentials*.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Gemini -----------------------------------------------------------
    gemini_api_keys: str = Field(default="", description="Comma-separated key pool")
    gemini_extraction_model: str = "gemini-2.5-flash"
    gemini_judge_model: str = "gemini-2.5-pro"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_vision_model: str = "gemini-2.5-flash"
    gemini_embedding_dim: int = 768
    gemini_max_attempts_per_request: int = 6
    gemini_cooldown_base_seconds: float = 30.0
    gemini_cooldown_max_seconds: float = 900.0

    # ---- Google Drive -----------------------------------------------------
    google_service_account_json: str = ""
    drive_page_size: int = 200
    # In mock mode the fake Drive is seeded from this local directory, so the
    # whole app is usable end to end without any credentials.
    mock_drive_dir: str = str(REPO_ROOT / "data" / "synthetic" / "cvs")
    mock_drive_folder_name: str = "Synthetic CVs (mock)"

    # ---- Storage ----------------------------------------------------------
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'app.db').as_posix()}"
    storage_dir: str = str(REPO_ROOT / "data")

    # ---- Routing thresholds (per-job defaults) ----------------------------
    shortlist_score_min: float = 75.0
    reject_score_max: float = 45.0
    confidence_min: float = 0.7
    # The judge scores on a four-level rubric (0/40/70/100), so consecutive
    # levels are 30 points apart. A cap below 30 would read a one-level
    # difference of opinion as disagreement; 35 means only a two-level gap
    # between two scorers measuring the same thing caps confidence.
    disagreement_cap: float = 35.0
    disagreement_confidence_ceiling: float = 0.65
    years_conflict_tolerance: float = 1.5
    evidence_match_threshold: float = 0.8
    min_source_quality: float = 0.55

    # ---- Extraction -------------------------------------------------------
    extraction_max_retries: int = 2
    high_confidence_threshold: float = 0.7

    # ---- Behaviour --------------------------------------------------------
    mock_mode: bool = True
    log_level: str = "INFO"

    @field_validator("mock_mode", mode="before")
    @classmethod
    def _coerce_bool(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return v

    # ---- Derived ----------------------------------------------------------
    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.gemini_api_keys.split(",") if k.strip()]

    @property
    def raw_dir(self) -> Path:
        p = Path(self.storage_dir) / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def text_dir(self) -> Path:
        p = Path(self.storage_dir) / "text"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def service_account_email(self) -> str:
        """Read the ``client_email`` out of the service-account JSON, if present."""
        path = Path(self.google_service_account_json) if self.google_service_account_json else None
        if path and path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")).get("client_email", "")
            except (json.JSONDecodeError, OSError):
                return ""
        return ""

    def default_thresholds(self) -> dict[str, float]:
        return {
            "shortlist_score_min": self.shortlist_score_min,
            "reject_score_max": self.reject_score_max,
            "confidence_min": self.confidence_min,
            "disagreement_cap": self.disagreement_cap,
            "years_conflict_tolerance": self.years_conflict_tolerance,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper — drop the memoised settings object."""
    get_settings.cache_clear()
