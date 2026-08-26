"""Settings page: Gemini key-pool status and the service-account e-mail."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..core.drive_client import DriveAuthError, DriveClient
from ..core.gemini_client import GeminiClient
from .deps import get_drive, get_gemini

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings")
def read_settings(
    settings: Settings = Depends(get_settings),
    gemini: GeminiClient = Depends(get_gemini),
    drive: DriveClient = Depends(get_drive),
) -> dict:
    now = time.monotonic()
    keys = []
    for status in gemini.key_pool_status():
        cooldown_remaining = max(0.0, float(status["cooldown_until"]) - now)
        keys.append(
            {
                "index": status["index"],
                # Never the full key: index plus last-4 only.
                "label": f"key[{status['index']}] {status['last4']}",
                "requests": status["requests"],
                "failures": status["failures"],
                "rate_limit_hits": status["rate_limit_hits"],
                "cooling_down": cooldown_remaining > 0,
                "cooldown_seconds_remaining": round(cooldown_remaining, 1),
            }
        )
    try:
        drive_status = drive.status()
    except DriveAuthError as exc:
        drive_status = {"connected": False, "error": str(exc), "service_account_email": ""}
    return {
        "mock_mode": settings.mock_mode,
        "models": {
            "extraction": settings.gemini_extraction_model,
            "judge": settings.gemini_judge_model,
            "embedding": settings.gemini_embedding_model,
            "vision": settings.gemini_vision_model,
        },
        "key_pool": {
            "size": len(keys),
            "available": sum(1 for k in keys if not k["cooling_down"]),
            "keys": keys,
        },
        "drive": {
            "connected": bool(drive_status.get("connected")),
            "service_account_email": drive_status.get("service_account_email")
            or settings.service_account_email,
            "error": drive_status.get("error"),
        },
        "defaults": settings.default_thresholds(),
    }
