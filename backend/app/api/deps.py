"""Shared dependencies: DB session, Gemini gateway, Drive client."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.drive_client import DriveClient, GoogleDriveClient, InMemoryDriveClient
from ..core.gemini_client import GeminiClient, ModelConfig, UsageStore
from ..core.gemini_transport import GenaiTransport, MockTransport
from ..db import get_session_factory
from ..models import ApiKeyUsage

logger = logging.getLogger(__name__)

MOCK_KEYS = ["mock-key-0001", "mock-key-0002"]

_drive_client: DriveClient | None = None
_gemini_client: GeminiClient | None = None
_usage_store: "DbUsageStore | None" = None


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


class DbUsageStore(UsageStore):
    """Per-key counters, buffered in memory and written behind the request.

    Counters are bookkeeping, not business data: the live pool state already
    lives in the client, and these rows exist only so the numbers survive a
    restart.  Writing them on every single Gemini call would open a second
    connection in the middle of whatever transaction the request is running,
    which on SQLite is how you get "database is locked".  So they are buffered
    and flushed opportunistically, and a failed flush is retried later rather
    than propagated into the caller's Gemini call.
    """

    #: Flush after this many buffered updates, so a long job still persists.
    FLUSH_EVERY = 25

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory
        self._pending: dict[int, dict[str, Any]] = {}
        self._since_flush = 0

    def load(self) -> dict[int, dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.scalars(select(ApiKeyUsage)).all()
            return {
                r.key_index: {
                    "requests": r.requests,
                    "failures": r.failures,
                    "rate_limit_hits": r.rate_limit_hits,
                    "cooldown_until": r.cooldown_until,
                }
                for r in rows
            }

    def save(self, key_index: int, record: dict[str, Any]) -> None:
        self._pending[key_index] = dict(record)
        self._since_flush += 1
        if self._since_flush >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> bool:
        """Best-effort write-behind. Returns True when the buffer was drained."""
        if not self._pending:
            return True
        snapshot = dict(self._pending)
        try:
            with self._session_factory() as session:
                for key_index, record in snapshot.items():
                    row = session.scalar(
                        select(ApiKeyUsage).where(ApiKeyUsage.key_index == key_index)
                    )
                    if row is None:
                        row = ApiKeyUsage(key_index=key_index)
                        session.add(row)
                    row.key_last4 = str(record.get("last4", ""))
                    row.requests = int(record.get("requests", 0))
                    row.failures = int(record.get("failures", 0))
                    row.rate_limit_hits = int(record.get("rate_limit_hits", 0))
                    row.cooldown_until = float(record.get("cooldown_until", 0.0))
                session.commit()
        except SQLAlchemyError as exc:
            # Keep the counters buffered and try again on the next flush; never
            # fail a screening run over usage bookkeeping.
            logger.warning("deferring Gemini key usage flush: %s", exc)
            return False
        for key_index, record in snapshot.items():
            if self._pending.get(key_index) == record:
                self._pending.pop(key_index, None)
        self._since_flush = 0
        return True


def build_gemini(settings: Settings) -> GeminiClient:
    """Construct the one Gemini gateway for this process."""
    models = ModelConfig(
        extraction=settings.gemini_extraction_model,
        judge=settings.gemini_judge_model,
        embedding=settings.gemini_embedding_model,
        vision=settings.gemini_vision_model,
        embedding_dim=settings.gemini_embedding_dim,
    )
    keys = settings.api_key_list
    if settings.mock_mode or not keys:
        transport: Any = MockTransport(dim=settings.gemini_embedding_dim)
        keys = keys or MOCK_KEYS
    else:
        transport = GenaiTransport()
    return GeminiClient(
        transport=transport,
        keys=keys,
        models=models,
        base_cooldown=settings.gemini_cooldown_base_seconds,
        max_cooldown=settings.gemini_cooldown_max_seconds,
        max_attempts=settings.gemini_max_attempts_per_request,
        usage_store=get_usage_store(),
    )


def get_usage_store() -> DbUsageStore:
    global _usage_store
    if _usage_store is None:
        _usage_store = DbUsageStore(get_session_factory())
    return _usage_store


def flush_usage() -> None:
    """Drain the buffered key counters; safe to call from a request boundary."""
    if _usage_store is not None:
        _usage_store.flush()


def build_drive(settings: Settings) -> DriveClient:
    if settings.mock_mode or not settings.google_service_account_json:
        client = InMemoryDriveClient()
        source = Path(settings.mock_drive_dir) if settings.mock_drive_dir else None
        if source and source.is_dir():
            client.load_directory("mock-folder-1", settings.mock_drive_folder_name, source)
        return client
    return GoogleDriveClient(
        settings.google_service_account_json, page_size=settings.drive_page_size
    )


def get_gemini(settings: Settings = Depends(get_settings)) -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = build_gemini(settings)
    return _gemini_client


def get_drive(settings: Settings = Depends(get_settings)) -> DriveClient:
    global _drive_client
    if _drive_client is None:
        _drive_client = build_drive(settings)
    return _drive_client


def set_gemini(client: GeminiClient | None) -> None:
    """Override the process-wide Gemini gateway (tests, scripts)."""
    global _gemini_client, _usage_store
    _gemini_client = client
    if client is None:
        _usage_store = None


def set_drive(client: DriveClient | None) -> None:
    global _drive_client
    _drive_client = client


def loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback
