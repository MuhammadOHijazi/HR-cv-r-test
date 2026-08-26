"""Database engine, session factory and schema creation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

# SQLite serialises writers. Without WAL a reader blocks a writer outright, and
# without a busy timeout a second connection fails instantly with "database is
# locked" instead of waiting its turn — which is exactly what happens when a
# background write (the Gemini key-usage counters) lands while a request is
# mid-transaction.
SQLITE_BUSY_TIMEOUT_MS = 10_000


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine():
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    is_sqlite = url.startswith("sqlite")
    connect_args = (
        {"check_same_thread": False, "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000}
        if is_sqlite
        else {}
    )
    engine = create_engine(url, connect_args=connect_args, future=True)
    if is_sqlite:
        _apply_sqlite_pragmas(engine, in_memory=":memory:" in url)
    return engine


def _apply_sqlite_pragmas(engine, *, in_memory: bool) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            if not in_memory:
                # WAL lets readers and one writer proceed concurrently.
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def configure(engine) -> None:
    """Point the module at an already-built engine."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def reset() -> None:
    """Drop the cached engine so the next call rebuilds from current settings."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


def init_db() -> None:
    from .models import entities  # noqa: F401  (registers the mappers)

    Base.metadata.create_all(bind=get_engine())


def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
