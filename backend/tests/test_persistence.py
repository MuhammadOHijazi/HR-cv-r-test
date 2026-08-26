"""Storage-layer behaviour: SQLite pragmas and the write-behind usage store.

The regression this file exists for: the Gemini key-usage counters used to be
written on their own connection on every single API call.  When that landed in
the middle of a request's open transaction, SQLite returned "database is
locked" and a review correction failed with a 500.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from backend.app.api.deps import DbUsageStore, build_gemini, flush_usage, set_gemini
from backend.app.db import get_engine, get_session_factory
from backend.app.models import ApiKeyUsage, AuditLog


# ---------------------------------------------------------------------------
# SQLite configuration
# ---------------------------------------------------------------------------


def test_sqlite_runs_in_wal_mode(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"


def test_sqlite_has_a_busy_timeout(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() >= 1000


def test_foreign_keys_are_enforced(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_counters_are_not_written_inside_an_open_transaction(engine, session):
    """The shape of the old deadlock: a write-behind buffer simply defers.

    SQLite serialises writers, so a second connection genuinely cannot write
    while a request holds a write lock — which is why the counters are buffered
    instead of written through. Recording one here must touch no second
    connection at all.
    """
    session.add(AuditLog(entity_type="probe", entity_id=1, action="open"))
    session.flush()  # holds a write lock, deliberately not committed

    store = DbUsageStore(get_session_factory())
    store.save(0, {"last4": "...1111", "requests": 3, "failures": 0, "rate_limit_hits": 0})

    session.commit()
    # Only now, outside the transaction, does the counter reach the database.
    assert store.flush() is True
    with get_session_factory()() as check:
        assert check.scalar(select(ApiKeyUsage).where(ApiKeyUsage.key_index == 0)) is not None


def test_a_review_correction_survives_the_screening_transaction(client, tmp_path, drive):
    """End-to-end regression: this returned a 500 'database is locked'."""
    from .test_api import PARTIAL_CV_LINES, _lines, approved_job

    from scripts.generate_test_data import render_pdf

    path = tmp_path / "partial.pdf"
    render_pdf(_lines(PARTIAL_CV_LINES, 6), path)
    drive.add_folder("folder-a", "Applications")
    drive.add_file("folder-a", "file-partial", "partial.pdf", path.read_bytes())

    job, _ = approved_job(client)
    client.put(
        f"/api/jobs/{job['id']}/config",
        json={
            "shortlist_score_min": 99.0,
            "reject_score_max": 1.0,
            "confidence_min": 0.7,
            "disagreement_cap": 35.0,
            "years_conflict_tolerance": 1.5,
        },
    )
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/sync")
    client.post(f"/api/jobs/{job['id']}/screen")

    queue = client.get(f"/api/jobs/{job['id']}/review").json()
    assert queue
    response = client.post(
        f"/api/review/{queue[0]['review_entry_id']}/correct",
        json={"corrections": {"stated_years_experience": 6.0, "computed_years": 6.0}},
    )
    assert response.status_code == 200, response.text


def test_key_counters_persist_after_an_api_request(engine, settings, session, drive):
    """The post-request middleware is what actually drains the buffer.

    This one wires up the app exactly as it ships — including the DB-backed
    usage store — rather than the in-memory store the other API tests use.
    """
    from fastapi.testclient import TestClient

    from backend.app.api import deps
    from backend.app.main import create_app

    deps.set_gemini(None)
    deps.set_drive(drive)
    with TestClient(create_app()) as client:
        client.post(
            "/api/jobs",
            json={"title": "X", "raw_jd_text": "Must have:\n- Python\nResponsibilities:\n- Build"},
        )
        assert client.post("/api/jobs/1/structure").status_code == 200

    rows = session.scalars(select(ApiKeyUsage)).all()
    assert rows, "key usage must be persisted once the request finishes"
    assert sum(r.requests for r in rows) >= 1
    assert all(r.key_last4.startswith("...") for r in rows)
    deps.set_gemini(None)
    deps.set_drive(None)


# ---------------------------------------------------------------------------
# Write-behind usage store
# ---------------------------------------------------------------------------


def record(**overrides):
    base = {"last4": "...1111", "requests": 1, "failures": 0, "rate_limit_hits": 0, "cooldown_until": 0.0}
    base.update(overrides)
    return base


def test_saving_does_not_write_immediately(engine, session):
    store = DbUsageStore(get_session_factory())
    store.save(0, record())
    assert session.scalar(select(ApiKeyUsage)) is None, "counters are buffered, not written"


def test_flush_persists_the_buffer(engine, session):
    store = DbUsageStore(get_session_factory())
    store.save(0, record(requests=7))
    store.flush()
    row = session.scalar(select(ApiKeyUsage).where(ApiKeyUsage.key_index == 0))
    assert row.requests == 7 and row.key_last4 == "...1111"


def test_flush_updates_an_existing_row(engine, session):
    store = DbUsageStore(get_session_factory())
    store.save(0, record(requests=1))
    store.flush()
    store.save(0, record(requests=9, rate_limit_hits=2))
    store.flush()
    rows = session.scalars(select(ApiKeyUsage)).all()
    assert len(rows) == 1
    assert rows[0].requests == 9 and rows[0].rate_limit_hits == 2


def test_flushing_an_empty_buffer_is_a_no_op(engine):
    assert DbUsageStore(get_session_factory()).flush() is True


def test_the_buffer_auto_flushes_after_enough_updates(engine, session):
    store = DbUsageStore(get_session_factory())
    for i in range(DbUsageStore.FLUSH_EVERY):
        store.save(0, record(requests=i))
    assert session.scalar(select(ApiKeyUsage)) is not None


def test_counters_survive_a_restart(engine):
    store = DbUsageStore(get_session_factory())
    store.save(0, record(requests=5, failures=2))
    store.flush()
    revived = DbUsageStore(get_session_factory())
    assert revived.load()[0]["requests"] == 5
    assert revived.load()[0]["failures"] == 2


def test_a_failing_flush_keeps_the_counters_buffered(engine):
    """A bookkeeping failure must never surface to the caller."""

    class Broken:
        def __call__(self):
            raise RuntimeError("no session for you")

    class BrokenSession:
        def __enter__(self):
            from sqlalchemy.exc import OperationalError

            raise OperationalError("SELECT 1", {}, Exception("database is locked"))

        def __exit__(self, *exc):
            return False

    store = DbUsageStore(lambda: BrokenSession())
    store.save(0, record(requests=4))
    assert store.flush() is False, "a locked database must not raise"
    # The value is still buffered, so a later flush can persist it.
    store._session_factory = get_session_factory()
    assert store.flush() is True
    assert store.load()[0]["requests"] == 4


def test_gemini_calls_never_fail_because_of_the_usage_store(engine, settings, transport):
    """The gateway keeps working even when counters cannot be persisted."""
    from backend.app.core.gemini_client import GeminiClient, ModelConfig

    class BrokenStore:
        def load(self):
            return {}

        def save(self, key_index, record):
            return None

    client = GeminiClient(
        transport=transport,
        keys=["k-aaaa1111"],
        models=ModelConfig("f", "p", "e", "f", 8),
        usage_store=BrokenStore(),
    )
    assert client.embed(["hello"])


def test_flush_usage_is_safe_before_any_client_exists():
    set_gemini(None)
    flush_usage()  # must not raise


def test_the_app_builds_one_shared_usage_store(engine, settings):
    """Two clients in one process must not keep separate buffers."""
    set_gemini(None)
    first = build_gemini(settings)
    second = build_gemini(settings)
    assert first.pool._store is second.pool._store
    set_gemini(None)
