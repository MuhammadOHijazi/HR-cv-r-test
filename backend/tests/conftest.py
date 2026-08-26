"""Shared pytest fixtures.

Every test runs against an isolated on-disk SQLite database and a temporary
storage directory, with the deterministic ``MockTransport`` standing in for
Gemini and ``InMemoryDriveClient`` standing in for Drive.  No credentials, no
network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import config as config_module  # noqa: E402
from backend.app import db as db_module  # noqa: E402
from backend.app.core.drive_client import InMemoryDriveClient  # noqa: E402
from backend.app.core.gemini_client import GeminiClient, ModelConfig  # noqa: E402
from backend.app.core.gemini_transport import MockTransport  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings pointing at a temp DB + storage, in mock mode."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("GEMINI_API_KEYS", "k-aaaa1111,k-bbbb2222")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    config_module.reset_settings_cache()
    settings = config_module.get_settings()
    yield settings
    config_module.reset_settings_cache()


@pytest.fixture
def engine(settings):
    """The production engine builder, pointed at the temp database.

    Deliberately not a hand-rolled ``create_engine``: the SQLite pragmas that
    keep concurrent writes from deadlocking live in ``_make_engine``, so a test
    engine built any other way would not be testing what actually ships.
    """
    db_module.reset()
    eng = db_module.get_engine()
    assert eng.url.database == settings.database_url.replace("sqlite:///", "")
    db_module.init_db()
    yield eng
    eng.dispose()
    db_module.reset()


@pytest.fixture
def session(engine):
    factory = db_module.get_session_factory()
    with factory() as s:
        yield s


@pytest.fixture
def transport():
    return MockTransport(dim=64)


@pytest.fixture
def gemini(transport):
    return GeminiClient(
        transport=transport,
        keys=["key-aaaa1111", "key-bbbb2222"],
        models=ModelConfig(
            extraction="mock-flash",
            judge="mock-pro",
            embedding="mock-embed",
            vision="mock-flash",
            embedding_dim=64,
        ),
    )


@pytest.fixture
def drive():
    return InMemoryDriveClient()


@pytest.fixture
def client(settings, engine, gemini, drive):
    """FastAPI TestClient with Gemini and Drive replaced by the fakes."""
    from fastapi.testclient import TestClient

    from backend.app.api import deps
    from backend.app.main import create_app

    deps.set_gemini(gemini)
    deps.set_drive(drive)
    app = create_app()
    with TestClient(app) as c:
        yield c
    deps.set_gemini(None)
    deps.set_drive(None)


class FakeClock:
    """Controllable monotonic clock for backoff-timing tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def fake_clock():
    return FakeClock()
