"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import deps, drive, jobs, review, screening, settings_api
from .config import get_settings
from .db import init_db

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db()
        from .scripts_support import seed_taxonomy_if_empty

        seed_taxonomy_if_empty()
        logger.info(
            "CV screening backend ready (mock_mode=%s, keys=%d)",
            settings.mock_mode,
            len(settings.api_key_list),
        )
        yield

    app = FastAPI(
        title="CV Screening Pipeline",
        version="2.0.0",
        description=(
            "The LLM reads, rules decide, humans judge. No candidate is finally "
            "rejected without a human action."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _flush_key_usage(request, call_next):
        """Persist buffered Gemini key counters once the request's work is done.

        Doing it here rather than inside the Gemini client keeps every counter
        write outside whatever transaction the request was running.
        """
        response = await call_next(request)
        deps.flush_usage()
        return response

    @app.get("/api/health", tags=["health"])
    def health() -> dict:
        return {
            "status": "ok",
            "mock_mode": settings.mock_mode,
            "gemini_keys_configured": len(settings.api_key_list),
        }

    app.include_router(jobs.router)
    app.include_router(drive.router)
    app.include_router(screening.router)
    app.include_router(review.router)
    app.include_router(settings_api.router)
    return app


app = create_app()
