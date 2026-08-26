"""Shared helpers for the end-to-end flow test and the dry-run harness.

Builds the whole system on top of a temp directory: synthetic corpus -> mock
Drive folders -> ingestion -> JD structuring + approval -> screening.  The
pipeline code exercised here is the real production code; only the Gemini
transport and the Drive service are fakes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core import textextract  # noqa: E402
from backend.app.core.drive_client import InMemoryDriveClient  # noqa: E402
from backend.app.core.gemini_transport import MockTransport  # noqa: E402
from scripts.generate_test_data import (  # noqa: E402
    BACKEND_JOB,
    JOB_DESCRIPTIONS,
    build_corpus,
    generate,
)

FOLDER_ID = "folder-synthetic-cvs"
FOLDER_NAME = "Synthetic CVs"


@dataclass
class Corpus:
    directory: Path
    manifest: dict[str, Any]
    drive: InMemoryDriveClient

    def expected_for(self, job_key: str) -> dict[str, dict[str, Any]]:
        return {
            cv["key"]: cv["expected"][job_key]
            for cv in self.manifest["cvs"]
            if job_key in cv["expected"]
        }

    def key_for_filename(self, filename: str) -> str:
        for cv in self.manifest["cvs"]:
            if cv["filename"] == filename:
                return cv["key"]
        return filename


def build_corpus_fixture(tmp_dir: Path, transport: MockTransport) -> Corpus:
    """Generate the corpus, load it into a fake Drive, prime the vision fake."""
    manifest = generate(tmp_dir)
    cv_dir = Path(tmp_dir) / "cvs"

    drive = InMemoryDriveClient()
    drive.add_folder(FOLDER_ID, FOLDER_NAME)
    for cv in build_corpus():
        path = cv_dir / cv.filename
        drive.add_file(FOLDER_ID, f"drive:{cv.key}", cv.filename, path.read_bytes())
        if cv.fmt == "scanned_pdf":
            _register_scanned(transport, path, "\n".join(cv.lines))

    return Corpus(cv_dir, manifest, drive)


def _register_scanned(transport: MockTransport, pdf_path: Path, text: str) -> None:
    """Tell the vision fake what the rasterised pages of this PDF say."""
    import pymupdf

    with pymupdf.open(stream=pdf_path.read_bytes(), filetype="pdf") as doc:
        images = textextract._render_pages(doc, doc.page_count)
    transport.register_vision(images, text)


def job_payload(job_key: str) -> dict[str, str]:
    spec = JOB_DESCRIPTIONS[job_key]
    return {"title": spec["title"], "raw_jd_text": spec["text"]}


DEFAULT_JOB = BACKEND_JOB
