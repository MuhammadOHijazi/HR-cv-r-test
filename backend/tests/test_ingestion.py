"""Phase 2 gate — Drive access, text extraction, dedup and archival."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.core import textextract
from backend.app.core.drive_client import (
    DriveAuthError,
    DriveFileInfo,
    GoogleDriveClient,
    InMemoryDriveClient,
    compute_md5,
    iter_supported,
)
from backend.app.core.ingestion import GeminiVisionFallback, IngestionService, _guess_name
from backend.app.core.textextract import (
    UnsupportedFormat,
    detect_mime,
    has_presentation_forms,
    normalise_extracted,
)
from backend.app.models import Candidate, CVFile, DriveFolder, Job, JobFolder

CV_LINES = [
    "Layla Haddad",
    "layla.haddad@example.com | +962 79 555 0101",
    "",
    "SKILLS",
    "Python, PostgreSQL, Docker, Kubernetes",
    "",
    "EXPERIENCE",
    "Principal Backend Engineer, Nimbus Systems, 2016 - present",
    "- Design and build scalable Python microservices",
]


@pytest.fixture
def pdf_bytes(tmp_path):
    from scripts.generate_test_data import render_pdf

    path = tmp_path / "cv.pdf"
    render_pdf(CV_LINES, path)
    return path.read_bytes()


@pytest.fixture
def docx_bytes(tmp_path):
    from scripts.generate_test_data import render_docx

    path = tmp_path / "cv.docx"
    render_docx(CV_LINES, path)
    return path.read_bytes()


@pytest.fixture
def scanned_pdf_bytes(tmp_path):
    from scripts.generate_test_data import render_scanned_pdf

    path = tmp_path / "scan.pdf"
    render_scanned_pdf(CV_LINES, path)
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,declared,expected",
    [
        ("cv.pdf", None, textextract.PDF_MIME),
        ("cv.PDF", None, textextract.PDF_MIME),
        ("cv.docx", None, textextract.DOCX_MIME),
        ("noext", textextract.PDF_MIME, textextract.PDF_MIME),
        ("cv.pdf", "application/octet-stream", textextract.PDF_MIME),
    ],
)
def test_supported_formats_are_detected(filename, declared, expected):
    assert detect_mime(filename, declared) == expected


@pytest.mark.parametrize("filename", ["cv.doc", "cv.txt", "cv.rtf", "photo.png", "noextension"])
def test_unsupported_formats_are_rejected(filename):
    with pytest.raises(UnsupportedFormat):
        detect_mime(filename)


def test_a_corrupt_pdf_raises_unsupported_format():
    with pytest.raises(UnsupportedFormat):
        textextract.extract(b"this is not a pdf", "cv.pdf")


def test_a_corrupt_docx_raises_unsupported_format():
    with pytest.raises(UnsupportedFormat):
        textextract.extract(b"not a docx either", "cv.docx")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_pdf_text_is_extracted(pdf_bytes):
    result = textextract.extract(pdf_bytes, "cv.pdf")
    assert "layla.haddad@example.com" in result.text
    assert "Kubernetes" in result.text
    assert result.method == "pymupdf"
    assert result.is_scanned is False


def test_docx_text_is_extracted(docx_bytes):
    result = textextract.extract(docx_bytes, "cv.docx")
    assert "layla.haddad@example.com" in result.text
    assert result.method == "python-docx"


def test_docx_tables_are_read(tmp_path):
    from docx import Document

    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skill"
    table.rows[0].cells[1].text = "Kubernetes"
    path = tmp_path / "table.docx"
    document.save(str(path))
    assert "Kubernetes" in textextract.extract(path.read_bytes(), "table.docx").text


def test_a_clean_pdf_clears_the_low_quality_threshold(pdf_bytes):
    from backend.app.core.routing import Thresholds

    quality = textextract.extract(pdf_bytes, "cv.pdf").source_quality
    assert quality > Thresholds().min_source_quality


def test_a_fuller_cv_scores_higher_than_a_sparse_one(tmp_path, pdf_bytes):
    """Source quality rewards a document with more to read."""
    from scripts.generate_test_data import render_pdf

    long_path = tmp_path / "long.pdf"
    render_pdf(CV_LINES + ["- " + line for line in CV_LINES] * 3, long_path)
    sparse = textextract.extract(pdf_bytes, "cv.pdf").source_quality
    full = textextract.extract(long_path.read_bytes(), "long.pdf").source_quality
    assert full > sparse


def test_an_empty_document_scores_the_floor(tmp_path):
    from scripts.generate_test_data import render_docx

    path = tmp_path / "blank.docx"
    render_docx([""], path)
    assert textextract.extract(path.read_bytes(), "blank.docx").source_quality <= 0.1


def test_extraction_result_serialises(pdf_bytes):
    payload = textextract.extract(pdf_bytes, "cv.pdf").as_dict()
    assert {"page_count", "is_scanned", "source_quality", "method", "chars"} <= set(payload)


# ---------------------------------------------------------------------------
# Scanned PDFs and the vision fallback
# ---------------------------------------------------------------------------


def test_an_image_only_pdf_is_detected_as_scanned(scanned_pdf_bytes):
    result = textextract.extract(scanned_pdf_bytes, "scan.pdf")
    assert result.is_scanned is True
    assert "no_text_layer" in result.warnings


def test_a_scanned_pdf_without_vision_gets_the_lowest_quality(scanned_pdf_bytes):
    result = textextract.extract(scanned_pdf_bytes, "scan.pdf", vision=None)
    assert result.source_quality <= 0.1
    assert "vision_fallback_unavailable" in result.warnings


def test_the_vision_fallback_recovers_the_text(scanned_pdf_bytes):
    class Vision:
        def __init__(self):
            self.pages = 0

        def transcribe(self, images):
            self.pages = len(images)
            return "\n".join(CV_LINES)

    vision = Vision()
    result = textextract.extract(scanned_pdf_bytes, "scan.pdf", vision=vision)
    assert vision.pages >= 1
    assert "Kubernetes" in result.text
    assert result.method == "gemini-vision"


def test_ocr_output_never_scores_as_high_as_a_text_layer(scanned_pdf_bytes, pdf_bytes):
    class Vision:
        def transcribe(self, images):
            return "\n".join(CV_LINES)

    scanned = textextract.extract(scanned_pdf_bytes, "scan.pdf", vision=Vision())
    digital = textextract.extract(pdf_bytes, "cv.pdf")
    assert scanned.source_quality < digital.source_quality
    assert scanned.source_quality <= 0.6


def test_a_failing_vision_fallback_degrades_gracefully(scanned_pdf_bytes):
    class Vision:
        def transcribe(self, images):
            raise RuntimeError("model unavailable")

    result = textextract.extract(scanned_pdf_bytes, "scan.pdf", vision=Vision())
    assert result.text == ""
    assert "vision_fallback_failed" in result.warnings


def test_the_vision_fallback_goes_through_the_gemini_gateway(gemini, transport):
    fallback = GeminiVisionFallback(gemini)
    transport.register_vision([b"page-one"], "recovered text")
    assert fallback.transcribe([b"page-one"]) == "recovered text"
    assert any(c["kind"] == "vision" for c in transport.calls)


# ---------------------------------------------------------------------------
# Arabic normalisation
# ---------------------------------------------------------------------------


def test_arabic_presentation_forms_normalise_to_base_letters():
    # The shaped forms a PDF producer may emit for "بايثون".
    shaped = "ﺑﺎﻳﺜﻮﻥ"
    assert has_presentation_forms(shaped)
    assert normalise_extracted(shaped) == "بايثون"


def test_ordinary_text_passes_through_normalisation_unchanged():
    assert normalise_extracted("Python, PostgreSQL") == "Python, PostgreSQL"


def test_bidi_controls_and_nbsp_are_stripped():
    assert normalise_extracted("a‏b c") == "ab c"


def test_normalising_empty_text_is_safe():
    assert normalise_extracted("") == ""
    assert normalise_extracted(None) == ""


# ---------------------------------------------------------------------------
# Drive client
# ---------------------------------------------------------------------------


def test_the_in_memory_drive_lists_folders_and_files(pdf_bytes):
    drive = InMemoryDriveClient()
    drive.add_folder("f1", "Applications")
    info = drive.add_file("f1", "file1", "cv.pdf", pdf_bytes)
    assert [f.name for f in drive.list_folders()] == ["Applications"]
    assert [f.id for f in drive.list_files("f1")] == ["file1"]
    assert info.md5_checksum == compute_md5(pdf_bytes)


def test_listing_an_unknown_folder_returns_nothing():
    assert InMemoryDriveClient().list_files("nope") == []


def test_downloading_an_unknown_file_raises():
    with pytest.raises(FileNotFoundError):
        InMemoryDriveClient().download("nope")


def test_adding_a_file_to_an_unknown_folder_creates_it(pdf_bytes):
    drive = InMemoryDriveClient()
    drive.add_file("new-folder", "f", "cv.pdf", pdf_bytes)
    assert [f.id for f in drive.list_folders()] == ["new-folder"]


def test_the_drive_status_exposes_the_service_account_email():
    status = InMemoryDriveClient(email="sa@project.iam.gserviceaccount.com").status()
    assert status["connected"] is True
    assert status["service_account_email"] == "sa@project.iam.gserviceaccount.com"


def test_loading_a_local_directory_registers_the_supported_files(tmp_path, pdf_bytes, docx_bytes):
    source = tmp_path / "drive-folder"
    source.mkdir()
    (source / "a.pdf").write_bytes(pdf_bytes)
    (source / "b.docx").write_bytes(docx_bytes)
    (source / "c.txt").write_text("ignored", encoding="utf-8")
    drive = InMemoryDriveClient()
    drive.load_directory("f1", "Folder", source)
    assert sorted(f.name for f in drive.list_files("f1")) == ["a.pdf", "b.docx"]


def test_unsupported_drive_files_are_filtered_out():
    files = [
        DriveFileInfo("1", "cv.pdf", textextract.PDF_MIME, "m1", 10, "f"),
        DriveFileInfo("2", "notes.txt", "text/plain", "m2", 10, "f"),
        DriveFileInfo("3", "cv.docx", textextract.DOCX_MIME, "m3", 10, "f"),
    ]
    assert [f.id for f in iter_supported(files)] == ["1", "3"]


def test_a_missing_service_account_file_reports_a_clear_error(tmp_path):
    client = GoogleDriveClient(str(tmp_path / "absent.json"))
    status = client.status()
    assert status["connected"] is False
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in status["error"]
    with pytest.raises(DriveAuthError):
        client.list_folders()


def test_the_google_client_talks_to_the_injected_service():
    """Folder and file listing is exercised against a stubbed Drive service."""

    class Files:
        def __init__(self):
            self.queries = []

        def list(self, **kwargs):
            self.queries.append(kwargs["q"])
            payload = (
                {"files": [{"id": "fold1", "name": "CVs"}]}
                if "application/vnd.google-apps.folder" in kwargs["q"]
                else {
                    "files": [
                        {
                            "id": "f1",
                            "name": "cv.pdf",
                            "mimeType": textextract.PDF_MIME,
                            "md5Checksum": "abc",
                            "size": "42",
                        }
                    ]
                }
            )
            return type("R", (), {"execute": lambda _self: payload})()

    class Service:
        def __init__(self):
            self._files = Files()

        def files(self):
            return self._files

    client = GoogleDriveClient("unused.json", service=Service())
    assert [f.name for f in client.list_folders()] == ["CVs"]
    files = client.list_files("fold1")
    assert files[0].md5_checksum == "abc" and files[0].size == 42


# ---------------------------------------------------------------------------
# Ingestion, dedup and archival
# ---------------------------------------------------------------------------


@pytest.fixture
def service(session, settings, gemini):
    drive = InMemoryDriveClient()
    drive.add_folder("f1", "Applications")
    return IngestionService(
        session, drive, gemini, raw_dir=settings.raw_dir, text_dir=settings.text_dir
    ), drive


def test_refreshing_folders_mirrors_drive_into_the_database(service, session):
    ingestion, drive = service
    drive.add_folder("f2", "Referrals")
    ingestion.refresh_folders()
    assert {f.name for f in session.scalars(select(DriveFolder)).all()} == {
        "Applications",
        "Referrals",
    }


def test_refreshing_folders_updates_a_renamed_folder(service, session):
    ingestion, drive = service
    ingestion.refresh_folders()
    drive._folders["f1"]["name"] = "Applications 2026"
    ingestion.refresh_folders()
    folders = session.scalars(select(DriveFolder)).all()
    assert len(folders) == 1 and folders[0].name == "Applications 2026"


def test_a_file_is_ingested_with_its_text_and_raw_archive(service, session, pdf_bytes):
    ingestion, drive = service
    info = drive.add_file("f1", "file1", "cv.pdf", pdf_bytes)
    outcome = ingestion.ingest_one(info)
    assert outcome.status == "ingested"
    row = session.get(CVFile, outcome.cv_file_id)
    assert Path(row.raw_path).is_file(), "the original bytes must be archived"
    assert Path(row.text_path).is_file(), "the extracted text must be archived"
    assert "Kubernetes" in Path(row.text_path).read_text(encoding="utf-8")


def test_the_same_bytes_are_never_ingested_twice(service, pdf_bytes):
    ingestion, drive = service
    first = drive.add_file("f1", "file1", "cv.pdf", pdf_bytes)
    ingestion.ingest_one(first)
    second = drive.add_file("f1", "file2", "cv-copy.pdf", pdf_bytes)
    outcome = ingestion.ingest_one(second)
    assert outcome.status == "duplicate_file"
    assert "md5" in outcome.detail


def test_deduplication_works_across_folders(service, pdf_bytes):
    ingestion, drive = service
    drive.add_folder("f2", "Referrals")
    ingestion.ingest_one(drive.add_file("f1", "a", "cv.pdf", pdf_bytes))
    outcome = ingestion.ingest_one(drive.add_file("f2", "b", "cv.pdf", pdf_bytes))
    assert outcome.status == "duplicate_file"


def test_a_file_with_no_drive_checksum_is_still_deduplicated(service, pdf_bytes):
    ingestion, drive = service
    first = drive.add_file("f1", "a", "cv.pdf", pdf_bytes)
    first.md5_checksum = None
    ingestion.ingest_one(first)
    second = drive.add_file("f1", "b", "cv.pdf", pdf_bytes)
    second.md5_checksum = None
    assert ingestion.ingest_one(second).status == "duplicate_file"


def test_two_files_from_one_person_collapse_onto_one_candidate(service, session, pdf_bytes, docx_bytes):
    ingestion, drive = service
    first = ingestion.ingest_one(drive.add_file("f1", "a", "cv.pdf", pdf_bytes))
    second = ingestion.ingest_one(drive.add_file("f1", "b", "cv.docx", docx_bytes))
    assert second.status == "duplicate_candidate_file"
    assert first.candidate_id == second.candidate_id
    assert len(session.scalars(select(Candidate)).all()) == 1


def test_different_people_get_different_candidates(service, session, tmp_path):
    from scripts.generate_test_data import render_pdf

    ingestion, drive = service
    for index, email in enumerate(["a@example.com", "b@example.com"]):
        path = tmp_path / f"cv{index}.pdf"
        render_pdf(["Person Name", email, "SKILLS", "Python"], path)
        ingestion.ingest_one(drive.add_file("f1", f"id{index}", path.name, path.read_bytes()))
    assert len(session.scalars(select(Candidate)).all()) == 2


def test_identity_falls_back_to_the_checksum_when_there_is_no_contact(service, session, tmp_path):
    from scripts.generate_test_data import render_pdf

    ingestion, drive = service
    path = tmp_path / "anon.pdf"
    render_pdf(["SKILLS", "Python, Docker, Kubernetes, PostgreSQL"], path)
    outcome = ingestion.ingest_one(drive.add_file("f1", "anon", "anon.pdf", path.read_bytes()))
    candidate = session.get(Candidate, outcome.candidate_id)
    assert candidate.canonical_key.startswith("anon|")


def test_a_bad_file_does_not_abort_the_whole_sync(service, session, pdf_bytes):
    ingestion, drive = service
    drive.add_file("f1", "good", "cv.pdf", pdf_bytes)
    drive.add_file("f1", "bad", "broken.pdf", b"definitely not a pdf")
    report = ingestion.ingest_files(drive.list_files("f1"))
    assert report.total == 2
    assert report.ingested == 1
    assert report.errors == 1
    assert report.outcomes[1].status == "error"


def test_sync_reports_progress(service, pdf_bytes, tmp_path):
    from scripts.generate_test_data import render_pdf

    ingestion, drive = service
    for i in range(3):
        path = tmp_path / f"p{i}.pdf"
        render_pdf([f"Person {i}", f"p{i}@example.com", "SKILLS", "Python"], path)
        drive.add_file("f1", f"id{i}", path.name, path.read_bytes())
    seen: list[tuple[int, int]] = []
    ingestion.ingest_files(drive.list_files("f1"), progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_sync_only_reads_the_folders_assigned_to_the_job(service, session, pdf_bytes, tmp_path):
    from scripts.generate_test_data import render_pdf

    ingestion, drive = service
    drive.add_folder("f2", "Other")
    drive.add_file("f1", "in", "cv.pdf", pdf_bytes)
    other = tmp_path / "other.pdf"
    render_pdf(["Other Person", "other@example.com", "SKILLS", "Python"], other)
    drive.add_file("f2", "out", "other.pdf", other.read_bytes())

    job = Job(title="Backend", raw_jd_text="x")
    session.add(job)
    session.commit()
    session.add(JobFolder(job_id=job.id, folder_id="f1"))
    session.commit()
    report = ingestion.sync_job(job.id)
    assert report.total == 1


def test_syncing_stamps_the_folder(service, session, pdf_bytes):
    ingestion, drive = service
    ingestion.refresh_folders()
    drive.add_file("f1", "a", "cv.pdf", pdf_bytes)
    job = Job(title="Backend", raw_jd_text="x")
    session.add(job)
    session.commit()
    session.add(JobFolder(job_id=job.id, folder_id="f1"))
    session.commit()
    ingestion.sync_job(job.id)
    folder = session.scalar(select(DriveFolder).where(DriveFolder.folder_id == "f1"))
    assert folder.last_synced_at is not None


def test_the_sync_report_serialises(service, pdf_bytes):
    ingestion, drive = service
    drive.add_file("f1", "a", "cv.pdf", pdf_bytes)
    payload = ingestion.ingest_files(drive.list_files("f1")).as_dict()
    assert payload["total"] == 1 and payload["ingested"] == 1
    assert payload["outcomes"][0]["status"] == "ingested"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Layla Haddad\nlayla@example.com", "Layla Haddad"),
        ("layla@example.com\nLayla Haddad", "Layla Haddad"),
        ("SKILLS:\nPython", None),
        ("", None),
    ],
)
def test_name_guessing(text, expected):
    assert _guess_name(text) == expected
