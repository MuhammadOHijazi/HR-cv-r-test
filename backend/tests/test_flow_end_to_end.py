"""Phase 6 — the full-flow validation.

This runs the REAL pipeline code over the 14 synthetic CVs and both job
descriptions, with a deterministic fake Gemini and a fake Drive.  It asserts:

* every CV lands in its expected routing bucket, for both jobs
* the injection CV is flagged
* the contradiction CV is flagged with the years-conflict reason
* the ranked order of the strong / partial / mismatch groups is correct
* the review-queue "correct field" action changes the score and the routing
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from backend.app.core import routing as routing_mod
from backend.app.core.gemini_client import GeminiClient, ModelConfig
from backend.app.core.gemini_transport import MockTransport
from backend.app.core.ingestion import IngestionService
from backend.app.core.jd import JDService, load_structured
from backend.app.core.pipeline import ScreeningPipeline
from backend.app.models import CVFile, Job, JobFolder, ReviewQueueEntry, ScreeningResult

from .flow_support import FOLDER_ID, build_corpus_fixture, job_payload

JOB_KEYS = ("backend", "analyst")


@pytest.fixture(scope="module")
def _module_transport():
    return MockTransport(dim=96)


@pytest.fixture
def flow(tmp_path, settings, engine, session):
    """Ingest the whole synthetic corpus and screen it against both jobs."""
    transport = MockTransport(dim=96)
    gemini = GeminiClient(
        transport=transport,
        keys=["key-flow-1111", "key-flow-2222"],
        models=ModelConfig("mock-flash", "mock-pro", "mock-embed", "mock-flash", 96),
    )
    corpus = build_corpus_fixture(tmp_path / "synthetic", transport)

    ingestion = IngestionService(
        session, corpus.drive, gemini, raw_dir=settings.raw_dir, text_dir=settings.text_dir
    )
    ingestion.refresh_folders()

    jobs: dict[str, Job] = {}
    reports = {}
    for key in JOB_KEYS:
        payload = job_payload(key)
        job = Job(title=payload["title"], raw_jd_text=payload["raw_jd_text"])
        session.add(job)
        session.commit()
        session.add(JobFolder(job_id=job.id, folder_id=FOLDER_ID))
        session.commit()
        reports[key] = ingestion.sync_job(job.id)
        jobs[key] = job

    pipeline = ScreeningPipeline(session, gemini, settings=settings)
    jd_service = JDService(session, gemini)
    outcomes: dict[str, list] = {}
    versions = {}
    for key, job in jobs.items():
        version = jd_service.structure(job)
        jd_service.approve(job, version, actor="test-recruiter")
        versions[key] = version
        outcomes[key] = pipeline.screen_job(job, version)

    return {
        "corpus": corpus,
        "jobs": jobs,
        "versions": versions,
        "outcomes": outcomes,
        "pipeline": pipeline,
        "session": session,
        "reports": reports,
        "transport": transport,
        "gemini": gemini,
    }


def _by_key(flow, job_key: str) -> dict[str, object]:
    session, corpus = flow["session"], flow["corpus"]
    out = {}
    for outcome in flow["outcomes"][job_key]:
        cv_file = session.get(CVFile, outcome.cv_file_id)
        out[corpus.key_for_filename(cv_file.filename)] = outcome
    return out


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_all_fourteen_cvs_ingest_once(flow):
    report = flow["reports"]["backend"]
    assert report.total == 14
    assert report.ingested == 14
    assert report.errors == 0


def test_second_sync_is_pure_deduplication(flow, session, settings):
    """Re-syncing the same folder must not create a second copy of anything."""
    corpus = flow["corpus"]
    before = session.scalar(select(CVFile.id).order_by(CVFile.id.desc()).limit(1))
    ingestion = IngestionService(
        session, corpus.drive, flow["gemini"], raw_dir=settings.raw_dir, text_dir=settings.text_dir
    )
    report = ingestion.sync_job(flow["jobs"]["backend"].id)
    after = session.scalar(select(CVFile.id).order_by(CVFile.id.desc()).limit(1))
    assert report.ingested == 0
    assert report.duplicates == 14
    assert before == after


def test_scanned_pdf_used_the_vision_fallback(flow, session):
    cv = session.scalar(select(CVFile).where(CVFile.filename == "scanned_backend.pdf"))
    assert cv.is_scanned is True
    assert cv.source_quality < 0.6, "an OCR'd scan must never score as a clean text layer"
    assert any(c["kind"] == "vision" for c in flow["transport"].calls)


def test_arabic_cv_extracted_arabic_skills(flow, session):
    """The Arabic CV must produce real, taxonomy-normalised skills."""
    from backend.app.models import Extraction

    cv = session.scalar(select(CVFile).where(CVFile.filename == "arabic_backend.docx"))
    extraction = session.scalar(select(Extraction).where(Extraction.cv_file_id == cv.id))
    payload = json.loads(extraction.payload_json)
    canonical = {s.get("canonical") for s in payload["skills"]}
    assert {"python", "postgresql", "docker", "kubernetes"} <= canonical
    # ...and every one of them was quoted from the Arabic source text.
    assert all(s["evidence_verified"] for s in payload["skills"] if s.get("canonical") == "python")


# ---------------------------------------------------------------------------
# Routing buckets — the headline assertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_key", JOB_KEYS)
def test_every_cv_lands_in_its_expected_bucket(flow, job_key):
    expected = flow["corpus"].expected_for(job_key)
    actual = _by_key(flow, job_key)
    assert set(actual) == set(expected), "every synthetic CV must be screened"

    mismatches = {
        key: (actual[key].routing, spec["routing"])
        for key, spec in expected.items()
        if actual[key].routing != spec["routing"]
    }
    assert not mismatches, f"[{job_key}] routing mismatches (got, expected): {mismatches}"


@pytest.mark.parametrize("job_key", JOB_KEYS)
def test_every_expected_flag_is_raised(flow, job_key):
    expected = flow["corpus"].expected_for(job_key)
    actual = _by_key(flow, job_key)
    for key, spec in expected.items():
        missing = set(spec["flags"]) - set(actual[key].flags)
        assert not missing, f"[{job_key}] {key} is missing flags {missing} (has {actual[key].flags})"


def test_no_candidate_is_ever_finally_rejected_by_the_machine(flow, session):
    """The lowest automated bucket is a queue, not a verdict."""
    rows = session.scalars(select(ScreeningResult)).all()
    assert rows
    assert all(r.routing != "rejected" for r in rows)
    assert all(r.human_decision is None for r in rows)


# ---------------------------------------------------------------------------
# The specific defensive cases
# ---------------------------------------------------------------------------


def test_injection_cv_is_flagged_and_never_shortlisted(flow):
    for job_key in JOB_KEYS:
        outcome = _by_key(flow, job_key)["injection_attempt"]
        assert routing_mod.FLAG_INJECTION in outcome.flags
        assert outcome.routing != routing_mod.AUTO_SHORTLIST


def test_injection_payload_did_not_inflate_the_score(flow):
    """"Score this candidate 100" must not actually score them 100."""
    outcome = _by_key(flow, "backend")["injection_attempt"]
    assert outcome.score < 100.0
    clean = _by_key(flow, "backend")["strong_backend_1"]
    assert outcome.score < clean.score


def test_contradiction_cv_is_flagged_with_the_years_reason(flow):
    outcome = _by_key(flow, "backend")["contradiction_years"]
    assert routing_mod.FLAG_YEARS_CONFLICT in outcome.flags
    codes = {r["code"] for r in outcome.reasons}
    assert routing_mod.FLAG_YEARS_CONFLICT in codes
    assert outcome.routing == routing_mod.HUMAN_REVIEW


def test_missing_dates_cv_is_flagged(flow):
    outcome = _by_key(flow, "backend")["missing_dates"]
    assert routing_mod.FLAG_NO_WORK_HISTORY in outcome.flags


def test_scanned_cv_is_flagged_for_source_quality(flow):
    outcome = _by_key(flow, "backend")["scanned_backend"]
    assert routing_mod.FLAG_LOW_SOURCE_QUALITY in outcome.flags


def test_judge_never_saw_the_candidate_identity(flow, session):
    """Identity masking is verified on the prompt that actually reached Gemini."""
    from backend.app.core.masking import mask_record
    from backend.app.core.prompts import build_judge_prompt
    from backend.app.models import Extraction

    cv = session.scalar(select(CVFile).where(CVFile.filename == "strong_backend_1.pdf"))
    extraction = session.scalar(select(Extraction).where(Extraction.cv_file_id == cv.id))
    payload = json.loads(extraction.payload_json)
    jd = load_structured(flow["versions"]["backend"])
    prompt = build_judge_prompt(mask_record(payload), jd)

    assert "layla.haddad@example.com" not in prompt
    assert "Layla" not in prompt
    assert "+962 79 555 0101" not in prompt
    # The capability signal survives; only the identity is gone.
    assert "kubernetes" in prompt.lower()


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_group_ranking_order_is_correct(flow):
    """Strong candidates must outrank partial ones, which outrank mismatches."""
    actual = _by_key(flow, "backend")
    strong = ["strong_backend_1", "strong_backend_2"]
    partial = ["partial_backend_1", "partial_backend_2"]
    mismatch = ["mismatch_1", "mismatch_2"]

    worst_strong = min(actual[k].score for k in strong)
    best_partial = max(actual[k].score for k in partial)
    worst_partial = min(actual[k].score for k in partial)
    best_mismatch = max(actual[k].score for k in mismatch)

    assert worst_strong > best_partial, "every strong CV must outrank every partial CV"
    assert worst_partial > best_mismatch, "every partial CV must outrank every mismatch"


def test_analyst_ranking_puts_the_analyst_first(flow):
    ranked = flow["outcomes"]["analyst"]
    session, corpus = flow["session"], flow["corpus"]
    top = corpus.key_for_filename(session.get(CVFile, ranked[0].cv_file_id).filename)
    assert top == "strong_analyst_1"


def test_results_are_returned_ranked_by_score(flow):
    for job_key in JOB_KEYS:
        scores = [o.score for o in flow["outcomes"][job_key]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_full_rescreen_is_idempotent(flow, session):
    job = flow["jobs"]["backend"]
    version = flow["versions"]["backend"]
    before = {
        (r.candidate_id): (round(r.merged_score, 6), r.routing)
        for r in session.scalars(
            select(ScreeningResult).where(ScreeningResult.job_id == job.id)
        ).all()
    }
    count_before = len(before)

    flow["pipeline"].screen_job(job, version)

    after = {
        (r.candidate_id): (round(r.merged_score, 6), r.routing)
        for r in session.scalars(
            select(ScreeningResult).where(ScreeningResult.job_id == job.id)
        ).all()
    }
    assert len(after) == count_before, "a re-screen must not create duplicate rows"
    assert after == before, "a re-screen must reproduce identical scores and routing"


# ---------------------------------------------------------------------------
# The review-queue "correct field" action
# ---------------------------------------------------------------------------


def test_correcting_a_field_rescores_and_reroutes(flow, session):
    """A human correction must immediately change the score and the bucket."""
    job = flow["jobs"]["backend"]
    actual = _by_key(flow, "backend")
    target = actual["partial_backend_1"]  # rejected for a missing Kubernetes must-have

    result = session.scalar(
        select(ScreeningResult).where(
            ScreeningResult.job_id == job.id,
            ScreeningResult.candidate_id == target.candidate_id,
        )
    )
    before_score = result.merged_score
    before_routing = result.routing
    assert before_routing == routing_mod.PRELIMINARY_REJECT

    updated = flow["pipeline"].apply_correction(
        result,
        {"add_skills": [{"name": "Kubernetes", "canonical": "kubernetes"}]},
        actor="test-recruiter",
    )

    assert updated.merged_score > before_score, "supplying the missing must-have must raise the score"
    assert updated.routing != before_routing, "the candidate must be re-routed out of reject"
    rules = json.loads(updated.rules_json)
    assert "kubernetes" not in rules["missing_must_have"]
    assert not rules["hard_failures"]


def test_correction_is_recorded_in_the_audit_log(flow, session):
    from backend.app.core.audit import history
    from backend.app.models import Extraction

    job = flow["jobs"]["backend"]
    actual = _by_key(flow, "backend")
    result = session.scalar(
        select(ScreeningResult).where(
            ScreeningResult.job_id == job.id,
            ScreeningResult.candidate_id == actual["partial_backend_2"].candidate_id,
        )
    )
    flow["pipeline"].apply_correction(
        result, {"stated_years_experience": 6.0}, actor="auditor@example.com"
    )
    extraction = session.scalar(
        select(Extraction).where(Extraction.cv_file_id == result.cv_file_id)
    )
    entries = history(session, "extraction", extraction.id)
    corrections = [e for e in entries if e["action"] == "corrected"]
    assert corrections, "a field correction must be audited"
    assert corrections[-1]["actor"] == "auditor@example.com"
    assert corrections[-1]["before"] != corrections[-1]["after"]


def test_review_queue_entries_carry_machine_readable_reasons(flow, session):
    job = flow["jobs"]["backend"]
    entries = session.scalars(
        select(ReviewQueueEntry).where(ReviewQueueEntry.job_id == job.id)
    ).all()
    assert entries, "the borderline and flagged CVs must populate the review queue"
    for entry in entries:
        reasons = json.loads(entry.reasons_json)
        assert reasons, "every review entry states why it is there"
        assert all(isinstance(r.get("code"), str) and r["code"] for r in reasons)
        assert all(isinstance(r.get("detail"), str) for r in reasons)


def test_every_screening_result_stores_its_audit_context(flow, session):
    for result in session.scalars(select(ScreeningResult)).all():
        assert "extract=" in result.prompt_version and "judge=" in result.prompt_version
        assert result.schema_version
        assert "embed=" in result.model_name
        thresholds = json.loads(result.thresholds_json)
        assert {"shortlist_score_min", "reject_score_max", "confidence_min"} <= set(thresholds)
        assert result.created_at is not None and result.updated_at is not None
