"""Run three synthetic CVs through the REAL Gemini API.

This is the only test that needs credentials. It skips cleanly with a clear
message when ``GEMINI_API_KEYS`` is not set, so it is safe to run anywhere.

What it exercises, against the live API:
  * JD structuring    (structured output with a response schema)
  * CV extraction     (structured output + evidence verification)
  * embeddings        (the semantic scorer's vectors)
  * the LLM judge     (rubric scoring on an identity-masked record)
  * key rotation      (every call goes through the one gateway)

Run:
    python scripts/live_smoke_test.py
    python scripts/live_smoke_test.py --keep     # keep the scratch database
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SMOKE_CVS = ("strong_backend_1", "mismatch_1", "injection_attempt")
EXIT_SKIPPED = 0
EXIT_FAILED = 1


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def skip(reason: str) -> int:
    print("\n" + "-" * 72)
    print("LIVE SMOKE TEST SKIPPED")
    print("-" * 72)
    print(reason)
    print(
        "\nTo run it, put one or more real Gemini API keys in .env:\n"
        "    GEMINI_API_KEYS=key-one,key-two\n"
        "    MOCK_MODE=false\n"
        "and run this script again. Everything else in the test suite runs\n"
        "without credentials."
    )
    return EXIT_SKIPPED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the scratch database")
    args = parser.parse_args()

    import os

    workdir = Path(tempfile.mkdtemp(prefix="cv-live-smoke-"))
    os.environ["DATABASE_URL"] = f"sqlite:///{(workdir / 'smoke.db').as_posix()}"
    os.environ["STORAGE_DIR"] = str(workdir / "storage")
    os.environ["MOCK_MODE"] = "false"

    from backend.app import config as config_module

    config_module.reset_settings_cache()
    settings = config_module.get_settings()

    if not settings.api_key_list:
        shutil.rmtree(workdir, ignore_errors=True)
        return skip("GEMINI_API_KEYS is empty, so there is no live API to call.")

    try:
        from backend.app.core.gemini_transport import GenaiTransport
    except Exception as exc:  # pragma: no cover - environment dependent
        shutil.rmtree(workdir, ignore_errors=True)
        return skip(f"the google-genai SDK is not importable: {exc}")

    from backend.app import db as db_module
    from backend.app.core.gemini_client import (
        AllKeysExhausted,
        GeminiClient,
        ModelConfig,
    )
    from backend.app.core.ingestion import IngestionService
    from backend.app.core.jd import JDService, load_structured
    from backend.app.core.pipeline import ScreeningPipeline
    from backend.app.models import CVFile, Job, JobFolder
    from backend.app.core.drive_client import InMemoryDriveClient
    from scripts.generate_test_data import JOB_DESCRIPTIONS, build_corpus, generate

    db_module.reset()
    db_module.init_db()

    banner("Live Gemini smoke test")
    print(f"keys in pool : {len(settings.api_key_list)}")
    print(f"extraction   : {settings.gemini_extraction_model}")
    print(f"judge        : {settings.gemini_judge_model}")
    print(f"embeddings   : {settings.gemini_embedding_model}")
    print(f"scratch dir  : {workdir}")

    client = GeminiClient(
        transport=GenaiTransport(),
        keys=settings.api_key_list,
        models=ModelConfig(
            extraction=settings.gemini_extraction_model,
            judge=settings.gemini_judge_model,
            embedding=settings.gemini_embedding_model,
            vision=settings.gemini_vision_model,
            embedding_dim=settings.gemini_embedding_dim,
        ),
        base_cooldown=settings.gemini_cooldown_base_seconds,
        max_cooldown=settings.gemini_cooldown_max_seconds,
    )

    generate(workdir / "synthetic")
    cv_dir = workdir / "synthetic" / "cvs"
    wanted = {cv.key: cv for cv in build_corpus() if cv.key in SMOKE_CVS}

    drive = InMemoryDriveClient()
    drive.add_folder("live-folder", "Live smoke test")
    for key, cv in wanted.items():
        drive.add_file("live-folder", f"live:{key}", cv.filename, (cv_dir / cv.filename).read_bytes())

    failures: list[str] = []
    session = db_module.get_session_factory()()
    try:
        job = Job(
            title=JOB_DESCRIPTIONS["backend"]["title"],
            raw_jd_text=JOB_DESCRIPTIONS["backend"]["text"],
        )
        session.add(job)
        session.commit()
        session.add(JobFolder(job_id=job.id, folder_id="live-folder"))
        session.commit()

        banner("1/4  Structuring the job description with Gemini")
        jd_service = JDService(session, client)
        version = jd_service.structure(job)
        structured = load_structured(version)
        print(f"title        : {structured['title']}")
        print(f"must have    : {[m['canonical'] for m in structured['must_have']]}")
        print(f"nice to have : {[m['canonical'] for m in structured['nice_to_have']]}")
        print(f"thresholds   : {structured['thresholds']}")
        if not structured["must_have"]:
            failures.append("the model returned no must-have requirements")
        jd_service.approve(job, version, actor="live-smoke-test")

        banner("2/4  Ingesting and extracting three CVs")
        ingestion = IngestionService(
            session, drive, client, raw_dir=settings.raw_dir, text_dir=settings.text_dir
        )
        ingestion.refresh_folders()
        report = ingestion.sync_job(job.id)
        print(f"ingested     : {report.ingested} of {report.total}")
        if report.errors:
            failures.append(f"{report.errors} file(s) failed to ingest")

        banner("3/4  Scoring against the approved job description")
        pipeline = ScreeningPipeline(session, client, settings=settings)
        outcomes = pipeline.screen_job(job, version, actor="live-smoke-test")

        print(f"\n{'cv':<24}{'score':>8}{'conf':>8}  routing")
        print("-" * 72)
        by_key = {}
        for outcome in outcomes:
            cv_file = session.get(CVFile, outcome.cv_file_id)
            key = next((k for k, cv in wanted.items() if cv.filename == cv_file.filename), cv_file.filename)
            by_key[key] = outcome
            print(f"{key:<24}{outcome.score:>8.1f}{outcome.confidence:>8.2f}  {outcome.routing}")
            if outcome.flags:
                print(f"{'':<24}{'':>16}  flags: {', '.join(outcome.flags)}")

        banner("4/4  Checking the live results against what the design requires")
        checks = [
            (
                "the strong CV outscores the unrelated CV",
                "strong_backend_1" in by_key
                and "mismatch_1" in by_key
                and by_key["strong_backend_1"].score > by_key["mismatch_1"].score,
            ),
            (
                "the injection CV is flagged",
                "injection_attempt" in by_key
                and "injection_suspicion" in by_key["injection_attempt"].flags,
            ),
            (
                "the injection CV was not shortlisted by its own instruction",
                "injection_attempt" in by_key
                and by_key["injection_attempt"].routing != "auto_shortlist",
            ),
            (
                "no candidate was finally rejected by the machine",
                all(o.routing != "rejected" for o in outcomes),
            ),
            ("every CV produced a score", all(o.score >= 0 for o in outcomes)),
        ]
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                failures.append(label)

        banner("Gemini key pool after the run")
        for status in client.key_pool_status():
            print(
                f"  key[{status['index']}] {status['last4']}  "
                f"requests={status['requests']}  failures={status['failures']}  "
                f"rate_limit_hits={status['rate_limit_hits']}"
            )

        report_path = workdir / "live_smoke_result.json"
        report_path.write_text(
            json.dumps(
                {
                    "structured_jd": structured,
                    "outcomes": {k: o.as_dict() for k, o in by_key.items()},
                    "key_pool": client.key_pool_status(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nfull result written to {report_path}")

    except AllKeysExhausted as exc:
        print(f"\nEvery key is cooling down: {exc}")
        print("This is the designed behaviour under quota pressure, not a crash.")
        failures.append("all keys exhausted")
    finally:
        session.close()
        if not args.keep:
            db_module.reset()
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"\nscratch directory kept at {workdir}")

    banner("RESULT")
    if failures:
        print("LIVE SMOKE TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_FAILED
    print("LIVE SMOKE TEST PASSED — the real Gemini API drives the whole pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
