"""The three scripts in ``scripts/``: data generation, taxonomy seeding, smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.models import SkillTaxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# generate_test_data.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from scripts.generate_test_data import generate

    out = tmp_path_factory.mktemp("corpus")
    return generate(out), out


def test_the_corpus_has_fourteen_cvs(corpus):
    manifest, _ = corpus
    assert manifest["counts"]["total"] == 14


def test_the_corpus_covers_every_required_group(corpus):
    manifest, _ = corpus
    counts = manifest["counts"]
    assert counts["strong"] == 3
    assert counts["partial"] == 3
    assert counts["mismatch"] == 2
    assert counts["special"] == 6


def test_the_corpus_defines_two_job_descriptions(corpus):
    manifest, _ = corpus
    assert set(manifest["jobs"]) == {"backend", "analyst"}
    for spec in manifest["jobs"].values():
        assert spec["title"] and "Must have" in spec["text"]


def test_every_cv_is_a_real_file_on_disk(corpus):
    manifest, out = corpus
    for cv in manifest["cvs"]:
        path = out / "cvs" / cv["filename"]
        assert path.is_file(), cv["filename"]
        assert path.stat().st_size > 400, f"{cv['filename']} looks empty"


def test_every_cv_ships_an_expected_outcome(corpus):
    manifest, _ = corpus
    buckets = {"auto_shortlist", "human_review", "preliminary_reject"}
    for cv in manifest["cvs"]:
        assert set(cv["expected"]) == {"backend", "analyst"}
        for spec in cv["expected"].values():
            assert spec["routing"] in buckets
            assert isinstance(spec["flags"], list)
        assert cv["note"], f"{cv['key']} should explain what it tests"


def test_the_corpus_covers_the_required_special_cases(corpus):
    manifest, _ = corpus
    keys = {cv["key"] for cv in manifest["cvs"]}
    assert {
        "arabic_backend",
        "mixed_backend",
        "contradiction_years",
        "missing_dates",
        "injection_attempt",
        "scanned_backend",
    } <= keys


def test_both_document_formats_are_produced(corpus):
    manifest, _ = corpus
    formats = {cv["format"] for cv in manifest["cvs"]}
    assert {"pdf", "docx", "scanned_pdf"} == formats


def test_the_generator_is_deterministic(tmp_path):
    from scripts.generate_test_data import generate

    first = generate(tmp_path / "a")
    second = generate(tmp_path / "b")
    assert first["cvs"] == second["cvs"]
    text_a = (tmp_path / "a" / "cvs" / "strong_backend_1.txt").read_text(encoding="utf-8")
    text_b = (tmp_path / "b" / "cvs" / "strong_backend_1.txt").read_text(encoding="utf-8")
    assert text_a == text_b


def test_the_injection_cv_actually_contains_a_payload(corpus):
    from backend.app.core.injection import scan

    _, out = corpus
    text = (out / "cvs" / "injection_attempt.txt").read_text(encoding="utf-8")
    assert scan(text).suspected


def test_the_arabic_cv_is_actually_arabic(corpus):
    _, out = corpus
    text = (out / "cvs" / "arabic_backend.txt").read_text(encoding="utf-8")
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    assert arabic > 100, "the Arabic CV must be predominantly Arabic"


def test_the_mixed_cv_contains_both_scripts(corpus):
    _, out = corpus
    text = (out / "cvs" / "mixed_backend.txt").read_text(encoding="utf-8")
    assert any("؀" <= ch <= "ۿ" for ch in text)
    assert any("a" <= ch.lower() <= "z" for ch in text)


def test_the_contradiction_cv_really_contradicts_itself(corpus):
    from backend.app.core import regexlayer

    _, out = corpus
    text = (out / "cvs" / "contradiction_years.txt").read_text(encoding="utf-8")
    findings = regexlayer.run(text)
    assert findings.stated_years is not None
    assert findings.computed_years is not None
    assert abs(findings.stated_years - findings.computed_years) > 1.5


def test_the_missing_dates_cv_really_has_no_dates(corpus):
    from backend.app.core import regexlayer

    _, out = corpus
    text = (out / "cvs" / "missing_dates.txt").read_text(encoding="utf-8")
    assert regexlayer.run(text).date_ranges == []


def test_the_scanned_pdf_really_has_no_text_layer(corpus):
    from backend.app.core import textextract

    _, out = corpus
    data = (out / "cvs" / "scanned_backend.pdf").read_bytes()
    assert textextract.extract(data, "scanned_backend.pdf").is_scanned is True


def test_the_generator_runs_as_a_command(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_test_data.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    assert "14 synthetic CVs" in result.stdout
    assert (tmp_path / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# seed_taxonomy.py
# ---------------------------------------------------------------------------


def test_seeding_writes_the_whole_taxonomy(engine, session):
    from scripts.seed_taxonomy import seed

    assert seed() >= 100
    rows = session.scalars(select(SkillTaxonomy)).all()
    assert len(rows) >= 100
    canonical = {r.canonical for r in rows}
    assert {"python", "kubernetes", "power bi"} <= canonical


def test_seeding_twice_is_a_no_op(engine):
    from scripts.seed_taxonomy import seed

    seed()
    assert seed() == 0


def test_forced_seeding_re_syncs(engine):
    from scripts.seed_taxonomy import seed

    seed()
    assert seed(force=True) >= 100


def test_a_skill_can_be_added_at_runtime(engine, session):
    from scripts.seed_taxonomy import add

    add("quantum computing:ml:qc,الحوسبة الكمية")
    row = session.scalar(
        select(SkillTaxonomy).where(SkillTaxonomy.canonical == "quantum computing")
    )
    assert row is not None
    assert json.loads(row.aliases_json) == ["qc", "الحوسبة الكمية"]


def test_a_runtime_skill_is_usable_by_the_taxonomy(engine):
    from scripts.seed_taxonomy import add, seed

    seed()
    add("quantum computing:ml:qc,الحوسبة الكمية")
    from backend.app.scripts_support import load_taxonomy

    taxonomy = load_taxonomy()
    assert taxonomy.normalise("QC") == "quantum computing"
    assert taxonomy.normalise("الحوسبة الكمية") == "quantum computing"


def test_a_malformed_add_specification_is_rejected(engine):
    from scripts.seed_taxonomy import add

    with pytest.raises(SystemExit):
        add("no-category-here")


# ---------------------------------------------------------------------------
# live_smoke_test.py
# ---------------------------------------------------------------------------


def test_the_live_smoke_test_skips_cleanly_without_keys(tmp_path):
    """The credential-gated script must never fail a credential-free run."""
    env = {
        **_clean_env(),
        "GEMINI_API_KEYS": "",
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'x.db').as_posix()}",
        "STORAGE_DIR": str(tmp_path / "storage"),
    }
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "live_smoke_test.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "LIVE SMOKE TEST SKIPPED" in result.stdout
    assert "GEMINI_API_KEYS" in result.stdout


def test_the_live_smoke_test_names_the_three_cvs_it_runs():
    from scripts.live_smoke_test import SMOKE_CVS

    from scripts.generate_test_data import build_corpus

    keys = {cv.key for cv in build_corpus()}
    assert len(SMOKE_CVS) == 3
    assert set(SMOKE_CVS) <= keys


def _clean_env() -> dict[str, str]:
    import os

    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GEMINI_", "GOOGLE_", "DATABASE_", "STORAGE_", "MOCK_"))
    }
