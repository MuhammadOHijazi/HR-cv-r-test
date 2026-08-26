"""Phase 3 gate — validator, evidence verifier, years cross-check, retry loop."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from backend.app.core import regexlayer
from backend.app.core.evidence import (
    EvidenceCheck,
    apply_to_confidence,
    normalise,
    verification_rate,
    verify_quote,
)
from backend.app.core.extraction import (
    Extractor,
    _repair_prompt,
    validate_payload,
)
from backend.app.core.gemini_client import GeminiClient, ModelConfig
from backend.app.core.taxonomy import DEFAULT_TAXONOMY, SkillEntry, Taxonomy

# The fixture is anchored to the current year so that "9 years of experience"
# always agrees with the date range, whatever year the suite runs in.
START_YEAR = dt.date.today().year - 9
GRAD_YEAR = START_YEAR - 1

SOURCE = f"""Layla Haddad
layla.haddad@example.com | +962 79 555 0101

SUMMARY
Senior backend engineer with 9 years of experience building Python services.

SKILLS
Python, PostgreSQL, Docker, Kubernetes

EXPERIENCE
Principal Backend Engineer, Nimbus Systems, {START_YEAR} - present
- Design and build scalable Python microservices for high-traffic APIs

EDUCATION
Bachelor of Science in Computer Science, University of Jordan, {GRAD_YEAR}
"""


class QueueTransport:
    """Returns each queued response in turn; repeats the last one thereafter."""

    def __init__(self, responses):
        self.responses = [
            r if isinstance(r, str) else json.dumps(r) for r in responses
        ]
        self.prompts: list[str] = []

    def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]

    def embed(self, *, api_key, model, texts):
        return [[0.0] * 8 for _ in texts]


def make_extractor(responses, **kwargs):
    transport = QueueTransport(responses)
    client = GeminiClient(
        transport=transport,
        keys=["k-aaaa1111", "k-bbbb2222"],
        models=ModelConfig("flash", "pro", "embed", "flash", 8),
        clock=lambda: 0.0,
    )
    return Extractor(client, **kwargs), transport


def valid_payload(**overrides):
    payload = {
        "full_name": "Layla Haddad",
        "email": "layla.haddad@example.com",
        "phone": "+962795550101",
        "stated_years_experience": 9.0,
        "skills": [
            {
                "name": "Python",
                "level": None,
                "evidence_quote": "Python, PostgreSQL, Docker, Kubernetes",
                "confidence": 0.9,
            }
        ],
        "education": [
            {
                "degree": "bachelor",
                "field": "Computer Science",
                "institution": "University of Jordan",
                "graduation_year": GRAD_YEAR,
                "evidence_quote": f"Bachelor of Science in Computer Science, University of Jordan, {GRAD_YEAR}",
                "confidence": 0.9,
            }
        ],
        "work_history": [
            {
                "title": "Principal Backend Engineer",
                "company": "Nimbus Systems",
                "from_date": f"{START_YEAR}-01-01",
                "to_date": None,
                "highlights": ["Design and build scalable Python microservices for high-traffic APIs"],
                "evidence_quote": f"Principal Backend Engineer, Nimbus Systems, {START_YEAR} - present",
                "confidence": 0.9,
            }
        ],
        "languages": ["English"],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_a_well_formed_payload_validates():
    assert validate_payload(valid_payload()) == []


def test_top_level_must_be_an_object():
    assert "top level must be a JSON object" in validate_payload([1, 2, 3])[0]


@pytest.mark.parametrize("key", ["skills", "education", "work_history"])
def test_missing_required_keys_are_reported(key):
    payload = valid_payload()
    del payload[key]
    assert any(f"missing required key '{key}'" in p for p in validate_payload(payload))


@pytest.mark.parametrize("key", ["skills", "education", "work_history"])
def test_required_keys_must_be_arrays(key):
    problems = validate_payload(valid_payload(**{key: {"not": "a list"}}))
    assert any(f"'{key}' must be an array" in p for p in problems)


def test_contact_fields_must_be_strings_or_null():
    problems = validate_payload(valid_payload(email=12345))
    assert any("'email' must be a string or null" in p for p in problems)


def test_null_contact_fields_are_allowed():
    assert validate_payload(valid_payload(email=None, phone=None, full_name=None)) == []


def test_stated_years_must_be_a_number():
    problems = validate_payload(valid_payload(stated_years_experience="nine"))
    assert any("must be a number or null" in p for p in problems)


def test_stated_years_must_be_in_a_human_range():
    problems = validate_payload(valid_payload(stated_years_experience=180))
    assert any("between 0 and 60" in p for p in problems)


def test_a_skill_needs_a_name():
    payload = valid_payload()
    payload["skills"][0]["name"] = "   "
    assert any("skills[0].name" in p for p in validate_payload(payload))


def test_a_skill_item_must_be_an_object():
    assert any("skills[0] must be an object" in p for p in validate_payload(valid_payload(skills=["Python"])))


def test_confidence_is_required():
    payload = valid_payload()
    del payload["skills"][0]["confidence"]
    assert any("skills[0].confidence is required" in p for p in validate_payload(payload))


@pytest.mark.parametrize("value", [-0.1, 1.4])
def test_confidence_must_be_within_zero_and_one(value):
    payload = valid_payload()
    payload["skills"][0]["confidence"] = value
    assert any("within [0,1]" in p for p in validate_payload(payload))


def test_confidence_must_be_numeric():
    payload = valid_payload()
    payload["skills"][0]["confidence"] = "high"
    assert any("must be a number" in p for p in validate_payload(payload))


def test_dates_must_be_iso_formatted():
    payload = valid_payload()
    payload["work_history"][0]["from_date"] = "March 2016"
    assert any("ISO date YYYY-MM-DD" in p for p in validate_payload(payload))


def test_impossible_calendar_dates_are_rejected():
    payload = valid_payload()
    payload["work_history"][0]["from_date"] = "2016-02-31"
    assert any("not a real calendar date" in p for p in validate_payload(payload))


def test_an_end_date_before_the_start_is_rejected():
    payload = valid_payload()
    payload["work_history"][0]["from_date"] = "2020-01-01"
    payload["work_history"][0]["to_date"] = "2018-01-01"
    assert any("to_date is before from_date" in p for p in validate_payload(payload))


def test_a_null_end_date_means_a_current_role():
    assert validate_payload(valid_payload()) == []


def test_highlights_must_be_an_array():
    payload = valid_payload()
    payload["work_history"][0]["highlights"] = "one long string"
    assert any("highlights must be an array" in p for p in validate_payload(payload))


def test_graduation_year_must_be_plausible():
    payload = valid_payload()
    payload["education"][0]["graduation_year"] = 1200
    assert any("graduation_year" in p for p in validate_payload(payload))


def test_a_null_graduation_year_is_allowed():
    payload = valid_payload()
    payload["education"][0]["graduation_year"] = None
    assert validate_payload(payload) == []


def test_non_object_education_and_work_items_are_reported():
    problems = validate_payload(valid_payload(education=["BSc"], work_history=["job"]))
    assert any("education[0] must be an object" in p for p in problems)
    assert any("work_history[0] must be an object" in p for p in problems)


# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------


def test_an_exact_quote_verifies():
    check = verify_quote("Docker, Kubernetes", "Python, PostgreSQL, Docker, Kubernetes")
    assert check.verified and check.ratio == 1.0 and check.reason == "exact"


def test_whitespace_differences_still_verify():
    check = verify_quote("Docker,    Kubernetes", "Python, Docker, Kubernetes")
    assert check.verified


def test_case_differences_still_verify():
    assert verify_quote("DOCKER, KUBERNETES", "docker, kubernetes").verified


def test_a_near_miss_verifies_fuzzily():
    check = verify_quote(
        "Design and build scalable Python microservices for high traffic APIs", SOURCE
    )
    assert check.verified
    assert 0.8 <= check.ratio <= 1.0


def test_an_invented_quote_does_not_verify():
    check = verify_quote("Led the Mars colonisation programme", SOURCE)
    assert not check.verified and check.reason == "not_found"


def test_a_missing_quote_does_not_verify():
    check = verify_quote(None, SOURCE)
    assert not check.verified and check.reason == "missing_quote"


def test_an_empty_quote_does_not_verify():
    assert verify_quote("   ", SOURCE).reason == "missing_quote"


def test_no_source_text_means_nothing_verifies():
    assert verify_quote("anything", "").reason == "no_source_text"


def test_a_quote_of_only_punctuation_cannot_verify():
    assert verify_quote("...", "").verified is False


def test_a_quote_longer_than_the_source_is_handled():
    check = verify_quote("a very long quote indeed", "short")
    assert not check.verified


def test_arabic_diacritics_are_normalised_away():
    assert normalise("مُهَنْدِس") == normalise("مهندس")


def test_arabic_quotes_verify():
    source = "المهارات: بايثون، بوستجريس، دوكر، كوبرنيتس"
    assert verify_quote("بايثون، بوستجريس", source).verified


def test_an_unverifiable_quote_zeroes_the_confidence():
    check = EvidenceCheck("made up", False, 0.2, "not_found")
    assert apply_to_confidence(0.95, check) == 0.0


def test_a_verified_quote_keeps_the_confidence():
    check = EvidenceCheck("real", True, 1.0, "exact")
    assert apply_to_confidence(0.95, check) == 0.95


def test_verification_rate_of_nothing_is_zero():
    assert verification_rate([]) == 0.0


def test_verification_rate_counts_the_verified_share():
    checks = [
        EvidenceCheck("a", True, 1.0, "exact"),
        EvidenceCheck("b", False, 0.0, "not_found"),
    ]
    assert verification_rate(checks) == 0.5


# ---------------------------------------------------------------------------
# Regex layer
# ---------------------------------------------------------------------------


def test_regex_layer_finds_the_email_and_phone():
    findings = regexlayer.run(SOURCE)
    assert findings.emails == ["layla.haddad@example.com"]
    assert findings.phones[0].startswith("+962")


def test_regex_layer_finds_stated_years():
    findings = regexlayer.run(SOURCE)
    assert findings.stated_years == 9.0
    assert "9 years" in findings.stated_years_quote


def test_regex_layer_reads_arabic_years():
    years, _ = regexlayer.find_stated_years("خبرة 7 سنوات في تطوير البرمجيات")
    assert years == 7.0


def test_regex_layer_normalises_arabic_indic_digits():
    assert regexlayer.normalise_digits("٧ سنوات") == "7 سنوات"


def test_a_salary_figure_is_not_mistaken_for_years():
    years, _ = regexlayer.find_stated_years("Compensation: 90 years-equivalent bonus pool")
    assert years is None


@pytest.mark.parametrize(
    "token,expected",
    [
        ("2016", dt.date(2016, 1, 1)),
        ("03/2019", dt.date(2019, 3, 1)),
        ("2019-07", dt.date(2019, 7, 1)),
        ("March 2018", dt.date(2018, 3, 1)),
        ("Sept 2018", dt.date(2018, 9, 1)),
        ("present", None),
        ("", None),
        ("garbage", None),
    ],
)
def test_date_token_parsing(token, expected):
    assert regexlayer.parse_date_token(token) == expected


def test_overlapping_employment_is_not_double_counted():
    ranges = [
        regexlayer.DateRange(dt.date(2016, 1, 1), dt.date(2020, 1, 1), "a"),
        regexlayer.DateRange(dt.date(2018, 1, 1), dt.date(2022, 1, 1), "b"),
    ]
    assert regexlayer.computed_years_from_ranges(ranges) == 6.0


def test_computed_years_of_nothing_is_none():
    assert regexlayer.computed_years_from_ranges([]) is None


def test_a_reversed_date_range_is_discarded():
    assert regexlayer.find_date_ranges("Engineer, 2020 - 2016") == []


def test_identity_key_normalises_email_and_phone():
    a = regexlayer.identity_key("Layla@Example.COM", "+962 79 555 0101", "x")
    b = regexlayer.identity_key("layla@example.com", "+962795550101", "y")
    assert a == b


def test_identity_key_falls_back_to_the_checksum():
    assert regexlayer.identity_key(None, None, "abc123") == "anon|abc123"


def test_phone_normalisation_handles_international_prefixes():
    assert regexlayer.normalise_phone("00962795550101") == "+962795550101"


# ---------------------------------------------------------------------------
# Extractor: happy path
# ---------------------------------------------------------------------------


def test_extraction_returns_verified_evidence():
    extractor, _ = make_extractor([valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.retries == 0
    assert result.evidence_verification_rate == 1.0
    assert result.mean_confidence > 0.8


def test_the_regex_layer_overrides_the_model_on_contact_details():
    extractor, _ = make_extractor([valid_payload(email="wrong@wrong.com", phone="000")])
    result = extractor.extract(SOURCE)
    assert result.payload["email"] == "layla.haddad@example.com"
    assert result.payload["phone"].startswith("+962")


def test_skills_are_normalised_against_the_taxonomy():
    payload = valid_payload()
    payload["skills"][0]["name"] = "Postgres"
    payload["skills"][0]["evidence_quote"] = "Python, PostgreSQL, Docker, Kubernetes"
    extractor, _ = make_extractor([payload])
    result = extractor.extract(SOURCE)
    assert result.payload["skills"][0]["canonical"] == "postgresql"
    assert result.payload["skills"][0]["category"] == "database"


def test_an_arabic_skill_normalises_to_the_same_canonical():
    assert DEFAULT_TAXONOMY.normalise("كوبرنيتس") == "kubernetes"
    assert DEFAULT_TAXONOMY.normalise("Kubernetes") == "kubernetes"


def test_an_unknown_skill_survives_normalisation():
    assert DEFAULT_TAXONOMY.normalise("Underwater Basket Weaving") == "underwater basket weaving"


def test_an_empty_skill_normalises_to_empty():
    assert DEFAULT_TAXONOMY.normalise("   ") == ""


def test_the_taxonomy_is_extensible():
    taxonomy = Taxonomy([SkillEntry("quantum computing", ["qc", "الحوسبة الكمية"], "ml")])
    assert taxonomy.normalise("QC") == "quantum computing"
    assert taxonomy.normalise("الحوسبة الكمية") == "quantum computing"
    assert taxonomy.category_of("quantum computing") == "ml"


def test_the_seed_taxonomy_has_around_a_hundred_bilingual_skills():
    assert len(DEFAULT_TAXONOMY) >= 100
    arabic = [
        e for e in DEFAULT_TAXONOMY.entries()
        if any(any("؀" <= ch <= "ۿ" for ch in a) for a in e.aliases)
    ]
    assert len(arabic) >= 80, "the taxonomy must be genuinely bilingual"


# ---------------------------------------------------------------------------
# Extractor: evidence enforcement
# ---------------------------------------------------------------------------


def test_an_unverifiable_quote_zeroes_that_fields_confidence():
    payload = valid_payload()
    payload["skills"][0]["evidence_quote"] = "Fluent in seventeen dead languages"
    extractor, _ = make_extractor([payload])
    result = extractor.extract(SOURCE)
    assert result.payload["skills"][0]["confidence"] == 0.0
    assert result.payload["skills"][0]["evidence_verified"] is False


def test_a_verified_quote_keeps_its_confidence():
    extractor, _ = make_extractor([valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.payload["skills"][0]["confidence"] == 0.9
    assert result.payload["skills"][0]["evidence_verified"] is True


def test_a_missing_quote_also_zeroes_the_confidence():
    payload = valid_payload()
    payload["skills"][0]["evidence_quote"] = None
    extractor, _ = make_extractor([payload])
    result = extractor.extract(SOURCE)
    assert result.payload["skills"][0]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Extractor: years cross-check
# ---------------------------------------------------------------------------


def test_agreeing_years_raise_no_conflict():
    extractor, _ = make_extractor([valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.years_conflict is False
    assert "stated_vs_computed_years_conflict" not in result.warnings


def test_a_large_gap_between_stated_and_computed_years_is_flagged():
    payload = valid_payload(stated_years_experience=25.0)
    extractor, _ = make_extractor([payload])
    result = extractor.extract(SOURCE)
    assert result.years_conflict is True
    assert "stated_vs_computed_years_conflict" in result.warnings


def test_a_gap_inside_the_tolerance_is_not_flagged():
    extractor, _ = make_extractor([valid_payload()], years_tolerance=100.0)
    result = extractor.extract(SOURCE)
    assert result.years_conflict is False


def test_the_conflict_never_picks_a_winner():
    """Both figures are kept; the router decides what to do about the gap."""
    payload = valid_payload(stated_years_experience=25.0)
    extractor, _ = make_extractor([payload])
    result = extractor.extract(SOURCE)
    assert result.stated_years == 25.0
    assert result.computed_years is not None and result.computed_years < 25.0


def test_missing_dates_produce_a_warning():
    payload = valid_payload()
    payload["work_history"][0]["from_date"] = None
    extractor, _ = make_extractor([payload])
    result = extractor.extract("Backend Engineer at Amber Systems\nPython and PostgreSQL")
    assert "no_dates_in_work_history" in result.warnings


def test_no_work_history_produces_a_warning():
    extractor, _ = make_extractor([valid_payload(work_history=[])])
    result = extractor.extract("Just a name and an email\nsomeone@example.com")
    assert "no_work_history" in result.warnings


# ---------------------------------------------------------------------------
# Extractor: bounded retry
# ---------------------------------------------------------------------------


def test_an_invalid_response_is_re_prompted_and_recovers():
    bad = valid_payload()
    bad["skills"][0]["confidence"] = 5.0
    extractor, transport = make_extractor([bad, valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.retries == 1
    assert len(transport.prompts) == 2
    assert "REJECTED BY THE SCHEMA VALIDATOR" in transport.prompts[1]


def test_the_repair_prompt_quotes_the_validator_error():
    bad = valid_payload()
    bad["work_history"][0]["from_date"] = "last March"
    extractor, transport = make_extractor([bad, valid_payload()])
    extractor.extract(SOURCE)
    assert "ISO date YYYY-MM-DD" in transport.prompts[1]


def test_malformed_json_is_also_re_prompted():
    extractor, transport = make_extractor(["this is not json", valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.retries == 1
    assert any("malformed_json" in w for w in result.warnings)


def test_retries_are_bounded_and_the_candidate_survives():
    bad = valid_payload(skills="not a list")
    extractor, transport = make_extractor([bad], max_retries=2)
    result = extractor.extract(SOURCE)
    assert len(transport.prompts) == 3, "one attempt plus exactly two retries"
    assert "extraction_failed_after_retries" in result.warnings
    assert result.payload["skills"] == []


def test_a_failed_extraction_still_keeps_the_regex_findings():
    extractor, _ = make_extractor(["garbage"], max_retries=1)
    result = extractor.extract(SOURCE)
    assert result.payload["email"] == "layla.haddad@example.com"
    assert result.stated_years == 9.0


def test_zero_retries_is_honoured():
    extractor, transport = make_extractor([valid_payload(skills="nope")], max_retries=0)
    extractor.extract(SOURCE)
    assert len(transport.prompts) == 1


def test_repair_prompt_lists_at_most_eight_problems():
    prompt = _repair_prompt("base", [f"problem {i}" for i in range(20)])
    assert prompt.count("- problem") == 8


def test_the_extraction_prompt_wraps_the_cv_as_untrusted_data():
    extractor, transport = make_extractor([valid_payload()])
    extractor.extract(SOURCE)
    prompt = transport.prompts[0]
    assert "UNTRUSTED DATA" in prompt
    assert "<<<BEGIN_UNTRUSTED_DATA>>>" in prompt
    assert "never instructions to follow" in prompt


def test_the_extraction_prompt_carries_the_regex_hints():
    extractor, transport = make_extractor([valid_payload()])
    extractor.extract(SOURCE)
    assert "layla.haddad@example.com" in transport.prompts[0]
    assert "deterministic pre-pass" in transport.prompts[0]


def test_the_result_records_its_prompt_and_schema_versions():
    extractor, _ = make_extractor([valid_payload()])
    result = extractor.extract(SOURCE)
    assert result.prompt_version and result.schema_version
    assert result.as_dict()["mean_confidence"] == result.mean_confidence
