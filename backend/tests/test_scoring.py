"""Phase 4 gate — the three scorers, the saturation curve and the merge."""

from __future__ import annotations

import math

import pytest
from sqlalchemy import select

from backend.app.core.confidence import agreement_from_disagreement, assemble
from backend.app.core.gemini_client import GeminiClient, ModelConfig
from backend.app.core.jd import JDNotApproved, JDService, normalise_structured
from backend.app.core.masking import mask_record, mask_text
from backend.app.core.scoring import (
    EmbeddingStore,
    JudgeResult,
    LLMJudge,
    RulesGate,
    SemanticScorer,
    _clamp_rubric,
    content_hash,
    contrast,
    cosine,
    merge_scores,
    pack_vector,
    scorer_disagreement,
    unpack_vector,
    years_score,
)
from backend.app.models import EmbeddingCache

JD = {
    "title": "Senior Backend Engineer",
    "must_have": [
        {"skill": "Python", "canonical": "python"},
        {"skill": "Kubernetes", "canonical": "kubernetes"},
    ],
    "nice_to_have": [{"skill": "Kafka", "canonical": "kafka"}],
    "responsibilities": [
        "Design and build scalable Python microservices",
        "Operate services on Kubernetes",
    ],
    "weights": {
        "must_have_skills": 0.30,
        "preferred_skills": 0.15,
        "experience_years": 0.20,
        "similar_experience": 0.25,
        "education": 0.10,
    },
    "thresholds": {
        "min_years_experience": 5.0,
        "required_degree": "bachelor",
        "required_certifications": [],
    },
}


def extraction(
    *,
    skills=(("python", 0.9), ("kubernetes", 0.9)),
    years=7.0,
    degree=("bachelor", 0.9),
    quality=0.95,
    conflict=False,
    verified=1.0,
    mean_conf=0.9,
    highlights=("Design and build scalable Python microservices",),
):
    return {
        "payload": {
            "skills": [
                {"name": name, "canonical": name, "confidence": conf} for name, conf in skills
            ],
            "education": (
                [{"degree": degree[0], "confidence": degree[1]}] if degree else []
            ),
            "work_history": [
                {
                    "title": "Backend Engineer",
                    "from_date": "2018-01-01",
                    "to_date": None,
                    "highlights": list(highlights),
                }
            ],
        },
        "stated_years": years,
        "computed_years": years,
        "years_conflict": conflict,
        "source_quality": quality,
        "evidence_verification_rate": verified,
        "mean_confidence": mean_conf,
        "field_confidence": {f"skill:{n}": c for n, c in skills},
    }


# ---------------------------------------------------------------------------
# Years saturation curve
# ---------------------------------------------------------------------------


def test_no_years_scores_zero():
    assert years_score(None, 5.0) == 0.0


def test_zero_years_scores_zero():
    assert years_score(0.0, 5.0) == 0.0


def test_meeting_the_requirement_scores_three_quarters():
    assert years_score(5.0, 5.0) == pytest.approx(0.75, abs=1e-6)


def test_double_the_requirement_is_worth_far_less_than_double():
    assert years_score(10.0, 5.0) == pytest.approx(0.9375, abs=1e-4)


def test_the_curve_is_monotonic():
    values = [years_score(y, 5.0) for y in range(0, 30)]
    assert values == sorted(values)


def test_the_curve_never_reaches_one():
    assert years_score(30.0, 5.0) < 1.0
    assert years_score(200.0, 5.0) <= 1.0


def test_returns_diminish_with_every_extra_year():
    gains = [years_score(y + 1, 5.0) - years_score(y, 5.0) for y in range(0, 15)]
    assert all(a > b for a, b in zip(gains, gains[1:])), "each extra year must be worth less"


def test_the_curve_is_not_linear():
    linear = years_score(5.0, 5.0) * 2
    assert years_score(10.0, 5.0) < linear


def test_with_no_stated_requirement_the_curve_anchors_at_three_years():
    assert years_score(3.0, 0.0) == pytest.approx(0.75, abs=1e-6)


def test_negative_years_are_clamped():
    assert years_score(-4.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# Rules gate
# ---------------------------------------------------------------------------


def test_a_fully_qualified_candidate_passes_the_gate():
    result = RulesGate().evaluate(extraction(), JD)
    assert result.passed
    assert result.must_have_coverage == 1.0
    assert not result.hard_failures and not result.soft_failures


def test_a_missing_must_have_on_good_data_is_a_hard_failure():
    result = RulesGate().evaluate(extraction(skills=(("python", 0.9),)), JD)
    assert result.has_hard_failure
    assert result.missing_must_have == ["kubernetes"]
    assert result.hard_failures[0]["rule"] == "missing_must_have_skill"


def test_the_same_gap_on_unreliable_data_is_only_a_soft_failure():
    """Never auto-reject on evidence we do not trust."""
    poor = extraction(skills=(("python", 0.2),), quality=0.3, verified=0.2, mean_conf=0.3)
    result = RulesGate().evaluate(poor, JD)
    assert not result.has_hard_failure
    assert result.soft_failures


def test_a_low_confidence_matched_skill_is_a_soft_failure():
    result = RulesGate().evaluate(extraction(skills=(("python", 0.9), ("kubernetes", 0.2))), JD)
    assert not result.has_hard_failure
    assert any(f["rule"] == "must_have_skill_low_confidence" for f in result.soft_failures)


def test_too_few_years_on_good_data_is_a_hard_failure():
    result = RulesGate().evaluate(extraction(years=2.0), JD)
    assert any(f["rule"] == "min_years_not_met" for f in result.hard_failures)


def test_too_few_years_with_a_years_conflict_is_only_soft():
    result = RulesGate().evaluate(extraction(years=2.0, conflict=True), JD)
    assert not result.has_hard_failure
    assert any(f["rule"] == "min_years_not_met" for f in result.soft_failures)


def test_unknown_years_is_a_soft_failure_not_a_rejection():
    data = extraction()
    data["stated_years"] = None
    data["computed_years"] = None
    result = RulesGate().evaluate(data, JD)
    assert not result.has_hard_failure
    assert any(f["rule"] == "min_years_unknown" for f in result.soft_failures)


def test_meeting_the_years_floor_exactly_passes():
    result = RulesGate().evaluate(extraction(years=5.0), JD)
    assert not any(f["rule"] == "min_years_not_met" for f in result.hard_failures)


def test_a_degree_below_the_requirement_is_a_hard_failure():
    result = RulesGate().evaluate(extraction(degree=("diploma", 0.9)), JD)
    assert any(f["rule"] == "required_degree_not_met" for f in result.hard_failures)


def test_a_higher_degree_satisfies_the_requirement():
    result = RulesGate().evaluate(extraction(degree=("phd", 0.9)), JD)
    assert not result.hard_failures


def test_a_low_confidence_degree_gap_is_only_soft():
    result = RulesGate().evaluate(extraction(degree=("diploma", 0.2)), JD)
    assert not result.has_hard_failure
    assert any(f["rule"] == "required_degree_not_met" for f in result.soft_failures)


def test_no_education_at_all_is_a_soft_failure():
    result = RulesGate().evaluate(extraction(degree=None), JD)
    assert not result.has_hard_failure
    assert any(f["rule"] == "required_degree_not_met" for f in result.soft_failures)


def test_a_missing_certification_is_a_hard_failure():
    jd = {**JD, "thresholds": {**JD["thresholds"], "required_certifications": ["CKA"]}}
    result = RulesGate().evaluate(extraction(), jd)
    assert any(f["rule"] == "missing_certification" for f in result.hard_failures)


def test_a_present_certification_passes():
    jd = {**JD, "thresholds": {**JD["thresholds"], "required_certifications": ["CKA"]}}
    data = extraction(highlights=("Earned the CKA certification in 2021",))
    result = RulesGate().evaluate(data, jd)
    assert not any(f["rule"] == "missing_certification" for f in result.hard_failures)


def test_a_jd_with_no_must_haves_gives_full_coverage():
    jd = {**JD, "must_have": []}
    assert RulesGate().evaluate(extraction(), jd).must_have_coverage == 1.0


def test_a_years_conflict_prefers_the_lower_figure():
    data = extraction()
    data.update(stated_years=20.0, computed_years=4.0, years_conflict=True)
    result = RulesGate().evaluate(data, JD)
    assert any(f["rule"] == "min_years_not_met" for f in result.soft_failures)


# ---------------------------------------------------------------------------
# Semantic scorer
# ---------------------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_of_mismatched_lengths_is_zero():
    assert cosine([1.0], [1.0, 2.0]) == 0.0


def test_cosine_of_empty_vectors_is_zero():
    assert cosine([], []) == 0.0


def test_vectors_round_trip_through_the_blob_packing():
    vector = [0.25, -0.5, 0.125]
    assert unpack_vector(pack_vector(vector)) == pytest.approx(vector)


def test_content_hash_is_stable_and_model_scoped():
    assert content_hash("abc", "m1") == content_hash("abc", "m1")
    assert content_hash("abc", "m1") != content_hash("abc", "m2")
    assert content_hash("abc", "m1") != content_hash("abd", "m1")


def test_contrast_of_the_background_level_is_zero():
    assert contrast(0.3, 0.3) == 0.0


def test_contrast_of_a_perfect_match_is_one():
    assert contrast(1.0, 0.2) == pytest.approx(1.0)


def test_contrast_is_clamped_below_the_baseline():
    assert contrast(0.1, 0.4) == 0.0


def test_contrast_handles_a_degenerate_baseline():
    assert contrast(1.0, 1.0) == 0.0


class CountingTransport:
    def __init__(self):
        self.embed_calls = 0
        self.embedded: list[str] = []

    def generate(self, **kwargs):
        return "{}"

    def embed(self, *, api_key, model, texts):
        self.embed_calls += 1
        self.embedded.extend(texts)
        return [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]


def make_store(session):
    transport = CountingTransport()
    client = GeminiClient(
        transport=transport,
        keys=["k-1111"],
        models=ModelConfig("f", "p", "embed-model", "f", 3),
        clock=lambda: 0.0,
    )
    return EmbeddingStore(session, client), transport


def test_embeddings_are_cached_in_the_database(session):
    store, transport = make_store(session)
    store.embed_many(["alpha", "beta"])
    assert transport.embed_calls == 1
    assert session.scalar(select(EmbeddingCache).where(EmbeddingCache.content_hash != "")) is not None


def test_the_same_text_is_never_embedded_twice(session):
    store, transport = make_store(session)
    store.embed_many(["alpha", "beta"])
    store.embed_many(["alpha", "beta"])
    assert transport.embed_calls == 1, "the second call must be served from the cache"


def test_only_the_new_texts_are_sent_on_a_partial_hit(session):
    store, transport = make_store(session)
    store.embed_many(["alpha"])
    store.embed_many(["alpha", "gamma"])
    assert transport.embedded == ["alpha", "gamma"]


def test_duplicates_within_one_batch_are_embedded_once(session):
    store, transport = make_store(session)
    vectors = store.embed_many(["same", "same", "other"])
    assert transport.embedded == ["same", "other"]
    assert vectors[0] == vectors[1]


def test_an_empty_batch_makes_no_call(session):
    store, transport = make_store(session)
    assert store.embed_many([]) == []
    assert transport.embed_calls == 0


def test_semantic_scoring_needs_both_sides(session):
    store, _ = make_store(session)
    scorer = SemanticScorer(store)
    no_history = extraction()
    no_history["payload"]["work_history"] = []
    assert scorer.score(no_history, JD).score == 0.0
    assert scorer.score(extraction(), {**JD, "responsibilities": []}).score == 0.0


def test_semantic_scoring_reports_its_pairs(session):
    store, _ = make_store(session)
    result = SemanticScorer(store).score(extraction(), JD)
    assert result.responsibility_count == 2
    assert len(result.pairs) == 2
    assert all("similarity" in p and "calibrated" in p for p in result.pairs)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class JudgeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
        import json

        self.prompts.append(prompt)
        return json.dumps(self.payload)

    def embed(self, *, api_key, model, texts):
        return [[1.0] for _ in texts]


def make_judge(payload):
    transport = JudgeTransport(payload)
    client = GeminiClient(
        transport=transport,
        keys=["k-1111"],
        models=ModelConfig("f", "p", "e", "f", 3),
        clock=lambda: 0.0,
    )
    return LLMJudge(client), transport


def dims(**overrides):
    base = {
        d: {"score": 70, "rubric_level": "solid", "evidence_quote": "python", "rationale": "r"}
        for d in ("preferred_skills", "similar_experience", "education")
    }
    base.update(overrides)
    return {"dimensions": base}


def test_the_judge_scores_every_dimension():
    judge, _ = make_judge(dims())
    result = judge.score(extraction(), JD)
    assert set(result.dimensions) == {"preferred_skills", "similar_experience", "education"}
    assert result.mean_score == 70.0


def test_the_judge_never_sees_the_candidate_identity():
    judge, transport = make_judge(dims())
    data = extraction()
    data["payload"]["full_name"] = "Layla Haddad"
    data["payload"]["email"] = "layla@example.com"
    data["payload"]["phone"] = "+962795550101"
    judge.score(data, JD)
    prompt = transport.prompts[0]
    assert "Layla" not in prompt
    assert "layla@example.com" not in prompt
    assert "+962795550101" not in prompt


def test_the_judge_prompt_wraps_the_record_as_untrusted_data():
    judge, transport = make_judge(dims())
    judge.score(extraction(), JD)
    assert "<<<BEGIN_UNTRUSTED_DATA>>>" in transport.prompts[0]
    assert "<<<BEGIN_JOB_SPEC>>>" in transport.prompts[0]
    assert "rubric" in transport.prompts[0].lower()


def test_a_score_with_an_unverifiable_quote_is_forced_to_zero():
    judge, _ = make_judge(
        dims(
            preferred_skills={
                "score": 100,
                "evidence_quote": "Invented a time machine",
                "rationale": "",
            }
        )
    )
    result = judge.score(extraction(), JD)
    assert result.dimensions["preferred_skills"]["score"] == 0.0
    assert result.dimensions["preferred_skills"]["evidence_verified"] is False


def test_a_score_with_a_verified_quote_stands():
    judge, _ = make_judge(dims())
    result = judge.score(extraction(), JD)
    assert result.dimensions["preferred_skills"]["score"] == 70.0
    assert result.dimensions["preferred_skills"]["evidence_verified"] is True


def test_a_missing_dimension_scores_zero():
    judge, _ = make_judge({"dimensions": {}})
    result = judge.score(extraction(), JD)
    assert result.mean_score == 0.0


def test_the_judge_reports_its_evidence_verification_rate():
    judge, _ = make_judge(
        dims(education={"score": 100, "evidence_quote": "not in the record", "rationale": ""})
    )
    result = judge.score(extraction(), JD)
    assert result.evidence_verified_rate == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 0.0), (12, 0.0), (35, 40.0), (55, 40.0), (60, 70.0), (85, 70.0), (95, 100.0), (140, 100.0)],
)
def test_scores_snap_onto_the_four_rubric_levels(raw, expected):
    assert _clamp_rubric(raw) == expected


@pytest.mark.parametrize("raw", [None, "high", object()])
def test_an_unusable_judge_score_becomes_zero(raw):
    assert _clamp_rubric(raw) == 0.0


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_masking_removes_every_identity_field():
    masked = mask_record(
        {
            "full_name": "Layla Haddad",
            "email": "layla@example.com",
            "phone": "+962795550101",
            "nationality": "Jordanian",
            "gender": "female",
            "age": 34,
            "skills": [{"name": "python", "confidence": 0.9}],
        }
    )
    assert set(masked) == {"skills"}


def test_masking_scrubs_identity_out_of_free_text():
    masked = mask_record(
        {
            "full_name": "Layla Haddad",
            "work_history": [
                {"highlights": ["Layla led the team; reach her at layla@example.com"]}
            ],
        }
    )
    text = masked["work_history"][0]["highlights"][0]
    assert "Layla" not in text and "layla@example.com" not in text


def test_masking_keeps_the_capability_signal():
    masked = mask_record(
        {"full_name": "Omar", "skills": [{"name": "kubernetes", "confidence": 0.9}]}
    )
    assert masked["skills"][0]["name"] == "kubernetes"


@pytest.mark.parametrize(
    "text",
    [
        "Nationality: Jordanian",
        "Date of birth: 1991-04-02",
        "Gender: female",
        "photo: headshot.jpg",
        "Marital status: married",
    ],
)
def test_identity_signals_in_text_are_removed(text):
    assert mask_text(text).strip() in {"[REDACTED]", "[REDACTED] [REDACTED]", ""}


def test_masking_empty_text_is_safe():
    assert mask_text("") == ""


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def rules_for(**kwargs):
    return RulesGate().evaluate(extraction(**kwargs), JD)


class FakeSemantic:
    def __init__(self, score):
        self.score = score
        self.pairs = []
        self.responsibility_count = 0
        self.highlight_count = 0
        self.baseline = 0.0

    def as_dict(self):
        return {"score": self.score}


def judge_result(preferred=70, similar=70, education=70):
    return JudgeResult(
        {
            "preferred_skills": {"score": preferred},
            "similar_experience": {"score": similar},
            "education": {"score": education},
        },
        1.0,
    )


def test_a_perfect_candidate_approaches_a_hundred():
    merged = merge_scores(rules_for(years=25.0), FakeSemantic(1.0), judge_result(100, 100, 100), JD)
    assert merged.total > 95


def test_an_empty_candidate_scores_near_zero():
    empty_jd = {**JD, "must_have": []}
    merged = merge_scores(
        RulesGate().evaluate(
            {"payload": {"skills": [], "education": [], "work_history": []}}, empty_jd
        ),
        FakeSemantic(0.0),
        judge_result(0, 0, 0),
        empty_jd,
    )
    # Only the must-have dimension can score, and only because this JD states
    # no must-haves at all; everything the candidate could earn is zero.
    assert merged.total <= 30.0
    assert merged.dimensions["similar_experience"]["contribution"] == 0.0
    assert merged.dimensions["experience_years"]["contribution"] == 0.0
    assert merged.dimensions["education"]["contribution"] == 0.0


def test_the_merge_is_bounded_to_zero_and_one_hundred():
    merged = merge_scores(rules_for(), FakeSemantic(1.0), judge_result(100, 100, 100), JD)
    assert 0.0 <= merged.total <= 100.0


def test_the_weights_are_normalised_to_sum_to_one():
    jd = {**JD, "weights": {k: v * 3 for k, v in JD["weights"].items()}}
    merged = merge_scores(rules_for(), FakeSemantic(0.5), judge_result(), jd)
    assert sum(merged.weights.values()) == pytest.approx(1.0)


def test_zero_weights_do_not_divide_by_zero():
    jd = {**JD, "weights": {k: 0.0 for k in JD["weights"]}}
    merged = merge_scores(rules_for(), FakeSemantic(0.5), judge_result(), jd)
    assert merged.total >= 0.0


def test_every_dimension_reports_its_contribution():
    merged = merge_scores(rules_for(), FakeSemantic(0.5), judge_result(), JD)
    assert set(merged.dimensions) == set(JD["weights"])
    for entry in merged.dimensions.values():
        assert {"raw", "weight", "contribution"} == set(entry)
    total = sum(e["contribution"] for e in merged.dimensions.values())
    assert total == pytest.approx(merged.total, abs=1e-3)


def test_a_hard_rule_failure_caps_the_total():
    merged = merge_scores(
        rules_for(skills=(("python", 0.9),)), FakeSemantic(1.0), judge_result(100, 100, 100), JD
    )
    assert merged.total <= 40.0


def test_similar_experience_blends_the_semantic_and_judge_views():
    high = merge_scores(rules_for(), FakeSemantic(1.0), judge_result(similar=100), JD)
    low = merge_scores(rules_for(), FakeSemantic(0.0), judge_result(similar=0), JD)
    assert high.dimensions["similar_experience"]["raw"] == pytest.approx(1.0)
    assert low.dimensions["similar_experience"]["raw"] == pytest.approx(0.0)


def test_education_blends_the_rules_and_judge_views():
    merged = merge_scores(rules_for(), FakeSemantic(0.5), judge_result(education=0), JD)
    assert 0.0 < merged.dimensions["education"]["raw"] < 1.0


# ---------------------------------------------------------------------------
# Disagreement and confidence
# ---------------------------------------------------------------------------


def test_agreeing_scorers_produce_no_disagreement():
    value = scorer_disagreement(rules_for(), FakeSemantic(0.7), judge_result(similar=70, education=100))
    assert value == 0.0


def test_a_split_on_similar_experience_is_disagreement():
    value = scorer_disagreement(rules_for(), FakeSemantic(0.1), judge_result(similar=100, education=100))
    assert value == pytest.approx(90.0)


def test_a_split_on_education_is_disagreement():
    value = scorer_disagreement(rules_for(), FakeSemantic(0.7), judge_result(similar=70, education=0))
    assert value == pytest.approx(100.0)


def test_preferred_skills_alone_never_count_as_disagreement():
    """The rules gate does not score nice-to-haves, so it cannot disagree about them."""
    value = scorer_disagreement(rules_for(), FakeSemantic(0.7), judge_result(preferred=0, similar=70, education=100))
    assert value == 0.0


def test_agreement_is_the_complement_of_disagreement():
    assert agreement_from_disagreement(0.0) == 1.0
    assert agreement_from_disagreement(100.0) == 0.0
    assert agreement_from_disagreement(40.0) == pytest.approx(0.6)


def test_agreement_is_clamped():
    assert agreement_from_disagreement(-10.0) == 1.0
    assert agreement_from_disagreement(400.0) == 0.0


def test_confidence_mixes_all_four_components():
    result = assemble(
        mean_field_confidence=1.0,
        source_quality=1.0,
        evidence_verification_rate=1.0,
        disagreement=0.0,
    )
    assert result.value == pytest.approx(1.0)
    assert set(result.components) == {
        "field_confidence",
        "source_quality",
        "evidence_verification",
        "scorer_agreement",
    }


def test_confidence_of_nothing_is_zero():
    result = assemble(
        mean_field_confidence=0.0,
        source_quality=0.0,
        evidence_verification_rate=0.0,
        disagreement=100.0,
    )
    assert result.value == 0.0


def test_poor_source_quality_drags_confidence_down():
    good = assemble(
        mean_field_confidence=0.9, source_quality=0.95, evidence_verification_rate=1.0, disagreement=0.0
    )
    poor = assemble(
        mean_field_confidence=0.9, source_quality=0.2, evidence_verification_rate=1.0, disagreement=0.0
    )
    assert poor.value < good.value


def test_unverified_evidence_drags_confidence_down():
    good = assemble(
        mean_field_confidence=0.9, source_quality=0.9, evidence_verification_rate=1.0, disagreement=0.0
    )
    poor = assemble(
        mean_field_confidence=0.9, source_quality=0.9, evidence_verification_rate=0.1, disagreement=0.0
    )
    assert poor.value < good.value


def test_disagreement_beyond_the_cap_hard_caps_confidence():
    result = assemble(
        mean_field_confidence=1.0,
        source_quality=1.0,
        evidence_verification_rate=1.0,
        disagreement=90.0,
        disagreement_cap=35.0,
        disagreement_ceiling=0.65,
    )
    assert result.value <= 0.65
    assert "scorer_disagreement" in result.caps_applied


def test_disagreement_inside_the_cap_applies_no_ceiling():
    result = assemble(
        mean_field_confidence=1.0,
        source_quality=1.0,
        evidence_verification_rate=1.0,
        disagreement=20.0,
        disagreement_cap=35.0,
    )
    assert result.caps_applied == []
    assert result.value > 0.65


def test_a_missing_critical_field_caps_confidence():
    result = assemble(
        mean_field_confidence=1.0,
        source_quality=1.0,
        evidence_verification_rate=1.0,
        disagreement=0.0,
        missing_critical_fields=True,
    )
    assert result.value <= 0.6
    assert "missing_critical_field" in result.caps_applied


def test_a_failed_extraction_caps_confidence_hardest():
    result = assemble(
        mean_field_confidence=1.0,
        source_quality=1.0,
        evidence_verification_rate=1.0,
        disagreement=0.0,
        extraction_failed=True,
    )
    assert result.value <= 0.3
    assert "extraction_failed" in result.caps_applied


def test_confidence_components_are_clamped():
    result = assemble(
        mean_field_confidence=5.0,
        source_quality=-2.0,
        evidence_verification_rate=None,
        disagreement=0.0,
    )
    assert result.components["field_confidence"] == 1.0
    assert result.components["source_quality"] == 0.0
    assert result.components["evidence_verification"] == 0.0


# ---------------------------------------------------------------------------
# JD structuring
# ---------------------------------------------------------------------------


def test_jd_weights_are_normalised():
    structured = normalise_structured(
        {"title": "X", "weights": {"must_have_skills": 2, "education": 2}}
    )
    assert sum(structured["weights"].values()) == pytest.approx(1.0)


def test_jd_weights_fall_back_to_sensible_defaults():
    structured = normalise_structured({"title": "X"})
    assert sum(structured["weights"].values()) == pytest.approx(1.0)
    assert structured["weights"]["must_have_skills"] > 0


def test_jd_skills_are_canonicalised_and_deduplicated():
    structured = normalise_structured(
        {"must_have": [{"skill": "Postgres"}, {"skill": "PostgreSQL"}, {"skill": "  "}]}
    )
    assert len(structured["must_have"]) == 1
    assert structured["must_have"][0]["canonical"] == "postgresql"


def test_jd_accepts_bare_skill_strings():
    structured = normalise_structured({"nice_to_have": ["Kafka"]})
    assert structured["nice_to_have"][0]["canonical"] == "kafka"


@pytest.mark.parametrize("value", ["Bachelor", "  master ", "PHD"])
def test_a_valid_degree_requirement_is_normalised(value):
    structured = normalise_structured({"thresholds": {"required_degree": value}})
    assert structured["thresholds"]["required_degree"] == value.strip().lower()


@pytest.mark.parametrize("value", ["wizard", "", None, 42])
def test_an_invalid_degree_requirement_becomes_null(value):
    structured = normalise_structured({"thresholds": {"required_degree": value}})
    assert structured["thresholds"]["required_degree"] is None


@pytest.mark.parametrize("value,expected", [("7", 7.0), (-3, 0.0), (None, 0.0), ("lots", 0.0)])
def test_min_years_is_coerced(value, expected):
    structured = normalise_structured({"thresholds": {"min_years_experience": value}})
    assert structured["thresholds"]["min_years_experience"] == expected


def test_an_empty_jd_still_produces_a_usable_shape():
    structured = normalise_structured({})
    assert structured["title"] == "Untitled role"
    assert structured["must_have"] == [] and structured["responsibilities"] == []


def test_screening_refuses_an_unapproved_jd(session, gemini):
    from backend.app.models import Job

    job = Job(title="X", raw_jd_text="Must have: Python\nResponsibilities:\n- Build things")
    session.add(job)
    session.commit()
    service = JDService(session, gemini)
    service.structure(job)
    with pytest.raises(JDNotApproved):
        service.active_version(job)


def test_approving_a_version_makes_it_active(session, gemini):
    from backend.app.models import Job

    job = Job(title="X", raw_jd_text="Must have:\n- Python\nResponsibilities:\n- Build services")
    session.add(job)
    session.commit()
    service = JDService(session, gemini)
    version = service.structure(job)
    service.approve(job, version, actor="recruiter")
    assert service.active_version(job).id == version.id
    assert job.status == "ready"


def test_editing_creates_a_new_unapproved_version(session, gemini):
    from backend.app.models import Job

    job = Job(title="X", raw_jd_text="Must have:\n- Python\nResponsibilities:\n- Build")
    session.add(job)
    session.commit()
    service = JDService(session, gemini)
    first = service.structure(job)
    service.approve(job, first, actor="r")
    second = service.edit(job, first, {"title": "Edited", "must_have": [{"skill": "Go"}]}, actor="r")
    assert second.version == first.version + 1
    assert second.approved is False
    assert service.active_version(job).id == first.id, "history stays immutable until re-approval"
