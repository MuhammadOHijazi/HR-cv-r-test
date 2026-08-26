"""Human field corrections: what each operation does to the extraction payload.

A human correction is authoritative. The person has just read the source
document, so the corrected field gets confidence 1.0 and counts as verified —
and the whole record is then re-scored and re-routed through the ordinary
pipeline, never through a special case.
"""

from __future__ import annotations

import pytest

from backend.app.core.pipeline import HUMAN_CONFIDENCE, apply_field_corrections


def payload():
    return {
        "full_name": "Layla Haddad",
        "email": "layla@example.com",
        "phone": "+962795550101",
        "stated_years_experience": 9.0,
        "skills": [
            {"name": "Python", "canonical": "python", "confidence": 0.9},
            {"name": "Docker", "canonical": "docker", "confidence": 0.4},
        ],
        "education": [{"degree": "bachelor", "confidence": 0.8}],
        "work_history": [{"title": "Engineer", "from_date": "2018-01-01", "highlights": []}],
    }


def confidences():
    return {"skill:Python": 0.9, "skill:Docker": 0.4}


def apply(corrections):
    return apply_field_corrections(payload(), confidences(), corrections)


# ---------------------------------------------------------------------------
# Scalar fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("full_name", "Layla H."),
        ("email", "new@example.com"),
        ("phone", "+962790000000"),
        ("stated_years_experience", 11.0),
    ],
)
def test_a_scalar_field_can_be_corrected(field, value):
    result, _ = apply({field: value})
    assert result[field] == value


def test_a_scalar_field_can_be_cleared():
    result, _ = apply({"stated_years_experience": None})
    assert result["stated_years_experience"] is None


def test_untouched_fields_are_preserved():
    result, _ = apply({"full_name": "X"})
    assert result["email"] == "layla@example.com"
    assert len(result["skills"]) == 2


def test_the_original_payload_is_not_mutated():
    original = payload()
    apply_field_corrections(original, confidences(), {"full_name": "Someone Else"})
    assert original["full_name"] == "Layla Haddad"


def test_no_corrections_changes_nothing():
    result, conf = apply({})
    assert result == payload()
    assert conf == confidences()


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_adding_a_skill_marks_it_human_verified():
    result, conf = apply({"add_skills": [{"name": "Kubernetes", "canonical": "kubernetes"}]})
    added = result["skills"][-1]
    assert added["canonical"] == "kubernetes"
    assert added["confidence"] == HUMAN_CONFIDENCE
    assert added["evidence_verified"] is True
    assert added["source"] == "human_correction"
    assert conf["skill:Kubernetes"] == HUMAN_CONFIDENCE


def test_a_bare_string_skill_is_accepted():
    result, _ = apply({"add_skills": ["Terraform"]})
    assert result["skills"][-1]["name"] == "Terraform"
    assert result["skills"][-1]["canonical"] == "terraform"


def test_an_added_skill_carries_a_traceable_quote():
    result, _ = apply({"add_skills": [{"name": "Kafka"}]})
    assert "human correction" in result["skills"][-1]["evidence_quote"]


def test_a_supplied_quote_is_kept():
    result, _ = apply(
        {"add_skills": [{"name": "Kafka", "evidence_quote": "Built Kafka consumers"}]}
    )
    assert result["skills"][-1]["evidence_quote"] == "Built Kafka consumers"


def test_a_skill_can_be_removed_by_name():
    result, conf = apply({"remove_skills": ["Docker"]})
    assert [s["name"] for s in result["skills"]] == ["Python"]
    assert "skill:Docker" not in conf


def test_a_skill_can_be_removed_by_canonical_form():
    result, _ = apply({"remove_skills": ["python"]})
    assert [s["name"] for s in result["skills"]] == ["Docker"]


def test_removing_an_absent_skill_is_harmless():
    result, _ = apply({"remove_skills": ["Fortran"]})
    assert len(result["skills"]) == 2


def test_a_skills_confidence_can_be_corrected():
    result, conf = apply(
        {"update_skill_confidence": [{"name": "Docker", "confidence": 0.95}]}
    )
    docker = next(s for s in result["skills"] if s["name"] == "Docker")
    assert docker["confidence"] == 0.95
    assert docker["evidence_verified"] is True
    assert conf["skill:Docker"] == 0.95


def test_updating_the_confidence_of_an_absent_skill_is_harmless():
    result, _ = apply({"update_skill_confidence": [{"name": "Rust", "confidence": 0.9}]})
    assert len(result["skills"]) == 2


# ---------------------------------------------------------------------------
# Education and work history
# ---------------------------------------------------------------------------


def test_education_can_be_added():
    result, conf = apply({"add_education": [{"degree": "master", "field": "CS"}]})
    added = result["education"][-1]
    assert added["degree"] == "master"
    assert added["confidence"] == HUMAN_CONFIDENCE
    assert added["evidence_verified"] is True
    assert any(k.startswith("education:master") for k in conf)


def test_work_history_can_be_added():
    result, conf = apply(
        {"add_work_history": [{"title": "Lead Engineer", "from_date": "2015-01-01"}]}
    )
    added = result["work_history"][-1]
    assert added["title"] == "Lead Engineer"
    assert added["highlights"] == []
    assert added["confidence"] == HUMAN_CONFIDENCE
    assert any(k.startswith("work:Lead Engineer") for k in conf)


def test_supplied_highlights_are_kept():
    result, _ = apply(
        {"add_work_history": [{"title": "Lead", "highlights": ["Ran the platform team"]}]}
    )
    assert result["work_history"][-1]["highlights"] == ["Ran the platform team"]


def test_several_corrections_apply_together():
    result, conf = apply(
        {
            "add_skills": [{"name": "Kubernetes"}],
            "remove_skills": ["Docker"],
            "add_education": [{"degree": "master"}],
            "stated_years_experience": 12.0,
        }
    )
    names = [s["name"] for s in result["skills"]]
    assert "Kubernetes" in names and "Docker" not in names
    assert result["education"][-1]["degree"] == "master"
    assert result["stated_years_experience"] == 12.0
    assert conf["skill:Kubernetes"] == HUMAN_CONFIDENCE


def test_a_correction_on_an_empty_payload_is_safe():
    result, conf = apply_field_corrections({}, {}, {"add_skills": [{"name": "Go"}]})
    assert result["skills"][0]["name"] == "Go"
    assert conf["skill:Go"] == HUMAN_CONFIDENCE
