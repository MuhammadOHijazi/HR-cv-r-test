"""Phase 5 gate — every FastAPI endpoint, happy path and error path."""

from __future__ import annotations

import pytest

from backend.app.schemas.api import REJECT_REASONS

BACKEND_JD = """Senior Backend Engineer

Must have:
- Strong Python and PostgreSQL
- Docker and Kubernetes in production
- At least 5 years of professional backend experience
- Bachelor degree in Computer Science

Nice to have:
- Kafka, Redis, Terraform

Responsibilities:
- Design and build scalable Python microservices for high-traffic APIs
- Own PostgreSQL schema design, query performance and migrations
- Deploy and operate services on Kubernetes with Docker
"""

CV_LINES = [
    "Layla Haddad",
    "layla.haddad@example.com | +962 79 555 0101",
    "",
    "SUMMARY",
    "Senior backend engineer with 9 years of experience building Python services.",
    "",
    "SKILLS",
    "Python, PostgreSQL, Docker, Kubernetes, Kafka, Redis, Terraform, Git, Linux",
    "",
    "EXPERIENCE",
    "Principal Backend Engineer, Nimbus Systems, {start} - present",
    "- Design and build scalable Python microservices for high-traffic APIs",
    "- Own PostgreSQL schema design, query performance and migrations",
    "- Deploy and operate services on Kubernetes with Docker",
    "",
    "EDUCATION",
    "Bachelor of Science in Computer Science, University of Jordan, {grad}",
]

WEAK_CV_LINES = [
    "Salma Aziz",
    "salma.aziz@example.com | +962 79 555 0707",
    "",
    "SUMMARY",
    "Graphic designer with 7 years of experience in brand and print design.",
    "",
    "SKILLS",
    "Photoshop, Illustrator, InDesign, typography, brand identity",
    "",
    "EXPERIENCE",
    "Senior Graphic Designer, Cedar Creative, {start} - present",
    "- Produced brand identity systems for regional retail clients",
    "",
    "EDUCATION",
    "Bachelor of Fine Arts, Jordan University, {grad}",
]


PARTIAL_CV_LINES = [
    "Tarek Mansour",
    "tarek.mansour@example.com | +962 79 555 0404",
    "",
    "SUMMARY",
    "Backend developer with 15 years of experience in Python web services.",
    "",
    "SKILLS",
    "Python, PostgreSQL, Docker, Git, Linux",
    "",
    "EXPERIENCE",
    "Backend Developer, Cedar Software, {start} - present",
    "- Design and build scalable Python microservices for high-traffic APIs",
    "- Own PostgreSQL schema design, query performance and migrations",
    "",
    "EDUCATION",
    "Bachelor of Science in Computer Science, Lebanese University, {grad}",
]


def _lines(template, years_ago):
    import datetime as dt

    year = dt.date.today().year
    return [
        line.format(start=year - years_ago, grad=year - years_ago - 1) for line in template
    ]


@pytest.fixture
def loaded_drive(drive, tmp_path):
    """A mock Drive folder holding two real CV files."""
    from scripts.generate_test_data import render_docx, render_pdf

    drive.add_folder("folder-a", "Applications")
    strong = tmp_path / "strong.pdf"
    render_pdf(_lines(CV_LINES, 9), strong)
    drive.add_file("folder-a", "file-strong", "strong.pdf", strong.read_bytes())

    weak = tmp_path / "weak.docx"
    render_docx(_lines(WEAK_CV_LINES, 7), weak)
    drive.add_file("folder-a", "file-weak", "weak.docx", weak.read_bytes())
    return drive


def make_job(client, title="Senior Backend Engineer", text=BACKEND_JD):
    response = client.post("/api/jobs", json={"title": title, "raw_jd_text": text})
    assert response.status_code == 201
    return response.json()


def approved_job(client):
    job = make_job(client)
    version = client.post(f"/api/jobs/{job['id']}/structure").json()
    client.post(f"/api/jobs/{job['id']}/versions/{version['version']}/approve", json={})
    return job, version


def screened_job(client, loaded_drive):
    job, version = approved_job(client)
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/sync")
    client.post(f"/api/jobs/{job['id']}/screen")
    return job, version


# ---------------------------------------------------------------------------
# Health and settings
# ---------------------------------------------------------------------------


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "mock_mode" in body


def test_settings_reports_the_key_pool(client):
    body = client.get("/api/settings").json()
    pool = body["key_pool"]
    assert pool["size"] >= 1
    assert pool["available"] <= pool["size"]
    for key in pool["keys"]:
        assert {"index", "label", "requests", "failures", "rate_limit_hits"} <= set(key)


def test_settings_never_leaks_a_full_key(client):
    body = client.get("/api/settings").text
    assert "key-aaaa1111" not in body and "key-bbbb2222" not in body
    assert "...1111" in body


def test_settings_reports_the_models_and_service_account(client):
    body = client.get("/api/settings").json()
    assert set(body["models"]) == {"extraction", "judge", "embedding", "vision"}
    assert "service_account_email" in body["drive"]
    assert set(body["defaults"]) >= {"shortlist_score_min", "confidence_min"}


# ---------------------------------------------------------------------------
# Jobs and JD structuring
# ---------------------------------------------------------------------------


def test_creating_and_listing_a_job(client):
    job = make_job(client)
    assert job["title"] == "Senior Backend Engineer"
    assert job["status"] == "draft"
    assert job["counts"] == {
        "auto_shortlist": 0,
        "human_review": 0,
        "preliminary_reject": 0,
        "total": 0,
    }
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job["id"]]


def test_creating_a_job_requires_a_title_and_text(client):
    assert client.post("/api/jobs", json={"title": "", "raw_jd_text": "x"}).status_code == 422
    assert client.post("/api/jobs", json={"title": "x", "raw_jd_text": ""}).status_code == 422


def test_fetching_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999").status_code == 404


def test_structuring_produces_an_unapproved_version(client):
    job = make_job(client)
    version = client.post(f"/api/jobs/{job['id']}/structure").json()
    assert version["version"] == 1
    assert version["approved"] is False
    structured = version["structured"]
    assert {"canonical": "python", "skill": "python", "importance": "must"} in structured["must_have"]
    assert structured["thresholds"]["min_years_experience"] == 5.0
    assert structured["thresholds"]["required_degree"] == "bachelor"
    assert structured["responsibilities"]


def test_structuring_an_unknown_job_is_a_404(client):
    assert client.post("/api/jobs/999/structure").status_code == 404


def test_structuring_an_empty_jd_is_a_400(client, session):
    from backend.app.models import Job

    job = Job(title="Empty", raw_jd_text="   ")
    session.add(job)
    session.commit()
    response = client.post(f"/api/jobs/{job.id}/structure")
    assert response.status_code == 400
    assert "no raw JD text" in response.json()["detail"]


def test_approving_a_version_marks_the_job_ready(client):
    job, version = approved_job(client)
    body = client.get(f"/api/jobs/{job['id']}").json()
    assert body["approved"] is True
    assert body["active_jd_version"] == version["version"]
    assert body["status"] == "ready"


def test_approving_an_unknown_version_is_a_404(client):
    job = make_job(client)
    assert client.post(f"/api/jobs/{job['id']}/versions/9/approve", json={}).status_code == 404


def test_an_empty_jd_version_cannot_be_approved(client):
    job = make_job(client)
    version = client.post(f"/api/jobs/{job['id']}/structure").json()
    client.put(
        f"/api/jobs/{job['id']}/versions/{version['version']}",
        json={"structured": {"title": "Empty", "must_have": [], "nice_to_have": []}},
    )
    response = client.post(f"/api/jobs/{job['id']}/versions/2/approve", json={})
    assert response.status_code == 400
    assert "no requirements" in response.json()["detail"]


def test_editing_creates_a_new_version_and_leaves_history_intact(client):
    job, version = approved_job(client)
    edited = client.put(
        f"/api/jobs/{job['id']}/versions/{version['version']}",
        json={"structured": {"title": "Revised", "must_have": [{"skill": "Go"}]}},
    ).json()
    assert edited["version"] == 2
    assert edited["approved"] is False
    versions = client.get(f"/api/jobs/{job['id']}/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["approved"] is True, "the approved version must be untouched"


def test_editing_an_unknown_version_is_a_404(client):
    job = make_job(client)
    response = client.put(f"/api/jobs/{job['id']}/versions/7", json={"structured": {}})
    assert response.status_code == 404


def test_listing_versions_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/versions").status_code == 404


def test_the_active_jd_is_served(client):
    job, _ = approved_job(client)
    body = client.get(f"/api/jobs/{job['id']}/jd").json()
    assert body["version"] == 1
    assert body["structured"]["must_have"]


def test_the_active_jd_of_an_unapproved_job_is_a_409(client):
    job = make_job(client)
    assert client.get(f"/api/jobs/{job['id']}/jd").status_code == 409


# ---------------------------------------------------------------------------
# Per-job configuration
# ---------------------------------------------------------------------------


def test_job_config_defaults_are_served(client):
    job = make_job(client)
    config = client.get(f"/api/jobs/{job['id']}/config").json()
    assert config["shortlist_score_min"] == 75.0
    assert config["confidence_min"] == 0.7


def test_job_config_can_be_updated(client):
    job = make_job(client)
    payload = {
        "shortlist_score_min": 80.0,
        "reject_score_max": 40.0,
        "confidence_min": 0.8,
        "disagreement_cap": 30.0,
        "years_conflict_tolerance": 2.0,
    }
    assert client.put(f"/api/jobs/{job['id']}/config", json=payload).json()["confidence_min"] == 0.8
    assert client.get(f"/api/jobs/{job['id']}/config").json()["shortlist_score_min"] == 80.0


def test_an_inverted_config_is_rejected(client):
    job = make_job(client)
    payload = {
        "shortlist_score_min": 40.0,
        "reject_score_max": 60.0,
        "confidence_min": 0.7,
        "disagreement_cap": 30.0,
        "years_conflict_tolerance": 2.0,
    }
    response = client.put(f"/api/jobs/{job['id']}/config", json=payload)
    assert response.status_code == 400
    assert "must be below" in response.json()["detail"]


def test_out_of_range_config_values_are_rejected(client):
    job = make_job(client)
    payload = {
        "shortlist_score_min": 900.0,
        "reject_score_max": 40.0,
        "confidence_min": 5.0,
        "disagreement_cap": 30.0,
        "years_conflict_tolerance": 2.0,
    }
    assert client.put(f"/api/jobs/{job['id']}/config", json=payload).status_code == 422


def test_config_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/config").status_code == 404


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------


def test_drive_status_reports_the_service_account(client):
    body = client.get("/api/drive/status").json()
    assert body["connected"] is True
    assert "service_account_email" in body


def test_folders_are_empty_until_refreshed(client, loaded_drive):
    assert client.get("/api/drive/folders").json() == []
    refreshed = client.post("/api/drive/folders/refresh").json()
    assert [f["folder_id"] for f in refreshed] == ["folder-a"]
    assert [f["name"] for f in client.get("/api/drive/folders").json()] == ["Applications"]


def test_folders_can_be_assigned_to_a_job(client, loaded_drive):
    job = make_job(client)
    client.post("/api/drive/folders/refresh")
    assert client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]}).json() == [
        "folder-a"
    ]
    assert client.get(f"/api/jobs/{job['id']}/folders").json() == ["folder-a"]


def test_assigning_an_unknown_folder_is_a_400(client, loaded_drive):
    job = make_job(client)
    client.post("/api/drive/folders/refresh")
    response = client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["nope"]})
    assert response.status_code == 400
    assert "refresh the folder list" in response.json()["detail"]


def test_reassigning_folders_replaces_the_previous_set(client, loaded_drive):
    job = make_job(client)
    loaded_drive.add_folder("folder-b", "Referrals")
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-b"]})
    assert client.get(f"/api/jobs/{job['id']}/folders").json() == ["folder-b"]


def test_assigning_folders_to_an_unknown_job_is_a_404(client):
    assert client.post("/api/jobs/999/folders", json={"folder_ids": []}).status_code == 404


def test_syncing_without_a_folder_is_a_400(client):
    job = make_job(client)
    response = client.post(f"/api/jobs/{job['id']}/sync")
    assert response.status_code == 400
    assert "assign at least one Drive folder" in response.json()["detail"]


def test_syncing_ingests_the_folder(client, loaded_drive):
    job = make_job(client)
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    body = client.post(f"/api/jobs/{job['id']}/sync").json()
    assert body["total"] == 2 and body["ingested"] == 2 and body["errors"] == 0


def test_a_second_sync_only_finds_duplicates(client, loaded_drive):
    job = make_job(client)
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/sync")
    body = client.post(f"/api/jobs/{job['id']}/sync").json()
    assert body["ingested"] == 0 and body["duplicates"] == 2


def test_sync_status_before_any_run(client):
    job = make_job(client)
    assert client.get(f"/api/jobs/{job['id']}/sync/status").json()["status"] == "never_run"


def test_sync_status_after_a_run(client, loaded_drive):
    job = make_job(client)
    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/sync")
    body = client.get(f"/api/jobs/{job['id']}/sync/status").json()
    assert body["status"] == "completed"
    assert body["processed"] == body["total"] == 2
    assert body["finished_at"] is not None


def test_sync_status_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/sync/status").status_code == 404


# ---------------------------------------------------------------------------
# Screening and results
# ---------------------------------------------------------------------------


def test_screening_an_unapproved_job_is_a_409(client, loaded_drive):
    job = make_job(client)
    response = client.post(f"/api/jobs/{job['id']}/screen")
    assert response.status_code == 409
    assert "approved JD" in response.json()["detail"]


def test_screening_an_unknown_job_is_a_404(client):
    assert client.post("/api/jobs/999/screen").status_code == 404


def test_screening_produces_routed_results(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    results = client.get(f"/api/jobs/{job['id']}/results").json()
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["routing"] == "auto_shortlist"
    assert results[1]["routing"] in {"preliminary_reject", "human_review"}


def test_results_carry_the_dimension_breakdown_and_evidence(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    top = client.get(f"/api/jobs/{job['id']}/results").json()[0]
    assert set(top["dimensions"]) == {
        "must_have_skills",
        "preferred_skills",
        "experience_years",
        "similar_experience",
        "education",
    }
    assert set(top["evidence"]) == {"preferred_skills", "similar_experience", "education"}
    for entry in top["evidence"].values():
        assert "quote" in entry and "verified" in entry
    assert top["confidence_detail"]["components"]
    assert top["audit"]["prompt_version"] and top["audit"]["model_name"]


def test_the_job_dashboard_counts_each_bucket(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    counts = client.get(f"/api/jobs/{job['id']}").json()["counts"]
    assert counts["total"] == 2
    assert sum(counts[b] for b in ("auto_shortlist", "human_review", "preliminary_reject")) == 2


def test_results_can_be_filtered_by_routing(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    shortlisted = client.get(f"/api/jobs/{job['id']}/results?routing=auto_shortlist").json()
    assert shortlisted and all(r["routing"] == "auto_shortlist" for r in shortlisted)


def test_results_can_be_filtered_by_score_and_confidence(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    assert client.get(f"/api/jobs/{job['id']}/results?min_score=99.9").json() == []
    assert client.get(f"/api/jobs/{job['id']}/results?max_score=0").json() == []
    assert client.get(f"/api/jobs/{job['id']}/results?min_confidence=1.0").json() == []


def test_results_can_be_filtered_by_flag(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    assert client.get(f"/api/jobs/{job['id']}/results?flag=nonexistent_flag").json() == []


def test_results_can_be_sorted(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    ascending = client.get(f"/api/jobs/{job['id']}/results?sort=score&order=asc").json()
    assert [r["score"] for r in ascending] == sorted(r["score"] for r in ascending)
    by_confidence = client.get(f"/api/jobs/{job['id']}/results?sort=confidence").json()
    assert len(by_confidence) == 2


def test_an_unknown_sort_field_is_a_400(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    response = client.get(f"/api/jobs/{job['id']}/results?sort=favourite_colour")
    assert response.status_code == 400
    assert "sort must be one of" in response.json()["detail"]


def test_results_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/results").status_code == 404


def test_rescreening_is_idempotent(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    before = client.get(f"/api/jobs/{job['id']}/results").json()
    client.post(f"/api/jobs/{job['id']}/screen")
    after = client.get(f"/api/jobs/{job['id']}/results").json()
    assert len(after) == len(before)
    assert [(r["candidate_id"], r["score"], r["routing"]) for r in after] == [
        (r["candidate_id"], r["score"], r["routing"]) for r in before
    ]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_a_candidate_record_includes_the_source_text(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    candidate_id = client.get(f"/api/jobs/{job['id']}/results").json()[0]["candidate_id"]
    body = client.get(f"/api/candidates/{candidate_id}").json()
    assert body["email"] == "layla.haddad@example.com"
    assert "Kubernetes" in body["source_text"]
    assert body["files"][0]["filename"] == "strong.pdf"
    assert body["results"]


def test_a_candidate_can_be_scoped_to_one_job(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    candidate_id = client.get(f"/api/jobs/{job['id']}/results").json()[0]["candidate_id"]
    body = client.get(f"/api/candidates/{candidate_id}?job_id={job['id']}").json()
    assert all(r["job_id"] == job["id"] for r in body["results"])


def test_an_unknown_candidate_is_a_404(client):
    assert client.get("/api/candidates/999").status_code == 404


# ---------------------------------------------------------------------------
# Review queue and the three human actions
# ---------------------------------------------------------------------------


def test_the_reject_reasons_are_a_closed_list(client):
    assert client.get("/api/review/reasons").json() == list(REJECT_REASONS)


def test_the_review_queue_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/review").status_code == 404


@pytest.fixture
def review_entry(client, loaded_drive, session, tmp_path):
    """Force one candidate into the review queue by tightening the job config."""
    job, _ = approved_job(client)
    client.put(
        f"/api/jobs/{job['id']}/config",
        json={
            "shortlist_score_min": 99.0,
            "reject_score_max": 1.0,
            "confidence_min": 0.7,
            "disagreement_cap": 35.0,
            "years_conflict_tolerance": 1.5,
        },
    )
    # A candidate missing Kubernetes whose stated years contradict their dates:
    # the flag keeps them in review, and the gap is what a human would correct.
    from scripts.generate_test_data import render_pdf

    partial = tmp_path / "partial.pdf"
    render_pdf(_lines(PARTIAL_CV_LINES, 6), partial)
    loaded_drive.add_file("folder-a", "file-partial", "partial.pdf", partial.read_bytes())

    client.post("/api/drive/folders/refresh")
    client.post(f"/api/jobs/{job['id']}/folders", json={"folder_ids": ["folder-a"]})
    client.post(f"/api/jobs/{job['id']}/sync")
    client.post(f"/api/jobs/{job['id']}/screen")
    queue = client.get(f"/api/jobs/{job['id']}/review").json()
    assert queue, "the tightened thresholds must produce a review queue"
    return job, queue


def test_review_entries_lead_with_the_routing_reason(client, review_entry):
    _, queue = review_entry
    entry = queue[0]
    assert entry["reasons"], "the reviewer must be told why before seeing the score"
    assert all("code" in r and "detail" in r for r in entry["reasons"])
    assert entry["dimensions"] and entry["evidence"]
    assert entry["review_status"] == "open"


def test_approving_from_review_shortlists_the_candidate(client, review_entry):
    job, queue = review_entry
    entry_id = queue[0]["review_entry_id"]
    body = client.post(f"/api/review/{entry_id}/approve", json={}).json()
    assert body["routing"] == "auto_shortlist"
    assert body["human_decision"] == "approved"
    remaining = client.get(f"/api/jobs/{job['id']}/review").json()
    assert entry_id not in [e["review_entry_id"] for e in remaining]


def test_rejecting_from_review_requires_a_listed_reason(client, review_entry):
    _, queue = review_entry
    response = client.post(
        f"/api/review/{queue[0]['review_entry_id']}/reject",
        json={"reason": "i just did not like them"},
    )
    assert response.status_code == 400
    assert "reason must be one of" in response.json()["detail"]


def test_rejecting_from_review_records_the_decision(client, review_entry):
    job, queue = review_entry
    entry_id = queue[0]["review_entry_id"]
    body = client.post(
        f"/api/review/{entry_id}/reject",
        json={"reason": "insufficient_experience", "note": "needs more depth"},
    ).json()
    assert body["routing"] == "rejected"
    assert body["human_decision"] == "rejected"
    remaining = client.get(f"/api/jobs/{job['id']}/review").json()
    assert entry_id not in [e["review_entry_id"] for e in remaining]


def test_a_resolved_entry_cannot_be_actioned_twice(client, review_entry):
    _, queue = review_entry
    entry_id = queue[0]["review_entry_id"]
    client.post(f"/api/review/{entry_id}/approve", json={})
    assert client.post(f"/api/review/{entry_id}/approve", json={}).status_code == 409


def test_actioning_an_unknown_entry_is_a_404(client):
    assert client.post("/api/review/999/approve", json={}).status_code == 404
    assert (
        client.post("/api/review/999/reject", json={"reason": "irrelevant_background"}).status_code
        == 404
    )


def _entry_missing_must_haves(queue):
    for entry in queue:
        if entry["rules"]["missing_must_have"]:
            return entry
    raise AssertionError("expected a queue entry with unmet must-haves")


def test_correcting_a_field_rescores_immediately(client, review_entry):
    """Supplying a genuinely missing must-have must raise the score."""
    _, queue = review_entry
    entry = _entry_missing_must_haves(queue)
    before_score = entry["score"]
    missing = entry["rules"]["missing_must_have"]

    body = client.post(
        f"/api/review/{entry['review_entry_id']}/correct",
        json={"corrections": {"add_skills": [{"name": s, "canonical": s} for s in missing]}},
    ).json()

    assert body["before"]["score"] == before_score
    assert body["score"] > before_score, "a correction must re-score the candidate"
    assert body["rules"]["missing_must_have"] == []
    assert body["audit"]["updated_at"] >= body["audit"]["created_at"]


def test_a_correction_can_change_the_routing_bucket(client, review_entry):
    """Clearing every hard failure must move the candidate out of its bucket."""
    _, queue = review_entry
    entry = _entry_missing_must_haves(queue)
    missing = entry["rules"]["missing_must_have"]

    body = client.post(
        f"/api/review/{entry['review_entry_id']}/correct",
        json={
            "corrections": {
                "add_skills": [{"name": s, "canonical": s} for s in missing],
                "add_education": [{"degree": "bachelor", "field": "Computer Science"}],
                "stated_years_experience": 9.0,
                "computed_years": 9.0,
            }
        },
    ).json()

    assert body["rules"]["hard_failures"] == []
    assert body["score"] > body["before"]["score"]


def test_a_preliminary_reject_reports_its_unmet_must_haves(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    rejects = [
        r
        for r in client.get(f"/api/jobs/{job['id']}/results").json()
        if r["routing"] == "preliminary_reject"
    ]
    assert rejects, "the designer CV must land in preliminary reject"
    assert rejects[0]["rules"]["missing_must_have"]
    assert rejects[0]["rules"]["hard_failures"]


def test_correcting_with_no_changes_is_a_400(client, review_entry):
    _, queue = review_entry
    response = client.post(
        f"/api/review/{queue[0]['review_entry_id']}/correct", json={"corrections": {}}
    )
    assert response.status_code == 400


def test_correcting_an_unknown_entry_is_a_404(client):
    response = client.post("/api/review/999/correct", json={"corrections": {"full_name": "X"}})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Preliminary rejects need a human
# ---------------------------------------------------------------------------


def test_preliminary_rejects_are_listed_for_confirmation(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    rejects = client.get(f"/api/jobs/{job['id']}/preliminary-rejects").json()
    assert rejects
    assert all(r["routing"] == "preliminary_reject" for r in rejects)
    assert all(r["human_decision"] is None for r in rejects)


def test_confirming_rejects_records_the_human_decision(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    rejects = client.get(f"/api/jobs/{job['id']}/preliminary-rejects").json()
    ids = [r["id"] for r in rejects]
    body = client.post(f"/api/jobs/{job['id']}/confirm-rejects", json={"result_ids": ids}).json()
    assert body["count"] == len(ids)
    after = client.get(f"/api/jobs/{job['id']}/results").json()
    confirmed = [r for r in after if r["id"] in ids]
    assert all(r["routing"] == "rejected" and r["human_decision"] == "rejected" for r in confirmed)


def test_confirming_rejects_ignores_results_in_other_buckets(client, loaded_drive):
    job, _ = screened_job(client, loaded_drive)
    shortlisted = [
        r["id"]
        for r in client.get(f"/api/jobs/{job['id']}/results").json()
        if r["routing"] == "auto_shortlist"
    ]
    body = client.post(
        f"/api/jobs/{job['id']}/confirm-rejects", json={"result_ids": shortlisted}
    ).json()
    assert body["count"] == 0, "only preliminary rejects may be confirmed"


def test_confirming_rejects_on_an_unknown_job_is_a_404(client):
    assert (
        client.post("/api/jobs/999/confirm-rejects", json={"result_ids": []}).status_code == 404
    )


def test_preliminary_rejects_of_an_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/999/preliminary-rejects").status_code == 404
