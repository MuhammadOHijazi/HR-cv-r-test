# CV Screening Pipeline — Implementation Plan

Implements the "CV Screening Pipeline" architecture document, v2.

## Core principles (non-negotiable)

1. **The LLM reads, rules decide, humans judge.** No candidate is ever *finally*
   rejected by the machine. The lowest automated bucket is `preliminary_reject`,
   a queue awaiting one-click human batch confirmation.
2. **Every score carries verified evidence.** Every evidence quote returned by a
   model is checked against the source text (fuzzy ratio >= 0.8 on
   whitespace-normalised text). An unverifiable quote forces that field's
   confidence to 0.0.
3. **Low confidence beats a high score.** Confidence below the per-job threshold
   always routes to human review, regardless of score.
4. **Identity is masked from all evaluators.** Name, email, phone, photo
   references, age, gender and nationality signals are stripped before the judge
   prompt is built.

## Repo tree (final)

    .
    |-- backend/
    |   |-- app/
    |   |   |-- main.py              FastAPI app factory + router mounting
    |   |   |-- config.py            pydantic-settings; every env var + default
    |   |   |-- db.py                engine / session / Base / init_db()
    |   |   |-- api/
    |   |   |   |-- deps.py          DB session dependency, service singletons
    |   |   |   |-- jobs.py          jobs + JD structuring / approval / versions
    |   |   |   |-- drive.py         folder listing, assignment, sync
    |   |   |   |-- candidates.py    candidate record + source-text viewer
    |   |   |   |-- screening.py     run screening, ranked results
    |   |   |   |-- review.py        review queue + the three human actions
    |   |   |   `-- settings_api.py  key-pool status, service-account email
    |   |   |-- core/
    |   |   |   |-- gemini_client.py THE single Gemini gateway (key rotation)
    |   |   |   |-- prompts.py       versioned prompt templates + JSON schemas
    |   |   |   |-- drive_client.py  service-account Drive gateway
    |   |   |   |-- ingestion.py     sync orchestration, dedup, archival
    |   |   |   |-- textextract.py   PyMuPDF / python-docx / vision fallback
    |   |   |   |-- regexlayer.py    deterministic email/phone/url/date layer
    |   |   |   |-- extraction.py    schema extraction + validation + retry
    |   |   |   |-- evidence.py      verbatim-quote verification
    |   |   |   |-- taxonomy.py      bilingual AR/EN skill normalisation
    |   |   |   |-- jd.py            JD structuring / approval / versioning
    |   |   |   |-- scoring.py       rules gate + semantic + judge + merge
    |   |   |   |-- confidence.py    confidence assembly
    |   |   |   |-- routing.py       two-axis decision matrix + flags
    |   |   |   |-- injection.py     injection heuristic + untrusted-data wrap
    |   |   |   |-- masking.py       identity masking
    |   |   |   `-- audit.py         audit-log helper
    |   |   |-- models/entities.py   all SQLAlchemy models
    |   |   `-- schemas/api.py       pydantic request/response models
    |   `-- tests/                   pytest suite (unit + integration + e2e)
    |-- frontend/                    React + Vite, six pages
    |-- scripts/
    |   |-- generate_test_data.py    2 JDs + 14 synthetic CVs + expectations
    |   |-- live_smoke_test.py       3 CVs through the REAL Gemini API
    |   `-- seed_taxonomy.py         ~100 bilingual skills into the DB
    |-- .env.example
    `-- PLAN.md  DECISIONS.md  TEST_REPORT.md  README.md

## Database schema

| table | key columns |
|---|---|
| `jobs` | id, title, raw_jd_text, status, active_jd_version_id, created_at |
| `jd_versions` | id, job_id, version, structured_json (must / nice / weights / thresholds / responsibilities), approved, approved_by, approved_at, source_model, prompt_version, created_at |
| `job_configs` | job_id, shortlist_score_min, reject_score_max, confidence_min, disagreement_cap, years_conflict_tolerance, weights_json |
| `drive_folders` | id, folder_id (unique), name, connected_at, last_synced_at |
| `job_folders` | job_id, folder_id |
| `candidates` | id, canonical_key (normalised email+phone, unique), full_name, email, phone |
| `cv_files` | id, candidate_id, drive_file_id, folder_id, filename, mime_type, md5_checksum (unique), size, raw_path, text_path, source_quality, is_scanned, page_count, ingested_at |
| `extractions` | id, cv_file_id, schema_version, prompt_version, model, payload_json, field_confidence_json, evidence_json, stated_years, computed_years, years_conflict, retries |
| `embeddings` | id, content_hash (unique), model, dim, vector BLOB |
| `screening_results` | id, job_id, jd_version_id, candidate_id, cv_file_id, rules_json, semantic_json, judge_json, merged_score, confidence, routing, flags_json, dimension_breakdown_json, prompt_version, schema_version, model_name, thresholds_json, created_at; unique (job_id, candidate_id) so a full re-screen is idempotent |
| `review_queue` | id, screening_result_id, reasons_json, status, resolved_action, resolved_at |
| `audit_log` | id, entity_type, entity_id, action, actor, before_json, after_json, created_at |
| `api_key_usage` | id, key_index, key_last4, requests, failures, rate_limit_hits, cooldown_until, last_used_at |
| `skills_taxonomy` | id, canonical (unique), aliases_json (AR + EN), category |
| `sync_runs` | id, job_id, status, total, processed, new_files, duplicates, errors_json, started_at, finished_at |

## API endpoints

    GET    /api/health
    GET    /api/settings                       key-pool status + SA email
    GET    /api/jobs                           jobs + per-bucket counts
    POST   /api/jobs                           create job from raw JD text
    GET    /api/jobs/{id}
    POST   /api/jobs/{id}/structure            Gemini -> new unapproved version
    GET    /api/jobs/{id}/versions
    PUT    /api/jobs/{id}/versions/{v}         edit -> creates a new version
    POST   /api/jobs/{id}/versions/{v}/approve
    GET/PUT /api/jobs/{id}/config              per-job thresholds
    GET    /api/drive/status
    GET    /api/drive/folders
    POST   /api/drive/folders/refresh
    GET/POST /api/jobs/{id}/folders            folder assignment
    POST   /api/jobs/{id}/sync                 sync now
    GET    /api/jobs/{id}/sync/status
    POST   /api/jobs/{id}/screen               run screening (idempotent)
    GET    /api/jobs/{id}/results              ranked / sortable / filterable
    GET    /api/candidates/{id}                full record incl. source text
    GET    /api/jobs/{id}/review               review queue
    POST   /api/review/{id}/approve
    POST   /api/review/{id}/reject             closed-list reason
    POST   /api/review/{id}/correct            correction -> re-score + re-route
    POST   /api/jobs/{id}/confirm-rejects      human batch confirmation

## Gemini client module boundary

`backend/app/core/gemini_client.py` is the **only** module that talks to the
Gemini SDK. Everything else depends on this interface:

    class GeminiClient:
        generate_structured(prompt, response_schema, *, model=None, temperature=0.0) -> dict
        judge(prompt, response_schema, *, model=None) -> dict
        embed(texts, *, model=None) -> list[list[float]]
        vision_extract(prompt, images, response_schema=None) -> dict
        key_pool_status() -> list[KeyStatus]

Internals: `_KeyPool` (round-robin cursor, per-key `cooldown_until`, counters
persisted to `api_key_usage`), an injectable `Transport` protocol (tests supply a
fake), exponential backoff starting 30 s doubling to a 900 s cap, retry of the
*same* request on the next healthy key, and `AllKeysExhausted` when every key is
cooling down. Logs only `key[i] ...abcd` — never a full key.

## Scoring and routing maths

* **Rules gate** — min years, required degree, required certs, must-have skills.
  A failure on a HIGH-confidence field is a `hard_fail`; the same failure on a
  LOW-confidence field is a `soft_fail` that forces human review.
* **Semantic** — mean of the top-k cosine similarities between JD
  responsibilities and candidate work-history highlights, embeddings cached by
  content hash.
* **Judge** — 4-level written rubric per dimension (0 / 40 / 70 / 100),
  temperature 0, evidence quote mandatory per dimension.
* **Years score** — saturation curve `1 - exp(-k * y / required)` with `k` chosen
  so `y == required` -> 0.75 and `y == 2*required` -> ~0.94.
* **Merge** — weighted sum of dimension scores using the approved JD weights,
  scaled to 0-100.
* **Confidence** — 0.40 mean field confidence + 0.20 source quality + 0.25
  evidence-verification rate + 0.15 scorer agreement; disagreement above the
  per-job cap hard-caps confidence at 0.65.
* **Routing** — the two-axis matrix, thresholds read from `job_configs`.

## Phase gates

P0 scaffold -> P1 Gemini client -> P2 Drive ingestion -> P3 extraction ->
P4 JD + scoring + routing -> P5 API + frontend -> P6 synthetic data, e2e and
coverage -> P7 documentation.
