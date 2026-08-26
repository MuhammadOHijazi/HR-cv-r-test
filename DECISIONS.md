# DECISIONS

Every judgement call made while building this, and why. The brief said to
choose sensibly where a detail was unspecified and record it here; this is that
record. Decisions are grouped by the area they affect.

---

## Environment and dependencies

### D1. Python 3.14 rather than 3.11
The brief locked "Python 3.11+". The only interpreter on this machine is
3.14.0, which satisfies that. Every pinned dependency has a working cp314
wheel, so nothing was compromised.

### D2. `import pymupdf`, not `import fitz`
PyMuPDF 1.28 no longer ships the legacy `fitz` alias — `import fitz` raises
`ModuleNotFoundError`. `backend/app/core/textextract.py` imports `pymupdf`
directly. Anyone porting this to an older PyMuPDF should note the reverse
applies below 1.24.

### D3. PDF rendering in the generator uses PyMuPDF, not reportlab
Both are installed and both were acceptable per the brief ("use
reportlab/python-docx"). PyMuPDF was already a hard dependency for *reading*
PDFs, so using it for writing avoids a second PDF engine in the fixture path
and guarantees the generator and the extractor agree about what a PDF is.
reportlab remains in the dependency list as the brief specified it.

---

## Synthetic corpus

### D4. Date ranges in the corpus are anchored to the current year
A fixture that hard-codes "2016 - present" alongside "9 years of experience"
silently drifts into a stated-vs-computed contradiction as real time passes,
and the flow test would then start failing on a calendar boundary rather than
on a code change. Every date in `scripts/generate_test_data.py` is written as
`Y(n)` — "n years before today" — so the corpus stays self-consistent forever.
`contradiction_years` is the deliberate exception: its dates are *meant* to
disagree with its claim.

### D5. The Arabic and mixed-language CVs are DOCX, not PDF
PyMuPDF's built-in fonts have no Arabic glyph coverage, and writing Arabic into
a PDF requires either a system TTF (not portable — a Linux CI box has no
`C:\Windows\Fonts`) or PyMuPDF's HTML box, which stores Arabic as *presentation
forms* in visual order. Neither is representative: real Arabic CVs come out of
Word and InDesign, which store logical-order Unicode — exactly what python-docx
produces. Rendering the Arabic fixtures as DOCX therefore gives a *more*
realistic fixture and a deterministic one on every platform.

This is a fixture-generation limit, not a pipeline limit. The pipeline handles
Arabic PDFs: `textextract.normalise_extracted()` NFKC-folds Arabic presentation
forms back to base letters and strips bidi controls, which is precisely what a
real Arabic PDF needs, and it is covered by
`test_arabic_presentation_forms_normalise_to_base_letters`.

### D6. Expected routing buckets follow the spec's rules, not first intuition
Several CVs were initially labelled "borderline -> human review" out of habit.
The spec is explicit that *a must-have failure on a high-confidence field routes
toward reject*, so `partial_backend_1` (no Kubernetes) and `partial_backend_2`
(four years against a five-year floor) correctly land in `preliminary_reject` —
still queued for human confirmation, never final. The expectations were
corrected to match the specified rule rather than the code being bent to match
the guess.

Likewise `arabic_backend` and `mixed_backend` were initially expected in review.
They meet every must-have, so they auto-shortlist — and that is the *strongest*
demonstration of "both must work through the same code path": a qualified
Arabic candidate reaches the same bucket as a qualified English one.

---

## The Gemini client

### D7. `Transport` is a protocol; the SDK lives behind it
`gemini_client.py` contains no Gemini import at all. `gemini_transport.py` holds
`GenaiTransport` (the only `google-genai` importer) and `MockTransport`. This is
what makes the entire suite runnable without credentials while keeping *one*
gateway, as the brief required.

### D8. Each key is tried at most once per request
`max_attempts` is clamped to the pool size. Retrying the same key inside a
single request would just re-hit the quota that made it cool down; rotating
across the pool once and then raising `AllKeysExhausted` lets the caller pause
and resume, which is the behaviour the brief asked for.

### D9. Key-usage counters are written behind the request, not through
Originally every Gemini call persisted its counters immediately on a fresh
session. On SQLite that opens a second connection in the middle of whatever
transaction the request is running, and returns `database is locked` — which is
exactly how a review correction failed with a 500 during the browser
walkthrough. Counters are now buffered in memory and flushed by an HTTP
middleware once the request's work is done. A failed flush is retried later and
never propagates into a Gemini call: bookkeeping must not be able to fail a
screening run.

### D10. SQLite runs in WAL mode with a busy timeout and foreign keys on
`journal_mode=WAL` lets readers and one writer proceed concurrently,
`busy_timeout=10000` makes a contending writer wait rather than fail instantly,
and `foreign_keys=ON` actually enforces the relationships the schema declares.
Turning FK enforcement on immediately caught two tests inserting orphan
`job_folders` rows, which is the point.

---

## Extraction

### D11. The deterministic layer overrides the model on contact details
Regex finds e-mail addresses and phone numbers perfectly; an LLM can
hallucinate them. Where the two disagree, `regexlayer` wins, and its findings
are also passed into the prompt as verified hints.

### D12. A failed extraction degrades, it does not discard
After `EXTRACTION_MAX_RETRIES` re-prompts, the pipeline keeps the deterministic
findings, records `extraction_failed_after_retries`, caps confidence at 0.3 and
routes to human review. Dropping the candidate entirely would silently lose a
real applicant to a transient model failure.

### D13. Years conflicts are reported, never resolved
When stated and computed years disagree beyond tolerance, both figures are
stored and the conflict is flagged. The rules gate then treats the years
evidence as unreliable, which downgrades a min-years failure from hard to soft.
The machine does not get to decide which number the candidate meant.

---

## Scoring

### D14. Semantic similarity is calibrated against a control sentence
Raw cosine is not comparable across embedding models — genuinely related short
texts sit anywhere from 0.3 to 0.9 depending on the provider — so a fixed
threshold would be meaningless and a fixed linear scaling made the semantic
scorer disagree with everything. The scorer embeds a deliberately unrelated
control sentence alongside the job's responsibilities, measures what "no
relationship" scores on *this* model, and reports the distance above that
floor.

An earlier version used the mean of the candidate's own similarity grid as the
baseline. That fails when a CV has only one or two highlights: the grid average
is then dominated by the good matches and cancels out the very signal being
measured. The fixed control has no such degeneracy.

### D15. Semantic score is coverage, not the best few matches
Averaging only the top-k best-matching responsibilities let a single lucky
pairing carry an otherwise unrelated CV — a graphic designer scored 32 against a
backend role. The score is now the mean over *every* responsibility of how well
the candidate's best highlight matches it, which is what "similar experience"
actually means.

### D16. Scorer disagreement compares only what is measured twice
Comparing the scorers' overall verdicts is meaningless: the rules gate measures
hard requirements while the judge also weighs nice-to-haves, so a candidate who
clears every must-have but few preferred skills looks like a disagreement when
the scorers are simply answering different questions. Only two quantities are
genuinely measured twice by independent methods — *similar experience*
(embeddings vs. the judge's rubric) and *education fit* (deterministic degree
comparison vs. the judge) — and disagreement is the larger of those two gaps.

### D17. The disagreement cap defaults to 35, not 25
The judge scores on a four-level rubric (0/40/70/100), so consecutive levels are
30 points apart. Any cap below 30 reads a one-level difference of opinion as
disagreement and caps confidence on perfectly ordinary candidates. 35 means only
a two-level gap counts.

### D18. The years curve is anchored at 0.75, and at three years when unstated
`years == required` scores 0.75 and `2 x required` about 0.94, so meeting the
bar is clearly rewarded while the tenth year adds far less than the third. When
a JD states no minimum the curve anchors at three years, so more experience
still scores higher instead of everything collapsing to a constant.

### D19. A hard must-have failure caps the merged score at 40
Without this a candidate could fail a stated requirement and still out-rank
someone who met it, purely on nice-to-haves. The cap keeps the ranking honest;
the *routing* decision is still made separately by the router.

---

## Routing

### D20. A review flag outranks the preliminary-reject rule
The spec's second bullet routes a confident must-have failure toward reject, and
its third routes "any review flag" to human review. For a candidate with both,
the flag wins. A flag means something about the record is untrustworthy — bad
OCR, a years contradiction, an injection attempt — and queueing a rejection,
however routine, on evidence already marked as suspect is exactly the failure
mode the architecture exists to prevent.

### D21. There is no `rejected` routing outcome
`route()` can only return `auto_shortlist`, `human_review` or
`preliminary_reject`. `rejected` is set solely by a human action endpoint. This
is enforced by a test that sweeps the routing input space and asserts
`rejected` never appears.

### D22. Rejection reasons are a closed list served by the API
`GET /api/review/reasons` returns the seven permitted reasons and the reject
endpoint 400s on anything else, so rejections stay analysable instead of
becoming free text.

---

## Prompt-injection defence

### D23. Delimiters in CV content are neutralised, not just wrapped
Wrapping untrusted text in `<<<BEGIN_UNTRUSTED_DATA>>>` is useless if the CV can
write the closing delimiter itself. `neutralise_delimiters()` rewrites any
`<<<..._...>>>` token found in the source before wrapping, so the block cannot
be closed early or forged. A test asserts the closing delimiter appears exactly
once in the final prompt.

### D24. A detected injection changes the routing, never the score
The heuristic sets a review flag. It does not penalise the candidate — someone
may legitimately have "ignore previous instructions" in a CV about prompt
engineering. A test asserts the injection CV's score stays below the honest
strong CVs, proving the payload had no effect.

---

## Frontend

### D25. Hash routing and a Vite dev proxy
`HashRouter` means the built `dist/` works from any static host with no
server-side rewrite rules. In development, Vite proxies `/api` to port 8000 so
the frontend is origin-relative and there are no CORS surprises; the backend
also allows the Vite origin explicitly.

### D26. Hand-rolled CSS, no design system
The brief said "clean, functional, no design system required". A single
`styles.css` keeps the dependency tree at three packages (react, react-dom,
react-router-dom) and the production bundle at ~64 kB gzipped.

### D27. The review card leads with the reason, not the score
Each queue entry renders "Why this is here" above the candidate's name and
score, so the reviewer reads the machine's uncertainty before they see a number
that might anchor them.

---

## Mock mode

### D28. Mock mode is a usable product mode, not just a test double
With `MOCK_MODE=true` the fake Drive is seeded from `MOCK_DRIVE_DIR` (the
generated synthetic corpus by default), so the entire application — sync,
screen, review, correct — is clickable end to end with no credentials at all.
That is what makes the README's five-minute walkthrough possible.

### D29. The mock Gemini shares the application's real skills taxonomy
The fake originally kept its own hard-coded skill list, which made the Arabic CV
extract almost nothing and fail its must-haves for the wrong reason. It now
reads `SEED_TAXONOMY`, matches any alias in any language, and folds tokens
through the taxonomy when producing embeddings — so it behaves like a genuinely
multilingual model while staying completely deterministic. It also extracts
whatever a CV lists under "Skills" even when the taxonomy has never heard of it,
which is what a real extractor does and what stops non-technical CVs extracting
to nothing.

### D30. The mock vision fallback transcribes only what it was told
`MockTransport.register_vision()` maps page-image hashes to known text. An
unregistered scanned page transcribes to nothing, which correctly drives the CV
into review with `low_ocr_quality` rather than inventing content. That is the
honest mock-mode answer, and it is why the scanned CV scores 0.0 in a bare mock
run but recovers its text in the flow test.

---

## Testing

### D31. The test engine is the production engine
`conftest.py` builds its engine through `db.get_engine()` rather than calling
`create_engine` itself. An earlier hand-rolled test engine silently skipped the
SQLite pragmas, so the tests were not exercising the configuration that ships.

### D32. Warnings are errors
`filterwarnings = ["error"]` with two narrow third-party ignores. This is what
surfaced the deprecated `@app.on_event("startup")`, now migrated to a lifespan
handler.

### D33. The live smoke test is a script, not a pytest case
The brief asked for `scripts/live_smoke_test.py` and for a suite with no skips
other than the credential-gated test. Keeping it as a script gives a suite with
**zero** skips, and a pytest case asserts the script exits cleanly with a clear
message when no keys are configured.
