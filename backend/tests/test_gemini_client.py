"""Phase 1 gate — the Gemini client's rotation, failover, backoff and accounting."""

from __future__ import annotations

import json

import pytest

from backend.app.core.gemini_client import (
    AllKeysExhausted,
    GeminiClient,
    InMemoryUsageStore,
    KeyPool,
    MalformedResponse,
    ModelConfig,
    NonRetryableError,
    TransportError,
    mask_key,
)

KEYS = ["AAAAkey-1111", "BBBBkey-2222", "CCCCkey-3333"]


class ScriptedTransport:
    """A transport that replays a scripted outcome per call."""

    def __init__(self, script=None, *, default="{}"):
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict] = []

    def _next(self, api_key: str, kind: str):
        self.calls.append({"key": api_key, "kind": kind})
        outcome = self.script.pop(0) if self.script else self.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
        self.calls[-1:] = self.calls[-1:]
        return self._next(api_key, "generate")

    def embed(self, *, api_key, model, texts):
        result = self._next(api_key, "embed")
        return result if isinstance(result, list) else [[0.1, 0.2] for _ in texts]


def build(transport, clock=None, keys=KEYS, **kwargs):
    return GeminiClient(
        transport=transport,
        keys=keys,
        models=ModelConfig("flash", "pro", "embed", "flash", 8),
        clock=clock or (lambda: 0.0),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Key masking — no full key may ever be logged
# ---------------------------------------------------------------------------


def test_mask_key_shows_only_last_four():
    assert mask_key("SUPERSECRETKEY9876") == "...9876"
    assert "SUPERSECRET" not in mask_key("SUPERSECRETKEY9876")


def test_mask_key_handles_short_values():
    assert mask_key("ab") == "...****"


def test_status_never_contains_a_full_key(fake_clock):
    client = build(ScriptedTransport(), clock=fake_clock)
    blob = json.dumps(client.key_pool_status())
    for key in KEYS:
        assert key not in blob
    assert "...1111" in blob


def test_log_output_masks_keys(caplog, fake_clock):
    transport = ScriptedTransport([TransportError(429, "quota"), "{}"])
    client = build(transport, clock=fake_clock)
    with caplog.at_level("WARNING"):
        client.generate_structured("p", {})
    text = caplog.text
    assert text
    for key in KEYS:
        assert key not in text
    assert "...1111" in text


# ---------------------------------------------------------------------------
# Rotation order
# ---------------------------------------------------------------------------


def test_requests_round_robin_across_the_pool(fake_clock):
    transport = ScriptedTransport()
    client = build(transport, clock=fake_clock)
    for _ in range(6):
        client.generate_structured("p", {})
    used = [c["key"] for c in transport.calls]
    assert used == KEYS * 2


def test_pool_acquire_wraps_around(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock)
    assert [pool.acquire()[0] for _ in range(5)] == [0, 1, 2, 0, 1]


def test_empty_key_pool_is_rejected():
    with pytest.raises(ValueError, match="GEMINI_API_KEYS"):
        KeyPool([])


# ---------------------------------------------------------------------------
# 429 failover
# ---------------------------------------------------------------------------


def test_rate_limited_key_fails_over_to_the_next_key(fake_clock):
    transport = ScriptedTransport([TransportError(429, "quota exceeded"), '{"ok": true}'])
    client = build(transport, clock=fake_clock)
    assert client.generate_structured("p", {}) == {"ok": True}
    assert [c["key"] for c in transport.calls] == [KEYS[0], KEYS[1]]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_every_retryable_status_rotates(status, fake_clock):
    transport = ScriptedTransport([TransportError(status, "boom"), '{"ok": 1}'])
    client = build(transport, clock=fake_clock)
    assert client.generate_structured("p", {}) == {"ok": 1}
    assert len(transport.calls) == 2


def test_the_same_request_is_replayed_on_the_next_key(fake_clock):
    transport = ScriptedTransport([TransportError(429, "q"), '{"v": 2}'])
    client = build(transport, clock=fake_clock)
    client.generate_structured("the-identical-prompt", {})
    assert len(transport.calls) == 2
    assert transport.calls[0]["kind"] == transport.calls[1]["kind"] == "generate"


def test_a_non_retryable_status_does_not_rotate(fake_clock):
    transport = ScriptedTransport([TransportError(400, "bad request"), '{"ok": 1}'])
    client = build(transport, clock=fake_clock)
    with pytest.raises(NonRetryableError):
        client.generate_structured("p", {})
    assert len(transport.calls) == 1, "a bad request must not burn the whole pool"


def test_an_untyped_exception_also_rotates(fake_clock):
    transport = ScriptedTransport([RuntimeError("socket died"), '{"ok": 1}'])
    client = build(transport, clock=fake_clock)
    assert client.generate_structured("p", {}) == {"ok": 1}
    assert len(transport.calls) == 2


# ---------------------------------------------------------------------------
# Backoff timing (fake clock)
# ---------------------------------------------------------------------------


def test_cooldown_starts_at_thirty_seconds(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0)
    assert pool.record_failure(0, 429) == 30.0
    assert pool.status()[0].cooldown_until == fake_clock.now + 30.0


def test_cooldown_doubles_on_consecutive_failures(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0, max_cooldown=900.0)
    assert [pool.record_failure(0, 429) for _ in range(6)] == [30, 60, 120, 240, 480, 900]


def test_cooldown_is_capped_at_fifteen_minutes(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0, max_cooldown=900.0)
    for _ in range(12):
        cooldown = pool.record_failure(0, 429)
    assert cooldown == 900.0


def test_a_cooling_key_is_skipped_until_it_expires(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0)
    pool.record_failure(0, 429)
    assert pool.acquire()[0] == 1
    fake_clock.advance(29)
    assert 0 not in {pool.acquire()[0] for _ in range(3)}
    fake_clock.advance(2)
    assert 0 in {pool.acquire()[0] for _ in range(3)}


def test_success_clears_the_cooldown_and_the_streak(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0)
    pool.record_failure(0, 429)
    pool.record_failure(0, 429)
    pool.record_success(0)
    assert pool.status()[0].cooldown_until == 0.0
    assert pool.status()[0].consecutive_failures == 0
    assert pool.record_failure(0, 429) == 30.0, "backoff must restart, not resume"


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


def test_all_keys_exhausted_when_every_key_is_cooling(fake_clock):
    transport = ScriptedTransport([TransportError(429, "q")] * 3)
    client = build(transport, clock=fake_clock)
    with pytest.raises(AllKeysExhausted) as exc:
        client.generate_structured("p", {})
    assert exc.value.retry_after > 0
    assert "cooling down" in str(exc.value)


def test_exhaustion_error_reports_the_shortest_wait(fake_clock):
    pool = KeyPool(KEYS, clock=fake_clock, base_cooldown=30.0)
    pool.record_failure(0, 429)
    fake_clock.advance(10)
    pool.record_failure(1, 429)
    pool.record_failure(2, 429)
    assert pool.time_until_available() == pytest.approx(20.0)
    with pytest.raises(AllKeysExhausted):
        pool.acquire()


def test_a_job_can_resume_after_the_cooldown_expires(fake_clock):
    transport = ScriptedTransport([TransportError(429, "q")] * 3)
    client = build(transport, clock=fake_clock)
    with pytest.raises(AllKeysExhausted):
        client.generate_structured("p", {})
    fake_clock.advance(31)
    transport.script = ['{"resumed": true}']
    assert client.generate_structured("p", {}) == {"resumed": True}


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_counters_track_requests_failures_and_rate_limits(fake_clock):
    transport = ScriptedTransport([TransportError(429, "q"), "{}"])
    client = build(transport, clock=fake_clock)
    client.generate_structured("p", {})
    status = {s["index"]: s for s in client.key_pool_status()}
    assert status[0]["failures"] == 1
    assert status[0]["rate_limit_hits"] == 1
    assert status[1]["requests"] == 1
    assert status[1]["failures"] == 0


def test_a_5xx_is_not_counted_as_a_rate_limit(fake_clock):
    transport = ScriptedTransport([TransportError(503, "unavailable"), "{}"])
    client = build(transport, clock=fake_clock)
    client.generate_structured("p", {})
    assert client.key_pool_status()[0]["rate_limit_hits"] == 0
    assert client.key_pool_status()[0]["failures"] == 1


def test_counters_persist_through_the_usage_store(fake_clock):
    store = InMemoryUsageStore()
    first = build(ScriptedTransport(), clock=fake_clock, usage_store=store)
    for _ in range(3):
        first.generate_structured("p", {})

    revived = build(ScriptedTransport(), clock=fake_clock, usage_store=store)
    assert sum(s["requests"] for s in revived.key_pool_status()) == 3


def test_usage_store_survives_a_failure_too(fake_clock):
    store = InMemoryUsageStore()
    client = build(
        ScriptedTransport([TransportError(429, "q"), "{}"]), clock=fake_clock, usage_store=store
    )
    client.generate_structured("p", {})
    assert store.load()[0]["rate_limit_hits"] == 1


# ---------------------------------------------------------------------------
# Typed wrappers
# ---------------------------------------------------------------------------


def test_judge_uses_the_judge_tier_model(fake_clock):
    seen = {}

    class T(ScriptedTransport):
        def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
            seen["model"] = model
            seen["temperature"] = temperature
            return '{"dimensions": {}}'

    client = build(T(), clock=fake_clock)
    client.judge("prompt", {})
    assert seen["model"] == "pro"
    assert seen["temperature"] == 0.0, "the judge must always run at temperature 0"


def test_generate_structured_uses_the_extraction_tier(fake_clock):
    seen = {}

    class T(ScriptedTransport):
        def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
            seen["model"] = model
            return "{}"

    client = build(T(), clock=fake_clock)
    client.generate_structured("p", {})
    assert seen["model"] == "flash"


def test_embed_returns_one_vector_per_text(fake_clock):
    client = build(ScriptedTransport(), clock=fake_clock)
    assert len(client.embed(["a", "b", "c"])) == 3


def test_embed_of_nothing_makes_no_call(fake_clock):
    transport = ScriptedTransport()
    client = build(transport, clock=fake_clock)
    assert client.embed([]) == []
    assert transport.calls == []


def test_vision_extract_passes_the_images_through(fake_clock):
    seen = {}

    class T(ScriptedTransport):
        def generate(self, *, api_key, model, prompt, response_schema, temperature, images=None):
            seen["images"] = list(images or [])
            seen["model"] = model
            return '{"text": "recovered"}'

    client = build(T(), clock=fake_clock)
    assert client.vision_extract("p", [b"png1", b"png2"], {"type": "object"}) == {
        "text": "recovered"
    }
    assert seen["images"] == [b"png1", b"png2"]
    assert seen["model"] == "flash"


def test_vision_extract_without_a_schema_returns_raw_text(fake_clock):
    client = build(ScriptedTransport(["plain text"]), clock=fake_clock)
    assert client.vision_extract("p", [b"x"]) == "plain text"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_fenced_json_is_unwrapped(fake_clock):
    client = build(ScriptedTransport(['```json\n{"a": 1}\n```']), clock=fake_clock)
    assert client.generate_structured("p", {}) == {"a": 1}


def test_malformed_json_raises_malformed_response(fake_clock):
    client = build(ScriptedTransport(["not json at all"]), clock=fake_clock)
    with pytest.raises(MalformedResponse):
        client.generate_structured("p", {})


def test_a_json_array_is_rejected(fake_clock):
    client = build(ScriptedTransport(["[1, 2, 3]"]), clock=fake_clock)
    with pytest.raises(MalformedResponse):
        client.generate_structured("p", {})


def test_a_non_string_response_is_rejected(fake_clock):
    client = build(ScriptedTransport([12345]), clock=fake_clock)
    with pytest.raises(NonRetryableError):
        client.generate_structured("p", {})


def test_a_dict_response_passes_straight_through(fake_clock):
    client = build(ScriptedTransport([{"already": "parsed"}]), clock=fake_clock)
    assert client.generate_structured("p", {}) == {"already": "parsed"}


def test_max_attempts_never_exceeds_the_pool_size(fake_clock):
    transport = ScriptedTransport([TransportError(429, "q")] * 10)
    client = build(transport, clock=fake_clock, max_attempts=99)
    with pytest.raises(AllKeysExhausted):
        client.generate_structured("p", {})
    assert len(transport.calls) == len(KEYS), "each key is tried at most once per request"


def test_a_single_key_pool_still_works(fake_clock):
    transport = ScriptedTransport(['{"ok": 1}'])
    client = build(transport, clock=fake_clock, keys=["ZZZZkey-9999"])
    assert client.generate_structured("p", {}) == {"ok": 1}
