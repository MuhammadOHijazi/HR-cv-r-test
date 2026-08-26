"""The single Gemini gateway for the whole application.

Nothing else in the codebase is allowed to import the Gemini SDK.  Extraction,
JD structuring, judging, embeddings and vision all go through this module so that
key rotation, cooldown, retry and usage accounting happen in exactly one place.

Design
------
* ``GEMINI_API_KEYS`` is a comma-separated pool.  Requests round-robin over it.
* A 429 / quota / 5xx response marks the key as cooling down (exponential
  backoff starting at 30 s, doubling to a 900 s cap) and the *same* request is
  retried on the next healthy key.
* When every key is cooling down we raise :class:`AllKeysExhausted` so callers
  can pause and resume a job instead of crashing.
* Per-key counters are persisted through an injectable
  :class:`UsageStore`, and surfaced in the Settings page.
* Logs only ever contain ``key[i] ...abcd`` — never a full key.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

logger = logging.getLogger(__name__)

RATE_LIMIT_STATUSES = {429}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class GeminiError(RuntimeError):
    """Base class for gateway errors."""


class AllKeysExhausted(GeminiError):
    """Raised when every key in the pool is cooling down."""

    def __init__(self, retry_after: float, message: str | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            message
            or f"All Gemini API keys are cooling down; retry in {retry_after:.0f}s"
        )


class TransportError(GeminiError):
    """A transport-level failure carrying an HTTP-ish status code."""

    def __init__(self, status: int, message: str = "") -> None:
        self.status = status
        super().__init__(f"gemini transport error {status}: {message}")


class NonRetryableError(GeminiError):
    """A failure that no amount of key rotation will fix (e.g. a bad request)."""


# ---------------------------------------------------------------------------
# Interfaces the tests substitute
# ---------------------------------------------------------------------------


class Transport(Protocol):
    """The thin seam between this gateway and the actual Gemini SDK."""

    def generate(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        images: Sequence[bytes] | None = None,
    ) -> str:
        """Return the raw model text (JSON when a schema was requested)."""

    def embed(self, *, api_key: str, model: str, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class UsageStore(Protocol):
    """Persistence for per-key counters."""

    def load(self) -> dict[int, dict[str, Any]]: ...

    def save(self, key_index: int, record: dict[str, Any]) -> None: ...


class InMemoryUsageStore:
    """Default store — used by tests and by mock mode."""

    def __init__(self) -> None:
        self._data: dict[int, dict[str, Any]] = {}

    def load(self) -> dict[int, dict[str, Any]]:
        return {k: dict(v) for k, v in self._data.items()}

    def save(self, key_index: int, record: dict[str, Any]) -> None:
        self._data[key_index] = dict(record)


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------


def mask_key(key: str) -> str:
    """Return a log-safe representation of an API key."""
    return f"...{key[-4:]}" if len(key) >= 4 else "...****"


@dataclass
class KeyStatus:
    index: int
    last4: str
    requests: int = 0
    failures: int = 0
    rate_limit_hits: int = 0
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_used_at: float | None = None

    def is_available(self, now: float) -> bool:
        return self.cooldown_until <= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "last4": self.last4,
            "requests": self.requests,
            "failures": self.failures,
            "rate_limit_hits": self.rate_limit_hits,
            "cooldown_until": self.cooldown_until,
            "consecutive_failures": self.consecutive_failures,
            "last_used_at": self.last_used_at,
        }


class KeyPool:
    """Round-robin pool with per-key exponential cooldown."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        base_cooldown: float = 30.0,
        max_cooldown: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
        store: UsageStore | None = None,
    ) -> None:
        if not keys:
            raise ValueError("GEMINI_API_KEYS is empty: at least one key is required")
        self._keys = list(keys)
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self._clock = clock
        self._store = store or InMemoryUsageStore()
        self._cursor = 0
        self._status: dict[int, KeyStatus] = {}
        persisted = self._store.load()
        for i, key in enumerate(self._keys):
            rec = persisted.get(i, {})
            self._status[i] = KeyStatus(
                index=i,
                last4=mask_key(key),
                requests=int(rec.get("requests", 0)),
                failures=int(rec.get("failures", 0)),
                rate_limit_hits=int(rec.get("rate_limit_hits", 0)),
                cooldown_until=float(rec.get("cooldown_until", 0.0)),
                consecutive_failures=int(rec.get("consecutive_failures", 0)),
            )

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    def status(self) -> list[KeyStatus]:
        return [self._status[i] for i in range(len(self._keys))]

    def key_at(self, index: int) -> str:
        return self._keys[index]

    def time_until_available(self) -> float:
        now = self._clock()
        waits = [max(0.0, s.cooldown_until - now) for s in self._status.values()]
        return min(waits) if waits else 0.0

    # -- selection ----------------------------------------------------------
    def acquire(self) -> tuple[int, str]:
        """Return the next available ``(index, key)`` in round-robin order."""
        now = self._clock()
        n = len(self._keys)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            if self._status[idx].is_available(now):
                self._cursor = (idx + 1) % n
                return idx, self._keys[idx]
        raise AllKeysExhausted(self.time_until_available())

    def acquire_excluding(self, tried: set[int]) -> tuple[int, str]:
        """Like :meth:`acquire` but skips keys already tried for this request."""
        now = self._clock()
        n = len(self._keys)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            if idx in tried:
                continue
            if self._status[idx].is_available(now):
                self._cursor = (idx + 1) % n
                return idx, self._keys[idx]
        raise AllKeysExhausted(self.time_until_available())

    # -- accounting ---------------------------------------------------------
    def record_success(self, index: int) -> None:
        st = self._status[index]
        st.requests += 1
        st.consecutive_failures = 0
        st.cooldown_until = 0.0
        st.last_used_at = self._clock()
        self._persist(index)

    def record_failure(self, index: int, status_code: int | None) -> float:
        """Mark a failure, applying cooldown for retryable statuses.

        Returns the cooldown duration applied (0.0 when none).
        """
        st = self._status[index]
        st.requests += 1
        st.failures += 1
        st.last_used_at = self._clock()
        if status_code in RATE_LIMIT_STATUSES:
            st.rate_limit_hits += 1
        cooldown = 0.0
        if status_code is None or status_code in RETRYABLE_STATUSES:
            st.consecutive_failures += 1
            cooldown = min(
                self.base_cooldown * math.pow(2, st.consecutive_failures - 1),
                self.max_cooldown,
            )
            st.cooldown_until = self._clock() + cooldown
            logger.warning(
                "gemini key[%d] %s cooling down %.0fs after status %s",
                index,
                st.last4,
                cooldown,
                status_code,
            )
        self._persist(index)
        return cooldown

    def _persist(self, index: int) -> None:
        self._store.save(index, self._status[index].to_dict())


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    extraction: str = "gemini-2.5-flash"
    judge: str = "gemini-2.5-pro"
    embedding: str = "gemini-embedding-001"
    vision: str = "gemini-2.5-flash"
    embedding_dim: int = 768
    extra: dict[str, Any] = field(default_factory=dict)


class GeminiClient:
    """Key-rotating gateway in front of every Gemini call the app makes."""

    def __init__(
        self,
        *,
        transport: Transport,
        keys: Sequence[str],
        models: ModelConfig | None = None,
        base_cooldown: float = 30.0,
        max_cooldown: float = 900.0,
        max_attempts: int = 6,
        clock: Callable[[], float] = time.monotonic,
        usage_store: UsageStore | None = None,
    ) -> None:
        self.transport = transport
        self.models = models or ModelConfig()
        self.max_attempts = max_attempts
        self.pool = KeyPool(
            keys,
            base_cooldown=base_cooldown,
            max_cooldown=max_cooldown,
            clock=clock,
            store=usage_store,
        )

    # -- public API ---------------------------------------------------------
    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        raw = self._run(
            lambda key: self.transport.generate(
                api_key=key,
                model=model or self.models.extraction,
                prompt=prompt,
                response_schema=response_schema,
                temperature=temperature,
            )
        )
        return _parse_json(raw)

    def judge(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Judge calls always run at temperature 0 on the judge-tier model."""
        raw = self._run(
            lambda key: self.transport.generate(
                api_key=key,
                model=model or self.models.judge,
                prompt=prompt,
                response_schema=response_schema,
                temperature=0.0,
            )
        )
        return _parse_json(raw)

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        return self._run(
            lambda key: self.transport.embed(
                api_key=key, model=model or self.models.embedding, texts=list(texts)
            )
        )

    def vision_extract(
        self,
        prompt: str,
        images: Sequence[bytes],
        response_schema: dict[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any] | str:
        raw = self._run(
            lambda key: self.transport.generate(
                api_key=key,
                model=model or self.models.vision,
                prompt=prompt,
                response_schema=response_schema,
                temperature=0.0,
                images=list(images),
            )
        )
        return _parse_json(raw) if response_schema else raw

    def key_pool_status(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.pool.status()]

    # -- engine -------------------------------------------------------------
    def _run(self, call: Callable[[str], Any]) -> Any:
        """Execute ``call`` against healthy keys, rotating on retryable errors."""
        tried: set[int] = set()
        last_error: Exception | None = None
        attempts = min(self.max_attempts, max(len(self.pool), 1))
        for _ in range(attempts):
            try:
                index, key = self.pool.acquire_excluding(tried)
            except AllKeysExhausted:
                if last_error is not None:
                    raise AllKeysExhausted(self.pool.time_until_available()) from last_error
                raise
            tried.add(index)
            try:
                result = call(key)
            except NonRetryableError:
                self.pool.record_failure(index, status_code=400)
                raise
            except TransportError as exc:
                if exc.status not in RETRYABLE_STATUSES:
                    self.pool.record_failure(index, status_code=exc.status)
                    raise NonRetryableError(str(exc)) from exc
                self.pool.record_failure(index, status_code=exc.status)
                last_error = exc
                logger.warning(
                    "gemini key[%d] %s failed with %s; rotating",
                    index,
                    self.pool.status()[index].last4,
                    exc.status,
                )
                continue
            except Exception as exc:  # transport blew up without a status
                self.pool.record_failure(index, status_code=None)
                last_error = exc
                logger.warning(
                    "gemini key[%d] %s raised %s; rotating",
                    index,
                    self.pool.status()[index].last4,
                    type(exc).__name__,
                )
                continue
            self.pool.record_success(index)
            return result
        raise AllKeysExhausted(self.pool.time_until_available()) from last_error


def _parse_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise NonRetryableError(f"expected JSON text from Gemini, got {type(raw).__name__}")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedResponse(f"Gemini returned non-JSON output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MalformedResponse("Gemini returned JSON that is not an object")
    return parsed


class MalformedResponse(GeminiError):
    """The model returned something that is not the requested JSON object."""
