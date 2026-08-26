"""Verbatim-evidence verification.

Every important field the model returns must be accompanied by a quote it claims
came from the source document.  We check that claim: the quote has to appear in
the source text with a fuzzy ratio of at least ``threshold`` (default 0.8) after
whitespace normalisation.  A quote that cannot be found forces that field's
confidence to 0.0 — the model does not get to assert facts we cannot trace.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_WS_RE = re.compile(r"\s+")
# Arabic diacritics and the tatweel elongation mark carry no lexical meaning.
_ARABIC_NOISE = re.compile(r"[ـً-ٰٟ]")


def normalise(text: str) -> str:
    """Whitespace/diacritic-normalised, case-folded form used for matching."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _ARABIC_NOISE.sub("", out)
    out = _WS_RE.sub(" ", out)
    return out.strip().casefold()


@dataclass
class EvidenceCheck:
    quote: str | None
    verified: bool
    ratio: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "quote": self.quote,
            "verified": self.verified,
            "ratio": round(self.ratio, 4),
            "reason": self.reason,
        }


def verify_quote(quote: str | None, source_text: str, *, threshold: float = 0.8) -> EvidenceCheck:
    """Check that ``quote`` really appears in ``source_text``."""
    if quote is None or not quote.strip():
        return EvidenceCheck(quote, False, 0.0, "missing_quote")
    if not source_text or not source_text.strip():
        return EvidenceCheck(quote, False, 0.0, "no_source_text")

    nq = normalise(quote)
    ns = normalise(source_text)
    if not nq:
        return EvidenceCheck(quote, False, 0.0, "empty_after_normalisation")
    if nq in ns:
        return EvidenceCheck(quote, True, 1.0, "exact")

    ratio = _best_window_ratio(nq, ns)
    if ratio >= threshold:
        return EvidenceCheck(quote, True, ratio, "fuzzy")
    return EvidenceCheck(quote, False, ratio, "not_found")


def _best_window_ratio(needle: str, haystack: str) -> float:
    """Best similarity of ``needle`` against any same-length window of ``haystack``.

    A full pairwise scan is quadratic, so we anchor on the longest shared block
    that :class:`SequenceMatcher` finds and compare only the windows around it.
    """
    if len(needle) > len(haystack):
        return SequenceMatcher(None, needle, haystack).ratio()

    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    block = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    if block.size == 0:
        return 0.0

    best = 0.0
    span = len(needle)
    centre = block.b - block.a
    for start in {max(0, centre), max(0, block.b - span), block.b, 0}:
        window = haystack[start : start + span]
        if not window:
            continue
        best = max(best, SequenceMatcher(None, needle, window, autojunk=False).ratio())
    return best


def verification_rate(checks: list[EvidenceCheck]) -> float:
    """Share of supplied quotes that verified; 0.0 when nothing was supplied."""
    if not checks:
        return 0.0
    return sum(1.0 for c in checks if c.verified) / len(checks)


def apply_to_confidence(confidence: float, check: EvidenceCheck) -> float:
    """An unverifiable quote zeroes the field's confidence."""
    return confidence if check.verified else 0.0
