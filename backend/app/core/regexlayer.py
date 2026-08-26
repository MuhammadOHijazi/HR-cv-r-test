"""Deterministic pre-extraction layer.

Anything a regular expression can find reliably — e-mail addresses, phone
numbers, URLs, date ranges, stated years of experience — is found here first.
The LLM never gets the chance to hallucinate these, and the results are handed
to it as verified hints.  Everything returned carries the exact source span so
it doubles as its own evidence.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;<>()]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+|00)?\d[\d\s().\-]{7,17}\d")
YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?|سنوات|سنة|عام|أعوام)"
    r"(?:\s*(?:of|من)?\s*(?:experience|خبرة))?",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "أغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
_PRESENT = {"present", "current", "now", "today", "ongoing", "الآن", "حتى الآن", "الحاضر", "حاليا", "حالياً"}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_PRESENT_ALT = "|".join(sorted(_PRESENT, key=len, reverse=True))
_DATE_TOKEN = rf"(?:(?:{_MONTH_ALT})[a-z]*\.?\s+)?\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}-\d{{1,2}}"
DATE_RANGE_RE = re.compile(
    rf"({_DATE_TOKEN})\s*(?:-|–|—|to|until|حتى|إلى)\s*({_DATE_TOKEN}|{_PRESENT_ALT})",
    re.IGNORECASE,
)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass
class DateRange:
    start: dt.date
    end: dt.date | None
    quote: str

    @property
    def months(self) -> int:
        end = self.end or dt.date.today()
        return max(0, (end.year - self.start.year) * 12 + (end.month - self.start.month))

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_date": self.start.isoformat(),
            "to_date": self.end.isoformat() if self.end else None,
            "quote": self.quote,
            "months": self.months,
        }


@dataclass
class RegexFindings:
    emails: list[str]
    phones: list[str]
    urls: list[str]
    date_ranges: list[DateRange]
    stated_years: float | None
    stated_years_quote: str | None

    def as_hints(self) -> dict[str, Any]:
        return {
            "emails": self.emails,
            "phones": self.phones,
            "urls": self.urls,
            "date_ranges": [d.as_dict() for d in self.date_ranges],
            "stated_years_experience": self.stated_years,
        }

    @property
    def computed_years(self) -> float | None:
        return computed_years_from_ranges(self.date_ranges)


def normalise_digits(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_ARABIC_DIGITS)


def normalise_phone(raw: str) -> str:
    """Digits only, with a leading ``+`` preserved for international numbers."""
    digits = re.sub(r"\D", "", normalise_digits(raw))
    if raw.strip().startswith("+") or digits.startswith("00"):
        digits = digits.lstrip("0")
        return f"+{digits}"
    return digits


def normalise_email(raw: str) -> str:
    return (raw or "").strip().lower()


def parse_date_token(token: str) -> dt.date | None:
    """Parse the loose date forms CVs actually use."""
    t = normalise_digits(token).strip().lower().rstrip(".")
    if not t:
        return None
    if t in _PRESENT:
        return None
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", t)
    if m:
        month = min(12, max(1, int(m.group(1))))
        return dt.date(int(m.group(2)), month, 1)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", t)
    if m:
        month = min(12, max(1, int(m.group(2))))
        return dt.date(int(m.group(1)), month, 1)
    m = re.fullmatch(r"([a-z؀-ۿ]+)\.?\s+(\d{4})", t)
    if m:
        month = _MONTHS.get(m.group(1)[:9]) or _MONTHS.get(m.group(1)[:3])
        if month:
            return dt.date(int(m.group(2)), month, 1)
        return dt.date(int(m.group(2)), 1, 1)
    m = re.fullmatch(r"(\d{4})", t)
    if m:
        return dt.date(int(m.group(1)), 1, 1)
    return None


def find_date_ranges(text: str) -> list[DateRange]:
    ranges: list[DateRange] = []
    for match in DATE_RANGE_RE.finditer(text or ""):
        start = parse_date_token(match.group(1))
        if start is None:
            continue
        end_token = match.group(2).strip().lower()
        end = None if end_token in _PRESENT else parse_date_token(match.group(2))
        if end is not None and end < start:
            continue
        ranges.append(DateRange(start, end, match.group(0).strip()))
    return ranges


def merge_intervals(ranges: list[DateRange]) -> list[tuple[dt.date, dt.date]]:
    """Collapse overlapping employment periods so they are not double counted."""
    today = dt.date.today()
    spans = sorted(((r.start, r.end or today) for r in ranges), key=lambda s: s[0])
    merged: list[tuple[dt.date, dt.date]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def computed_years_from_ranges(ranges: list[DateRange]) -> float | None:
    """Total non-overlapping employment length, in years."""
    if not ranges:
        return None
    months = 0
    for start, end in merge_intervals(ranges):
        months += max(0, (end.year - start.year) * 12 + (end.month - start.month))
    return round(months / 12.0, 2)


def find_stated_years(text: str) -> tuple[float | None, str | None]:
    """Largest explicitly stated years-of-experience figure, plus its quote."""
    best: float | None = None
    quote: str | None = None
    for match in YEARS_RE.finditer(normalise_digits(text or "")):
        value = float(match.group(1))
        if value > 60:  # not a duration; almost certainly a year or an amount
            continue
        if best is None or value > best:
            best = value
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            quote = text[line_start : line_end if line_end != -1 else len(text)].strip()
    return best, quote


def run(text: str) -> RegexFindings:
    """Run the whole deterministic layer over a CV's text."""
    source = text or ""
    emails: list[str] = []
    for raw in EMAIL_RE.findall(source):
        value = normalise_email(raw)
        if value not in emails:
            emails.append(value)
    urls: list[str] = []
    for raw in URL_RE.findall(source):
        if raw not in urls and "@" not in raw:
            urls.append(raw)
    phones: list[str] = []
    for raw in PHONE_RE.findall(normalise_digits(source)):
        value = normalise_phone(raw)
        if 7 <= len(value.lstrip("+")) <= 15 and value not in phones:
            phones.append(value)
    stated, stated_quote = find_stated_years(source)
    return RegexFindings(
        emails=emails,
        phones=phones,
        urls=urls,
        date_ranges=find_date_ranges(source),
        stated_years=stated,
        stated_years_quote=stated_quote,
    )


def identity_key(email: str | None, phone: str | None, fallback: str) -> str:
    """Stable candidate identity: normalised e-mail + normalised phone."""
    e = normalise_email(email or "")
    p = normalise_phone(phone or "") if phone else ""
    if e or p:
        return f"{e}|{p}"
    return f"anon|{fallback}"
