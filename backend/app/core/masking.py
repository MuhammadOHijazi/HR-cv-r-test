"""Identity masking.

Everything that reaches an evaluator (the LLM judge, and the free-text blocks
handed to it) goes through here first.  Names, e-mail addresses, phone numbers,
photo references, age, gender and nationality signals are stripped so the
evaluator scores capability, not identity.
"""

from __future__ import annotations

import re
from typing import Any

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")
URL_RE = re.compile(r"https?://\S+|(?:www\.)\S+")
PHOTO_RE = re.compile(
    r"(?:photo|picture|image|headshot|صورة)\s*[:\-]?\s*\S*|\S+\.(?:jpg|jpeg|png|gif|bmp)",
    re.IGNORECASE,
)
AGE_RE = re.compile(
    r"\b(?:age|aged|date\s+of\s+birth|dob|born|العمر|تاريخ\s+الميلاد)\b\s*[:\-]?\s*[^\n,;]{0,24}",
    re.IGNORECASE,
)
GENDER_RE = re.compile(
    r"\b(?:gender|sex|male|female|mr\.?|mrs\.?|ms\.?|miss|الجنس|ذكر|أنثى)\b[^\n,;]{0,16}",
    re.IGNORECASE,
)
NATIONALITY_RE = re.compile(
    r"\b(?:nationality|citizenship|marital\s+status|religion|passport|"
    r"الجنسية|الحالة\s+الاجتماعية|الديانة)\b\s*[:\-]?\s*[^\n,;]{0,32}",
    re.IGNORECASE,
)

REDACTED = "[REDACTED]"


def mask_text(text: str, *, names: list[str] | None = None) -> str:
    """Remove identity signals from a free-text block."""
    if not text:
        return ""
    out = text
    for name in sorted(filter(None, names or []), key=len, reverse=True):
        for token in [name] + [p for p in name.split() if len(p) > 2]:
            out = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", REDACTED, out, flags=re.IGNORECASE)
    out = EMAIL_RE.sub(REDACTED, out)
    out = PHONE_RE.sub(REDACTED, out)
    out = URL_RE.sub(REDACTED, out)
    out = PHOTO_RE.sub(REDACTED, out)
    out = AGE_RE.sub(REDACTED, out)
    out = NATIONALITY_RE.sub(REDACTED, out)
    out = GENDER_RE.sub(REDACTED, out)
    return out


IDENTITY_FIELDS = ("full_name", "email", "phone", "photo", "age", "gender", "nationality", "date_of_birth")


def mask_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return an identity-masked copy of an extraction payload.

    Skills, work history and education survive; every identity field is dropped
    and every free-text field is scrubbed.
    """
    names = [record.get("full_name")] if record.get("full_name") else []
    masked: dict[str, Any] = {}
    for key, value in record.items():
        if key in IDENTITY_FIELDS:
            continue
        masked[key] = _mask_value(value, names)
    return masked


def _mask_value(value: Any, names: list[str]) -> Any:
    if isinstance(value, str):
        return mask_text(value, names=names)
    if isinstance(value, list):
        return [_mask_value(v, names) for v in value]
    if isinstance(value, dict):
        return {
            k: _mask_value(v, names)
            for k, v in value.items()
            if k not in IDENTITY_FIELDS
        }
    return value


def contains_identity(text: str) -> bool:
    """True when an obvious identity signal survived masking (test helper)."""
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or NATIONALITY_RE.search(text))
