"""Audit-log helper.

Every human action and every machine decision that a person could later question
is recorded with before/after values.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AuditLog


def record(
    session: Session,
    entity_type: str,
    entity_id: int,
    action: str,
    *,
    actor: str = "system",
    before: Any = None,
    after: Any = None,
) -> AuditLog:
    entry = AuditLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor=actor,
        before_json=json.dumps(before, ensure_ascii=False, default=str),
        after_json=json.dumps(after, ensure_ascii=False, default=str),
    )
    session.add(entry)
    session.flush()
    return entry


def history(session: Session, entity_type: str, entity_id: int) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(AuditLog)
        .where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
        .order_by(AuditLog.id)
    ).all()
    return [
        {
            "id": r.id,
            "action": r.action,
            "actor": r.actor,
            "before": json.loads(r.before_json or "null"),
            "after": json.loads(r.after_json or "null"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
