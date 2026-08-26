"""Small helpers shared between the app and the scripts in ``scripts/``."""

from __future__ import annotations

import json

from sqlalchemy import select

from .core.taxonomy import SEED_TAXONOMY, Taxonomy
from .db import get_session_factory
from .models import SkillTaxonomy


def seed_taxonomy_if_empty() -> int:
    """Seed ``skills_taxonomy`` from the code-level seed. Returns rows inserted."""
    factory = get_session_factory()
    with factory() as session:
        existing = session.scalar(select(SkillTaxonomy).limit(1))
        if existing is not None:
            return 0
        for entry in SEED_TAXONOMY:
            session.add(
                SkillTaxonomy(
                    canonical=entry.canonical,
                    aliases_json=json.dumps(entry.aliases, ensure_ascii=False),
                    category=entry.category,
                )
            )
        session.commit()
        return len(SEED_TAXONOMY)


def load_taxonomy() -> Taxonomy:
    """Build a taxonomy from the DB, falling back to the code-level seed."""
    factory = get_session_factory()
    with factory() as session:
        rows = session.scalars(select(SkillTaxonomy)).all()
    return Taxonomy.from_rows(rows) if rows else Taxonomy()
