"""Seed or extend the bilingual skills taxonomy in the database.

The ~100-skill seed lives in ``backend/app/core/taxonomy.py``; this script
mirrors it into the ``skills_taxonomy`` table so it can be extended at runtime
without a code change.

Run:
    python scripts/seed_taxonomy.py            # seed only if the table is empty
    python scripts/seed_taxonomy.py --force    # re-sync every seed entry
    python scripts/seed_taxonomy.py --list     # show what is in the database
    python scripts/seed_taxonomy.py --add "quantum computing:ml:qc,الحوسبة الكمية"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select  # noqa: E402

from backend.app.core.taxonomy import SEED_TAXONOMY  # noqa: E402
from backend.app.db import get_session_factory, init_db  # noqa: E402
from backend.app.models import SkillTaxonomy  # noqa: E402


def seed(force: bool = False) -> int:
    """Write the code-level seed into the database. Returns rows written."""
    init_db()
    written = 0
    with get_session_factory()() as session:
        existing = {row.canonical: row for row in session.scalars(select(SkillTaxonomy)).all()}
        if existing and not force:
            print(f"skills_taxonomy already holds {len(existing)} skills; use --force to re-sync")
            return 0
        for entry in SEED_TAXONOMY:
            row = existing.get(entry.canonical)
            if row is None:
                session.add(
                    SkillTaxonomy(
                        canonical=entry.canonical,
                        aliases_json=json.dumps(entry.aliases, ensure_ascii=False),
                        category=entry.category,
                    )
                )
            else:
                row.aliases_json = json.dumps(entry.aliases, ensure_ascii=False)
                row.category = entry.category
            written += 1
        session.commit()
    return written


def add(spec: str) -> None:
    """Add one skill from a ``canonical:category:alias,alias`` specification."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit('--add expects "canonical:category[:alias,alias]"')
    canonical, category = parts[0].strip(), parts[1].strip()
    aliases = [a.strip() for a in (parts[2] if len(parts) > 2 else "").split(",") if a.strip()]

    init_db()
    with get_session_factory()() as session:
        row = session.scalar(select(SkillTaxonomy).where(SkillTaxonomy.canonical == canonical))
        if row is None:
            row = SkillTaxonomy(canonical=canonical)
            session.add(row)
        row.category = category
        row.aliases_json = json.dumps(aliases, ensure_ascii=False)
        session.commit()
    print(f"stored {canonical} ({category}) with {len(aliases)} aliases")


def show() -> None:
    init_db()
    with get_session_factory()() as session:
        rows = session.scalars(select(SkillTaxonomy).order_by(SkillTaxonomy.category)).all()
    if not rows:
        print("skills_taxonomy is empty — run without arguments to seed it")
        return
    print(f"{len(rows)} skills in the database\n")
    for row in rows:
        aliases = json.loads(row.aliases_json or "[]")
        print(f"  [{row.category:>14}] {row.canonical:<26} {', '.join(aliases)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-sync every seed entry")
    parser.add_argument("--list", action="store_true", help="print the stored taxonomy")
    parser.add_argument("--add", metavar="SPEC", help='"canonical:category[:alias,alias]"')
    args = parser.parse_args()

    if args.list:
        show()
        return 0
    if args.add:
        add(args.add)
        return 0

    written = seed(force=args.force)
    if written:
        print(f"seeded {written} skills into skills_taxonomy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
