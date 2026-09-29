"""
    Read-only drift check: SpiceDB (the authority) against the Postgres mirror.

    Writes nothing. Prints ids only (never names or emails). Exits 1 if any
    drift is found, so it can run from cron or CI:

        uv run python scripts/check_permission_consistency.py

    What it checks:
      membership  department edges vs user_department (both directions, the
                  manager flag, and users holding member AND manager at once)
      ownership   every live document has exactly one owner edge, and it
                  matches document.owner_department_id
      shares      shared_department/viewer edges vs document_share
      lifecycle   edges on unknown or deleted documents, edges held by unknown
                  or deactivated users, leftover rows on deleted documents
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import func, select

# Allow `python scripts/check_permission_consistency.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from permrag.config import get_settings
from permrag.db.models import (
    Department,
    Document,
    DocumentPage,
    DocumentShare,
    User,
    UserDepartment,
)
from permrag.db.session import dispose_engine, get_session_factory
from permrag.logging_config import configure_logging
from permrag.permissions.client import (
    REL_MANAGER,
    REL_OWNER_DEPARTMENT,
    REL_SHARED_DEPARTMENT,
    REL_VIEWER,
    TYPE_DEPARTMENT,
    TYPE_DOCUMENT,
    get_spicedb_client,
)


async def collect() -> dict[str, list[str]]:
    spicedb = get_spicedb_client()
    # No zed_token: fully consistent reads, so the check never sees stale data.
    department_edges = await spicedb.read_relationships(resource_type=TYPE_DEPARTMENT)
    document_edges = await spicedb.read_relationships(resource_type=TYPE_DOCUMENT)

    async with get_session_factory()() as session:
        users = {str(u.id): u.is_active for u in (await session.execute(select(User))).scalars()}
        departments = {str(d.id) for d in (await session.execute(select(Department))).scalars()}
        mirror_members = {
            (str(m.user_id), str(m.department_id)): m.is_manager
            for m in (await session.execute(select(UserDepartment))).scalars()
        }
        documents = {
            str(d.id): (d.status, str(d.owner_department_id))
            for d in (await session.execute(select(Document))).scalars()
        }
        mirror_shares = {
            (
                str(s.document_id),
                REL_SHARED_DEPARTMENT if s.department_id else REL_VIEWER,
                str(s.department_id or s.user_id),
            )
            for s in (await session.execute(select(DocumentShare))).scalars()
        }
        page_counts = {
            str(doc_id): count
            for doc_id, count in (
                await session.execute(
                    select(DocumentPage.document_id, func.count()).group_by(DocumentPage.document_id)
                )
            ).all()
        }

    drift: dict[str, list[str]] = defaultdict(list)

    # -- membership ----------------------------------------------------------
    relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in department_edges:
        relations[(edge.subject_id, edge.resource_id)].add(edge.relation)

    for (user, dept), held in relations.items():
        pair = f"user={user} department={dept}"
        if (user, dept) not in mirror_members:
            drift["membership edge with no user_department row"].append(pair)
        elif (REL_MANAGER in held) != mirror_members[(user, dept)]:
            drift["manager flag differs from SpiceDB"].append(
                f"{pair} spicedb={sorted(held)} mirror.is_manager={mirror_members[(user, dept)]}"
            )
        if len(held) > 1:
            drift["user holds both member and manager"].append(pair)
        if user not in users:
            drift["membership edge for unknown user"].append(pair)
        elif not users[user]:
            drift["deactivated user still holds a membership edge"].append(pair)
        if dept not in departments:
            drift["membership edge on unknown department"].append(pair)

    for (user, dept) in mirror_members:
        if (user, dept) not in relations:
            drift["user_department row with no SpiceDB edge"].append(f"user={user} department={dept}")

    # -- ownership -----------------------------------------------------------
    owners: dict[str, set[str]] = defaultdict(set)
    for edge in document_edges:
        if edge.relation == REL_OWNER_DEPARTMENT:
            owners[edge.resource_id].add(edge.subject_id)

    for doc, (status, owner) in documents.items():
        if status == "deleted":
            continue
        found = owners.get(doc, set())
        if not found:
            drift["live document with no owner edge"].append(f"document={doc}")
        elif found != {owner}:
            drift["owner edges differ from document.owner_department_id"].append(
                f"document={doc} spicedb={sorted(found)} postgres={owner}"
            )

    # -- shares --------------------------------------------------------------
    spicedb_shares = {
        (e.resource_id, e.relation, e.subject_id)
        for e in document_edges
        if e.relation in (REL_SHARED_DEPARTMENT, REL_VIEWER)
    }
    for doc, relation, subject in sorted(spicedb_shares - mirror_shares):
        drift["share edge with no document_share row"].append(f"document={doc} {relation}={subject}")
    for doc, relation, subject in sorted(mirror_shares - spicedb_shares):
        drift["document_share row with no SpiceDB edge"].append(f"document={doc} {relation}={subject}")
    for doc, relation, subject in spicedb_shares:
        if relation == REL_VIEWER and subject in users and not users[subject]:
            drift["deactivated user still holds a viewer edge"].append(f"document={doc} user={subject}")

    # -- lifecycle -----------------------------------------------------------
    for doc in sorted({e.resource_id for e in document_edges}):
        if doc not in documents:
            drift["edges on a document with no Postgres row"].append(f"document={doc}")
        elif documents[doc][0] == "deleted":
            drift["deleted document still has edges"].append(f"document={doc}")
    for doc, (status, _) in documents.items():
        if status == "deleted" and page_counts.get(doc):
            drift["deleted document still has document_page rows"].append(f"document={doc}")
        if status == "deleted" and any(s[0] == doc for s in mirror_shares):
            drift["deleted document still has document_share rows"].append(f"document={doc}")

    print(
        f"checked: {len(users)} users, {len(departments)} departments, {len(documents)} documents, "
        f"{len(department_edges)} department edges, {len(document_edges)} document edges"
    )
    return drift


async def main() -> int:
    configure_logging(get_settings().log_level)
    try:
        drift = await collect()
    finally:
        await dispose_engine()

    if not drift:
        print("OK: SpiceDB and the Postgres mirror agree.")
        return 0

    total = sum(len(items) for items in drift.values())
    print(f"DRIFT: {total} problem(s) in {len(drift)} categories")
    for category, items in drift.items():
        print(f"\n[{category}] ({len(items)})")
        for item in items:
            print(f"  {item}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
