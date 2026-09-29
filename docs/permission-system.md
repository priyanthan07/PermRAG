# Permission system: how it works, where it can break, how it scales

Status as of 2026-09-29. Everything below is either read from the code or
measured; measurement setup is stated with each number.

## 1. Model

SpiceDB is the only authority for access. Postgres holds a **mirror** for the
admin panel, an **audit log**, and the **checkpoint** (newest ZedToken).

```
permission view = viewer + owner_department->membership + shared_department->membership
membership      = member + manager        (department)
administer      = owner_department->manage (defined, not yet enforced anywhere)
```

`view` is a union of paths. Access disappears only when **every** path is gone
(verified on 2026-09-29, see section 3).

## 2. Where each operation writes

Order for every write (`PermissionService`):
1. validate (subject/resource exist, user active),
2. stage + flush mirror and audit rows (constraint errors abort here, before SpiceDB),
3. SpiceDB write under the checkpoint lock, token stored in a short separate transaction,
4. the request transaction commits mirror + audit.

| Operation | SpiceDB | Postgres mirror | audit | checkpoint |
|---|---|---|---|---|
| Add / change membership | TOUCH target relation + DELETE the other, one atomic write | `user_department` upsert | `department.member.grant` | yes |
| Remove membership | DELETE member + manager | row deleted | `department.member.revoke` | yes |
| Set owner (ingest) | read current owners, TOUCH new + DELETE others, one atomic write under the lock | `document.owner_department_id` | `document.owner.set` | yes |
| Share / unshare department | `shared_department` | `document_share` | `document.share.department.*` | yes |
| Grant / revoke user | `viewer` | `document_share` | `document.share.user.*` | yes |
| Delete document | delete by filter (all edges) | status `deleted`; share + page rows deleted | `document.relationships.purge` | yes |
| Deactivate user | delete by filter (department edges, viewer edges) | membership + share rows deleted | `user.relationships.purge` | yes |
| Read (chat, list) | LookupResources `view` pinned `at_least_as_fresh` to the checkpoint | live-document filter | -- | read |

## 3. Break points found, and their status

Verified on 2026-09-29 with a temporary integration suite against real SpiceDB
(in-memory and Postgres-backed) and a separate Postgres test database. Each
fixed item P1-P9, P14 and P15 had a test that failed on the original code and
passed after the fix (the UI checkbox in P3 was not covered). An end-to-end
check through the HTTP API (22 checks) confirmed that only permitted passages
reach the LLM prompt. That suite has since been removed from the repository;
these results describe the code as of that date. P10 is a script.
The live drift from P1 was repaired on 2026-09-29 through the fixed service
(audited, actor = the admin account); the consistency check reports OK.

| # | Break point | Status |
|---|---|---|
| P1 | Promote/demote left the old relation; a demoted manager kept `manager` in SpiceDB. **Present in live data** (3 pairs) | Fixed: one atomic TOUCH+DELETE |
| P2 | Owner change left the old owner edge; old department kept `view` | Fixed: replace owners atomically under the lock |
| P3 | Same `external_id` from another department (UI derives it from the filename) silently took the document over | Fixed: 409 unless `transfer_ownership=true`; UI has an explicit checkbox |
| P4 | Delete left `document_share` / `document_page` rows; re-ingest then skipped pages | Fixed |
| P5 | Share with a non-existent department wrote the SpiceDB edge, then failed (500) | Fixed: 404 before any write |
| P6 | Deactivated users could be granted access again | Fixed: 409 |
| P7 | SpiceDB written before Postgres: a Postgres failure left a grant with no audit or mirror row | Fixed: Postgres staged and flushed first |
| P8 | Checkpoint token could move backwards under concurrent writes (reads then pinned to a stale revision); concurrent first writes also collided inserting the checkpoint row | Fixed: lock taken before the SpiceDB write; insert-if-missing with ON CONFLICT |
| P9 | Checkpoint row lock held until the request committed; an ingestion blocked every grant/revoke while embedding | Fixed: lock held for one SpiceDB round-trip on a dedicated pool |
| P10 | No drift detection | Fixed: `scripts/check_permission_consistency.py` (read-only, exit 1 on drift) |
| P11 | Suspected: delete-by-filter past 1000 edges | **Not a bug.** 1500-edge deletes complete on both in-memory and Postgres-backed SpiceDB; the 1000 limit only caps an explicitly requested page size |
| P12 | Suspected: reading past 1000 relationships | **Not a bug.** Same result; 1501-edge read complete |
| P13 | Users see at most `MAX_PERMITTED_DOCUMENTS` (5000) documents; beyond that, silently invisible | Documented; flagged in logs and `RetrievalResult.truncated`. Needs the redesign (section 6) |
| P14 | LookupResources yields a document once **per path** (owner, share, direct grant). **Present in live data**: one user got 14 ids for 8 documents. Inflated `permitted_document_count` (chat UI, `query_log`, Langfuse) and duplicates counted toward the cap | Fixed: de-duplicated in the client; only distinct documents count toward the limit |
| P15 | Paged lookup shrank the last page to fit the limit; SpiceDB rejects a cursor follow-up with different parameters (`INVALID_ARGUMENT`), so any cap that is not a multiple of 1000 failed every chat for users past the last full page. Latent at the default 5000 | Fixed: constant page size for the whole lookup |

### Residual risks (known, not fixed)
- **Commit failure after the SpiceDB write** (e.g. Postgres dies between steps
  3 and 4): SpiceDB has the change, the mirror/audit do not. Narrow window;
  the consistency check reports it. Closing it fully needs an outbox.
- **Vector chunk deletion failure on document delete** is logged, not raised:
  access is already gone and the row is `deleted`, so the chunks are
  unreachable, but they occupy space until cleaned up.
- After an ownership transfer, Qdrant payloads of *unchanged* pages keep the
  old `department_id`. Retrieval does not filter on it, so this is cosmetic.
- Model limits: no way to exclude one member of the owning department from one
  document; no expiring grants; no nested departments; admins are global and
  `administer`/`manage` are not enforced; no endpoint to reactivate a user.

## 4. Verified SpiceDB limits (v1.35.3 defaults, `spicedb serve --help`)

| Flag | Default |
|---|---|
| `--max-lookup-resources-limit` | 1000 (the app pages with a cursor) |
| `--max-read-relationships-limit` | 1000 |
| `--max-delete-relationships-limit` | 1000 |
| `--write-relationships-max-updates-per-call` | 1000 |
| `--datastore-revision-quantization-interval` | 5s (why reads must be pinned to the checkpoint) |

## 5. Measurements

Setup (2026-09-29): one Windows machine, Docker; SpiceDB v1.35.3 on the
**Postgres** datastore in a separate test database; app Postgres in a separate
test database; Qdrant v1.13.1 with a temporary 40,000-point collection,
256-dim random vectors, keyword index on `document_id`. Medians of 5 runs.
Absolute numbers are machine-specific; the growth is the point. The benchmark
code has since been removed. Measured before the P14/P15 fixes; in the
benchmark every document had exactly one path and the cap was a multiple of
1000, so neither fix changes what was measured.

**Read path, per chat question** (before the LLM is called):

| Visible documents | LookupResources | checkpoint + Postgres live filter | Qdrant `MatchAny` | Total |
|---:|---:|---:|---:|---:|
| 1,000 | 75.5 ms | 15.7 ms | 16.8 ms | 108 ms |
| 5,000 | 467.7 ms | 31.5 ms | 19.8 ms | 519 ms |
| 20,000 | 2,015.6 ms | 3,019.9 ms | 48.9 ms | 5,084 ms |

- LookupResources grows roughly linearly with the number of visible documents.
- The Postgres `IN (…)` filter is cheap at 5,000 ids and jumps to ~3 s at 20,000
  (derived as the median of lookup+filter minus the median of lookup alone).
- Qdrant filtering is not the bottleneck at this size.
- With the default cap (5,000) a user with more visible documents already pays
  ~0.5 s per question **and** silently misses documents past the cap.

**Write path** (50 `add_user_to_department` calls):

| | Before fixes | After fixes |
|---|---:|---:|
| sequential | 72.3 grants/s | 32.8 grants/s |
| concurrent | 74.6 grants/s | 22.1 grants/s |

The fixes cost write throughput: every write now takes the checkpoint lock
before its SpiceDB round-trip and commits a second short transaction, plus two
validation reads. The "before" numbers came with the token-ordering bug (P8).
Not profiled further. For an organisation, permission changes are rare
compared with questions; bulk onboarding is where this matters (section 6).

## 6. Scaling beyond this: proposed redesign (not implemented)

The read path scales with *documents a user can see*. It should scale with
*principals a user holds* (a handful of departments + themselves).

1. **Principals on chunks.** Each Qdrant point carries
   `allowed = ["dept:<owner>", "dept:<shared>"…, "user:<viewer>"…]`, keyword-indexed.
2. **Query by principal.** `principals(user) = ["user:<id>"] +
   ["dept:<d>" for d in LookupResources(department, membership, user)]`,
   then `MatchAny(principals)` -- a few values regardless of document count.
3. **SpiceDB stays the authority.** Post-check the top-K candidates with
   `CheckBulkPermissions(view)` pinned to the checkpoint (available in the
   installed client, authzed 1.25.0). A stale payload can only cause a
   candidate to be *dropped* (fail closed), never a leak. Oversample K.
4. **Payload sync via an outbox.** Share / unshare / owner changes write an
   outbox row in the same transaction as the mirror; a worker applies
   `set_payload` by `document_id` filter. The SpiceDB write stays synchronous.
5. **Removes** the 5,000 cap, the per-question `IN (…)` over all visible ids,
   and LookupResources over documents.
6. **Bulk grant API.** One `WriteRelationships` carries up to 1,000 updates
   under one checkpoint lock, amortising the per-write cost measured above.

Trade-offs: every share change touches Qdrant payloads; recall depends on K
when payloads lag; the ingestion and sharing paths both change, and existing
chunks need a one-off payload backfill.

## 7. How to verify

```
make check-permissions        # read-only drift check against the real data
```

Prints ids only and exits 1 if SpiceDB and the Postgres mirror disagree.
