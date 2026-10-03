import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from permrag.db.models import (
    AuditLog,
    Department,
    Document,
    DocumentShare,
    PermissionCheckpoint,
    User,
    UserDepartment,
)
from permrag.db.session import checkpoint_scope
from permrag.exceptions import ConflictError, NotFoundError
from permrag.permissions.client import (
    REL_MANAGER,
    REL_MEMBER,
    REL_OWNER_DEPARTMENT,
    REL_SHARED_DEPARTMENT,
    REL_VIEWER,
    TYPE_DEPARTMENT,
    TYPE_DOCUMENT,
    TYPE_USER,
    RelationshipTuple,
    SpiceDBClient,
)

logger = logging.getLogger(__name__)

class PermissionService:
    """
        Owns every write to SpiceDB and the Postgres mirror that shadows it.

        Every write follows the same order:

        1. Validate (the subject and resource exist; the user is active).
        2. Stage the mirror and audit rows and flush them, so a constraint
           failure aborts the request before SpiceDB is touched.
        3. Write SpiceDB under the checkpoint lock (``_write_and_checkpoint``).

        The request's transaction commits the mirror and audit rows afterwards.
    """

    def __init__(self, spicedb: SpiceDBClient, session: AsyncSession) -> None:
        self._spicedb = spicedb
        self._session = session

    # -- ZedToken checkpoint ------------------------------------------------
    async def get_checkpoint_token(self) -> str | None:
        """Newest ZedToken observed. Reads are pinned to this."""
        result = await self._session.execute(
            select(PermissionCheckpoint).where(PermissionCheckpoint.id == 1)
        )
        checkpoint = result.scalar_one_or_none()
        return checkpoint.zed_token if checkpoint else None

    async def _write_and_checkpoint(self, apply: Callable[[], Awaitable[str]]) -> str:
        """
            Run a SpiceDB write and record its ZedToken, strictly in write order.

            ``apply`` performs the write (and any read it depends on) and returns
            the new token. It runs while the checkpoint row is locked, in a short
            transaction of its own:

            - Ordering: writers queue on the lock *before* touching SpiceDB, so
              tokens are stored in the order the writes happened. ZedTokens are
              opaque and cannot be compared, so ordering is the only way to
              guarantee the stored token never moves backwards.
            - Duration: the lock covers one SpiceDB round-trip, never the
              caller's whole request -- an ingestion keeps its transaction open
              while it embeds.
        """
        async with checkpoint_scope() as session:
            checkpoint = await self._lock_checkpoint(session)
            token = await apply()
            checkpoint.zed_token = token
        return token

    @staticmethod
    async def _lock_checkpoint(session: AsyncSession) -> PermissionCheckpoint:
        locked = select(PermissionCheckpoint).where(PermissionCheckpoint.id == 1).with_for_update()
        checkpoint = (await session.execute(locked)).scalar_one_or_none()
        if checkpoint is None:
            # First write on a fresh database. Concurrent first writers race on
            # the insert; ON CONFLICT lets the losers fall through to the lock.
            await session.execute(
                pg_insert(PermissionCheckpoint).values(id=1).on_conflict_do_nothing(index_elements=["id"])
            )
            checkpoint = (await session.execute(locked)).scalar_one()
        return checkpoint

    async def _write(
        self,
        creates: list[RelationshipTuple] | None = None,
        deletes: list[RelationshipTuple] | None = None,
    ) -> str:
        return await self._write_and_checkpoint(
            lambda: self._spicedb.write_relationships(creates=creates, deletes=deletes)
        )

    # -- validation ---------------------------------------------------------

    async def _require_active_user(self, user_id: uuid.UUID) -> None:
        user = await self._session.get(User, user_id)
        if user is None:
            raise NotFoundError(f"User {user_id} not found")
        if not user.is_active:
            # Deactivation strips every edge; granting again would quietly
            # undo that.
            raise ConflictError(f"User {user_id} is deactivated; access cannot be granted")

    async def _require_department(self, department_id: uuid.UUID) -> None:
        if await self._session.get(Department, department_id) is None:
            raise NotFoundError(f"Department {department_id} not found")

    async def _require_live_document(self, document_id: uuid.UUID) -> None:
        document = await self._session.get(Document, document_id)
        if document is None or document.status == "deleted":
            raise NotFoundError(f"Document {document_id} not found")

    # -- audit --------------------------------------------------------------

    async def _audit(
        self,
        actor_id: uuid.UUID | None,
        action: str,
        resource_type: str,
        resource_id: str,
        detail: dict | None = None,
    ) -> None:
        self._session.add(
            AuditLog(
                created_at=datetime.now(UTC),
                actor_user_id=actor_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail or {},
            )
        )
        await self._session.flush()

    # -- department membership ---------------------------------------------

    @staticmethod
    def _membership(department_id: uuid.UUID, relation: str, user_id: uuid.UUID) -> RelationshipTuple:
        return RelationshipTuple(
            resource_type=TYPE_DEPARTMENT,
            resource_id=str(department_id),
            relation=relation,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )

    async def add_user_to_department(
        self,
        user_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        is_manager: bool = False,
    ) -> str:
        """Grant membership, or change its role. Exactly one relation remains."""
        await self._require_active_user(user_id)
        await self._require_department(department_id)

        # Atomic upsert: two concurrent grants for the same pair must not both
        # try to insert (unique constraint) and fail one request with a 500.
        await self._session.execute(
            pg_insert(UserDepartment)
            .values(user_id=user_id, department_id=department_id, is_manager=is_manager, granted_by_id=actor_id)
            .on_conflict_do_update(
                index_elements=["user_id", "department_id"],
                set_={"is_manager": is_manager, "updated_at": func.now()},
            )
        )

        await self._audit(
            actor_id,
            "department.member.grant",
            "department",
            str(department_id),
            {"user_id": str(user_id), "is_manager": is_manager},
        )

        # One atomic write: grant the target relation and drop the other, so a
        # promotion or demotion never leaves both behind.
        relation, stale = (REL_MANAGER, REL_MEMBER) if is_manager else (REL_MEMBER, REL_MANAGER)
        return await self._write(
            creates=[self._membership(department_id, relation, user_id)],
            deletes=[self._membership(department_id, stale, user_id)],
        )

    async def remove_user_from_department(
        self,
        user_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Revoke department membership. Both relations are removed."""
        await self._session.execute(
            delete(UserDepartment).where(
                UserDepartment.user_id == user_id,
                UserDepartment.department_id == department_id,
            )
        )
        await self._audit(
            actor_id,
            "department.member.revoke",
            "department",
            str(department_id),
            {"user_id": str(user_id)},
        )
        return await self._write(
            deletes=[self._membership(department_id, relation, user_id) for relation in (REL_MEMBER, REL_MANAGER)]
        )

    # -- document ownership -------------------------------------------------

    async def set_document_owner(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Bind a document to its single owning department.

        Called at the end of a successful ingest. Chunks are never searchable
        before this edge exists: a document is only in a permitted set once it
        has an edge and a committed row. Any previous owner's edge is removed
        in the same atomic write.
        """
        await self._audit(
            actor_id,
            "document.owner.set",
            "document",
            str(document_id),
            {"department_id": str(department_id)},
        )

        owner = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_OWNER_DEPARTMENT,
            subject_type=TYPE_DEPARTMENT,
            subject_id=str(department_id),
        )

        async def replace_owner() -> str:
            # Read inside the lock: two concurrent owner changes must not both
            # see the same old owner and each leave their own edge behind.
            current = await self._spicedb.read_relationships(
                resource_type=TYPE_DOCUMENT,
                resource_id=str(document_id),
                relation=REL_OWNER_DEPARTMENT,
            )
            stale = [edge for edge in current if edge.subject_id != str(department_id)]
            if stale:
                logger.info(
                    "document owner changed",
                    extra={"document_id": str(document_id), "removed_owners": [e.subject_id for e in stale]},
                )
            return await self._spicedb.write_relationships(creates=[owner], deletes=stale)

        return await self._write_and_checkpoint(replace_owner)

    # -- cross-department sharing ------------------------------------------

    @staticmethod
    def _document_edge(
        document_id: uuid.UUID, relation: str, subject_type: str, subject_id: uuid.UUID
    ) -> RelationshipTuple:
        return RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=relation,
            subject_type=subject_type,
            subject_id=str(subject_id),
        )

    async def share_document_with_department(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        reason: str | None = None,
    ) -> str:
        await self._require_live_document(document_id)
        await self._require_department(department_id)

        await self._session.execute(
            pg_insert(DocumentShare)
            .values(document_id=document_id, department_id=department_id, granted_by_id=actor_id, reason=reason)
            .on_conflict_do_nothing(index_elements=["document_id", "department_id"])
        )

        await self._audit(
            actor_id,
            "document.share.department.grant",
            "document",
            str(document_id),
            {"department_id": str(department_id), "reason": reason},
        )
        return await self._write(
            creates=[self._document_edge(document_id, REL_SHARED_DEPARTMENT, TYPE_DEPARTMENT, department_id)]
        )

    async def unshare_document_with_department(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        await self._session.execute(
            delete(DocumentShare).where(
                DocumentShare.document_id == document_id,
                DocumentShare.department_id == department_id,
            )
        )
        await self._audit(
            actor_id,
            "document.share.department.revoke",
            "document",
            str(document_id),
            {"department_id": str(department_id)},
        )
        return await self._write(
            deletes=[self._document_edge(document_id, REL_SHARED_DEPARTMENT, TYPE_DEPARTMENT, department_id)]
        )

    # -- person-level overrides --------------------------------------------

    async def grant_document_to_user(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        reason: str | None = None,
    ) -> str:
        """Give one person access without moving them between departments."""
        await self._require_live_document(document_id)
        await self._require_active_user(user_id)

        await self._session.execute(
            pg_insert(DocumentShare)
            .values(document_id=document_id, user_id=user_id, granted_by_id=actor_id, reason=reason)
            .on_conflict_do_nothing(index_elements=["document_id", "user_id"])
        )

        await self._audit(
            actor_id,
            "document.share.user.grant",
            "document",
            str(document_id),
            {"user_id": str(user_id), "reason": reason},
        )
        return await self._write(creates=[self._document_edge(document_id, REL_VIEWER, TYPE_USER, user_id)])

    async def revoke_document_from_user(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        await self._session.execute(
            delete(DocumentShare).where(
                DocumentShare.document_id == document_id,
                DocumentShare.user_id == user_id,
            )
        )
        await self._audit(
            actor_id,
            "document.share.user.revoke",
            "document",
            str(document_id),
            {"user_id": str(user_id)},
        )
        return await self._write(deletes=[self._document_edge(document_id, REL_VIEWER, TYPE_USER, user_id)])

    # -- teardown -----------------------------------------------------------

    async def purge_document_relationships(
        self,
        document_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Remove every edge touching a document.

        Called before the document's chunks are deleted from the vector store,
        so access disappears ahead of the content rather than after it.
        """
        await self._audit(actor_id, "document.relationships.purge", "document", str(document_id))
        return await self._write_and_checkpoint(
            lambda: self._spicedb.delete_by_filter(resource_type=TYPE_DOCUMENT, resource_id=str(document_id))
        )

    async def purge_user_relationships(
        self,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Strip a deactivated user from every department and override."""
        await self._session.execute(delete(UserDepartment).where(UserDepartment.user_id == user_id))
        await self._session.execute(delete(DocumentShare).where(DocumentShare.user_id == user_id))
        await self._audit(actor_id, "user.relationships.purge", "user", str(user_id))

        async def purge() -> str:
            await self._spicedb.delete_by_filter(
                resource_type=TYPE_DEPARTMENT,
                subject_type=TYPE_USER,
                subject_id=str(user_id),
            )
            # The second delete is the later revision, so its token covers both.
            return await self._spicedb.delete_by_filter(
                resource_type=TYPE_DOCUMENT,
                relation=REL_VIEWER,
                subject_type=TYPE_USER,
                subject_id=str(user_id),
            )

        return await self._write_and_checkpoint(purge)

    # -- reads --------------------------------------------------------------

    async def end_transaction(self) -> None:
        """
            End the current transaction and return its pooled connection.

            For read paths that go on to slow, database-free work (embedding,
            search, the LLM call): holding the connection meanwhile would cap
            concurrent requests at the pool size.
        """
        await self._session.commit()

    async def list_viewable_document_ids(
        self, user_id: uuid.UUID, limit: int
    ) -> tuple[list[str], str | None]:
        """Every document this user may view, pinned to the newest write."""
        token = await self.get_checkpoint_token()
        permitted_ids, observed_token = await self._spicedb.lookup_viewable_documents(
            user_id=str(user_id), zed_token=token, limit=limit
        )

        if not permitted_ids:
            return permitted_ids, observed_token

        as_uuid = [uuid.UUID(value) for value in permitted_ids]
        result = await self._session.execute(
            select(Document.id).where(Document.id.in_(as_uuid), Document.status != "deleted")
        )
        live_ids = {str(row[0]) for row in result.all()}

        filtered = [doc_id for doc_id in permitted_ids if doc_id in live_ids]
        dropped = len(permitted_ids) - len(filtered)
        if dropped:
            logger.warning(
                "dropped permitted document ids with no live row in postgres",
                extra={"user_id": str(user_id), "dropped": dropped},
            )

        return filtered, observed_token

    async def user_can_view(self, user_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        token = await self.get_checkpoint_token()
        return await self._spicedb.check_permission(
            user_id=str(user_id), document_id=str(document_id), zed_token=token
        )

    async def list_document_access(self, document_id: uuid.UUID) -> list[RelationshipTuple]:
        """Raw edges on a document, for the admin "who has access" panel."""
        token = await self.get_checkpoint_token()
        return await self._spicedb.read_relationships(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            zed_token=token,
        )
