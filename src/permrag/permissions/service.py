import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from permrag.db.models import AuditLog, DocumentShare, PermissionCheckpoint, UserDepartment
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
    """Owns every write to SpiceDB and the Postgres mirror that shadows it."""

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

    async def _store_checkpoint_token(self, token: str) -> None:
        result = await self._session.execute(
            select(PermissionCheckpoint).where(PermissionCheckpoint.id == 1)
        )
        checkpoint = result.scalar_one_or_none()
        if checkpoint is None:
            checkpoint = PermissionCheckpoint(id=1, zed_token=token)
            self._session.add(checkpoint)
        else:
            checkpoint.zed_token = token
        await self._session.flush()

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

    async def add_user_to_department(
        self,
        user_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        is_manager: bool = False,
    ) -> str:
        """Grant department membership. SpiceDB first, then the mirror."""
        relation = REL_MANAGER if is_manager else REL_MEMBER
        tup = RelationshipTuple(
            resource_type=TYPE_DEPARTMENT,
            resource_id=str(department_id),
            relation=relation,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )

        token = await self._spicedb.write_relationships(creates=[tup])
        await self._store_checkpoint_token(token)

        existing = await self._session.execute(
            select(UserDepartment).where(
                UserDepartment.user_id == user_id,
                UserDepartment.department_id == department_id,
            )
        )
        row = existing.scalar_one_or_none()
        if row is None:
            self._session.add(
                UserDepartment(
                    user_id=user_id,
                    department_id=department_id,
                    is_manager=is_manager,
                    granted_by_id=actor_id,
                )
            )
        else:
            row.is_manager = is_manager

        await self._audit(
            actor_id,
            "department.member.grant",
            "department",
            str(department_id),
            {"user_id": str(user_id), "is_manager": is_manager},
        )
        await self._session.flush()
        return token

    async def remove_user_from_department(
        self,
        user_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Revoke department membership. Both relations are removed."""
        deletes = [
            RelationshipTuple(
                resource_type=TYPE_DEPARTMENT,
                resource_id=str(department_id),
                relation=relation,
                subject_type=TYPE_USER,
                subject_id=str(user_id),
            )
            for relation in (REL_MEMBER, REL_MANAGER)
        ]

        token = await self._spicedb.write_relationships(deletes=deletes)
        await self._store_checkpoint_token(token)

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
        await self._session.flush()
        return token

    # -- document ownership -------------------------------------------------

    async def set_document_owner(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Bind a document to its owning department.

        Called during ingestion *before* any chunk is written to the vector
        store, so a chunk is never searchable before its ownership edge exists.
        """
        tup = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_OWNER_DEPARTMENT,
            subject_type=TYPE_DEPARTMENT,
            subject_id=str(department_id),
        )
        token = await self._spicedb.write_relationships(creates=[tup])
        await self._store_checkpoint_token(token)
        await self._audit(
            actor_id,
            "document.owner.set",
            "document",
            str(document_id),
            {"department_id": str(department_id)},
        )
        return token

    # -- cross-department sharing ------------------------------------------

    async def share_document_with_department(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        reason: str | None = None,
    ) -> str:
        tup = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_SHARED_DEPARTMENT,
            subject_type=TYPE_DEPARTMENT,
            subject_id=str(department_id),
        )
        token = await self._spicedb.write_relationships(creates=[tup])
        await self._store_checkpoint_token(token)

        existing = await self._session.execute(
            select(DocumentShare).where(
                DocumentShare.document_id == document_id,
                DocumentShare.department_id == department_id,
            )
        )
        if existing.scalar_one_or_none() is None:
            self._session.add(
                DocumentShare(
                    document_id=document_id,
                    department_id=department_id,
                    granted_by_id=actor_id,
                    reason=reason,
                )
            )

        await self._audit(
            actor_id,
            "document.share.department.grant",
            "document",
            str(document_id),
            {"department_id": str(department_id), "reason": reason},
        )
        await self._session.flush()
        return token

    async def unshare_document_with_department(
        self,
        document_id: uuid.UUID,
        department_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        tup = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_SHARED_DEPARTMENT,
            subject_type=TYPE_DEPARTMENT,
            subject_id=str(department_id),
        )
        token = await self._spicedb.write_relationships(deletes=[tup])
        await self._store_checkpoint_token(token)

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
        await self._session.flush()
        return token

    # -- person-level overrides --------------------------------------------

    async def grant_document_to_user(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
        reason: str | None = None,
    ) -> str:
        """Give one person access without moving them between departments."""
        tup = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_VIEWER,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )
        token = await self._spicedb.write_relationships(creates=[tup])
        await self._store_checkpoint_token(token)

        existing = await self._session.execute(
            select(DocumentShare).where(
                DocumentShare.document_id == document_id,
                DocumentShare.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none() is None:
            self._session.add(
                DocumentShare(
                    document_id=document_id,
                    user_id=user_id,
                    granted_by_id=actor_id,
                    reason=reason,
                )
            )

        await self._audit(
            actor_id,
            "document.share.user.grant",
            "document",
            str(document_id),
            {"user_id": str(user_id), "reason": reason},
        )
        await self._session.flush()
        return token

    async def revoke_document_from_user(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        tup = RelationshipTuple(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
            relation=REL_VIEWER,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )
        token = await self._spicedb.write_relationships(deletes=[tup])
        await self._store_checkpoint_token(token)

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
        await self._session.flush()
        return token

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
        token = await self._spicedb.delete_by_filter(
            resource_type=TYPE_DOCUMENT,
            resource_id=str(document_id),
        )
        await self._store_checkpoint_token(token)
        await self._audit(actor_id, "document.relationships.purge", "document", str(document_id))
        return token

    async def purge_user_relationships(
        self,
        user_id: uuid.UUID,
        actor_id: uuid.UUID | None,
    ) -> str:
        """Strip a deactivated user from every department and override."""
        token = await self._spicedb.delete_by_filter(
            resource_type=TYPE_DEPARTMENT,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )
        token = await self._spicedb.delete_by_filter(
            resource_type=TYPE_DOCUMENT,
            relation=REL_VIEWER,
            subject_type=TYPE_USER,
            subject_id=str(user_id),
        )
        await self._store_checkpoint_token(token)

        await self._session.execute(delete(UserDepartment).where(UserDepartment.user_id == user_id))
        await self._session.execute(delete(DocumentShare).where(DocumentShare.user_id == user_id))
        await self._audit(actor_id, "user.relationships.purge", "user", str(user_id))
        await self._session.flush()
        return token

    # -- reads --------------------------------------------------------------

    async def list_viewable_document_ids(
        self, user_id: uuid.UUID, limit: int
    ) -> tuple[list[str], str | None]:
        """Every document this user may view, pinned to the newest write."""
        token = await self.get_checkpoint_token()
        return await self._spicedb.lookup_viewable_documents(
            user_id=str(user_id), zed_token=token, limit=limit
        )

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
        