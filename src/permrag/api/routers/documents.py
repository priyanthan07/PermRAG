import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from permrag.api.deps import (
    AdminUser,
    CurrentUser,
    DbSession,
    PermissionServiceDep,
    get_store,
)
from permrag.api.schemas import (
    AccessEntry,
    DocumentAccessResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentResponse,
    MessageResponse,
    ShareResponse,
    ShareWithDepartmentRequest,
    ShareWithUserRequest,
)
from permrag.db.models import Department, Document, User
from permrag.exceptions import NotFoundError
from permrag.ingestion.pipeline import IngestionPipeline, PageInput
from permrag.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

StoreDep = Annotated[QdrantVectorStore, Depends(get_store)]


@router.post(
    "",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    payload: DocumentIngestRequest,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
    store: StoreDep,
) -> DocumentIngestResponse:
    """Ingest or re-ingest a document.

    Re-posting the same ``external_id`` performs an incremental update: pages
    whose content hash is unchanged are skipped without an embedding call.
    """
    department = await session.execute(
        select(Department).where(Department.id == payload.owner_department_id)
    )
    if department.scalar_one_or_none() is None:
        raise NotFoundError(f"Department {payload.owner_department_id} not found")

    pipeline = IngestionPipeline(session=session, vector_store=store, permissions=permissions)

    result = await pipeline.ingest(
        external_id=payload.external_id,
        title=payload.title,
        owner_department_id=payload.owner_department_id,
        pages=[PageInput(page_number=p.page_number, content=p.content) for p in payload.pages],
        actor_id=admin.id,
        source_uri=payload.source_uri,
    )

    return DocumentIngestResponse(
        document_id=result.document_id,
        pages_total=result.pages_total,
        pages_indexed=result.pages_indexed,
        pages_skipped=result.pages_skipped,
        pages_removed=result.pages_removed,
        chunks_written=result.chunks_written,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_visible_documents(
    session: DbSession,
    user: CurrentUser,
    permissions: PermissionServiceDep,
) -> list[Document]:
    """List documents this user may view.

    The id set comes from SpiceDB; Postgres is used only to hydrate titles and
    metadata for ids already cleared by the permission system.
    """
    permitted_ids, _ = await permissions.list_viewable_document_ids(user.id, limit=5000)
    if not permitted_ids:
        return []

    as_uuid = [uuid.UUID(value) for value in permitted_ids]
    result = await session.execute(
        select(Document)
        .where(Document.id.in_(as_uuid), Document.status != "deleted")
        .order_by(Document.title)
    )
    return list(result.scalars().all())


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
    permissions: PermissionServiceDep,
) -> Document:
    if not await permissions.user_can_view(user.id, document_id):
        # Same response as a genuinely missing document, so this endpoint
        # cannot be used to probe which document ids exist.
        raise NotFoundError(f"Document {document_id} not found")

    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise NotFoundError(f"Document {document_id} not found")
    return document


@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
    store: StoreDep,
) -> MessageResponse:
    pipeline = IngestionPipeline(session=session, vector_store=store, permissions=permissions)
    await pipeline.delete_document(document_id=document_id, actor_id=admin.id)
    return MessageResponse(message="Document deleted and all access revoked")


# --- sharing -----------------------------------------------------------------


@router.post("/{document_id}/shares/departments", response_model=ShareResponse)
async def share_with_department(
    document_id: uuid.UUID,
    payload: ShareWithDepartmentRequest,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> ShareResponse:
    await _assert_document_exists(session, document_id)
    token = await permissions.share_document_with_department(
        document_id=document_id,
        department_id=payload.department_id,
        actor_id=admin.id,
        reason=payload.reason,
    )
    return ShareResponse(
        document_id=document_id,
        zed_token=token,
        message="Department granted access",
    )


@router.delete("/{document_id}/shares/departments/{department_id}", response_model=ShareResponse)
async def unshare_with_department(
    document_id: uuid.UUID,
    department_id: uuid.UUID,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> ShareResponse:
    token = await permissions.unshare_document_with_department(
        document_id=document_id, department_id=department_id, actor_id=admin.id
    )
    return ShareResponse(
        document_id=document_id,
        zed_token=token,
        message="Department access revoked",
    )


@router.post("/{document_id}/shares/users", response_model=ShareResponse)
async def share_with_user(
    document_id: uuid.UUID,
    payload: ShareWithUserRequest,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> ShareResponse:
    """Person-level override: one user, one document, no department change."""
    await _assert_document_exists(session, document_id)

    target = await session.execute(select(User).where(User.id == payload.user_id))
    if target.scalar_one_or_none() is None:
        raise NotFoundError(f"User {payload.user_id} not found")

    token = await permissions.grant_document_to_user(
        document_id=document_id,
        user_id=payload.user_id,
        actor_id=admin.id,
        reason=payload.reason,
    )
    return ShareResponse(
        document_id=document_id,
        zed_token=token,
        message="User granted access",
    )


@router.delete("/{document_id}/shares/users/{user_id}", response_model=ShareResponse)
async def revoke_from_user(
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> ShareResponse:
    token = await permissions.revoke_document_from_user(
        document_id=document_id, user_id=user_id, actor_id=admin.id
    )
    return ShareResponse(
        document_id=document_id,
        zed_token=token,
        message="User access revoked",
    )


@router.get("/{document_id}/access", response_model=DocumentAccessResponse)
async def list_document_access(
    document_id: uuid.UUID,
    _: AdminUser,
    permissions: PermissionServiceDep,
) -> DocumentAccessResponse:
    """Who currently has access, read straight from the permission graph."""
    tuples = await permissions.list_document_access(document_id)
    return DocumentAccessResponse(
        document_id=document_id,
        entries=[
            AccessEntry(
                relation=t.relation,
                subject_type=t.subject_type,
                subject_id=t.subject_id,
            )
            for t in tuples
        ],
    )


async def _assert_document_exists(session, document_id: uuid.UUID) -> None:
    result = await session.execute(select(Document).where(Document.id == document_id))
    if result.scalar_one_or_none() is None:
        raise NotFoundError(f"Document {document_id} not found")
    