"""FastAPI dependencies: session, current user, and the service graph."""

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from permrag.db.models import User
from permrag.db.session import get_session_factory
from permrag.exceptions import AuthenticationError, PermissionDeniedError
from permrag.permissions.client import SpiceDBClient
from permrag.permissions.service import PermissionService
from permrag.rag.answerer import Answerer
from permrag.rag.retriever import PermissionAwareRetriever
from permrag.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

async def get_db_session() -> AsyncIterator[AsyncSession]:
    """One session per request, committed on success, rolled back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DbSession = Annotated[AsyncSession, Depends(get_db_session)]

def get_spicedb(request: Request) -> SpiceDBClient:
    """Client created once in the lifespan hook and stored on app state."""
    return request.app.state.spicedb


def get_store(request: Request) -> QdrantVectorStore:
    return request.app.state.vector_store


def get_permission_service(
    session: DbSession,
    spicedb: Annotated[SpiceDBClient, Depends(get_spicedb)],
) -> PermissionService:
    return PermissionService(spicedb=spicedb, session=session)

PermissionServiceDep = Annotated[PermissionService, Depends(get_permission_service)]

def get_retriever(
    store: Annotated[QdrantVectorStore, Depends(get_store)],
    permissions: PermissionServiceDep,
) -> PermissionAwareRetriever:
    return PermissionAwareRetriever(vector_store=store, permissions=permissions)


def get_answerer(
    retriever: Annotated[PermissionAwareRetriever, Depends(get_retriever)],
    session: DbSession,
) -> Answerer:
    return Answerer(retriever=retriever, session=session)

async def get_current_user(
    session: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> User:
    """
        Resolve the bearer token to a live user row.

        The token is not trusted on its own: the user is re-read from the database
        on every request, so a deactivated account stops working immediately
        instead of when its token happens to expire.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token")

    # Imported here to keep the auth primitives out of module import order.
    from permrag.security.tokens import decode_access_token

    payload = decode_access_token(credentials.credentials)

    result = await session.execute(select(User).where(User.id == payload.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise AuthenticationError("User no longer exists")
    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

async def require_admin(user: CurrentUser) -> User:
    if not user.is_admin:
        raise PermissionDeniedError("Administrator privileges required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
