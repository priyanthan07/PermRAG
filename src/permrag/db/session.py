"""
    Async engine and session factory.

    A single engine is created per process and reused. Sessions are handed out
    per request through the context manager below, which guarantees a rollback on
    any exception -- the failure mode that leaves a raw psycopg connection stuck
    in an aborted transaction.
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from permrag.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

# Dedicated pool for ZedToken checkpoint writes. A permission write holds its
# request connection while it waits for the checkpoint lock; if the checkpoint
# came from the same pool, enough concurrent writers could take every
# connection and wait on each other forever.
_checkpoint_engine: AsyncEngine | None = None
_checkpoint_factory: async_sessionmaker[AsyncSession] | None = None
CHECKPOINT_POOL_SIZE = 2

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=False,
        )
        logger.info("database engine created", extra={"host": settings.postgres_host})
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def checkpoint_scope() -> AsyncIterator[AsyncSession]:
    """Short transaction on the checkpoint pool. Commits on success, rolls back on any exception."""
    global _checkpoint_engine, _checkpoint_factory
    if _checkpoint_factory is None:
        settings = get_settings()
        _checkpoint_engine = create_async_engine(
            settings.database_url,
            pool_size=CHECKPOINT_POOL_SIZE,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        _checkpoint_factory = async_sessionmaker(
            bind=_checkpoint_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )

    async with _checkpoint_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the pools. Called from the FastAPI lifespan shutdown hook."""
    global _engine, _session_factory, _checkpoint_engine, _checkpoint_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database engine disposed")
    if _checkpoint_engine is not None:
        await _checkpoint_engine.dispose()
        _checkpoint_engine = None
        _checkpoint_factory = None
        