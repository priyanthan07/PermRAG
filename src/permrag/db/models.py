"""
    ORM models.

    Two things worth understanding about this schema:

    1. ``user_department`` and ``document_share`` are *mirrors*. SpiceDB is the
    authority for access decisions; these tables exist so the admin panel can
    list and paginate grants without walking the permission graph. They must
    never be consulted on the retrieval path.

    2. ``permission_checkpoint`` holds the ZedToken from the most recent SpiceDB
    write. Reads are pinned to it so a revoke is visible on the very next
    query rather than "eventually" (the New Enemy Problem).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from permrag.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

class Department(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "department"

    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    memberships: Mapped[list["UserDepartment"]] = relationship(back_populates="department", cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="owner_department")

class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "app_user"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # UserDepartment has two FKs to app_user (user_id and granted_by_id), so
    # the join column must be stated explicitly.
    memberships: Mapped[list["UserDepartment"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="UserDepartment.user_id",
    )
    
class UserDepartment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Mirror of the SpiceDB ``department#member`` relationship."""

    __tablename__ = "user_department"
    __table_args__ = (UniqueConstraint("user_id", "department_id", name="uq_user_department_pair"),)

    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("department.id", ondelete="CASCADE"), nullable=False)
    is_manager: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    granted_by_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="SET NULL"))
    user: Mapped["User"] = relationship(back_populates="memberships", foreign_keys=[user_id])
    department: Mapped["Department"] = relationship(back_populates="memberships")
    
class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'indexing', 'indexed', 'failed', 'deleted')",
            name="status_valid",
        ),
        Index("ix_document_owner_status", "owner_department_id", "status"),
    )

    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    owner_department_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("department.id", ondelete="RESTRICT"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    owner_department: Mapped["Department"] = relationship(back_populates="documents")
    pages: Mapped[list["DocumentPage"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    shares: Mapped[list["DocumentShare"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class DocumentPage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
        One page of a document, with the hash that drives incremental re-indexing.
        Freshness is tracked per page, not per document: on re-sync only pages
        whose ``content_hash`` changed are re-embedded.
    """

    __tablename__ = "document_page"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page_number"),
        Index("ix_document_page_hash", "content_hash"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="pages")

class DocumentShare(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
        Mirror of SpiceDB ``document#shared_department`` / ``document#viewer``.

        Exactly one of ``department_id`` or ``user_id`` is set, enforced by a
        check constraint.
    """

    __tablename__ = "document_share"
    __table_args__ = (
        CheckConstraint(
            "(department_id IS NOT NULL AND user_id IS NULL) "
            "OR (department_id IS NULL AND user_id IS NOT NULL)",
            name="exactly_one_subject",
        ),
        UniqueConstraint("document_id", "department_id", name="uq_share_document_department"),
        UniqueConstraint("document_id", "user_id", name="uq_share_document_user"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("document.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("department.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="CASCADE"))
    granted_by_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="SET NULL"))
    reason: Mapped[str | None] = mapped_column(Text)
    document: Mapped["Document"] = relationship(back_populates="shares")

class PermissionCheckpoint(TimestampMixin, Base):
    """
        Single-row table holding the newest ZedToken observed from SpiceDB.

        Every permission write stores its returned token here. Every permission
        read is then pinned with ``at_least_as_fresh``, which is what makes a
        revoke take effect on the next query instead of after a cache expires.
    """

    __tablename__ = "permission_checkpoint"
    __table_args__ = (CheckConstraint("id = 1", name="singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    zed_token: Mapped[str | None] = mapped_column(Text)

class AuditLog(UUIDPrimaryKeyMixin, Base):
    """
        Append-only record of every permission-changing action.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_resource", "resource_type", "resource_id"),
        Index("ix_audit_log_created", "created_at"),
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    
class QueryLog(UUIDPrimaryKeyMixin, Base):
    """
        One row per answered question.

        ``permitted_document_ids`` and ``retrieved_chunk_ids`` are stored so a
        permission incident can be reconstructed after the fact: what the user was
        allowed to see at that moment, and what actually reached the model.
    """

    __tablename__ = "query_log"
    __table_args__ = (Index("ix_query_log_user_created", "user_id", "created_at"),)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("app_user.id", ondelete="SET NULL"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    permitted_document_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    permitted_document_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    zed_token: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    
