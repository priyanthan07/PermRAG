"""initial_schema

Revision ID: 7dd784305a19
Revises: 
Create Date: 2026-09-07 10:05:14.154937

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7dd784305a19'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('app_user',
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('full_name', sa.String(length=255), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('is_admin', sa.Boolean(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_app_user'))
    )
    op.create_index(op.f('ix_app_user_email'), 'app_user', ['email'], unique=True)
    op.create_table('department',
    sa.Column('slug', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_department')),
    sa.UniqueConstraint('slug', name=op.f('uq_department_slug'))
    )
    op.create_table('permission_checkpoint',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('zed_token', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('id = 1', name=op.f('ck_permission_checkpoint_singleton')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_permission_checkpoint'))
    )
    op.create_table('audit_log',
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor_user_id', sa.UUID(), nullable=True),
    sa.Column('action', sa.String(length=64), nullable=False),
    sa.Column('resource_type', sa.String(length=64), nullable=False),
    sa.Column('resource_id', sa.String(length=255), nullable=False),
    sa.Column('detail', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['actor_user_id'], ['app_user.id'], name=op.f('fk_audit_log_actor_user_id_app_user'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_log'))
    )
    op.create_index('ix_audit_log_created', 'audit_log', ['created_at'], unique=False)
    op.create_index('ix_audit_log_resource', 'audit_log', ['resource_type', 'resource_id'], unique=False)
    op.create_table('document',
    sa.Column('external_id', sa.String(length=255), nullable=False),
    sa.Column('title', sa.String(length=512), nullable=False),
    sa.Column('source_uri', sa.Text(), nullable=True),
    sa.Column('owner_department_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('pending', 'indexing', 'indexed', 'failed', 'deleted')", name=op.f('ck_document_status_valid')),
    sa.ForeignKeyConstraint(['owner_department_id'], ['department.id'], name=op.f('fk_document_owner_department_id_department'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document')),
    sa.UniqueConstraint('external_id', name=op.f('uq_document_external_id'))
    )
    op.create_index('ix_document_owner_status', 'document', ['owner_department_id', 'status'], unique=False)
    op.create_table('query_log',
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('answer', sa.Text(), nullable=True),
    sa.Column('permitted_document_count', sa.Integer(), nullable=False),
    sa.Column('permitted_document_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('retrieved_chunk_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('zed_token', sa.Text(), nullable=True),
    sa.Column('trace_id', sa.String(length=64), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['app_user.id'], name=op.f('fk_query_log_user_id_app_user'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_query_log'))
    )
    op.create_index('ix_query_log_user_created', 'query_log', ['user_id', 'created_at'], unique=False)
    op.create_table('user_department',
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('department_id', sa.UUID(), nullable=False),
    sa.Column('is_manager', sa.Boolean(), nullable=False),
    sa.Column('granted_by_id', sa.UUID(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['department_id'], ['department.id'], name=op.f('fk_user_department_department_id_department'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['granted_by_id'], ['app_user.id'], name=op.f('fk_user_department_granted_by_id_app_user'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['app_user.id'], name=op.f('fk_user_department_user_id_app_user'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_department')),
    sa.UniqueConstraint('user_id', 'department_id', name='uq_user_department_pair')
    )
    op.create_table('document_page',
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('char_count', sa.Integer(), nullable=False),
    sa.Column('chunk_count', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], name=op.f('fk_document_page_document_id_document'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_page')),
    sa.UniqueConstraint('document_id', 'page_number', name='uq_document_page_number')
    )
    op.create_index('ix_document_page_hash', 'document_page', ['content_hash'], unique=False)
    op.create_table('document_share',
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('department_id', sa.UUID(), nullable=True),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.Column('granted_by_id', sa.UUID(), nullable=True),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('(department_id IS NOT NULL AND user_id IS NULL) OR (department_id IS NULL AND user_id IS NOT NULL)', name=op.f('ck_document_share_exactly_one_subject')),
    sa.ForeignKeyConstraint(['department_id'], ['department.id'], name=op.f('fk_document_share_department_id_department'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], name=op.f('fk_document_share_document_id_document'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['granted_by_id'], ['app_user.id'], name=op.f('fk_document_share_granted_by_id_app_user'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['app_user.id'], name=op.f('fk_document_share_user_id_app_user'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_share')),
    sa.UniqueConstraint('document_id', 'department_id', name='uq_share_document_department'),
    sa.UniqueConstraint('document_id', 'user_id', name='uq_share_document_user')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('document_share')
    op.drop_index('ix_document_page_hash', table_name='document_page')
    op.drop_table('document_page')
    op.drop_table('user_department')
    op.drop_index('ix_query_log_user_created', table_name='query_log')
    op.drop_table('query_log')
    op.drop_index('ix_document_owner_status', table_name='document')
    op.drop_table('document')
    op.drop_index('ix_audit_log_resource', table_name='audit_log')
    op.drop_index('ix_audit_log_created', table_name='audit_log')
    op.drop_table('audit_log')
    op.drop_table('permission_checkpoint')
    op.drop_table('department')
    op.drop_index(op.f('ix_app_user_email'), table_name='app_user')
    op.drop_table('app_user')
    # ### end Alembic commands ###
