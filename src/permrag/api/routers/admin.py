import logging
import uuid

from fastapi import APIRouter, status
from sqlalchemy import select

from permrag.api.deps import AdminUser, DbSession, PermissionServiceDep
from permrag.api.schemas import (
    DepartmentCreate,
    DepartmentResponse,
    MembershipRequest,
    MembershipResponse,
    MessageResponse,
    UserCreate,
    UserResponse,
)
from permrag.db.models import Department, User, UserDepartment
from permrag.exceptions import ConflictError, NotFoundError
from permrag.security.passwords import hash_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# --- departments -------------------------------------------------------------

@router.post(
    "/departments",
    response_model=DepartmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_department(
    payload: DepartmentCreate,
    session: DbSession,
    _: AdminUser,
) -> Department:
    existing = await session.execute(select(Department).where(Department.slug == payload.slug))
    if existing.scalar_one_or_none() is not None:
        raise ConflictError(f"Department '{payload.slug}' already exists")

    department = Department(
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
    )
    session.add(department)
    await session.flush()

    logger.info("department created", extra={"department_id": str(department.id)})
    return department

@router.get("/departments", response_model=list[DepartmentResponse])
async def list_departments(session: DbSession, _: AdminUser) -> list[Department]:
    result = await session.execute(select(Department).order_by(Department.name))
    return list(result.scalars().all())

# --- users -------------------------------------------------------------------
@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    session: DbSession,
    _: AdminUser,
) -> User:
    """Provision an account. Department membership is granted separately."""
    email = payload.email.lower()
    existing = await session.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none() is not None:
        raise ConflictError(f"User '{email}' already exists")

    user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        is_admin=payload.is_admin,
        is_active=True,
    )
    session.add(user)
    await session.flush()

    logger.info("user provisioned", extra={"user_id": str(user.id)})
    return user

@router.get("/users", response_model=list[UserResponse])
async def list_users(session: DbSession, _: AdminUser) -> list[User]:
    result = await session.execute(select(User).order_by(User.email))
    return list(result.scalars().all())


@router.post("/users/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user(
    user_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> MessageResponse:
    """Disable an account and strip every relationship it holds.

    Deactivating the row alone would leave the SpiceDB edges intact, so the
    account would still appear in access listings and would regain everything
    if reactivated. Both are removed together.
    """
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError(f"User {user_id} not found")

    await permissions.purge_user_relationships(user_id=user_id, actor_id=admin.id)
    user.is_active = False
    await session.flush()

    logger.info("user deactivated", extra={"user_id": str(user_id)})
    return MessageResponse(message="User deactivated and all access revoked")

# --- membership --------------------------------------------------------------


@router.post("/memberships", response_model=MembershipResponse)
async def add_membership(
    payload: MembershipRequest,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> MembershipResponse:
    await _assert_exists(session, User, payload.user_id, "User")
    await _assert_exists(session, Department, payload.department_id, "Department")

    token = await permissions.add_user_to_department(
        user_id=payload.user_id,
        department_id=payload.department_id,
        actor_id=admin.id,
        is_manager=payload.is_manager,
    )
    return MembershipResponse(
        user_id=payload.user_id,
        department_id=payload.department_id,
        is_manager=payload.is_manager,
        zed_token=token,
    )
    
@router.delete("/memberships", response_model=MessageResponse)
async def remove_membership(
    user_id: uuid.UUID,
    department_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
    permissions: PermissionServiceDep,
) -> MessageResponse:
    await permissions.remove_user_from_department(
        user_id=user_id, department_id=department_id, actor_id=admin.id
    )
    return MessageResponse(message="Membership revoked")

@router.get("/departments/{department_id}/members", response_model=list[UserResponse])
async def list_department_members(
    department_id: uuid.UUID,
    session: DbSession,
    _: AdminUser,
) -> list[User]:
    """Read from the Postgres mirror.

    This is a listing convenience only. Access decisions never read this table.
    """
    result = await session.execute(
        select(User)
        .join(UserDepartment, UserDepartment.user_id == User.id)
        .where(UserDepartment.department_id == department_id)
        .order_by(User.email)
    )
    return list(result.scalars().all())

async def _assert_exists(session, model, record_id: uuid.UUID, label: str) -> None:
    result = await session.execute(select(model).where(model.id == record_id))
    if result.scalar_one_or_none() is None:
        raise NotFoundError(f"{label} {record_id} not found")
