import logging
from datetime import UTC, datetime

from fastapi import APIRouter, status
from sqlalchemy import select

from permrag.api.deps import CurrentUser, DbSession
from permrag.api.schemas import CurrentUserResponse, LoginRequest, TokenResponse
from permrag.db.models import User
from permrag.exceptions import AuthenticationError
from permrag.security.passwords import verify_password
from permrag.security.tokens import create_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: DbSession) -> TokenResponse:
    result = await session.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()

    # Same error for unknown email and wrong password, so the endpoint cannot
    # be used to enumerate which addresses have accounts.
    if user is None or not verify_password(payload.password, user.hashed_password):
        logger.warning("failed login attempt", extra={"email": payload.email})
        raise AuthenticationError("Incorrect email or password")

    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    user.last_login_at = datetime.now(UTC)
    await session.flush()

    token, expires_in = create_access_token(
        user_id=user.id, email=user.email, is_admin=user.is_admin
    )
    logger.info("login succeeded", extra={"user_id": str(user.id)})
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=CurrentUserResponse, status_code=status.HTTP_200_OK)
async def read_current_user(user: CurrentUser) -> User:
    return user
