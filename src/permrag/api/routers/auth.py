import asyncio
import logging
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import APIRouter, Request, status
from sqlalchemy import select

from permrag.api.deps import CurrentUser, DbSession
from permrag.api.schemas import CurrentUserResponse, LoginRequest, TokenResponse
from permrag.db.models import User
from permrag.exceptions import AuthenticationError
from permrag.security.passwords import hash_password, verify_password
from permrag.security.throttle import login_throttle
from permrag.security.tokens import create_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """A real bcrypt hash to verify against when the email is unknown."""
    return hash_password("timing-equaliser-not-a-real-account")


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request, session: DbSession) -> TokenResponse:
    email = payload.email.lower()
    client_ip = request.client.host if request.client else "unknown"
    throttle_key = f"{email}|{client_ip}"
    login_throttle.check(throttle_key)

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    # Same error for unknown email and wrong password, so the endpoint cannot
    # be used to enumerate which addresses have accounts -- and a bcrypt check
    # runs either way, so the response time does not give it away either.
    # bcrypt is deliberately slow CPU work: run it off the event loop.
    stored_hash = user.hashed_password if user is not None else await asyncio.to_thread(_dummy_hash)
    password_ok = await asyncio.to_thread(verify_password, payload.password, stored_hash)

    if user is None or not password_ok:
        login_throttle.record_failure(throttle_key)
        logger.warning("failed login attempt", extra={"email": email})
        raise AuthenticationError("Incorrect email or password")

    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    login_throttle.reset(throttle_key)
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
