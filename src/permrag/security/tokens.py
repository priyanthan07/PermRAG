import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from permrag.config import get_settings
from permrag.exceptions import AuthenticationError

@dataclass(frozen=True, slots=True)
class TokenPayload:
    user_id: uuid.UUID
    email: str
    is_admin: bool
    expires_at: datetime
    
def create_access_token(user_id: uuid.UUID, email: str, is_admin: bool) -> tuple[str, int]:
    """Issue a signed token. Returns (token, expires_in_seconds)."""
    settings = get_settings()
    expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    expires_at = datetime.now(UTC) + expires_delta

    claims = {
        "sub": str(user_id),
        "email": email,
        "is_admin": is_admin,
        "iat": int(datetime.now(UTC).timestamp()),
        "exp": int(expires_at.timestamp()),
    }

    token = jwt.encode(
        claims,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, int(expires_delta.total_seconds())

def decode_access_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Token is invalid") from exc

    try:
        return TokenPayload(
            user_id=uuid.UUID(claims["sub"]),
            email=claims["email"],
            is_admin=bool(claims.get("is_admin", False)),
            expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
        )
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Token payload is malformed") from exc
    