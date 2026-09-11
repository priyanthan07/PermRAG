
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import select

# Allow `python scripts/bootstrap.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from permrag.config import get_settings  # noqa: E402
from permrag.db.models import PermissionCheckpoint, User  # noqa: E402
from permrag.db.session import dispose_engine, session_scope  # noqa: E402
from permrag.logging_config import configure_logging  # noqa: E402
from permrag.permissions.client import get_spicedb_client  # noqa: E402
from permrag.security.passwords import hash_password  # noqa: E402

logger = logging.getLogger("bootstrap")

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "permrag.zed"


async def apply_spicedb_schema() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"SpiceDB schema not found at {SCHEMA_PATH}")

    client = get_spicedb_client()
    await client.write_schema(SCHEMA_PATH.read_text(encoding="utf-8"))
    logger.info("spicedb schema applied")


async def ensure_checkpoint_row() -> None:
    """Create the singleton ZedToken row so the first write has a home."""
    async with session_scope() as session:
        result = await session.execute(
            select(PermissionCheckpoint).where(PermissionCheckpoint.id == 1)
        )
        if result.scalar_one_or_none() is None:
            session.add(PermissionCheckpoint(id=1, zed_token=None))
            logger.info("permission checkpoint row created")


async def ensure_admin_user() -> None:
    settings = get_settings()
    email = settings.bootstrap_admin_email.lower()
    password = settings.bootstrap_admin_password.get_secret_value()

    if password == "admin":
        logger.warning(
            "bootstrap admin is using the default password; change it before "
            "exposing this service"
        )

    async with session_scope() as session:
        result = await session.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none() is not None:
            logger.info("admin already exists, skipping", extra={"email": email})
            return

        session.add(
            User(
                email=email,
                full_name="Bootstrap Administrator",
                hashed_password=hash_password(password),
                is_admin=True,
                is_active=True,
            )
        )
        logger.info("admin user created", extra={"email": email})


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("bootstrap starting", extra={"environment": settings.environment})
    try:
        await apply_spicedb_schema()
        await ensure_checkpoint_row()
        await ensure_admin_user()
        logger.info("bootstrap complete")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
    