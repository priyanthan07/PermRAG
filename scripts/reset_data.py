"""
    DESTRUCTIVE: drop and recreate the application and SpiceDB databases.

    Used by `make reset`. Postgres is external to docker compose, so
    `docker compose down -v` only wipes Qdrant; without this step the
    documents, page hashes and permission graph would survive a "reset", and
    re-uploading would skip every page as unchanged while Qdrant is empty.

    Database names come from Settings (POSTGRES_DB, SPICEDB_DB). Refuses to
    run without --yes:

        uv run python scripts/reset_data.py --yes
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psycopg

from permrag.config import get_settings
from permrag.logging_config import configure_logging

logger = logging.getLogger("reset_data")

MAINTENANCE_DB = "postgres"


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    targets = (settings.postgres_db, settings.spicedb_db)

    if "--yes" not in sys.argv:
        sys.exit(f"Refusing to drop {', '.join(targets)} without --yes.")

    # CREATE/DROP DATABASE cannot run inside a transaction block.
    with psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        dbname=MAINTENANCE_DB,
        sslmode=settings.postgres_ssl_mode,
        autocommit=True,
    ) as conn:
        for name in targets:
            # Names come from our own Settings, never user input; identifiers
            # cannot be parameterised. FORCE (Postgres 13+) closes open sessions,
            # e.g. an API still running on the host.
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            conn.execute(f'CREATE DATABASE "{name}"')
            logger.info("database recreated", extra={"db": name})


if __name__ == "__main__":
    main()
