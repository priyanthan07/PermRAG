"""
    One-time setup: create both application databases on the shared Postgres
    instance -- `permrag` (app data) and `spicedb` (permission graph).

    Nothing runs this automatically -- it replaces scripts/init_databases.sh,
    which only worked via Docker's docker-entrypoint-initdb.d hook on the local
    Postgres container. Run once, by hand, after pointing .env at your external
    instance:

        uv run python scripts/provision_databases.py

    Connects to Postgres's own default `postgres` database, since neither
    target database exists yet to connect to directly. If your provider names
    its default database something else, set POSTGRES_MAINTENANCE_DB in .env.

    Reads every credential from Settings -- nothing pasted or hardcoded here.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psycopg  # noqa: E402
from permrag.config import get_settings  # noqa: E402
from permrag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger("provision_databases")

MAINTENANCE_DB = "postgres"

def _ensure_database(conn: psycopg.Connection, db_name: str) -> None:
    exists = conn.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
    ).fetchone()

    if exists:
        logger.info("database already exists, skipping", extra={"db": db_name})
        return

    # db_name always comes from our own Settings, never user input -- safe
    # to interpolate directly since identifiers can't be parameterised.
    conn.execute(f'CREATE DATABASE "{db_name}"')
    logger.info("database created", extra={"db": db_name})
    
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    # CREATE DATABASE cannot run inside a transaction block -- autocommit
    # must be set at connect time.
    conn = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        dbname=MAINTENANCE_DB,
        sslmode=settings.postgres_ssl_mode,
        autocommit=True,
    )

    try:
        _ensure_database(conn, settings.postgres_db)
        _ensure_database(conn, settings.spicedb_db)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
    