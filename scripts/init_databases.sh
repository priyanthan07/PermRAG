#!/bin/bash
# Runs once, on first initialisation of the Postgres volume.
# Creates the second database that SpiceDB owns, so application data and
# authorization data never share a schema even though they share a server.
set -euo pipefail

SPICEDB_DB="${SPICEDB_DB:-spicedb}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE ${SPICEDB_DB}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${SPICEDB_DB}')\gexec

    GRANT ALL PRIVILEGES ON DATABASE ${SPICEDB_DB} TO ${POSTGRES_USER};
EOSQL

echo "init_databases.sh: ensured database '${SPICEDB_DB}' exists"
