"""Authorization layer. SpiceDB is the single source of truth for access."""

from permrag.permissions.client import SpiceDBClient, get_spicedb_client
from permrag.permissions.service import PermissionService

__all__ = ["PermissionService", "SpiceDBClient", "get_spicedb_client"]
