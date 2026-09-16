import logging
from dataclasses import dataclass

import grpc
from authzed.api.v1 import (
    Cursor,
    CheckPermissionRequest,
    CheckPermissionResponse,
    Client,
    Consistency,
    DeleteRelationshipsRequest,
    LookupResourcesRequest,
    ObjectReference,
    ReadRelationshipsRequest,
    Relationship,
    RelationshipFilter,
    RelationshipUpdate,
    SubjectFilter,
    SubjectReference,
    WriteRelationshipsRequest,
    WriteSchemaRequest,
    ZedToken,
)
from grpcutil import bearer_token_credentials, insecure_bearer_token_credentials

from permrag.config import get_settings
from permrag.exceptions import PermissionSystemError

logger = logging.getLogger(__name__)

# Object types, mirroring schema/permrag.zed. Keep in sync with that file.
TYPE_USER = "user"
TYPE_DEPARTMENT = "department"
TYPE_DOCUMENT = "document"

# Relations
REL_MEMBER = "member"
REL_MANAGER = "manager"
REL_OWNER_DEPARTMENT = "owner_department"
REL_SHARED_DEPARTMENT = "shared_department"
REL_VIEWER = "viewer"

# SpiceDB refuses an optional_limit above this on LookupResources.
SPICEDB_MAX_PAGE_SIZE = 1000

# Permissions
PERM_VIEW = "view"
PERM_ADMINISTER = "administer"


@dataclass(frozen=True, slots=True)
class RelationshipTuple:
    """A single (resource, relation, subject) edge in the permission graph."""

    resource_type: str
    resource_id: str
    relation: str
    subject_type: str
    subject_id: str
    subject_relation: str | None = None

    def to_proto(self) -> Relationship:
        subject = SubjectReference(
            object=ObjectReference(
                object_type=self.subject_type,
                object_id=self.subject_id,
            ),
            optional_relation=self.subject_relation or "",
        )
        return Relationship(
            resource=ObjectReference(
                object_type=self.resource_type,
                object_id=self.resource_id,
            ),
            relation=self.relation,
            subject=subject,
        )


def user_subject(user_id: str) -> SubjectReference:
    return SubjectReference(object=ObjectReference(object_type=TYPE_USER, object_id=user_id))


class SpiceDBClient:
    """Async SpiceDB client. One instance per process, shared by all requests."""

    def __init__(self) -> None:
        settings = get_settings()
        token = settings.spicedb_token.get_secret_value()

        if settings.spicedb_insecure:
            credentials = insecure_bearer_token_credentials(token)
        else:
            credentials = bearer_token_credentials(token)

        # Client auto-selects the asyncio channel because it is constructed
        # inside a running event loop (the FastAPI lifespan hook).
        self._client = Client(settings.spicedb_endpoint, credentials)
        self._endpoint = settings.spicedb_endpoint
        logger.info("spicedb client initialised", extra={"endpoint": self._endpoint})

    # -- schema ------------------------------------------------------------

    async def write_schema(self, schema_text: str) -> None:
        """Apply the .zed schema. Idempotent; safe to call on every startup."""
        try:
            await self._client.WriteSchema(WriteSchemaRequest(schema=schema_text))
            logger.info("spicedb schema applied")
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Failed to write schema: {exc}") from exc

    # -- writes ------------------------------------------------------------

    async def write_relationships(
        self,
        creates: list[RelationshipTuple] | None = None,
        deletes: list[RelationshipTuple] | None = None,
    ) -> str:
        """Apply relationship changes atomically. Returns the new ZedToken.

        Creates use TOUCH rather than CREATE so re-granting an existing
        permission is idempotent instead of an error.
        """
        updates: list[RelationshipUpdate] = []

        for tup in creates or []:
            updates.append(
                RelationshipUpdate(
                    operation=RelationshipUpdate.Operation.OPERATION_TOUCH,
                    relationship=tup.to_proto(),
                )
            )
        for tup in deletes or []:
            updates.append(
                RelationshipUpdate(
                    operation=RelationshipUpdate.Operation.OPERATION_DELETE,
                    relationship=tup.to_proto(),
                )
            )

        if not updates:
            raise ValueError("write_relationships called with no updates")

        try:
            response = await self._client.WriteRelationships(
                WriteRelationshipsRequest(updates=updates)
            )
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Relationship write failed: {exc}") from exc

        logger.info(
            "relationships written",
            extra={"creates": len(creates or []), "deletes": len(deletes or [])},
        )
        return response.written_at.token

    async def delete_by_filter(
        self,
        resource_type: str,
        resource_id: str | None = None,
        relation: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
    ) -> str:
        """Bulk-delete matching relationships. Returns the new ZedToken.

        Used when a document or department is removed and every edge touching
        it must go, without enumerating them first.
        """
        subject_filter = None
        if subject_type:
            subject_filter = SubjectFilter(
                subject_type=subject_type,
                optional_subject_id=subject_id or "",
            )

        rel_filter = RelationshipFilter(
            resource_type=resource_type,
            optional_resource_id=resource_id or "",
            optional_relation=relation or "",
            optional_subject_filter=subject_filter,
        )

        try:
            response = await self._client.DeleteRelationships(
                DeleteRelationshipsRequest(relationship_filter=rel_filter)
            )
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Relationship delete failed: {exc}") from exc

        logger.info(
            "relationships deleted by filter",
            extra={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "count": response.relationships_deleted_count,
            },
        )
        return response.deleted_at.token

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _consistency(zed_token: str | None) -> Consistency:
        """Pin the read to at least the freshness of the last write.

        Without a token we fall back to ``fully_consistent``: slower, but it
        never serves a stale permission. Choosing ``minimize_latency`` here
        would reintroduce exactly the revocation window this design exists to
        close.
        """
        if zed_token:
            return Consistency(at_least_as_fresh=ZedToken(token=zed_token))
        return Consistency(fully_consistent=True)

    async def lookup_viewable_documents(
        self,
        user_id: str,
        zed_token: str | None = None,
        limit: int = 1000,
    ) -> tuple[list[str], str | None]:
        """Return every document id this user may view.

        This is the call that gates retrieval. Its result becomes the
        mandatory filter passed into the vector search.

        SpiceDB rejects any optional_limit above SPICEDB_MAX_PAGE_SIZE, so
        `limit` here is the total we want, not the page size: the lookup is
        paged with a cursor and the pages are concatenated. Without this,
        any organisation whose users can see more than that many documents
        would fail closed on every single query.
        """
        document_ids: list[str] = []
        observed_token: str | None = None
        cursor: Cursor | None = None

        try:
            while len(document_ids) < limit:
                page_size = min(SPICEDB_MAX_PAGE_SIZE, limit - len(document_ids))
                request = LookupResourcesRequest(
                    consistency=self._consistency(zed_token),
                    resource_object_type=TYPE_DOCUMENT,
                    permission=PERM_VIEW,
                    subject=user_subject(user_id),
                    optional_limit=page_size,
                    optional_cursor=cursor,
                )

                received = 0
                next_cursor: Cursor | None = None

                async for response in self._client.LookupResources(request):
                    document_ids.append(response.resource_object_id)
                    received += 1
                    if response.looked_up_at.token:
                        observed_token = response.looked_up_at.token
                    if response.HasField("after_result_cursor"):
                        next_cursor = response.after_result_cursor

                # A short page means the result set is exhausted. Checking
                # this rather than only the cursor avoids looping forever if
                # the server keeps returning one.
                if received < page_size or next_cursor is None:
                    break

                cursor = next_cursor
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Permission lookup failed: {exc}") from exc

        if len(document_ids) >= limit:
            # Silently truncating would make documents invisible with no
            # signal, which looks identical to a permission problem.
            logger.warning(
                "permission lookup hit the configured ceiling; results truncated",
                extra={"user_id": user_id, "limit": limit},
            )

        logger.info(
            "permission lookup complete",
            extra={"user_id": user_id, "document_count": len(document_ids)},
        )
        return document_ids, observed_token

    async def check_permission(
        self,
        user_id: str,
        document_id: str,
        permission: str = PERM_VIEW,
        zed_token: str | None = None,
    ) -> bool:
        """Point check for a single document. Used as a post-retrieval assertion."""
        request = CheckPermissionRequest(
            consistency=self._consistency(zed_token),
            resource=ObjectReference(object_type=TYPE_DOCUMENT, object_id=document_id),
            permission=permission,
            subject=user_subject(user_id),
        )
        try:
            response = await self._client.CheckPermission(request)
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Permission check failed: {exc}") from exc

        return (
            response.permissionship
            == CheckPermissionResponse.Permissionship.PERMISSIONSHIP_HAS_PERMISSION
        )

    async def read_relationships(
        self,
        resource_type: str,
        resource_id: str | None = None,
        relation: str | None = None,
        zed_token: str | None = None,
    ) -> list[RelationshipTuple]:
        """List existing relationships. Powers the admin "who has access" view."""
        request = ReadRelationshipsRequest(
            consistency=self._consistency(zed_token),
            relationship_filter=RelationshipFilter(
                resource_type=resource_type,
                optional_resource_id=resource_id or "",
                optional_relation=relation or "",
            ),
        )

        results: list[RelationshipTuple] = []
        try:
            async for response in self._client.ReadRelationships(request):
                rel = response.relationship
                results.append(
                    RelationshipTuple(
                        resource_type=rel.resource.object_type,
                        resource_id=rel.resource.object_id,
                        relation=rel.relation,
                        subject_type=rel.subject.object.object_type,
                        subject_id=rel.subject.object.object_id,
                        subject_relation=rel.subject.optional_relation or None,
                    )
                )
        except grpc.RpcError as exc:
            raise PermissionSystemError(f"Relationship read failed: {exc}") from exc

        return results


_client: SpiceDBClient | None = None


def get_spicedb_client() -> SpiceDBClient:
    """Process-wide singleton. Do not construct SpiceDBClient directly."""
    global _client
    if _client is None:
        _client = SpiceDBClient()
    return _client
