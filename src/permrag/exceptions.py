class PermRAGError(Exception):
    """Base class for every error this application raises deliberately."""

    status_code: int = 500
    detail: str = "Internal error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class ConfigurationError(PermRAGError):
    status_code = 500
    detail = "Service is misconfigured"


class AuthenticationError(PermRAGError):
    status_code = 401
    detail = "Could not validate credentials"


class PermissionDeniedError(PermRAGError):
    status_code = 403
    detail = "You do not have access to this resource"


class NotFoundError(PermRAGError):
    status_code = 404
    detail = "Resource not found"


class ConflictError(PermRAGError):
    status_code = 409
    detail = "Resource already exists"


class ValidationError(PermRAGError):
    status_code = 422
    detail = "Request failed validation"


class PermissionSystemError(PermRAGError):
    """SpiceDB was unreachable or returned an error.

    Retrieval treats this as fatal and returns nothing. Failing closed is the
    only safe behaviour when the authorization system cannot be consulted.
    """

    status_code = 503
    detail = "Permission system unavailable"


class VectorStoreError(PermRAGError):
    status_code = 503
    detail = "Vector store unavailable"


class LLMError(PermRAGError):
    status_code = 503
    detail = "Language model unavailable"
    