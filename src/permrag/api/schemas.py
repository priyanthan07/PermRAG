import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field

# --- auth ------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CurrentUserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    is_admin: bool
    is_active: bool


# --- departments ---------------

class DepartmentCreate(BaseModel):
    slug: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class DepartmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    created_at: datetime


# --- users -------------

class UserCreate(BaseModel):
    """Admin-provisioned account creation.

    There is no department field a user can set for themselves. Membership is
    granted separately, by an admin, through the membership endpoint.
    """

    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=72)
    is_admin: bool = False


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    is_admin: bool
    is_active: bool
    created_at: datetime


class MembershipRequest(BaseModel):
    user_id: uuid.UUID
    department_id: uuid.UUID
    is_manager: bool = False


class MembershipResponse(BaseModel):
    user_id: uuid.UUID
    department_id: uuid.UUID
    is_manager: bool
    zed_token: str


# --- documents ------------

class PagePayload(BaseModel):
    page_number: int = Field(ge=1)
    content: str


class DocumentIngestRequest(BaseModel):
    external_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=512)
    owner_department_id: uuid.UUID
    source_uri: str | None = None
    pages: list[PagePayload] = Field(min_length=1)


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    pages_total: int
    pages_indexed: int
    pages_skipped: int
    pages_removed: int
    chunks_written: int


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    external_id: str
    title: str
    source_uri: str | None
    owner_department_id: uuid.UUID
    status: str
    indexed_at: datetime | None
    created_at: datetime


# --- sharing -------------------

class ShareWithDepartmentRequest(BaseModel):
    department_id: uuid.UUID
    reason: str | None = None


class ShareWithUserRequest(BaseModel):
    user_id: uuid.UUID
    reason: str | None = None


class ShareResponse(BaseModel):
    document_id: uuid.UUID
    zed_token: str
    message: str


class AccessEntry(BaseModel):
    relation: str
    subject_type: str
    subject_id: str


class DocumentAccessResponse(BaseModel):
    document_id: uuid.UUID
    entries: list[AccessEntry]


# --- chat ------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class CitationResponse(BaseModel):
    index: int
    document_id: str
    title: str
    page_number: int
    source_uri: str | None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    permitted_document_count: int
    retrieved_chunk_count: int
    trace_id: str | None
    latency_ms: int


# --- misc -----------------------
class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class MessageResponse(BaseModel):
    message: str
    