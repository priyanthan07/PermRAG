from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # --- Application ---
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_port: int = 8000
    
    # --- Postgres ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "permrag"
    postgres_password: SecretStr = SecretStr("permrag")
    postgres_db: str = "permrag"
    spicedb_db: str = "spicedb"
    
    db_pool_size: int = 10
    db_max_overflow: int = 5
    postgres_ssl_mode: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ] = "disable"
    
    # --- SpiceDB ---
    spicedb_endpoint: str = "localhost:50051"
    spicedb_token: SecretStr = SecretStr("permrag-local-dev-key")
    spicedb_insecure: bool = True
    
    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "permrag_chunks"
    
    # --- OpenAI ---
    openai_api_key: SecretStr
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 3
    
    # --- Langfuse ---
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True
    
    # --- Auth ---
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    
    # --- Bootstrap ---
    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_password: SecretStr = SecretStr("change-me")
    
    # --- Retrieval ---
    retrieval_candidate_limit: int = Field(default=40, ge=1, le=500)
    retrieval_final_limit: int = Field(default=8, ge=1, le=50)
    max_permitted_documents: int = Field(default=5000, ge=1)
    chunk_size_words: int = Field(default=280, ge=50)
    chunk_overlap_words: int = Field(default=50, ge=0)
    
    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL for the application database."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password.get_secret_value(),
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
                query=f"ssl={self.postgres_ssl_mode}",
            )
        )
        
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Synchronous URL. Alembic runs migrations through this one."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg",
                username=self.postgres_user,
                password=self.postgres_password.get_secret_value(),
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
                query=f"sslmode={self.postgres_ssl_mode}",
            )
        )
    
    @computed_field  # type: ignore[prop-decorator]
    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this rather than instantiating Settings()."""
    return Settings()  # type: ignore[call-arg]
