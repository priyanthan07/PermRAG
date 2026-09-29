from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Reranker tokens kept free for the question when sizing chunks (~45 words).
QUESTION_TOKEN_RESERVE = 64

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
    api_host: str = "0.0.0.0"
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
    
    # --- LLM provider ---
    # Which backend serves embeddings and answer generation. Only the chosen
    # provider's credentials need to be present.
    llm_provider: Literal["openai", "gemini"] = "openai"
    llm_timeout_seconds: float = 60.0
    
    # --- OpenAI ---
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 3
    
    # --- Gemini ---
    gemini_api_key: SecretStr | None = None
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimensions: int = 3072
    gemini_thinking_level: Literal["LOW", "MEDIUM", "HIGH"] | None = "LOW"
    
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
    
    # --- Reranking ---
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = ""
    reranker_batch_size: int = Field(default=32, ge=1, le=256)
    reranker_max_length: int = Field(default=512, ge=64, le=2048)
    reranker_min_score: float = 0.0
    reranker_threads: int = Field(default=4, ge=0, le=64)
    
    # --- Evaluation (DeepEval) ---
    eval_enabled: bool = True
    eval_judge_model: str = "gemini-3.1-flash-lite"
    eval_faithfulness_threshold: float = 0.5
    eval_answer_relevancy_threshold: float = 0.5
    eval_contextual_relevancy_threshold: float = 0.5
    
    max_permitted_documents: int = Field(default=5000, ge=1)

    # --- Chunking ---
    # Measured in reranker tokens (see ingestion/chunker.py). The reranker reads
    # [CLS] question [SEP] chunk [SEP] within reranker_max_length, so a chunk
    # may use at most reranker_max_length - 3 - QUESTION_TOKEN_RESERVE tokens
    # (445 at the defaults). 400 leaves headroom; the 60-token overlap (15%)
    # keeps a sentence cut at a boundary whole in one of the two chunks.
    chunk_size_tokens: int = Field(default=400, ge=32)
    chunk_overlap_tokens: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        budget = self.reranker_max_length - 3 - QUESTION_TOKEN_RESERVE
        if self.reranker_enabled and self.chunk_size_tokens > budget:
            # Too-long chunks are not an error at query time: the reranker
            # just truncates them and scores only the head. Refuse to start.
            raise ValueError(
                f"CHUNK_SIZE_TOKENS={self.chunk_size_tokens} does not fit the reranker window: "
                f"at most {budget} ({self.reranker_max_length} - 3 special tokens - "
                f"{QUESTION_TOKEN_RESERVE} reserved for the question)"
            )
        return self

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
    def embedding_dimensions(self) -> int:
        """
            Vector width for the active provider.
    
            The Qdrant collection is sized from this. Changing provider or model
            changes the width, and an existing collection is never resized in
            place -- it has to be dropped and rebuilt.
        """
        if self.llm_provider == "gemini":
            return self.gemini_embedding_dimensions
        return self.openai_embedding_dimensions
    
    @computed_field  # type: ignore[prop-decorator]
    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this rather than instantiating Settings()."""
    return Settings()  # type: ignore[call-arg]
