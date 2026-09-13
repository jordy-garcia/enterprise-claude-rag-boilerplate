"""Application settings loaded from environment variables via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the FastAPI + Bedrock + pgvector application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="enterprise-claude-rag-boilerplate", alias="APP_NAME")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        alias="CORS_ORIGINS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    # ── AWS / Bedrock ────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        alias="BEDROCK_MODEL_ID",
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        alias="BEDROCK_EMBEDDING_MODEL_ID",
    )
    bedrock_max_tokens: int = Field(default=4096, alias="BEDROCK_MAX_TOKENS", ge=1, le=200_000)
    bedrock_temperature: float = Field(default=0.7, alias="BEDROCK_TEMPERATURE", ge=0.0, le=1.0)
    bedrock_top_p: float = Field(default=0.9, alias="BEDROCK_TOP_P", ge=0.0, le=1.0)

    # ── Claude ───────────────────────────────────────────────────────────────
    claude_system_prompt: str = Field(
        default=(
            "You are a helpful enterprise assistant. "
            "Answer accurately and cite context when available."
        ),
        alias="CLAUDE_SYSTEM_PROMPT",
    )

    # ── PostgreSQL / pgvector ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/rag",
        alias="DATABASE_URL",
    )
    pgvector_table: str = Field(default="documents", alias="PGVECTOR_TABLE")
    pgvector_embedding_dim: int = Field(default=1024, alias="PGVECTOR_EMBEDDING_DIM", ge=1)
    pgvector_distance: Literal["cosine", "inner_product"] = Field(
        default="cosine",
        alias="PGVECTOR_DISTANCE",
    )
    rag_engine: Literal["pgvector", "mock"] = Field(default="mock", alias="RAG_ENGINE")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K", ge=1, le=50)

    # ── OpenTelemetry ────────────────────────────────────────────────────────
    otel_service_name: str = Field(
        default="enterprise-claude-rag-boilerplate",
        alias="OTEL_SERVICE_NAME",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_traces_exporter: Literal["otlp", "console", "none"] = Field(
        default="otlp",
        alias="OTEL_TRACES_EXPORTER",
    )
    otel_sdk_disabled: bool = Field(default=False, alias="OTEL_SDK_DISABLED")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        """Allow comma-separated CORS origins in addition to JSON lists."""
        if isinstance(value, str) and not value.startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton (safe for FastAPI Depends)."""
    return Settings()
