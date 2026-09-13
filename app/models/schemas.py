"""Pydantic v2 models for chat / RAG request and response contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageRole(StrEnum):
    """Allowed roles in a Claude 3 conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Single turn in a Claude Messages API conversation."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(..., min_length=1, description="Plain-text content for this turn.")


class ChatRequest(BaseModel):
    """Incoming chat generation request.

    Attributes:
        messages: Full conversation history (must end with a user turn).
        use_rag: When True, retrieve and inject enterprise context before generation.
        system_prompt: Optional per-request system prompt override.
        max_tokens: Optional per-request token budget override.
        temperature: Optional per-request sampling temperature override.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(..., min_length=1)
    use_rag: bool = Field(default=False, description="Enable RAG context injection.")
    system_prompt: str | None = Field(default=None, max_length=16_000)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def validate_message_sequence(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure history ends with a user message (Claude Messages API requirement)."""
        if messages[-1].role != MessageRole.USER:
            raise ValueError("The last message in the conversation must have role 'user'.")
        return messages


class RagRequest(BaseModel):
    """RAG-dedicated completion request (retrieval is always enabled)."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(..., min_length=1)
    system_prompt: str | None = Field(default=None, max_length=16_000)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def validate_message_sequence(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role != MessageRole.USER:
            raise ValueError("The last message in the conversation must have role 'user'.")
        return messages

    def to_chat_request(self) -> ChatRequest:
        """Map to ChatRequest with ``use_rag=True``."""
        return ChatRequest(
            messages=self.messages,
            use_rag=True,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class UsageMetadata(BaseModel):
    """Token usage reported by Bedrock / Claude."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    """Successful chat generation response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model_id: str
    use_rag: bool
    rag_context_injected: bool = False
    stop_reason: str | None = None
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestRequest(BaseModel):
    """Upsert a document chunk into the pgvector store."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=1024)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: UUID | None = Field(
        default=None,
        description="Optional stable UUID; generated when omitted.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional precomputed embedding; otherwise Titan embeds ``content``.",
    )


class DocumentIngestResponse(BaseModel):
    """Result of a successful document upsert."""

    id: str
    source: str
    embedding_dimensions: int
    created: bool = Field(description="True when inserted; False when updated.")


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str = "ok"
    app_name: str
    environment: str
    version: str


class ReadinessCheck(BaseModel):
    """Single dependency check within a readiness probe."""

    name: str
    status: Literal["ok", "error", "skipped"]
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness probe: app + optional Postgres / RAG backend."""

    status: Literal["ready", "not_ready"]
    app_name: str
    environment: str
    version: str
    checks: list[ReadinessCheck]


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by global exception handlers."""

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
