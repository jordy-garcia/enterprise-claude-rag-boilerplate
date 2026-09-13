"""FastAPI dependency providers (composition root for DI)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.bedrock_client import BedrockClient
from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.db.pgvector_client import PgVectorClient
from app.services.claude_service import ClaudeService
from app.services.rag_engine import BaseRagEngine, MockRagEngine, PgVectorRagEngine


def get_bedrock_client(request: Request) -> BedrockClient:
    """Prefer the lifespan-scoped BedrockClient (single instance per process)."""
    existing = getattr(request.app.state, "bedrock_client", None)
    if isinstance(existing, BedrockClient):
        return existing
    return BedrockClient(settings=get_settings())


def get_pgvector_client(request: Request) -> PgVectorClient:
    """Resolve the PgVectorClient attached during application lifespan."""
    client = getattr(request.app.state, "pgvector_client", None)
    if client is None:
        raise ServiceUnavailableError(
            "PgVector is not available. Set RAG_ENGINE=pgvector and configure DATABASE_URL.",
            details={"rag_engine": get_settings().rag_engine},
        )
    return client


def get_rag_engine(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    bedrock_client: Annotated[BedrockClient, Depends(get_bedrock_client)],
) -> BaseRagEngine:
    """Provide the active RAG engine (pgvector production or in-memory mock)."""
    if settings.rag_engine == "mock":
        existing_mock = getattr(request.app.state, "rag_engine", None)
        if isinstance(existing_mock, MockRagEngine):
            return existing_mock
        return MockRagEngine()

    existing = getattr(request.app.state, "rag_engine", None)
    if isinstance(existing, BaseRagEngine):
        return existing

    pgvector_client = get_pgvector_client(request)
    return PgVectorRagEngine(
        pgvector_client=pgvector_client,
        embed_fn=bedrock_client.embed_text,
        default_top_k=settings.rag_top_k,
    )


def get_claude_service(
    settings: Annotated[Settings, Depends(get_settings)],
    bedrock_client: Annotated[BedrockClient, Depends(get_bedrock_client)],
    rag_engine: Annotated[BaseRagEngine, Depends(get_rag_engine)],
) -> ClaudeService:
    """Compose ClaudeService with its collaborators for request-scoped use."""
    return ClaudeService(
        settings=settings,
        bedrock_client=bedrock_client,
        rag_engine=rag_engine,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClaudeServiceDep = Annotated[ClaudeService, Depends(get_claude_service)]
BedrockClientDep = Annotated[BedrockClient, Depends(get_bedrock_client)]
RagEngineDep = Annotated[BaseRagEngine, Depends(get_rag_engine)]
PgVectorClientDep = Annotated[PgVectorClient, Depends(get_pgvector_client)]
