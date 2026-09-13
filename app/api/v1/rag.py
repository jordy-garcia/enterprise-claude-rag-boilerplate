"""Dedicated RAG completion and document ingestion endpoints (API v1)."""

from fastapi import APIRouter, status

from app.api.dependencies import BedrockClientDep, ClaudeServiceDep, PgVectorClientDep
from app.models.schemas import (
    ChatResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    ErrorResponse,
    RagRequest,
)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/completions",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Bedrock invocation failure"},
    },
    summary="Generate a Claude completion with RAG always enabled",
)
async def create_rag_completion(
    payload: RagRequest,
    claude_service: ClaudeServiceDep,
) -> ChatResponse:
    """Always retrieve context before calling Claude (dedicated RAG path)."""
    return await claude_service.generate(payload.to_chat_request())


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "pgvector unavailable"},
    },
    summary="Upsert a document embedding into pgvector",
)
async def ingest_document(
    payload: DocumentIngestRequest,
    pgvector_client: PgVectorClientDep,
    bedrock_client: BedrockClientDep,
) -> DocumentIngestResponse:
    """Embed (unless provided) and upsert a chunk into the vector store."""
    embedding = payload.embedding
    if embedding is None:
        embedding = await bedrock_client.embed_text(payload.content)

    result = await pgvector_client.upsert_document(
        content=payload.content,
        source=payload.source,
        embedding=embedding,
        metadata=payload.metadata,
        document_id=payload.document_id,
    )
    return DocumentIngestResponse(
        id=str(result.id),
        source=payload.source,
        embedding_dimensions=pgvector_client.embedding_dimensions,
        created=result.created,
    )
