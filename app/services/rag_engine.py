"""RAG engine: abstract contract, mock stub, and pgvector production backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import RagEngineError
from app.core.logging import get_logger
from app.core.telemetry import get_tracer
from app.db.pgvector_client import PgVectorClient

logger = get_logger(__name__)
tracer = get_tracer(__name__)

EmbedFn = Callable[[str], Awaitable[list[float]]]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single piece of retrieved knowledge to inject into the prompt."""

    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRagEngine(ABC):
    """Pluggable RAG retrieval interface."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for ``query``."""

    async def build_context(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Retrieve chunks and format them as a single context block."""
        with tracer.start_as_current_span("rag.build_context"):
            try:
                chunks = await self.retrieve(query, top_k=top_k, filters=filters)
            except RagEngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("rag_retrieval_failed", query_preview=query[:120])
                raise RagEngineError(
                    message="Failed to retrieve RAG context",
                    details={"query_preview": query[:120]},
                ) from exc

            if not chunks:
                return ""

            sections: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                sections.append(
                    f"[{index}] source={chunk.source} score={chunk.score:.3f}\n{chunk.content}"
                )
            return "\n\n".join(sections)


class PgVectorRagEngine(BaseRagEngine):
    """Production RAG engine: embed query → pgvector similarity → top-K chunks."""

    def __init__(
        self,
        pgvector_client: PgVectorClient,
        embed_fn: EmbedFn,
        *,
        default_top_k: int = 5,
    ) -> None:
        self._pgvector = pgvector_client
        self._embed = embed_fn
        self._default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed ``query`` and run cosine / inner-product search in PostgreSQL."""
        resolved_top_k = top_k or self._default_top_k

        with tracer.start_as_current_span(
            "rag.pgvector.retrieve",
            attributes={"rag.top_k": resolved_top_k},
        ):
            embedding = await self._embed(query)
            hits = await self._pgvector.similarity_search(
                embedding,
                top_k=resolved_top_k,
                filters=filters,
            )

        chunks = [
            RetrievedChunk(
                content=hit.content,
                source=hit.source,
                score=hit.score,
                metadata={**hit.metadata, "document_id": str(hit.id)},
            )
            for hit in hits
        ]
        logger.info("rag_pgvector_retrieve", query_preview=query[:80], chunks=len(chunks))
        return chunks


class MockRagEngine(BaseRagEngine):
    """Deterministic in-memory RAG stub for local development and tests."""

    def __init__(self, corpus: list[RetrievedChunk] | None = None) -> None:
        self._corpus: list[RetrievedChunk] = corpus or [
            RetrievedChunk(
                content=(
                    "This enterprise Claude RAG boilerplate uses FastAPI, "
                    "AWS Bedrock, pgvector, and OpenTelemetry."
                ),
                source="docs/architecture.md",
                score=0.92,
                metadata={"doc_type": "architecture"},
            ),
            RetrievedChunk(
                content=(
                    "Claude 3 on Bedrock expects the Messages API payload with "
                    "roles 'user' and 'assistant', plus an optional system prompt."
                ),
                source="docs/bedrock-claude.md",
                score=0.88,
                metadata={"doc_type": "integration"},
            ),
            RetrievedChunk(
                content=(
                    "Use asyncio.to_thread around boto3 InvokeModel calls to keep "
                    "the FastAPI event loop non-blocking under high concurrency."
                ),
                source="docs/concurrency.md",
                score=0.85,
                metadata={"doc_type": "performance"},
            ),
        ]

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Naive keyword overlap ranking over the in-memory corpus."""
        _ = filters
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        if not query_terms:
            return self._corpus[:top_k]

        scored: list[RetrievedChunk] = []
        for chunk in self._corpus:
            content_terms = {term.lower() for term in chunk.content.split()}
            overlap = len(query_terms & content_terms)
            adjusted_score = chunk.score + (0.05 * overlap)
            scored.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    score=min(adjusted_score, 1.0),
                    metadata=chunk.metadata,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
