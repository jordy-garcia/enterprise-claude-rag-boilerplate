"""Async PostgreSQL + pgvector client (SQLAlchemy + asyncpg)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, MetaData, Table, Text, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import Select

from app.core.config import Settings
from app.core.exceptions import RagEngineError
from app.core.logging import get_logger
from app.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

DistanceMetric = Literal["cosine", "inner_product"]

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One row returned by a pgvector similarity query."""

    id: UUID | str
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """Outcome of inserting or updating a document row."""

    id: UUID
    created: bool


class PgVectorClient:
    """Async connection pool + cosine / inner-product similarity search.

    Uses SQLAlchemy 2.0 async engine (asyncpg driver) and the ``pgvector``
    SQLAlchemy type so distance operators (``<=>``, ``<#>``) are first-class.
    """

    def __init__(self, settings: Settings) -> None:
        if not _TABLE_NAME_RE.match(settings.pgvector_table):
            raise ValueError(f"Invalid PGVECTOR_TABLE name: {settings.pgvector_table!r}")

        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._documents = self._build_table(
            table_name=settings.pgvector_table,
            dimensions=settings.pgvector_embedding_dim,
        )

    @staticmethod
    def _build_table(*, table_name: str, dimensions: int) -> Table:
        metadata = MetaData()
        return Table(
            table_name,
            metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
            Column("content", Text, nullable=False),
            Column("source", Text, nullable=False),
            Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
            Column("embedding", Vector(dimensions), nullable=False),
        )

    @property
    def engine(self) -> AsyncEngine:
        """Expose the async engine for advanced consumers."""
        return self._engine

    @property
    def embedding_dimensions(self) -> int:
        """Configured embedding vector size."""
        return self._settings.pgvector_embedding_dim

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a request-scoped async SQLAlchemy session."""
        async with self._session_factory() as session:
            yield session

    async def connect(self) -> None:
        """Warm the pool and ensure the ``vector`` extension is available."""
        with tracer.start_as_current_span("pgvector.connect"):
            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info(
                "pgvector_connected",
                table=self._settings.pgvector_table,
                dimensions=self._settings.pgvector_embedding_dim,
            )

    async def close(self) -> None:
        """Dispose the async engine / connection pool."""
        await self._engine.dispose()
        logger.info("pgvector_pool_closed")

    async def ping(self) -> bool:
        """Return True when the database accepts a trivial query."""
        with tracer.start_as_current_span("pgvector.ping"):
            try:
                async with self._engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                return True
            except Exception:  # noqa: BLE001
                logger.exception("pgvector_ping_failed")
                return False

    def _validate_embedding(self, embedding: Sequence[float]) -> list[float]:
        vector_values = [float(value) for value in embedding]
        if len(vector_values) != self._settings.pgvector_embedding_dim:
            raise RagEngineError(
                message="Embedding dimension mismatch",
                details={
                    "expected": self._settings.pgvector_embedding_dim,
                    "received": len(vector_values),
                },
            )
        return vector_values

    async def upsert_document(
        self,
        *,
        content: str,
        source: str,
        embedding: Sequence[float],
        metadata: dict[str, Any] | None = None,
        document_id: UUID | None = None,
    ) -> UpsertResult:
        """Insert or update a single document row (ON CONFLICT by ``id``)."""
        vector_values = self._validate_embedding(embedding)
        doc_id = document_id or uuid4()
        meta = metadata or {}

        with tracer.start_as_current_span(
            "pgvector.upsert_document",
            attributes={"db.system": "postgresql", "db.operation": "upsert"},
        ):
            try:
                async with self.session() as session:
                    exists_result = await session.execute(
                        select(self._documents.c.id).where(self._documents.c.id == doc_id)
                    )
                    existed = exists_result.scalar_one_or_none() is not None

                    stmt = (
                        pg_insert(self._documents)
                        .values(
                            id=doc_id,
                            content=content,
                            source=source,
                            metadata=meta,
                            embedding=vector_values,
                        )
                        .on_conflict_do_update(
                            index_elements=[self._documents.c.id],
                            set_={
                                "content": content,
                                "source": source,
                                "metadata": meta,
                                "embedding": vector_values,
                            },
                        )
                    )
                    await session.execute(stmt)
                    await session.commit()
            except RagEngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("pgvector_upsert_failed", document_id=str(doc_id))
                raise RagEngineError(
                    message="Failed to upsert document into pgvector",
                    details={"document_id": str(doc_id), "error": str(exc)},
                ) from exc

        created = not existed
        logger.info(
            "pgvector_upsert_document",
            document_id=str(doc_id),
            created=created,
            source=source,
        )
        return UpsertResult(id=doc_id, created=created)

    async def similarity_search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        distance: DistanceMetric | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        """Return the top-K nearest documents for ``embedding``.

        Distance operators:
        - ``cosine`` → ``<=>`` (cosine distance; score = 1 - distance)
        - ``inner_product`` → ``<#>`` (negative inner product; score = -distance)
        """
        metric = distance or self._settings.pgvector_distance
        vector_values = self._validate_embedding(embedding)

        with tracer.start_as_current_span(
            "pgvector.similarity_search",
            attributes={
                "db.system": "postgresql",
                "db.operation": "similarity_search",
                "rag.top_k": top_k,
                "rag.distance": metric,
            },
        ):
            try:
                stmt = self._build_similarity_statement(
                    embedding=vector_values,
                    top_k=top_k,
                    metric=metric,
                    filters=filters,
                )
                async with self.session() as session:
                    result = await session.execute(stmt)
                    rows = result.mappings().all()
            except RagEngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("pgvector_similarity_search_failed")
                raise RagEngineError(
                    message="pgvector similarity search failed",
                    details={"error": str(exc)},
                ) from exc

        hits = [
            VectorSearchHit(
                id=row["id"],
                content=row["content"],
                source=row["source"],
                score=float(row["score"]),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        ]
        logger.info("pgvector_similarity_search", hits=len(hits), top_k=top_k, distance=metric)
        return hits

    def _build_similarity_statement(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metric: DistanceMetric,
        filters: dict[str, Any] | None,
    ) -> Select[Any]:
        """Compose an ORDER BY distance LIMIT statement."""
        embedding_col = self._documents.c.embedding

        if metric == "cosine":
            distance_expr = embedding_col.cosine_distance(embedding)
            score_expr = (1 - distance_expr).label("score")
            order_expr = distance_expr
        elif metric == "inner_product":
            distance_expr = embedding_col.max_inner_product(embedding)
            score_expr = (-distance_expr).label("score")
            order_expr = distance_expr
        else:
            raise RagEngineError(
                message=f"Unsupported distance metric: {metric}",
                details={"distance": metric},
            )

        stmt: Select[Any] = (
            select(
                self._documents.c.id,
                self._documents.c.content,
                self._documents.c.source,
                self._documents.c.metadata,
                score_expr,
            )
            .order_by(order_expr)
            .limit(top_k)
        )

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(self._documents.c.metadata.contains({key: value}))

        return stmt
