"""Shared pytest fixtures: async client, mocked Bedrock, mock RAG."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.pgvector_client import UpsertResult

# Force safe defaults before Settings / app are imported.
os.environ["RAG_ENGINE"] = "mock"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["LOG_JSON"] = "false"
os.environ["APP_ENV"] = "development"
os.environ["APP_DEBUG"] = "true"


@pytest.fixture
def mock_bedrock_response() -> dict[str, Any]:
    """Canonical Claude Messages API payload returned by Bedrock."""
    return {
        "id": "msg_test_001",
        "type": "message",
        "role": "assistant",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "content": [{"type": "text", "text": "Hello from mocked Claude."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


@pytest.fixture
def app(mock_bedrock_response: dict[str, Any]) -> Iterator[Any]:
    """Build a fresh FastAPI app with Bedrock mocked (no AWS calls)."""
    from app.api.dependencies import get_bedrock_client
    from app.core.bedrock_client import BedrockClient
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()

    settings = get_settings()
    bedrock = BedrockClient(settings=settings)
    bedrock.invoke_claude = AsyncMock(return_value=mock_bedrock_response)  # type: ignore[method-assign]
    bedrock.embed_text = AsyncMock(  # type: ignore[method-assign]
        return_value=[0.1] * settings.pgvector_embedding_dim
    )
    bedrock.invoke_model = AsyncMock(return_value=mock_bedrock_response)  # type: ignore[method-assign]

    application = create_app()
    application.dependency_overrides[get_bedrock_client] = lambda: bedrock
    # Ensure readiness / DI share the same mocked client even before lifespan.
    application.state.bedrock_client = bedrock

    yield application

    application.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
async def client(app: Any) -> AsyncIterator[AsyncClient]:
    """Async HTTP client with ASGI lifespan (startup/shutdown) enabled."""
    async with app.router.lifespan_context(app):
        # Re-apply mock after lifespan replaces app.state.bedrock_client.
        from app.api.dependencies import get_bedrock_client

        bedrock = app.dependency_overrides[get_bedrock_client]()
        app.state.bedrock_client = bedrock

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as async_client:
            yield async_client


@pytest.fixture
def mock_pgvector_client() -> AsyncMock:
    """In-memory stand-in for PgVectorClient upsert/ping."""
    client = AsyncMock()
    client.embedding_dimensions = 1024
    client.ping = AsyncMock(return_value=True)
    client.upsert_document = AsyncMock(return_value=UpsertResult(id=uuid4(), created=True))
    return client
