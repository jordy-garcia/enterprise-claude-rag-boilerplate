"""Chat / RAG endpoint tests with Bedrock mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.api.dependencies import get_bedrock_client, get_pgvector_client, get_rag_engine
from app.db.pgvector_client import UpsertResult
from app.services.rag_engine import MockRagEngine, RetrievedChunk


@pytest.mark.asyncio
async def test_chat_completion_without_rag(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "use_rag": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "Hello from mocked Claude."
    assert payload["use_rag"] is False
    assert payload["rag_context_injected"] is False
    assert payload["usage"]["input_tokens"] == 12
    assert payload["usage"]["output_tokens"] == 8
    assert payload["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_chat_completion_with_rag_injects_context(
    client: AsyncClient,
    app: Any,
) -> None:
    rag = MockRagEngine(
        corpus=[
            RetrievedChunk(
                content="Enterprise RAG uses pgvector cosine distance for retrieval.",
                source="docs/rag.md",
                score=0.97,
            )
        ]
    )
    app.dependency_overrides[get_rag_engine] = lambda: rag

    response = await client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "How does enterprise RAG retrieve context?",
                }
            ],
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["use_rag"] is True
    assert payload["rag_context_injected"] is True
    assert payload["content"] == "Hello from mocked Claude."

    bedrock = app.dependency_overrides[get_bedrock_client]()
    assert isinstance(bedrock.invoke_claude, AsyncMock)
    bedrock.invoke_claude.assert_awaited_once()
    call_kwargs = bedrock.invoke_claude.await_args.kwargs
    user_content = call_kwargs["messages"][-1]["content"]
    assert "retrieved_context" in user_content
    assert "pgvector cosine distance" in user_content


@pytest.mark.asyncio
async def test_rag_completions_always_enables_retrieval(
    client: AsyncClient,
    app: Any,
) -> None:
    rag = MockRagEngine(
        corpus=[
            RetrievedChunk(
                content="Dedicated /rag path always injects context.",
                source="docs/rag-api.md",
                score=0.99,
            )
        ]
    )
    app.dependency_overrides[get_rag_engine] = lambda: rag

    response = await client.post(
        "/api/v1/rag/completions",
        json={
            "messages": [{"role": "user", "content": "What does the RAG path do?"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["use_rag"] is True
    assert payload["rag_context_injected"] is True

    bedrock = app.dependency_overrides[get_bedrock_client]()
    user_content = bedrock.invoke_claude.await_args.kwargs["messages"][-1]["content"]
    assert "Dedicated /rag path" in user_content


@pytest.mark.asyncio
async def test_ingest_document_requires_pgvector(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/rag/documents",
        json={
            "content": "A knowledge chunk",
            "source": "docs/example.md",
            "metadata": {"team": "platform"},
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_ingest_document_upserts_with_embedding(
    client: AsyncClient,
    app: Any,
    mock_pgvector_client: AsyncMock,
) -> None:
    doc_id = uuid4()
    mock_pgvector_client.upsert_document = AsyncMock(
        return_value=UpsertResult(id=doc_id, created=True)
    )
    app.dependency_overrides[get_pgvector_client] = lambda: mock_pgvector_client

    response = await client.post(
        "/api/v1/rag/documents",
        json={
            "content": "A knowledge chunk about Bedrock",
            "source": "docs/bedrock.md",
            "metadata": {"team": "platform"},
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == str(doc_id)
    assert payload["created"] is True
    assert payload["embedding_dimensions"] == 1024
    assert payload["source"] == "docs/bedrock.md"

    bedrock = app.dependency_overrides[get_bedrock_client]()
    bedrock.embed_text.assert_awaited_once()
    mock_pgvector_client.upsert_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_validation_rejects_empty_messages(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/completions",
        json={"messages": [], "use_rag": False},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "request_validation_error"


@pytest.mark.asyncio
async def test_chat_validation_requires_final_user_turn(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "assistant", "content": "I speak first"}],
            "use_rag": False,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "request_validation_error"


@pytest.mark.asyncio
async def test_chat_stream_sse_tokens(
    client: AsyncClient,
    app: Any,
) -> None:
    """SSE endpoint streams token events from a mocked Bedrock stream."""

    async def fake_claude_stream(**_kwargs: Any) -> Any:
        for part in ("Hello", " ", "from", " ", "stream"):
            yield part

    bedrock = app.dependency_overrides[get_bedrock_client]()
    bedrock.invoke_claude_stream = fake_claude_stream

    async with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Stream please"}],
            "use_rag": False,
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    assert "event: meta" in body
    assert "event: token" in body
    assert '"content": "Hello"' in body
    assert '"content": "stream"' in body
    assert "event: done" in body
    assert '"execution_time"' in body
    assert "event: error" not in body


@pytest.mark.asyncio
async def test_chat_stream_with_rag_includes_meta_flag(
    client: AsyncClient,
    app: Any,
) -> None:
    async def fake_claude_stream(**_kwargs: Any) -> Any:
        yield "ok"

    rag = MockRagEngine(
        corpus=[
            RetrievedChunk(
                content="Streaming works with retrieved pgvector context.",
                source="docs/stream.md",
                score=0.95,
            )
        ]
    )
    app.dependency_overrides[get_rag_engine] = lambda: rag
    bedrock = app.dependency_overrides[get_bedrock_client]()
    bedrock.invoke_claude_stream = fake_claude_stream

    async with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Explain streaming RAG"}],
            "use_rag": True,
        },
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert '"rag_context_injected": true' in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_bedrock_failure_returns_502(
    client: AsyncClient,
    app: Any,
) -> None:
    from app.core.exceptions import BedrockInvocationError

    bedrock = app.dependency_overrides[get_bedrock_client]()
    bedrock.invoke_claude = AsyncMock(side_effect=BedrockInvocationError("simulated failure"))

    response = await client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "boom"}],
            "use_rag": False,
        },
    )

    assert response.status_code == 502
    payload = response.json()
    assert payload["error_code"] == "bedrock_invocation_error"
