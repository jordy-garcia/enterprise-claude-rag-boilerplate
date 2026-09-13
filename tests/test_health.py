"""Health and readiness endpoint tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_name"] == "enterprise-claude-rag-boilerplate"
    assert "version" in payload
    assert payload["environment"] == "development"


@pytest.mark.asyncio
async def test_ready_returns_ready_with_mock_rag(client: AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    check_names = {item["name"]: item for item in payload["checks"]}
    assert check_names["app"]["status"] == "ok"
    assert check_names["rag_engine"]["detail"] == "mock"
    assert check_names["postgres"]["status"] == "skipped"
    assert check_names["bedrock"]["status"] == "ok"
