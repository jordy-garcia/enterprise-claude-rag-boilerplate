"""Async-friendly AWS Bedrock client wrapping the synchronous boto3 SDK."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings
from app.core.exceptions import BedrockInvocationError
from app.core.logging import get_logger
from app.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class BedrockClient:
    """Injectable Bedrock Runtime client.

    ``boto3`` is blocking; every network call is offloaded with
    ``asyncio.to_thread`` so FastAPI's event loop stays responsive under load.
    OpenTelemetry spans wrap each invocation for distributed latency tracing.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: BaseClient | None = None

    def _build_client(self) -> BaseClient:
        """Create a bedrock-runtime client using settings credentials/profile."""
        session_kwargs: dict[str, Any] = {"region_name": self._settings.aws_region}

        if self._settings.aws_profile:
            session_kwargs["profile_name"] = self._settings.aws_profile
        elif self._settings.aws_access_key_id and self._settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = self._settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self._settings.aws_secret_access_key

        session = boto3.Session(**session_kwargs)
        return session.client("bedrock-runtime")

    @property
    def client(self) -> BaseClient:
        """Lazy-initialized boto3 bedrock-runtime client (singleton per instance)."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _invoke_sync(
        self,
        *,
        body: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous InvokeModel call (runs inside a worker thread)."""
        resolved_model_id = model_id or self._settings.bedrock_model_id
        try:
            response = self.client.invoke_model(
                modelId=resolved_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw_body: bytes = response["body"].read()
            parsed: dict[str, Any] = json.loads(raw_body)
            return parsed
        except (ClientError, BotoCoreError, json.JSONDecodeError, KeyError) as exc:
            logger.exception("bedrock_invoke_failed", model_id=resolved_model_id)
            raise BedrockInvocationError(
                message=f"Bedrock InvokeModel failed: {exc}",
                details={"model_id": resolved_model_id},
            ) from exc

    async def invoke_model(
        self,
        *,
        body: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke Bedrock without blocking the asyncio event loop."""
        resolved_model_id = model_id or self._settings.bedrock_model_id
        with tracer.start_as_current_span(
            "bedrock.invoke_model",
            attributes={
                "gen_ai.system": "aws.bedrock",
                "gen_ai.request.model": resolved_model_id,
                "peer.service": "aws.bedrock",
            },
        ):
            return await asyncio.to_thread(
                self._invoke_sync,
                body=body,
                model_id=resolved_model_id,
            )

    async def invoke_claude(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a Claude 3 Messages API payload via Bedrock."""
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self._settings.bedrock_max_tokens,
            "temperature": temperature
            if temperature is not None
            else self._settings.bedrock_temperature,
            "top_p": top_p if top_p is not None else self._settings.bedrock_top_p,
            "messages": messages,
        }
        if system:
            body["system"] = system

        with tracer.start_as_current_span("bedrock.invoke_claude"):
            return await self.invoke_model(body=body, model_id=model_id)

    async def embed_text(
        self,
        text: str,
        *,
        model_id: str | None = None,
        dimensions: int | None = None,
    ) -> list[float]:
        """Generate an embedding vector via Amazon Titan Embed on Bedrock.

        Titan Embed Text v2 accepts ``dimensions`` to match ``PGVECTOR_EMBEDDING_DIM``.
        """
        resolved_model_id = model_id or self._settings.bedrock_embedding_model_id
        resolved_dimensions = dimensions or self._settings.pgvector_embedding_dim

        body: dict[str, Any] = {
            "inputText": text,
            "dimensions": resolved_dimensions,
            "normalize": True,
        }

        with tracer.start_as_current_span(
            "bedrock.embed_text",
            attributes={
                "gen_ai.request.model": resolved_model_id,
                "embedding.dimensions": resolved_dimensions,
            },
        ):
            raw = await self.invoke_model(body=body, model_id=resolved_model_id)

        embedding = raw.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BedrockInvocationError(
                message="Bedrock embedding response missing 'embedding' vector",
                details={"model_id": resolved_model_id},
            )
        return [float(value) for value in embedding]
