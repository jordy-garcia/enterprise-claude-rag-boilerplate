"""Async-friendly AWS Bedrock client wrapping the synchronous boto3 SDK."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
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

    def _build_claude_body(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Assemble a Claude Messages API payload for Bedrock."""
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
        return body

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
        body = self._build_claude_body(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        with tracer.start_as_current_span("bedrock.invoke_claude"):
            return await self.invoke_model(body=body, model_id=model_id)

    async def invoke_model_with_response_stream(
        self,
        *,
        body: dict[str, Any],
        model_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas from Bedrock ``InvokeModelWithResponseStream``.

        The boto3 event iterator is consumed on a worker thread; text chunks are
        forwarded to the caller through an ``asyncio.Queue`` so the FastAPI
        event loop stays non-blocking.
        """
        resolved_model_id = model_id or self._settings.bedrock_model_id
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None | BaseException] = asyncio.Queue()
        started = time.perf_counter()

        def _produce() -> None:
            try:
                response = self.client.invoke_model_with_response_stream(
                    modelId=resolved_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                event_stream = response.get("body")
                if event_stream is None:
                    raise BedrockInvocationError(
                        message="Bedrock stream response missing body",
                        details={"model_id": resolved_model_id},
                    )

                for event in event_stream:
                    chunk = event.get("chunk") if isinstance(event, dict) else None
                    if not chunk:
                        continue
                    raw_bytes = chunk.get("bytes")
                    if not raw_bytes:
                        continue
                    payload = json.loads(raw_bytes)
                    text = _extract_text_delta(payload)
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)

                loop.call_soon_threadsafe(queue.put_nowait, None)
            except (ClientError, BotoCoreError, json.JSONDecodeError, KeyError) as exc:
                logger.exception("bedrock_stream_failed", model_id=resolved_model_id)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    BedrockInvocationError(
                        message=f"Bedrock InvokeModelWithResponseStream failed: {exc}",
                        details={"model_id": resolved_model_id},
                    ),
                )
            except BaseException as exc:  # noqa: BLE001 — forward to async consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        with tracer.start_as_current_span(
            "bedrock.invoke_model_with_response_stream",
            attributes={
                "gen_ai.system": "aws.bedrock",
                "gen_ai.request.model": resolved_model_id,
                "gen_ai.operation.name": "chat_stream",
                "peer.service": "aws.bedrock",
            },
        ) as span:
            producer = asyncio.create_task(asyncio.to_thread(_produce))
            token_count = 0
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        if isinstance(item, BedrockInvocationError):
                            raise item
                        raise BedrockInvocationError(
                            message=f"Bedrock stream failed: {item}",
                            details={"model_id": resolved_model_id},
                        ) from item
                    token_count += 1
                    yield item
            finally:
                await producer
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                span.set_attribute("stream.token_events", token_count)
                span.set_attribute("stream.duration_ms", duration_ms)
                logger.info(
                    "bedrock_stream_completed",
                    model_id=resolved_model_id,
                    token_events=token_count,
                    execution_time=duration_ms,
                )

    async def invoke_claude_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream Claude text deltas via Bedrock response streaming."""
        body = self._build_claude_body(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        async for text in self.invoke_model_with_response_stream(
            body=body,
            model_id=model_id,
        ):
            yield text

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


def _extract_text_delta(payload: dict[str, Any]) -> str | None:
    """Pull plain text from a Claude Messages stream event payload."""
    event_type = payload.get("type")
    if event_type == "content_block_delta":
        delta = payload.get("delta") or {}
        if delta.get("type") == "text_delta":
            text = delta.get("text")
            return str(text) if text else None
    # Some Bedrock wrappers surface deltas under nested keys
    if "delta" in payload and isinstance(payload["delta"], dict):
        text = payload["delta"].get("text")
        if text:
            return str(text)
    return None
