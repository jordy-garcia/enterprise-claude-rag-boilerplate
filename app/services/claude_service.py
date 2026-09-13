"""Claude orchestration service: RAG injection + Bedrock Messages API."""

from __future__ import annotations

from typing import Any

from app.core.bedrock_client import BedrockClient
from app.core.config import Settings
from app.core.exceptions import ValidationAppError
from app.core.logging import get_logger
from app.core.telemetry import get_tracer
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    UsageMetadata,
)
from app.services.rag_engine import BaseRagEngine

logger = get_logger(__name__)
tracer = get_tracer(__name__)

_RAG_CONTEXT_TEMPLATE = (
    "Use the following retrieved enterprise context to answer the user. "
    "If the context is insufficient, say so explicitly.\n\n"
    "<retrieved_context>\n{context}\n</retrieved_context>\n\n"
    "User question:\n{question}"
)


class ClaudeService:
    """Coordinates prompt formatting, optional RAG enrichment, and Bedrock calls."""

    def __init__(
        self,
        settings: Settings,
        bedrock_client: BedrockClient,
        rag_engine: BaseRagEngine,
    ) -> None:
        self._settings = settings
        self._bedrock = bedrock_client
        self._rag = rag_engine

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate an assistant reply for the given chat request."""
        with tracer.start_as_current_span(
            "claude.generate",
            attributes={"rag.enabled": request.use_rag},
        ):
            messages = self._to_bedrock_messages(request.messages)
            rag_context_injected = False

            if request.use_rag:
                last_user_content = request.messages[-1].content
                context = await self._rag.build_context(
                    last_user_content,
                    top_k=self._settings.rag_top_k,
                )
                if context:
                    messages[-1] = {
                        "role": MessageRole.USER.value,
                        "content": _RAG_CONTEXT_TEMPLATE.format(
                            context=context,
                            question=last_user_content,
                        ),
                    }
                    rag_context_injected = True
                    logger.info("rag_context_injected", context_chars=len(context))
                else:
                    logger.info("rag_enabled_no_context")

            system_prompt = request.system_prompt or self._settings.claude_system_prompt

            raw_response = await self._bedrock.invoke_claude(
                messages=messages,
                system=system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return self._build_response(
                raw_response=raw_response,
                use_rag=request.use_rag,
                rag_context_injected=rag_context_injected,
            )

    def _to_bedrock_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert Pydantic chat messages to Claude 3 Messages API dicts."""
        if not messages:
            raise ValidationAppError("At least one message is required.")

        return [{"role": message.role.value, "content": message.content} for message in messages]

    def _build_response(
        self,
        *,
        raw_response: dict[str, Any],
        use_rag: bool,
        rag_context_injected: bool,
    ) -> ChatResponse:
        """Map Bedrock Claude response payload into our API schema."""
        content_blocks = raw_response.get("content") or []
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))

        usage_raw = raw_response.get("usage") or {}
        usage = UsageMetadata(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )

        return ChatResponse(
            content="".join(text_parts).strip(),
            model_id=str(raw_response.get("model") or self._settings.bedrock_model_id),
            use_rag=use_rag,
            rag_context_injected=rag_context_injected,
            stop_reason=raw_response.get("stop_reason"),
            usage=usage,
            metadata={
                "id": raw_response.get("id"),
                "type": raw_response.get("type"),
                "role": raw_response.get("role"),
            },
        )
