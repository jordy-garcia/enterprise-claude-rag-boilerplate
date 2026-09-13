"""Chat generation endpoints (API v1), including SSE streaming."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import ClaudeServiceDep
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/completions",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Bedrock invocation failure"},
    },
    summary="Generate a Claude completion (optional RAG via use_rag)",
)
async def create_chat_completion(
    payload: ChatRequest,
    claude_service: ClaudeServiceDep,
) -> ChatResponse:
    """Generate an assistant response from conversation history.

    Set ``use_rag=true`` to retrieve enterprise context via the injected
    ``BaseRagEngine`` before calling Claude on AWS Bedrock.
    """
    return await claude_service.generate(payload)


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Bedrock invocation failure"},
    },
    summary="Stream a Claude completion via Server-Sent Events (SSE)",
)
async def create_chat_stream(
    payload: ChatRequest,
    claude_service: ClaudeServiceDep,
) -> StreamingResponse:
    """Stream Claude tokens in real time (``text/event-stream``).

    Optional RAG runs before the first token. OpenTelemetry span
    ``claude.generate_stream`` and structured logs capture total stream
    ``execution_time`` through the final ``done`` / ``error`` event.
    """

    async def event_generator() -> AsyncIterator[str]:
        async for frame in claude_service.generate_claude_stream(payload):
            yield frame

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
