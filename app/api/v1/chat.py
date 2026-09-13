"""Chat generation endpoints (API v1)."""

from fastapi import APIRouter, status

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
