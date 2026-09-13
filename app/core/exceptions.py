"""Domain and HTTP exception hierarchy for the application."""

from typing import Any


class AppException(Exception):
    """Base application exception with HTTP-friendly metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_code: str = "internal_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class BedrockInvocationError(AppException):
    """Raised when AWS Bedrock InvokeModel fails."""

    def __init__(
        self,
        message: str = "Failed to invoke Bedrock model",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=502,
            error_code="bedrock_invocation_error",
            details=details,
        )


class RagEngineError(AppException):
    """Raised when the RAG retrieval pipeline fails."""

    def __init__(
        self,
        message: str = "RAG context retrieval failed",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=500,
            error_code="rag_engine_error",
            details=details,
        )


class ValidationAppError(AppException):
    """Raised for domain-level validation failures."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=422,
            error_code="validation_error",
            details=details,
        )


class ServiceUnavailableError(AppException):
    """Raised when a required backend (e.g. pgvector) is not available."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=503,
            error_code="service_unavailable",
            details=details,
        )
