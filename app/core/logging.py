"""Structured JSON logging for CloudWatch / Datadog via structlog."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp

from app.core.config import Settings


def _add_otel_trace_context(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject OpenTelemetry trace/span IDs into every log event."""
    span = trace.get_current_span()
    context = span.get_span_context()
    if context is not None and context.is_valid:
        event_dict["trace_id"] = format(context.trace_id, "032x")
        event_dict["span_id"] = format(context.span_id, "016x")
    else:
        event_dict.setdefault("trace_id", None)
        event_dict.setdefault("span_id", None)
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure stdlib + structlog for JSON (prod) or console (dev) output."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        _add_otel_trace_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quieten noisy third-party loggers in production JSON mode
    for name in ("uvicorn.access", "uvicorn.error", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING if settings.log_json else log_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger."""
    return structlog.get_logger(name)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured log per request with path and execution_time.

    For ``StreamingResponse`` (SSE), ``execution_time`` covers the full body
    drain — not only header send — so CloudWatch / Datadog see total stream
    latency.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._logger = get_logger("app.request")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        path = request.url.path
        method = request.method

        response = await call_next(request)
        status_code = response.status_code

        if isinstance(response, StreamingResponse):
            original_body_iterator = response.body_iterator

            async def timed_stream() -> AsyncIterator[Any]:
                try:
                    async for chunk in original_body_iterator:
                        yield chunk
                finally:
                    self._log_completion(
                        path=path,
                        method=method,
                        status_code=status_code,
                        started=started,
                        streaming=True,
                    )

            response.body_iterator = timed_stream()
            return response

        self._log_completion(
            path=path,
            method=method,
            status_code=status_code,
            started=started,
            streaming=False,
        )
        return response

    def _log_completion(
        self,
        *,
        path: str,
        method: str,
        status_code: int,
        started: float,
        streaming: bool,
    ) -> None:
        execution_time_ms = round((time.perf_counter() - started) * 1000, 2)
        self._logger.info(
            "request_completed",
            path=path,
            method=method,
            status_code=status_code,
            execution_time=execution_time_ms,
            streaming=streaming,
        )
