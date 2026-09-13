"""OpenTelemetry tracing setup for FastAPI and outbound HTTP clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.semconv.resource import ResourceAttributes

from app.core.config import Settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)

_provider: TracerProvider | None = None


def setup_telemetry(settings: Settings) -> TracerProvider | None:
    """Initialize the global TracerProvider and instrument HTTPX.

    FastAPI instrumentation is applied later via ``instrument_fastapi`` once
    the app instance exists.
    """
    global _provider  # noqa: PLW0603

    if settings.otel_sdk_disabled or settings.otel_traces_exporter == "none":
        logger.info("otel_disabled", reason="OTEL_SDK_DISABLED or exporter=none")
        return None

    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: settings.otel_service_name,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_traces_exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    AsyncPGInstrumentor().instrument()
    _provider = provider

    logger.info(
        "otel_initialized",
        exporter=settings.otel_traces_exporter,
        endpoint=settings.otel_exporter_otlp_endpoint,
    )
    return provider


def instrument_fastapi(app: FastAPI, settings: Settings) -> None:
    """Attach OpenTelemetry middleware to a FastAPI application."""
    if settings.otel_sdk_disabled or settings.otel_traces_exporter == "none":
        return

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,/health,/ready,/docs,/redoc,/openapi.json",
    )
    logger.info("otel_fastapi_instrumented")


def shutdown_telemetry() -> None:
    """Flush and shut down the TracerProvider."""
    global _provider  # noqa: PLW0603
    if _provider is not None:
        _provider.shutdown()
        _provider = None
        logger.info("otel_shutdown")


def get_tracer(name: str = "app") -> trace.Tracer:
    """Return a tracer bound to the global provider."""
    return trace.get_tracer(name)
