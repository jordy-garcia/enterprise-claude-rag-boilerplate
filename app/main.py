"""FastAPI application entrypoint: telemetry, logging, routers, exception handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.v1 import chat as chat_router
from app.api.v1 import rag as rag_router
from app.core.bedrock_client import BedrockClient
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import RequestLoggingMiddleware, configure_logging, get_logger
from app.core.telemetry import instrument_fastapi, setup_telemetry, shutdown_telemetry
from app.db.pgvector_client import PgVectorClient
from app.models.schemas import ErrorResponse, HealthResponse, ReadinessCheck, ReadinessResponse
from app.services.rag_engine import MockRagEngine, PgVectorRagEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: pgvector pool, RAG engine wiring, OTEL shutdown."""
    settings = get_settings()

    app.state.settings = settings
    app.state.bedrock_client = BedrockClient(settings=settings)
    app.state.pgvector_client = None
    app.state.rag_engine = None

    if settings.rag_engine == "pgvector":
        pgvector_client = PgVectorClient(settings=settings)
        try:
            await pgvector_client.connect()
        except Exception:
            logger.exception("pgvector_startup_failed")
            await pgvector_client.close()
            raise
        app.state.pgvector_client = pgvector_client
        app.state.rag_engine = PgVectorRagEngine(
            pgvector_client=pgvector_client,
            embed_fn=app.state.bedrock_client.embed_text,
            default_top_k=settings.rag_top_k,
        )
        logger.info("rag_engine_ready", engine="pgvector")
    else:
        app.state.rag_engine = MockRagEngine()
        logger.info("rag_engine_ready", engine="mock")

    logger.info(
        "app_started",
        app_name=settings.app_name,
        env=settings.app_env,
        model=settings.bedrock_model_id,
        version=__version__,
    )
    yield

    if app.state.pgvector_client is not None:
        await app.state.pgvector_client.close()
    shutdown_telemetry()
    logger.info("app_stopped", app_name=settings.app_name)


def create_app() -> FastAPI:
    """Application factory — keeps imports testable and configuration explicit."""
    settings = get_settings()
    configure_logging(settings)
    setup_telemetry(settings)

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Enterprise starter kit for Anthropic Claude via AWS Bedrock "
            "with pgvector RAG, structured logging, and OpenTelemetry."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
    )

    # Order: last added = outermost. Request logging should wrap the stack.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)

    instrument_fastapi(application, settings)
    register_exception_handlers(application)
    register_routes(application)
    return application


async def _build_readiness(app: FastAPI) -> ReadinessResponse:
    """Evaluate dependency health for Kubernetes-style readiness probes."""
    settings = get_settings()
    checks: list[ReadinessCheck] = [
        ReadinessCheck(name="app", status="ok", detail=f"version={__version__}"),
        ReadinessCheck(
            name="rag_engine",
            status="ok",
            detail=settings.rag_engine,
        ),
    ]

    ready = True
    pgvector_client = getattr(app.state, "pgvector_client", None)
    if settings.rag_engine == "pgvector":
        if pgvector_client is None:
            checks.append(
                ReadinessCheck(
                    name="postgres",
                    status="error",
                    detail="PgVectorClient not initialized",
                )
            )
            ready = False
        else:
            ping_ok = await pgvector_client.ping()
            checks.append(
                ReadinessCheck(
                    name="postgres",
                    status="ok" if ping_ok else "error",
                    detail="SELECT 1" if ping_ok else "ping failed",
                )
            )
            if not ping_ok:
                ready = False
    else:
        checks.append(
            ReadinessCheck(
                name="postgres",
                status="skipped",
                detail="RAG_ENGINE=mock",
            )
        )

    bedrock = getattr(app.state, "bedrock_client", None)
    if isinstance(bedrock, BedrockClient):
        checks.append(
            ReadinessCheck(
                name="bedrock",
                status="ok",
                detail=f"model={settings.bedrock_model_id}",
            )
        )
    else:
        checks.append(
            ReadinessCheck(name="bedrock", status="error", detail="client missing")
        )
        ready = False

    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        app_name=settings.app_name,
        environment=settings.app_env,
        version=__version__,
        checks=checks,
    )


def register_routes(application: FastAPI) -> None:
    """Mount health, readiness, and versioned API routes."""
    settings = get_settings()

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Liveness probe",
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            environment=settings.app_env,
            version=__version__,
        )

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["health"],
        summary="Readiness probe (Postgres when RAG_ENGINE=pgvector)",
        responses={503: {"model": ReadinessResponse}},
    )
    async def readiness_check(request: Request) -> JSONResponse:
        payload = await _build_readiness(request.app)
        status_code = (
            status.HTTP_200_OK
            if payload.status == "ready"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(status_code=status_code, content=payload.model_dump())

    application.include_router(chat_router.router, prefix=settings.api_v1_prefix)
    application.include_router(rag_router.router, prefix=settings.api_v1_prefix)


def register_exception_handlers(application: FastAPI) -> None:
    """Wire domain and framework exceptions into a uniform JSON envelope."""

    @application.exception_handler(AppException)
    async def app_exception_handler(
        _request: Request,
        exc: AppException,
    ) -> JSONResponse:
        payload = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        payload = ErrorResponse(
            error_code="request_validation_error",
            message="Request validation failed",
            details={"errors": _serialize_validation_errors(exc.errors())},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("unhandled_exception", error=str(exc))
        payload = ErrorResponse(
            error_code="internal_error",
            message="An unexpected error occurred",
            details={},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )


def _serialize_validation_errors(errors: list[Any]) -> list[dict[str, Any]]:
    """Make Pydantic/FastAPI validation errors JSON-serializable."""
    serialized: list[dict[str, Any]] = []
    for error in errors:
        item = dict(error)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {key: str(value) for key, value in ctx.items()}
        serialized.append(item)
    return serialized


app = create_app()
