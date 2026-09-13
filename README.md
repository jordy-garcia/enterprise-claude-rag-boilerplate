# enterprise-claude-rag-boilerplate

Production-ready FastAPI starter for Anthropic Claude on **AWS Bedrock**, with **pgvector** RAG, **structured JSON logging**, and **OpenTelemetry** tracing.

**Author:** Jordy Garcia | Senior Software Architect

Managed with [uv](https://docs.astral.sh/uv/) for environments, lockfiles, and packaging.

## Tech stack

| Layer | Choice |
| --- | --- |
| Package / env manager | uv (`pyproject.toml` + `uv.lock`) |
| API | FastAPI (async) |
| Validation / settings | Pydantic v2 + pydantic-settings |
| AWS SDK | boto3 via `asyncio.to_thread` |
| Vector store | PostgreSQL + pgvector (`asyncpg` / SQLAlchemy async) |
| Logging | structlog (JSON for CloudWatch / Datadog) |
| Tracing | OpenTelemetry (FastAPI, HTTPX, asyncpg + Bedrock spans) |
| Tests | pytest + pytest-asyncio + httpx |
| Server | Uvicorn |
| Python | ≥ 3.11 |

## Project layout

```text
app/
├── api/
│   ├── dependencies.py
│   └── v1/
│       ├── chat.py          # Chat completions
│       └── rag.py           # RAG completions + document ingest
├── core/
│   ├── config.py
│   ├── bedrock_client.py
│   ├── logging.py           # JSON structured logger + request middleware
│   ├── telemetry.py         # OpenTelemetry TracerProvider setup
│   └── exceptions.py
├── db/
│   ├── pgvector_client.py   # Async pool, similarity search, upsert
│   └── schema.sql           # Reference DDL for documents + vector index
├── models/schemas.py
├── services/
│   ├── claude_service.py
│   └── rag_engine.py        # PgVectorRagEngine (+ MockRagEngine for tests)
└── main.py                  # CORS, logging, OTEL, /health, /ready
tests/
├── conftest.py
├── test_health.py
└── test_chat.py
```

## Local setup (uv)

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate it
source .venv/bin/activate

# 3. Install runtime + dev dependencies
uv sync --extra dev

# 4. Configure environment
cp .env.example .env
# Edit AWS credentials; keep RAG_ENGINE=mock until Postgres is ready

# 5. Run the API with hot reload
uv run uvicorn app.main:app --reload
```

Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Useful uv commands

```bash
uv add <package>
uv add --dev <package>
uv remove <package>
uv lock
uv run pytest
uv run uvicorn app.main:app --reload
```

## Tests

Bedrock is mocked in the suite — no AWS credentials required:

```bash
uv sync --extra dev
uv run pytest
uv run pytest -v --tb=short
```

## Enable production RAG (pgvector)

1. Run PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.
2. Apply [`app/db/schema.sql`](app/db/schema.sql) (adjust `vector(1024)` if you change dimensions).
3. Set in `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
PGVECTOR_TABLE=documents
PGVECTOR_EMBEDDING_DIM=1024
PGVECTOR_DISTANCE=cosine   # or inner_product
RAG_ENGINE=pgvector
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

Pipeline when `use_rag=true`:

1. `BedrockClient.embed_text()` → Titan embedding (async, traced).
2. `PgVectorClient.similarity_search()` → `<=>` (cosine) or `<#>` (inner product).
3. Top-K chunks injected into the Claude Messages prompt.
4. `BedrockClient.invoke_claude()` generates the answer.

## Observability

### Structured JSON logging

`app/core/logging.py` configures structlog with:

- `timestamp` (ISO-8601 UTC)
- `level`
- `trace_id` / `span_id` (from the active OpenTelemetry span)
- `path`, `method`, `status_code`, `execution_time` (ms) via `RequestLoggingMiddleware`

Compatible with CloudWatch Logs Insights and Datadog log pipelines (`LOG_JSON=true`).

### OpenTelemetry

`app/core/telemetry.py` initializes a `TracerProvider` and instruments:

- FastAPI (inbound requests)
- HTTPX (outbound HTTP)
- asyncpg (SQL)
- Manual spans around Bedrock `invoke_model` / `embed_text` / Claude generation

Export via OTLP (`OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://localhost:4317`). Disable for local/tests with `OTEL_SDK_DISABLED=true`.

## API overview

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (Postgres when `RAG_ENGINE=pgvector`) |
| `POST` | `/api/v1/chat/completions` | Claude completion; optional `use_rag` |
| `POST` | `/api/v1/rag/completions` | Claude completion with RAG always on |
| `POST` | `/api/v1/rag/documents` | Upsert a document embedding into pgvector |

```bash
# Optional RAG via flag
curl -s http://127.0.0.1:8000/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "How does cosine search work in this stack?"}
    ],
    "use_rag": true
  }'

# Dedicated RAG path
curl -s http://127.0.0.1:8000/api/v1/rag/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Summarize our retrieval strategy"}
    ]
  }'

# Ingest a document (requires RAG_ENGINE=pgvector)
curl -s http://127.0.0.1:8000/api/v1/rag/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "pgvector stores embeddings for enterprise RAG.",
    "source": "docs/rag.md",
    "metadata": {"team": "platform"}
  }'
```

## Docker

```bash
docker build -t enterprise-claude-rag-boilerplate .
docker run --rm -p 8000:8000 --env-file .env enterprise-claude-rag-boilerplate
```

## Environment variables

See [`.env.example`](.env.example). Highlights:

| Variable | Purpose |
| --- | --- |
| `BEDROCK_MODEL_ID` | Claude model on Bedrock |
| `BEDROCK_EMBEDDING_MODEL_ID` | Titan (or other) embedding model |
| `DATABASE_URL` | SQLAlchemy async URL (`postgresql+asyncpg://...`) |
| `RAG_ENGINE` | `pgvector` or `mock` |
| `PGVECTOR_DISTANCE` | `cosine` (`<=>`) or `inner_product` (`<#>`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector |
| `LOG_JSON` | JSON logs for CloudWatch / Datadog |

Prefer IAM roles in AWS over long-lived access keys.

## Author

**Jordy Garcia** | Senior Software Architect

## License

See [LICENSE](LICENSE).
