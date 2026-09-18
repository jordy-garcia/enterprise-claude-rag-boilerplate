# Enterprise Claude RAG Boilerplate

A production-shaped FastAPI service for **Claude on AWS Bedrock** with an optional **pgvector RAG** path, designed so platform and product teams can ship enterprise assistants without reinventing authz-adjacent plumbing, retrieval wiring, or observability every time.

**Author:** Jordy Garcia | Senior Software Architect

---

## The problem this solves

Most “Claude + RAG” demos fail the jump to enterprise for predictable reasons:

1. **Vendor lock-in at the wrong layer** — calling Anthropic’s public API directly when the organization already standardized on Bedrock (IAM, VPC endpoints, model access, audit).
2. **Retrieval bolted on as an afterthought** — embeddings, distance metrics, and prompt injection living inline in route handlers, so swapping stores or turning RAG off becomes a rewrite.
3. **No operability story** — unstructured logs, no correlation between a slow chat and a slow embed/search/LLM hop, and health checks that lie when Postgres is down.
4. **Local/prod divergence** — developers cannot exercise the API without AWS credentials *and* a vector DB on day one, so the “starter” never gets adopted.

This repository is a **bounded, replaceable skeleton**: a chat API that can answer with or without retrieved context, runs against Bedrock, stores vectors in Postgres when you need them, and emits logs/traces that survive a CloudWatch or Datadog pipeline. It is intentionally *not* a full knowledge platform (chunking pipelines, ACL-aware retrieval, eval harnesses). Those belong as adjacent services; this service owns the request path.

---

## Design goals

| Goal | What it means here |
| --- | --- |
| Ship the request path first | Completions, optional RAG, SSE streaming, document upsert — not a document portal |
| Keep I/O boundaries explicit | Routers → services → Bedrock / RAG / DB; no SDK calls in handlers |
| Fail closed on readiness | `/ready` reflects Postgres when RAG is real; `/health` stays cheap for liveness |
| Develop without the full stack | `RAG_ENGINE=mock` + mocked Bedrock in tests |
| Observability as a default | JSON logs with `trace_id` / `span_id`; OTEL across FastAPI, SQL, and Bedrock |

---

## Architecture

```text
                    ┌─────────────────────────────────────────┐
                    │              FastAPI (main)              │
                    │  CORS · request logging · OTEL · DI      │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        /api/v1/chat          /api/v1/rag            /health · /ready
              │                     │
              └──────────┬──────────┘
                         ▼
                 ClaudeService
            (prompt prep · RAG flag · stream)
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      BaseRagEngine              BedrockClient
   (mock | pgvector)         (embed · invoke · stream)
            │                         │
            ▼                         ▼
     PgVectorClient              AWS Bedrock
     (asyncpg pool)           Claude + Titan embeds
```

### Request path

1. Validate the conversation contract (Pydantic; last turn must be `user`).
2. If RAG is enabled, embed the latest user text, run similarity search, format a context block, and inject it into the Messages payload.
3. Call Claude on Bedrock (sync completion or SSE token stream).
4. Return a typed response (or stream frames) with usage/timing metadata where available.

Chat exposes RAG as a **per-request flag** (`use_rag`). The dedicated RAG routes force retrieval on and own ingest. That split keeps a general chat surface usable for non-grounded flows while giving product teams a clear “always grounded” API.

---

## Architectural decisions

### Bedrock instead of the Anthropic public API

Enterprise constraints usually win: model enablement, IAM roles (prefer over long-lived keys), PrivateLink, and centralized spend. The service talks to Bedrock via `boto3`, wrapped so call sites stay async-friendly (`asyncio.to_thread`) and individually traced. Swapping model IDs is configuration, not a code fork.

### Pluggable RAG behind an interface

`BaseRagEngine` defines `retrieve` / `build_context`. Production uses `PgVectorRagEngine`; local and CI use `MockRagEngine`. The chat service depends on the abstraction, not on SQL. Replacing pgvector with OpenSearch, a managed vector service, or a sidecar later should not rewrite handlers.

**Why Postgres + pgvector (default prod engine):** many enterprises already run Postgres for OLTP. Co-locating vectors avoids a second operational plane early on. Distance metric (`cosine` / `inner_product`) and embedding dimension are settings — change Titan (or another embed model) and align `PGVECTOR_EMBEDDING_DIM` + `schema.sql`.

**What we deliberately do not do in-process:** heavy document chunking, ACL filtering, hybrid BM25+vector, or re-ranking. Ingest accepts a content payload and upserts an embedding; upstream systems own how documents are split and authorized.

### Async FastAPI with a sync AWS SDK

FastAPI and asyncpg are native async; Bedrock’s SDK is not. Isolating Bedrock behind a client and offloading to threads keeps the event loop responsive without pretending boto3 is async. Spans wrap embed and invoke so latency attribution stays honest.

### Structured logs + OpenTelemetry from day zero

Debugging “chat was slow” without knowing whether embed, ANN search, or generation dominated is expensive. structlog emits JSON (CloudWatch / Datadog–friendly) and attaches active `trace_id` / `span_id`. OTEL instruments FastAPI, HTTPX, asyncpg, plus manual Bedrock spans. Disable the SDK locally with `OTEL_SDK_DISABLED=true` so tests do not require a collector.

### Health vs readiness

- **`/health`** — process is up (orchestrator liveness).
- **`/ready`** — dependencies required to serve traffic; when `RAG_ENGINE=pgvector`, Postgres connectivity is checked.

Lying readiness is worse than a failed deploy.

### Packaging and runtime

- **[uv](https://docs.astral.sh/uv/)** for lockfiles and reproducible envs (`pyproject.toml` + `uv.lock`).
- **Multi-stage Docker** — builder with uv, slim runtime, non-root user, image `HEALTHCHECK` on `/health`.
- **Python ≥ 3.11**, Pydantic v2 settings — typed config, fail fast on bad env.

### Testing stance

The suite mocks Bedrock. No AWS credentials required to prove routing, validation, and orchestration. RAG can stay on `mock` until a real Postgres is available. That keeps CI cheap and the contract of the API honest.

---

## Tech stack

| Concern | Choice | Rationale (short) |
| --- | --- | --- |
| API | FastAPI + Uvicorn | Async request path, OpenAPI for free |
| Config / contracts | Pydantic v2 + pydantic-settings | Typed boundaries at the edge |
| LLM / embeds | AWS Bedrock (Claude + Titan) | Enterprise control plane |
| Vectors | PostgreSQL + pgvector | Familiar ops, good enough ANN to start |
| Logging | structlog (JSON) | Machine-parseable, correlatable |
| Tracing | OpenTelemetry (OTLP) | Vendor-neutral export |
| Tooling | uv, pytest, ruff, mypy | Reproducible + strict |

---

## Project layout

Boundaries match the architecture: **API** is thin, **services** own orchestration, **core** owns cross-cutting I/O and observability, **db** owns the vector store.

```text
app/
├── api/
│   ├── dependencies.py      # DI: settings, Bedrock, RAG engine
│   └── v1/
│       ├── chat.py          # Completions + SSE (optional RAG)
│       └── rag.py           # Always-on RAG + document upsert
├── core/
│   ├── config.py            # Environment → Settings
│   ├── bedrock_client.py    # Bedrock invoke / embed / stream
│   ├── logging.py           # JSON logger + request middleware
│   ├── telemetry.py         # TracerProvider + instrumentations
│   └── exceptions.py        # Domain errors → HTTP mapping
├── db/
│   ├── pgvector_client.py   # Pool, similarity search, upsert
│   └── schema.sql           # Reference DDL + vector index
├── models/schemas.py        # Request/response contracts
├── services/
│   ├── claude_service.py    # Prompt prep + generation
│   └── rag_engine.py        # BaseRagEngine + pgvector / mock
└── main.py                  # Lifespan, middleware, probes, routers
tests/                       # Health + chat; Bedrock mocked
```

---

## RAG pipeline (production)

When retrieval is enabled (`use_rag=true` or `/api/v1/rag/completions`):

1. `BedrockClient.embed_text()` — Titan embedding (traced).
2. `PgVectorClient.similarity_search()` — `<=>` (cosine) or `<#>` (inner product).
3. Top-K chunks formatted and injected into the Claude Messages prompt.
4. `BedrockClient.invoke_claude()` (or stream) produces the answer.

Enable with Postgres + pgvector:

1. Apply [`app/db/schema.sql`](app/db/schema.sql) (adjust `vector(1024)` if dimensions change).
2. Set:

```bash
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
PGVECTOR_TABLE=documents
PGVECTOR_EMBEDDING_DIM=1024
PGVECTOR_DISTANCE=cosine   # or inner_product
RAG_ENGINE=pgvector
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

---

## Observability

**Logs** (`app/core/logging.py`): ISO-8601 UTC timestamp, level, `trace_id` / `span_id`, plus `path`, `method`, `status_code`, `execution_time` via `RequestLoggingMiddleware`. Set `LOG_JSON=true` for CloudWatch Logs Insights / Datadog.

**Traces** (`app/core/telemetry.py`): FastAPI, HTTPX, asyncpg, and manual spans around Bedrock embed/invoke and Claude generation. Export OTLP to `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`).

---

## API surface

| Method | Path | Role |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Readiness (Postgres when `RAG_ENGINE=pgvector`) |
| `POST` | `/api/v1/chat/completions` | Claude completion; RAG optional |
| `POST` | `/api/v1/chat/stream` | SSE token stream (`text/event-stream`) |
| `POST` | `/api/v1/rag/completions` | Completion with RAG always on |
| `POST` | `/api/v1/rag/documents` | Upsert document + embedding |

```bash
# Optional RAG
curl -s http://127.0.0.1:8000/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "How does cosine search work in this stack?"}
    ],
    "use_rag": true
  }'

# Always-on RAG
curl -s http://127.0.0.1:8000/api/v1/rag/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Summarize our retrieval strategy"}
    ]
  }'

# SSE stream
curl -N http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "messages": [{"role": "user", "content": "Stream a short greeting"}],
    "use_rag": false
  }'

# Ingest (requires RAG_ENGINE=pgvector)
curl -s http://127.0.0.1:8000/api/v1/rag/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "pgvector stores embeddings for enterprise RAG.",
    "source": "docs/rag.md",
    "metadata": {"team": "platform"}
  }'
```

---

## Local setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev

cp .env.example .env
# Configure AWS (prefer profile/role). Keep RAG_ENGINE=mock until Postgres is ready.

uv run uvicorn app.main:app --reload
```

OpenAPI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

```bash
uv add <package>
uv lock
uv run pytest
uv run uvicorn app.main:app --reload
```

### Tests

```bash
uv sync --extra dev
uv run pytest
```

Bedrock is mocked — no cloud credentials required for the default suite.

### Docker

```bash
docker build -t enterprise-claude-rag-boilerplate .
docker run --rm -p 8000:8000 --env-file .env enterprise-claude-rag-boilerplate
```

---

## Configuration

See [`.env.example`](.env.example). High-signal variables:

| Variable | Purpose |
| --- | --- |
| `BEDROCK_MODEL_ID` | Claude model on Bedrock |
| `BEDROCK_EMBEDDING_MODEL_ID` | Embedding model (e.g. Titan) |
| `DATABASE_URL` | SQLAlchemy async URL (`postgresql+asyncpg://...`) |
| `RAG_ENGINE` | `pgvector` or `mock` |
| `PGVECTOR_DISTANCE` | `cosine` (`<=>`) or `inner_product` (`<#>`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector |
| `OTEL_SDK_DISABLED` | Disable tracing (local/CI) |
| `LOG_JSON` | JSON logs for log platforms |

Prefer IAM roles in AWS over long-lived access keys.

---

## Non-goals (for clarity)

This boilerplate does **not** attempt to be:

- A multi-tenant document CMS or ACL-aware search layer
- An evaluation / red-team harness for RAG quality
- A workflow orchestrator (agents, tools, multi-step planners)
- A replacement for your org’s API gateway, authN/Z, or secrets manager

Use it as the **generation + retrieval request service**, and compose those concerns around it.

---

## Author

**Jordy Garcia** | Senior Software Architect

## License

See [LICENSE](LICENSE).
