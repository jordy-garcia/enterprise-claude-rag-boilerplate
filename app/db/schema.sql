-- Reference schema for enterprise RAG with pgvector.
-- Apply against your PostgreSQL instance before enabling RAG_ENGINE=pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Match PGVECTOR_EMBEDDING_DIM (default 1024 for Titan Embed Text v2)
    embedding   vector(1024) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Cosine distance ANN index (swap to vector_ip_ops for inner product)
CREATE INDEX IF NOT EXISTS documents_embedding_cosine_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents
    USING gin (metadata);
