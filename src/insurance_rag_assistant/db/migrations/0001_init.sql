CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id    text PRIMARY KEY,
    title          text NOT NULL,
    document_type  text NOT NULL,
    coverage       text NOT NULL,
    jurisdiction   text NOT NULL,
    language       text NOT NULL CHECK (language IN ('en', 'fr')),
    version        text NOT NULL,
    effective_date date NOT NULL,
    status         text NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'archived', 'draft')),
    source_path    text
);

CREATE INDEX IF NOT EXISTS documents_filters_idx
    ON documents (status, coverage, jurisdiction, language);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      text PRIMARY KEY,
    document_id   text NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    section_title text NOT NULL,
    section_path  text[] NOT NULL,
    chunk_index   integer NOT NULL CHECK (chunk_index >= 0),
    page_start    integer,
    page_end      integer,
    text          text NOT NULL,
    content_hash  char(64) NOT NULL,
    embedding     vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS kb_versions (
    kb_version      text PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),
    embedding_model text NOT NULL,
    chunk_count     integer NOT NULL
);
