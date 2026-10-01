-- Hybrid search: per-chunk language configuration + full-text index.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS fts_config regconfig NOT NULL DEFAULT 'simple'::regconfig;

UPDATE chunks c
SET fts_config = CASE d.language
        WHEN 'fr' THEN 'french'::regconfig
        WHEN 'en' THEN 'english'::regconfig
        ELSE 'simple'::regconfig
    END
FROM documents d
WHERE d.document_id = c.document_id;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS text_search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector(fts_config, coalesce(section_title, '')), 'A')
        || setweight(to_tsvector(fts_config, coalesce(text, '')), 'B')
    ) STORED;

CREATE INDEX IF NOT EXISTS chunks_text_search_idx
    ON chunks USING gin (text_search);
