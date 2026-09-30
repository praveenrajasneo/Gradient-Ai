CREATE TABLE IF NOT EXISTS companies (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    source_type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES companies(id),
    input_hash TEXT NOT NULL,
    config JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    error TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id UUID NOT NULL,
    company_id BIGINT NOT NULL REFERENCES companies(id),
    source TEXT NOT NULL REFERENCES sources(name),
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL,
    author TEXT,
    published_at TIMESTAMPTZ,
    collected_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, company_id)
);
CREATE TABLE IF NOT EXISTS aspects (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    analysis_id UUID NOT NULL REFERENCES analyses(id),
    document_id UUID NOT NULL,
    company_id BIGINT NOT NULL,
    aspect TEXT NOT NULL CHECK (aspect IN ('workload','work_life_balance','management',
        'career_growth','promotion','compensation','job_security','team_culture')),
    sentiment TEXT NOT NULL CHECK (sentiment IN ('positive','negative','neutral','mixed')),
    pain_point TEXT,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    evidence JSONB NOT NULL,
    pain_points JSONB NOT NULL DEFAULT '[]',
    UNIQUE (analysis_id, document_id, company_id, aspect),
    FOREIGN KEY (document_id, company_id) REFERENCES documents(id, company_id)
);
CREATE INDEX IF NOT EXISTS aspects_analysis_sentiment ON aspects(analysis_id, sentiment);

CREATE TABLE IF NOT EXISTS analysis_documents (
    analysis_id UUID NOT NULL REFERENCES analyses(id),
    document_id UUID NOT NULL,
    company_id BIGINT NOT NULL,
    role TEXT,
    snapshot JSONB NOT NULL,
    PRIMARY KEY(analysis_id, document_id, company_id),
    FOREIGN KEY(document_id, company_id) REFERENCES documents(id, company_id)
);
