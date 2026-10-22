CREATE TABLE IF NOT EXISTS research_projects (
    project_id BIGSERIAL PRIMARY KEY,
    project_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS candidate_materials (
    candidate_id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL
        REFERENCES research_projects(project_id)
        ON DELETE CASCADE,
    material_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'REVIEWING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(project_id, material_id)
);

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id BIGSERIAL PRIMARY KEY,
    project_id BIGINT
        REFERENCES research_projects(project_id)
        ON DELETE SET NULL,
    material_id TEXT,
    action_type TEXT NOT NULL,
    action_details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO research_projects (project_name)
VALUES ('Solar Materials')
ON CONFLICT (project_name) DO NOTHING;