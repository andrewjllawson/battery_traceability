-- Adds synthesis steps to an existing Hazbat database (new databases get this from schema.sql).
-- Run once:  psql "$DATABASE_URL" -f migrations/002_material_synthesis.sql
CREATE TABLE IF NOT EXISTS tbl_material_synthesis (
    material_id     TEXT PRIMARY KEY REFERENCES tbl_materials(material_id) ON DELETE CASCADE,
    precursors      TEXT,
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tbl_material_synthesis_steps (
    step_id         SERIAL PRIMARY KEY,
    material_id     TEXT NOT NULL REFERENCES tbl_materials(material_id) ON DELETE CASCADE,
    step_order      INTEGER NOT NULL CHECK (step_order > 0),
    step            TEXT NOT NULL,
    description     TEXT,
    UNIQUE (material_id, step_order)
);
