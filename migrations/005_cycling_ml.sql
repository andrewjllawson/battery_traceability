-- Cycle summaries, cell features and the model registry for an existing Hazbat database
-- (new databases get them from schemas/0000_schema.sql).
-- Run once:  gosu postgres psql -d bmac -v ON_ERROR_STOP=1 -f migrations/005_cycling_ml.sql
-- then rerun scripts.dbgen (or restart the app) to grant the roles access.
-- Filled by scripts/extract_cycle_features.py and the training scripts, not by the pages.
-- Model values are DOUBLE PRECISION, not NUMERIC, so they come back as floats.

-- Create the tables as their owner, so scripts.dbgen can grant them to the other roles.
SET ROLE hazbat_admin;

-- One row per full cycle of an uploaded .mpr file (services/cycle_features.py).
-- Cycles the quality gate drops are kept with passed_gate = FALSE, so the gate can be reported.
CREATE TABLE IF NOT EXISTS tbl_cycle_summary (
    cycling_id      INTEGER NOT NULL REFERENCES tbl_cycling_data(cycling_id) ON DELETE CASCADE,
    record_id       TEXT NOT NULL,
    cycle_no        INTEGER NOT NULL CHECK (cycle_no > 0),
    discharge_mah   DOUBLE PRECISION CHECK (discharge_mah >= 0),
    charge_mah      DOUBLE PRECISION CHECK (charge_mah >= 0),
    coulombic_eff   DOUBLE PRECISION CHECK (coulombic_eff > 0),
    charge_time_s   DOUBLE PRECISION CHECK (charge_time_s >= 0),
    passed_gate     BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (cycling_id, cycle_no)
);
CREATE INDEX IF NOT EXISTS idx_cycle_summary_record ON tbl_cycle_summary (record_id);

-- Early-life features of one .mpr file (Severson et al. 2019): the ones a .mpr can give
-- (REDUCED_FEATURES). Capacity in Ah, charge time in minutes. Internal resistance and
-- temperature are not recorded yet; add their columns when they are. Features are NULL when
-- the test is too short; feature_error says why.
CREATE TABLE IF NOT EXISTS tbl_cell_features (
    cycling_id          INTEGER PRIMARY KEY REFERENCES tbl_cycling_data(cycling_id) ON DELETE CASCADE,
    record_id           TEXT NOT NULL,
    n_cycles            INTEGER CHECK (n_cycles >= 0),
    log_var_dq          DOUBLE PRECISION,
    log_abs_min_dq      DOUBLE PRECISION,
    log_abs_skew_dq     DOUBLE PRECISION,
    log_abs_kurt_dq     DOUBLE PRECISION,
    qd_cycle2           DOUBLE PRECISION,
    qd_max_minus_cycle2 DOUBLE PRECISION,
    fade_slope          DOUBLE PRECISION,
    fade_intercept      DOUBLE PRECISION,
    charge_time_2to6    DOUBLE PRECISION,
    v_grid_max          DOUBLE PRECISION,                 -- voltage range of the ΔQ(V) curve
    v_grid_min          DOUBLE PRECISION,
    gate                JSONB NOT NULL DEFAULT '[]',      -- the check_cycles() report
    feature_error       TEXT,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cell_features_record ON tbl_cell_features (record_id);

-- Every model training run, for any data type (mixing, cycling, ...). One row per model and
-- split; metrics holds the full scores and the shared result shape (importance, scatter).
CREATE TABLE IF NOT EXISTS tbl_model_runs (
    run_id          SERIAL PRIMARY KEY,
    data_type       TEXT NOT NULL,                        -- 'mixing', 'cycling'
    model           TEXT NOT NULL,                        -- e.g. 'Gradient boosting', 'Linear fade'
    target          TEXT NOT NULL,
    features        TEXT[] NOT NULL DEFAULT '{}',
    split_method    TEXT NOT NULL,                        -- e.g. 'leave-one-stage-out'
    n_train         INTEGER CHECK (n_train >= 0),
    n_test          INTEGER CHECK (n_test >= 0),
    metric_name     TEXT NOT NULL,                        -- e.g. 'MAE (W)', 'MAPE (%)'
    metric          DOUBLE PRECISION,
    baseline_metric DOUBLE PRECISION,
    metrics         JSONB NOT NULL DEFAULT '{}',
    params          JSONB NOT NULL DEFAULT '{}',
    data_source     TEXT,                                 -- input files and their SHA-256
    code_version    TEXT,                                 -- git commit
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_runs_type ON tbl_model_runs (data_type, trained_at DESC);

INSERT INTO tbl_unprivileged_tables (table_name) VALUES
    ('tbl_cycle_summary'), ('tbl_cell_features'), ('tbl_model_runs')
ON CONFLICT (table_name) DO NOTHING;
