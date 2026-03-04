-- ─────────────────────────────────────────────────────────────────────────────
-- Bank-Marketing ML — PostgreSQL schema
-- Run: psql -U postgres -d bank_marketing -f data/sql/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Raw data ──────────────────────────────────────────────────────────────────
-- Mirrors the bank-full.csv (ucimlrepo id=222): 45 211 rows × 16 features
CREATE TABLE IF NOT EXISTS raw_features (
    id               SERIAL PRIMARY KEY,
    -- Demographic
    age              INTEGER,
    job              VARCHAR(30),
    marital          VARCHAR(20),
    education        VARCHAR(40),
    -- Financial
    "default"        VARCHAR(10),
    balance          INTEGER,           -- average yearly balance (euros)
    housing          VARCHAR(10),
    loan             VARCHAR(10),
    -- Last contact
    contact          VARCHAR(20),
    day_of_week      INTEGER,           -- day of month (1-31)
    month            VARCHAR(5),
    duration         INTEGER,           -- excluded from models (data leakage)
    -- Campaign
    campaign         INTEGER,
    pdays            INTEGER,           -- 999 = never previously contacted
    previous         INTEGER,
    poutcome         VARCHAR(20),
    -- Target
    subscribed       VARCHAR(5),
    -- Metadata
    ingested_at      TIMESTAMP DEFAULT NOW()
);

-- ── Processed / feature-engineered data ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS processed_features (
    id                      SERIAL PRIMARY KEY,
    raw_id                  INTEGER REFERENCES raw_features(id),
    -- Encoded features (one-hot / ordinal stored as JSONB for flexibility)
    features_json           JSONB NOT NULL,
    -- Binary target (1 = subscribed, 0 = not)
    subscribed              SMALLINT NOT NULL,
    split                   VARCHAR(10),   -- 'train', 'val', 'test'
    processed_at            TIMESTAMP DEFAULT NOW()
);

-- ── Model predictions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    raw_id              INTEGER REFERENCES raw_features(id),
    model_name          VARCHAR(100) NOT NULL,
    mlflow_run_id       VARCHAR(64),
    predicted_label     SMALLINT NOT NULL,           -- 0 or 1
    predicted_proba     NUMERIC(6, 5),               -- probability of class=1
    actual_label        SMALLINT,
    predicted_at        TIMESTAMP DEFAULT NOW()
);

-- ── Experiment summary (denormalised snapshot for quick reporting) ─────────────
CREATE TABLE IF NOT EXISTS experiment_results (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   VARCHAR(64) UNIQUE,
    model_name      VARCHAR(100),
    roc_auc         NUMERIC(6, 4),
    f1_score        NUMERIC(6, 4),
    precision_score NUMERIC(6, 4),
    recall_score    NUMERIC(6, 4),
    accuracy        NUMERIC(6, 4),
    params_json     JSONB,
    trained_at      TIMESTAMP DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_raw_features_subscribed   ON raw_features(subscribed);
CREATE INDEX IF NOT EXISTS idx_processed_split           ON processed_features(split);
CREATE INDEX IF NOT EXISTS idx_predictions_model         ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_experiment_model          ON experiment_results(model_name);
