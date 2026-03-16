-- Customer Analytics Platform — Database Schema
-- Run once to initialise all tables

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    plan            VARCHAR(20) NOT NULL,
    channel         VARCHAR(30) NOT NULL,
    country         VARCHAR(50) NOT NULL,
    age             INT,
    is_churned      BOOLEAN NOT NULL DEFAULT FALSE,
    churned_at      TIMESTAMPTZ,
    ltv_actual      NUMERIC(10,2) DEFAULT 0
);

-- User events (session activity)
CREATE TABLE IF NOT EXISTS user_events (
    event_id        BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    event_type      VARCHAR(50) NOT NULL,  -- login, feature_use, support_ticket, etc.
    event_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB
);

-- Subscriptions
CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id          BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    plan            VARCHAR(20) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    mrr             NUMERIC(10,2) NOT NULL  -- monthly recurring revenue
);

-- Transactions
CREATE TABLE IF NOT EXISTS transactions (
    tx_id           BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    amount          NUMERIC(10,2) NOT NULL,
    currency        VARCHAR(5) NOT NULL DEFAULT 'EUR',
    status          VARCHAR(20) NOT NULL,  -- success, failed, refunded
    tx_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- A/B tests
CREATE TABLE IF NOT EXISTS ab_tests (
    test_id         BIGSERIAL PRIMARY KEY,
    test_name       VARCHAR(100) NOT NULL UNIQUE,
    hypothesis      TEXT,
    metric          VARCHAR(50) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'running',
    result          VARCHAR(20),  -- significant, not_significant, inconclusive
    p_value         NUMERIC(6,4),
    lift            NUMERIC(6,4)
);

-- A/B test assignments
CREATE TABLE IF NOT EXISTS ab_assignments (
    assignment_id   BIGSERIAL PRIMARY KEY,
    test_id         BIGINT NOT NULL REFERENCES ab_tests(test_id),
    user_id         UUID NOT NULL REFERENCES users(user_id),
    variant         VARCHAR(20) NOT NULL,  -- control, treatment
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    converted       BOOLEAN DEFAULT FALSE,
    converted_at    TIMESTAMPTZ,
    UNIQUE(test_id, user_id)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    pred_id         BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    model_name      VARCHAR(50) NOT NULL,
    prediction      NUMERIC(10,4) NOT NULL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(50),
    features        JSONB
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_user_id    ON user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_at         ON user_events(event_at);
CREATE INDEX IF NOT EXISTS idx_txn_user_id       ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, predicted_at);
CREATE INDEX IF NOT EXISTS idx_ab_assign_test    ON ab_assignments(test_id, variant);