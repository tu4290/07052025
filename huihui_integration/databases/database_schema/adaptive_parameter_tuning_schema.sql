-- =============================================================================
-- Schema: HUIHUI_ADAPTIVE_PARAMETER_TUNING
-- Author: EOTS v2.5 AI Architecture Division
-- Version: 1.0
-- Description:
-- This schema provides the necessary tables for the HuiHui AI system's
-- elite Adaptive Parameter Tuning engine. It allows for dynamic, hands-off
-- optimization of expert parameters based on real-world performance,
-- ensuring the system adapts to changing market dynamics without manual
-- intervention.
--
-- Features:
-- - Persistence for expert parameter configurations and safe bounds.
-- - Tracking of current, optimized parameter values for different market regimes.
-- - A detailed audit trail for all parameter adjustments.
-- - Performance outcome logging linked to the specific parameters used.
-- - State persistence for the Bayesian optimization models.
-- =============================================================================

-- Best practice: Ensure all objects are created within a specific schema
-- CREATE SCHEMA IF NOT EXISTS huihui_learning;
-- SET search_path TO huihui_learning;

-- =============================================================================
-- 1. TRIGGER FUNCTION FOR AUTO-UPDATING 'updated_at' TIMESTAMPS
-- =============================================================================
-- This function is a standard PostgreSQL utility to automatically update the
-- 'updated_at' column on any row modification.

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = now();
   RETURN NEW;
END;
$$ language 'plpgsql';

COMMENT ON FUNCTION update_updated_at_column() IS 'This trigger function automatically updates the "updated_at" column to the current timestamp whenever a row is modified.';


-- =============================================================================
-- 2. TABLE: expert_parameter_configs
-- Description: Defines the tunable parameters for each expert.
-- =============================================================================
CREATE TABLE IF NOT EXISTS expert_parameter_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    expert_id TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    initial_value REAL NOT NULL,
    min_bound REAL NOT NULL,
    max_bound REAL NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (expert_id, parameter_name)
);

COMMENT ON TABLE expert_parameter_configs IS 'Defines the set of tunable parameters for each expert, including their safe operating bounds and initial default values.';
COMMENT ON COLUMN expert_parameter_configs.expert_id IS 'Identifier for the expert system, e.g., "flow_analytics".';
COMMENT ON COLUMN expert_parameter_configs.parameter_name IS 'The name of the parameter, e.g., "high_volume_threshold".';
COMMENT ON COLUMN expert_parameter_configs.min_bound IS 'The absolute minimum safe value for this parameter.';
COMMENT ON COLUMN expert_parameter_configs.max_bound IS 'The absolute maximum safe value for this parameter.';

CREATE INDEX IF NOT EXISTS idx_expert_parameter_configs_expert_id ON expert_parameter_configs(expert_id);

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON expert_parameter_configs
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();


-- =============================================================================
-- 3. TABLE: current_parameter_values
-- Description: Stores the live, optimized value for each parameter and regime.
-- =============================================================================
CREATE TABLE IF NOT EXISTS current_parameter_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id UUID NOT NULL REFERENCES expert_parameter_configs(id) ON DELETE CASCADE,
    market_regime TEXT NOT NULL DEFAULT 'global',
    current_value REAL NOT NULL,
    last_updated_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (config_id, market_regime)
);

COMMENT ON TABLE current_parameter_values IS 'Stores the current, live, optimized value for each parameter, potentially specialized for different market regimes.';
COMMENT ON COLUMN current_parameter_values.config_id IS 'Foreign key linking to the parameter''s definition.';
COMMENT ON COLUMN current_parameter_values.market_regime IS 'The market regime this specific value applies to (e.g., "high_volatility", "global").';
COMMENT ON COLUMN current_parameter_values.current_value IS 'The live, optimized value currently in use by the system.';
COMMENT ON COLUMN current_parameter_values.last_updated_by IS 'Identifier for the process that last updated this value, e.g., "bayesian_optimizer".';

CREATE INDEX IF NOT EXISTS idx_current_parameter_values_config_id ON current_parameter_values(config_id);
CREATE INDEX IF NOT EXISTS idx_current_parameter_values_market_regime ON current_parameter_values(market_regime);

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON current_parameter_values
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();


-- =============================================================================
-- 4. TABLE: performance_outcomes
-- Description: Logs the outcome of each prediction/trade event.
-- =============================================================================
CREATE TABLE IF NOT EXISTS performance_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    expert_id TEXT NOT NULL,
    prediction_id TEXT NOT NULL UNIQUE,
    outcome_metric REAL NOT NULL,
    market_regime TEXT NOT NULL,
    parameters_used JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE performance_outcomes IS 'Logs the performance outcome (e.g., P&L, accuracy) of a specific prediction event, capturing the exact set of parameters used.';
COMMENT ON COLUMN performance_outcomes.prediction_id IS 'A unique ID for the prediction/trade event being evaluated.';
COMMENT ON COLUMN performance_outcomes.outcome_metric IS 'The performance score for this event (e.g., P&L, Sharpe ratio, +1 for win, -1 for loss).';
COMMENT ON COLUMN performance_outcomes.parameters_used IS 'A JSONB blob of all parameter key-value pairs used to generate this prediction.';

CREATE INDEX IF NOT EXISTS idx_performance_outcomes_expert_id_timestamp ON performance_outcomes(expert_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_outcomes_market_regime ON performance_outcomes(market_regime);
CREATE INDEX IF NOT EXISTS idx_performance_outcomes_parameters_gin ON performance_outcomes USING GIN (parameters_used);


-- =============================================================================
-- 5. TABLE: parameter_adjustment_logs
-- Description: Immutable audit trail of all parameter changes.
-- =============================================================================
CREATE TABLE IF NOT EXISTS parameter_adjustment_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameter_config_id UUID NOT NULL REFERENCES expert_parameter_configs(id) ON DELETE RESTRICT,
    market_regime TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    reason TEXT NOT NULL,
    performance_delta REAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE parameter_adjustment_logs IS 'Immutable audit trail for every automatic change made to a parameter, providing full explainability.';
COMMENT ON COLUMN parameter_adjustment_logs.reason IS 'Human-readable reason for the change, generated by the optimization engine.';
COMMENT ON COLUMN parameter_adjustment_logs.performance_delta IS 'The predicted or measured change in performance that prompted the adjustment.';

CREATE INDEX IF NOT EXISTS idx_parameter_adjustment_logs_param_id_timestamp ON parameter_adjustment_logs(parameter_config_id, timestamp DESC);


-- =============================================================================
-- 6. TABLE: bayesian_optimizer_state
-- Description: Persists the state of the Bayesian optimization models.
-- =============================================================================
CREATE TABLE IF NOT EXISTS bayesian_optimizer_state (
    optimizer_key TEXT PRIMARY KEY,
    state_json JSONB NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE bayesian_optimizer_state IS 'Persists the learned state of Bayesian optimizers to allow learning across system restarts.';
COMMENT ON COLUMN bayesian_optimizer_state.optimizer_key IS 'Unique key for an optimizer, e.g., "flow_analytics:high_volatility:high_volume_threshold".';
COMMENT ON COLUMN bayesian_optimizer_state.state_json IS 'JSONB blob containing the serialized state of the Gaussian Process model (e.g., X_samples, Y_samples).';

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON bayesian_optimizer_state
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
