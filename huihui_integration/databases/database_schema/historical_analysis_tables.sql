-- Historical Analysis Tables for EOTS v2.5
-- ==========================================
-- These tables enable powerful historical comparison and pattern recognition

-- 1. Daily Regime Snapshots Table
CREATE TABLE IF NOT EXISTS daily_regime_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    primary_regime VARCHAR(100) NOT NULL,
    regime_confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    regime_duration_minutes INTEGER DEFAULT 0,
    regime_transition_count INTEGER DEFAULT 0,
    secondary_regimes JSONB DEFAULT '[]',
    market_conditions JSONB DEFAULT '{}', -- VIX, SPY close, volume, etc.
    volatility_environment VARCHAR(50),
    flow_intensity_score DECIMAL(5,4) DEFAULT 0.0000,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(symbol, date),
    CONSTRAINT chk_regime_confidence CHECK (regime_confidence_score >= 0.0 AND regime_confidence_score <= 1.0),
    CONSTRAINT chk_flow_intensity CHECK (flow_intensity_score >= 0.0 AND flow_intensity_score <= 1.0)
);

-- 2. Key Level Performance Tracking Table
CREATE TABLE IF NOT EXISTS key_level_performance (
    level_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    level_price DECIMAL(10,4) NOT NULL,
    level_type VARCHAR(50) NOT NULL, -- support, resistance, gamma_wall, max_pain, etc.
    level_source VARCHAR(50) NOT NULL, -- A-DAG, GEX, OI, etc.
    hit_count INTEGER DEFAULT 0,
    bounce_count INTEGER DEFAULT 0,
    break_count INTEGER DEFAULT 0,
    bounce_accuracy DECIMAL(5,4) DEFAULT 0.0000,
    break_significance DECIMAL(5,4) DEFAULT 0.0000,
    time_to_test_minutes INTEGER,
    hold_duration_minutes INTEGER,
    max_distance_from_level DECIMAL(10,4),
    conviction_score DECIMAL(5,4) DEFAULT 0.0000,
    market_regime_context VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT chk_bounce_accuracy CHECK (bounce_accuracy >= 0.0 AND bounce_accuracy <= 1.0),
    CONSTRAINT chk_break_significance CHECK (break_significance >= 0.0 AND break_significance <= 1.0),
    CONSTRAINT chk_conviction_score CHECK (conviction_score >= 0.0 AND conviction_score <= 1.0)
);

-- 3. Flow Pattern Library Table
CREATE TABLE IF NOT EXISTS flow_pattern_library (
    pattern_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_signature JSONB NOT NULL, -- Encoded pattern characteristics
    pattern_strength DECIMAL(5,4) DEFAULT 0.0000,
    market_outcome VARCHAR(50), -- breakout_up, breakout_down, reversal, continuation
    follow_through_success BOOLEAN,
    follow_through_magnitude DECIMAL(8,4),
    regime_context VARCHAR(100),
    volatility_environment VARCHAR(50),
    time_horizon_minutes INTEGER,
    success_rate DECIMAL(5,4) DEFAULT 0.0000,
    sample_size INTEGER DEFAULT 1,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT chk_pattern_strength CHECK (pattern_strength >= 0.0 AND pattern_strength <= 1.0),
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- 4. Enhanced AI Performance Tracking (extends existing ai_predictions)
CREATE TABLE IF NOT EXISTS ai_performance_daily_summary (
    summary_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,4) DEFAULT 0.0000,
    avg_confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    regime_prediction_accuracy JSONB DEFAULT '{}', -- accuracy by regime
    prediction_type_performance JSONB DEFAULT '{}', -- accuracy by prediction type
    high_confidence_accuracy DECIMAL(5,4) DEFAULT 0.0000, -- accuracy for confidence > 0.8
    medium_confidence_accuracy DECIMAL(5,4) DEFAULT 0.0000, -- accuracy for 0.5-0.8
    low_confidence_accuracy DECIMAL(5,4) DEFAULT 0.0000, -- accuracy for < 0.5
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(date, symbol),
    CONSTRAINT chk_accuracy_rate CHECK (accuracy_rate >= 0.0 AND accuracy_rate <= 1.0),
    CONSTRAINT chk_avg_confidence CHECK (avg_confidence_score >= 0.0 AND avg_confidence_score <= 1.0)
);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_regime_snapshots_symbol_date ON daily_regime_snapshots(symbol, date);
CREATE INDEX IF NOT EXISTS idx_regime_snapshots_regime ON daily_regime_snapshots(primary_regime);
CREATE INDEX IF NOT EXISTS idx_regime_snapshots_date ON daily_regime_snapshots(date);

CREATE INDEX IF NOT EXISTS idx_key_level_symbol_date ON key_level_performance(symbol, date);
CREATE INDEX IF NOT EXISTS idx_key_level_type ON key_level_performance(level_type);
CREATE INDEX IF NOT EXISTS idx_key_level_price ON key_level_performance(level_price);
CREATE INDEX IF NOT EXISTS idx_key_level_regime ON key_level_performance(market_regime_context);

CREATE INDEX IF NOT EXISTS idx_flow_pattern_symbol_date ON flow_pattern_library(symbol, date);
CREATE INDEX IF NOT EXISTS idx_flow_pattern_name ON flow_pattern_library(pattern_name);
CREATE INDEX IF NOT EXISTS idx_flow_pattern_outcome ON flow_pattern_library(market_outcome);
CREATE INDEX IF NOT EXISTS idx_flow_pattern_regime ON flow_pattern_library(regime_context);

CREATE INDEX IF NOT EXISTS idx_ai_performance_date ON ai_performance_daily_summary(date);
CREATE INDEX IF NOT EXISTS idx_ai_performance_symbol ON ai_performance_daily_summary(symbol);
CREATE INDEX IF NOT EXISTS idx_ai_performance_accuracy ON ai_performance_daily_summary(accuracy_rate);
