-- =====================================================
-- HUIHUI USAGE MONITORING TABLES - SUPABASE SCHEMA
-- =====================================================
-- 
-- Comprehensive usage monitoring and optimization tables for HuiHui-MoE
-- expert system. Tracks rate limits, token usage, performance metrics,
-- and provides optimization recommendations.
--
-- Author: EOTS v2.5 AI Optimization Division
-- Version: 1.0.0 - "PRODUCTION MONITORING"
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- HUIHUI USAGE RECORDS TABLE
-- =====================================================
-- Stores detailed usage records for each expert call
CREATE TABLE IF NOT EXISTS huihui_usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expert_name VARCHAR(50) NOT NULL,
    request_type VARCHAR(50) NOT NULL, -- 'analysis', 'prediction', 'synthesis'
    
    -- Token usage metrics
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    
    -- Performance metrics
    processing_time_seconds DECIMAL(8,3) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_type VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    timeout_occurred BOOLEAN DEFAULT FALSE,
    
    -- Market context
    market_condition VARCHAR(20) NOT NULL DEFAULT 'normal', -- 'normal', 'volatile', 'crisis'
    vix_level DECIMAL(6,2),
    symbol VARCHAR(20),
    
    -- Security and audit
    api_token_hash VARCHAR(64), -- Last 8 chars of token for audit
    user_session_id VARCHAR(100),
    
    -- Metadata
    request_metadata JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_processing_time CHECK (processing_time_seconds >= 0),
    CONSTRAINT chk_tokens CHECK (input_tokens >= 0 AND output_tokens >= 0 AND total_tokens >= 0),
    CONSTRAINT chk_market_condition CHECK (market_condition IN ('normal', 'volatile', 'crisis'))
);

-- =====================================================
-- HUIHUI USAGE PATTERNS TABLE
-- =====================================================
-- Aggregated usage patterns for optimization analysis
CREATE TABLE IF NOT EXISTS huihui_usage_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expert_name VARCHAR(50) NOT NULL,
    analysis_period VARCHAR(20) NOT NULL, -- '1h', '24h', '7d', '30d'
    
    -- Request rate metrics
    total_requests INTEGER NOT NULL,
    avg_requests_per_minute DECIMAL(8,3) NOT NULL,
    peak_requests_per_minute DECIMAL(8,3) NOT NULL,
    
    -- Token usage metrics
    avg_input_tokens DECIMAL(8,1) NOT NULL,
    avg_output_tokens DECIMAL(8,1) NOT NULL,
    avg_total_tokens DECIMAL(8,1) NOT NULL,
    max_input_tokens INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    max_total_tokens INTEGER NOT NULL,
    
    -- Performance metrics
    avg_processing_time DECIMAL(8,3) NOT NULL,
    max_processing_time DECIMAL(8,3) NOT NULL,
    success_rate DECIMAL(5,4) NOT NULL,
    timeout_rate DECIMAL(5,4) NOT NULL,
    
    -- Market condition distribution
    normal_condition_requests INTEGER DEFAULT 0,
    volatile_condition_requests INTEGER DEFAULT 0,
    crisis_condition_requests INTEGER DEFAULT 0,
    
    -- Analysis metadata
    analysis_start_time TIMESTAMPTZ NOT NULL,
    analysis_end_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT chk_timeout_rate CHECK (timeout_rate >= 0.0 AND timeout_rate <= 1.0)
);

-- =====================================================
-- HUIHUI OPTIMIZATION RECOMMENDATIONS TABLE
-- =====================================================
-- Stores optimization recommendations based on usage patterns
CREATE TABLE IF NOT EXISTS huihui_optimization_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expert_name VARCHAR(50) NOT NULL,
    
    -- Current configuration
    current_rate_limit INTEGER NOT NULL,
    current_token_limit INTEGER NOT NULL,
    current_timeout_seconds INTEGER NOT NULL,
    
    -- Recommended configuration
    recommended_rate_limit INTEGER NOT NULL,
    recommended_token_limit INTEGER NOT NULL,
    recommended_timeout_seconds INTEGER NOT NULL,
    
    -- Recommendation metrics
    confidence_score DECIMAL(5,4) NOT NULL,
    urgency_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    market_condition_factor VARCHAR(30) NOT NULL, -- 'normal_conditions', 'volatility_optimized', 'crisis_optimized'
    
    -- Analysis basis
    based_on_requests INTEGER NOT NULL,
    analysis_period_hours INTEGER NOT NULL,
    peak_usage_factor DECIMAL(6,3) NOT NULL,
    
    -- Reasoning and metadata
    reasoning TEXT NOT NULL,
    implementation_priority INTEGER DEFAULT 5, -- 1-10 scale
    estimated_improvement_percent DECIMAL(5,2),
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'implemented', 'rejected'
    implemented_at TIMESTAMPTZ,
    implemented_by VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_urgency_level CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_priority CHECK (implementation_priority >= 1 AND implementation_priority <= 10),
    CONSTRAINT chk_status CHECK (status IN ('pending', 'approved', 'implemented', 'rejected'))
);

-- =====================================================
-- HUIHUI SYSTEM HEALTH TABLE
-- =====================================================
-- Tracks overall system health and performance
CREATE TABLE IF NOT EXISTS huihui_system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- System metrics
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_percent DECIMAL(5,2),
    gpu_memory_used_percent DECIMAL(5,2),
    gpu_temperature DECIMAL(5,1),
    gpu_load_percent DECIMAL(5,2),
    
    -- Ollama health
    ollama_healthy BOOLEAN NOT NULL,
    ollama_response_time_ms INTEGER,
    
    -- Expert availability
    experts_available JSONB DEFAULT '{}', -- {"market_regime": true, "options_flow": true, ...}
    
    -- Performance summary
    total_requests_last_hour INTEGER DEFAULT 0,
    avg_response_time_last_hour DECIMAL(8,3),
    error_rate_last_hour DECIMAL(5,4),
    
    -- Market context
    current_market_condition VARCHAR(20),
    current_vix_level DECIMAL(6,2),
    
    -- Timestamps
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_cpu_usage CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    CONSTRAINT chk_memory_usage CHECK (memory_usage_percent >= 0 AND memory_usage_percent <= 100),
    CONSTRAINT chk_error_rate CHECK (error_rate_last_hour >= 0.0 AND error_rate_last_hour <= 1.0)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Usage records indexes
CREATE INDEX IF NOT EXISTS idx_huihui_usage_records_expert_time 
ON huihui_usage_records(expert_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_huihui_usage_records_market_condition 
ON huihui_usage_records(market_condition, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_huihui_usage_records_success 
ON huihui_usage_records(success, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_huihui_usage_records_symbol 
ON huihui_usage_records(symbol, created_at DESC);

-- Usage patterns indexes
CREATE INDEX IF NOT EXISTS idx_huihui_usage_patterns_expert_period 
ON huihui_usage_patterns(expert_name, analysis_period, created_at DESC);

-- Optimization recommendations indexes
CREATE INDEX IF NOT EXISTS idx_huihui_optimization_expert_status 
ON huihui_optimization_recommendations(expert_name, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_huihui_optimization_urgency 
ON huihui_optimization_recommendations(urgency_level, created_at DESC);

-- System health indexes
CREATE INDEX IF NOT EXISTS idx_huihui_system_health_time 
ON huihui_system_health(recorded_at DESC);

-- =====================================================
-- VIEWS FOR EASY QUERYING
-- =====================================================

-- Current usage summary view
CREATE OR REPLACE VIEW huihui_current_usage_summary AS
SELECT 
    expert_name,
    COUNT(*) as requests_last_hour,
    AVG(processing_time_seconds) as avg_processing_time,
    AVG(total_tokens) as avg_tokens,
    MAX(total_tokens) as max_tokens,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as success_rate,
    SUM(CASE WHEN timeout_occurred THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as timeout_rate
FROM huihui_usage_records 
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY expert_name;

-- Peak usage analysis view
CREATE OR REPLACE VIEW huihui_peak_usage_analysis AS
WITH hourly_usage AS (
    SELECT 
        expert_name,
        DATE_TRUNC('hour', created_at) as hour,
        COUNT(*) as requests_per_hour
    FROM huihui_usage_records 
    WHERE created_at >= NOW() - INTERVAL '7 days'
    GROUP BY expert_name, DATE_TRUNC('hour', created_at)
)
SELECT 
    expert_name,
    MAX(requests_per_hour) as peak_requests_per_hour,
    AVG(requests_per_hour) as avg_requests_per_hour,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY requests_per_hour) as p95_requests_per_hour
FROM hourly_usage
GROUP BY expert_name;

-- Market condition impact view
CREATE OR REPLACE VIEW huihui_market_condition_impact AS
SELECT 
    expert_name,
    market_condition,
    COUNT(*) as request_count,
    AVG(processing_time_seconds) as avg_processing_time,
    AVG(total_tokens) as avg_tokens,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as success_rate
FROM huihui_usage_records 
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY expert_name, market_condition
ORDER BY expert_name, market_condition;

-- =====================================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- =====================================================

-- Function to clean old usage records (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_huihui_usage_records()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM huihui_usage_records 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to generate usage pattern summary
CREATE OR REPLACE FUNCTION generate_huihui_usage_pattern(
    p_expert_name VARCHAR(50),
    p_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    total_requests INTEGER,
    avg_requests_per_minute DECIMAL,
    peak_requests_per_minute DECIMAL,
    avg_total_tokens DECIMAL,
    max_total_tokens INTEGER,
    success_rate DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_requests,
        (COUNT(*)::DECIMAL / (p_hours * 60)) as avg_requests_per_minute,
        -- Peak calculation would need more complex logic
        (COUNT(*)::DECIMAL / (p_hours * 60)) * 2 as peak_requests_per_minute, -- Simplified
        AVG(hur.total_tokens) as avg_total_tokens,
        MAX(hur.total_tokens) as max_total_tokens,
        (SUM(CASE WHEN hur.success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) as success_rate
    FROM huihui_usage_records hur
    WHERE hur.expert_name = p_expert_name 
    AND hur.created_at >= NOW() - (p_hours || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL DATA AND CONFIGURATION
-- =====================================================

-- Insert initial system health record
INSERT INTO huihui_system_health (
    ollama_healthy,
    current_market_condition,
    total_requests_last_hour,
    avg_response_time_last_hour,
    error_rate_last_hour
) VALUES (
    TRUE,
    'normal',
    0,
    0.0,
    0.0
) ON CONFLICT DO NOTHING;

-- =====================================================
-- COMMENTS AND DOCUMENTATION
-- =====================================================

COMMENT ON TABLE huihui_usage_records IS 'Detailed usage records for each HuiHui expert call with performance metrics';
COMMENT ON TABLE huihui_usage_patterns IS 'Aggregated usage patterns for optimization analysis';
COMMENT ON TABLE huihui_optimization_recommendations IS 'System-generated optimization recommendations based on usage patterns';
COMMENT ON TABLE huihui_system_health IS 'Overall system health and performance monitoring';

COMMENT ON COLUMN huihui_usage_records.expert_name IS 'Name of the HuiHui expert (market_regime, options_flow, sentiment, orchestrator)';
COMMENT ON COLUMN huihui_usage_records.processing_time_seconds IS 'Time taken to process the request in seconds';
COMMENT ON COLUMN huihui_usage_records.market_condition IS 'Market condition during the request (normal, volatile, crisis)';
COMMENT ON COLUMN huihui_optimization_recommendations.confidence_score IS 'Confidence in the recommendation (0.0 to 1.0)';
COMMENT ON COLUMN huihui_optimization_recommendations.urgency_level IS 'How urgently the recommendation should be implemented';
