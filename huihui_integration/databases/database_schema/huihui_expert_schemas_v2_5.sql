-- =====================================================
-- HUIHUI EXPERT SCHEMAS v2.5 - SUPABASE SCHEMA
-- =====================================================
-- 
-- Comprehensive expert-specific schema tables for HuiHui-MoE
-- expert system. Supports Market Regime, Options Flow, Sentiment,
-- and Orchestrator experts with full Pydantic model compatibility.
--
-- Author: EOTS v2.5 AI Schema Division
-- Version: 2.5.0 - "EXPERT SCHEMA FOUNDATION"
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- HUIHUI MARKET REGIME ANALYSIS TABLE
-- =====================================================
-- Stores comprehensive market regime analysis results
CREATE TABLE IF NOT EXISTS huihui_market_regime_analysis (
    -- Primary identifiers
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    expert_type VARCHAR(20) NOT NULL DEFAULT 'market_regime',
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Core regime analysis
    current_regime_id INTEGER NOT NULL,
    regime_confidence DECIMAL(5,4) NOT NULL,
    regime_probability DECIMAL(5,4) NOT NULL,
    regime_name VARCHAR(50) NOT NULL,
    regime_description TEXT,
    
    -- VRI 3.0 Components
    vri_3_composite DECIMAL(8,4) NOT NULL,
    volatility_regime_score DECIMAL(8,4) NOT NULL,
    flow_intensity_score DECIMAL(8,4) NOT NULL,
    regime_stability_score DECIMAL(8,4) NOT NULL,
    transition_momentum_score DECIMAL(8,4) NOT NULL,
    
    -- Regime characteristics
    volatility_level VARCHAR(20) NOT NULL,
    trend_direction VARCHAR(20) NOT NULL,
    flow_pattern VARCHAR(30) NOT NULL,
    risk_appetite VARCHAR(20) NOT NULL,
    
    -- Transition prediction
    predicted_regime_id INTEGER,
    transition_probability DECIMAL(5,4),
    expected_transition_timeframe VARCHAR(30),
    transition_confidence DECIMAL(5,4),
    
    -- Performance metrics
    confidence_score DECIMAL(5,4) NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    data_quality_score DECIMAL(5,4) NOT NULL,
    
    -- Supporting data
    supporting_indicators JSONB DEFAULT '{}',
    regime_factors JSONB DEFAULT '{}',
    historical_context JSONB DEFAULT '{}',
    
    -- Quality and metadata
    quality_metrics JSONB DEFAULT '{}',
    error_flags TEXT[],
    warning_flags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_regime_confidence CHECK (regime_confidence >= 0.0 AND regime_confidence <= 1.0),
    CONSTRAINT chk_regime_probability CHECK (regime_probability >= 0.0 AND regime_probability <= 1.0),
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    CONSTRAINT chk_volatility_level CHECK (volatility_level IN ('low', 'medium', 'high', 'extreme')),
    CONSTRAINT chk_trend_direction CHECK (trend_direction IN ('bullish', 'bearish', 'neutral', 'mixed')),
    CONSTRAINT chk_risk_appetite CHECK (risk_appetite IN ('risk_on', 'risk_off', 'neutral', 'mixed'))
);

-- =====================================================
-- HUIHUI OPTIONS FLOW ANALYSIS TABLE
-- =====================================================
-- Stores comprehensive options flow analysis results
CREATE TABLE IF NOT EXISTS huihui_options_flow_analysis (
    -- Primary identifiers
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    expert_type VARCHAR(20) NOT NULL DEFAULT 'options_flow',
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Core flow metrics
    vapi_fa_z_score DECIMAL(8,4) NOT NULL,
    dwfd_z_score DECIMAL(8,4) NOT NULL,
    tw_laf_score DECIMAL(8,4) NOT NULL,
    gib_oi_based DECIMAL(8,4) NOT NULL,
    
    -- SDAG Analysis
    sdag_multiplicative DECIMAL(8,4) NOT NULL,
    sdag_directional DECIMAL(8,4) NOT NULL,
    sdag_weighted DECIMAL(8,4) NOT NULL,
    sdag_volatility_focused DECIMAL(8,4) NOT NULL,
    
    -- DAG Analysis
    dag_multiplicative DECIMAL(8,4) NOT NULL,
    dag_additive DECIMAL(8,4) NOT NULL,
    dag_weighted DECIMAL(8,4) NOT NULL,
    dag_consensus DECIMAL(8,4) NOT NULL,
    
    -- Flow classification
    flow_type VARCHAR(30) NOT NULL,
    flow_subtype VARCHAR(30),
    flow_intensity VARCHAR(20) NOT NULL,
    directional_bias VARCHAR(20) NOT NULL,
    
    -- Participant analysis
    institutional_probability DECIMAL(5,4) NOT NULL,
    retail_probability DECIMAL(5,4) NOT NULL,
    dealer_probability DECIMAL(5,4) NOT NULL,
    
    -- Intelligence metrics
    sophistication_score DECIMAL(5,4) NOT NULL,
    information_content DECIMAL(5,4) NOT NULL,
    market_impact_potential DECIMAL(5,4) NOT NULL,
    
    -- Gamma dynamics
    gamma_exposure DECIMAL(12,2),
    delta_exposure DECIMAL(12,2),
    vanna_exposure DECIMAL(12,2),
    charm_exposure DECIMAL(12,2),
    
    -- Risk assessment
    tail_risk_probability DECIMAL(5,4),
    liquidity_risk_score DECIMAL(5,4),
    execution_risk_score DECIMAL(5,4),
    
    -- Performance metrics
    confidence_score DECIMAL(5,4) NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    data_quality_score DECIMAL(5,4) NOT NULL,
    
    -- Supporting data
    flow_breakdown JSONB DEFAULT '{}',
    participant_details JSONB DEFAULT '{}',
    risk_metrics JSONB DEFAULT '{}',
    
    -- Quality and metadata
    quality_metrics JSONB DEFAULT '{}',
    error_flags TEXT[],
    warning_flags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_flow_probabilities CHECK (
        institutional_probability + retail_probability + dealer_probability <= 1.01 AND
        institutional_probability >= 0.0 AND retail_probability >= 0.0 AND dealer_probability >= 0.0
    ),
    CONSTRAINT chk_flow_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_flow_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    CONSTRAINT chk_flow_intensity CHECK (flow_intensity IN ('low', 'medium', 'high', 'extreme')),
    CONSTRAINT chk_directional_bias CHECK (directional_bias IN ('bullish', 'bearish', 'neutral', 'mixed'))
);

-- =====================================================
-- HUIHUI SENTIMENT ANALYSIS TABLE
-- =====================================================
-- Stores comprehensive sentiment analysis results
CREATE TABLE IF NOT EXISTS huihui_sentiment_analysis (
    -- Primary identifiers
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    expert_type VARCHAR(20) NOT NULL DEFAULT 'sentiment',
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Core sentiment metrics
    overall_sentiment_score DECIMAL(6,4) NOT NULL,
    sentiment_direction VARCHAR(20) NOT NULL,
    sentiment_strength VARCHAR(20) NOT NULL,
    
    -- Sentiment components
    price_action_sentiment DECIMAL(6,4) NOT NULL,
    volume_sentiment DECIMAL(6,4) NOT NULL,
    options_sentiment DECIMAL(6,4) NOT NULL,
    news_sentiment DECIMAL(6,4) NOT NULL,
    social_sentiment DECIMAL(6,4) NOT NULL,
    
    -- Behavioral analysis
    fear_greed_index DECIMAL(6,4) NOT NULL,
    contrarian_signals DECIMAL(6,4) NOT NULL,
    crowd_psychology_state VARCHAR(30) NOT NULL,
    behavioral_biases TEXT[],
    
    -- Market microstructure sentiment
    liquidity_sentiment DECIMAL(6,4) NOT NULL,
    volatility_sentiment DECIMAL(6,4) NOT NULL,
    momentum_sentiment DECIMAL(6,4) NOT NULL,
    
    -- Risk regime analysis
    current_risk_regime VARCHAR(30) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    tail_risk_probability DECIMAL(5,4) NOT NULL,
    
    -- Sentiment dynamics
    sentiment_momentum DECIMAL(6,4) NOT NULL,
    sentiment_volatility DECIMAL(6,4) NOT NULL,
    sentiment_persistence DECIMAL(5,4) NOT NULL,
    reversal_probability DECIMAL(5,4) NOT NULL,
    
    -- Performance metrics
    confidence_score DECIMAL(5,4) NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    data_quality_score DECIMAL(5,4) NOT NULL,
    
    -- Supporting data
    sentiment_breakdown JSONB DEFAULT '{}',
    behavioral_indicators JSONB DEFAULT '{}',
    risk_indicators JSONB DEFAULT '{}',
    
    -- Quality and metadata
    quality_metrics JSONB DEFAULT '{}',
    error_flags TEXT[],
    warning_flags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_sentiment_score CHECK (overall_sentiment_score >= -1.0 AND overall_sentiment_score <= 1.0),
    CONSTRAINT chk_sentiment_direction CHECK (sentiment_direction IN ('bullish', 'bearish', 'neutral', 'mixed')),
    CONSTRAINT chk_sentiment_strength CHECK (sentiment_strength IN ('weak', 'moderate', 'strong', 'extreme')),
    CONSTRAINT chk_sentiment_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_sentiment_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'extreme')),
    CONSTRAINT chk_tail_risk_probability CHECK (tail_risk_probability >= 0.0 AND tail_risk_probability <= 1.0)
);

-- =====================================================
-- HUIHUI ORCHESTRATOR DECISIONS TABLE
-- =====================================================
-- Stores comprehensive orchestrator decision results
CREATE TABLE IF NOT EXISTS huihui_orchestrator_decisions (
    -- Primary identifiers
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    expert_type VARCHAR(20) NOT NULL DEFAULT 'orchestrator',
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Orchestration decision
    decision_type VARCHAR(30) NOT NULL,
    priority_level VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    
    -- Expert coordination
    experts_consulted TEXT[] NOT NULL,
    expert_consensus_score DECIMAL(5,4) NOT NULL,
    conflicting_signals TEXT[],
    
    -- Strategic synthesis
    market_regime_assessment TEXT NOT NULL,
    flow_assessment TEXT NOT NULL,
    sentiment_assessment TEXT NOT NULL,
    overall_market_view TEXT NOT NULL,
    
    -- Recommendations
    primary_recommendation TEXT NOT NULL,
    secondary_recommendation TEXT,
    risk_warnings TEXT[],
    opportunity_highlights TEXT[],
    
    -- Resource allocation
    component_routing JSONB DEFAULT '{}',
    parameter_adjustments JSONB DEFAULT '{}',
    resource_allocation JSONB DEFAULT '{}',
    
    -- Strategic context
    market_context JSONB DEFAULT '{}',
    risk_context JSONB DEFAULT '{}',
    opportunity_context JSONB DEFAULT '{}',
    
    -- Performance tracking
    decision_effectiveness DECIMAL(5,4),
    outcome_tracking JSONB DEFAULT '{}',
    
    -- Learning data
    decision_reasoning TEXT NOT NULL,
    learning_insights TEXT[],
    adaptation_suggestions TEXT[],
    
    -- Performance metrics
    processing_time_ms INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    data_quality_score DECIMAL(5,4) NOT NULL,
    
    -- Quality and metadata
    quality_metrics JSONB DEFAULT '{}',
    error_flags TEXT[],
    warning_flags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_orchestrator_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_expert_consensus_score CHECK (expert_consensus_score >= 0.0 AND expert_consensus_score <= 1.0),
    CONSTRAINT chk_orchestrator_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    CONSTRAINT chk_priority_level CHECK (priority_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_decision_effectiveness CHECK (decision_effectiveness IS NULL OR (decision_effectiveness >= 0.0 AND decision_effectiveness <= 1.0))
);

-- =====================================================
-- HUIHUI UNIFIED EXPERT RESPONSES TABLE
-- =====================================================
-- Stores unified responses from all experts with common metadata
CREATE TABLE IF NOT EXISTS huihui_unified_expert_responses (
    -- Primary identifiers
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    expert_type VARCHAR(20) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Expert-specific analysis IDs (only one should be populated)
    market_regime_analysis_id UUID REFERENCES huihui_market_regime_analysis(id),
    options_flow_analysis_id UUID REFERENCES huihui_options_flow_analysis(id),
    sentiment_analysis_id UUID REFERENCES huihui_sentiment_analysis(id),
    orchestrator_decision_id UUID REFERENCES huihui_orchestrator_decisions(id),
    
    -- Common response data
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    warning_message TEXT,
    
    -- Quality metrics
    overall_confidence_score DECIMAL(5,4) NOT NULL,
    data_quality_score DECIMAL(5,4) NOT NULL,
    model_performance_score DECIMAL(5,4) NOT NULL,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    validation_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    
    -- Usage tracking
    request_id VARCHAR(100),
    user_session_id VARCHAR(100),
    api_version VARCHAR(10) NOT NULL DEFAULT 'v2.5',
    
    -- Performance metrics
    total_processing_time_ms INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Metadata
    request_metadata JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_unified_confidence_score CHECK (overall_confidence_score >= 0.0 AND overall_confidence_score <= 1.0),
    CONSTRAINT chk_unified_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    CONSTRAINT chk_unified_model_performance_score CHECK (model_performance_score >= 0.0 AND model_performance_score <= 1.0),
    CONSTRAINT chk_expert_type CHECK (expert_type IN ('market_regime', 'options_flow', 'sentiment', 'orchestrator')),
    CONSTRAINT chk_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'timeout')),
    CONSTRAINT chk_validation_status CHECK (validation_status IN ('pending', 'passed', 'failed', 'warning')),
    
    -- Ensure only one expert analysis ID is populated
    CONSTRAINT chk_single_expert_reference CHECK (
        (CASE WHEN market_regime_analysis_id IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN options_flow_analysis_id IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN sentiment_analysis_id IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN orchestrator_decision_id IS NOT NULL THEN 1 ELSE 0 END) = 1
    )
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Market Regime Analysis indexes
CREATE INDEX IF NOT EXISTS idx_market_regime_ticker_time 
ON huihui_market_regime_analysis(ticker, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_market_regime_analysis_id 
ON huihui_market_regime_analysis(analysis_id);

CREATE INDEX IF NOT EXISTS idx_market_regime_current_regime 
ON huihui_market_regime_analysis(current_regime_id, timestamp DESC);

-- Options Flow Analysis indexes
CREATE INDEX IF NOT EXISTS idx_options_flow_ticker_time 
ON huihui_options_flow_analysis(ticker, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_options_flow_analysis_id 
ON huihui_options_flow_analysis(analysis_id);

CREATE INDEX IF NOT EXISTS idx_options_flow_type_intensity 
ON huihui_options_flow_analysis(flow_type, flow_intensity, timestamp DESC);

-- Sentiment Analysis indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time 
ON huihui_sentiment_analysis(ticker, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_id 
ON huihui_sentiment_analysis(analysis_id);

CREATE INDEX IF NOT EXISTS idx_sentiment_direction_strength 
ON huihui_sentiment_analysis(sentiment_direction, sentiment_strength, timestamp DESC);

-- Orchestrator Decisions indexes
CREATE INDEX IF NOT EXISTS idx_orchestrator_ticker_time 
ON huihui_orchestrator_decisions(ticker, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_orchestrator_analysis_id 
ON huihui_orchestrator_decisions(analysis_id);

CREATE INDEX IF NOT EXISTS idx_orchestrator_decision_type_priority 
ON huihui_orchestrator_decisions(decision_type, priority_level, timestamp DESC);

-- Unified Expert Responses indexes
CREATE INDEX IF NOT EXISTS idx_unified_expert_type_time 
ON huihui_unified_expert_responses(expert_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_unified_analysis_id 
ON huihui_unified_expert_responses(analysis_id);

CREATE INDEX IF NOT EXISTS idx_unified_ticker_status 
ON huihui_unified_expert_responses(ticker, status, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_unified_request_id 
ON huihui_unified_expert_responses(request_id);

-- =====================================================
-- VIEWS FOR ANALYSIS AND REPORTING
-- =====================================================

-- Current expert analysis summary
CREATE OR REPLACE VIEW huihui_current_expert_summary AS
SELECT 
    uer.expert_type,
    uer.ticker,
    uer.timestamp,
    uer.overall_confidence_score,
    uer.data_quality_score,
    uer.status,
    CASE 
        WHEN uer.expert_type = 'market_regime' THEN mra.regime_name
        WHEN uer.expert_type = 'options_flow' THEN ofa.flow_type
        WHEN uer.expert_type = 'sentiment' THEN sa.sentiment_direction
        WHEN uer.expert_type = 'orchestrator' THEN od.decision_type
    END as primary_result,
    CASE 
        WHEN uer.expert_type = 'market_regime' THEN mra.regime_confidence
        WHEN uer.expert_type = 'options_flow' THEN ofa.confidence_score
        WHEN uer.expert_type = 'sentiment' THEN sa.confidence_score
        WHEN uer.expert_type = 'orchestrator' THEN od.confidence_score
    END as expert_confidence
FROM huihui_unified_expert_responses uer
LEFT JOIN huihui_market_regime_analysis mra ON uer.market_regime_analysis_id = mra.id
LEFT JOIN huihui_options_flow_analysis ofa ON uer.options_flow_analysis_id = ofa.id
LEFT JOIN huihui_sentiment_analysis sa ON uer.sentiment_analysis_id = sa.id
LEFT JOIN huihui_orchestrator_decisions od ON uer.orchestrator_decision_id = od.id
WHERE uer.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY uer.timestamp DESC;

-- Expert performance analysis
CREATE OR REPLACE VIEW huihui_expert_performance_analysis AS
SELECT 
    expert_type,
    COUNT(*) as total_analyses,
    AVG(overall_confidence_score) as avg_confidence,
    AVG(data_quality_score) as avg_data_quality,
    AVG(total_processing_time_ms) as avg_processing_time_ms,
    COUNT(CASE WHEN status = 'completed' THEN 1 END)::DECIMAL / COUNT(*) as success_rate,
    DATE_TRUNC('hour', timestamp) as analysis_hour
FROM huihui_unified_expert_responses
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY expert_type, DATE_TRUNC('hour', timestamp)
ORDER BY analysis_hour DESC, expert_type;

-- =====================================================
-- CLEANUP FUNCTIONS
-- =====================================================

-- Function to cleanup old expert analysis data
CREATE OR REPLACE FUNCTION cleanup_old_expert_analysis_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete old market regime analyses (older than 90 days)
    DELETE FROM huihui_market_regime_analysis 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old options flow analyses (older than 90 days)
    DELETE FROM huihui_options_flow_analysis 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    -- Delete old sentiment analyses (older than 90 days)
    DELETE FROM huihui_sentiment_analysis 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    -- Delete old orchestrator decisions (older than 90 days)
    DELETE FROM huihui_orchestrator_decisions 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    -- Delete old unified responses (older than 90 days)
    DELETE FROM huihui_unified_expert_responses 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- GRANTS AND PERMISSIONS
-- =====================================================

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO huihui_app_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO huihui_app_user;
-- GRANT EXECUTE ON FUNCTION cleanup_old_expert_analysis_data() TO huihui_app_user;

-- =====================================================
-- COMPLETION MESSAGE
-- =====================================================
-- HuiHui Expert Schemas v2.5 tables created successfully!
-- 
-- Tables created:
-- 1. huihui_market_regime_analysis - VRI 3.0 components, 20 regime classifications
-- 2. huihui_options_flow_analysis - SDAG/DAG analysis, participant classification
-- 3. huihui_sentiment_analysis - Multi-component sentiment, behavioral analysis
-- 4. huihui_orchestrator_decisions - Decision tracking, expert coordination
-- 5. huihui_unified_expert_responses - Unified response structure
-- 
-- Features:
-- - Full Pydantic model compatibility
-- - Comprehensive indexing for performance
-- - Data validation constraints
-- - Cleanup functions for maintenance
-- - Analysis views for reporting
-- - Extensible design for future enhancements
-- =====================================================