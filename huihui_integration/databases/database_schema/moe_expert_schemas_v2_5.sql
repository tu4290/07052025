-- =====================================================
-- MOE (MIXTURE OF EXPERTS) SCHEMAS v2.5 - SUPABASE SCHEMA
-- =====================================================
-- 
-- Comprehensive Mixture of Experts database schema for EOTS v2.5
-- Supports expert routing, gating networks, performance tracking,
-- and hierarchical expert coordination with full Pydantic compatibility.
--
-- Author: EOTS v2.5 MOE Schema Division
-- Version: 2.5.0 - "MOE FOUNDATION SCHEMA"
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- MOE EXPERT REGISTRY TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_expert_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expert_id VARCHAR(50) NOT NULL UNIQUE,
    expert_name VARCHAR(100) NOT NULL,
    expert_type VARCHAR(30) NOT NULL,
    expert_category VARCHAR(30) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    specialization_domains TEXT[] NOT NULL,
    capability_scores JSONB DEFAULT '{}',
    success_rate DECIMAL(5,4) DEFAULT 0.0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0.0,
    total_queries_processed INTEGER DEFAULT 0,
    confidence_reliability DECIMAL(5,4) DEFAULT 0.0,
    computational_cost DECIMAL(8,4) DEFAULT 0.0,
    memory_requirements_mb INTEGER DEFAULT 0,
    max_concurrent_requests INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'active',
    availability_score DECIMAL(5,4) DEFAULT 1.0,
    last_health_check TIMESTAMPTZ DEFAULT NOW(),
    description TEXT,
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_expert_status CHECK (status IN ('active', 'inactive', 'maintenance', 'deprecated')),
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT chk_availability_score CHECK (availability_score >= 0.0 AND availability_score <= 1.0)
);

-- =====================================================
-- MOE GATING NETWORK TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_gating_network (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    routing_id VARCHAR(100) NOT NULL UNIQUE,
    request_id VARCHAR(100) NOT NULL,
    input_text TEXT NOT NULL,
    input_complexity_score DECIMAL(5,4) NOT NULL,
    input_domain_classification VARCHAR(50) NOT NULL,
    selected_expert_id VARCHAR(50) NOT NULL REFERENCES moe_expert_registry(expert_id),
    selection_confidence DECIMAL(5,4) NOT NULL,
    selection_reasoning TEXT NOT NULL,
    alternative_experts JSONB DEFAULT '[]',
    expert_scores JSONB DEFAULT '{}',
    routing_strategy VARCHAR(30) NOT NULL,
    context_factors JSONB DEFAULT '{}',
    market_conditions JSONB DEFAULT '{}',
    user_preferences JSONB DEFAULT '{}',
    predicted_success_probability DECIMAL(5,4) NOT NULL,
    predicted_response_time_ms DECIMAL(8,2) NOT NULL,
    predicted_confidence_score DECIMAL(5,4) NOT NULL,
    actual_success BOOLEAN,
    actual_response_time_ms DECIMAL(8,2),
    actual_confidence_score DECIMAL(5,4),
    routing_accuracy_score DECIMAL(5,4),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    processing_time_ms DECIMAL(8,2) NOT NULL,
    CONSTRAINT chk_gating_selection_confidence CHECK (selection_confidence >= 0.0 AND selection_confidence <= 1.0),
    CONSTRAINT chk_gating_routing_strategy CHECK (routing_strategy IN ('confidence_based', 'load_balanced', 'cost_optimized', 'performance_optimized', 'hybrid'))
);

-- =====================================================
-- MOE EXPERT RESPONSES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_expert_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    response_id VARCHAR(100) NOT NULL UNIQUE,
    routing_id VARCHAR(100) NOT NULL REFERENCES moe_gating_network(routing_id),
    expert_id VARCHAR(50) NOT NULL REFERENCES moe_expert_registry(expert_id),
    request_text TEXT NOT NULL,
    request_type VARCHAR(30) NOT NULL,
    request_priority INTEGER DEFAULT 3,
    response_text TEXT NOT NULL,
    response_confidence DECIMAL(5,4) NOT NULL,
    response_quality_score DECIMAL(5,4) NOT NULL,
    processing_time_ms DECIMAL(8,2) NOT NULL,
    tokens_used INTEGER,
    computational_cost DECIMAL(8,4),
    relevance_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    completeness_score DECIMAL(5,4),
    coherence_score DECIMAL(5,4),
    context_used JSONB DEFAULT '{}',
    model_parameters JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT TRUE,
    error_type VARCHAR(50),
    error_message TEXT,
    warnings TEXT[],
    request_timestamp TIMESTAMPTZ NOT NULL,
    response_timestamp TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_response_confidence CHECK (response_confidence >= 0.0 AND response_confidence <= 1.0),
    CONSTRAINT chk_request_priority CHECK (request_priority >= 1 AND request_priority <= 5)
);

-- =====================================================
-- MOE ENSEMBLE DECISIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_ensemble_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ensemble_id VARCHAR(100) NOT NULL UNIQUE,
    request_id VARCHAR(100) NOT NULL,
    participating_experts TEXT[] NOT NULL,
    expert_weights JSONB DEFAULT '{}',
    consensus_strategy VARCHAR(30) NOT NULL,
    expert_responses JSONB DEFAULT '{}',
    expert_confidences JSONB DEFAULT '{}',
    expert_processing_times JSONB DEFAULT '{}',
    consensus_score DECIMAL(5,4) NOT NULL,
    agreement_level VARCHAR(20) NOT NULL,
    conflicting_opinions JSONB DEFAULT '[]',
    majority_opinion TEXT,
    final_response TEXT NOT NULL,
    final_confidence DECIMAL(5,4) NOT NULL,
    decision_quality_score DECIMAL(5,4) NOT NULL,
    weight_distribution JSONB DEFAULT '{}',
    aggregation_method VARCHAR(30) NOT NULL,
    uncertainty_quantification DECIMAL(5,4),
    total_processing_time_ms DECIMAL(8,2) NOT NULL,
    ensemble_efficiency_score DECIMAL(5,4),
    cost_benefit_ratio DECIMAL(8,4),
    ensemble_accuracy DECIMAL(5,4),
    prediction_calibration DECIMAL(5,4),
    robustness_score DECIMAL(5,4),
    market_context JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_ensemble_consensus_score CHECK (consensus_score >= 0.0 AND consensus_score <= 1.0),
    CONSTRAINT chk_ensemble_final_confidence CHECK (final_confidence >= 0.0 AND final_confidence <= 1.0),
    CONSTRAINT chk_ensemble_agreement_level CHECK (agreement_level IN ('high', 'medium', 'low', 'conflicted'))
);

-- =====================================================
-- MOE EXPERT PERFORMANCE TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_expert_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expert_id VARCHAR(50) NOT NULL REFERENCES moe_expert_registry(expert_id),
    performance_period_start TIMESTAMPTZ NOT NULL,
    performance_period_end TIMESTAMPTZ NOT NULL,
    total_queries INTEGER NOT NULL DEFAULT 0,
    successful_queries INTEGER NOT NULL DEFAULT 0,
    failed_queries INTEGER NOT NULL DEFAULT 0,
    timeout_queries INTEGER NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,4) NOT NULL,
    avg_response_time_ms DECIMAL(8,2) NOT NULL,
    median_response_time_ms DECIMAL(8,2) NOT NULL,
    p95_response_time_ms DECIMAL(8,2) NOT NULL,
    avg_confidence_score DECIMAL(5,4) NOT NULL,
    avg_quality_score DECIMAL(5,4) NOT NULL,
    confidence_calibration DECIMAL(5,4) NOT NULL,
    tokens_per_second DECIMAL(8,2),
    cost_per_query DECIMAL(8,4),
    resource_utilization DECIMAL(5,4),
    domain_performance JSONB DEFAULT '{}',
    query_type_performance JSONB DEFAULT '{}',
    complexity_performance JSONB DEFAULT '{}',
    relative_performance_score DECIMAL(5,4),
    rank_among_peers INTEGER,
    improvement_trend VARCHAR(20),
    adaptation_rate DECIMAL(5,4),
    learning_effectiveness DECIMAL(5,4),
    knowledge_retention DECIMAL(5,4),
    performance_notes TEXT,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_perf_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0)
);

-- =====================================================
-- MOE SYSTEM HEALTH TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS moe_system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    health_check_id VARCHAR(100) NOT NULL UNIQUE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    total_active_experts INTEGER NOT NULL,
    total_inactive_experts INTEGER NOT NULL,
    system_availability DECIMAL(5,4) NOT NULL,
    overall_performance_score DECIMAL(5,4) NOT NULL,
    cpu_utilization DECIMAL(5,4),
    memory_utilization DECIMAL(5,4),
    network_utilization DECIMAL(5,4),
    storage_utilization DECIMAL(5,4),
    avg_system_response_time_ms DECIMAL(8,2) NOT NULL,
    throughput_queries_per_second DECIMAL(8,2) NOT NULL,
    error_rate DECIMAL(5,4) NOT NULL,
    timeout_rate DECIMAL(5,4) NOT NULL,
    avg_response_quality DECIMAL(5,4) NOT NULL,
    user_satisfaction_score DECIMAL(5,4),
    system_reliability_score DECIMAL(5,4) NOT NULL,
    overall_health_status VARCHAR(20) NOT NULL,
    critical_issues INTEGER DEFAULT 0,
    warning_issues INTEGER DEFAULT 0,
    expert_health_summary JSONB DEFAULT '{}',
    performance_trends JSONB DEFAULT '{}',
    capacity_status JSONB DEFAULT '{}',
    active_alerts JSONB DEFAULT '[]',
    resolved_alerts JSONB DEFAULT '[]',
    maintenance_windows JSONB DEFAULT '[]',
    health_check_duration_ms DECIMAL(8,2) NOT NULL,
    health_check_version VARCHAR(10) DEFAULT '2.5',
    CONSTRAINT chk_system_availability CHECK (system_availability >= 0.0 AND system_availability <= 1.0),
    CONSTRAINT chk_system_health_status CHECK (overall_health_status IN ('excellent', 'good', 'fair', 'poor', 'critical'))
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_expert_registry_type_status ON moe_expert_registry(expert_type, status);
CREATE INDEX IF NOT EXISTS idx_gating_network_timestamp ON moe_gating_network(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_expert_responses_expert_time ON moe_expert_responses(expert_id, response_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ensemble_decisions_timestamp ON moe_ensemble_decisions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_expert_performance_period ON moe_expert_performance(expert_id, performance_period_start DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON moe_system_health(timestamp DESC);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================
CREATE OR REPLACE VIEW moe_current_expert_status AS
SELECT 
    er.expert_id,
    er.expert_name,
    er.expert_type,
    er.status,
    er.success_rate,
    er.avg_response_time_ms,
    er.total_queries_processed,
    CASE 
        WHEN er.last_health_check > NOW() - INTERVAL '5 minutes' THEN 'healthy'
        WHEN er.last_health_check > NOW() - INTERVAL '15 minutes' THEN 'warning'
        ELSE 'critical'
    END as health_status
FROM moe_expert_registry er
WHERE er.status = 'active';

-- =====================================================
-- INSERT DEFAULT EXPERTS
-- =====================================================
INSERT INTO moe_expert_registry (
    expert_id, expert_name, expert_type, expert_category,
    model_name, model_version, specialization_domains, capability_scores
) VALUES 
('huihui_market_regime', 'HuiHui Market Regime Expert', 'market_analysis', 'financial',
 'huihui-moe-abliterated', '5b-a1.7b', ARRAY['market_regime', 'volatility', 'trend_analysis'], 
 '{"accuracy": 0.85, "speed": 0.90, "reliability": 0.88}'),

('huihui_options_flow', 'HuiHui Options Flow Expert', 'options_analysis', 'financial',
 'huihui-moe-abliterated', '5b-a1.7b', ARRAY['options_flow', 'gamma_exposure', 'participant_analysis'],
 '{"accuracy": 0.87, "speed": 0.85, "reliability": 0.90}'),

('huihui_sentiment', 'HuiHui Sentiment Expert', 'sentiment_analysis', 'financial',
 'huihui-moe-abliterated', '5b-a1.7b', ARRAY['sentiment', 'behavioral_analysis', 'risk_regime'],
 '{"accuracy": 0.82, "speed": 0.92, "reliability": 0.85}'),

('huihui_orchestrator', 'HuiHui Meta-Orchestrator', 'meta_orchestration', 'coordination',
 'huihui-moe-abliterated', '5b-a1.7b', ARRAY['coordination', 'synthesis', 'decision_making'],
 '{"accuracy": 0.88, "speed": 0.80, "reliability": 0.92}')

ON CONFLICT (expert_id) DO UPDATE SET
    updated_at = NOW(),
    model_version = EXCLUDED.model_version,
    capability_scores = EXCLUDED.capability_scores;
