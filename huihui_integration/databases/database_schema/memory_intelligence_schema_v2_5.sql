-- Memory Intelligence Schema v2.5 - "THE APEX PREDATOR'S BRAIN DATABASE"
-- ========================================================================
--
-- This schema defines the database structure for the Memory Intelligence Engine,
-- storing patterns, relations, observations, and insights for persistent learning.
--
-- Tables:
-- - memory_entities: Core knowledge entities (patterns, outcomes, insights)
-- - memory_relations: Relationships between entities
-- - memory_observations: Market observations and events
-- - memory_insights: AI-generated insights and recommendations
-- - memory_performance: Performance tracking for memory-based predictions
--
-- Author: EOTS v2.5 Development Team - "Database Memory Division"
-- Version: 2.5.0 - "PERSISTENT INTELLIGENCE SCHEMA"

-- ===================================================================
-- MEMORY ENTITIES TABLE
-- Stores core knowledge entities like patterns, outcomes, and insights
-- ===================================================================

CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id VARCHAR(36) PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_name VARCHAR(255) NOT NULL,
    description TEXT,
    symbol VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Indexes for performance
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_entity_type CHECK (entity_type IN ('pattern', 'outcome', 'insight', 'signal', 'strategy', 'news_event', 'market_regime'))
);

-- Indexes for memory_entities
CREATE INDEX IF NOT EXISTS idx_memory_entities_symbol ON memory_entities(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_entities_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_memory_entities_confidence ON memory_entities(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_entities_created ON memory_entities(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_entities_active ON memory_entities(is_active);
CREATE INDEX IF NOT EXISTS idx_memory_entities_metadata ON memory_entities USING GIN(metadata);

-- ===================================================================
-- MEMORY RELATIONS TABLE
-- Stores relationships between memory entities
-- ===================================================================

CREATE TABLE IF NOT EXISTS memory_relations (
    relation_id VARCHAR(36) PRIMARY KEY,
    source_entity_id VARCHAR(36) NOT NULL,
    target_entity_id VARCHAR(36) NOT NULL,
    relation_type VARCHAR(50) NOT NULL,
    relation_strength DECIMAL(5,4) DEFAULT 0.5000,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_validated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Foreign key constraints
    FOREIGN KEY (source_entity_id) REFERENCES memory_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES memory_entities(entity_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_relation_strength CHECK (relation_strength >= 0.0 AND relation_strength <= 1.0),
    CONSTRAINT chk_relation_type CHECK (relation_type IN ('causes', 'correlates_with', 'predicts', 'follows', 'contradicts', 'confirms', 'triggers')),
    CONSTRAINT chk_different_entities CHECK (source_entity_id != target_entity_id)
);

-- Indexes for memory_relations
CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_type ON memory_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_relations_strength ON memory_relations(relation_strength DESC);
CREATE INDEX IF NOT EXISTS idx_memory_relations_created ON memory_relations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_relations_active ON memory_relations(is_active);

-- ===================================================================
-- MEMORY OBSERVATIONS TABLE
-- Stores market observations, events, and their outcomes
-- ===================================================================

CREATE TABLE IF NOT EXISTS memory_observations (
    observation_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    observation_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success_rate DECIMAL(5,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}',
    related_entity_id VARCHAR(36),
    validation_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraint (optional relation to entity)
    FOREIGN KEY (related_entity_id) REFERENCES memory_entities(entity_id) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT chk_validation_status CHECK (validation_status IN ('pending', 'confirmed', 'rejected', 'expired'))
);

-- Indexes for memory_observations
CREATE INDEX IF NOT EXISTS idx_memory_observations_symbol ON memory_observations(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_observations_type ON memory_observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_memory_observations_timestamp ON memory_observations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memory_observations_success ON memory_observations(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_memory_observations_status ON memory_observations(validation_status);
CREATE INDEX IF NOT EXISTS idx_memory_observations_metadata ON memory_observations USING GIN(metadata);

-- ===================================================================
-- MEMORY INSIGHTS TABLE
-- Stores AI-generated insights and recommendations
-- ===================================================================

CREATE TABLE IF NOT EXISTS memory_insights (
    insight_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    insight_type VARCHAR(50) NOT NULL,
    insight_text TEXT NOT NULL,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    source_entities TEXT[], -- Array of entity IDs that contributed to this insight
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    validation_outcome VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT chk_insight_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_validation_outcome CHECK (validation_outcome IS NULL OR validation_outcome IN ('correct', 'incorrect', 'partial', 'unknown'))
);

-- Indexes for memory_insights
CREATE INDEX IF NOT EXISTS idx_memory_insights_symbol ON memory_insights(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_insights_type ON memory_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_memory_insights_confidence ON memory_insights(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_insights_created ON memory_insights(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_insights_expires ON memory_insights(expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_insights_active ON memory_insights(is_active);

-- ===================================================================
-- MEMORY PERFORMANCE TABLE
-- Tracks performance of memory-based predictions and recommendations
-- ===================================================================

CREATE TABLE IF NOT EXISTS memory_performance (
    performance_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_data JSONB NOT NULL,
    actual_outcome JSONB,
    accuracy_score DECIMAL(5,4),
    related_entities TEXT[], -- Array of entity IDs used in prediction
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    outcome_timestamp TIMESTAMP WITH TIME ZONE,
    evaluation_status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    CONSTRAINT chk_accuracy_score CHECK (accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0)),
    CONSTRAINT chk_evaluation_status CHECK (evaluation_status IN ('pending', 'evaluated', 'expired', 'invalid'))
);

-- Indexes for memory_performance
CREATE INDEX IF NOT EXISTS idx_memory_performance_symbol ON memory_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_performance_type ON memory_performance(prediction_type);
CREATE INDEX IF NOT EXISTS idx_memory_performance_accuracy ON memory_performance(accuracy_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_performance_prediction_time ON memory_performance(prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memory_performance_status ON memory_performance(evaluation_status);

-- ===================================================================
-- MEMORY ANALYTICS VIEWS
-- Useful views for analyzing memory intelligence performance
-- ===================================================================

-- View: Entity Performance Summary
CREATE OR REPLACE VIEW memory_entity_performance AS
SELECT 
    e.entity_id,
    e.entity_name,
    e.entity_type,
    e.symbol,
    e.confidence_score,
    COUNT(r.relation_id) as relation_count,
    AVG(r.relation_strength) as avg_relation_strength,
    e.access_count,
    e.created_at,
    e.last_accessed
FROM memory_entities e
LEFT JOIN memory_relations r ON (e.entity_id = r.source_entity_id OR e.entity_id = r.target_entity_id)
WHERE e.is_active = TRUE
GROUP BY e.entity_id, e.entity_name, e.entity_type, e.symbol, e.confidence_score, e.access_count, e.created_at, e.last_accessed;

-- View: Symbol Intelligence Summary
CREATE OR REPLACE VIEW memory_symbol_summary AS
SELECT 
    symbol,
    COUNT(DISTINCT entity_id) as total_entities,
    COUNT(DISTINCT CASE WHEN entity_type = 'pattern' THEN entity_id END) as pattern_count,
    COUNT(DISTINCT CASE WHEN entity_type = 'outcome' THEN entity_id END) as outcome_count,
    AVG(confidence_score) as avg_confidence,
    MAX(created_at) as last_entity_created,
    COUNT(DISTINCT observation_id) as observation_count
FROM memory_entities e
LEFT JOIN memory_observations o USING (symbol)
WHERE e.is_active = TRUE
GROUP BY symbol;

-- View: Recent High-Confidence Insights
CREATE OR REPLACE VIEW memory_recent_insights AS
SELECT 
    i.insight_id,
    i.symbol,
    i.insight_type,
    i.insight_text,
    i.confidence_score,
    i.created_at,
    i.expires_at,
    CASE 
        WHEN i.expires_at IS NULL OR i.expires_at > NOW() THEN 'active'
        ELSE 'expired'
    END as status
FROM memory_insights i
WHERE i.is_active = TRUE 
    AND i.confidence_score >= 0.7
    AND i.created_at >= NOW() - INTERVAL '7 days'
ORDER BY i.confidence_score DESC, i.created_at DESC;

-- ===================================================================
-- MEMORY MAINTENANCE FUNCTIONS
-- Functions for maintaining and optimizing memory data
-- ===================================================================

-- Function: Update entity access tracking
CREATE OR REPLACE FUNCTION update_entity_access(entity_id_param VARCHAR(36))
RETURNS VOID AS $$
BEGIN
    UPDATE memory_entities 
    SET last_accessed = NOW(), 
        access_count = access_count + 1
    WHERE entity_id = entity_id_param;
END;
$$ LANGUAGE plpgsql;

-- Function: Clean expired insights
CREATE OR REPLACE FUNCTION clean_expired_insights()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    UPDATE memory_insights 
    SET is_active = FALSE 
    WHERE expires_at IS NOT NULL 
        AND expires_at < NOW() 
        AND is_active = TRUE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate pattern success rate
CREATE OR REPLACE FUNCTION calculate_pattern_success_rate(pattern_entity_id VARCHAR(36))
RETURNS DECIMAL(5,4) AS $$
DECLARE
    success_rate DECIMAL(5,4);
BEGIN
    SELECT AVG(
        CASE 
            WHEN (mp.metadata->>'outcome_data'->>'success')::boolean = true THEN 1.0
            ELSE 0.0
        END
    ) INTO success_rate
    FROM memory_performance mp
    WHERE pattern_entity_id = ANY(mp.related_entities)
        AND mp.evaluation_status = 'evaluated';
    
    RETURN COALESCE(success_rate, 0.0);
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- INITIAL DATA AND TRIGGERS
-- ===================================================================

-- Trigger: Update last_accessed when entity is queried
CREATE OR REPLACE FUNCTION trigger_update_entity_access()
RETURNS TRIGGER AS $$
BEGIN
    -- This would be called by application logic when entities are accessed
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO eots_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO eots_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO eots_user;

-- ===================================================================
-- SCHEMA COMPLETE
-- ===================================================================

-- Memory Intelligence Schema v2.5 installation complete!
-- This schema provides the foundation for persistent, learning intelligence
-- that remembers patterns, connects insights, and evolves over time.
--
-- Next steps:
-- 1. Run this schema against your database
-- 2. Configure the Memory Intelligence Engine to use these tables
-- 3. Integrate with MCP servers for enhanced intelligence
-- 4. Start collecting and storing trading patterns
-- 5. Watch your system become smarter over time!
--
-- "THE APEX PREDATOR'S BRAIN IS NOW READY TO LEARN AND REMEMBER!"
