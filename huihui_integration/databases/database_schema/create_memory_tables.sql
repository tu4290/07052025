-- Create Memory Intelligence Tables for EOTS v2.5
-- ================================================

-- Memory Entities Table
CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_type VARCHAR(100) NOT NULL,
    entity_name VARCHAR(255) NOT NULL,
    description TEXT,
    symbol VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- Memory Relations Table
CREATE TABLE IF NOT EXISTS memory_relations (
    relation_id VARCHAR(255) PRIMARY KEY,
    source_entity_id VARCHAR(255) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL,
    relation_type VARCHAR(100) NOT NULL,
    relation_strength DECIMAL(5,4) DEFAULT 0.5000,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_validated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT chk_relation_strength CHECK (relation_strength >= 0.0 AND relation_strength <= 1.0)
);

-- Memory Observations Table
CREATE TABLE IF NOT EXISTS memory_observations (
    observation_id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    observation_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success_rate DECIMAL(5,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}',
    related_entity_id VARCHAR(255),
    validation_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_memory_entities_symbol ON memory_entities(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_entities_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_memory_entities_confidence ON memory_entities(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_entities_created ON memory_entities(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_type ON memory_relations(relation_type);

CREATE INDEX IF NOT EXISTS idx_memory_observations_symbol ON memory_observations(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_observations_type ON memory_observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_memory_observations_timestamp ON memory_observations(timestamp DESC);
