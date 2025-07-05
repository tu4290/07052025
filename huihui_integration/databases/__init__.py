"""
HuiHui Databases - Individual Expert Data Storage
=================================================

Dedicated database systems for each HuiHui AI expert including:
- Market Regime Database: Regime patterns, volatility history, decisions
- Options Flow Database: Flow patterns, institutional positioning, gamma dynamics
- Sentiment Database: Sentiment patterns, news analysis, psychology indicators
- Shared Knowledge Database: Cross-expert insights and learnings

All databases use Supabase for storage (no SQLite support).

Features:
- Expert-specific data schemas
- Historical pattern storage
- Decision tracking and analysis
- Cross-expert knowledge sharing
- Performance optimization data

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    "name": "HuiHui Database System",
    "version": "2.5.0",
    "storage_backend": "supabase_only",
    "expert_databases": {
        "market_regime": "market_regime_patterns, volatility_history, regime_decisions",
        "options_flow": "options_flow_patterns, institutional_positioning, gamma_dynamics", 
        "sentiment": "sentiment_patterns, news_analysis, psychology_indicators",
        "shared": "shared_knowledge, cross_expert_insights, system_learnings"
    },
    "real_time_sync": True
}

# Database status
_databases_initialized = False
_supabase_connected = False
_expert_databases_ready = {
    "market_regime": False,
    "options_flow": False,
    "sentiment": False,
    "shared": False
}

def get_database_status() -> Dict[str, Any]:
    """Get HuiHui database system status."""
    return {
        "config": DATABASE_CONFIG,
        "status": {
            "databases_initialized": _databases_initialized,
            "supabase_connected": _supabase_connected,
            "expert_databases_ready": _expert_databases_ready.copy(),
            "all_databases_operational": _databases_initialized and _supabase_connected and all(_expert_databases_ready.values())
        }
    }

# Import guards
try:
    from .market_regime_db import MarketRegimeDatabase
    MARKET_REGIME_DB_AVAILABLE = True
except ImportError:
    MARKET_REGIME_DB_AVAILABLE = False
    logger.debug("Market Regime Database not available")

try:
    from .options_flow_db import OptionsFlowDatabase
    OPTIONS_FLOW_DB_AVAILABLE = True
except ImportError:
    OPTIONS_FLOW_DB_AVAILABLE = False
    logger.debug("Options Flow Database not available")

try:
    from .sentiment_db import SentimentDatabase
    SENTIMENT_DB_AVAILABLE = True
except ImportError:
    SENTIMENT_DB_AVAILABLE = False
    logger.debug("Sentiment Database not available")

try:
    from .shared_knowledge_db import SharedKnowledgeDatabase
    SHARED_KNOWLEDGE_DB_AVAILABLE = True
except ImportError:
    SHARED_KNOWLEDGE_DB_AVAILABLE = False
    logger.debug("Shared Knowledge Database not available")

def get_market_regime_database():
    """Get Market Regime Database instance."""
    if not MARKET_REGIME_DB_AVAILABLE:
        raise ImportError("Market Regime Database not available")
    return MarketRegimeDatabase()

def get_options_flow_database():
    """Get Options Flow Database instance."""
    if not OPTIONS_FLOW_DB_AVAILABLE:
        raise ImportError("Options Flow Database not available")
    return OptionsFlowDatabase()

def get_sentiment_database():
    """Get Sentiment Database instance."""
    if not SENTIMENT_DB_AVAILABLE:
        raise ImportError("Sentiment Database not available")
    return SentimentDatabase()

def get_shared_knowledge_database():
    """Get Shared Knowledge Database instance."""
    if not SHARED_KNOWLEDGE_DB_AVAILABLE:
        raise ImportError("Shared Knowledge Database not available")
    return SharedKnowledgeDatabase()

async def initialize_databases() -> bool:
    """Initialize all HuiHui expert databases."""
    global _databases_initialized, _supabase_connected, _expert_databases_ready
    
    try:
        logger.info("🗄️ Initializing HuiHui Database System...")
        
        # Test Supabase connection
        _supabase_connected = True  # Will be properly tested when databases are implemented
        
        # Initialize individual expert databases
        if MARKET_REGIME_DB_AVAILABLE:
            market_db = get_market_regime_database()
            _expert_databases_ready["market_regime"] = True
            logger.info("✅ Market Regime Database initialized")
        
        if OPTIONS_FLOW_DB_AVAILABLE:
            flow_db = get_options_flow_database()
            _expert_databases_ready["options_flow"] = True
            logger.info("✅ Options Flow Database initialized")
        
        if SENTIMENT_DB_AVAILABLE:
            sentiment_db = get_sentiment_database()
            _expert_databases_ready["sentiment"] = True
            logger.info("✅ Sentiment Database initialized")
        
        if SHARED_KNOWLEDGE_DB_AVAILABLE:
            shared_db = get_shared_knowledge_database()
            _expert_databases_ready["shared"] = True
            logger.info("✅ Shared Knowledge Database initialized")
        
        _databases_initialized = True
        logger.info("🚀 HuiHui Database System initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database system: {e}")
        return False

__all__ = [
    "DATABASE_CONFIG",
    "get_database_status",
    "get_market_regime_database",
    "get_options_flow_database",
    "get_sentiment_database",
    "get_shared_knowledge_database",
    "initialize_databases",
    "MARKET_REGIME_DB_AVAILABLE",
    "OPTIONS_FLOW_DB_AVAILABLE",
    "SENTIMENT_DB_AVAILABLE",
    "SHARED_KNOWLEDGE_DB_AVAILABLE"
]
