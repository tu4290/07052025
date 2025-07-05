"""
Market Regime Expert - HuiHui Specialist #1
===========================================

Specialized AI expert for market regime and volatility analysis including:
- VRI (Volatility Regime Indicator) analysis
- Regime transition detection
- Volatility pattern recognition
- Structural market analysis
- Risk assessment and warnings

This expert focuses on understanding market structure and volatility dynamics
to provide regime-specific insights for the EOTS trading system.

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Expert configuration
EXPERT_CONFIG = {
    "name": "Market Regime & Volatility Analysis Expert",
    "expert_id": "market_regime",
    "specialization": "VRI analysis, volatility patterns, regime detection",
    "temperature": 0.1,  # Low temperature for consistent regime analysis
    "max_tokens": 2000,
    "keywords": ["regime", "volatility", "VRI", "market", "structure", "risk"],
    "eots_metrics": ["VRI_2_0_Und", "volatility_clustering", "regime_transitions"],
    "database_tables": ["market_regime_patterns", "volatility_history", "regime_decisions"]
}

# Expert status
_expert_initialized = False
_database_connected = False
_learning_enabled = False

def get_expert_info() -> Dict[str, Any]:
    """Get Market Regime Expert information."""
    return {
        "config": EXPERT_CONFIG,
        "status": {
            "initialized": _expert_initialized,
            "database_connected": _database_connected,
            "learning_enabled": _learning_enabled
        }
    }

# Import guards
try:
    from .expert import MarketRegimeExpert
    EXPERT_AVAILABLE = True
except ImportError:
    EXPERT_AVAILABLE = False
    logger.debug("Market Regime Expert implementation not available")

try:
    from .database import MarketRegimeDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.debug("Market Regime Database not available")

try:
    from .learning import MarketRegimeLearning
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    logger.debug("Market Regime Learning not available")

try:
    from .prompts import MarketRegimePrompts
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.debug("Market Regime Prompts not available")

def get_market_regime_expert():
    """Get Market Regime Expert instance."""
    if not EXPERT_AVAILABLE:
        raise ImportError("Market Regime Expert not available")
    return MarketRegimeExpert()

def get_market_regime_database():
    """Get Market Regime Database instance."""
    if not DATABASE_AVAILABLE:
        raise ImportError("Market Regime Database not available")
    return MarketRegimeDatabase()

def get_market_regime_learning():
    """Get Market Regime Learning instance."""
    if not LEARNING_AVAILABLE:
        raise ImportError("Market Regime Learning not available")
    return MarketRegimeLearning()

def get_market_regime_prompts():
    """Get Market Regime Prompts instance."""
    if not PROMPTS_AVAILABLE:
        raise ImportError("Market Regime Prompts not available")
    return MarketRegimePrompts()

__all__ = [
    "EXPERT_CONFIG",
    "get_expert_info",
    "get_market_regime_expert",
    "get_market_regime_database", 
    "get_market_regime_learning",
    "get_market_regime_prompts",
    "EXPERT_AVAILABLE",
    "DATABASE_AVAILABLE",
    "LEARNING_AVAILABLE",
    "PROMPTS_AVAILABLE"
]
