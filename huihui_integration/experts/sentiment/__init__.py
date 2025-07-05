"""
Sentiment Expert - HuiHui Specialist #3
=======================================

Specialized AI expert for sentiment and news intelligence analysis including:
- News sentiment analysis and interpretation
- Market psychology indicators
- Social media sentiment tracking
- Fear/greed cycle analysis
- Contrarian signal detection
- Behavioral analysis insights

This expert focuses on understanding market psychology and sentiment dynamics
to provide sentiment-specific insights for the EOTS trading system.

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Expert configuration
EXPERT_CONFIG = {
    "name": "Sentiment & News Intelligence Expert",
    "expert_id": "sentiment",
    "specialization": "News analysis, market psychology, sentiment indicators",
    "temperature": 0.15,  # Slightly higher temperature for sentiment nuance
    "max_tokens": 2000,
    "keywords": ["sentiment", "news", "psychology", "fear", "greed", "social", "behavioral"],
    "eots_metrics": ["sentiment_score", "news_impact", "fear_greed_index", "social_sentiment"],
    "database_tables": ["sentiment_patterns", "news_analysis", "psychology_indicators"],
    "data_sources": ["Alpha Vantage", "Brave Search", "HotNews", "Social Media APIs"]
}

# Expert status
_expert_initialized = False
_database_connected = False
_learning_enabled = False

def get_expert_info() -> Dict[str, Any]:
    """Get Sentiment Expert information."""
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
    from .expert import SentimentExpert
    EXPERT_AVAILABLE = True
except ImportError:
    EXPERT_AVAILABLE = False
    logger.debug("Sentiment Expert implementation not available")

try:
    from .database import SentimentDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.debug("Sentiment Database not available")

try:
    from .learning import SentimentLearning
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    logger.debug("Sentiment Learning not available")

try:
    from .prompts import SentimentPrompts
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.debug("Sentiment Prompts not available")

def get_sentiment_expert():
    """Get Sentiment Expert instance."""
    if not EXPERT_AVAILABLE:
        raise ImportError("Sentiment Expert not available")
    return SentimentExpert()

def get_sentiment_database():
    """Get Sentiment Database instance."""
    if not DATABASE_AVAILABLE:
        raise ImportError("Sentiment Database not available")
    return SentimentDatabase()

def get_sentiment_learning():
    """Get Sentiment Learning instance."""
    if not LEARNING_AVAILABLE:
        raise ImportError("Sentiment Learning not available")
    return SentimentLearning()

def get_sentiment_prompts():
    """Get Sentiment Prompts instance."""
    if not PROMPTS_AVAILABLE:
        raise ImportError("Sentiment Prompts not available")
    return SentimentPrompts()

__all__ = [
    "EXPERT_CONFIG",
    "get_expert_info",
    "get_sentiment_expert",
    "get_sentiment_database",
    "get_sentiment_learning",
    "get_sentiment_prompts", 
    "EXPERT_AVAILABLE",
    "DATABASE_AVAILABLE",
    "LEARNING_AVAILABLE",
    "PROMPTS_AVAILABLE"
]
