"""
HuiHui Experts - Specialized AI Experts
=======================================

The 3 specialized HuiHui AI experts for EOTS v2.5:

1. Market Regime Expert - VRI analysis, volatility patterns, regime detection
2. Options Flow Expert - VAPI-FA, DWFD, institutional flow analysis
3. Sentiment Expert - News analysis, market psychology, sentiment indicators

Note: The 4th expert (Meta-Orchestrator) is integrated as its_orchestrator_v2_5.py
in the core_analytics_engine directory.

Each expert has:
- Dedicated learning algorithms
- Individual database storage
- Specialized prompts and fine-tuning
- Performance tracking and optimization

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Expert availability status
EXPERT_STATUS = {
    "market_regime": False,
    "options_flow": False,
    "sentiment": False
}

# Expert metadata
EXPERT_METADATA = {
    "market_regime": {
        "name": "Market Regime & Volatility Analysis Expert",
        "specialization": "VRI analysis, volatility patterns, regime detection",
        "temperature": 0.1,
        "max_tokens": 2000,
        "keywords": ["regime", "volatility", "VRI", "market", "structure"]
    },
    "options_flow": {
        "name": "Options Flow & Institutional Behavior Expert", 
        "specialization": "VAPI-FA, DWFD, institutional flow analysis",
        "temperature": 0.1,
        "max_tokens": 2000,
        "keywords": ["options", "flow", "VAPI-FA", "DWFD", "gamma", "institutional"]
    },
    "sentiment": {
        "name": "Sentiment & News Intelligence Expert",
        "specialization": "News analysis, market psychology, sentiment indicators",
        "temperature": 0.15,  # Slightly higher for sentiment nuance
        "max_tokens": 2000,
        "keywords": ["sentiment", "news", "psychology", "fear", "greed", "social"]
    }
}

def get_expert_status() -> Dict[str, Any]:
    """Get status of all HuiHui experts."""
    return {
        "experts": EXPERT_STATUS.copy(),
        "metadata": EXPERT_METADATA,
        "all_available": all(EXPERT_STATUS.values()),
        "available_count": sum(EXPERT_STATUS.values())
    }

def get_expert_metadata(expert_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for specific expert."""
    return EXPERT_METADATA.get(expert_name)

# Import guards for experts
try:
    from .market_regime.expert import MarketRegimeExpert
    EXPERT_STATUS["market_regime"] = True
    logger.debug("Market Regime Expert available")
except ImportError:
    logger.debug("Market Regime Expert not available")

try:
    from .options_flow.expert import OptionsFlowExpert
    EXPERT_STATUS["options_flow"] = True
    logger.debug("Options Flow Expert available")
except ImportError:
    logger.debug("Options Flow Expert not available")

try:
    from .sentiment.expert import SentimentExpert
    EXPERT_STATUS["sentiment"] = True
    logger.debug("Sentiment Expert available")
except ImportError:
    logger.debug("Sentiment Expert not available")

# Convenience functions
def get_available_experts() -> List[str]:
    """Get list of available expert names."""
    return [name for name, available in EXPERT_STATUS.items() if available]

def is_expert_available(expert_name: str) -> bool:
    """Check if specific expert is available."""
    return EXPERT_STATUS.get(expert_name, False)

__all__ = [
    "EXPERT_STATUS",
    "EXPERT_METADATA", 
    "get_expert_status",
    "get_expert_metadata",
    "get_available_experts",
    "is_expert_available"
]
