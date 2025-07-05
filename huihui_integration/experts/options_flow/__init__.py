"""
Options Flow Expert - HuiHui Specialist #2
==========================================

Specialized AI expert for options flow and institutional behavior analysis including:
- VAPI-FA (Volume-Adjusted Put/Call Imbalance Flow Analysis)
- DWFD (Dollar-Weighted Flow Dynamics)
- TW-LAF (Time-Weighted Large Activity Flow)
- GIB (Gamma Imbalance Barometer)
- Institutional positioning analysis
- Dealer hedging behavior

This expert focuses on understanding institutional options flow patterns
and gamma dynamics to provide flow-specific insights for the EOTS trading system.

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Expert configuration
EXPERT_CONFIG = {
    "name": "Options Flow & Institutional Behavior Expert",
    "expert_id": "options_flow",
    "specialization": "VAPI-FA, DWFD, institutional flow analysis",
    "temperature": 0.1,  # Low temperature for consistent flow analysis
    "max_tokens": 2000,
    "keywords": ["options", "flow", "VAPI-FA", "DWFD", "gamma", "institutional", "dealer"],
    "eots_metrics": ["VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Und", "GIB_OI_based_Und"],
    "database_tables": ["options_flow_patterns", "institutional_positioning", "gamma_dynamics"]
}

# Expert status
_expert_initialized = False
_database_connected = False
_learning_enabled = False

def get_expert_info() -> Dict[str, Any]:
    """Get Options Flow Expert information."""
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
    from .expert import OptionsFlowExpert
    EXPERT_AVAILABLE = True
except ImportError:
    EXPERT_AVAILABLE = False
    logger.debug("Options Flow Expert implementation not available")

try:
    from .database import OptionsFlowDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.debug("Options Flow Database not available")

try:
    from .learning import OptionsFlowLearning
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    logger.debug("Options Flow Learning not available")

try:
    from .prompts import OptionsFlowPrompts
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.debug("Options Flow Prompts not available")

def get_options_flow_expert():
    """Get Options Flow Expert instance."""
    if not EXPERT_AVAILABLE:
        raise ImportError("Options Flow Expert not available")
    return OptionsFlowExpert()

def get_options_flow_database():
    """Get Options Flow Database instance."""
    if not DATABASE_AVAILABLE:
        raise ImportError("Options Flow Database not available")
    return OptionsFlowDatabase()

def get_options_flow_learning():
    """Get Options Flow Learning instance."""
    if not LEARNING_AVAILABLE:
        raise ImportError("Options Flow Learning not available")
    return OptionsFlowLearning()

def get_options_flow_prompts():
    """Get Options Flow Prompts instance."""
    if not PROMPTS_AVAILABLE:
        raise ImportError("Options Flow Prompts not available")
    return OptionsFlowPrompts()

__all__ = [
    "EXPERT_CONFIG",
    "get_expert_info",
    "get_options_flow_expert",
    "get_options_flow_database",
    "get_options_flow_learning", 
    "get_options_flow_prompts",
    "EXPERT_AVAILABLE",
    "DATABASE_AVAILABLE",
    "LEARNING_AVAILABLE",
    "PROMPTS_AVAILABLE"
]
