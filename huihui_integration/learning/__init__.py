"""
HuiHui Learning - Advanced Learning and Feedback Systems
========================================================

Advanced learning systems for HuiHui AI experts including:
- Individual expert learning algorithms
- Cross-expert knowledge sharing
- Performance-based optimization
- Feedback loop management
- Continuous improvement systems

Features:
- Expert-specific learning models
- Knowledge transfer between experts
- Performance tracking and optimization
- Adaptive prompt engineering
- Success/failure pattern analysis

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Learning configuration
LEARNING_CONFIG = {
    "name": "HuiHui Learning System",
    "version": "2.5.0",
    "learning_enabled": True,
    "experts_learning": ["market_regime", "options_flow", "sentiment"],
    "learning_methods": ["feedback_loops", "performance_tracking", "knowledge_sharing"],
    "optimization_enabled": True
}

# Learning status
_learning_initialized = False
_feedback_enabled = False
_knowledge_sharing_enabled = False

def get_learning_status() -> Dict[str, Any]:
    """Get HuiHui learning system status."""
    return {
        "config": LEARNING_CONFIG,
        "status": {
            "learning_initialized": _learning_initialized,
            "feedback_enabled": _feedback_enabled,
            "knowledge_sharing_enabled": _knowledge_sharing_enabled,
            "all_systems_active": all([
                _learning_initialized,
                _feedback_enabled,
                _knowledge_sharing_enabled
            ])
        }
    }

# Import guards
try:
    from .feedback_loops import HuiHuiFeedbackSystem, get_feedback_system
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    logger.debug("Feedback System not available")

try:
    from .performance_tracking import HuiHuiPerformanceTracker
    PERFORMANCE_TRACKING_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKING_AVAILABLE = False
    logger.debug("Performance Tracking not available")

try:
    from .knowledge_sharing import HuiHuiKnowledgeSharing
    KNOWLEDGE_SHARING_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SHARING_AVAILABLE = False
    logger.debug("Knowledge Sharing not available")

def get_feedback_system_instance():
    """Get HuiHui Feedback System instance."""
    if not FEEDBACK_AVAILABLE:
        raise ImportError("Feedback System not available")
    return get_feedback_system()

async def initialize_learning() -> bool:
    """Initialize the complete HuiHui learning system."""
    global _learning_initialized, _feedback_enabled, _knowledge_sharing_enabled
    
    try:
        logger.info("🧠 Initializing HuiHui Learning System...")
        
        # Initialize feedback system
        if FEEDBACK_AVAILABLE:
            feedback_system = get_feedback_system_instance()
            _feedback_enabled = True
            logger.info("✅ Feedback system initialized")
        
        # Initialize performance tracking
        if PERFORMANCE_TRACKING_AVAILABLE:
            logger.info("✅ Performance tracking available")
        
        # Initialize knowledge sharing
        if KNOWLEDGE_SHARING_AVAILABLE:
            _knowledge_sharing_enabled = True
            logger.info("✅ Knowledge sharing initialized")
        
        _learning_initialized = True
        logger.info("🚀 HuiHui Learning System initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize learning system: {e}")
        return False

__all__ = [
    "LEARNING_CONFIG",
    "get_learning_status",
    "get_feedback_system_instance",
    "initialize_learning",
    "FEEDBACK_AVAILABLE",
    "PERFORMANCE_TRACKING_AVAILABLE",
    "KNOWLEDGE_SHARING_AVAILABLE"
]
