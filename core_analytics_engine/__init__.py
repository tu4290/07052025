
"""
EOTS v2.5 Core Analytics Engine - HUIHUI AI INTEGRATION
========================================================

This module contains the core analytics components of the EOTS system,
now fully integrated with the HuiHui Expert Coordinator system.

All AI intelligence is now routed through HuiHui Expert Coordinator:
- Market Regime Expert for VRI analysis and regime detection
- Options Flow Expert for VAPI-FA, DWFD, and institutional flow analysis  
- Sentiment Expert for market intelligence and behavioral analysis
- Expert Coordinator for intelligent consensus and coordination

Author: EOTS v2.5 Development Team - "HuiHui AI Integration Division"
"""

# Core analytics modules
from .its_orchestrator_v2_5 import ITSOrchestratorV2_5
from .eots_metrics import MetricsCalculatorV2_5
from .market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from .market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from .news_intelligence_engine_v2_5 import NewsIntelligenceEngineV2_5

# HuiHui AI-integrated modules (replacing legacy static modules)
from .huihui_ai_integration_v2_5 import (
    HuiHuiAIIntegrationV2_5,
    get_unified_ai_intelligence_system,
    run_unified_learning_for_symbol
)
from .adaptive_learning_integration_v2_5 import (
    AdaptiveLearningIntegrationV2_5,
    get_adaptive_learning_integration,
    run_daily_unified_learning,
    run_weekly_unified_learning
)
from .atif_engine_v2_5 import ATIFEngineV2_5
from .ai_predictions_manager_v2_5 import AIPredictionsManagerV2_5

# MCP integration
from .mcp_unified_manager_v2_5 import MCPUnifiedManagerV2_5

# Elite Metrics Definitions - Updated to use consolidated elite_intelligence
from .eots_metrics.elite_intelligence import MarketRegime, FlowType, EliteConfig, ConvexValueColumns, EliteImpactColumns

# Export all components
__all__ = [
    # Core analytics
    'ITSOrchestratorV2_5',
    'MetricsCalculatorV2_5', 
    'MarketIntelligenceEngineV2_5',
    'MarketRegimeEngineV2_5',
    'NewsIntelligenceEngineV2_5',
    
    # HuiHui AI Integration (NEW)
    'HuiHuiAIIntegrationV2_5',
    'get_unified_ai_intelligence_system',
    'run_unified_learning_for_symbol',
    'AdaptiveLearningIntegrationV2_5',
    'get_adaptive_learning_integration', 
    'run_daily_unified_learning',
    'run_weekly_unified_learning',
    'ATIFEngineV2_5',
    'AIPredictionsManagerV2_5',
    
    # MCP integration
    'MCPUnifiedManagerV2_5',

    # Elite Metrics Definitions
    'MarketRegime',
    'FlowType',
    'EliteConfig',
    'ConvexValueColumns',
    'EliteImpactColumns'
]

# Legacy module aliases for backward compatibility during transition
UnifiedAIIntelligenceSystemV2_5 = HuiHuiAIIntegrationV2_5
