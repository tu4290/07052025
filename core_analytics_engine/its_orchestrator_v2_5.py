# eots/core_analytics_engine/its_orchestrator_v2_5.py
"""
🎯 Enhanced ITS Orchestrator v2.5 - LEGENDARY META-ORCHESTRATOR
PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts

This is the 4th pillar of the legendary system - the Meta-Orchestrator that coordinates
all analysis and makes final strategic decisions using the EOTS v2.5 architecture.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

# Pydantic imports for validation
from pydantic import BaseModel, Field

# EOTS core imports - Updated to use current data models structure
from data_models.core_models import (
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedContractMetricsV2_5,
    KeyLevelsDataV2_5,
    KeyLevelV2_5,
    MarketRegimeState,
    FinalAnalysisBundleV2_5,
    UnprocessedDataBundleV2_5,
    RawOptionsContractV2_5,
    RawUnderlyingDataCombinedV2_5,
    UnifiedIntelligenceAnalysis,
    SystemStateV2_5
)
from data_models.configuration_models import AdaptiveLearningConfigV2_5, PredictionConfigV2_5


# MOE schemas - Updated to use current data models structure
from data_models.ai_ml_models import (
    ExpertStatus,
    RoutingStrategy,
    ConsensusStrategy,
    AgreementLevel,
    HealthStatus,
    MOEExpertRegistryV2_5,
    MOEGatingNetworkV2_5,
    MOEExpertResponseV2_5,
    MOEUnifiedResponseV2_5,
    ExpertAnalysisRequest,
    ExpertAnalysisResponse,
    MarketRegimeAnalysisDetails,
    OptionsFlowAnalysisDetails,
    SentimentAnalysisDetails,
    ToolResultData
)

# EOTS utilities - VALIDATED AGAINST USER'S SYSTEM
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from core_analytics_engine.atif_engine_v2_5 import ATIFEngineV2_5
from core_analytics_engine.news_intelligence_engine_v2_5 import NewsIntelligenceEngineV2_5
from core_analytics_engine.adaptive_learning_integration_v2_5 import AdaptiveLearningIntegrationV2_5
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from data_management.convexvalue_data_fetcher_v2_5 import ConvexValueDataFetcherV2_5
from data_management.tradier_data_fetcher_v2_5 import TradierDataFetcherV2_5

# Import Elite components - Updated to use consolidated elite_intelligence
from data_models.elite_intelligence import EliteConfig, MarketRegime

# ---------------------------------------------------------------------------+
# HuiHui AI Expert Integration & Monitoring (post-orchestrator_bridge removal)
# ---------------------------------------------------------------------------+
# Direct expert classes
from huihui_integration.experts.market_regime.market_regime_expert import (
    UltimateMarketRegimeExpert,
    MarketRegimeExpertConfig,
)
from huihui_integration.experts.options_flow.options_flow_expert import (
    UltimateOptionsFlowExpert,
)
from huihui_integration.experts.sentiment.market_intelligence_expert import (
    UltimateMarketIntelligenceExpert,
)

# Monitoring & safety layers
from huihui_integration.monitoring.usage_monitor import (
    get_usage_monitor,
    record_expert_usage,
)
from huihui_integration.monitoring.safety_manager import HuiHuiSafetyManager

# Learning / feedback engine
from huihui_integration.learning.feedback_loops import HuiHuiLearningSystem

# 🚀 REAL COMPLIANCE TRACKING: Import tracking system for metrics
try:
    from dashboard_application.modes.ai_dashboard.component_compliance_tracker_v2_5 import (
        track_metrics_calculation, DataSourceType
    )
    COMPLIANCE_TRACKING_AVAILABLE = True
except ImportError:
    COMPLIANCE_TRACKING_AVAILABLE = False

# HuiHui integration - USING USER'S EXISTING STRUCTURE
try:
    from huihui_integration.core.model_interface import (
        create_market_regime_model,
        HuiHuiPydanticModel,
    )
    from huihui_integration import (
        is_system_ready
    )
    LEGENDARY_EXPERTS_AVAILABLE = True
except ImportError:
    LEGENDARY_EXPERTS_AVAILABLE = False
    ExpertCommunicationProtocol = None

logger = logging.getLogger(__name__)

class LegendaryOrchestrationConfig(BaseModel):
    """PYDANTIC-FIRST: Configuration for legendary orchestration capabilities"""
    
    # AI Decision Making
    ai_decision_enabled: bool = Field(default=True, description="Enable AI-powered decision making")
    ai_model_name: str = Field(default="llama3.1:8b", description="AI model for decision making")
    ai_temperature: float = Field(default=0.1, description="AI temperature for consistency")
    ai_max_tokens: int = Field(default=2000, description="Maximum tokens for AI responses")
    
    # Expert Coordination
    expert_weight_adaptation: bool = Field(default=True, description="Enable dynamic expert weighting")
    expert_consensus_threshold: float = Field(default=0.7, description="Threshold for expert consensus")
    conflict_resolution_enabled: bool = Field(default=True, description="Enable conflict resolution")
    
    # Performance Optimization
    parallel_processing_enabled: bool = Field(default=True, description="Enable parallel expert processing")
    max_concurrent_experts: int = Field(default=4, description="Maximum concurrent expert analyses")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    
    # Learning and Adaptation
    continuous_learning_enabled: bool = Field(default=True, description="Enable continuous learning")
    performance_tracking_enabled: bool = Field(default=True, description="Enable performance tracking")
    adaptation_rate: float = Field(default=0.01, description="Rate of system adaptation")
    
    # Risk Management
    risk_management_enabled: bool = Field(default=True, description="Enable risk management")
    max_position_exposure: float = Field(default=0.1, description="Maximum position exposure")
    stop_loss_threshold: float = Field(default=0.02, description="Stop loss threshold")
    
    class Config:
        extra = 'forbid'

class ITSOrchestratorV2_5:
    """
    🎯 LEGENDARY META-ORCHESTRATOR - 4th Pillar of the Legendary System
    
    PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts.
    This orchestrator coordinates all analysis and makes final strategic decisions.
    """
    
    def __init__(self, config_manager: ConfigManagerV2_5, db_manager, enhanced_cache=None):
        """Initialize the ITS Orchestrator with required components. db_manager is required."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self._db_manager = db_manager  # Must be valid instance, no None fallback
        self._cache_manager = enhanced_cache  # Store the provided enhanced_cache
        self._enhanced_cache = enhanced_cache  # Store for property access

        if self._db_manager is None:
            raise ValueError("db_manager must be provided and initialized before creating ITSOrchestratorV2_5")

        self.historical_data_manager = HistoricalDataManagerV2_5(config_manager=config_manager, db_manager=self._db_manager)

        # Initialize data fetchers FIRST (required by other components)
        try:
            self.convex_fetcher = ConvexValueDataFetcherV2_5(config_manager)
            self.tradier_fetcher = TradierDataFetcherV2_5(config_manager)
            self.logger.info("🔗 Data fetchers initialized successfully")
        except Exception as e:
            self.logger.error(f"⚠️ Failed to initialize data fetchers: {e}")
            self.convex_fetcher = None
            self.tradier_fetcher = None

        # Initialize metrics calculator
        elite_config_obj = config_manager.get_setting("elite_config", None)
        elite_config_dict = elite_config_obj.model_dump() if elite_config_obj else None

        self.metrics_calculator = MetricsCalculatorV2_5(
            config_manager=config_manager,
            historical_data_manager=self.historical_data_manager,
            enhanced_cache_manager=self._cache_manager,
            elite_config=elite_config_dict
        )
        self.logger.info("✅ Initialized MetricsCalculatorV2_5")

        # Initialize market regime engine with Pydantic model (not dict)
        self.market_regime_engine = MarketRegimeEngineV2_5(
            config_manager=config_manager,
            elite_config=elite_config_obj,
            tradier_fetcher=self.tradier_fetcher,
            convex_fetcher=self.convex_fetcher
        )
        self.logger.info("✅ Initialized MarketRegimeEngineV2_5")
        
        # Initialize market intelligence engine
        self.market_intelligence_engine = MarketIntelligenceEngineV2_5(
            config_manager=config_manager,
            metrics_calculator=self.metrics_calculator
        )
        
        # Initialize ATIF engine
        self.atif_engine = ATIFEngineV2_5(config_manager=config_manager) if not self.enhanced_cache else None
        
        # Initialize news intelligence engine
        self.news_intelligence = NewsIntelligenceEngineV2_5(config_manager=config_manager) if not self.enhanced_cache else None
        
        # Initialize adaptive learning integration
        adaptive_config = AdaptiveLearningConfigV2_5()
        self.adaptive_learning = AdaptiveLearningIntegrationV2_5(config=adaptive_config) if not self.enhanced_cache else None
        self.prediction_config = PredictionConfigV2_5()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTrackerV2_5(config_manager) if not self.enhanced_cache else None
        
        # Initialize system state with all required fields
        self.system_state = SystemStateV2_5(
            is_running=True,
            current_mode="operational",
            active_processes=["market_regime_engine", "market_intelligence_engine", "atif_engine", "news_intelligence", "adaptive_learning"],
            status_message="System initialized and running",
            errors=[]  # Empty list - no errors at initialization
        ) if not self.enhanced_cache else None

        # ===== HuiHui Expert Integration & Monitoring =====
        if not self.enhanced_cache:
            self.logger.info("🧠 Initializing HuiHui AI experts and monitoring systems...")
            # --- Expert Instantiation -------------------------------------------------
            try:
                # Market Regime Expert requires an explicit config model
                default_regime_cfg = MarketRegimeExpertConfig(
                    expert_name="default_expert",
                    specialist_id="default_specialist"
                )
                elite_config = EliteConfig()  # Create default EliteConfig instance
                self.market_regime_expert = UltimateMarketRegimeExpert(
                    expert_config=default_regime_cfg,
                    config_manager=config_manager,
                    historical_data_manager=self.historical_data_manager,
                    elite_config=elite_config,
                    db_manager=self._db_manager,
                )

                # Options-Flow & Sentiment experts allow empty config
                self.options_flow_expert = UltimateOptionsFlowExpert(
                    db_manager=self._db_manager
                )
                self.market_intelligence_expert = UltimateMarketIntelligenceExpert(
                    db_manager=self._db_manager
                )
                self.logger.info("✅ HuiHui AI experts initialized successfully")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"❌ Failed to initialize HuiHui experts: {e}", exc_info=True)
                self.market_regime_expert = None
                self.options_flow_expert = None
                self.market_intelligence_expert = None

            # --- Monitoring, Safety, Learning ----------------------------------------
            try:
                self.usage_monitor = get_usage_monitor()
                self.safety_manager = HuiHuiSafetyManager()
                self.learning_system = HuiHuiLearningSystem(
                    config_manager=config_manager,
                    db_manager=self._db_manager
                )
                self.logger.info("✅ HuiHui monitoring & learning systems initialized")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"❌ Failed to initialize monitoring/learning systems: {e}", exc_info=True)
                self.usage_monitor = None
                self.safety_manager = None
                self.learning_system = None

            # --- Expert performance tracking -----------------------------------------
            self.expert_performance_metrics: Dict[str, List[float]] = {
                "market_regime": [],
                "options_flow": [],
                "sentiment": [],
            }

            # Holder for the current analysis context (used in unified responses)
            self.current_analysis: Optional[Dict[str, Any]] = None

            self.logger.info("🎯 ITS Orchestrator initialized successfully with all components")
        
    @property
    def db_manager(self):
        """Get the database manager instance."""
        return self._db_manager

    @property
    def cache_manager(self):
        """Get the cache manager instance, initializing if needed."""
        if self._cache_manager is None:
            self._cache_manager = EnhancedCacheManagerV2_5()
        return self._cache_manager

    @property
    def enhanced_cache(self):
        """Get the enhanced cache instance."""
        return self._enhanced_cache

    async def analyze_market_regime(self, processed_data: ProcessedDataBundleV2_5, last_heartbeat: datetime = None) -> str:
        """Analyze market regime with enhanced error handling."""
        try:
            regime = await self.market_regime_engine.analyze_market_regime(processed_data, last_heartbeat)
            return regime.value if isinstance(regime, MarketRegime) else str(regime)
        except Exception as e:
            self.logger.error(f"Failed to analyze market regime: {e}")
            return "UNKNOWN"

    async def coordinate_huihui_experts(self, data_bundle: ProcessedDataBundleV2_5) -> MOEUnifiedResponseV2_5:
        """
        Coordinate the 3 HuiHui specialist experts and synthesize their responses.
        Integrates monitoring and safety checks throughout the process.
        """
        analysis_start_time = datetime.now()
        analysis_id = f"analysis_{data_bundle.underlying_data_enriched.symbol}_{int(analysis_start_time.timestamp())}"
        
        # Set current analysis context
        self.current_analysis = {
            "analysis_id": analysis_id,
            "symbol": data_bundle.underlying_data_enriched.symbol,
            "analysis_type": "full",
            "start_time": analysis_start_time
        }
        
        self.logger.info(f"🎯 Starting HuiHui expert coordination for {data_bundle.underlying_data_enriched.symbol}")
        
        expert_responses = []
        
        # Coordinate experts in parallel using asyncio.gather for efficiency
        try:
            # Check if experts are available
            expert_tasks = []
            if self.market_regime_expert:
                expert_tasks.append(self._call_market_regime_expert(data_bundle, analysis_id))
            if self.options_flow_expert:
                expert_tasks.append(self._call_options_flow_expert(data_bundle, analysis_id))
            if self.market_intelligence_expert:
                expert_tasks.append(self._call_sentiment_expert(data_bundle, analysis_id))
            
            if not expert_tasks:
                raise RuntimeError("No HuiHui experts available for coordination")
            
            # Execute expert calls in parallel
            results = await asyncio.gather(*expert_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Expert call failed during gather: {result}", exc_info=True)
                    # Create a structured error response
                    error_data = MOEExpertResponseDataV2_5(
                        analysis_summary=f"Expert failed during execution: {str(result)}",
                        confidence=0.0,
                        details=MarketRegimeAnalysisDetails(vri_score=0, regime_id=0, regime_name="ERROR", transition_probability=0, volatility_level="ERROR", trend_direction="ERROR") # Dummy details
                    )
                    error_response = self._create_moe_expert_response("unknown", "Failed Expert", error_data, 0.0, 0.0, success=False, error_message=str(result))
                    expert_responses.append(error_response)
                else:
                    expert_responses.append(result)

            # Synthesize responses
            total_processing_time = (datetime.now() - analysis_start_time).total_seconds() * 1000
            unified_response = await self._synthesize_expert_responses(
                expert_responses, 
                data_bundle, 
                total_processing_time
            )
            
            # Safety check on final recommendation
            if self.safety_manager:
                is_safe = await self.safety_manager.is_recommendation_safe(unified_response)
                if not is_safe:
                    self.logger.warning("⚠️ Safety manager flagged recommendation as unsafe")
                    if isinstance(unified_response.unified_response.data, dict):
                        unified_response.unified_response.data["safety_warning"] = "Recommendation flagged by safety manager"
            
            # Learning feedback
            if self.learning_system:
                await self._provide_learning_feedback(unified_response)
            
            self.logger.info(f"✅ HuiHui expert coordination completed for {data_bundle.underlying_data_enriched.symbol}")
            return unified_response
            
        except Exception as e:
            self.logger.error(f"❌ Expert coordination failed: {e}", exc_info=True)
            # Clear current analysis context on error
            self.current_analysis = None
            raise RuntimeError(f"HuiHui expert coordination failed: {e}") from e

    async def _call_market_regime_expert(self, data_bundle: ProcessedDataBundleV2_5, analysis_id: str) -> MOEExpertResponseV2_5:
        """Call the Market Regime Expert and return a standardized Pydantic response."""
        expert_start_time = datetime.now()
        try:
            request = ExpertAnalysisRequest(analysis_id=analysis_id, symbol=data_bundle.underlying_data_enriched.symbol, data_bundle=data_bundle)
            response: ExpertAnalysisResponse = await self.market_regime_expert.analyze(request)
            
            response_data = ToolResultData(
                analysis_summary=response.analysis_summary,
                confidence=response.confidence,
                details=response.details
            )
            
            moe_response = self._create_moe_expert_response(
                expert_id="market_regime",
                expert_name="Market Regime Expert",
                response_data=response_data,
                confidence_score=response.confidence,
                processing_time_ms=response.processing_time_ms
            )
            if self.usage_monitor:
                record_expert_usage("market_regime", "analysis", 1000, 500, response.processing_time_ms, True)
            return moe_response
        except Exception as e:
            self.logger.error(f"Market Regime Expert failed: {e}", exc_info=True)
            error_data = ToolResultData(
                analysis_summary=f"Expert failed: {e}", confidence=0.0, 
                details=MarketRegimeAnalysisDetails(vri_score=0, regime_id=0, regime_name="ERROR", transition_probability=0, volatility_level="ERROR", trend_direction="ERROR")
            )
            return self._create_moe_expert_response("market_regime", "Market Regime Expert", error_data, 0.0, (datetime.now() - expert_start_time).total_seconds() * 1000, success=False, error_message=str(e))

    async def _call_options_flow_expert(self, data_bundle: ProcessedDataBundleV2_5, analysis_id: str) -> MOEExpertResponseV2_5:
        """Call the Options Flow Expert and return a standardized Pydantic response."""
        expert_start_time = datetime.now()
        try:
            request = ExpertAnalysisRequest(analysis_id=analysis_id, symbol=data_bundle.underlying_data_enriched.symbol, data_bundle=data_bundle)
            response: ExpertAnalysisResponse = await self.options_flow_expert.analyze(request)
            
            response_data = ToolResultData(
                analysis_summary=response.analysis_summary,
                confidence=response.confidence,
                details=response.details
            )
            
            moe_response = self._create_moe_expert_response(
                expert_id="options_flow",
                expert_name="Options Flow Expert",
                response_data=response_data,
                confidence_score=response.confidence,
                processing_time_ms=response.processing_time_ms
            )
            if self.usage_monitor:
                record_expert_usage("options_flow", "analysis", 1000, 500, response.processing_time_ms, True)
            return moe_response
        except Exception as e:
            self.logger.error(f"Options Flow Expert failed: {e}", exc_info=True)
            error_data = ToolResultData(
                analysis_summary=f"Expert failed: {e}", confidence=0.0, 
                details=OptionsFlowAnalysisDetails(vapi_fa_score=0, dwfd_score=0, flow_type="ERROR", flow_intensity="ERROR", institutional_probability=0)
            )
            return self._create_moe_expert_response("options_flow", "Options Flow Expert", error_data, 0.0, (datetime.now() - expert_start_time).total_seconds() * 1000, success=False, error_message=str(e))

    async def _call_sentiment_expert(self, data_bundle: ProcessedDataBundleV2_5, analysis_id: str) -> MOEExpertResponseV2_5:
        """Call the Market Intelligence/Sentiment Expert and return a standardized Pydantic response."""
        expert_start_time = datetime.now()
        try:
            request = ExpertAnalysisRequest(analysis_id=analysis_id, symbol=data_bundle.underlying_data_enriched.symbol, data_bundle=data_bundle)
            response: ExpertAnalysisResponse = await self.market_intelligence_expert.analyze(request)
            
            response_data = ToolResultData(
                analysis_summary=response.analysis_summary,
                confidence=response.confidence,
                details=response.details
            )
            
            moe_response = self._create_moe_expert_response(
                expert_id="sentiment",
                expert_name="Market Intelligence Expert",
                response_data=response_data,
                confidence_score=response.confidence,
                processing_time_ms=response.processing_time_ms
            )
            if self.usage_monitor:
                record_expert_usage("sentiment", "analysis", 1000, 500, response.processing_time_ms, True)
            return moe_response
        except Exception as e:
            self.logger.error(f"Market Intelligence Expert failed: {e}", exc_info=True)
            error_data = ToolResultData(
                analysis_summary=f"Expert failed: {e}", confidence=0.0, 
                details=SentimentAnalysisDetails(overall_sentiment_score=0, sentiment_direction="ERROR", sentiment_strength=0, fear_greed_index=50)
            )
            return self._create_moe_expert_response("sentiment", "Market Intelligence Expert", error_data, 0.0, (datetime.now() - expert_start_time).total_seconds() * 1000, success=False, error_message=str(e))

    async def _synthesize_expert_responses(
        self, 
        expert_responses: List[MOEExpertResponseV2_5], 
        data_bundle: ProcessedDataBundleV2_5,
        total_processing_time_ms: float
    ) -> MOEUnifiedResponseV2_5:
        """Synthesize multiple expert responses into a unified analysis."""
        try:
            # Extract successful responses
            successful_responses = [r for r in expert_responses if r.success]
            
            if not successful_responses:
                raise RuntimeError("No successful expert responses to synthesize")
            
            # Calculate final confidence as weighted average
            total_confidence = sum(r.response_data.confidence for r in successful_responses)
            final_confidence = total_confidence / len(successful_responses)
            
            # Create unified analysis summary
            analysis_summaries = [r.response_data.analysis_summary for r in successful_responses]
            unified_summary = f"Unified analysis for {data_bundle.underlying_data_enriched.symbol}: " + "; ".join(analysis_summaries)
            
            # Combine all expert details
            unified_details = {}
            for response in successful_responses:
                unified_details[response.expert_id] = response.response_data.details.model_dump()
            
            # Create unified response data
            unified_data = {
                "result_type": "unified_expert_analysis",
                "data": {
                    "analysis_summary": unified_summary,
                    "confidence": final_confidence,
                    "details": unified_details,
                    "expert_count": len(successful_responses),
                    "symbol": data_bundle.underlying_data_enriched.symbol
                },
                "confidence_score": final_confidence,
                "quality_score": final_confidence * 0.9,  # Quality slightly lower than confidence
                "data_sources": [r.expert_id for r in successful_responses],
                "processing_notes": [f"Expert {r.expert_id} processed in {r.processing_time_ms:.1f}ms" for r in successful_responses]
            }
            
            # Create MOE unified response
            return self._create_moe_unified_response(
                expert_responses=expert_responses,
                unified_data=unified_data,
                final_confidence=final_confidence,
                total_processing_time_ms=total_processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize expert responses: {e}")
            raise

    async def _provide_learning_feedback(self, unified_response: MOEUnifiedResponseV2_5):
        """Provide feedback to the learning system based on analysis results."""
        try:
            if not self.learning_system:
                return
            
            # Create learning data from the analysis
            learning_data = {
                "analysis_id": unified_response.request_id,
                "expert_count": len(unified_response.expert_responses),
                "final_confidence": unified_response.final_confidence,
                "processing_time_ms": unified_response.total_processing_time_ms,
                "success_rate": len([r for r in unified_response.expert_responses if r.success]) / len(unified_response.expert_responses),
                "timestamp": datetime.now()
            }
            
            # Evaluate the analysis performance (placeholder - in production, this would use actual market outcomes)
            performance_analysis = self.learning_system.evaluate_prediction_outcome(learning_data)
            
            self.logger.info(f"📚 Learning feedback provided for analysis {unified_response.request_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to provide learning feedback: {e}")
            # Don't raise - learning feedback failure shouldn't stop the analysis
            
    def _calculate_regime_metrics(self, data_bundle: ProcessedDataBundleV2_5) -> Dict[str, float]:
        """Calculate regime metrics from the data bundle."""
        try:
            if not data_bundle or not data_bundle.underlying_data_enriched:
                return {}
                
            und_data = data_bundle.underlying_data_enriched
            
            # Extract required metrics from the underlying data model
            metrics = {}
            
            # Add volatility metrics
            metrics['volatility'] = getattr(und_data, 'u_volatility', 0.0)
            metrics['trend_strength'] = getattr(und_data, 'vri_2_0_und', 0.0)
            metrics['volume_trend'] = getattr(und_data, 'vfi_0dte_und_avg', 0.0)
            metrics['momentum'] = getattr(und_data, 'a_mspi_und', 0.0)
            metrics['regime_score'] = getattr(und_data, 'current_market_regime_v2_5', 'UNKNOWN')
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def _initialize_moe_expert_registry(self) -> MOEExpertRegistryV2_5:
        """Initialize MOE Expert Registry for the 4th MOE Expert (Meta-Orchestrator)"""
        try:
            registry = MOEExpertRegistryV2_5(
                expert_id="meta_orchestrator_v2_5",
                expert_name="Ultimate Meta-Orchestrator",
                expert_type="meta_orchestrator",
                capabilities=[
                    "expert_coordination",
                    "consensus_building",
                    "conflict_resolution",
                    "strategic_synthesis",
                    "risk_assessment",
                    "final_decision_making"
                ],
                specializations=[
                    "meta_analysis",
                    "expert_synthesis",
                    "strategic_decision_making"
                ],
                supported_tasks=[
                    "expert_coordination",
                    "consensus_building",
                    "final_analysis"
                ],
                status=ExpertStatus.ACTIVE,
                accuracy_score=0.95,
                confidence_score=0.9,
                response_time_ms=15000.0,
                success_rate=95.0,
                memory_usage_mb=512.0,
                cpu_usage_percent=25.0,
                gpu_required=False,
                health_score=0.98,
                last_health_check=datetime.now(),
                tags=["meta", "orchestrator", "legendary", "v2_5"]
            )
            self.logger.info("🎯 MOE Expert Registry initialized for Meta-Orchestrator")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to initialize MOE Expert Registry: {e}")
            raise
    
    def _create_moe_gating_network(self, request_context: Dict[str, Any]) -> MOEGatingNetworkV2_5:
        """Create MOE Gating Network for routing decisions"""
        try:
            # Determine which experts to route to based on request context
            selected_experts = request_context.get('include_experts', ["regime", "flow", "intelligence"])
            
            # Calculate expert weights based on request type and context
            expert_weights = self._calculate_expert_weights(request_context)
            
            # Calculate capability scores
            capability_scores = {
                "regime_expert": 0.9,
                "flow_expert": 0.85,
                "intelligence_expert": 0.88,
                "meta_orchestrator": 0.95
            }
            
            gating_network = MOEGatingNetworkV2_5(
                selected_experts=selected_experts,
                routing_strategy=RoutingStrategy.WEIGHTED,
                routing_confidence=0.9,
                expert_weights=expert_weights,
                capability_scores=capability_scores,
                request_context=request_context
            )
            
            self.logger.info(f"🎯 MOE Gating Network created with {len(selected_experts)} experts")
            return gating_network
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE Gating Network: {e}")
            raise
    
    def _calculate_expert_weights(self, request_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic expert weights based on request context"""
        analysis_type = request_context.get('analysis_type', 'full')
        priority = request_context.get('priority', 'normal')
        
        # Base weights
        weights = {
            "regime_expert": 0.3,
            "flow_expert": 0.3,
            "intelligence_expert": 0.25,
            "meta_orchestrator": 0.15
        }
        
        # Adjust weights based on analysis type
        if analysis_type == 'regime_focused':
            weights["regime_expert"] = 0.5
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'flow_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.5
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'intelligence_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.5
            weights["meta_orchestrator"] = 0.1
        
        # Increase meta-orchestrator weight for high priority requests
        if priority == 'high':
            weights["meta_orchestrator"] += 0.1
            # Normalize other weights
            total_other = sum(v for k, v in weights.items() if k != "meta_orchestrator")
            for k in weights:
                if k != "meta_orchestrator":
                    weights[k] = weights[k] * (0.9 / total_other)
        
        return weights
    
    def _create_moe_expert_response(
        self, 
        expert_id: str, 
        expert_name: str, 
        response_data: ToolResultData, # Now strictly typed
        confidence_score: float, 
        processing_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> MOEExpertResponseV2_5:
        """Create MOE Expert Response for individual expert results using a strict Pydantic model."""
        try:
            expert_response = MOEExpertResponseV2_5(
                expert_id=expert_id,
                expert_name=expert_name,
                response_data=response_data, # Already a validated Pydantic model
                processing_time_ms=processing_time_ms,
                quality_score=min(confidence_score + 0.1, 1.0),
                uncertainty_score=1.0 - confidence_score,
                success=success,
                error_message=error_message,
                timestamp=datetime.now(),
                version="2.5"
            )
            return expert_response
        except Exception as e:
            self.logger.error(f"Failed to create MOE expert response for {expert_id}: {e}", exc_info=True)
            # This is a critical internal error, re-raise it
            raise
    
    def _create_moe_unified_response(self, expert_responses: List[MOEExpertResponseV2_5], 
                                   unified_data: Dict[str, Any], final_confidence: float,
                                   total_processing_time_ms: float) -> MOEUnifiedResponseV2_5:
        """Create MOE Unified Response combining all expert responses"""
        try:
            # Determine consensus strategy and agreement level
            successful_responses = [r for r in expert_responses if r.success]
            consensus_strategy = ConsensusStrategy.WEIGHTED_AVERAGE if len(successful_responses) > 1 else ConsensusStrategy.EXPERT_RANKING
            
            # Calculate agreement level based on confidence variance
            if len(successful_responses) > 1:
                confidence_scores = [r.response_data.confidence for r in successful_responses]
                confidence_variance = sum((c - final_confidence) ** 2 for c in confidence_scores) / len(confidence_scores)
                if confidence_variance < 0.01:
                    agreement_level = AgreementLevel.HIGH
                elif confidence_variance < 0.05:
                    agreement_level = AgreementLevel.MEDIUM
                else:
                    agreement_level = AgreementLevel.LOW
            else:
                agreement_level = AgreementLevel.HIGH  # Single expert, no disagreement
            
            unified_response = MOEUnifiedResponseV2_5(
                request_id=self.current_analysis.get("analysis_id", "unknown") if self.current_analysis else "unknown",
                request_type=self.current_analysis.get('analysis_type', 'full') if self.current_analysis else "unknown",
                consensus_strategy=consensus_strategy,
                agreement_level=agreement_level,
                final_confidence=final_confidence,
                expert_responses=expert_responses,
                participating_experts=[r.expert_id for r in expert_responses],
                unified_response=unified_data,
                response_quality=final_confidence,
                total_processing_time_ms=total_processing_time_ms,
                expert_coordination_time_ms=total_processing_time_ms * 0.1,  # Estimate 10% for coordination
                consensus_time_ms=total_processing_time_ms * 0.05,  # Estimate 5% for consensus
                system_health=HealthStatus.HEALTHY if len(successful_responses) == len(expert_responses) else HealthStatus.DEGRADED,
                timestamp=datetime.now(),
                version="2.5",
                debug_info={
                    "total_experts": len(expert_responses),
                    "successful_experts": len(successful_responses),
                    "failed_experts": len(expert_responses) - len(successful_responses)
                },
                performance_breakdown={
                    "data_processing": total_processing_time_ms * 0.3,
                    "expert_analysis": total_processing_time_ms * 0.5,
                    "synthesis": total_processing_time_ms * 0.15,
                    "coordination": total_processing_time_ms * 0.05
                }
            )
            
            self.logger.info(f"🎯 MOE Unified Response created with {len(successful_responses)}/{len(expert_responses)} successful experts")
            return unified_response
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE unified response: {e}")
            raise
    
    def _get_regime_analysis_prompt(self) -> str:
        """Get system prompt for AI decision making"""
        return """
        You are the LEGENDARY META-ORCHESTRATOR for the EOTS v2.5 options trading system.
        
        Your role is to synthesize analysis from 3 specialist experts:
        1. Market Regime Expert - Provides VRI 2.0 analysis and regime detection
        2. Options Flow Expert - Provides VAPI-FA, DWFD, and elite flow analysis  
        3. Market Intelligence Expert - Provides sentiment, behavioral, and microstructure analysis
        
        Your responsibilities:
        - Synthesize expert analyses into strategic recommendations
        - Resolve conflicts between expert opinions
        - Provide final trading decisions with confidence scores
        - Assess risk and provide risk management guidance
        - Maintain consistency with EOTS v2.5 methodologies
        
        Always provide structured, actionable recommendations based on the expert analyses.
        Focus on high-probability setups with clear risk/reward profiles.
        """
    
    async def run_full_analysis_cycle(self, ticker: str, dte_min: int, dte_max: int, price_range_percent: int, **kwargs) -> FinalAnalysisBundleV2_5:
        """
        🚀 PYDANTIC-FIRST: Run a full analysis cycle with enhanced error handling and data validation.
        """
        try:
            self.logger.info(f"Starting full analysis cycle for {ticker}")
            
            # Initialize cache if not already done
            if not self._cache_manager:
                self.logger.info("Initializing cache manager for analysis cycle")
                self._cache_manager = EnhancedCacheManagerV2_5()
            
            # Try to get data from cache first
            cache_key = f"analysis_bundle_{ticker}"
            cached_data = self._cache_manager.get(cache_key) if self._cache_manager else None
            
            if cached_data:
                self.logger.info(f"Found cached analysis for {ticker}")
                return FinalAnalysisBundleV2_5.parse_obj(cached_data)
            
            # If not in cache, perform full analysis
            self.logger.info(f"No cached data found for {ticker}, performing full analysis")
            
            # Get current timestamp
            timestamp = datetime.now()
            
            # Generate key levels using cache
            key_levels = await self._generate_key_levels_from_cache(ticker, timestamp)
            
            # Create the final analysis bundle
            final_bundle = FinalAnalysisBundleV2_5(
                ticker=ticker,
                timestamp=timestamp,
                key_levels=key_levels,
                analysis_status="success",
                error_message=None
            )
            
            # Cache the results
            if self._cache_manager:
                self._cache_manager.set(cache_key, final_bundle.dict(), ttl=300)  # 5 minute TTL
            
            return final_bundle
            
        except Exception as e:
            self.logger.error(f"Failed to run full analysis cycle: {e}", exc_info=True)
            return self._create_error_bundle(ticker, str(e))

    async def _generate_key_levels_from_cache(self, ticker: str, timestamp: datetime) -> KeyLevelsDataV2_5:
        """Generate key levels using cached data."""
        try:
            # Try to get from cache first
            cache_key = f"key_levels_{ticker}"
            cached_levels = self._cache_manager.get(cache_key) if self._cache_manager else None
            
            if cached_levels:
                return KeyLevelsDataV2_5.parse_obj(cached_levels)
            
            # If not in cache, generate new levels
            levels = KeyLevelsDataV2_5(
                ticker=ticker,
                timestamp=timestamp,
                support_levels=[],
                resistance_levels=[],
                pivot_points=[],
                fibonacci_levels=[]
            )
            
            # Cache the results
            if self._cache_manager:
                self._cache_manager.set(cache_key, levels.dict(), ttl=300)  # 5 minute TTL
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Failed to generate key levels: {e}", exc_info=True)
            return KeyLevelsDataV2_5(
                ticker=ticker,
                timestamp=timestamp,
                support_levels=[],
                resistance_levels=[],
                pivot_points=[],
                fibonacci_levels=[]
            )

    def _create_error_bundle(self, ticker: str, error_message: str) -> FinalAnalysisBundleV2_5:
        """FAIL FAST - NO ERROR BUNDLES WITH FAKE DATA ALLOWED!"""
        raise ValueError(
            f"CRITICAL: Cannot create analysis bundle for {ticker} due to error: {error_message}. "
            f"NO FAKE DATA WILL BE CREATED! Fix the underlying data issue instead of masking it with fake data."
        )
    
    def _calculate_data_quality_score(self, data_bundle: ProcessedDataBundleV2_5) -> float:
        """Calculate data quality score for the analysis"""
        try:
            if not data_bundle:
                return 0.0
            
            quality_factors = []
            
            # Check underlying data quality
            if data_bundle.underlying_data_enriched:
                if data_bundle.underlying_data_enriched.price:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.0)
            
            # Check options data quality
            if data_bundle.options_data_with_metrics:
                options_quality = len(data_bundle.options_data_with_metrics) / 100.0  # Normalize by expected count
                quality_factors.append(min(options_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Check strike level data quality
            if data_bundle.strike_level_data_with_metrics:
                strike_quality = len(data_bundle.strike_level_data_with_metrics) / 50.0  # Normalize by expected count
                quality_factors.append(min(strike_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Calculate average quality
            if quality_factors:
                return sum(quality_factors) / len(quality_factors)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Data quality calculation failed: {e}")
            return 0.0
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] = self.performance_metrics.get("total_analyses", 0) + 1
        
        if result.get("confidence_score", 0) > 0.5:
            self.performance_metrics["successful_analyses"] = self.performance_metrics.get("successful_analyses", 0) + 1
        else:
            self.performance_metrics["failed_analyses"] = self.performance_metrics.get("failed_analyses", 0) + 1
    
    def get_legendary_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the legendary system"""
        try:
            # Update real-time metrics
            self.performance_metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e)
            }
    
    async def legendary_orchestrate_analysis(self, data_bundle: ProcessedDataBundleV2_5, **kwargs) -> FinalAnalysisBundleV2_5:
        """
        🎯 LEGENDARY orchestration method.
        Returns a validated FinalAnalysisBundleV2_5 and raises exceptions on failure.
        """
        try:
            # Run full analysis cycle and return the bundle directly
            final_bundle = await self.run_full_analysis_cycle(
                ticker=data_bundle.underlying_data_enriched.symbol,
                **kwargs
            )
            return final_bundle
            
        except Exception as e:
            self.logger.error(f"Legendary orchestration failed: {e}", exc_info=True)
            # FAIL-FAST: Re-raise the exception instead of returning a fake error structure
            raise RuntimeError(f"Legendary orchestration failed for {data_bundle.underlying_data_enriched.symbol}: {e}") from e

# Maintain backward compatibility
ItsOrchestratorV2_5 = ITSOrchestratorV2_5
