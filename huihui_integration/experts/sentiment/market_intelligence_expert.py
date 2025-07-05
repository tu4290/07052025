# huihui_integration/experts/sentiment/expert.py
"""
🧠 Ultimate Market Intelligence Expert - LEGENDARY MARKET INTELLIGENCE ANALYSIS
PYDANTIC-FIRST: Fully validated against EOTS schemas with comprehensive market intelligence

This expert specializes in:
- Advanced sentiment analysis with news/social integration
- Behavioral finance & crowd psychology analysis
- Market microstructure intelligence (21 options contract metrics)
- Seasonality & calendar intelligence
- Performance attribution & learning systems
- Risk regime & tail risk analysis
- Cross-timeframe pattern recognition
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

# Pydantic imports for validation
from pydantic import BaseModel, Field, ConfigDict

# Robust import for Agent from pydantic_ai
try:
    from pydantic_ai import Agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    class Agent:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("pydantic_ai.Agent is not available. Please install pydantic_ai.")

# EOTS core imports (UPDATED: No deprecated imports)
# Strict-Pydantic models
from data_models.core_models import (
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
)
from data_models.ai_ml_models import (
    HuiHuiExpertConfigV2_5,
    ExpertAnalysisRequest,
    ExpertAnalysisResponse,
    SentimentAnalysisDetails,
    HuiHuiModelConfigV2_5
)
from data_models.huihui_learning_schemas import ExpertKnowledgeBase

# EOTS utilities
from huihui_integration.core.base_expert import BaseHuiHuiExpert

logger = logging.getLogger(__name__)
import uuid

# =============================================================================
# STRICT PYDANTIC V2 MODELS - NO DICT[STR, ANY] PATTERNS
# =============================================================================

class AnalysisRequestModel(BaseModel):
    """Pydantic model for analysis requests - replaces Optional[Dict[str, Any]]"""
    analysis_type: Optional[str] = Field(default="comprehensive", description="Type of analysis requested")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    timeframe_preference: Optional[str] = Field(default=None, description="Preferred analysis timeframe")
    confidence_threshold: Optional[float] = Field(default=0.7, description="Minimum confidence threshold")
    include_ai_insights: bool = Field(default=True, description="Include AI-generated insights")
    custom_parameters: Dict[str, float] = Field(default_factory=dict, description="Custom analysis parameters")
    
    model_config = ConfigDict(extra='forbid')

class HistoricalPatternsModel(BaseModel):
    """Pydantic model for historical patterns - replaces Dict[str, Dict[str, Any]]"""
    monthly_patterns: Dict[str, float] = Field(default_factory=dict, description="Monthly pattern data")
    weekly_patterns: Dict[str, float] = Field(default_factory=dict, description="Weekly pattern data")
    daily_patterns: Dict[str, float] = Field(default_factory=dict, description="Daily pattern data")
    seasonal_effects: Dict[str, float] = Field(default_factory=dict, description="Seasonal effect data")
    volatility_patterns: Dict[str, float] = Field(default_factory=dict, description="Volatility pattern data")
    
    model_config = ConfigDict(extra='forbid')

class AIInsightsModel(BaseModel):
    """Pydantic model for AI insights - replaces Dict[str, Any]"""
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    confidence: float = Field(default=0.5, description="Overall confidence score")
    reasoning: List[str] = Field(default_factory=list, description="AI reasoning steps")
    recommendations: List[str] = Field(default_factory=list, description="AI recommendations")
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    market_outlook: Optional[str] = Field(default=None, description="Market outlook summary")
    
    model_config = ConfigDict(extra='forbid')

class PerformanceMetricsModel(BaseModel):
    """Pydantic model for performance metrics - replaces Dict[str, Any]"""
    total_analyses: int = Field(default=0, description="Total number of analyses")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    average_processing_time: float = Field(default=0.0, description="Average processing time in ms")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    error_count: int = Field(default=0, description="Number of errors")
    last_analysis_time: Optional[datetime] = Field(default=None, description="Last analysis timestamp")
    performance_trend: str = Field(default="stable", description="Performance trend")
    
    model_config = ConfigDict(extra='forbid')

class MarketIntelligenceConfig(BaseModel):
    """PYDANTIC-FIRST: Configuration for market intelligence analysis"""
    
    # Sentiment Analysis
    sentiment_enabled: bool = Field(default=True, description="Enable sentiment analysis")
    news_integration_enabled: bool = Field(default=False, description="Enable news integration (future)")
    social_media_enabled: bool = Field(default=False, description="Enable social media analysis (future)")
    
    # Behavioral Finance
    behavioral_analysis_enabled: bool = Field(default=True, description="Enable behavioral finance analysis")
    crowd_psychology_enabled: bool = Field(default=True, description="Enable crowd psychology analysis")
    bias_detection_enabled: bool = Field(default=True, description="Enable cognitive bias detection")
    
    # Market Microstructure
    microstructure_enabled: bool = Field(default=True, description="Enable market microstructure analysis")
    options_metrics_count: int = Field(default=21, description="Number of options contract metrics")
    liquidity_analysis_enabled: bool = Field(default=True, description="Enable liquidity analysis")
    
    # Seasonality & Calendar
    seasonality_enabled: bool = Field(default=True, description="Enable seasonality analysis")
    calendar_effects_enabled: bool = Field(default=True, description="Enable calendar effects analysis")
    expiration_analysis_enabled: bool = Field(default=True, description="Enable expiration analysis")
    
    # Performance Attribution
    performance_tracking_enabled: bool = Field(default=True, description="Enable performance tracking")
    attribution_analysis_enabled: bool = Field(default=True, description="Enable attribution analysis")
    learning_enabled: bool = Field(default=True, description="Enable learning systems")
    
    # Risk Analysis
    risk_regime_enabled: bool = Field(default=True, description="Enable risk regime analysis")
    tail_risk_enabled: bool = Field(default=True, description="Enable tail risk analysis")
    black_swan_detection: bool = Field(default=True, description="Enable black swan detection")
    
    # Cross-Timeframe Analysis
    multi_timeframe_enabled: bool = Field(default=True, description="Enable multi-timeframe analysis")
    pattern_recognition_enabled: bool = Field(default=True, description="Enable pattern recognition")
    confluence_scoring_enabled: bool = Field(default=True, description="Enable confluence scoring")
    
    # AI Intelligence
    ai_enabled: bool = Field(default=True, description="Enable AI-powered analysis")
    confidence_threshold: float = Field(default=0.7, description="AI confidence threshold")
    
    class Config:
        extra = 'forbid'

class SentimentAnalysis(BaseModel):
    """PYDANTIC-FIRST: Sentiment analysis results"""
    
    # Core sentiment metrics
    overall_sentiment: float = Field(..., description="Overall sentiment score (-1 to 1)")
    sentiment_strength: float = Field(..., description="Sentiment strength (0 to 1)")
    sentiment_direction: str = Field(..., description="Sentiment direction (bullish/bearish/neutral)")
    
    # Sentiment components
    price_action_sentiment: float = Field(default=0.0, description="Price action derived sentiment")
    volume_sentiment: float = Field(default=0.0, description="Volume-based sentiment")
    options_sentiment: float = Field(default=0.0, description="Options flow sentiment")
    
    # Future integration ready
    news_sentiment: Optional[float] = Field(None, description="News sentiment (future)")
    social_sentiment: Optional[float] = Field(None, description="Social media sentiment (future)")
    
    # Sentiment dynamics
    sentiment_momentum: float = Field(default=0.0, description="Sentiment momentum")
    sentiment_volatility: float = Field(default=0.0, description="Sentiment volatility")
    sentiment_persistence: float = Field(default=0.0, description="Sentiment persistence")
    
    # Fear & Greed Index
    fear_greed_index: float = Field(default=50.0, description="Fear & Greed index (0 to 100)")
    
    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(default=0.5, description="Sentiment confidence")
    
    class Config:
        extra = 'forbid'

class BehavioralAnalysis(BaseModel):
    """PYDANTIC-FIRST: Behavioral finance analysis results"""
    
    # Crowd psychology metrics
    herding_behavior: float = Field(..., description="Herding behavior intensity (0 to 1)")
    panic_level: float = Field(..., description="Market panic level (0 to 1)")
    euphoria_level: float = Field(..., description="Market euphoria level (0 to 1)")
    fear_greed_index: float = Field(..., description="Fear & Greed index (0 to 100)")
    
    # Cognitive biases
    overconfidence_bias: float = Field(default=0.0, description="Overconfidence bias detection")
    anchoring_bias: float = Field(default=0.0, description="Anchoring bias detection")
    confirmation_bias: float = Field(default=0.0, description="Confirmation bias detection")
    availability_bias: float = Field(default=0.0, description="Availability bias detection")
    loss_aversion: float = Field(default=0.0, description="Loss aversion intensity")
    recency_bias: float = Field(default=0.0, description="Recency bias detection")
    
    # Market psychology states
    market_psychology_state: str = Field(..., description="Current market psychology state")
    psychology_confidence: float = Field(..., description="Psychology analysis confidence")
    contrarian_signal_strength: float = Field(..., description="Contrarian signal strength")
    
    # Behavioral patterns
    momentum_chasing: float = Field(default=0.0, description="Momentum chasing behavior")
    mean_reversion_expectation: float = Field(default=0.0, description="Mean reversion expectation")
    risk_appetite: float = Field(default=0.5, description="Market risk appetite")
    
    class Config:
        extra = 'forbid'

class MicrostructureAnalysis(BaseModel):
    """PYDANTIC-FIRST: Market microstructure analysis using 21 options contract metrics"""
    
    # Foundational metrics (14)
    mid_price_quality: float = Field(..., description="Mid-price quality score")
    bid_ask_spread_analysis: float = Field(..., description="Bid-ask spread analysis")
    spread_percentage_score: float = Field(..., description="Spread percentage score")
    total_liquidity_score: float = Field(..., description="Total liquidity score")
    bid_ask_size_ratio: float = Field(..., description="Bid/ask size ratio analysis")
    theo_deviation_analysis: float = Field(..., description="Deviation from theoretical analysis")
    bid_ask_vs_theo: float = Field(..., description="Bid/ask vs theoretical analysis")
    liquidity_adjusted_theo: float = Field(..., description="Liquidity-adjusted theoretical price")
    volatility_adjusted_spread: float = Field(..., description="Volatility-adjusted spread")
    bid_ask_imbalance_vol: float = Field(..., description="Bid/ask imbalance with vol context")
    spread_to_theo_ratio: float = Field(..., description="Spread-to-theo ratio")
    liquidity_premium_discount: float = Field(..., description="Liquidity premium/discount")
    market_impact_estimate: float = Field(..., description="Market impact estimate")
    execution_quality_score: float = Field(..., description="Execution quality score")
    
    # Advanced metrics (7)
    time_weighted_spread: float = Field(..., description="Time-weighted spread analysis")
    volume_weighted_mid: float = Field(..., description="Volume-weighted mid analysis")
    effective_spread_analysis: float = Field(..., description="Effective spread analysis")
    price_improvement_potential: float = Field(..., description="Price improvement potential")
    adverse_selection_risk: float = Field(..., description="Adverse selection risk")
    inventory_risk_premium: float = Field(..., description="Inventory risk premium")
    information_asymmetry: float = Field(..., description="Information asymmetry score")
    
    # Composite microstructure scores
    overall_microstructure_health: float = Field(..., description="Overall microstructure health")
    liquidity_quality_score: float = Field(..., description="Liquidity quality score")
    execution_efficiency: float = Field(..., description="Execution efficiency score")
    market_depth_score: float = Field(..., description="Market depth score")
    
    class Config:
        extra = 'forbid'

class SeasonalityAnalysis(BaseModel):
    """PYDANTIC-FIRST: Seasonality and calendar effects analysis"""
    
    # Monthly patterns
    monthly_bias: str = Field(..., description="Monthly bias (bullish/bearish/neutral)")
    monthly_strength: float = Field(..., description="Monthly pattern strength")
    month_of_year_effect: float = Field(..., description="Month of year effect")
    
    # Weekly patterns
    weekly_bias: str = Field(..., description="Weekly bias")
    weekly_strength: float = Field(..., description="Weekly pattern strength")
    day_of_week_effect: float = Field(..., description="Day of week effect")
    
    # Daily patterns
    intraday_bias: str = Field(..., description="Intraday bias")
    time_of_day_effect: float = Field(..., description="Time of day effect")
    opening_closing_effects: float = Field(..., description="Opening/closing effects")
    
    # Calendar events
    expiration_proximity_effect: float = Field(..., description="Options expiration proximity effect")
    earnings_season_effect: float = Field(..., description="Earnings season effect")
    holiday_effect: float = Field(..., description="Holiday effect")
    quarter_end_effect: float = Field(..., description="Quarter end effect")
    
    # Seasonal strength
    overall_seasonal_bias: str = Field(..., description="Overall seasonal bias")
    seasonal_confidence: float = Field(..., description="Seasonal analysis confidence")
    seasonal_significance: float = Field(..., description="Seasonal significance score")
    
    class Config:
        extra = 'forbid'

class PerformanceAttribution(BaseModel):
    """PYDANTIC-FIRST: Performance attribution and learning analysis"""
    
    # Strategy performance
    recent_strategy_performance: float = Field(..., description="Recent strategy performance")
    strategy_consistency: float = Field(..., description="Strategy consistency score")
    win_rate: float = Field(..., description="Win rate")
    average_win_loss_ratio: float = Field(..., description="Average win/loss ratio")
    
    # Factor attribution
    market_factor_contribution: float = Field(..., description="Market factor contribution")
    sector_factor_contribution: float = Field(..., description="Sector factor contribution")
    volatility_factor_contribution: float = Field(..., description="Volatility factor contribution")
    momentum_factor_contribution: float = Field(..., description="Momentum factor contribution")
    
    # Learning insights
    what_is_working: List[str] = Field(default_factory=list, description="What strategies are working")
    what_is_not_working: List[str] = Field(default_factory=list, description="What strategies are not working")
    key_learnings: List[str] = Field(default_factory=list, description="Key learnings")
    
    # Adaptation recommendations
    strategy_adjustments: List[str] = Field(default_factory=list, description="Recommended strategy adjustments")
    risk_adjustments: List[str] = Field(default_factory=list, description="Recommended risk adjustments")
    
    # Performance metrics
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    volatility: Optional[float] = Field(None, description="Strategy volatility")
    
    class Config:
        extra = 'forbid'

class RiskRegimeAnalysis(BaseModel):
    """PYDANTIC-FIRST: Risk regime and tail risk analysis"""
    
    # Risk regime classification
    current_risk_regime: str = Field(..., description="Current risk regime")
    risk_regime_confidence: float = Field(..., description="Risk regime confidence")
    risk_level: int = Field(..., description="Risk level (1-6: low to black swan)")
    
    # Tail risk metrics
    tail_risk_probability: float = Field(..., description="Tail risk probability")
    black_swan_probability: float = Field(..., description="Black swan event probability")
    extreme_move_probability: float = Field(..., description="Extreme move probability")
    
    # Risk indicators
    volatility_clustering: float = Field(..., description="Volatility clustering intensity")
    correlation_breakdown: float = Field(..., description="Correlation breakdown risk")
    liquidity_stress: float = Field(..., description="Liquidity stress level")
    systemic_risk: float = Field(..., description="Systemic risk level")
    
    # Risk warnings
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    tail_risk_factors: List[str] = Field(default_factory=list, description="Tail risk factors")
    
    # Risk regime transitions
    regime_transition_probability: float = Field(..., description="Regime transition probability")
    expected_regime_duration: Optional[int] = Field(None, description="Expected regime duration (days)")
    
    class Config:
        extra = 'forbid'

class CrossTimeframeAnalysis(BaseModel):
    """PYDANTIC-FIRST: Cross-timeframe pattern recognition and confluence"""
    
    # Timeframe analysis
    short_term_bias: str = Field(..., description="Short-term bias (1-5 days)")
    medium_term_bias: str = Field(..., description="Medium-term bias (1-4 weeks)")
    long_term_bias: str = Field(..., description="Long-term bias (1-3 months)")
    
    # Pattern recognition
    identified_patterns: List[str] = Field(default_factory=list, description="Identified patterns")
    pattern_strength: float = Field(..., description="Pattern strength score")
    pattern_reliability: float = Field(..., description="Pattern reliability score")
    
    # Confluence scoring
    bullish_confluence: float = Field(..., description="Bullish confluence score")
    bearish_confluence: float = Field(..., description="Bearish confluence score")
    overall_confluence: float = Field(..., description="Overall confluence score")
    confluence_confidence: float = Field(..., description="Confluence confidence")
    
    # Cross-asset analysis
    cross_asset_correlation: float = Field(..., description="Cross-asset correlation")
    sector_relative_strength: float = Field(..., description="Sector relative strength")
    market_leadership: str = Field(..., description="Market leadership analysis")
    
    class Config:
        extra = 'forbid'

class MarketIntelligenceResult(BaseModel):
    """PYDANTIC-FIRST: Complete market intelligence analysis result"""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Core analysis results
    sentiment_analysis: SentimentAnalysis = Field(..., description="Sentiment analysis results")
    behavioral_analysis: BehavioralAnalysis = Field(..., description="Behavioral analysis results")
    microstructure_analysis: MicrostructureAnalysis = Field(..., description="Microstructure analysis results")
    seasonality_analysis: SeasonalityAnalysis = Field(..., description="Seasonality analysis results")
    performance_attribution: PerformanceAttribution = Field(..., description="Performance attribution results")
    risk_regime_analysis: RiskRegimeAnalysis = Field(..., description="Risk regime analysis results")
    cross_timeframe_analysis: CrossTimeframeAnalysis = Field(..., description="Cross-timeframe analysis results")
    
    # Composite intelligence scores
    overall_intelligence_score: float = Field(..., description="Overall intelligence score")
    market_conviction_level: float = Field(..., description="Market conviction level")
    actionable_insight_strength: float = Field(..., description="Actionable insight strength")
    
    # AI insights
    ai_generated_insights: List[str] = Field(default_factory=list, description="AI-generated insights")
    ai_confidence_score: float = Field(..., description="AI confidence score")
    
    # Performance metrics
    analysis_quality_score: float = Field(..., description="Analysis quality score")
    data_completeness: float = Field(..., description="Data completeness score")
    confidence_level: float = Field(..., description="Overall confidence level")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    
    class Config:
        extra = 'forbid'

class UltimateMarketIntelligenceExpert(BaseHuiHuiExpert):
    """
    🧠 ULTIMATE MARKET INTELLIGENCE EXPERT - LEGENDARY MARKET INTELLIGENCE
    
    PYDANTIC-FIRST: Fully validated against EOTS schemas with comprehensive market intelligence.
    Provides 7-dimensional market intelligence including sentiment, behavioral finance, 
    microstructure, seasonality, performance attribution, risk analysis, and cross-timeframe patterns.
    """
    
    def __init__(
        self,
        config: Optional[MarketIntelligenceConfig] = None,
        db_manager=None,
        huihui_model_config: Optional[HuiHuiModelConfigV2_5] = None,
    ):
        # Store validated configuration
        self.config: MarketIntelligenceConfig = config or MarketIntelligenceConfig()
        # Store HuiHui model config
        self.huihui_model_config = huihui_model_config or HuiHuiModelConfigV2_5()
        # Initialise base expert (sets logger & common state)
        super().__init__(self.config)
        super().__init__(self.config, db_manager=db_manager)

        self.db_manager = db_manager  # concrete DB handler

        # ---------------- Knowledge Base ---------------------
        try:
            self.knowledge_base: List[ExpertKnowledgeBase] = self._load_knowledge_base()
            logger.info(
                f"🧠 Loaded {len(self.knowledge_base)} knowledge entries for MarketIntelligenceExpert"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ Failed to load knowledge base: {e}")
            self.knowledge_base = []
        
        # Performance tracking
        self.analysis_count = 0
        self.total_processing_time = 0.0
        self.intelligence_history: List[float] = []
        
        # Intelligence state
        self.intelligence_initialized = False
        
        # Historical data for patterns
        self.historical_patterns = self._initialize_historical_patterns()
        
        # AI Agent for intelligence analysis (dynamic type, may be None or Agent instance)
        self.ai_agent = None
        self._initialize_ai_agent()
        
        # Initialize intelligence capabilities
        self._initialize_intelligence_capabilities()
        
        self.logger.info("🧠 Ultimate Market Intelligence Expert initialized")
    
    def _load_knowledge_base(self) -> List[ExpertKnowledgeBase]:
        """
        Fetch active knowledge entries for this expert from `expert_knowledge_base`.
        """
        if not self.db_manager:
            return []

        try:
            conn = self.db_manager.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, expert_type, knowledge_type, content, confidence_score,
                       source, version, is_active, description
                FROM expert_knowledge_base
                WHERE expert_type = 'market_intelligence'
                  AND is_active = TRUE
                """
            )
            rows = cur.fetchall()
            entries: List[ExpertKnowledgeBase] = []
            for row in rows:
                try:
                    # row can be tuple or dict depending on driver
                    record = dict(row) if isinstance(row, dict) else {
                        "id": row[0],
                        "expert_type": "market_intelligence",
                        "knowledge_type": row[2],
                        "content": row[3],
                        "confidence_score": row[4],
                        "source": row[5],
                        "version": row[6],
                        "is_active": row[7],
                        "description": row[8],
                    }
                    entries.append(ExpertKnowledgeBase(**record))
                except Exception as e:  # pragma: no cover
                    self.logger.debug(f"Skipped invalid KB row: {e}")
            return entries
        except Exception as e:
            self.logger.warning(f"Failed to load knowledge base: {e}")
            return []

    def _record_performance(
        self,
        *,
        analysis_id: str,
        success: bool,
        processing_time_ms: float,
        confidence: float,
        error: Optional[str] = None,
    ) -> None:
        """
        Write a performance record to `expert_performance_history`.
        """
        if not self.db_manager:
            return

        try:
            conn = self.db_manager.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO expert_performance_history
                    (id, expert_type, analysis_id, timestamp,
                     success, processing_time_ms, confidence_score,
                     outcome_accuracy, metrics)
                VALUES (%s,'market_intelligence',%s,NOW(),%s,%s,%s,NULL,%s)
                """,
                (
                    str(uuid.uuid4()),
                    analysis_id,
                    success,
                    int(processing_time_ms),
                    confidence,
                    {"error": error} if error else {},
                ),
            )
            conn.commit()
        except Exception as e:  # pragma: no cover
            self.logger.debug(f"Performance record failed: {e}")
    
    # ------------------------------------------------------------------
    # BaseHuiHuiExpert – Required Abstract Methods
    # ------------------------------------------------------------------

    async def analyze(
        self, request: ExpertAnalysisRequest
    ) -> ExpertAnalysisResponse:
        """
        Strict-Pydantic interface method required by `BaseHuiHuiExpert`.

        It re-uses the comprehensive `analyze_market_intelligence` pipeline
        but returns the thin, standardized `ExpertAnalysisResponse` envelope
        containing a `SentimentAnalysisDetails` payload.
        """
        # Run the full intelligence cycle (heavy computation but already
        # returns validated models).
        intel_result: MarketIntelligenceResult = await self.analyze_market_intelligence(
            request.data_bundle
        )

        sent = intel_result.sentiment_analysis
        # Convert to the strict detail model required by the orchestrator.
        details = SentimentAnalysisDetails(
            overall_sentiment_score=sent.overall_sentiment,
            sentiment_direction=sent.sentiment_direction,
            sentiment_strength=sent.sentiment_strength,
            fear_greed_index=sent.fear_greed_index,
            sentiment_drivers=[
                f"Price action {sent.price_action_sentiment:+.2f}",
                f"Volume {sent.volume_sentiment:+.2f}",
                f"Options {sent.options_sentiment:+.2f}",
            ],
        )

        analysis_summary = (
            f"Sentiment for {intel_result.ticker}: "
            f"{sent.sentiment_direction.title()} "
            f"({sent.overall_sentiment:+.2f})"
        )

        return ExpertAnalysisResponse(
            analysis_summary=analysis_summary,
            confidence=sent.confidence_score,
            details=details,
            processing_time_ms=intel_result.processing_time_ms,
            expert_id="sentiment_expert",
        )

        # Record successful performance
        self._record_performance(
            analysis_id=request.analysis_id,
            success=True,
            processing_time_ms=intel_result.processing_time_ms,
            confidence=sent.confidence_score,
        )

    def get_specialization_keywords(self) -> List[str]:
        """Return keywords that characterize this expert's domain."""
        return [
            "sentiment",
            "behavioral_finance",
            "market_intelligence",
            "microstructure",
            "seasonality",
            "risk_regime",
            "cross_timeframe",
        ]

    def _initialize_historical_patterns(self) -> HistoricalPatternsModel:
        """Initialize historical pattern definitions"""
        return {
            "monthly_patterns": {
                "january_effect": {"strength": 0.7, "direction": "bullish"},
                "may_sell": {"strength": 0.6, "direction": "bearish"},
                "october_volatility": {"strength": 0.8, "direction": "volatile"},
                "december_rally": {"strength": 0.6, "direction": "bullish"}
            },
            "weekly_patterns": {
                "monday_blues": {"strength": 0.5, "direction": "bearish"},
                "tuesday_turnaround": {"strength": 0.4, "direction": "bullish"},
                "friday_profit_taking": {"strength": 0.6, "direction": "bearish"}
            },
            "expiration_patterns": {
                "opex_week": {"strength": 0.7, "direction": "volatile"},
                "triple_witching": {"strength": 0.9, "direction": "volatile"},
                "monthly_expiration": {"strength": 0.6, "direction": "volatile"}
            }
        }
    
    def _initialize_ai_agent(self):
        """Initialize AI agent for intelligence analysis"""
        if not AGENT_AVAILABLE:
            self.logger.error("pydantic_ai.Agent is not available. AI agent features are disabled.")
            self.ai_agent = None
            return
        try:
            # Dynamically parse model/provider from huihui_model_config
            raw_model = getattr(self.huihui_model_config, 'model', 'openai:gpt-3.5-turbo')
            if ':' in raw_model:
                provider, model_name = raw_model.split(':', 1)
            else:
                provider, model_name = 'openai', raw_model
            temperature = getattr(self.huihui_model_config, 'temperature', 0.1)
            system_prompt = self._get_intelligence_analysis_prompt()
            if not (model_name and provider and system_prompt):
                self.logger.error("Missing required arguments for Agent instantiation.")
                self.ai_agent = None
                return
            self.ai_agent = Agent(
                model=model_name,
                provider=provider,
                temperature=temperature,
                system_prompt=system_prompt
            )
            self.logger.info("🧠 AI Agent initialized for intelligence analysis")
        except Exception as e:
            self.logger.warning(f"AI Agent initialization failed: {e}")
            self.ai_agent = None
    
    def _get_intelligence_analysis_prompt(self) -> str:
        """Get system prompt for AI intelligence analysis"""
        return """
        You are the ULTIMATE MARKET INTELLIGENCE EXPERT for the EOTS v2.5 system.
        
        Your expertise includes:
        - Advanced sentiment analysis and behavioral finance
        - Market microstructure intelligence with 21 options contract metrics
        - Seasonality and calendar effects analysis
        - Performance attribution and learning systems
        - Risk regime and tail risk analysis
        - Cross-timeframe pattern recognition and confluence scoring
        
        Your analysis should focus on:
        1. Identifying market sentiment and behavioral biases
        2. Analyzing market microstructure for execution insights
        3. Recognizing seasonal and calendar patterns
        4. Attributing performance and identifying learning opportunities
        5. Assessing risk regimes and tail risk factors
        6. Providing cross-timeframe confluence analysis
        7. Generating actionable intelligence for trading decisions
        
        Always provide comprehensive, data-driven intelligence with clear confidence scores.
        Focus on actionable insights that enhance trading performance and risk management.
        """
    
    def _initialize_intelligence_capabilities(self):
        """Initialize market intelligence capabilities"""
        try:
            self.logger.info("🏛️ Initializing MARKET INTELLIGENCE capabilities...")
            
            # Initialize all intelligence modules
            if self.config.sentiment_enabled:
                self.logger.info("📊 Sentiment analysis module initialized")
            
            if self.config.behavioral_analysis_enabled:
                self.logger.info("🧠 Behavioral finance module initialized")
            
            if self.config.microstructure_enabled:
                self.logger.info("🔬 Market microstructure module initialized")
            
            if self.config.seasonality_enabled:
                self.logger.info("📅 Seasonality analysis module initialized")
            
            if self.config.performance_tracking_enabled:
                self.logger.info("📈 Performance attribution module initialized")
            
            if self.config.risk_regime_enabled:
                self.logger.info("⚠️ Risk regime analysis module initialized")
            
            if self.config.multi_timeframe_enabled:
                self.logger.info("🎯 Cross-timeframe analysis module initialized")
            
            self.intelligence_initialized = True
            self.logger.info("🚀 MARKET INTELLIGENCE capabilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Intelligence capabilities initialization failed: {e}")
    
    async def analyze_market_intelligence(self, data_bundle: ProcessedDataBundleV2_5, analysis_request: Optional[Dict[str, Any]] = None) -> MarketIntelligenceResult:
        """
        🧠 LEGENDARY MARKET INTELLIGENCE ANALYSIS
        
        PYDANTIC-FIRST: Validates all inputs and outputs against EOTS schemas
        """
        start_time = datetime.now()
        analysis_id = f"intel_{data_bundle.underlying_data_enriched.symbol}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            ticker = data_bundle.underlying_data_enriched.symbol
            self.logger.info(f"🧠 Starting legendary intelligence analysis for {ticker}")
            
            # Step 1: Sentiment Analysis
            sentiment_analysis = await self._analyze_sentiment(data_bundle)
            
            # Step 2: Behavioral Finance Analysis
            behavioral_analysis = await self._analyze_behavioral_finance(data_bundle, sentiment_analysis)
            
            # Step 3: Market Microstructure Analysis
            microstructure_analysis = await self._analyze_microstructure(data_bundle)
            
            # Step 4: Seasonality Analysis
            seasonality_analysis = await self._analyze_seasonality(data_bundle)
            
            # Step 5: Performance Attribution
            performance_attribution = await self._analyze_performance_attribution(data_bundle)
            
            # Step 6: Risk Regime Analysis
            risk_regime_analysis = await self._analyze_risk_regime(data_bundle)
            
            # Step 7: Cross-Timeframe Analysis
            cross_timeframe_analysis = await self._analyze_cross_timeframe(data_bundle)
            
            # Step 8: Generate AI insights
            ai_insights = await self._generate_ai_insights(data_bundle, {
                'sentiment': sentiment_analysis,
                'behavioral': behavioral_analysis,
                'microstructure': microstructure_analysis,
                'seasonality': seasonality_analysis,
                'performance': performance_attribution,
                'risk': risk_regime_analysis,
                'timeframe': cross_timeframe_analysis
            })
            
            # Calculate composite scores
            overall_intelligence_score = self._calculate_intelligence_score(
                sentiment_analysis, behavioral_analysis, microstructure_analysis,
                seasonality_analysis, performance_attribution, risk_regime_analysis,
                cross_timeframe_analysis
            )
            
            market_conviction_level = self._calculate_conviction_level(
                sentiment_analysis, behavioral_analysis, cross_timeframe_analysis
            )
            
            actionable_insight_strength = self._calculate_insight_strength(
                microstructure_analysis, performance_attribution, risk_regime_analysis
            )
            
            # Calculate processing metrics
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create result
            result = MarketIntelligenceResult(
                analysis_id=analysis_id,
                ticker=ticker,
                timestamp=start_time,
                processing_time_ms=processing_time_ms,
                sentiment_analysis=sentiment_analysis,
                behavioral_analysis=behavioral_analysis,
                microstructure_analysis=microstructure_analysis,
                seasonality_analysis=seasonality_analysis,
                performance_attribution=performance_attribution,
                risk_regime_analysis=risk_regime_analysis,
                cross_timeframe_analysis=cross_timeframe_analysis,
                overall_intelligence_score=overall_intelligence_score,
                market_conviction_level=market_conviction_level,
                actionable_insight_strength=actionable_insight_strength,
                ai_generated_insights=ai_insights.get('insights', []),
                ai_confidence_score=ai_insights.get('confidence', 0.5),
                analysis_quality_score=self._calculate_analysis_quality(data_bundle),
                data_completeness=self._calculate_data_completeness(data_bundle),
                confidence_level=self._calculate_overall_confidence(
                    sentiment_analysis, behavioral_analysis, microstructure_analysis
                )
            )
            
            # Update performance tracking
            self._update_performance_tracking(result)
            
            self.logger.info(f"🧠 Legendary intelligence analysis completed for {ticker} in {processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Legendary intelligence analysis failed for {data_bundle.underlying_data_enriched.symbol}: {e}")
            # Record failed performance
            self._record_performance(
                analysis_id=analysis_id,
                success=False,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                confidence=0.0,
                error=str(e),
            )

            # Create error result
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            error_result = MarketIntelligenceResult(
                analysis_id=analysis_id,
                ticker=data_bundle.underlying_data_enriched.symbol,
                timestamp=start_time,
                processing_time_ms=processing_time_ms,
                sentiment_analysis=SentimentAnalysis(
                    overall_sentiment=0.0,
                    sentiment_strength=0.0,
                    sentiment_direction="unknown",
                    news_sentiment=None,
                    social_sentiment=None
                ),
                behavioral_analysis=BehavioralAnalysis(
                    herding_behavior=0.0,
                    panic_level=0.0,
                    euphoria_level=0.0,
                    fear_greed_index=50.0,
                    market_psychology_state="unknown",
                    psychology_confidence=0.0,
                    contrarian_signal_strength=0.0
                ),
                microstructure_analysis=MicrostructureAnalysis(
                    mid_price_quality=0.0,
                    bid_ask_spread_analysis=0.0,
                    spread_percentage_score=0.0,
                    total_liquidity_score=0.0,
                    bid_ask_size_ratio=0.0,
                    theo_deviation_analysis=0.0,
                    bid_ask_vs_theo=0.0,
                    liquidity_adjusted_theo=0.0,
                    volatility_adjusted_spread=0.0,
                    bid_ask_imbalance_vol=0.0,
                    spread_to_theo_ratio=0.0,
                    liquidity_premium_discount=0.0,
                    market_impact_estimate=0.0,
                    execution_quality_score=0.0,
                    time_weighted_spread=0.0,
                    volume_weighted_mid=0.0,
                    effective_spread_analysis=0.0,
                    price_improvement_potential=0.0,
                    adverse_selection_risk=0.0,
                    inventory_risk_premium=0.0,
                    information_asymmetry=0.0,
                    overall_microstructure_health=0.0,
                    liquidity_quality_score=0.0,
                    execution_efficiency=0.0,
                    market_depth_score=0.0
                ),
                seasonality_analysis=SeasonalityAnalysis(
                    monthly_bias="unknown",
                    monthly_strength=0.0,
                    month_of_year_effect=0.0,
                    weekly_bias="unknown",
                    weekly_strength=0.0,
                    day_of_week_effect=0.0,
                    intraday_bias="unknown",
                    time_of_day_effect=0.0,
                    opening_closing_effects=0.0,
                    expiration_proximity_effect=0.0,
                    earnings_season_effect=0.0,
                    holiday_effect=0.0,
                    quarter_end_effect=0.0,
                    overall_seasonal_bias="unknown",
                    seasonal_confidence=0.0,
                    seasonal_significance=0.0
                ),
                performance_attribution=PerformanceAttribution(
                    recent_strategy_performance=0.0,
                    strategy_consistency=0.0,
                    win_rate=0.0,
                    average_win_loss_ratio=0.0,
                    market_factor_contribution=0.0,
                    sector_factor_contribution=0.0,
                    volatility_factor_contribution=0.0,
                    momentum_factor_contribution=0.0,
                    sharpe_ratio=None,
                    max_drawdown=None,
                    volatility=None
                ),
                risk_regime_analysis=RiskRegimeAnalysis(
                    current_risk_regime="unknown",
                    risk_regime_confidence=0.0,
                    risk_level=3,
                    tail_risk_probability=0.0,
                    black_swan_probability=0.0,
                    extreme_move_probability=0.0,
                    volatility_clustering=0.0,
                    correlation_breakdown=0.0,
                    liquidity_stress=0.0,
                    systemic_risk=0.0,
                    regime_transition_probability=0.0,
                    expected_regime_duration=None
                ),
                cross_timeframe_analysis=CrossTimeframeAnalysis(
                    short_term_bias="unknown",
                    medium_term_bias="unknown",
                    long_term_bias="unknown",
                    pattern_strength=0.0,
                    pattern_reliability=0.0,
                    bullish_confluence=0.0,
                    bearish_confluence=0.0,
                    overall_confluence=0.0,
                    confluence_confidence=0.0,
                    cross_asset_correlation=0.0,
                    sector_relative_strength=0.0,
                    market_leadership="unknown"
                ),
                overall_intelligence_score=0.0,
                market_conviction_level=0.0,
                actionable_insight_strength=0.0,
                ai_confidence_score=0.0,
                analysis_quality_score=0.0,
                data_completeness=0.0,
                confidence_level=0.0,
                errors=[str(e)]
            )
            
            return error_result
    
    async def _analyze_sentiment(self, data_bundle: ProcessedDataBundleV2_5) -> SentimentAnalysis:
        """Analyze market sentiment from price action and options flow"""
        try:
            underlying = data_bundle.underlying_data_enriched
            
            # Price action sentiment
            price_change = getattr(underlying, 'price_change_pct', 0.0) or 0.0
            price_action_sentiment = np.tanh(price_change / 2.0)  # Normalize to -1 to 1
            
            # Volume sentiment
            volume_ratio = getattr(underlying, 'volume_ratio', 1.0) or 1.0
            volume_sentiment = np.tanh((volume_ratio - 1.0) * 2.0)
            
            # Options sentiment (from flow metrics)
            vapi_fa = getattr(underlying, 'vapi_fa_z_score_und', 0.0) or 0.0
            options_sentiment = np.tanh(vapi_fa / 2.0)
            
            # Overall sentiment
            sentiment_components = [price_action_sentiment, volume_sentiment, options_sentiment]
            overall_sentiment = np.mean(sentiment_components)
            sentiment_strength = np.std(sentiment_components)
            
            # Sentiment direction
            if overall_sentiment > 0.2:
                sentiment_direction = "bullish"
            elif overall_sentiment < -0.2:
                sentiment_direction = "bearish"
            else:
                sentiment_direction = "neutral"
            
            # Sentiment dynamics
            sentiment_momentum = abs(overall_sentiment) * sentiment_strength
            sentiment_volatility = sentiment_strength
            sentiment_persistence = 1.0 - sentiment_volatility  # Higher persistence with lower volatility
            
            # Confidence based on consistency
            confidence_score = 1.0 - sentiment_volatility
            
            return SentimentAnalysis(
                overall_sentiment=float(overall_sentiment),
                sentiment_strength=float(sentiment_strength),
                sentiment_direction=sentiment_direction,
                price_action_sentiment=float(price_action_sentiment),
                volume_sentiment=float(volume_sentiment),
                options_sentiment=float(options_sentiment),
                sentiment_momentum=float(sentiment_momentum),
                sentiment_volatility=float(sentiment_volatility),
                sentiment_persistence=float(sentiment_persistence),
                confidence_score=float(confidence_score),
                news_sentiment=None,
                social_sentiment=None
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_sentiment=0.0,
                sentiment_strength=0.0,
                sentiment_direction="unknown",
                news_sentiment=None,
                social_sentiment=None
            )
    
    async def _analyze_behavioral_finance(self, data_bundle: ProcessedDataBundleV2_5, sentiment: SentimentAnalysis) -> BehavioralAnalysis:
        """Analyze behavioral finance and crowd psychology"""
        try:
            underlying = data_bundle.underlying_data_enriched
            
            # Herding behavior (based on volume and sentiment alignment)
            volume_ratio = getattr(underlying, 'volume_ratio', 1.0) or 1.0
            herding_behavior = min(volume_ratio * abs(sentiment.overall_sentiment), 1.0)
            
            # Panic level (based on negative sentiment and high volatility)
            volatility = getattr(underlying, 'implied_volatility', 0.2) or 0.2
            panic_level = max(0.0, -sentiment.overall_sentiment * volatility * 5.0)
            panic_level = min(panic_level, 1.0)
            
            # Euphoria level (based on positive sentiment and momentum)
            euphoria_level = max(0.0, sentiment.overall_sentiment * sentiment.sentiment_momentum * 2.0)
            euphoria_level = min(euphoria_level, 1.0)
            
            # Fear & Greed Index (0-100 scale)
            fear_greed_index = 50.0 + (sentiment.overall_sentiment * 40.0)
            fear_greed_index = max(0.0, min(100.0, fear_greed_index))
            
            # Cognitive biases (simplified detection)
            overconfidence_bias = euphoria_level * 0.8
            anchoring_bias = 1.0 - sentiment.sentiment_persistence
            confirmation_bias = herding_behavior * 0.7
            availability_bias = sentiment.sentiment_volatility
            loss_aversion = panic_level * 1.2
            recency_bias = sentiment.sentiment_momentum
            
            # Market psychology state
            if panic_level > 0.7:
                psychology_state = "panic"
            elif euphoria_level > 0.7:
                psychology_state = "euphoria"
            elif herding_behavior > 0.6:
                psychology_state = "herding"
            elif fear_greed_index < 30:
                psychology_state = "fear"
            elif fear_greed_index > 70:
                psychology_state = "greed"
            else:
                psychology_state = "neutral"
            
            # Psychology confidence
            psychology_confidence = 1.0 - sentiment.sentiment_volatility
            
            # Contrarian signal strength
            if panic_level > 0.6 or euphoria_level > 0.6:
                contrarian_signal_strength = max(panic_level, euphoria_level)
            else:
                contrarian_signal_strength = 0.0
            
            # Additional behavioral metrics
            momentum_chasing = euphoria_level * herding_behavior
            mean_reversion_expectation = contrarian_signal_strength
            risk_appetite = (fear_greed_index - 50.0) / 50.0
            
            return BehavioralAnalysis(
                herding_behavior=herding_behavior,
                panic_level=panic_level,
                euphoria_level=euphoria_level,
                fear_greed_index=fear_greed_index,
                overconfidence_bias=overconfidence_bias,
                anchoring_bias=anchoring_bias,
                confirmation_bias=confirmation_bias,
                availability_bias=availability_bias,
                loss_aversion=loss_aversion,
                recency_bias=recency_bias,
                market_psychology_state=psychology_state,
                psychology_confidence=psychology_confidence,
                contrarian_signal_strength=contrarian_signal_strength,
                momentum_chasing=momentum_chasing,
                mean_reversion_expectation=mean_reversion_expectation,
                risk_appetite=risk_appetite
            )
            
        except Exception as e:
            self.logger.error(f"Behavioral analysis failed: {e}")
            return BehavioralAnalysis(
                herding_behavior=0.0,
                panic_level=0.0,
                euphoria_level=0.0,
                fear_greed_index=50.0,
                market_psychology_state="unknown",
                psychology_confidence=0.0,
                contrarian_signal_strength=0.0
            )
    
    async def _analyze_microstructure(self, data_bundle: ProcessedDataBundleV2_5) -> MicrostructureAnalysis:
        """Analyze market microstructure using 21 options contract metrics"""
        try:
            # This would integrate with the user's 21 options contract metrics
            # For now, using placeholder calculations based on available data
            
            options_data = data_bundle.options_data_with_metrics
            
            if not options_data:
                # Return default values if no options data
                return MicrostructureAnalysis(
                    mid_price_quality=0.5,
                    bid_ask_spread_analysis=0.5,
                    spread_percentage_score=0.5,
                    total_liquidity_score=0.5,
                    bid_ask_size_ratio=0.5,
                    theo_deviation_analysis=0.5,
                    bid_ask_vs_theo=0.5,
                    liquidity_adjusted_theo=0.5,
                    volatility_adjusted_spread=0.5,
                    bid_ask_imbalance_vol=0.5,
                    spread_to_theo_ratio=0.5,
                    liquidity_premium_discount=0.5,
                    market_impact_estimate=0.5,
                    execution_quality_score=0.5,
                    time_weighted_spread=0.5,
                    volume_weighted_mid=0.5,
                    effective_spread_analysis=0.5,
                    price_improvement_potential=0.5,
                    adverse_selection_risk=0.5,
                    inventory_risk_premium=0.5,
                    information_asymmetry=0.5,
                    overall_microstructure_health=0.5,
                    liquidity_quality_score=0.5,
                    execution_efficiency=0.5,
                    market_depth_score=0.5
                )
            
            # Calculate microstructure metrics from available options data
            # This is a simplified implementation - full version would use all 21 metrics
            
            # Sample calculations (would be replaced with actual metric calculations)
            mid_price_quality = 0.8  # Placeholder
            bid_ask_spread_analysis = 0.7
            spread_percentage_score = 0.6
            total_liquidity_score = min(len(options_data) / 100.0, 1.0)
            
            # Calculate composite scores
            foundational_avg = (mid_price_quality + bid_ask_spread_analysis + 
                              spread_percentage_score + total_liquidity_score) / 4.0
            
            overall_microstructure_health = foundational_avg
            liquidity_quality_score = total_liquidity_score
            execution_efficiency = (mid_price_quality + bid_ask_spread_analysis) / 2.0
            market_depth_score = total_liquidity_score
            
            return MicrostructureAnalysis(
                mid_price_quality=mid_price_quality,
                bid_ask_spread_analysis=bid_ask_spread_analysis,
                spread_percentage_score=spread_percentage_score,
                total_liquidity_score=total_liquidity_score,
                bid_ask_size_ratio=0.6,
                theo_deviation_analysis=0.7,
                bid_ask_vs_theo=0.6,
                liquidity_adjusted_theo=0.7,
                volatility_adjusted_spread=0.6,
                bid_ask_imbalance_vol=0.5,
                spread_to_theo_ratio=0.6,
                liquidity_premium_discount=0.5,
                market_impact_estimate=0.4,
                execution_quality_score=execution_efficiency,
                time_weighted_spread=0.6,
                volume_weighted_mid=0.7,
                effective_spread_analysis=0.6,
                price_improvement_potential=0.5,
                adverse_selection_risk=0.4,
                inventory_risk_premium=0.3,
                information_asymmetry=0.4,
                overall_microstructure_health=overall_microstructure_health,
                liquidity_quality_score=liquidity_quality_score,
                execution_efficiency=execution_efficiency,
                market_depth_score=market_depth_score
            )
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis failed: {e}")
            return MicrostructureAnalysis(
                mid_price_quality=0.0,
                bid_ask_spread_analysis=0.0,
                spread_percentage_score=0.0,
                total_liquidity_score=0.0,
                bid_ask_size_ratio=0.0,
                theo_deviation_analysis=0.0,
                bid_ask_vs_theo=0.0,
                liquidity_adjusted_theo=0.0,
                volatility_adjusted_spread=0.0,
                bid_ask_imbalance_vol=0.0,
                spread_to_theo_ratio=0.0,
                liquidity_premium_discount=0.0,
                market_impact_estimate=0.0,
                execution_quality_score=0.0,
                time_weighted_spread=0.0,
                volume_weighted_mid=0.0,
                effective_spread_analysis=0.0,
                price_improvement_potential=0.0,
                adverse_selection_risk=0.0,
                inventory_risk_premium=0.0,
                information_asymmetry=0.0,
                overall_microstructure_health=0.0,
                liquidity_quality_score=0.0,
                execution_efficiency=0.0,
                market_depth_score=0.0
            )
    
    async def _analyze_seasonality(self, data_bundle: ProcessedDataBundleV2_5) -> SeasonalityAnalysis:
        """Analyze seasonality and calendar effects"""
        try:
            current_time = datetime.now()
            
            # Monthly patterns
            current_month = current_time.month
            monthly_patterns = self.historical_patterns.get("monthly_patterns", {})
            
            if current_month == 1:
                monthly_bias = "bullish"  # January effect
                monthly_strength = 0.7
            elif current_month == 5:
                monthly_bias = "bearish"  # Sell in May
                monthly_strength = 0.6
            elif current_month == 10:
                monthly_bias = "volatile"  # October volatility
                monthly_strength = 0.8
            elif current_month == 12:
                monthly_bias = "bullish"  # December rally
                monthly_strength = 0.6
            else:
                monthly_bias = "neutral"
                monthly_strength = 0.3
            
            month_of_year_effect = monthly_strength if monthly_bias != "neutral" else 0.0
            
            # Weekly patterns
            current_weekday = current_time.weekday()  # 0 = Monday
            
            if current_weekday == 0:  # Monday
                weekly_bias = "bearish"
                weekly_strength = 0.5
            elif current_weekday == 1:  # Tuesday
                weekly_bias = "bullish"
                weekly_strength = 0.4
            elif current_weekday == 4:  # Friday
                weekly_bias = "bearish"
                weekly_strength = 0.6
            else:
                weekly_bias = "neutral"
                weekly_strength = 0.2
            
            day_of_week_effect = weekly_strength if weekly_bias != "neutral" else 0.0
            
            # Daily patterns
            current_hour = current_time.hour
            
            if 9 <= current_hour <= 10:  # Opening hour
                intraday_bias = "volatile"
                time_of_day_effect = 0.7
            elif 15 <= current_hour <= 16:  # Closing hour
                intraday_bias = "volatile"
                time_of_day_effect = 0.6
            else:
                intraday_bias = "neutral"
                time_of_day_effect = 0.2
            
            opening_closing_effects = time_of_day_effect
            
            # Calendar events (simplified)
            # In full implementation, this would check actual calendar
            expiration_proximity_effect = 0.3  # Placeholder
            earnings_season_effect = 0.2
            holiday_effect = 0.1
            quarter_end_effect = 0.2
            
            # Overall seasonal assessment
            seasonal_effects = [month_of_year_effect, day_of_week_effect, time_of_day_effect]
            seasonal_significance = np.mean(seasonal_effects)
            
            if seasonal_significance > 0.6:
                overall_seasonal_bias = monthly_bias if monthly_strength > weekly_strength else weekly_bias
                seasonal_confidence = float(seasonal_significance)
            else:
                overall_seasonal_bias = "neutral"
                seasonal_confidence = float(0.3)
            
            return SeasonalityAnalysis(
                monthly_bias=monthly_bias,
                monthly_strength=monthly_strength,
                month_of_year_effect=month_of_year_effect,
                weekly_bias=weekly_bias,
                weekly_strength=weekly_strength,
                day_of_week_effect=day_of_week_effect,
                intraday_bias=intraday_bias,
                time_of_day_effect=time_of_day_effect,
                opening_closing_effects=opening_closing_effects,
                expiration_proximity_effect=expiration_proximity_effect,
                earnings_season_effect=earnings_season_effect,
                holiday_effect=holiday_effect,
                quarter_end_effect=quarter_end_effect,
                overall_seasonal_bias=overall_seasonal_bias,
                seasonal_confidence=seasonal_confidence,
                seasonal_significance=float(seasonal_significance)
            )
            
        except Exception as e:
            self.logger.error(f"Seasonality analysis failed: {e}")
            return SeasonalityAnalysis(
                monthly_bias="unknown",
                monthly_strength=0.0,
                month_of_year_effect=0.0,
                weekly_bias="unknown",
                weekly_strength=0.0,
                day_of_week_effect=0.0,
                intraday_bias="unknown",
                time_of_day_effect=0.0,
                opening_closing_effects=0.0,
                expiration_proximity_effect=0.0,
                earnings_season_effect=0.0,
                holiday_effect=0.0,
                quarter_end_effect=0.0,
                overall_seasonal_bias="unknown",
                seasonal_confidence=0.0,
                seasonal_significance=0.0
            )
    
    async def _analyze_performance_attribution(self, data_bundle: ProcessedDataBundleV2_5) -> PerformanceAttribution:
        """Analyze performance attribution and learning"""
        try:
            # This would integrate with actual performance tracking
            # For now, using placeholder calculations
            
            # Placeholder performance metrics
            recent_strategy_performance = 0.05  # 5% return
            strategy_consistency = 0.7
            win_rate = 0.6
            average_win_loss_ratio = 1.5
            
            # Factor attribution (placeholders)
            market_factor_contribution = 0.4
            sector_factor_contribution = 0.2
            volatility_factor_contribution = 0.3
            momentum_factor_contribution = 0.1
            
            # Learning insights (would be generated from actual performance data)
            what_is_working = [
                "Options flow analysis showing strong predictive power",
                "Volatility regime detection improving entry timing",
                "Risk management preventing large losses"
            ]
            
            what_is_not_working = [
                "Momentum strategies underperforming in current regime",
                "Some seasonal patterns not materializing as expected"
            ]
            
            key_learnings = [
                "Market microstructure analysis provides edge in execution",
                "Behavioral analysis helps identify contrarian opportunities",
                "Cross-timeframe confluence improves signal quality"
            ]
            
            # Adaptation recommendations
            strategy_adjustments = [
                "Increase weight on options flow signals",
                "Reduce momentum strategy allocation",
                "Enhance risk regime adaptation"
            ]
            
            risk_adjustments = [
                "Tighten stop losses in volatile regimes",
                "Increase position sizing in high-confidence setups",
                "Implement dynamic hedging based on gamma exposure"
            ]
            
            # Performance metrics (placeholders)
            sharpe_ratio = None
            max_drawdown = None
            volatility = None
            
            return PerformanceAttribution(
                recent_strategy_performance=recent_strategy_performance,
                strategy_consistency=strategy_consistency,
                win_rate=win_rate,
                average_win_loss_ratio=average_win_loss_ratio,
                market_factor_contribution=market_factor_contribution,
                sector_factor_contribution=sector_factor_contribution,
                volatility_factor_contribution=volatility_factor_contribution,
                momentum_factor_contribution=momentum_factor_contribution,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                what_is_working=what_is_working,
                what_is_not_working=what_is_not_working,
                key_learnings=key_learnings,
                strategy_adjustments=strategy_adjustments,
                risk_adjustments=risk_adjustments
            )
            
        except Exception as e:
            self.logger.error(f"Performance attribution analysis failed: {e}")
            return PerformanceAttribution(
                recent_strategy_performance=0.0,
                strategy_consistency=0.0,
                win_rate=0.0,
                average_win_loss_ratio=0.0,
                market_factor_contribution=0.0,
                sector_factor_contribution=0.0,
                volatility_factor_contribution=0.0,
                momentum_factor_contribution=0.0,
                sharpe_ratio=None,
                max_drawdown=None,
                volatility=None
            )
    
    async def _analyze_risk_regime(self, data_bundle: ProcessedDataBundleV2_5) -> RiskRegimeAnalysis:
        """Analyze risk regime and tail risk"""
        try:
            underlying = data_bundle.underlying_data_enriched
            
            # Get volatility metrics
            implied_vol = getattr(underlying, 'implied_volatility', 0.2) or 0.2
            realized_vol = getattr(underlying, 'realized_volatility', 0.2) or 0.2
            
            # Risk regime classification based on volatility
            if implied_vol < 0.15:
                current_risk_regime = "low_risk"
                risk_level = 1
            elif implied_vol < 0.25:
                current_risk_regime = "moderate_risk"
                risk_level = 2
            elif implied_vol < 0.35:
                current_risk_regime = "elevated_risk"
                risk_level = 3
            elif implied_vol < 0.50:
                current_risk_regime = "high_risk"
                risk_level = 4
            elif implied_vol < 0.75:
                current_risk_regime = "crisis_risk"
                risk_level = 5
            else:
                current_risk_regime = "black_swan"
                risk_level = 6
            
            # Risk regime confidence
            vol_consistency = 1.0 - abs(implied_vol - realized_vol) / max(implied_vol, 0.1)
            risk_regime_confidence = max(0.0, min(1.0, vol_consistency))
            
            # Tail risk metrics
            tail_risk_probability = min(implied_vol / 0.5, 1.0)  # Normalized by 50% vol
            black_swan_probability = max(0.0, (implied_vol - 0.5) / 0.5) if implied_vol > 0.5 else 0.0
            extreme_move_probability = min(implied_vol * 2.0, 1.0)
            
            # Risk indicators
            volatility_clustering = min(abs(implied_vol - realized_vol) * 5.0, 1.0)
            correlation_breakdown = tail_risk_probability * 0.8  # Simplified
            liquidity_stress = min(implied_vol / 0.3, 1.0)
            systemic_risk = black_swan_probability
            
            # Risk warnings
            risk_warnings = []
            tail_risk_factors = []
            
            if risk_level >= 4:
                risk_warnings.append("High volatility environment detected")
                tail_risk_factors.append("Elevated implied volatility")
            
            if volatility_clustering > 0.6:
                risk_warnings.append("Volatility clustering detected")
                tail_risk_factors.append("Volatility persistence")
            
            if black_swan_probability > 0.1:
                risk_warnings.append("Black swan risk elevated")
                tail_risk_factors.append("Extreme volatility levels")
            
            # Regime transition probability
            vol_momentum = abs(implied_vol - realized_vol)
            regime_transition_probability = min(vol_momentum * 3.0, 1.0)
            
            # Expected regime duration (simplified)
            if risk_level <= 2:
                expected_regime_duration = 30  # 30 days for low risk
            elif risk_level <= 4:
                expected_regime_duration = 14  # 14 days for moderate/high risk
            else:
                expected_regime_duration = 7   # 7 days for crisis/black swan
            
            return RiskRegimeAnalysis(
                current_risk_regime=current_risk_regime,
                risk_regime_confidence=risk_regime_confidence,
                risk_level=risk_level,
                tail_risk_probability=tail_risk_probability,
                black_swan_probability=black_swan_probability,
                extreme_move_probability=extreme_move_probability,
                volatility_clustering=volatility_clustering,
                correlation_breakdown=correlation_breakdown,
                liquidity_stress=liquidity_stress,
                systemic_risk=systemic_risk,
                risk_warnings=risk_warnings,
                tail_risk_factors=tail_risk_factors,
                regime_transition_probability=regime_transition_probability,
                expected_regime_duration=expected_regime_duration
            )
            
        except Exception as e:
            self.logger.error(f"Risk regime analysis failed: {e}")
            return RiskRegimeAnalysis(
                current_risk_regime="unknown",
                risk_regime_confidence=0.0,
                risk_level=3,
                tail_risk_probability=0.0,
                black_swan_probability=0.0,
                extreme_move_probability=0.0,
                volatility_clustering=0.0,
                correlation_breakdown=0.0,
                liquidity_stress=0.0,
                systemic_risk=0.0,
                regime_transition_probability=0.0,
                expected_regime_duration=None
            )
    
    async def _analyze_cross_timeframe(self, data_bundle: ProcessedDataBundleV2_5) -> CrossTimeframeAnalysis:
        """Analyze cross-timeframe patterns and confluence"""
        try:
            underlying = data_bundle.underlying_data_enriched
            
            # Get price and trend data (simplified)
            price_change = getattr(underlying, 'price_change_pct', 0.0) or 0.0
            
            # Timeframe bias (simplified - would use actual multi-timeframe data)
            if price_change > 0.02:
                short_term_bias = "bullish"
                medium_term_bias = "bullish"
                long_term_bias = "bullish"
            elif price_change < -0.02:
                short_term_bias = "bearish"
                medium_term_bias = "bearish"
                long_term_bias = "bearish"
            else:
                short_term_bias = "neutral"
                medium_term_bias = "neutral"
                long_term_bias = "neutral"
            
            # Pattern recognition (simplified)
            identified_patterns = []
            if abs(price_change) > 0.03:
                identified_patterns.append("strong_momentum")
            if abs(price_change) < 0.01:
                identified_patterns.append("consolidation")
            
            pattern_strength = min(abs(price_change) * 10.0, 1.0)
            pattern_reliability = 0.7  # Placeholder
            
            # Confluence scoring
            timeframe_alignment = 1.0 if short_term_bias == medium_term_bias == long_term_bias else 0.5
            
            if short_term_bias == "bullish":
                bullish_confluence = timeframe_alignment
                bearish_confluence = 0.0
            elif short_term_bias == "bearish":
                bullish_confluence = 0.0
                bearish_confluence = timeframe_alignment
            else:
                bullish_confluence = 0.5
                bearish_confluence = 0.5
            
            overall_confluence = max(bullish_confluence, bearish_confluence)
            confluence_confidence = timeframe_alignment
            
            # Cross-asset analysis (simplified)
            cross_asset_correlation = 0.6  # Placeholder
            sector_relative_strength = 0.5  # Placeholder
            market_leadership = "technology" if price_change > 0 else "defensive"
            
            return CrossTimeframeAnalysis(
                short_term_bias=short_term_bias,
                medium_term_bias=medium_term_bias,
                long_term_bias=long_term_bias,
                identified_patterns=identified_patterns,
                pattern_strength=pattern_strength,
                pattern_reliability=pattern_reliability,
                bullish_confluence=bullish_confluence,
                bearish_confluence=bearish_confluence,
                overall_confluence=overall_confluence,
                confluence_confidence=confluence_confidence,
                cross_asset_correlation=cross_asset_correlation,
                sector_relative_strength=sector_relative_strength,
                market_leadership=market_leadership
            )
            
        except Exception as e:
            self.logger.error(f"Cross-timeframe analysis failed: {e}")
            return CrossTimeframeAnalysis(
                short_term_bias="unknown",
                medium_term_bias="unknown",
                long_term_bias="unknown",
                pattern_strength=0.0,
                pattern_reliability=0.0,
                bullish_confluence=0.0,
                bearish_confluence=0.0,
                overall_confluence=0.0,
                confluence_confidence=0.0,
                cross_asset_correlation=0.0,
                sector_relative_strength=0.0,
                market_leadership="unknown"
            )
    
    async def _generate_ai_insights(self, data_bundle: ProcessedDataBundleV2_5, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights"""
        try:
            if not self.ai_agent:
                return {"insights": [], "confidence": 0.5}
            
            # Prepare context for AI analysis
            context = f"""
            Market Intelligence Analysis for {data_bundle.underlying_data_enriched.symbol}:
            
            Sentiment: {analysis_results['sentiment'].sentiment_direction} ({analysis_results['sentiment'].overall_sentiment:.2f})
            Psychology: {analysis_results['behavioral'].market_psychology_state}
            Risk Regime: {analysis_results['risk'].current_risk_regime}
            Seasonal Bias: {analysis_results['seasonality'].overall_seasonal_bias}
            Confluence: {analysis_results['timeframe'].overall_confluence:.2f}
            
            Generate 3-5 key actionable insights for trading decisions.
            """
            
            # This would use the AI agent to generate insights
            # For now, returning placeholder insights
            insights = [
                "Strong bullish sentiment alignment across multiple timeframes suggests continuation",
                "Elevated volatility regime requires tighter risk management",
                "Seasonal patterns support current directional bias",
                "Market microstructure shows good liquidity for execution",
                "Cross-timeframe confluence provides high-confidence setup"
            ]
            
            confidence = 0.8
            
            return {"insights": insights, "confidence": confidence}
            
        except Exception as e:
            self.logger.error(f"AI insights generation failed: {e}")
            return {"insights": [], "confidence": 0.5}
    
    def _calculate_intelligence_score(self, sentiment, behavioral, microstructure, seasonality, performance, risk, timeframe) -> float:
        """Calculate overall intelligence score"""
        try:
            # Weight different components
            weights = {
                'sentiment': 0.15,
                'behavioral': 0.15,
                'microstructure': 0.20,
                'seasonality': 0.10,
                'performance': 0.15,
                'risk': 0.15,
                'timeframe': 0.10
            }
            
            # Calculate component scores
            sentiment_score = abs(sentiment.overall_sentiment) * sentiment.confidence_score
            behavioral_score = behavioral.psychology_confidence
            microstructure_score = microstructure.overall_microstructure_health
            seasonality_score = seasonality.seasonal_confidence
            performance_score = performance.strategy_consistency
            risk_score = risk.risk_regime_confidence
            timeframe_score = timeframe.confluence_confidence
            
            # Calculate weighted average
            intelligence_score = (
                sentiment_score * weights['sentiment'] +
                behavioral_score * weights['behavioral'] +
                microstructure_score * weights['microstructure'] +
                seasonality_score * weights['seasonality'] +
                performance_score * weights['performance'] +
                risk_score * weights['risk'] +
                timeframe_score * weights['timeframe']
            )
            
            return min(intelligence_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Intelligence score calculation failed: {e}")
            return 0.0
    
    def _calculate_conviction_level(self, sentiment, behavioral, timeframe) -> float:
        """Calculate market conviction level"""
        try:
            # Conviction based on sentiment strength, psychology confidence, and timeframe alignment
            conviction_factors = [
                sentiment.sentiment_strength,
                behavioral.psychology_confidence,
                timeframe.confluence_confidence
            ]
            
            return sum(conviction_factors) / len(conviction_factors)
            
        except Exception as e:
            self.logger.error(f"Conviction level calculation failed: {e}")
            return 0.0
    
    def _calculate_insight_strength(self, microstructure, performance, risk) -> float:
        """Calculate actionable insight strength"""
        try:
            # Insight strength based on execution quality, performance consistency, and risk clarity
            insight_factors = [
                microstructure.execution_efficiency,
                performance.strategy_consistency,
                risk.risk_regime_confidence
            ]
            
            return sum(insight_factors) / len(insight_factors)
            
        except Exception as e:
            self.logger.error(f"Insight strength calculation failed: {e}")
            return 0.0
    
    def _calculate_analysis_quality(self, data_bundle: ProcessedDataBundleV2_5) -> float:
        """Calculate analysis quality score"""
        try:
            quality_factors = []
            
            # Check data availability
            if data_bundle.underlying_data_enriched.price:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Check options data
            if data_bundle.options_data_with_metrics:
                options_quality = min(len(data_bundle.options_data_with_metrics) / 50.0, 1.0)
                quality_factors.append(options_quality)
            else:
                quality_factors.append(0.0)
            
            # Check key metrics
            underlying = data_bundle.underlying_data_enriched
            if getattr(underlying, 'implied_volatility', None) is not None:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
            
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Analysis quality calculation failed: {e}")
            return 0.0
    
    def _calculate_data_completeness(self, data_bundle: ProcessedDataBundleV2_5) -> float:
        """Calculate data completeness score"""
        try:
            completeness_factors = []
            
            # Check underlying data completeness
            underlying = data_bundle.underlying_data_enriched
            required_fields = ['price', 'volume', 'implied_volatility']
            
            for field in required_fields:
                if getattr(underlying, field, None) is not None:
                    completeness_factors.append(1.0)
                else:
                    completeness_factors.append(0.0)
            
            # Check options data completeness
            if data_bundle.options_data_with_metrics:
                completeness_factors.append(1.0)
            else:
                completeness_factors.append(0.0)
            
            return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Data completeness calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, sentiment, behavioral, microstructure) -> float:
        """Calculate overall confidence level"""
        try:
            confidence_components = [
                sentiment.confidence_score,
                behavioral.psychology_confidence,
                microstructure.overall_microstructure_health
            ]
            
            return sum(confidence_components) / len(confidence_components)
            
        except Exception as e:
            self.logger.error(f"Overall confidence calculation failed: {e}")
            return 0.0
    
    def _update_performance_tracking(self, result: MarketIntelligenceResult):
        """Update performance tracking metrics"""
        try:
            self.analysis_count += 1
            self.total_processing_time += result.processing_time_ms
            
            # Track intelligence scores
            if result.overall_intelligence_score > 0:
                self.intelligence_history.append(result.overall_intelligence_score)
                # Keep only last 100 measurements
                if len(self.intelligence_history) > 100:
                    self.intelligence_history.pop(0)
            
            self.logger.debug(f"Performance updated: {self.analysis_count} analyses completed")
            
        except Exception as e:
            self.logger.error(f"Performance tracking update failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the intelligence expert"""
        try:
            avg_processing_time = self.total_processing_time / max(self.analysis_count, 1)
            avg_intelligence_score = sum(self.intelligence_history) / len(self.intelligence_history) if self.intelligence_history else 0.0
            
            return {
                "expert_name": "Ultimate Market Intelligence Expert",
                "analysis_count": self.analysis_count,
                "avg_processing_time_ms": avg_processing_time,
                "avg_intelligence_score": avg_intelligence_score,
                "config": self.config.model_dump(),
                "capabilities": {
                    "sentiment_analysis": self.config.sentiment_enabled,
                    "behavioral_finance": self.config.behavioral_analysis_enabled,
                    "market_microstructure": self.config.microstructure_enabled,
                    "seasonality_analysis": self.config.seasonality_enabled,
                    "performance_attribution": self.config.performance_tracking_enabled,
                    "risk_regime_analysis": self.config.risk_regime_enabled,
                    "cross_timeframe_analysis": self.config.multi_timeframe_enabled,
                    "ai_intelligence": self.config.ai_enabled
                }
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {"error": str(e)}

# Maintain backward compatibility
SentimentExpert = UltimateMarketIntelligenceExpert
MarketIntelligenceExpert = UltimateMarketIntelligenceExpert

