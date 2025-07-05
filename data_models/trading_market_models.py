from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator, ConfigDict

class TickerContextDictV2_5(BaseModel):
    """
    Contextual information specific to a ticker, used for market analysis.
    This model is designed to be flexible and can include various market-related data points.
    """
    symbol: str = Field(..., description="The ticker symbol.")
    current_price: float = Field(..., description="Current price of the underlying asset.")
    market_cap: Optional[float] = Field(None, description="Market capitalization.")
    sector: Optional[str] = Field(None, description="Sector of the company.")
    industry: Optional[str] = Field(None, description="Industry of the company.")
    news_sentiment_score: Optional[float] = Field(None, description="Aggregated news sentiment score.")
    social_media_sentiment_score: Optional[float] = Field(None, description="Aggregated social media sentiment score.")
    analyst_ratings: Optional[Dict[str, Any]] = Field(None, description="Analyst ratings and price targets.")
    earnings_date: Optional[str] = Field(None, description="Upcoming earnings date.")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield.")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio.")
    beta: Optional[float] = Field(None, description="Beta of the stock.")
    historical_volatility_30d: Optional[float] = Field(None, description="30-day historical volatility.")
    implied_volatility_30d: Optional[float] = Field(None, description="30-day implied volatility.")
    options_volume_24h: Optional[int] = Field(None, description="24-hour options trading volume.")
    open_interest_total: Optional[int] = Field(None, description="Total open interest across all options.")
    put_call_ratio: Optional[float] = Field(None, description="Put/Call volume or OI ratio.")
    economic_data_impact: Optional[Dict[str, Any]] = Field(None, description="Impact from recent economic data.")
    key_technical_levels: Optional[List[float]] = Field(None, description="Important support and resistance levels.")
    market_regime_classification: Optional[str] = Field(None, description="Current market regime classification.")
    liquidity_score: Optional[float] = Field(None, description="Score indicating liquidity of the asset.")
    trading_volume_24h: Optional[int] = Field(None, description="24-hour trading volume of the underlying.")
    average_daily_volume_30d: Optional[int] = Field(None, description="30-day average daily trading volume.")
    # Add any other relevant context fields as needed

    model_config = ConfigDict(extra='allow') # Allow extra fields for flexibility

class DynamicThresholdsV2_5(BaseModel):
    """
    Defines dynamic thresholds for various metrics, adapting to market conditions.
    These thresholds are used to classify signals, identify significant movements,
    and adjust system behavior based on real-time market dynamics.
    """
    # Volatility thresholds
    volatility_high_threshold: float = Field(..., description="Threshold for classifying high volatility.")
    volatility_low_threshold: float = Field(..., description="Threshold for classifying low volatility.")
    
    # Flow thresholds
    flow_strong_buy_threshold: float = Field(..., description="Threshold for strong bullish options flow.")
    flow_strong_sell_threshold: float = Field(..., description="Threshold for strong bearish options flow.")
    flow_neutral_range: List[float] = Field(..., min_length=2, max_length=2, description="Range for neutral options flow [lower, upper].")
    
    # Sentiment thresholds
    sentiment_extreme_bullish: float = Field(..., description="Threshold for extreme bullish sentiment.")
    sentiment_extreme_bearish: float = Field(..., description="Threshold for extreme bearish sentiment.")
    
    # Liquidity thresholds
    liquidity_high_threshold: float = Field(..., description="Threshold for high liquidity conditions.")
    liquidity_low_threshold: float = Field(..., description="Threshold for low liquidity conditions.")
    
    # Signal strength thresholds
    signal_strong_threshold: float = Field(..., description="Threshold for a strong trading signal.")
    signal_moderate_threshold: float = Field(..., description="Threshold for a moderate trading signal.")
    
    # Risk management thresholds
    max_drawdown_alert_pct: float = Field(..., description="Percentage drawdown at which to trigger an alert.")
    max_position_size_pct: float = Field(..., description="Maximum percentage of portfolio to allocate to a single position.")
    
    # Adaptive learning thresholds
    model_performance_degradation_alert: float = Field(..., description="Threshold for alerting on model performance degradation.")
    data_freshness_stale_threshold_minutes: int = Field(..., description="Minutes after which data is considered stale.")

    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_threshold_consistency(self) -> 'DynamicThresholdsV2_5':
        """Ensures logical consistency between related thresholds."""
        if not (self.volatility_low_threshold < self.volatility_high_threshold):
            raise ValueError("volatility_low_threshold must be less than volatility_high_threshold")
        if not (self.flow_neutral_range[0] < self.flow_neutral_range[1]):
            raise ValueError("flow_neutral_range lower bound must be less than upper bound")
        if not (self.flow_strong_sell_threshold < self.flow_neutral_range[0] and self.flow_strong_buy_threshold > self.flow_neutral_range[1]):
            raise ValueError("Flow thresholds are inconsistent with neutral range")
        
        return self

class ATIFSituationalAssessmentProfileV2_5(BaseModel):
    """
    Represents a comprehensive situational assessment profile generated by the ATIF.
    This profile encapsulates various market conditions, system states, and intelligence
    metrics relevant for dynamic strategy adaptation and trade management.
    """
    timestamp: str = Field(..., description="Timestamp of the assessment.")
    symbol: str = Field(..., description="Ticker symbol of the asset being assessed.")
    market_regime: str = Field(..., description="Current market regime (e.g., 'bullish_trending', 'bearish_volatile').")
    volatility_regime: str = Field(..., description="Current volatility regime (e.g., 'high', 'low', 'expanding').")
    liquidity_conditions: str = Field(..., description="Current liquidity conditions (e.g., 'high', 'moderate', 'low').")
    sentiment_bias: str = Field(..., description="Overall market sentiment bias (e.g., 'bullish', 'bearish', 'neutral').")
    options_flow_bias: str = Field(..., description="Options flow directional bias (e.g., 'call_heavy', 'put_heavy', 'balanced').")
    key_levels_proximity: List[str] = Field(default_factory=list, description="List of key levels the price is currently near.")
    signal_confluence_score: float = Field(..., ge=0.0, le=1.0, description="Score indicating the confluence of various trading signals.")
    risk_appetite_index: float = Field(..., ge=0.0, le=1.0, description="Index reflecting the system's current risk appetite.")
    system_health_status: str = Field(..., description="Overall health status of the EOTS system.")
    active_recommendations_count: int = Field(..., ge=0, description="Number of active trade recommendations.")
    pending_orders_count: int = Field(..., ge=0, description="Number of pending trade orders.")
    news_impact_score: float = Field(..., ge=0.0, le=1.0, description="Score reflecting the impact of recent news.")
    economic_data_impact: str = Field(..., description="Summary of impact from recent economic data.")
    model_performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics of integrated AI models.")
    adaptive_learning_status: str = Field(..., description="Status of the adaptive learning system.")
    
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_assessment_consistency(self) -> 'ATIFSituationalAssessmentProfileV2_5':
        """Ensures logical consistency within the assessment profile."""
        # Example: If market_regime is 'bullish_trending', sentiment_bias should ideally be 'bullish'
        if 'bullish' in self.market_regime and self.sentiment_bias == 'bearish':
            import warnings
            warnings.warn("Situational assessment: Market regime and sentiment bias are conflicting.")
        return self

class ATIFStrategyDirectivePayloadV2_5(BaseModel):
    """
    Represents a strategic directive generated by the ATIF for trade management.
    This directive provides actionable instructions for adjusting or exiting existing
    trades, or for initiating new ones based on the current situational assessment.
    """
    directive_id: str = Field(..., description="Unique identifier for the directive.")
    timestamp: str = Field(..., description="Timestamp when the directive was issued.")
    symbol: str = Field(..., description="Ticker symbol the directive applies to.")
    directive_type: str = Field(..., description="Type of directive (e.g., 'adjust_stop_loss', 'take_profit', 'initiate_trade', 'exit_trade').")
    target_recommendation_id: Optional[str] = Field(None, description="ID of the recommendation this directive targets, if applicable.")
    action_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action (e.g., new_stop_loss_price, new_target_price, trade_details for new trade).")
    conviction_score: float = Field(..., ge=0.0, le=1.0, description="Conviction score for the directive (0.0 to 1.0).")
    rationale: str = Field(..., min_length=10, description="Detailed rationale for the directive.")
    assessment_profile: ATIFSituationalAssessmentProfileV2_5 = Field(..., description="The detailed situational assessment that led to this directive.")
    risk_adjustment_factor: float = Field(1.0, ge=0.0, le=2.0, description="Factor to adjust risk based on directive (1.0 for no change).")
    expected_impact: str = Field(..., description="Expected impact of executing this directive.")
    
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_action_parameters(self) -> 'ATIFStrategyDirectivePayloadV2_5':
        """Validates action parameters based on directive type."""
        if self.directive_type == 'adjust_stop_loss':
            if 'new_stop_loss_price' not in self.action_parameters:
                raise ValueError("adjust_stop_loss directive requires 'new_stop_loss_price' in action_parameters.")
        # Add more validation rules for other directive types as needed
        return self