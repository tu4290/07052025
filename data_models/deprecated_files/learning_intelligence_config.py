"""
Learning & Intelligence Configuration Models for EOTS v2.5

This module contains learning system configurations, HuiHui AI settings,
and intelligence framework parameters.

Extracted from configuration_models.py for better modularity.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# LEARNING SYSTEM CONFIGURATION
# =============================================================================

class MarketContextData(BaseModel):
    """Market conditions and context relevant to learning insights."""
    market_regime: Optional[str] = Field(None, description="Current market regime")
    volatility_level: Optional[float] = Field(None, description="Current volatility level")
    trend_direction: Optional[str] = Field(None, description="Current trend direction")
    volume_profile: Optional[str] = Field(None, description="Volume profile analysis")
    sector_rotation: Optional[str] = Field(None, description="Sector rotation pattern")
    economic_indicators: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Economic indicators")
    technical_indicators: Optional[Dict[str, float]] = Field(default=None, description="Technical indicator values")
    sentiment_metrics: Optional[Dict[str, float]] = Field(default=None, description="Market sentiment metrics")
    model_config = ConfigDict(extra='forbid')

class AdaptationSuggestion(BaseModel):
    """Suggested adaptation parameters based on learning insights."""
    parameter_name: Optional[str] = Field(None, description="Name of parameter to adjust")
    current_value: Optional[Union[str, int, float, bool]] = Field(None, description="Current parameter value")
    suggested_value: Optional[Union[str, int, float, bool]] = Field(None, description="Suggested new value")
    adjustment_reason: Optional[str] = Field(None, description="Reason for the adjustment")
    expected_impact: Optional[str] = Field(None, description="Expected impact of the change")
    confidence_level: Optional[float] = Field(None, description="Confidence in the suggestion")
    risk_assessment: Optional[str] = Field(None, description="Risk assessment of the change")
    rollback_plan: Optional[str] = Field(None, description="Plan for rolling back if needed")
    custom_adaptations: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom adaptation parameters")
    model_config = ConfigDict(extra='forbid')

class PerformanceMetricsSnapshot(BaseModel):
    """Performance metrics snapshot for before/after comparisons."""
    accuracy_pct: Optional[float] = Field(None, description="Accuracy percentage")
    precision_pct: Optional[float] = Field(None, description="Precision percentage")
    recall_pct: Optional[float] = Field(None, description="Recall percentage")
    f1_score: Optional[float] = Field(None, description="F1 score")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown_pct: Optional[float] = Field(None, description="Maximum drawdown percentage")
    win_rate_pct: Optional[float] = Field(None, description="Win rate percentage")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    avg_return_pct: Optional[float] = Field(None, description="Average return percentage")
    volatility_pct: Optional[float] = Field(None, description="Volatility percentage")
    execution_time_ms: Optional[float] = Field(None, description="Average execution time in milliseconds")
    resource_usage_pct: Optional[float] = Field(None, description="Resource usage percentage")
    custom_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom performance metrics")
    model_config = ConfigDict(extra='forbid')

class LearningInsightData(BaseModel):
    """Summarized learning insights from analysis."""
    insight_type: Optional[str] = Field(None, description="Type of insight discovered")
    insight_description: Optional[str] = Field(None, description="Description of the insight")
    confidence_score: Optional[float] = Field(None, description="Confidence in the insight")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence supporting the insight")
    potential_impact: Optional[str] = Field(None, description="Potential impact of applying the insight")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions based on insight")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors to consider")
    validation_criteria: List[str] = Field(default_factory=list, description="Criteria for validating the insight")
    custom_insights: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom insight data")
    model_config = ConfigDict(extra='forbid')

class ExpertAdaptationSummary(BaseModel):
    """Summary of adaptations for expert systems."""
    expert_type: Optional[str] = Field(None, description="Type of expert system")
    adaptations_applied: List[str] = Field(default_factory=list, description="List of adaptations applied")
    performance_before: Optional[PerformanceMetricsSnapshot] = Field(None, description="Performance before adaptations")
    performance_after: Optional[PerformanceMetricsSnapshot] = Field(None, description="Performance after adaptations")
    adaptation_effectiveness: Optional[float] = Field(None, description="Effectiveness of adaptations")
    rollback_required: Optional[bool] = Field(None, description="Whether rollback is required")
    next_review_date: Optional[datetime] = Field(None, description="Next review date for adaptations")
    custom_summary: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom summary data")
    model_config = ConfigDict(extra='forbid')

class ConfidenceUpdateData(BaseModel):
    """Data for updating confidence scores."""
    component_name: Optional[str] = Field(None, description="Name of component being updated")
    old_confidence: Optional[float] = Field(None, description="Previous confidence score")
    new_confidence: Optional[float] = Field(None, description="New confidence score")
    update_reason: Optional[str] = Field(None, description="Reason for confidence update")
    supporting_data: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Data supporting the update")
    validation_results: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Validation results")
    impact_assessment: Optional[str] = Field(None, description="Assessment of update impact")
    custom_updates: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom update data")
    model_config = ConfigDict(extra='forbid')

class LearningSystemConfig(BaseModel):
    """Configuration for the learning system."""
    enabled: bool = Field(True, description="Enable/disable learning system.")
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="Learning rate for adaptations.")
    adaptation_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for applying adaptations.")
    validation_window_days: int = Field(30, ge=7, description="Window for validating learning insights.")
    max_adaptations_per_cycle: int = Field(5, ge=1, description="Maximum adaptations per learning cycle.")
    rollback_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Threshold for rolling back adaptations.")
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# HUIHUI AI SYSTEM CONFIGURATION
# =============================================================================

class AnalysisContext(BaseModel): # Duplicated from hui_hui_config_schemas.py
    """Context data for HuiHui analysis requests."""
    market_conditions: Optional[str] = Field(default=None, description="Current market conditions")
    recent_news: List[str] = Field(default_factory=list, description="Recent news items")
    volatility_regime: Optional[str] = Field(default=None, description="Current volatility regime")
    market_sentiment: Optional[str] = Field(default=None, description="Market sentiment")
    time_of_day: Optional[str] = Field(default=None, description="Time context")
    custom_context: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom context fields")
    model_config = ConfigDict(extra='forbid')

class RequestMetadata(BaseModel): # Duplicated from hui_hui_config_schemas.py
    """Metadata for HuiHui analysis requests."""
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    priority: Optional[str] = Field(default=None, description="Request priority")
    source: Optional[str] = Field(default=None, description="Request source")
    custom_metadata: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom metadata fields")
    model_config = ConfigDict(extra='forbid')

class EOTSPrediction(BaseModel): # Duplicated from hui_hui_config_schemas.py
    """EOTS prediction data structure."""
    prediction_type: str = Field(description="Type of prediction")
    symbol: str = Field(description="Symbol for prediction")
    confidence: float = Field(description="Prediction confidence")
    timeframe: str = Field(description="Prediction timeframe")
    direction: str = Field(description="Predicted direction")
    target_price: Optional[float] = Field(default=None, description="Target price if applicable")
    probability: Optional[float] = Field(default=None, description="Probability estimate")
    custom_data: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom prediction data")
    model_config = ConfigDict(extra='forbid')

class TradingRecommendation(BaseModel): # Duplicated from hui_hui_config_schemas.py
    """Trading recommendation structure."""
    action: str = Field(description="Recommended action (buy/sell/hold)")
    symbol: str = Field(description="Symbol for recommendation")
    quantity: Optional[int] = Field(default=None, description="Recommended quantity")
    price_target: Optional[float] = Field(default=None, description="Price target")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss level")
    confidence: float = Field(description="Recommendation confidence")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for recommendation")
    risk_level: Optional[str] = Field(default=None, description="Risk level")
    timeframe: Optional[str] = Field(default=None, description="Recommendation timeframe")
    custom_attributes: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom recommendation attributes")
    model_config = ConfigDict(extra='forbid')

class PerformanceByCondition(BaseModel): # Duplicated from hui_hui_config_schemas.py
    """Performance metrics by market condition."""
    success_rate: Optional[float] = Field(default=None, description="Success rate for this condition")
    avg_processing_time: Optional[float] = Field(default=None, description="Average processing time")
    total_requests: int = Field(default=0, description="Total requests in this condition")
    avg_confidence: Optional[float] = Field(default=None, description="Average confidence score")
    error_rate: Optional[float] = Field(default=None, description="Error rate for this condition")
    custom_metrics: Optional[Dict[str, float]] = Field(default=None, description="Custom performance metrics")
    model_config = ConfigDict(extra='forbid')

class HuiHuiSystemConfig(BaseModel):
    """Configuration for the HuiHui AI system."""
    enabled: bool = Field(True, description="Enable/disable HuiHui AI system.")
    model_provider: Optional[str] = Field(default="openai", description="AI model provider for HuiHui.") # Made Optional
    model_name: Optional[str] = Field(default="gpt-4", description="AI model name for HuiHui.") # Made Optional
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature for responses.")
    max_tokens: int = Field(default=4000, ge=100, description="Maximum tokens for responses.")
    timeout_seconds: int = Field(default=30, ge=5, description="Timeout for AI requests.")
    retry_attempts: int = Field(default=3, ge=1, description="Number of retry attempts for failed requests.")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold for responses.")
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# INTELLIGENCE FRAMEWORK CONFIGURATION
# =============================================================================

class IntelligenceFrameworkConfig(BaseModel):
    """Configuration for the overall intelligence framework."""
    enabled: bool = Field(True, description="Enable/disable intelligence framework.")
    learning_system: LearningSystemConfig = Field(default_factory=LearningSystemConfig, description="Learning system configuration.")
    huihui_system: HuiHuiSystemConfig = Field(default_factory=HuiHuiSystemConfig, description="HuiHui AI system configuration.")
    intelligence_update_interval_seconds: int = Field(300, ge=60, description="Interval for intelligence updates.")
    cross_system_learning: bool = Field(True, description="Enable learning across different systems.")
    knowledge_persistence: bool = Field(True, description="Enable persistence of learned knowledge.")
    learning_params: Optional[Dict[str, Any]] = Field(default=None, description="Learning parameters configuration")
    model_config = ConfigDict(extra='forbid')