# hui_hui_config_schemas.py
# This module defines Pydantic models to replace Dict[str, Any] patterns in HuiHui AI system

from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

class AnalysisContext(BaseModel):
    """Context data for HuiHui analysis requests."""
    market_conditions: Optional[str] = Field(default=None, description="Current market conditions")
    recent_news: List[str] = Field(default_factory=list, description="Recent news items")
    volatility_regime: Optional[str] = Field(default=None, description="Current volatility regime")
    market_sentiment: Optional[str] = Field(default=None, description="Market sentiment")
    time_of_day: Optional[str] = Field(default=None, description="Time context")
    custom_context: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom context fields")
    model_config = ConfigDict(extra='forbid')

class RequestMetadata(BaseModel):
    """Metadata for HuiHui analysis requests."""
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    priority: Optional[str] = Field(default=None, description="Request priority")
    source: Optional[str] = Field(default=None, description="Request source")
    custom_metadata: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom metadata fields")
    model_config = ConfigDict(extra='forbid')

class EOTSPrediction(BaseModel):
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

class TradingRecommendation(BaseModel):
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

class PerformanceByCondition(BaseModel):
    """Performance metrics by market condition."""
    success_rate: Optional[float] = Field(default=None, description="Success rate for this condition")
    avg_processing_time: Optional[float] = Field(default=None, description="Average processing time")
    total_requests: int = Field(default=0, description="Total requests in this condition") # 0 is valid start
    avg_confidence: Optional[float] = Field(default=None, description="Average confidence score")
    error_rate: Optional[float] = Field(default=None, description="Error rate for this condition")
    custom_metrics: Optional[Dict[str, float]] = Field(default=None, description="Custom performance metrics")
    model_config = ConfigDict(extra='forbid')