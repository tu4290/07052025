
"""
Pydantic models for AI-driven predictions and performance tracking.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from typing_extensions import Literal

class AIPredictionV2_5(BaseModel):
    """Represents an AI-generated prediction for market analysis and trading decisions."""
    id: Optional[int] = Field(None, description="Auto-generated database ID", ge=1)
    symbol: str = Field(..., description="Trading symbol", max_length=20)
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment']
    prediction_value: Optional[float] = Field(None, description="Predicted numerical value")
    prediction_direction: Literal['UP', 'DOWN', 'NEUTRAL']
    confidence_score: float = Field(..., description="Model's confidence (0.0 to 1.0)", ge=0.0, le=1.0)
    time_horizon: str = Field(..., description="Time frame for the prediction (e.g., '1H', '1D')")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    target_timestamp: datetime = Field(..., description="When the prediction should be evaluated")
    actual_value: Optional[float] = Field(None, description="Actual observed value")
    actual_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']] = Field(None, description="Actual observed direction")
    prediction_accurate: Optional[bool] = Field(None, description="Whether the prediction was accurate")
    accuracy_score: Optional[float] = Field(None, description="Quantitative accuracy score (0.0 to 1.0)", ge=0.0, le=1.0)
    model_version: str = Field(default="v2.5")
    model_name: Optional[str] = Field(None)
    market_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AIPredictionPerformanceV2_5(BaseModel):
    """Tracks and analyzes the performance of AI predictions over time."""
    symbol: str = Field(..., max_length=20)
    model_name: str = Field(default="default")
    time_period_days: int = Field(..., ge=1, le=3650)
    total_predictions: int = Field(..., ge=0)
    correct_predictions: int = Field(..., ge=0)
    incorrect_predictions: int = Field(..., ge=0)
    pending_predictions: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    learning_score: float = Field(..., ge=0.0, le=1.0)
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN']
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class AIPredictionRequestV2_5(BaseModel):
    """Request model for creating new AI predictions."""
    symbol: str = Field(..., max_length=20)
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment']
    prediction_value: Optional[float] = Field(None, ge=0.0)
    prediction_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']]
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: Optional[str] = Field(None, max_length=100)
    time_horizon: str = Field(..., pattern=r'^\d+[mhdwMqy]$')
    target_timestamp: datetime
    market_context: Dict[str, Any] = Field(default_factory=dict)
    request_metadata: Dict[str, Any] = Field(default_factory=dict)

class AIPredictionSummaryV2_5(BaseModel):
    """Unified summary of AI predictions and their performance."""
    symbol: str = Field(..., description="Trading symbol")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    latest_prediction: Optional[AIPredictionV2_5] = Field(None, description="The most recent prediction for the symbol")
    overall_performance: Optional[AIPredictionPerformanceV2_5] = Field(None, description="Aggregated performance metrics")
    active_predictions_count: int = Field(..., description="Number of active predictions")
    pending_predictions_count: int = Field(..., description="Number of predictions awaiting outcome")
    prediction_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score of predictions")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Average confidence of active predictions")
    optimization_recommendations: List[str] = Field(default_factory=list, description="Recommendations for model optimization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
