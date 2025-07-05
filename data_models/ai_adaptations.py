
"""
Pydantic models for AI model adaptations and performance.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from typing_extensions import Literal

class AIAdaptationV2_5(BaseModel):
    """Tracks AI model adaptations and adjustments for market conditions."""
    id: Optional[int] = Field(None, description="Auto-generated database ID", ge=1)
    adaptation_type: Literal['signal_enhancement', 'threshold_adjustment', 'model_calibration', 'feature_engineering']
    adaptation_name: str = Field(..., min_length=3, max_length=100)
    adaptation_description: Optional[str] = Field(None, max_length=1000)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    implementation_status: Literal['PENDING', 'ACTIVE', 'INACTIVE', 'DEPRECATED', 'TESTING'] = "PENDING"
    market_context: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    parent_model_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AIAdaptationPerformanceV2_5(BaseModel):
    """Tracks performance metrics for AI model adaptations over time."""
    adaptation_id: int = Field(..., ge=1)
    symbol: str = Field(..., max_length=20)
    time_period_days: int = Field(..., ge=1)
    total_applications: int = Field(..., ge=0)
    successful_applications: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_improvement: float
    adaptation_score: float = Field(..., ge=0.0, le=1.0)
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN']
    avg_processing_time_ms: float = Field(..., ge=0.0)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
