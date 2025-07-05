# processed_data_config_schemas.py
# This module defines Pydantic models to replace Dict[str, Any] patterns in processed data

from typing import Dict, Any, Union
from pydantic import BaseModel, Field

class DynamicThresholds(BaseModel):
    """Dynamic thresholds used in analysis cycles."""
    volatility_threshold: float = Field(default=0.0, description="Volatility threshold")
    volume_threshold: float = Field(default=0.0, description="Volume threshold")
    price_threshold: float = Field(default=0.0, description="Price threshold")
    momentum_threshold: float = Field(default=0.0, description="Momentum threshold")
    trend_threshold: float = Field(default=0.0, description="Trend threshold")
    flow_threshold: float = Field(default=0.0, description="Flow threshold")
    regime_threshold: float = Field(default=0.0, description="Regime threshold")
    custom_thresholds: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Custom threshold values")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()