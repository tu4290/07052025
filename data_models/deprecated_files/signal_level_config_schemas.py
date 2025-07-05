# signal_level_config_schemas.py
# This module defines Pydantic models to replace Dict[str, Any] patterns in signal level schemas

from typing import Dict, Any, Union
from pydantic import BaseModel, Field

class SupportingMetrics(BaseModel):
    """Supporting metrics that contributed to triggering a signal."""
    vapi_fa_score: float = Field(default=0.0, description="VAPI FA score")
    a_mspi_score: float = Field(default=0.0, description="A-MSPI score")
    flow_divergence: float = Field(default=0.0, description="Flow divergence metric")
    volatility_spike: float = Field(default=0.0, description="Volatility spike indicator")
    momentum_strength: float = Field(default=0.0, description="Momentum strength")
    regime_confidence: float = Field(default=0.0, description="Regime confidence level")
    structural_support: float = Field(default=0.0, description="Structural support level")
from pydantic import BaseModel, Field, ConfigDict # Added ConfigDict
from typing import Dict, Any, Union, Optional # Added Optional

class SupportingMetrics(BaseModel):
    """Supporting metrics that contributed to triggering a signal."""
    vapi_fa_score: Optional[float] = Field(default=None, description="VAPI FA score")
    a_mspi_score: Optional[float] = Field(default=None, description="A-MSPI score")
    flow_divergence: Optional[float] = Field(default=None, description="Flow divergence metric")
    volatility_spike: Optional[float] = Field(default=None, description="Volatility spike indicator")
    momentum_strength: Optional[float] = Field(default=None, description="Momentum strength")
    regime_confidence: Optional[float] = Field(default=None, description="Regime confidence level")
    structural_support: Optional[float] = Field(default=None, description="Structural support level")
    volume_confirmation: Optional[float] = Field(default=None, description="Volume confirmation")
    price_action_quality: Optional[float] = Field(default=None, description="Price action quality")
    time_decay_factor: Optional[float] = Field(default=None, description="Time decay factor")
    custom_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom supporting metrics")
    model_config = ConfigDict(extra='forbid')