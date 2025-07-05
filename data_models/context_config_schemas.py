# context_config_schemas.py
# This module defines Pydantic models to replace Dict[str, Any] patterns in context schemas

from typing import Dict, Any, Union
from pydantic import BaseModel, Field

class AdvancedOptionsMetrics(BaseModel):
    """Advanced options metrics for context analysis."""
    gamma_exposure: float = Field(default=0.0, description="Gamma exposure metric")
    vanna_exposure: float = Field(default=0.0, description="Vanna exposure metric")
    charm_exposure: float = Field(default=0.0, description="Charm exposure metric")
    volga_exposure: float = Field(default=0.0, description="Volga exposure metric")
    theta_decay: float = Field(default=0.0, description="Theta decay metric")
    delta_hedging_flow: float = Field(default=0.0, description="Delta hedging flow")
    pin_risk: float = Field(default=0.0, description="Pin risk assessment")
    skew_metrics: float = Field(default=0.0, description="Volatility skew metrics")
    term_structure_slope: float = Field(default=0.0, description="Term structure slope")
    custom_metrics: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Custom options metrics")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
