"""
Pydantic models related to impact analysis calculations.
"""
from pydantic import BaseModel, Field
from typing import Optional

class EliteImpactResult(BaseModel):
    """
    Pydantic model for a single row of elite impact output.
    Refactored to remove default fallbacks like 0.0 or 'unknown'.
    Fields are Optional if their calculation might not yield a value,
    using None to represent missing/non-applicable data.
    """
    delta_impact_raw: Optional[float] = Field(default=None)
    gamma_impact_raw: Optional[float] = Field(default=None)
    vega_impact_raw: Optional[float] = Field(default=None)
    theta_impact_raw: Optional[float] = Field(default=None)
    vanna_impact_raw: Optional[float] = Field(default=None)
    vomma_impact_raw: Optional[float] = Field(default=None)
    charm_impact_raw: Optional[float] = Field(default=None)
    sdag_multiplicative: Optional[float] = Field(default=None)
    sdag_directional: Optional[float] = Field(default=None)
    sdag_weighted: Optional[float] = Field(default=None)
    sdag_volatility_focused: Optional[float] = Field(default=None)
    sdag_consensus: Optional[float] = Field(default=None)
    dag_multiplicative: Optional[float] = Field(default=None)
    dag_directional: Optional[float] = Field(default=None)
    dag_weighted: Optional[float] = Field(default=None)
    dag_volatility_focused: Optional[float] = Field(default=None)
    dag_consensus: Optional[float] = Field(default=None)
    strike_magnetism_index: Optional[float] = Field(default=None)
    volatility_pressure_index: Optional[float] = Field(default=None)
    flow_momentum_index: Optional[float] = Field(default=None)
    institutional_flow_score: Optional[float] = Field(default=None)
    regime_adjusted_gamma: Optional[float] = Field(default=None)
    regime_adjusted_delta: Optional[float] = Field(default=None)
    regime_adjusted_vega: Optional[float] = Field(default=None)
    cross_exp_gamma_surface: Optional[float] = Field(default=None)
    expiration_transition_factor: Optional[float] = Field(default=None)
    flow_velocity_5m: Optional[float] = Field(default=None)
    flow_velocity_15m: Optional[float] = Field(default=None)
    flow_acceleration: Optional[float] = Field(default=None)
    momentum_persistence: Optional[float] = Field(default=None)
    market_regime: Optional[str] = Field(default=None)
    flow_type: Optional[str] = Field(default=None)
    volatility_regime: Optional[str] = Field(default=None)
    elite_impact_score: Optional[float] = Field(default=None)
    prediction_confidence: Optional[float] = Field(default=None)
    signal_strength: Optional[float] = Field(default=None)

    class Config:
        extra = 'forbid' # Keep it strict

__all__ = ['EliteImpactResult']