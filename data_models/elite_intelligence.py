import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# =============================================================================
# CONSOLIDATED DEFINITIONS - Enums and Dataclasses
# =============================================================================

class MarketRegime(Enum):
    """Market regime classifications for dynamic adaptation"""
    LOW_VOL_TRENDING = "low_vol_trending"
    LOW_VOL_RANGING = "low_vol_ranging"
    MEDIUM_VOL_TRENDING = "medium_vol_trending"
    MEDIUM_VOL_RANGING = "medium_vol_ranging"
    HIGH_VOL_TRENDING = "high_vol_trending"
    HIGH_VOL_RANGING = "high_vol_ranging"
    STRESS_REGIME = "stress_regime"
    EXPIRATION_REGIME = "expiration_regime"
    REGIME_UNCLEAR_OR_TRANSITIONING = "regime_unclear_or_transitioning"

class FlowType(Enum):
    """Flow classification types for institutional intelligence"""
    RETAIL_UNSOPHISTICATED = "retail_unsophisticated"
    RETAIL_SOPHISTICATED = "retail_sophisticated"
    INSTITUTIONAL_HEDGING = "institutional_hedging"
    INSTITUTIONAL_DIRECTIONAL = "institutional_directional"
    MARKET_MAKER_FLOW = "market_maker_flow"
    MIXED_FLOW = "mixed_flow"
    FLOW_UNCLEAR = "flow_unclear"

class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    SUBDUED = "subdued"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"

@dataclass
class ConvexValueColumns:
    """Column mappings for ConvexValue data"""
    # Price and Greeks
    STRIKE: str = 'strike'
    OPTION_TYPE: str = 'opt_kind'
    DELTA: str = 'delta_contract'
    GAMMA: str = 'gamma_contract'
    VEGA: str = 'vega_contract'
    THETA: str = 'theta_contract'
    
    # Volume and OI
    VOLUME: str = 'volm'
    OPEN_INTEREST: str = 'open_interest'
    
    # Exposure metrics
    DXOI: str = 'dxoi'
    GXOI: str = 'gxoi'
    VXOI: str = 'vxoi'
    TXOI: str = 'txoi'
    
    # Flow metrics
    VOLUME_BS: str = 'volm_bs'
    VALUE_BS: str = 'value_bs'
    
    # Additional metrics
    IMPLIED_VOL: str = 'implied_volatility'
    DTE: str = 'dte_calc'

@dataclass
class EliteImpactColumns:
    """Column mappings for elite impact calculations"""
    # Elite scores
    ELITE_IMPACT_SCORE: str = 'elite_impact_score_und'
    INSTITUTIONAL_FLOW_SCORE: str = 'institutional_flow_score_und'
    FLOW_MOMENTUM_INDEX: str = 'flow_momentum_index_und'
    
    # Regime classifications
    MARKET_REGIME: str = 'market_regime_elite'
    FLOW_TYPE: str = 'flow_type_elite'
    VOLATILITY_REGIME: str = 'volatility_regime_elite'
    
    # Confidence metrics
    CONFIDENCE: str = 'confidence'
    TRANSITION_RISK: str = 'transition_risk'

class EliteConfig(BaseModel):
    """Configuration for elite impact calculations"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Thresholds
    large_trade_threshold: float = Field(default=100000, description="Threshold for large trades")
    institutional_volume_pct: float = Field(default=0.2, description="Percentage of volume considered institutional")
    
    # Weights
    volume_weight: float = Field(default=0.3, description="Weight for volume in calculations")
    oi_weight: float = Field(default=0.2, description="Weight for open interest")
    greek_weight: float = Field(default=0.3, description="Weight for Greeks")
    flow_weight: float = Field(default=0.2, description="Weight for flow metrics")
    
    # Regime detection
    volatility_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'subdued': 0.15,
            'normal': 0.25,
            'elevated': 0.40,
            'extreme': 0.60
        }
    )
    
    # Flow classification
    flow_size_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'retail': 10000,
            'sophisticated': 50000,
            'institutional': 100000
        }
    )

class EliteImpactResultsV2_5(BaseModel):
    """Results from elite impact calculations"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core scores
    elite_impact_score_und: float = Field(description="Overall elite impact score")
    institutional_flow_score_und: float = Field(description="Institutional flow score")
    flow_momentum_index_und: float = Field(description="Flow momentum index")
    
    # Regime classifications
    market_regime_elite: str = Field(description="Market regime classification")
    flow_type_elite: str = Field(description="Flow type classification")
    volatility_regime_elite: str = Field(description="Volatility regime classification")
    
    # Confidence metrics
    confidence: float = Field(description="Confidence in calculations")
    transition_risk: float = Field(description="Risk of regime transition")
    
    # Additional metrics
    large_trade_count: int = Field(default=0, description="Number of large trades")
    institutional_volume_ratio: float = Field(default=0.0, description="Ratio of institutional volume")
    smart_money_indicator: float = Field(default=0.0, description="Smart money flow indicator")

__all__ = [
    'EliteConfig', 'MarketRegime', 'FlowType', 'VolatilityRegime',
    'ConvexValueColumns', 'EliteImpactColumns', 'EliteImpactResultsV2_5'
]
