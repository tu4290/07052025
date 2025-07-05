"""
Compatibility layer for EliteConfig import
Re-exports EliteConfig from core_analytics_engine
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
from enum import Enum

class MarketRegime(str, Enum):
    LOW_VOL_RANGING = "low_vol_ranging"
    HIGH_VOL_TRENDING = "high_vol_trending"
    CONSOLIDATION = "consolidation"
    TRANSITION = "transition"
    UNDEFINED = "undefined"

class EliteConfig(BaseModel):
    """
    Centralized configuration model for Elite Options Trading System
    Consolidates settings from multiple components
    """
    system_name: str = Field(default="Elite Options Trading System v2.5")
    
    # Market Regime Settings
    market_regime: MarketRegime = Field(default=MarketRegime.UNDEFINED)
    regime_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Adaptive Learning Parameters
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    adaptation_cycle: int = Field(default=10, ge=1)
    
    # Risk Management
    max_trade_risk: float = Field(default=0.02, ge=0.0, le=0.1)
    portfolio_allocation_factor: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # Performance Tracking
    performance_window: int = Field(default=30, ge=1)
    
    # Expert System Configuration
    expert_weights: Dict[str, float] = Field(default_factory=lambda: {
        "market_regime": 0.3,
        "options_flow": 0.3,
        "sentiment": 0.2,
        "technical_analysis": 0.2
    })
    
    # Advanced Configuration
    advanced_settings: Optional[Dict[str, Any]] = Field(default=None)
    
    model_config = ConfigDict(
        title="Elite Options System Configuration",
        extra='forbid',  # Prevent additional fields
        strict=True,     # Enforce type checking
        validate_assignment=True  # Validate on attribute assignment
    )
    
    def get_regime_strategy(self) -> str:
        """Determine trading strategy based on market regime"""
        regime_strategies = {
            MarketRegime.LOW_VOL_RANGING: "mean_reversion",
            MarketRegime.HIGH_VOL_TRENDING: "trend_following",
            MarketRegime.CONSOLIDATION: "range_trading",
            MarketRegime.TRANSITION: "adaptive_hedging",
            MarketRegime.UNDEFINED: "conservative_cash"
        }
        return regime_strategies.get(self.market_regime, "conservative_cash")

# Re-export the EliteConfig for backward compatibility
__all__ = ['EliteConfig'] 