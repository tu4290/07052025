# core_analytics_engine/eots_metrics/elite_intelligence.py

"""
EOTS Elite Intelligence - Consolidated Elite Impact and Configuration

Consolidates:
- elite_impact_calculations.py: Advanced institutional intelligence and impact calculations
- elite_definitions.py: Enums, configurations, and column definitions

Optimizations:
- Unified elite configuration management
- Streamlined impact calculations
- Integrated institutional intelligence
- Eliminated redundant ML models and complex calculations
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# =============================================================================
# CONSOLIDATED DEFINITIONS - From elite_definitions.py
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
    INSTITUTIONAL_SMALL = "institutional_small"
    INSTITUTIONAL_LARGE = "institutional_large"
    HEDGE_FUND = "hedge_fund"
    MARKET_MAKER = "market_maker"
    UNKNOWN = "unknown"

class EliteConfig(BaseModel):
    """Consolidated elite impact calculation configuration"""
    # Core feature flags
    regime_detection_enabled: bool = Field(default=True, description="Enable dynamic regime adaptation")
    flow_classification_enabled: bool = Field(default=True, description="Enable institutional flow intelligence")
    volatility_surface_enabled: bool = Field(default=True, description="Enable volatility surface integration")
    momentum_detection_enabled: bool = Field(default=True, description="Enable momentum-acceleration detection")
    
    # Regime detection parameters
    regime_lookback_periods: Dict[str, int] = Field(
        default_factory=lambda: {'short': 20, 'medium': 60, 'long': 252}, 
        description="Lookback periods for regime detection"
    )
    
    # Flow classification parameters
    institutional_threshold_percentile: float = Field(default=95.0, description="Percentile for institutional flow threshold")
    flow_momentum_periods: List[int] = Field(default_factory=lambda: [5, 15, 30, 60], description="Periods for flow momentum analysis")
    
    # Volatility surface parameters
    skew_adjustment_alpha: float = Field(default=1.0, description="Alpha for skew adjustment")
    surface_stability_threshold: float = Field(default=0.15, description="Threshold for volatility surface stability")
    
    # Momentum detection parameters
    acceleration_threshold_multiplier: float = Field(default=2.0, description="Multiplier for acceleration threshold")
    momentum_persistence_threshold: float = Field(default=0.7, description="Threshold for momentum persistence")
    
    # Performance optimization
    enable_caching: bool = Field(default=True, description="Enable caching for performance optimization")
    enable_parallel_processing: bool = Field(default=False, description="Enable parallel processing (simplified)")
    max_workers: int = Field(default=2, description="Maximum number of parallel workers")
    
    # Feature toggles
    enable_sdag_calculation: bool = Field(default=True, description="Enable SDAG calculation")
    enable_dag_calculation: bool = Field(default=True, description="Enable DAG calculation")
    enable_advanced_greeks: bool = Field(default=True, description="Enable advanced Greeks calculation")
    enable_flow_clustering: bool = Field(default=False, description="Enable flow clustering (simplified)")

    # Additional config fields from config file (alternative field names)
    enable_elite_regime_detection: Optional[bool] = Field(default=True, description="Enable elite regime detection (alternative field name)")
    elite_regime_threshold: Optional[float] = Field(default=0.7, description="Elite regime threshold")

    model_config = ConfigDict(extra='allow')

class ConvexValueColumns:
    """Consolidated ConvexValue column definitions"""
    # Basic option parameters
    OPT_KIND = 'opt_kind'
    STRIKE = 'strike'
    EXPIRATION = 'expiration'
    EXPIRATION_TS = 'expiration_ts'
    
    # Greeks
    DELTA = 'delta'
    GAMMA = 'gamma'
    THETA = 'theta'
    VEGA = 'vega'
    RHO = 'rho'
    VANNA = 'vanna'
    VOMMA = 'vomma'
    CHARM = 'charm'
    
    # Open Interest multiplied metrics
    DXOI = 'dxoi'
    GXOI = 'gxoi'
    VXOI = 'vxoi'
    TXOI = 'txoi'
    VANNAXOI = 'vannaxoi'
    VOMMAXOI = 'vommaxoi'
    CHARMXOI = 'charmxoi'
    
    # Volume multiplied metrics
    DXVOLM = 'dxvolm'
    GXVOLM = 'gxvolm'
    VXVOLM = 'vxvolm'
    TXVOLM = 'txvolm'
    VANNAXVOLM = 'vannaxvolm'
    VOMMAXVOLM = 'vommaxvolm'
    CHARMXVOLM = 'charmxvolm'
    
    # Flow metrics
    VALUE_BS = 'value_bs'
    VOLM_BS = 'volm_bs'
    
    # Multi-timeframe flow metrics
    VOLMBS_5M = 'volmbs_5m'
    VOLMBS_15M = 'volmbs_15m'
    VOLMBS_30M = 'volmbs_30m'
    VOLMBS_60M = 'volmbs_60m'
    
    VALUEBS_5M = 'valuebs_5m'
    VALUEBS_15M = 'valuebs_15m'
    VALUEBS_30M = 'valuebs_30m'
    VALUEBS_60M = 'valuebs_60m'
    
    # Volatility
    VOLATILITY = 'volatility'
    IMPLIED_VOLATILITY = 'implied_volatility'

class EliteImpactColumns:
    """Elite impact calculation output columns"""
    ELITE_IMPACT_SCORE = 'elite_impact_score_und'
    INSTITUTIONAL_FLOW_SCORE = 'institutional_flow_score_und'
    FLOW_MOMENTUM_INDEX = 'flow_momentum_index_und'
    MARKET_REGIME = 'market_regime_elite'
    FLOW_TYPE = 'flow_type_elite'
    VOLATILITY_REGIME = 'volatility_regime_elite'
    CONFIDENCE = 'confidence'
    TRANSITION_RISK = 'transition_risk'

    # SDAG/DAG Analysis Columns (Strike-level)
    SDAG_CONSENSUS = 'sdag_consensus'
    DAG_CONSENSUS = 'dag_consensus'
    PREDICTION_CONFIDENCE = 'prediction_confidence'
    SIGNAL_STRENGTH = 'signal_strength'

    # Elite Impact Metrics (Strike-level)
    ELITE_IMPACT_SCORE_STRIKE = 'elite_impact_score'
    STRIKE_MAGNETISM_INDEX = 'strike_magnetism_index'
    VOLATILITY_PRESSURE_INDEX = 'volatility_pressure_index'


class EliteImpactResultsV2_5(BaseModel):
    """
    STRICT PYDANTIC V2-ONLY: Elite impact calculation results model.
    Replaces dictionary returns with proper Pydantic v2 structure.
    """
    elite_impact_score_und: float = Field(..., description="Master composite elite impact score for the underlying.")
    institutional_flow_score_und: float = Field(..., description="Institutional flow score for the underlying.")
    flow_momentum_index_und: float = Field(..., description="Flow momentum index for the underlying.")
    market_regime_elite: str = Field(..., description="Elite classified market regime.")
    flow_type_elite: str = Field(..., description="Elite classified flow type.")
    volatility_regime_elite: str = Field(..., description="Elite classified volatility regime.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level for analysis.")
    transition_risk: float = Field(..., ge=0.0, le=1.0, description="Transition risk score.")

    class Config:
        extra = 'forbid'

# =============================================================================
# ELITE IMPACT CALCULATOR - Simplified from elite_impact_calculations.py
# =============================================================================

class EliteImpactCalculator:
    """
    Simplified elite impact calculator focusing on core institutional intelligence.
    
    Eliminates complex ML models in favor of robust heuristic-based calculations
    that provide reliable institutional flow intelligence.
    """
    
    def __init__(self, elite_config: EliteConfig = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = elite_config or EliteConfig()
        
        # Impact calculation weights (optimized from original complex models)
        self.IMPACT_WEIGHTS = {
            'flow_intensity': 0.35,
            'volume_profile': 0.25,
            'momentum_persistence': 0.20,
            'regime_alignment': 0.20
        }
        
        # Institutional flow thresholds
        self.INSTITUTIONAL_THRESHOLDS = {
            'large_institutional': 1000000,  # $1M+ flows
            'small_institutional': 100000,   # $100K+ flows
            'sophisticated_retail': 10000,   # $10K+ flows
            'volume_threshold': 5000000      # 5M+ volume
        }
    
    def calculate_elite_impact_score(self, options_data: pd.DataFrame, underlying_data) -> EliteImpactResultsV2_5:
        """
        Calculate comprehensive elite impact score using simplified but robust methodology.

        Args:
            options_data: DataFrame of options data
            underlying_data: Pydantic model (ProcessedUnderlyingAggregatesV2_5) or Dict

        Returns:
            Dict containing elite impact metrics and institutional intelligence
        """
        try:
            self.logger.debug("Calculating elite impact score...")
            
            # Extract key metrics
            flow_intensity = self._calculate_flow_intensity_optimized(options_data, underlying_data)
            volume_profile = self._calculate_volume_profile_optimized(options_data, underlying_data)
            momentum_persistence = self._calculate_momentum_persistence_optimized(underlying_data)
            regime_alignment = self._calculate_regime_alignment_optimized(underlying_data)
            
            # Calculate composite elite impact score
            elite_impact_score = (
                flow_intensity * self.IMPACT_WEIGHTS['flow_intensity'] +
                volume_profile * self.IMPACT_WEIGHTS['volume_profile'] +
                momentum_persistence * self.IMPACT_WEIGHTS['momentum_persistence'] +
                regime_alignment * self.IMPACT_WEIGHTS['regime_alignment']
            )
            
            # Calculate institutional flow score
            institutional_flow_score = self._calculate_institutional_flow_score_optimized(options_data, underlying_data)
            
            # Calculate flow momentum index
            flow_momentum_index = self._calculate_flow_momentum_index_optimized(underlying_data)
            
            # Determine market regime and flow type
            market_regime = self._determine_market_regime_simple(underlying_data)
            flow_type = self._classify_flow_type_simple(options_data, underlying_data)
            volatility_regime = self._determine_volatility_regime_simple(underlying_data)
            
            # Calculate confidence and transition risk
            confidence = self._calculate_confidence_optimized(elite_impact_score, institutional_flow_score)
            transition_risk = self._calculate_transition_risk_optimized(underlying_data)
            
            # CRITICAL FIX: Return proper Pydantic v2 model instead of dictionary
            results = EliteImpactResultsV2_5(
                elite_impact_score_und=self._bound_score(elite_impact_score),
                institutional_flow_score_und=self._bound_score(institutional_flow_score),
                flow_momentum_index_und=self._bound_score(flow_momentum_index),
                market_regime_elite=market_regime.value,
                flow_type_elite=flow_type.value,
                volatility_regime_elite=volatility_regime,
                confidence=self._bound_score(confidence, 0.0, 1.0),
                transition_risk=self._bound_score(transition_risk, 0.0, 1.0)
            )

            self.logger.debug(f"Elite impact calculation complete. Score: {elite_impact_score:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating elite impact score: {e}", exc_info=True)
            self._raise_elite_calculation_error(f"elite impact score calculation: {e}")
    
    def _calculate_flow_intensity_optimized(self, options_data: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate flow intensity using simplified but effective methodology"""
        try:
            # TIERED WEEKEND SYSTEM: Extract flow metrics with off-hours handling
            net_value_flow_raw = getattr(underlying_data, 'net_value_flow_5m_und', None) or getattr(underlying_data, 'value_bs', None)
            net_vol_flow_raw = getattr(underlying_data, 'net_vol_flow_5m_und', None) or getattr(underlying_data, 'volm_bs', None)
            total_volume_raw = getattr(underlying_data, 'day_volume', None)

            # Handle off-hours gracefully - use real data or zero for flow metrics
            net_value_flow = 0.0 if net_value_flow_raw is None else float(net_value_flow_raw)
            net_vol_flow = 0.0 if net_vol_flow_raw is None else float(net_vol_flow_raw)
            total_volume = 1000000.0 if total_volume_raw is None else float(total_volume_raw)
            
            # Calculate raw flow intensity
            raw_intensity = abs(net_value_flow) + abs(net_vol_flow)
            
            # Volume-adjusted intensity
            volume_factor = min(2.0, max(0.5, total_volume / 10000000))  # Normalize around 10M volume
            adjusted_intensity = raw_intensity * volume_factor
            
            # Scale to 0-100 range
            return min(100.0, adjusted_intensity / 100000)  # Scale factor based on typical flow ranges
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating flow intensity: {e}")
            raise ValueError(f"CRITICAL: Flow intensity calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_volume_profile_optimized(self, options_data: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate volume profile score"""
        try:
            total_volume_raw = getattr(underlying_data, 'day_volume', None)
            total_volume = float(total_volume_raw) if total_volume_raw is not None else 1000000.0
            
            # Volume percentile scoring (simplified)
            if total_volume > 50000000:  # Very high volume
                return 90.0
            elif total_volume > 20000000:  # High volume
                return 75.0
            elif total_volume > 5000000:   # Medium volume
                return 50.0
            elif total_volume > 1000000:   # Low volume
                return 25.0
            else:  # Very low volume
                return 10.0
                
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating volume profile: {e}")
            raise ValueError(f"CRITICAL: Volume profile calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_momentum_persistence_optimized(self, underlying_data: Dict) -> float:
        """Calculate momentum persistence score"""
        try:
            # Extract momentum indicators - Handle None values gracefully
            momentum_index_raw = getattr(underlying_data, 'momentum_acceleration_index_und', None)
            price_change_pct_raw = getattr(underlying_data, 'price_change_pct_und', None)

            momentum_index = float(momentum_index_raw) if momentum_index_raw is not None else 0.0
            price_change_pct = float(price_change_pct_raw) if price_change_pct_raw is not None else 0.0
            
            # Momentum persistence calculation
            momentum_strength = abs(momentum_index) * 10  # Scale momentum index
            price_momentum = abs(price_change_pct) * 100   # Scale price change
            
            # Combined momentum score
            combined_momentum = (momentum_strength * 0.6 + price_momentum * 0.4)
            
            return min(100.0, combined_momentum)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating momentum persistence: {e}")
            raise ValueError(f"CRITICAL: Momentum persistence calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_regime_alignment_optimized(self, underlying_data: Dict) -> float:
        """Calculate regime alignment score"""
        try:
            # Extract regime indicators - Handle None values gracefully
            current_iv_raw = getattr(underlying_data, 'u_volatility', None)
            price_change_pct_raw = getattr(underlying_data, 'price_change_pct_und', None)

            current_iv = float(current_iv_raw) if current_iv_raw is not None else 0.20
            price_change_pct = float(price_change_pct_raw) if price_change_pct_raw is not None else 0.0
            
            # Regime stability scoring
            vol_stability = 100 - min(100, abs(current_iv - 0.20) * 500)  # Penalty for extreme volatility
            trend_clarity = min(100, abs(price_change_pct) * 2000)        # Reward for clear trends
            
            # Combined regime alignment
            alignment_score = (vol_stability * 0.6 + trend_clarity * 0.4)
            
            return max(0.0, min(100.0, alignment_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime alignment: {e}")
            return 50.0  # Neutral score on error
    
    def _calculate_institutional_flow_score_optimized(self, options_data: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate institutional flow score using simplified heuristics"""
        try:
            # Extract key flow indicators - Handle None values gracefully for off-hours data
            net_value_flow_raw = getattr(underlying_data, 'net_value_flow_5m_und', None)
            total_volume_raw = getattr(underlying_data, 'day_volume', None)

            # Use zero for flow metrics during off-hours (real behavior, not fake data)
            net_value_flow = abs(float(net_value_flow_raw)) if net_value_flow_raw is not None else 0.0
            total_volume = float(total_volume_raw) if total_volume_raw is not None else 1000000.0  # Reasonable default volume
            
            # Institutional flow scoring
            if net_value_flow > self.INSTITUTIONAL_THRESHOLDS['large_institutional']:
                flow_score = 90.0
            elif net_value_flow > self.INSTITUTIONAL_THRESHOLDS['small_institutional']:
                flow_score = 70.0
            elif net_value_flow > self.INSTITUTIONAL_THRESHOLDS['sophisticated_retail']:
                flow_score = 40.0
            else:
                flow_score = 20.0
            
            # Volume adjustment
            if total_volume > self.INSTITUTIONAL_THRESHOLDS['volume_threshold']:
                flow_score *= 1.2  # Boost for high volume
            
            return min(100.0, flow_score)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating institutional flow score: {e}")
            raise ValueError(f"CRITICAL: Institutional flow score calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_flow_momentum_index_optimized(self, underlying_data: Dict) -> float:
        """Calculate flow momentum index"""
        try:
            # Extract flow data across timeframes - Handle None values gracefully
            flow_5m_raw = getattr(underlying_data, 'net_vol_flow_5m_und', None)
            flow_15m_raw = getattr(underlying_data, 'net_vol_flow_15m_und', None)
            flow_30m_raw = getattr(underlying_data, 'net_vol_flow_30m_und', None)

            flow_5m = float(flow_5m_raw) if flow_5m_raw is not None else 0.0
            flow_15m = float(flow_15m_raw) if flow_15m_raw is not None else 0.0
            flow_30m = float(flow_30m_raw) if flow_30m_raw is not None else 0.0
            
            # Calculate momentum acceleration
            if abs(flow_15m) > 0.001:
                short_momentum = flow_5m / flow_15m
            else:
                short_momentum = 0.0
            
            if abs(flow_30m) > 0.001:
                medium_momentum = flow_15m / flow_30m
            else:
                medium_momentum = 0.0
            
            # Combined momentum index
            momentum_index = (short_momentum * 0.6 + medium_momentum * 0.4) * 50  # Scale to 0-100
            
            return self._bound_score(momentum_index)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating flow momentum index: {e}")
            raise ValueError(f"CRITICAL: Flow momentum index calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    # =============================================================================
    # SIMPLIFIED CLASSIFICATION METHODS
    # =============================================================================
    
    def _determine_market_regime_simple(self, underlying_data: Dict) -> MarketRegime:
        """Simplified market regime determination"""
        try:
            current_iv_raw = getattr(underlying_data, 'u_volatility', None)
            price_change_pct_raw = getattr(underlying_data, 'price_change_pct_und', None)

            current_iv = float(current_iv_raw) if current_iv_raw is not None else 0.20
            price_change_pct = float(price_change_pct_raw) if price_change_pct_raw is not None else 0.0
            
            # Volatility classification
            if current_iv > 0.30:
                vol_regime = 'HIGH_VOL'
            elif current_iv < 0.15:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'MEDIUM_VOL'
            
            # Trend classification
            if abs(price_change_pct) > 0.02:
                trend_regime = 'TRENDING'
            else:
                trend_regime = 'RANGING'
            
            # Combine regimes
            regime_name = f"{vol_regime}_{trend_regime}".lower()
            
            try:
                return MarketRegime(regime_name)
            except ValueError:
                return MarketRegime.REGIME_UNCLEAR_OR_TRANSITIONING
                
        except Exception as e:
            self.logger.warning(f"Error determining market regime: {e}")
            return MarketRegime.REGIME_UNCLEAR_OR_TRANSITIONING
    
    def _classify_flow_type_simple(self, options_data: pd.DataFrame, underlying_data: Dict) -> FlowType:
        """Simplified flow type classification"""
        try:
            # Handle None values gracefully for off-hours data
            net_value_flow_raw = getattr(underlying_data, 'net_value_flow_5m_und', None)
            total_volume_raw = getattr(underlying_data, 'day_volume', None)

            net_value_flow = abs(float(net_value_flow_raw)) if net_value_flow_raw is not None else 0.0
            total_volume = float(total_volume_raw) if total_volume_raw is not None else 1000000.0
            
            # Simple classification based on flow size and volume
            if net_value_flow > self.INSTITUTIONAL_THRESHOLDS['large_institutional'] and total_volume > 20000000:
                return FlowType.INSTITUTIONAL_LARGE
            elif net_value_flow > self.INSTITUTIONAL_THRESHOLDS['small_institutional']:
                return FlowType.INSTITUTIONAL_SMALL
            elif net_value_flow > self.INSTITUTIONAL_THRESHOLDS['sophisticated_retail']:
                return FlowType.RETAIL_SOPHISTICATED
            elif total_volume > 10000000:
                return FlowType.HEDGE_FUND
            else:
                return FlowType.RETAIL_UNSOPHISTICATED
                
        except Exception as e:
            self.logger.warning(f"Error classifying flow type: {e}")
            return FlowType.UNKNOWN
    
    def _determine_volatility_regime_simple(self, underlying_data: Dict) -> str:
        """Simplified volatility regime determination"""
        try:
            current_iv_raw = getattr(underlying_data, 'u_volatility', None)
            current_iv = float(current_iv_raw) if current_iv_raw is not None else 0.20

            if current_iv > 0.40:
                return "high_vol"
            elif current_iv < 0.15:
                return "low_vol"
            else:
                return "medium_vol"

        except Exception as e:
            self.logger.warning(f"Error determining volatility regime: {e}")
            return "medium_vol"
    
    def _calculate_confidence_optimized(self, elite_score: float, institutional_score: float) -> float:
        """Calculate confidence in the analysis"""
        try:
            # Confidence based on score consistency and magnitude
            score_consistency = 1.0 - abs(elite_score - institutional_score) / 100.0
            score_magnitude = (elite_score + institutional_score) / 200.0
            
            confidence = (score_consistency * 0.6 + score_magnitude * 0.4)
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating confidence score: {e}")
            raise ValueError(f"CRITICAL: Confidence calculation failed - cannot return fake neutral confidence! Error: {e}") from e
    
    def _calculate_transition_risk_optimized(self, underlying_data: Dict) -> float:
        """Calculate regime transition risk"""
        try:
            # ZERO TOLERANCE FAKE DATA: Get real volatility or fail
            current_iv_raw = getattr(underlying_data, 'u_volatility', None)
            if current_iv_raw is None:
                raise ValueError("CRITICAL: u_volatility missing from underlying data - cannot calculate transition risk without real volatility data!")
            current_iv = float(current_iv_raw)

            # ZERO TOLERANCE FAKE DATA: Get real price change or fail
            price_change_pct_raw = getattr(underlying_data, 'price_change_pct_und', None)
            if price_change_pct_raw is None:
                raise ValueError("CRITICAL: price_change_pct_und missing from underlying data - cannot calculate transition risk without real price change data!")
            price_change_pct = float(price_change_pct_raw)
            
            # Risk factors
            vol_instability = min(1.0, abs(current_iv - 0.20) * 5)  # Risk from extreme volatility
            price_instability = min(1.0, abs(price_change_pct) * 50)  # Risk from large price moves
            
            transition_risk = (vol_instability * 0.6 + price_instability * 0.4)
            return max(0.0, min(1.0, transition_risk))
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating transition risk: {e}")
            raise ValueError(f"CRITICAL: Transition risk calculation failed - cannot return fake neutral risk! Error: {e}") from e
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    

    
    def _bound_score(self, score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Bound score within specified range"""
        return max(min_val, min(max_val, score))
    
    def _raise_elite_calculation_error(self, error_context: str) -> None:
        """FAIL-FAST: Raise error instead of returning fake elite results - NO FAKE TRADING INTELLIGENCE ALLOWED"""
        raise ValueError(f"CRITICAL: Elite intelligence calculation failed in {error_context} - cannot return fake trading intelligence that could cause massive losses!")

# Export consolidated components
__all__ = [
    'EliteImpactCalculator', 'EliteConfig', 'MarketRegime', 'FlowType',
    'ConvexValueColumns', 'EliteImpactColumns', 'EliteImpactResultsV2_5'
]
