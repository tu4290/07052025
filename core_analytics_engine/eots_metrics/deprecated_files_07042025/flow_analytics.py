# core_analytics_engine/eots_metrics/flow_analytics.py

"""
EOTS Flow Analytics - Consolidated Flow Calculations

Consolidates:
- enhanced_flow_metrics.py: Tier 3 flow metrics (VAPI-FA, DWFD, TW-LAF)
- elite_flow_classifier.py: Institutional flow classification
- elite_momentum_detector.py: Momentum and acceleration detection

Optimizations:
- Unified flow classification logic
- Streamlined momentum calculations
- Integrated caching strategy
- Eliminated duplicate flow processing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from core_analytics_engine.eots_metrics.core_calculator import CoreCalculator
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class FlowType(Enum):
    """Consolidated flow classification types"""
    RETAIL_UNSOPHISTICATED = "retail_unsophisticated"
    RETAIL_SOPHISTICATED = "retail_sophisticated"
    INSTITUTIONAL_SMALL = "institutional_small"
    INSTITUTIONAL_LARGE = "institutional_large"
    HEDGE_FUND = "hedge_fund"
    MARKET_MAKER = "market_maker"
    UNKNOWN = "unknown"

class FlowAnalytics(CoreCalculator):
    """
    Consolidated flow analytics calculator.
    
    Combines functionality from:
    - EnhancedFlowMetricsCalculator: VAPI-FA, DWFD, TW-LAF calculations
    - EliteFlowClassifier: Institutional flow classification
    - EliteMomentumDetector: Momentum and acceleration detection
    """
    
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: Any = None):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config = elite_config or {}
        
        # Initialize flow classification components
        self.flow_scaler = StandardScaler()
        self.flow_model = None  # Would be loaded from pre-trained model in production
        self.is_flow_model_trained = False
        
        # Momentum detection cache
        self.momentum_cache = {}
        
        # Flow classification thresholds (optimized from original modules)
        self.INSTITUTIONAL_THRESHOLD_PERCENTILE = 95.0
        self.FLOW_MOMENTUM_PERIODS = [5, 15, 30, 60]
        self.ACCELERATION_THRESHOLD_MULTIPLIER = 2.0
    
    # =============================================================================
    # ENHANCED FLOW METRICS (Tier 3) - Consolidated from enhanced_flow_metrics.py
from data_models import ProcessedUnderlyingAggregatesV2_5 # Import the specific model

# =============================================================================
    
    def calculate_all_enhanced_flow_metrics(self, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> ProcessedUnderlyingAggregatesV2_5:
        """
        Orchestrates calculation of all enhanced flow metrics using proper Pydantic v2 architecture.

        Calculates:
        - VAPI-FA (Volume-Adjusted Premium Intensity with Flow Acceleration)
        - DWFD (Delta-Weighted Flow Divergence)
        - TW-LAF (Time-Weighted Liquidity-Adjusted Flow)
        - Flow Type Classification
        - Momentum Acceleration Index

        Args:
            und_data: Pydantic model (FoundationalMetricsOutput or ProcessedUnderlyingAggregatesV2_5)
            symbol: Symbol to calculate metrics for

        Returns:
            Updated Pydantic model instance with calculated flow metrics
        """
        self.logger.debug(f"Calculating enhanced flow metrics for {symbol}...")

        try:
            # Update calculation state
            self._calculation_state.update_state(current_symbol=symbol)

            # Classify flow type first (affects other calculations)
            flow_type = self._classify_flow_type_optimized(und_data)

            # Calculate momentum acceleration index
            momentum_index = self._calculate_momentum_acceleration_index_optimized(und_data)

            # Calculate enhanced flow metrics using proper Pydantic access
            vapi_fa_raw, vapi_fa_z_score, vapi_fa_pvr_5m, vapi_fa_flow_accel_5m = self._calculate_vapi_fa_optimized(und_data, symbol)
            dwfd_raw, dwfd_z_score, dwfd_fvd = self._calculate_dwfd_optimized(und_data, symbol)
            tw_laf_raw, tw_laf_z_score, tw_laf_liquidity_factor_5m, tw_laf_time_weighted_sum = self._calculate_tw_laf_optimized(und_data, symbol)

            self.logger.debug(f"Enhanced flow metrics calculation complete for {symbol}.")

            # STRICT PYDANTIC V2-ONLY: Update model fields directly, no dictionaries
            und_data.flow_type_elite = flow_type.value
            und_data.momentum_acceleration_index_und = momentum_index
            und_data.vapi_fa_raw_und = vapi_fa_raw
            und_data.vapi_fa_z_score_und = vapi_fa_z_score
            und_data.vapi_fa_pvr_5m_und = vapi_fa_pvr_5m
            und_data.vapi_fa_flow_accel_5m_und = vapi_fa_flow_accel_5m
            und_data.dwfd_raw_und = dwfd_raw
            und_data.dwfd_z_score_und = dwfd_z_score
            und_data.dwfd_fvd_und = dwfd_fvd
            und_data.tw_laf_raw_und = tw_laf_raw
            und_data.tw_laf_z_score_und = tw_laf_z_score
            und_data.tw_laf_liquidity_factor_5m_und = tw_laf_liquidity_factor_5m
            und_data.tw_laf_time_weighted_sum_und = tw_laf_time_weighted_sum

            return und_data

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating enhanced flow metrics for {symbol}: {e}", exc_info=True)
            self._raise_flow_calculation_error(f"enhanced flow metrics for {symbol}: {e}")
            return und_data # Return the original model on error, as per type hint
    
    def _calculate_vapi_fa_optimized(self, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> Tuple[float, float, float, float]:
        """Optimized VAPI-FA calculation with proper Pydantic access"""
        try:
            self.logger.debug(f"VAPI-FA calculation for {symbol}")

            # TIERED WEEKEND SYSTEM: Handle live vs off-hours data gracefully
            # Try live rolling data first, then fall back to snapshot data from ConvexValue
            net_value_flow_5m_raw = (
                getattr(und_data, 'net_value_flow_5m_und', None) or  # Live intraday data
                getattr(und_data, 'total_nvp', None) or              # Alternative live data
                getattr(und_data, 'value_bs', None)                  # ConvexValue snapshot data
            )

            net_vol_flow_5m_raw = (
                getattr(und_data, 'net_vol_flow_5m_und', None) or    # Live intraday data
                getattr(und_data, 'total_nvp_vol', None) or          # Alternative live data
                getattr(und_data, 'volm_bs', None)                   # ConvexValue snapshot data
            )

            # WEEKEND/OFF-HOURS HANDLING: Use snapshot data or return zero for flow metrics
            if net_value_flow_5m_raw is None:
                self.logger.warning(f"⚠️ No flow data available for {symbol} (off-hours) - using zero flow")
                net_value_flow_5m = 0.0
            else:
                net_value_flow_5m = float(net_value_flow_5m_raw)

            if net_vol_flow_5m_raw is None:
                self.logger.warning(f"⚠️ No volume flow data available for {symbol} (off-hours) - using zero flow")
                net_vol_flow_5m = 0.0
            else:
                net_vol_flow_5m = float(net_vol_flow_5m_raw)

            # Calculate 15m flow from 5m data if not available
            net_vol_flow_15m_raw = getattr(und_data, 'net_vol_flow_15m_und', None)
            if net_vol_flow_15m_raw is None:
                net_vol_flow_15m_raw = net_vol_flow_5m * 2.8
            net_vol_flow_15m = float(net_vol_flow_15m_raw)

            current_iv_raw = (
                getattr(und_data, 'u_volatility', None) or
                getattr(und_data, 'Current_Underlying_IV', None) or
                getattr(und_data, 'implied_volatility', None)
            )
            if current_iv_raw is None:
                raise ValueError(f"CRITICAL: No volatility data available for {symbol} - cannot calculate VAPI-FA without real IV data!")
            current_iv = float(current_iv_raw)
            
            # Calculate Price-to-Volume Ratio (PVR)
            pvr_5m = net_value_flow_5m / max(abs(net_vol_flow_5m), 0.001)
            
            # Volatility adjustment
            volatility_adjusted_pvr_5m = pvr_5m * current_iv
            
            # Flow acceleration calculation (optimized)
            flow_in_prior_5_to_10_min = (net_vol_flow_15m - net_vol_flow_5m) / 2.0
            flow_acceleration_5m = net_vol_flow_5m - flow_in_prior_5_to_10_min
            
            # VAPI-FA raw calculation
            vapi_fa_raw = volatility_adjusted_pvr_5m * flow_acceleration_5m
            
            # Unified caching and normalization
            vapi_fa_cache = self._add_to_intraday_cache(symbol, 'vapi_fa', vapi_fa_raw, max_size=200)
            vapi_fa_z_score = self._calculate_z_score_optimized(vapi_fa_cache, vapi_fa_raw)

            self.logger.debug(f"VAPI-FA results for {symbol}: raw={vapi_fa_raw:.2f}, z_score={vapi_fa_z_score:.2f}")

            # Return calculated metrics as tuple
            return (vapi_fa_raw, vapi_fa_z_score, pvr_5m, flow_acceleration_5m)

        except Exception as e:
            self.logger.error(f"Error calculating VAPI-FA for {symbol}: {e}", exc_info=True)
            return (0.0, 0.0, 0.0, 0.0) # Keep tuple return for this internal helper
    
    def _calculate_dwfd_optimized(self, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> Tuple[float, float, float]:
        """Optimized DWFD calculation with proper Pydantic access"""
        try:
            self.logger.debug(f"DWFD calculation for {symbol}")

            # Extract delta-weighted flows using proper Pydantic attribute access
            net_delta_flow_raw = getattr(und_data, 'net_cust_delta_flow_und', None)
            net_value_flow_raw = getattr(und_data, 'net_value_flow_5m_und', None) or getattr(und_data, 'value_bs', None)

            if net_delta_flow_raw is None:
                raise ValueError(f"CRITICAL: net_cust_delta_flow_und is None for {symbol}")
            if net_value_flow_raw is None:
                raise ValueError(f"CRITICAL: No value flow data available for {symbol}")

            net_delta_flow = float(net_delta_flow_raw)
            net_value_flow = float(net_value_flow_raw)

            # Calculate flow divergence
            if abs(net_delta_flow) > 0.001:
                flow_divergence = net_value_flow / net_delta_flow
            else:
                flow_divergence = 0.0

            # DWFD raw calculation
            dwfd_raw = flow_divergence * abs(net_delta_flow) ** 0.5

            # Unified caching and normalization
            dwfd_cache = self._add_to_intraday_cache(symbol, 'dwfd', dwfd_raw, max_size=200)
            dwfd_z_score = self._calculate_z_score_optimized(dwfd_cache, dwfd_raw)

            self.logger.debug(f"DWFD results for {symbol}: raw={dwfd_raw:.2f}, z_score={dwfd_z_score:.2f}")

            # Return calculated metrics as tuple
            return (dwfd_raw, dwfd_z_score, flow_divergence)

        except Exception as e:
            self.logger.error(f"Error calculating DWFD for {symbol}: {e}", exc_info=True)
            return (0.0, 0.0, 0.0) # Keep tuple return for this internal helper
    
    def _calculate_tw_laf_optimized(self, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> Tuple[float, float, float, float]:
        """Optimized TW-LAF calculation with proper Pydantic access"""
        try:
            self.logger.debug(f"TW-LAF calculation for {symbol}")

            # TIERED WEEKEND SYSTEM: Extract time-weighted flows with off-hours handling
            net_vol_flow_5m_raw = getattr(und_data, 'net_vol_flow_5m_und', None) or getattr(und_data, 'volm_bs', None)
            net_vol_flow_15m_raw = getattr(und_data, 'net_vol_flow_15m_und', None)
            net_vol_flow_30m_raw = getattr(und_data, 'net_vol_flow_30m_und', None)

            # Handle off-hours gracefully
            net_vol_flow_5m = 0.0 if net_vol_flow_5m_raw is None else float(net_vol_flow_5m_raw)
            net_vol_flow_15m = 0.0 if net_vol_flow_15m_raw is None else float(net_vol_flow_15m_raw)
            net_vol_flow_30m = 0.0 if net_vol_flow_30m_raw is None else float(net_vol_flow_30m_raw)

            # Calculate liquidity factor (simplified)
            total_volume_raw = getattr(und_data, 'day_volume', None)
            if total_volume_raw is None:
                raise ValueError(f"CRITICAL: day_volume missing for {symbol} - cannot calculate TW-LAF without real volume data!")
            total_volume = float(total_volume_raw)
            liquidity_factor = min(2.0, max(0.5, total_volume / 1000000))

            # Time-weighted sum calculation
            time_weighted_sum = (net_vol_flow_5m * 0.5 +
                               net_vol_flow_15m * 0.3 +
                               net_vol_flow_30m * 0.2)

            # TW-LAF raw calculation
            tw_laf_raw = time_weighted_sum * liquidity_factor

            # Unified caching and normalization
            tw_laf_cache = self._add_to_intraday_cache(symbol, 'tw_laf', tw_laf_raw, max_size=200)
            tw_laf_z_score = self._calculate_z_score_optimized(tw_laf_cache, tw_laf_raw)

            self.logger.debug(f"TW-LAF results for {symbol}: raw={tw_laf_raw:.2f}, z_score={tw_laf_z_score:.2f}")

            # Return calculated metrics as tuple
            return (tw_laf_raw, tw_laf_z_score, liquidity_factor, time_weighted_sum)

        except Exception as e:
            self.logger.error(f"Error calculating TW-LAF for {symbol}: {e}", exc_info=True)
            return (0.0, 0.0, 0.0, 0.0) # Keep tuple return for this internal helper
    
    # =============================================================================
    # FLOW CLASSIFICATION - Consolidated from elite_flow_classifier.py
    # =============================================================================
    
    def _classify_flow_type_optimized(self, und_data: ProcessedUnderlyingAggregatesV2_5) -> FlowType:
        """Optimized flow classification using simplified heuristics with proper Pydantic access"""
        try:
            # TIERED WEEKEND SYSTEM: Extract key flow indicators with off-hours handling
            total_volume_raw = getattr(und_data, 'day_volume', None)
            net_value_flow_raw = getattr(und_data, 'net_value_flow_5m_und', None) or getattr(und_data, 'value_bs', None)
            net_vol_flow_raw = getattr(und_data, 'net_vol_flow_5m_und', None) or getattr(und_data, 'volm_bs', None)

            # Handle off-hours gracefully - use real data or zero for flow metrics
            total_volume = 1000000.0 if total_volume_raw is None else float(total_volume_raw)
            net_value_flow = 0.0 if net_value_flow_raw is None else float(net_value_flow_raw)
            net_vol_flow = 0.0 if net_vol_flow_raw is None else float(net_vol_flow_raw)

            # Calculate flow intensity
            flow_intensity = abs(net_value_flow) + abs(net_vol_flow)

            # Volume-based classification (simplified from ML approach)
            if total_volume > 10000000:  # High volume threshold
                if flow_intensity > 1000000:
                    return FlowType.INSTITUTIONAL_LARGE
                elif flow_intensity > 100000:
                    return FlowType.INSTITUTIONAL_SMALL
                else:
                    return FlowType.HEDGE_FUND
            elif total_volume > 1000000:  # Medium volume threshold
                if flow_intensity > 50000:
                    return FlowType.RETAIL_SOPHISTICATED
                else:
                    return FlowType.RETAIL_UNSOPHISTICATED
            else:
                return FlowType.RETAIL_UNSOPHISTICATED

        except Exception as e:
            self.logger.warning(f"Error classifying flow type: {e}")
            return FlowType.UNKNOWN
    
    # =============================================================================
    # MOMENTUM DETECTION - Consolidated from elite_momentum_detector.py
    # =============================================================================
    
    def _calculate_momentum_acceleration_index_optimized(self, und_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Optimized momentum acceleration calculation with proper Pydantic access"""
        try:
            # TIERED WEEKEND SYSTEM: Extract flow series data with off-hours handling
            net_vol_flow_5m_raw = getattr(und_data, 'net_vol_flow_5m_und', None) or getattr(und_data, 'volm_bs', None)
            net_vol_flow_15m_raw = getattr(und_data, 'net_vol_flow_15m_und', None)

            # Handle off-hours gracefully - use real data or zero for flow metrics
            net_vol_flow_5m = 0.0 if net_vol_flow_5m_raw is None else float(net_vol_flow_5m_raw)
            net_vol_flow_15m = 0.0 if net_vol_flow_15m_raw is None else float(net_vol_flow_15m_raw)

            # TIERED WEEKEND SYSTEM: Handle 30m flow with off-hours handling
            net_vol_flow_30m_raw = getattr(und_data, 'net_vol_flow_30m_und', None)
            net_vol_flow_30m = 0.0 if net_vol_flow_30m_raw is None else float(net_vol_flow_30m_raw)

            # Create synthetic flow series for momentum calculation
            flow_series = [net_vol_flow_30m, net_vol_flow_15m, net_vol_flow_5m]

            # Calculate velocity (rate of change)
            if len(flow_series) >= 2:
                velocity = flow_series[-1] - flow_series[-2]
            else:
                velocity = 0.0

            # Calculate acceleration (rate of change of velocity)
            if len(flow_series) >= 3:
                prev_velocity = flow_series[-2] - flow_series[-3]
                acceleration = velocity - prev_velocity
            else:
                acceleration = 0.0

            # Momentum acceleration index
            momentum_index = (abs(velocity) * 0.6 + abs(acceleration) * 0.4) / max(abs(net_vol_flow_5m), 1.0)

            return self._bound_value(momentum_index, -10.0, 10.0)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating momentum acceleration index: {e}")
            raise ValueError(f"CRITICAL: Momentum acceleration index calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    # =============================================================================
    # OPTIMIZED UTILITIES
    # =============================================================================
    
    def _calculate_z_score_optimized(self, cache_data: List[float], current_value: float) -> float:
        """FAIL-FAST: Z-score calculation - NO FAKE DEFAULTS ALLOWED"""
        if not cache_data or len(cache_data) < 2:
            raise ValueError(f"CRITICAL: Insufficient cache data for Z-score calculation - need at least 2 data points, got {len(cache_data) if cache_data else 0}!")
        
        try:
            if len(cache_data) >= 10:
                # Use z-score for larger datasets
                mean_val = np.mean(cache_data)
                std_val = np.std(cache_data)
                return (current_value - mean_val) / max(std_val, 0.001)
            else:
                # Use percentile gauge for smaller datasets
                return self._calculate_percentile_gauge_value(cache_data, current_value)
                
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating z-score: {e}")
            raise ValueError(f"CRITICAL: Z-score calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _raise_flow_calculation_error(self, error_context: str) -> None:
        """FAIL-FAST: Raise error instead of returning fake flow metrics - NO FAKE TRADING DATA ALLOWED"""
        raise ValueError(f"CRITICAL: Flow analytics calculation failed in {error_context} - cannot return fake flow metrics that could cause massive trading losses!")

    # ELIMINATED DANGEROUS FAKE DATA FUNCTIONS:
    # - _get_default_flow_metrics_dict() - returned fake flow metrics dictionary
    # - _get_default_flow_metrics_model() - returned fake flow metrics in Pydantic model
    # These functions were EXTREMELY DANGEROUS as they created fake trading data that could cause massive losses.
    # All flow calculations now FAIL FAST with clear error messages when real data is unavailable.

# Export the consolidated calculator
__all__ = ['FlowAnalytics', 'FlowType']
