# core_analytics_engine/eots_metrics/flow_analytics.py

"""
EOTS Flow Analytics - Consolidated Flow Calculations

Consolidates:
- enhanced_flow_metrics.py: Tier 3 flow metrics (VAPI-FA, DWFD, TW-LAF)
- elite_flow_classifier.py: Institutional flow classification
- elite_momentum_detector.py: Momentum and acceleration detection

Optimizations:
- Unified flow classification logic with Pydantic-first models
- Streamlined momentum calculations with fail-fast error handling
- Integrated caching strategy for performance
- Eliminated duplicate flow processing and dictionary-based logic
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from core_analytics_engine.eots_metrics.core_calculator import CoreCalculator
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_models import ProcessedUnderlyingAggregatesV2_5

# === Adaptive-Parameter-Tuning (P2) =====================
# NOTE: Added by AI assistant per **canon directive**.
# It enables dynamic (non-hard-coded) thresholds.  This first
# patch wires the tuner in; later patches will add outcome
# recording & optimisation cycles.
try:
    from huihui_integration.learning.adaptive_parameter_tuning import (
        AdaptiveParameterTuner,
        ExpertParameterSet,
        ParameterConfig,
    )
    _ADAPTIVE_TUNER_AVAILABLE = True
except ImportError:  # graceful degradation
    _ADAPTIVE_TUNER_AVAILABLE = False

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

# =============================================================================
# PYDANTIC-FIRST INPUT/OUTPUT MODELS
# =============================================================================

class FlowAnalyticsInput(BaseModel):
    """
    Strictly typed input model for flow analytics calculations.
    Ensures that only required data is passed and validated.
    
    NOTE: This model is being expanded to include richer data for elite enhancements.
    """
    # Required for all calculations
    day_volume: Optional[float]
    net_cust_delta_flow_und: Optional[float]
    u_volatility: Optional[float]

    # Required for VAPI-FA and TW-LAF with tiered fallback logic
    net_value_flow_5m_und: Optional[float]
    net_vol_flow_5m_und: Optional[float]
    net_vol_flow_15m_und: Optional[float]
    net_vol_flow_30m_und: Optional[float]
    net_vol_flow_60m_und: Optional[float] # Added for multi-timeframe consistency
    value_bs: Optional[float]  # Fallback for net_value_flow
    volm_bs: Optional[float]   # Fallback for net_vol_flow

    # --- AI ENHANCEMENT FIELDS ---
    # The following fields are used by the AI's enhanced institutional detection logic.
    # They are optional to ensure backward compatibility and graceful degradation.
    gammas_call_buy: Optional[float] = None
    gammas_call_sell: Optional[float] = None
    gammas_put_buy: Optional[float] = None
    gammas_put_sell: Optional[float] = None
    value_call_buy: Optional[float] = None
    value_call_sell: Optional[float] = None
    value_put_buy: Optional[float] = None
    value_put_sell: Optional[float] = None
    vflowratio: Optional[float] = None

    model_config = ConfigDict(extra='ignore') # Ignore extra fields from the source model

class FlowAnalyticsOutput(BaseModel):
    """
    Strictly typed output model for flow analytics results.
    Ensures a consistent and predictable data contract.
    """
    flow_type_elite: FlowType
    flow_momentum_index_und: float
    vapi_fa_raw_und: float
    vapi_fa_z_score_und: float
    vapi_fa_pvr_5m_und: float
    vapi_fa_flow_accel_5m_und: float
    dwfd_raw_und: float
    dwfd_z_score_und: float
    dwfd_fvd_und: float
    tw_laf_raw_und: float
    tw_laf_z_score_und: float
    tw_laf_liquidity_factor_5m_und: float
    tw_laf_time_weighted_sum_und: float

    model_config = ConfigDict(extra='forbid')

# =============================================================================
# FLOW ANALYTICS CALCULATOR
# =============================================================================

from data_models.elite_config_models import EliteConfig

class FlowAnalytics(CoreCalculator):
    """
    Consolidated flow analytics calculator.

    Combines functionality from:
    - EnhancedFlowMetricsCalculator: VAPI-FA, DWFD, TW-LAF calculations
    - EliteFlowClassifier: Institutional flow classification
    - EliteMomentumDetector: Momentum and acceleration detection
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: Any = None):
        # Validate required parameters
        if config_manager is None:
            raise ValueError("config_manager is required and cannot be None")
        if historical_data_manager is None:
            raise ValueError("historical_data_manager is required and cannot be None")
        if not isinstance(enhanced_cache_manager, EnhancedCacheManagerV2_5):
            raise TypeError(f"enhanced_cache_manager must be an instance of EnhancedCacheManagerV2_5, got {type(enhanced_cache_manager)}")

        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Convert elite_config to Pydantic model if not already
        if elite_config is None:
            self.elite_config = EliteConfig()
        elif isinstance(elite_config, EliteConfig):
            self.elite_config = elite_config
        else:
            self.elite_config = EliteConfig.model_validate(elite_config)

        # Momentum detection cache
        self.momentum_cache: Dict[str, Any] = {}

        # Flow classification thresholds (were hard-coded).
        # They are now exposed as adaptive parameters with safe defaults.
        self._default_params = {
            "high_volume_threshold": 10_000_000.0,
            "medium_volume_threshold": 1_000_000.0,
            "institutional_flow_intensity": 1_000_000.0,
            "sophisticated_flow_intensity": 50_000.0,
        }

        # One-time registration of parameters with the global tuner.
        if _ADAPTIVE_TUNER_AVAILABLE:
            self._init_param_tuner_once()
    
    def calculate_all_enhanced_flow_metrics(
        self, 
        und_data: ProcessedUnderlyingAggregatesV2_5, 
        symbol: str
    ) -> FlowAnalyticsOutput:
        """
        Orchestrates calculation of all enhanced flow metrics using a strict Pydantic v2 architecture.

        Calculates:
        - VAPI-FA (Volume-Adjusted Premium Intensity with Flow Acceleration)
        - DWFD (Delta-Weighted Flow Divergence)
        - TW-LAF (Time-Weighted Liquidity-Adjusted Flow)
        - Flow Type Classification
        - Flow Momentum Index

        Args:
            und_data: A complete `ProcessedUnderlyingAggregatesV2_5` Pydantic model.
            symbol: The symbol to calculate metrics for.

        Returns:
            A `FlowAnalyticsOutput` Pydantic model containing all calculated flow metrics.
        """
        self.logger.debug(f"Calculating enhanced flow metrics for {symbol}...")

        try:
            # Update calculation state
            self._calculation_state.update_state(current_symbol=symbol)

            # Create a validated input object from the main data bundle.
            # This ensures this component only uses the data it explicitly needs.
            input_data = FlowAnalyticsInput.model_validate(und_data)

            # Classify flow type first (affects other calculations)
            flow_type = self._classify_flow_type_optimized(input_data)

            # Calculate momentum acceleration index
            momentum_index = self._calculate_momentum_acceleration_index_optimized(input_data)

            # Calculate enhanced flow metrics
            vapi_fa_raw, vapi_fa_z_score, vapi_fa_pvr_5m, vapi_fa_flow_accel_5m = self._calculate_vapi_fa_optimized(input_data, symbol)
            dwfd_raw, dwfd_z_score, dwfd_fvd = self._calculate_dwfd_optimized(input_data, symbol)
            tw_laf_raw, tw_laf_z_score, tw_laf_liquidity_factor_5m, tw_laf_time_weighted_sum = self._calculate_tw_laf_optimized(input_data, symbol)

            self.logger.debug(f"Enhanced flow metrics calculation complete for {symbol}.")

            # Construct and return the strictly-typed output model.
            return FlowAnalyticsOutput(
                flow_type_elite=flow_type,
                flow_momentum_index_und=momentum_index,
                vapi_fa_raw_und=vapi_fa_raw,
                vapi_fa_z_score_und=vapi_fa_z_score,
                vapi_fa_pvr_5m_und=vapi_fa_pvr_5m,
                vapi_fa_flow_accel_5m_und=vapi_fa_flow_accel_5m,
                dwfd_raw_und=dwfd_raw,
                dwfd_z_score_und=dwfd_z_score,
                dwfd_fvd_und=dwfd_fvd,
                tw_laf_raw_und=tw_laf_raw,
                tw_laf_z_score_und=tw_laf_z_score,
                tw_laf_liquidity_factor_5m_und=tw_laf_liquidity_factor_5m,
                tw_laf_time_weighted_sum_und=tw_laf_time_weighted_sum
            )

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating enhanced flow metrics for {symbol}: {e}", exc_info=True)
            self._raise_flow_calculation_error(f"enhanced flow metrics for {symbol}: {e}")
    
    # ---------------------------------------------------------------------+
    # Adaptive-parameter utility helpers
    # ---------------------------------------------------------------------+

    _PARAM_TUNER: Optional["AdaptiveParameterTuner"] = None  # type: ignore

    import asyncio

    @classmethod
    def _init_param_tuner_once(cls):
        """Register this expert's tunable parameters exactly once."""
        if cls._PARAM_TUNER is not None:
            return  # already initialised

        cls._PARAM_TUNER = AdaptiveParameterTuner()

        param_set = ExpertParameterSet(
            expert_id="flow_analytics",
            parameters=[
                ParameterConfig(
                    name="high_volume_threshold",
                    initial_value=10_000_000.0,
                    min_bound=1_000_000.0,
                    max_bound=25_000_000.0,
                    description="Volume above which we label trades as institutional."
                ),
                ParameterConfig(
                    name="medium_volume_threshold",
                    initial_value=1_000_000.0,
                    min_bound=100_000.0,
                    max_bound=5_000_000.0,
                    description="Volume above which we label trades as retail-sophisticated."
                ),
                ParameterConfig(
                    name="institutional_flow_intensity",
                    initial_value=1_000_000.0,
                    min_bound=100_000.0,
                    max_bound=5_000_000.0,
                    description="Value/volume flow intensity considered institutional."
                ),
                ParameterConfig(
                    name="sophisticated_flow_intensity",
                    initial_value=50_000.0,
                    min_bound=1_000.0,
                    max_bound=200_000.0,
                    description="Flow intensity considered retail-sophisticated."
                ),
            ],
        )
        asyncio.create_task(cls._PARAM_TUNER.register_expert(param_set))

    def _get_dyn_param(self, name: str) -> float:
        """Fetch dynamic parameter (fallback to default if tuner unavailable)."""
        if _ADAPTIVE_TUNER_AVAILABLE and self._PARAM_TUNER is not None:
            params = self._PARAM_TUNER.get_current_parameters("flow_analytics", market_regime="global")
            return params.get(name, self._default_params[name])
        return self._default_params[name]

    def _calculate_vapi_fa_optimized(self, input_data: FlowAnalyticsInput, symbol: str) -> Tuple[float, float, float, float]:
        """Optimized VAPI-FA calculation using a validated Pydantic input model."""
        try:
            self.logger.debug(f"VAPI-FA calculation for {symbol}")

            # TIERED WEEKEND SYSTEM: Handle live vs off-hours data gracefully
            net_value_flow_5m_raw = input_data.net_value_flow_5m_und or input_data.value_bs
            net_vol_flow_5m_raw = input_data.net_vol_flow_5m_und or input_data.volm_bs
            net_vol_flow_15m_raw = input_data.net_vol_flow_15m_und

            # WEEKEND/OFF-HOURS HANDLING: Use snapshot data or return zero for flow metrics
            net_value_flow_5m = 0.0 if net_value_flow_5m_raw is None else float(net_value_flow_5m_raw)
            net_vol_flow_5m = 0.0 if net_vol_flow_5m_raw is None else float(net_vol_flow_5m_raw)
            net_vol_flow_15m = net_vol_flow_5m * 2.8 if net_vol_flow_15m_raw is None else float(net_vol_flow_15m_raw)

            current_iv_raw = input_data.u_volatility
            if current_iv_raw is None:
                self._raise_flow_calculation_error(f"No volatility data available for {symbol} - cannot calculate VAPI-FA without real IV data!")
            current_iv = float(current_iv_raw)
            
            # Calculate Price-to-Volume Ratio (PVR)
            pvr_5m = net_value_flow_5m / max(abs(net_vol_flow_5m), EPSILON)
            
            # Volatility adjustment
            volatility_adjusted_pvr_5m = pvr_5m * current_iv
            
            # Flow acceleration calculation (optimized)
            flow_in_prior_5_to_10_min = (net_vol_flow_15m - net_vol_flow_5m) / 2.0
            flow_acceleration_5m = net_vol_flow_5m - flow_in_prior_5_to_10_min
            
            # VAPI-FA raw calculation
            vapi_fa_raw = volatility_adjusted_pvr_5m * flow_acceleration_5m
            
            # --- AI ENHANCEMENT (START) ---
            # NOTE: This enhancement was added by the AI assistant and may not fully adhere to the system guide.
            # It uses multi-timeframe flow consistency to provide a more robust signal.
            # It is designed to FAIL-FAST and will not use fake data.
            try:
                enhancement_factor = self._calculate_flow_consistency_enhancement(input_data)
                vapi_fa_raw *= enhancement_factor
                self.logger.info(f"Applied VAPI-FA enhancement factor of {enhancement_factor:.2f} for {symbol}")
            except ValueError as e:
                self.logger.warning(f"Could not apply VAPI-FA enhancement for {symbol}: {e}. Using original calculation.")
            # --- AI ENHANCEMENT (END) ---

            # Unified caching and normalization
            vapi_fa_cache = self._add_to_intraday_cache(symbol, 'vapi_fa', vapi_fa_raw, max_size=200)
            vapi_fa_z_score = self._calculate_z_score_optimized(vapi_fa_cache, vapi_fa_raw)

            self.logger.debug(f"VAPI-FA results for {symbol}: raw={vapi_fa_raw:.2f}, z_score={vapi_fa_z_score:.2f}")
            return (vapi_fa_raw, vapi_fa_z_score, pvr_5m, flow_acceleration_5m)

        except Exception as e:
            self._raise_flow_calculation_error(f"VAPI-FA calculation for {symbol}: {e}")
    
    def _calculate_dwfd_optimized(self, input_data: FlowAnalyticsInput, symbol: str) -> Tuple[float, float, float]:
        """
        Optimized and corrected DWFD calculation adhering to the EOTS Guide v2.5.
        This metric now correctly measures the divergence between value and volume,
        then weights it by delta flow to spot "smart money" positioning.
        """
        try:
            self.logger.debug(f"DWFD calculation for {symbol}")

            # --- FAIL-FAST Data Extraction ---
            net_delta_flow_raw = input_data.net_cust_delta_flow_und
            net_value_flow_raw = input_data.net_value_flow_5m_und or input_data.value_bs
            net_vol_flow_raw = input_data.net_vol_flow_5m_und or input_data.volm_bs

            if net_delta_flow_raw is None:
                self._raise_flow_calculation_error(f"net_cust_delta_flow_und is None for {symbol}")
            if net_value_flow_raw is None:
                self._raise_flow_calculation_error(f"No value flow data available for {symbol}")
            if net_vol_flow_raw is None:
                self._raise_flow_calculation_error(f"No volume flow data available for {symbol}")

            net_delta_flow = float(net_delta_flow_raw)
            net_value_flow = float(net_value_flow_raw)
            net_vol_flow = float(net_vol_flow_raw)

            # --- Corrected Flow Divergence Calculation ---
            # As per the guide, this measures the divergence between VALUE and VOLUME flow.
            # A high value indicates large premium per contract, a sign of sophisticated or urgent positioning.
            flow_divergence = net_value_flow / net_vol_flow if abs(net_vol_flow) > EPSILON else 0.0

            # --- Corrected DWFD Raw Calculation ---
            # The divergence is weighted by the SIGN and MAGNITUDE of the delta flow.
            # - The SIGN of delta flow indicates the direction of the "smart money" (positive=bullish, negative=bearish).
            # - The sqrt of the magnitude gives weight to stronger delta flows without letting extreme values dominate.
            dwfd_raw = flow_divergence * np.sign(net_delta_flow) * (abs(net_delta_flow) ** 0.5)

            # --- AI ENHANCEMENT (START) ---
            # This enhancement remains valid as it scores institutional conviction, which can amplify the base signal.
            try:
                enhancement_factor = self._calculate_institutional_bias_enhancement(input_data)
                dwfd_raw *= enhancement_factor
                self.logger.info(f"Applied DWFD enhancement factor of {enhancement_factor:.2f} for {symbol}")
            except ValueError as e:
                self.logger.warning(f"Could not apply DWFD enhancement for {symbol}: {e}. Using original calculation.")
            # --- AI ENHANCEMENT (END) ---

            # Unified caching and normalization
            dwfd_cache = self._add_to_intraday_cache(symbol, 'dwfd', dwfd_raw, max_size=200)
            dwfd_z_score = self._calculate_z_score_optimized(dwfd_cache, dwfd_raw)

            self.logger.debug(f"DWFD results for {symbol}: raw={dwfd_raw:.2f}, z_score={dwfd_z_score:.2f}")
            return (dwfd_raw, dwfd_z_score, flow_divergence)

        except Exception as e:
            self._raise_flow_calculation_error(f"DWFD calculation for {symbol}: {e}")
    
    # --- AI ENHANCEMENT HELPER METHODS ---

    def _calculate_flow_consistency_enhancement(self, input_data: FlowAnalyticsInput) -> float:
        """
        AI ENHANCEMENT: Calculates a flow consistency score based on multiple timeframes.
        This provides a more robust measure of sustained momentum.
        FAIL-FAST: Raises ValueError if any required data is missing.
        """
        timeframes = [
            input_data.net_vol_flow_5m_und,
            input_data.net_vol_flow_15m_und,
            input_data.net_vol_flow_30m_und,
            input_data.net_vol_flow_60m_und
        ]
        
        if any(tf is None for tf in timeframes):
            raise ValueError("Missing multi-timeframe flow data for consistency enhancement.")
            
        flow_directions = np.sign(timeframes)
        # Score is high if all flows are in the same direction
        consistency_score = abs(np.mean(flow_directions))
        
        # Calculate momentum (latest vs earliest)
        momentum = (timeframes[0] - timeframes[-1]) / max(abs(timeframes[-1]), EPSILON) if timeframes[-1] != 0 else 0
        
        # Return an enhancement factor. 1.0 is neutral.
        return 1.0 + (consistency_score * np.tanh(momentum) * 0.25) # tanh to bound momentum, 0.25 to moderate effect

    def _calculate_institutional_bias_enhancement(self, input_data: FlowAnalyticsInput) -> float:
        """
        AI ENHANCEMENT: Uses Greek-level buy/sell data to score institutional conviction.
        FAIL-FAST: Raises ValueError if any required data is missing.
        """
        required_fields = [
            'gammas_call_buy', 'gammas_call_sell', 'gammas_put_buy', 'gammas_put_sell',
            'vflowratio'
        ]
        
        for field in required_fields:
            if getattr(input_data, field, None) is None:
                raise ValueError(f"Missing required field '{field}' for institutional bias enhancement.")
        
        gamma_buy_total = input_data.gammas_call_buy + input_data.gammas_put_buy
        gamma_sell_total = input_data.gammas_call_sell + input_data.gammas_put_sell
        
        if gamma_sell_total == 0: # Avoid division by zero, but fail if it's a suspicious state
            if gamma_buy_total > 0:
                raise ValueError("Suspicious state: Zero gamma sell flow but positive gamma buy flow.")
            return 1.0 # Neutral enhancement if both are zero
            
        gamma_bias_ratio = gamma_buy_total / gamma_sell_total
        
        # vflowratio is a direct institutional indicator. Weight it heavily.
        # A value far from 0.5 indicates institutional activity.
        vflow_institutional_weight = abs(input_data.vflowratio - 0.5) * 2
        
        # Combine factors. A high ratio and high weight indicate strong institutional bias.
        bias_score = np.tanh(gamma_bias_ratio - 1) * vflow_institutional_weight # tanh to bound the ratio effect
        
        # Return an enhancement factor. 1.0 is neutral.
        return 1.0 + (bias_score * 0.5) # 0.5 to moderate the effect
        
    def _calculate_tw_laf_optimized(self, input_data: FlowAnalyticsInput, symbol: str) -> Tuple[float, float, float, float]:
        """Optimized TW-LAF calculation using a validated Pydantic input model."""
        try:
            self.logger.debug(f"TW-LAF calculation for {symbol}")

            # TIERED WEEKEND SYSTEM: Extract time-weighted flows with off-hours handling
            net_vol_flow_5m_raw = input_data.net_vol_flow_5m_und or input_data.volm_bs
            net_vol_flow_15m_raw = input_data.net_vol_flow_15m_und
            net_vol_flow_30m_raw = input_data.net_vol_flow_30m_und

            # Handle off-hours gracefully
            net_vol_flow_5m = 0.0 if net_vol_flow_5m_raw is None else float(net_vol_flow_5m_raw)
            net_vol_flow_15m = 0.0 if net_vol_flow_15m_raw is None else float(net_vol_flow_15m_raw)
            net_vol_flow_30m = 0.0 if net_vol_flow_30m_raw is None else float(net_vol_flow_30m_raw)

            # Calculate liquidity factor (simplified)
            total_volume_raw = input_data.day_volume
            if total_volume_raw is None:
                self._raise_flow_calculation_error(f"day_volume missing for {symbol} - cannot calculate TW-LAF without real volume data!")
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
            return (tw_laf_raw, tw_laf_z_score, liquidity_factor, time_weighted_sum)

        except Exception as e:
            self._raise_flow_calculation_error(f"TW-LAF calculation for {symbol}: {e}")
    
    def _classify_flow_type_optimized(self, input_data: FlowAnalyticsInput) -> FlowType:
        """Optimized flow classification using simplified heuristics and a validated Pydantic input model."""
        try:
            total_volume_raw = input_data.day_volume
            net_value_flow_raw = input_data.net_value_flow_5m_und or input_data.value_bs
            net_vol_flow_raw = input_data.net_vol_flow_5m_und or input_data.volm_bs

            # Handle off-hours gracefully - use real data or zero for flow metrics
            total_volume = 1000000.0 if total_volume_raw is None else float(total_volume_raw)
            net_value_flow = 0.0 if net_value_flow_raw is None else float(net_value_flow_raw)
            net_vol_flow = 0.0 if net_vol_flow_raw is None else float(net_vol_flow_raw)

            # Calculate flow intensity
            flow_intensity = abs(net_value_flow) + abs(net_vol_flow)

            # -----------------------------------------------------------------
            # Dynamic thresholds supplied by adaptive parameter tuner
            # -----------------------------------------------------------------
            high_vol_threshold = self._get_dyn_param("high_volume_threshold")
            med_vol_threshold = self._get_dyn_param("medium_volume_threshold")
            inst_flow_intensity = self._get_dyn_param("institutional_flow_intensity")
            sophist_flow_intensity = self._get_dyn_param("sophisticated_flow_intensity")

            # Volume-based classification (dynamic)
            if total_volume > high_vol_threshold:
                if flow_intensity > inst_flow_intensity:
                    return FlowType.INSTITUTIONAL_LARGE
                elif flow_intensity > sophist_flow_intensity:
                    return FlowType.INSTITUTIONAL_SMALL
                else:
                    return FlowType.HEDGE_FUND
            elif total_volume > med_vol_threshold:
                if flow_intensity > sophist_flow_intensity:
                    return FlowType.RETAIL_SOPHISTICATED
                else:
                    return FlowType.RETAIL_UNSOPHISTICATED
            else:
                return FlowType.RETAIL_UNSOPHISTICATED

        except Exception as e:
            self.logger.warning(f"Error classifying flow type: {e}, returning UNKNOWN")
            return FlowType.UNKNOWN
    
    def _calculate_momentum_acceleration_index_optimized(self, input_data: FlowAnalyticsInput) -> float:
        """Optimized momentum acceleration calculation using a validated Pydantic input model."""
        try:
            net_vol_flow_5m_raw = input_data.net_vol_flow_5m_und or input_data.volm_bs
            net_vol_flow_15m_raw = input_data.net_vol_flow_15m_und
            net_vol_flow_30m_raw = input_data.net_vol_flow_30m_und

            # Handle off-hours gracefully
            net_vol_flow_5m = 0.0 if net_vol_flow_5m_raw is None else float(net_vol_flow_5m_raw)
            net_vol_flow_15m = 0.0 if net_vol_flow_15m_raw is None else float(net_vol_flow_15m_raw)
            net_vol_flow_30m = 0.0 if net_vol_flow_30m_raw is None else float(net_vol_flow_30m_raw)

            # Create synthetic flow series for momentum calculation
            flow_series = [net_vol_flow_30m, net_vol_flow_15m, net_vol_flow_5m]

            velocity = flow_series[-1] - flow_series[-2] if len(flow_series) >= 2 else 0.0
            prev_velocity = flow_series[-2] - flow_series[-3] if len(flow_series) >= 3 else 0.0
            acceleration = velocity - prev_velocity

            momentum_index = (abs(velocity) * 0.6 + abs(acceleration) * 0.4) / max(abs(net_vol_flow_5m), EPSILON)
            return self._bound_value(momentum_index, -10.0, 10.0)
            
        except Exception as e:
            self._raise_flow_calculation_error(f"Momentum acceleration index calculation: {e}")
    
    def _calculate_z_score_optimized(self, cache_data: List[float], current_value: float) -> float:
        """FAIL-FAST: Z-score calculation. No fake defaults allowed."""
        if not cache_data or len(cache_data) < 2:
            self._raise_flow_calculation_error(f"Insufficient cache data for Z-score - need at least 2 data points, got {len(cache_data) if cache_data else 0}")
        
        try:
            mean_val = np.mean(cache_data)
            std_val = np.std(cache_data)
            return (current_value - mean_val) / max(std_val, EPSILON)
                
        except Exception as e:
            self._raise_flow_calculation_error(f"Z-score calculation failed: {e}")
    
    def _raise_flow_calculation_error(self, error_context: str) -> None:
        """FAIL-FAST: Raise error instead of returning fake flow metrics."""
        raise ValueError(f"CRITICAL: Flow analytics calculation failed in {error_context} - cannot return fake flow metrics that could cause massive trading losses!")

# Export the consolidated calculator
__all__ = ['FlowAnalytics', 'FlowType', 'FlowAnalyticsInput', 'FlowAnalyticsOutput']
