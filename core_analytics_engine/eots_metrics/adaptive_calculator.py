# core_analytics_engine/eots_metrics/adaptive_calculator.py

"""
EOTS Adaptive Calculator - Consolidated Adaptive Metrics with Regime Detection

Consolidates:
- adaptive_metrics.py: Tier 2 adaptive metrics (A-DAG, E-SDAG, D-TDPI, VRI 2.0)
- elite_regime_detector.py: Market regime classification
- elite_volatility_surface.py: Volatility surface analysis

Optimizations:
- Unified regime detection logic with Pydantic-first models
- Streamlined adaptive calculations with explicit data contracts
- Integrated volatility surface analysis
- Eliminated dictionary passing in favor of validated Pydantic models
- Replaced .to_dict() with .model_dump() and uses .model_validate() for strict validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field, ConfigDict

from core_analytics_engine.eots_metrics.core_calculator import CoreCalculator
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_models import ProcessedUnderlyingAggregatesV2_5, EOTSBaseModel

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class MarketRegime(Enum):
    """Consolidated market regime classifications"""
    LOW_VOL_TRENDING = "low_vol_trending"
    LOW_VOL_RANGING = "low_vol_ranging"
    MEDIUM_VOL_TRENDING = "medium_vol_trending"
    MEDIUM_VOL_RANGING = "medium_vol_ranging"
    HIGH_VOL_TRENDING = "high_vol_trending"
    HIGH_VOL_RANGING = "high_vol_ranging"
    STRESS_REGIME = "stress_regime"
    EXPIRATION_REGIME = "expiration_regime"
    REGIME_UNCLEAR_OR_TRANSITIONING = "regime_unclear_or_transitioning"

# =============================================================================
# PYDANTIC-FIRST DATA MODELS FOR ADAPTIVE CALCULATIONS
# =============================================================================

class AdaptiveMetricsInput(EOTSBaseModel):
    """
    Strictly typed input model for a single strike's adaptive calculations.
    Ensures all required data is present and validated before calculation.
    """
    # From underlying data
    underlying_price: float
    u_volatility: float
    price_change_pct_und: float
    day_volume: float
    net_cust_gamma_flow_und: float

    # From strike-level data
    strike: float
    dte: float
    implied_volatility: float
    total_gxoi_at_strike: float
    total_dxoi_at_strike: float
    total_txoi_at_strike: float
    total_vxoi_at_strike: float
    total_vannaxoi_at_strike: float
    total_volume: float

    model_config = ConfigDict(extra='ignore') # Ignore extra fields not needed for this calculation

class AdaptiveMetricsOutput(EOTSBaseModel):
    """
    Strictly typed output model containing all calculated adaptive metrics for a single strike.
    Ensures a consistent and predictable data contract for downstream consumers.
    """
    strike: float  # Key for merging back to the main DataFrame
    market_regime: MarketRegime
    volatility_regime: str

    # A-DAG
    a_dag_strike: float
    a_dag_adaptive_alpha: float
    a_dag_flow_alignment: float
    
    # E-SDAG
    e_sdag_mult_strike: float
    e_sdag_skew_adj: float
    
    # D-TDPI
    d_tdpi_strike: float
    d_tdpi_dte_scaling: float
    
    # VRI 2.0
    vri_2_0_strike: float
    vri_2_0_vol_mult: float
    
    # Concentration Indices
    gci_score: float
    dci_score: float
    
    # 0DTE Suite
    vri_0dte_strike: float
    e_vfi_sens_strike: float
    e_vvr_sens_strike: float
    vci_0dte_score: float

# =============================================================================
# ADAPTIVE CALCULATOR CLASS
# =============================================================================

class AdaptiveCalculator(CoreCalculator):
    """
    Consolidated adaptive calculator with regime detection and volatility surface analysis.
    Refactored to use a strict Pydantic-first architecture.
    """
    
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: Any = None):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config = elite_config or {}
        
        # Adaptive calculation thresholds (optimized from original modules)
        self.VOLATILITY_THRESHOLDS = {'low': 0.15, 'high': 0.30}
        self.DTE_THRESHOLDS = {'short': 7, 'medium': 30, 'long': 60}
        self.CONCENTRATION_WINDOW = 20
    
    def calculate_all_adaptive_metrics(self, df_strike: pd.DataFrame, und_data: ProcessedUnderlyingAggregatesV2_5) -> List[AdaptiveMetricsOutput]:
        """
        Orchestrates calculation of all adaptive metrics using a strict Pydantic v2 architecture.
        This method now returns a list of Pydantic models, not a DataFrame.

        Args:
            df_strike: DataFrame with strike-level data.
            und_data: A complete `ProcessedUnderlyingAggregatesV2_5` Pydantic model.

        Returns:
            List[AdaptiveMetricsOutput]: A list of Pydantic models, each containing the
                                         calculated adaptive metrics for a single strike.
        """
        if df_strike.empty:
            self.logger.warning("Input strike DataFrame is empty. Cannot calculate adaptive metrics.")
            return []

        self.logger.debug(f"Starting adaptive metrics calculation for {len(df_strike)} strikes...")

        # Prepare a base dictionary from the underlying data to avoid repeated attribute access in the loop.
        # This uses .model_dump() once, which is efficient.
        underlying_dict = und_data.model_dump()
        results: List[AdaptiveMetricsOutput] = []

        for _, row in df_strike.iterrows():
            try:
                # Combine data from the underlying model and the current strike row
                combined_data = {**underlying_dict, **row.to_dict()}

                # Use .model_validate() to create a strictly validated input object.
                # This is the FAIL-FAST entry point for each strike's calculation.
                input_model = AdaptiveMetricsInput.model_validate(combined_data)

                # Calculate all metrics for the validated input model
                output_model = self._calculate_metrics_for_strike(input_model)
                results.append(output_model)

            except Exception as e:
                # Log the error for the specific strike but continue processing others.
                # The "fail-fast" for this strike has already occurred via the exception.
                self.logger.error(f"Failed to calculate adaptive metrics for strike {row.get('strike', 'N/A')}: {e}", exc_info=True)
                # Do not append a fake or default result. The strike will be skipped.

        self.logger.debug(f"Adaptive metrics calculation complete. Successfully processed {len(results)} of {len(df_strike)} strikes.")
        return results

    def _calculate_metrics_for_strike(self, input_data: AdaptiveMetricsInput) -> AdaptiveMetricsOutput:
        """
        Calculates all adaptive metrics for a single, validated strike input.
        """
        # Determine market context (regime, volatility, etc.)
        market_regime = self._determine_market_regime_optimized(input_data)
        volatility_regime = self._determine_volatility_regime_optimized(input_data)
        volatility_context = self._get_volatility_context_optimized(input_data)
        dte_context = self._get_average_dte_context_optimized(input_data)

        # Calculate individual adaptive metrics
        a_dag_strike, a_dag_alpha, a_dag_align = self._calculate_a_dag_optimized(input_data, market_regime)
        e_sdag_strike, e_sdag_adj = self._calculate_e_sdag_optimized(input_data)
        d_tdpi_strike, d_tdpi_scale = self._calculate_d_tdpi_optimized(input_data)
        vri_2_0_strike, vri_2_0_mult = self._calculate_vri_2_0_optimized(input_data, volatility_context)
        gci_score, dci_score = self._calculate_concentration_indices_optimized(input_data)
        vri_0dte, vfi_0dte, vvr_0dte, vci_0dte = self._calculate_0dte_suite_optimized(input_data)

        # Assemble the validated output model
        return AdaptiveMetricsOutput(
            strike=input_data.strike,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            a_dag_strike=a_dag_strike,
            a_dag_adaptive_alpha=a_dag_alpha,
            a_dag_flow_alignment=a_dag_align,
            e_sdag_mult_strike=e_sdag_strike,
            e_sdag_skew_adj=e_sdag_adj,
            d_tdpi_strike=d_tdpi_strike,
            d_tdpi_dte_scaling=d_tdpi_scale,
            vri_2_0_strike=vri_2_0_strike,
            vri_2_0_vol_mult=vri_2_0_mult,
            gci_score=gci_score,
            dci_score=dci_score,
            vri_0dte_strike=vri_0dte,
            e_vfi_sens_strike=vfi_0dte,
            e_vvr_sens_strike=vvr_0dte,
            vci_0dte_score=vci_0dte
        )

    # =============================================================================
    # REGIME DETECTION & CONTEXT
    # =============================================================================
    
    def _determine_market_regime_optimized(self, input_data: AdaptiveMetricsInput) -> MarketRegime:
        """Optimized market regime detection using simplified heuristics on validated input."""
        # Volatility classification
        if input_data.u_volatility > self.VOLATILITY_THRESHOLDS['high']:
            vol_regime = 'HIGH_VOL'
        elif input_data.u_volatility < self.VOLATILITY_THRESHOLDS['low']:
            vol_regime = 'LOW_VOL'
        else:
            vol_regime = 'MEDIUM_VOL'
        
        # Trend classification
        if abs(input_data.price_change_pct_und) > 0.02:
            trend_regime = 'TRENDING'
        else:
            trend_regime = 'RANGING'
        
        # Special regime checks
        if input_data.u_volatility > 0.50 and input_data.day_volume > 50_000_000:
            return MarketRegime.STRESS_REGIME
        
        if input_data.dte <= 7:
            return MarketRegime.EXPIRATION_REGIME
        
        regime_name = f"{vol_regime}_{trend_regime}".lower()
        return MarketRegime(regime_name)

    def _determine_volatility_regime_optimized(self, input_data: AdaptiveMetricsInput) -> str:
        """Optimized volatility regime determination on validated input."""
        iv = input_data.implied_volatility
        if iv > 0.40:
            return "high_vol"
        elif iv < 0.15:
            return "low_vol"
        else:
            return "normal"

    def _get_volatility_context_optimized(self, input_data: AdaptiveMetricsInput) -> str:
        """Optimized volatility context determination on validated input."""
        if input_data.u_volatility > self.VOLATILITY_THRESHOLDS['high']:
            return 'HIGH_VOL'
        elif input_data.u_volatility < self.VOLATILITY_THRESHOLDS['low']:
            return 'LOW_VOL'
        else:
            return 'NORMAL_VOL'
    
    def _get_average_dte_context_optimized(self, input_data: AdaptiveMetricsInput) -> str:
        """Optimized DTE context determination on validated input."""
        if input_data.dte <= self.DTE_THRESHOLDS['short']:
            return 'SHORT_DTE'
        elif input_data.dte <= self.DTE_THRESHOLDS['medium']:
            return 'MEDIUM_DTE'
        else:
            return 'LONG_DTE'

    # =============================================================================
    # INDIVIDUAL ADAPTIVE METRIC CALCULATIONS
    # =============================================================================
    
    def _calculate_a_dag_optimized(self, input_data: AdaptiveMetricsInput, market_regime: MarketRegime) -> tuple[float, float, float]:
        """Calculates A-DAG on a validated input model."""
        regime_multipliers = {
            MarketRegime.HIGH_VOL_TRENDING: 1.5, MarketRegime.HIGH_VOL_RANGING: 1.2,
            MarketRegime.STRESS_REGIME: 2.0, MarketRegime.EXPIRATION_REGIME: 0.8
        }
        adaptive_alpha = regime_multipliers.get(market_regime, 1.0)
        flow_alignment = np.sign(input_data.net_cust_gamma_flow_und) if abs(input_data.net_cust_gamma_flow_und) > 1000 else 0
        
        a_dag_exposure = input_data.total_gxoi_at_strike * adaptive_alpha
        a_dag_score = a_dag_exposure * (1 + flow_alignment * 0.2)
        
        return a_dag_score, adaptive_alpha, flow_alignment
    
    def _calculate_e_sdag_optimized(self, input_data: AdaptiveMetricsInput) -> tuple[float, float]:
        """Calculates E-SDAG on a validated input model."""
        moneyness = input_data.strike / max(input_data.underlying_price, EPSILON)
        skew_adj = 1.1 if moneyness < 0.95 else (0.9 if moneyness > 1.05 else 1.0)
        e_sdag_score = input_data.total_gxoi_at_strike * skew_adj
        return e_sdag_score, skew_adj
    
    def _calculate_d_tdpi_optimized(self, input_data: AdaptiveMetricsInput) -> tuple[float, float]:
        """
        Calculates D-TDPI, enhanced to be fully adaptive as per the EOTS Guide v2.5.
        This version incorporates a dynamic Gaussian width based on underlying volatility
        to accurately model time decay pressure around the current price.
        """
        # DTE scaling remains a key component, emphasizing near-term expiries.
        dte_scaling = 2.0 if input_data.dte <= 7 else (1.5 if input_data.dte <= 30 else 1.0)
        
        # Dynamic Gaussian width based on underlying volatility.
        # Higher volatility leads to a wider distribution of potential price moves, so we widen the Gaussian.
        # A baseline width is set and then scaled by volatility.
        base_width = 0.02 # Corresponds to a 2% move
        dynamic_width = base_width * (1 + input_data.u_volatility)
        
        # Gaussian function to weigh theta exposure by strike proximity.
        # Strikes closer to the underlying price will have a weight closer to 1.
        price_distance = (input_data.strike - input_data.underlying_price) / input_data.underlying_price
        gaussian_weight = np.exp(-0.5 * (price_distance / dynamic_width) ** 2)
        
        # The final score is the raw theta exposure, scaled by DTE and weighted by its proximity to the current price.
        d_tdpi_score = input_data.total_txoi_at_strike * dte_scaling * gaussian_weight
        
        # The scaling factor returned now represents the combined effect of DTE and Gaussian weighting.
        combined_scaling_factor = dte_scaling * gaussian_weight
        return d_tdpi_score, combined_scaling_factor
    
    def _calculate_vri_2_0_optimized(self, input_data: AdaptiveMetricsInput, volatility_context: str) -> tuple[float, float]:
        """
        Calculates VRI 2.0, enhanced to be fully adaptive as per the EOTS Guide v2.5.
        This version directly uses volatility and Vanna exposure to create a more
        sensitive and accurate measure of volatility risk and hedging pressure.
        """
        # The core of VRI 2.0 is the interaction between Vega (volatility sensitivity) and Vanna (delta's sensitivity to IV).
        # High Vanna indicates that hedging behavior will change rapidly with IV, a key risk vector.
        # We use the raw underlying volatility for a more continuous, dynamic adjustment.
        
        # The multiplier is now a product of the underlying volatility and the Vanna exposure itself.
        # We use np.tanh on Vanna to bound its effect, preventing extreme values from dominating.
        # The sign of Vanna is preserved to indicate directional risk.
        vanna_component = np.tanh(input_data.total_vannaxoi_at_strike / 1e6) # Normalize Vanna before tanh
        
        # The dynamic multiplier directly reflects the current volatility level and the Vanna risk.
        dynamic_multiplier = (1 + input_data.u_volatility) * (1 + vanna_component)
        
        # The final score is the raw Vega exposure, amplified by our dynamic, risk-aware multiplier.
        vri_2_0_score = input_data.total_vxoi_at_strike * dynamic_multiplier
        
        return vri_2_0_score, dynamic_multiplier
    
    def _calculate_concentration_indices_optimized(self, input_data: AdaptiveMetricsInput) -> tuple[float, float]:
        """Calculates concentration indices on a validated input model."""
        # Note: This calculation is inherently limited without the full context of other strikes.
        # A more accurate implementation would require the full series.
        # This provides a strike-relative measure.
        gci_score = input_data.total_gxoi_at_strike / max(abs(input_data.net_cust_gamma_flow_und), EPSILON)
        dci_score = input_data.total_dxoi_at_strike / max(abs(input_data.net_cust_delta_flow_und), EPSILON)
        return gci_score, dci_score
    
    def _calculate_0dte_suite_optimized(self, input_data: AdaptiveMetricsInput) -> tuple[float, float, float, float]:
        """Calculates 0DTE suite metrics on a validated input model."""
        is_0dte = input_data.dte <= 2
        if not is_0dte:
            return 0.0, 0.0, 0.0, 0.0

        vri_0dte = (input_data.total_gxoi_at_strike * input_data.total_vxoi_at_strike) / max(input_data.total_volume, EPSILON)
        vfi_0dte = input_data.total_vxoi_at_strike * 2.0
        vvr_0dte = input_data.total_vxoi_at_strike / max(input_data.total_volume, EPSILON)
        vci_0dte = input_data.total_vannaxoi_at_strike / max(abs(input_data.total_vannaxoi_at_strike), EPSILON) # Simplified

        return vri_0dte, vfi_0dte, vvr_0dte, vci_0dte

# Export the consolidated calculator
__all__ = ['AdaptiveCalculator', 'MarketRegime', 'AdaptiveMetricsInput', 'AdaptiveMetricsOutput']
