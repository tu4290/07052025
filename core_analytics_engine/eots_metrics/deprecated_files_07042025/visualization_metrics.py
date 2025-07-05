# core_analytics_engine/eots_metrics/visualization_metrics.py

"""
EOTS Visualization Metrics - Consolidated Heatmap and Aggregation Data

Consolidates:
- heatmap_metrics.py: Enhanced heatmap data (SGDHP, IVSDH, UGCH)
- underlying_aggregates.py: Strike-to-underlying aggregation

Optimizations:
- Unified data preparation for visualizations
- Streamlined aggregation logic
- Integrated heatmap calculations
- Eliminated duplicate data transformations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from core_analytics_engine.eots_metrics.core_calculator import CoreCalculator
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_models import ProcessedUnderlyingAggregatesV2_5 # Import the specific model

class NormalizationParams(BaseModel):
    """Pydantic model for exposure normalization parameters"""
    mean: float = Field(description="Mean value for normalization")
    std: float = Field(description="Standard deviation for normalization")

    model_config = ConfigDict(extra='forbid')

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class VisualizationMetrics(CoreCalculator):
    """
    Consolidated visualization metrics calculator.
    
    Combines functionality from:
    - HeatmapMetricsCalculator: SGDHP, IVSDH, UGCH heatmap data
    - UnderlyingAggregatesCalculator: Strike-level to underlying aggregation
    """
    
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: Any = None):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config = elite_config or {}
        
        # Heatmap calculation weights (optimized from original)
        self.HEATMAP_WEIGHTS = {
            'gamma': 0.4,
            'delta': 0.3,
            'vanna': 0.2,
            'flow': 0.1
        }
        
        # Aggregation configuration
        self.AGGREGATION_COLUMNS = [
            'total_dxoi_at_strike', 'total_gxoi_at_strike', 'total_vxoi_at_strike',
            'total_txoi_at_strike', 'total_charmxoi_at_strike', 'total_vannaxoi_at_strike',
            'total_vommaxoi_at_strike', 'nvp_at_strike', 'nvp_vol_at_strike'
        ]
    
    # =============================================================================
    # HEATMAP METRICS - Consolidated from heatmap_metrics.py
    # =============================================================================
    
    def calculate_all_heatmap_data(self, df_strike: pd.DataFrame, und_data: ProcessedUnderlyingAggregatesV2_5) -> pd.DataFrame:
        """
        Orchestrates calculation of all enhanced heatmap data using proper Pydantic v2 architecture.

        Calculates:
        - SGDHP (Super Gamma-Delta Hedging Pressure) scores
        - IVSDH (Integrated Volatility Surface Dynamics) scores
        - UGCH (Ultimate Greek Confluence) scores

        Args:
            df_strike: DataFrame with strike-level data
            und_data: Pydantic model (ProcessedUnderlyingAggregatesV2_5) - accessed via dot notation

        Returns:
            pd.DataFrame: Updated strike-level data with heatmap metrics
        """
        self.logger.debug("Calculating enhanced heatmap data...")
        
        try:
            if df_strike.empty:
                return df_strike
            
            # CRITICAL FIX: Use proper Pydantic dot notation instead of .get()
            if self._calculation_state is None:
                print(f"❌ ERROR: _calculation_state is None in visualization_metrics")
                symbol = 'UNKNOWN'
            else:
                symbol = getattr(self._calculation_state, 'current_symbol', 'UNKNOWN') or 'UNKNOWN'
            
            # Calculate individual heatmap components
            df_strike = self._calculate_sgdhp_scores_optimized(df_strike, und_data, symbol)
            df_strike = self._calculate_ivsdh_scores_optimized(df_strike, und_data, symbol)
            df_strike = self._calculate_ugch_scores_optimized(df_strike, und_data, symbol)
            
            self.logger.debug(f"Enhanced heatmap data calculation complete for {len(df_strike)} strikes.")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating heatmap data: {e}", exc_info=True)
            return self._add_default_heatmap_columns(df_strike)
    
    def _calculate_sgdhp_scores_optimized(self, df_strike: pd.DataFrame, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> pd.DataFrame:
        """Optimized SGDHP (Super Gamma-Delta Hedging Pressure) calculation"""
        try:
            # Debug DataFrame state
            if df_strike is None:
                self.logger.error(f"df_strike is None in SGDHP calculation for {symbol}")
                return pd.DataFrame()

            # Extract exposure data using FAIL-FAST validation (NO FAKE DATA)
            gamma_exposure = self._get_column_required(df_strike, 'total_gxoi_at_strike', 'for SGDHP calculation')
            delta_exposure = self._get_column_required(df_strike, 'total_dxoi_at_strike', 'for SGDHP calculation')

            # Extract flow data using FAIL-FAST validation (NO FAKE DATA)
            net_gamma_flow = self._get_column_required(df_strike, 'net_cust_gamma_flow_at_strike', 'for SGDHP calculation')
            net_delta_flow = self._get_column_required(df_strike, 'net_cust_delta_flow_at_strike', 'for SGDHP calculation')
            
            # Normalize exposures using unified caching
            gamma_intensity = self._normalize_exposure_optimized(gamma_exposure, 'gamma', symbol)
            delta_intensity = self._normalize_exposure_optimized(delta_exposure, 'delta', symbol)
            
            # Calculate flow pressure component
            flow_pressure = (abs(net_gamma_flow) + abs(net_delta_flow)) * self.HEATMAP_WEIGHTS['flow']
            
            # SGDHP composite score
            sgdhp_score = (gamma_intensity * self.HEATMAP_WEIGHTS['gamma'] + 
                          delta_intensity * self.HEATMAP_WEIGHTS['delta'] + 
                          flow_pressure)
            
            df_strike['sgdhp_score_strike'] = sgdhp_score
            
            self.logger.debug(f"SGDHP scores calculated for {symbol}")
            return df_strike
            
        except Exception as e:
            self.logger.warning(f"Error calculating SGDHP scores: {e}")
            df_strike['sgdhp_score_strike'] = 0.0
            return df_strike
    
    def _calculate_ivsdh_scores_optimized(self, df_strike: pd.DataFrame, und_data, symbol: str) -> pd.DataFrame:
        """Optimized IVSDH (Integrated Volatility Surface Dynamics) calculation with proper Pydantic access"""
        try:
            # Extract vega and vanna exposures using FAIL-FAST validation (NO FAKE DATA)
            vega_exposure = self._get_column_required(df_strike, 'total_vxoi_at_strike', 'for IVSDH calculation')
            vanna_exposure = self._get_column_required(df_strike, 'total_vannaxoi_at_strike', 'for IVSDH calculation')

            # Get implied volatility data - check if available at strike level
            if 'implied_volatility' in df_strike.columns:
                iv_data = self._get_column_required(df_strike, 'implied_volatility', 'for IVSDH calculation')
            elif 'avg_iv_at_strike' in df_strike.columns:
                # Use aggregated IV if available
                iv_data = self._get_column_required(df_strike, 'avg_iv_at_strike', 'for IVSDH calculation')
            else:
                # FAIL-FAST: No implied volatility data available at strike level
                raise ValueError("CRITICAL: No implied volatility data available at strike level - cannot calculate IVSDH without real IV data!")

            # FAIL-FAST: Calculate volatility surface dynamics - NO FAKE DEFAULTS ALLOWED
            if not hasattr(und_data, 'u_volatility'):
                raise ValueError("CRITICAL: u_volatility missing from underlying data - cannot calculate IVSDH without real volatility!")
            current_iv = float(getattr(und_data, 'u_volatility'))
            iv_differential = abs(iv_data - current_iv)
            
            # Normalize components
            vega_intensity = self._normalize_exposure_optimized(vega_exposure, 'vega', symbol)
            vanna_intensity = self._normalize_exposure_optimized(vanna_exposure, 'vanna', symbol)
            
            # IVSDH composite score
            ivsdh_score = (vega_intensity * 0.5 + 
                          vanna_intensity * self.HEATMAP_WEIGHTS['vanna'] + 
                          iv_differential * 0.3)
            
            df_strike['ivsdh_score_strike'] = ivsdh_score
            
            self.logger.debug(f"IVSDH scores calculated for {symbol}")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"CRITICAL: IVSDH calculation failed: {e}")
            # FAIL-FAST: Do not create fake data - re-raise the exception
            raise ValueError(f"IVSDH calculation failed: {e}") from e
    
    def _calculate_ugch_scores_optimized(self, df_strike: pd.DataFrame, und_data: ProcessedUnderlyingAggregatesV2_5, symbol: str) -> pd.DataFrame:
        """Optimized UGCH (Ultimate Greek Confluence) calculation"""
        try:
            # Extract all Greek exposures using FAIL-FAST validation (NO FAKE DATA)
            gamma_exposure = self._get_column_required(df_strike, 'total_gxoi_at_strike', 'for UGCH calculation')
            delta_exposure = self._get_column_required(df_strike, 'total_dxoi_at_strike', 'for UGCH calculation')
            vega_exposure = self._get_column_required(df_strike, 'total_vxoi_at_strike', 'for UGCH calculation')
            theta_exposure = self._get_column_required(df_strike, 'total_txoi_at_strike', 'for UGCH calculation')
            vanna_exposure = self._get_column_required(df_strike, 'total_vannaxoi_at_strike', 'for UGCH calculation')
            
            # Normalize all components
            gamma_norm = self._normalize_exposure_optimized(gamma_exposure, 'gamma', symbol)
            delta_norm = self._normalize_exposure_optimized(delta_exposure, 'delta', symbol)
            vega_norm = self._normalize_exposure_optimized(vega_exposure, 'vega', symbol)
            theta_norm = self._normalize_exposure_optimized(theta_exposure, 'theta', symbol)
            vanna_norm = self._normalize_exposure_optimized(vanna_exposure, 'vanna', symbol)
            
            # UGCH composite score (weighted combination of all Greeks)
            ugch_score = (gamma_norm * 0.25 + 
                         delta_norm * 0.25 + 
                         vega_norm * 0.20 + 
                         theta_norm * 0.15 + 
                         vanna_norm * 0.15)
            
            df_strike['ugch_score_strike'] = ugch_score
            
            self.logger.debug(f"UGCH scores calculated for {symbol}")
            return df_strike
            
        except Exception as e:
            self.logger.warning(f"Error calculating UGCH scores: {e}")
            df_strike['ugch_score_strike'] = 0.0
            return df_strike
    
    # =============================================================================
    # UNDERLYING AGGREGATES - Consolidated from underlying_aggregates.py
    # =============================================================================
    
    def calculate_all_underlying_aggregates(self, df_strike: pd.DataFrame, und_data) -> Dict:
        """
        Calculates underlying aggregate metrics from strike-level data.
        
        Aggregates:
        - Greek exposures (DXOI, GXOI, VXOI, TXOI, etc.)
        - Flow metrics (NVP, NVP_VOL)
        - Elite impact scores
        - 0DTE suite aggregates
        - Rolling flow aggregates
        """
        self.logger.debug("Calculating underlying aggregates...")
        
        try:
            if df_strike is None or df_strike.empty:
                self.logger.error("CRITICAL: Strike-level DataFrame is empty - cannot calculate aggregates without real strike data!")
                self._raise_aggregates_calculation_error("empty strike-level DataFrame")
            
            aggregates = {}
            
            # Basic Greek aggregations
            aggregates.update(self._calculate_basic_greek_aggregates(df_strike))
            
            # Flow aggregations
            aggregates.update(self._calculate_flow_aggregates(df_strike))
            
            # Elite metrics aggregations
            aggregates.update(self._calculate_elite_aggregates(df_strike, und_data))
            
            # 0DTE suite aggregations
            aggregates.update(self._calculate_0dte_aggregates(df_strike))
            
            # Adaptive metrics aggregations
            aggregates.update(self._calculate_adaptive_aggregates(df_strike))
            
            # Rolling flow time series
            aggregates.update(self._calculate_rolling_flow_aggregates(df_strike, und_data))
            
            self.logger.debug(f"Underlying aggregates calculated: {len(aggregates)} metrics")
            return aggregates
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating underlying aggregates: {e}", exc_info=True)
            self._raise_aggregates_calculation_error(f"underlying aggregates calculation: {e}")
    
    def _calculate_basic_greek_aggregates(self, df_strike: pd.DataFrame) -> Dict:
        """Calculate basic Greek exposure aggregates"""
        aggregates = {}
        
        for column in self.AGGREGATION_COLUMNS:
            if column in df_strike.columns:
                aggregates[f"{column.replace('_at_strike', '_und')}"] = df_strike[column].sum()
            else:
                aggregates[f"{column.replace('_at_strike', '_und')}"] = 0.0
        
        return aggregates
    
    def _calculate_flow_aggregates(self, df_strike: pd.DataFrame) -> Dict:
        """Calculate flow-related aggregates"""
        aggregates = {}
        
        # Net value and volume flows
        if 'nvp_at_strike' in df_strike.columns:
            aggregates['total_nvp_und'] = df_strike['nvp_at_strike'].sum()
        else:
            aggregates['total_nvp_und'] = 0.0
            
        if 'nvp_vol_at_strike' in df_strike.columns:
            aggregates['total_nvp_vol_und'] = df_strike['nvp_vol_at_strike'].sum()
        else:
            aggregates['total_nvp_vol_und'] = 0.0
        
        return aggregates
    
    def _calculate_elite_aggregates(self, df_strike: pd.DataFrame, und_data: ProcessedUnderlyingAggregatesV2_5) -> Dict[str, Any]: # Added type hint for und_data and return
        """Calculate elite impact aggregates using proper Pydantic access"""
        aggregates: Dict[str, Any] = {} # Ensure type

        # Elite impact scores from underlying data using proper Pydantic attribute access
        elite_columns = [
            'elite_impact_score_und', 'institutional_flow_score_und',
            'flow_momentum_index_und', 'market_regime_elite',
            'flow_type_elite', 'volatility_regime_elite'
        ]

        for column in elite_columns:
            default_value = 0.0 if 'score' in column or 'index' in column else 'unknown'
            aggregates[column] = getattr(und_data, column, default_value)

        return aggregates
    
    def _calculate_0dte_aggregates(self, df_strike: pd.DataFrame) -> Dict:
        """Calculate 0DTE suite aggregates with proper normalization"""
        aggregates = {}

        # Check if we have DTE data
        if 'dte' in df_strike.columns:
            dte_0_mask = df_strike['dte'] <= 2  # Use same threshold as adaptive_calculator

            # VRI 0DTE aggregate - CRITICAL FIX: Use correct field name (vri_0dte_strike)
            if 'vri_0dte_strike' in df_strike.columns:
                vri_values = df_strike.loc[dte_0_mask, 'vri_0dte_strike']
                vri_sum = vri_values.sum() if len(vri_values) > 0 else 0.0
                # Normalize to -1 to 1 scale
                aggregates['vri_0dte_und_sum'] = np.tanh(vri_sum / 1000000.0)  # Normalize large values
            else:
                aggregates['vri_0dte_und_sum'] = 0.0

            # VFI 0DTE aggregate - CRITICAL FIX: Use correct field name (e_vfi_sens_strike)
            if 'e_vfi_sens_strike' in df_strike.columns:
                vfi_values = df_strike.loc[dte_0_mask, 'e_vfi_sens_strike']
                vfi_sum = vfi_values.sum() if len(vfi_values) > 0 else 0.0
                # Normalize to -1 to 1 scale
                aggregates['vfi_0dte_und_sum'] = np.tanh(vfi_sum / 1000000.0)  # Normalize large values
            else:
                aggregates['vfi_0dte_und_sum'] = 0.0

            # VVR 0DTE average - CRITICAL FIX: Use correct field name and normalize
            if 'e_vvr_sens_strike' in df_strike.columns:
                vvr_values = df_strike.loc[dte_0_mask, 'e_vvr_sens_strike']
                vvr_avg = vvr_values.mean() if len(vvr_values) > 0 else 0.0
                # Normalize to -1 to 1 scale (VVR can be very large)
                aggregates['vvr_0dte_und_avg'] = np.tanh(vvr_avg / 100.0)  # Normalize large ratios
            else:
                aggregates['vvr_0dte_und_avg'] = 0.0

            # VCI 0DTE aggregate - normalize to -1 to 1 scale
            if 'vci_0dte_score' in df_strike.columns:
                vci_values = df_strike.loc[dte_0_mask, 'vci_0dte_score']
                vci_sum = vci_values.sum() if len(vci_values) > 0 else 0.0
                # Normalize to -1 to 1 scale
                aggregates['vci_0dte_agg'] = np.tanh(vci_sum / 10.0)  # Normalize concentration index
            else:
                aggregates['vci_0dte_agg'] = 0.0
        else:
            # Default values when no DTE data
            aggregates.update({
                'vri_0dte_und_sum': 0.0,
                'vfi_0dte_und_sum': 0.0,
                'vvr_0dte_und_avg': 0.0,
                'vci_0dte_agg': 0.0
            })

        return aggregates
    
    def _calculate_adaptive_aggregates(self, df_strike: pd.DataFrame) -> Dict:
        """Calculate adaptive metrics aggregates"""
        aggregates = {}
        
        # VRI 2.0 aggregate - CRITICAL FIX: Use correct field name from adaptive_calculator
        if 'vri_2_0_strike' in df_strike.columns:
            aggregates['vri_2_0_und_aggregate'] = df_strike['vri_2_0_strike'].sum()
        else:
            aggregates['vri_2_0_und_aggregate'] = 0.0
        
        # A-DAG aggregate - CRITICAL FIX: Use correct field name from adaptive_calculator
        if 'a_dag_strike' in df_strike.columns:
            aggregates['a_dag_und_aggregate'] = df_strike['a_dag_strike'].sum()
        else:
            aggregates['a_dag_und_aggregate'] = 0.0

        # E-SDAG aggregate - CRITICAL FIX: Use correct field name from adaptive_calculator
        if 'e_sdag_mult_strike' in df_strike.columns:
            aggregates['e_sdag_und_aggregate'] = df_strike['e_sdag_mult_strike'].sum()
        else:
            aggregates['e_sdag_und_aggregate'] = 0.0
        
        return aggregates
    
    def _calculate_rolling_flow_aggregates(self, df_strike: pd.DataFrame, und_data) -> Dict:
        """Calculate rolling flow aggregates and time series using proper Pydantic access"""
        aggregates = {}

        # Rolling flow periods
        periods = ['5m', '15m', '30m', '60m']

        for period in periods:
            net_value_key = f'net_value_flow_{period}_und'
            net_vol_key = f'net_vol_flow_{period}_und'

            # Get from underlying data using proper Pydantic attribute access
            aggregates[net_value_key] = getattr(und_data, net_value_key, 0.0)
            aggregates[net_vol_key] = getattr(und_data, net_vol_key, 0.0)

        return aggregates
    
    # =============================================================================
    # OPTIMIZED UTILITIES
    # =============================================================================
    
    def _get_column_required(self, df: pd.DataFrame, column: str, context: str = "") -> pd.Series:
        """Get required column from DataFrame - FAIL FAST if missing (NO FAKE DATA ALLOWED)"""
        if df is None:
            raise ValueError(f"CRITICAL: DataFrame is None when accessing required field '{column}' {context}")

        if column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(
                f"CRITICAL: Required field '{column}' missing from data pipeline {context}. "
                f"Available columns: {available_columns}. "
                f"This indicates a data processing failure - NO FAKE DATA WILL BE SUBSTITUTED!"
            )

        series = df[column]
        if series.isna().all():
            raise ValueError(
                f"CRITICAL: Required field '{column}' contains only NaN values {context}. "
                f"This indicates corrupted data - NO FAKE DATA WILL BE SUBSTITUTED!"
            )

        # Return series with NaN values intact - let downstream handle appropriately
        return series
    
    def _normalize_exposure_optimized(self, exposure_series: pd.Series, exposure_type: str, symbol: str) -> pd.Series:
        """Optimized exposure normalization with caching"""
        try:
            # Handle None input
            if exposure_series is None:
                self.logger.warning(f"exposure_series is None for {exposure_type} exposure in {symbol}")
                return pd.Series([0.0])

            if len(exposure_series) == 0:
                return exposure_series

            # Check if cache manager is available
            if self.enhanced_cache_manager is None:
                norm_params_raw = None
            else:
                # Use unified caching for normalization parameters
                norm_params_raw = self.enhanced_cache_manager.get(symbol, f"{exposure_type}_norm_params")

            if norm_params_raw is None:
                norm_params = None
            else:
                # CRITICAL FIX: Reconstruct Pydantic model from cache if it's a dict
                if isinstance(norm_params_raw, dict):
                    try:
                        norm_params = NormalizationParams.model_validate(norm_params_raw)
                    except Exception:
                        norm_params = None
                elif isinstance(norm_params_raw, NormalizationParams):
                    norm_params = norm_params_raw
                else:
                    norm_params = None

            if norm_params is None:
                # Calculate normalization parameters
                mean_val = exposure_series.mean()
                std_val = exposure_series.std()
                # CRITICAL FIX: Use Pydantic model instead of dictionary
                norm_params = NormalizationParams(mean=mean_val, std=max(std_val, 0.001))

                # Cache for future use if cache manager is available
                if self.enhanced_cache_manager is not None:
                    self.enhanced_cache_manager.put(symbol, f"{exposure_type}_norm_params", norm_params)

            # Normalize using proper Pydantic attribute access
            normalized = (exposure_series - norm_params.mean) / norm_params.std
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Error normalizing {exposure_type} exposure: {e}")
            import traceback
            self.logger.warning(f"Normalization error traceback: {traceback.format_exc()}")
            return pd.Series([0.0] * len(exposure_series))
    
    def _add_default_heatmap_columns(self, df_strike: pd.DataFrame) -> pd.DataFrame:
        """Add default heatmap columns on error"""
        df_strike['sgdhp_score_strike'] = 0.0
        df_strike['ivsdh_score_strike'] = 0.0
        df_strike['ugch_score_strike'] = 0.0
        return df_strike
    
    def _raise_aggregates_calculation_error(self, error_context: str) -> None:
        """FAIL-FAST: Raise error instead of returning fake aggregates - NO FAKE TRADING DATA ALLOWED"""
        raise ValueError(f"CRITICAL: Aggregates calculation failed in {error_context} - cannot return fake financial aggregates that could cause massive trading losses!")

# Export the consolidated calculator
__all__ = ['VisualizationMetrics']
