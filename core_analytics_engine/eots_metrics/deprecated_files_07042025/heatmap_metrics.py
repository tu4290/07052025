import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from core_analytics_engine.eots_metrics.base_calculator import BaseCalculator

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class HeatmapMetricsCalculator(BaseCalculator):
    """
    Calculates Enhanced Heatmap Data (SGDHP, IVSDH, UGCH).
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: Any):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_all_heatmap_data(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """
        Orchestrates the calculation of all enhanced heatmap data.
        """
        self.logger.debug("Calculating enhanced heatmap data...")
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            
            gamma_exposure = df_strike['total_gxoi_at_strike'].fillna(0) if 'total_gxoi_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            delta_exposure = df_strike['total_dxoi_at_strike'].fillna(0) if 'total_dxoi_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            vanna_exposure = df_strike['total_vanna_at_strike'].fillna(0) if 'total_vanna_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            
            net_gamma_flow = df_strike['net_cust_gamma_flow_at_strike_proxy'].fillna(0) if 'net_cust_gamma_flow_at_strike_proxy' in df_strike.columns else pd.Series([0] * len(df_strike))
            net_delta_flow = df_strike['net_cust_delta_flow_at_strike'].fillna(0) if 'net_cust_delta_flow_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            net_vanna_flow = df_strike['net_cust_vanna_flow_at_strike_proxy'].fillna(0) if 'net_cust_vanna_flow_at_strike_proxy' in df_strike.columns else pd.Series([0] * len(df_strike))
            
            gamma_weight = 0.4
            delta_weight = 0.3
            vanna_weight = 0.2
            flow_weight = 0.1
            
            gamma_intensity = self._normalize_flow(gamma_exposure, 'gamma', symbol) * gamma_weight
            delta_intensity = self._normalize_flow(delta_exposure, 'delta', symbol) * delta_weight
            vanna_intensity = self._normalize_flow(vanna_exposure, 'vanna', symbol) * vanna_weight
            
            num_rows = len(df_strike)
            if len(gamma_intensity) == 1 and num_rows > 1:
                gamma_intensity = np.full(num_rows, gamma_intensity[0])
            if len(delta_intensity) == 1 and num_rows > 1:
                delta_intensity = np.full(num_rows, delta_intensity[0])
            if len(vanna_intensity) == 1 and num_rows > 1:
                vanna_intensity = np.full(num_rows, vanna_intensity[0])
            
            gamma_exposure_arr = pd.Series(pd.to_numeric(gamma_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_gamma_flow_arr = pd.Series(pd.to_numeric(net_gamma_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            delta_exposure_arr = pd.Series(pd.to_numeric(delta_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_delta_flow_arr = pd.Series(pd.to_numeric(net_delta_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            vanna_exposure_arr = pd.Series(pd.to_numeric(vanna_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_vanna_flow_arr = pd.Series(pd.to_numeric(net_vanna_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            
            flow_sum = (
                net_gamma_flow_arr +
                net_delta_flow_arr +
                net_vanna_flow_arr
            )
            flow_intensity = self._normalize_flow(flow_sum, 'combined_flow', str(symbol) if symbol is not None else "") * flow_weight
            
            composite_intensity = (
                gamma_intensity + delta_intensity + vanna_intensity + flow_intensity
            )
            
            df_strike['sgdhp_data'] = composite_intensity
            df_strike['ivsdh_data'] = vanna_intensity
            df_strike['ugch_data'] = delta_intensity
            
            df_strike = self._calculate_sgdhp_scores(df_strike, und_data)
            df_strike = self._calculate_ivsdh_scores(df_strike, und_data)
            df_strike = self._calculate_ugch_scores(df_strike, und_data)
            
            df_strike['heatmap_regime_scaling'] = 1.0
            df_strike['heatmap_gamma_component'] = gamma_intensity
            df_strike['heatmap_delta_component'] = delta_intensity
            df_strike['heatmap_vanna_component'] = vanna_intensity
            df_strike['heatmap_flow_component'] = flow_intensity
            
            self.logger.debug("Enhanced heatmap data calculation complete.")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced heatmap data: {e}", exc_info=True)
            df_strike['sgdhp_data'] = 0.0
            df_strike['ivsdh_data'] = 0.0
            df_strike['ugch_data'] = 0.0
            return df_strike

    def _calculate_sgdhp_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate SGDHP scores according to system guide specifications."""
        try:
            current_price = und_data.get('price', 0.0)
            if current_price <= 0:
                df_strike['sgdhp_score_strike'] = 0.0
                return df_strike
            
            gxoi_at_strike = df_strike['total_gxoi_at_strike'].fillna(0)
            dxoi_at_strike = df_strike['total_dxoi_at_strike'].fillna(0)
            strikes = df_strike['strike'].fillna(0)
            
            proximity_sensitivity = self._get_metric_config('heatmap_generation_settings', 'sgdhp_params.proximity_sensitivity_param', 0.05)
            price_proximity_factor = np.exp(-0.5 * ((strikes - current_price) / (current_price * proximity_sensitivity)) ** 2)

            max_abs_dxoi = dxoi_at_strike.abs().max()
            if max_abs_dxoi > 0:
                dxoi_normalized_impact = (1 + dxoi_at_strike.abs()) / (max_abs_dxoi + EPSILON)
            else:
                dxoi_normalized_impact = pd.Series([1.0] * len(df_strike))
            
            recent_flow_confirmation = pd.Series([0.1] * len(df_strike))
            
            sgdhp_scores = (
                (gxoi_at_strike * price_proximity_factor) * 
                np.sign(dxoi_at_strike) * 
                dxoi_normalized_impact * 
                (1 + recent_flow_confirmation)
            )
            
            df_strike['sgdhp_score_strike'] = sgdhp_scores
            
            self.logger.debug(f"SGDHP scores calculated: min={sgdhp_scores.min():.2f}, max={sgdhp_scores.max():.2f}, mean={sgdhp_scores.mean():.2f}")
            
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating SGDHP scores: {e}", exc_info=True)
            df_strike['sgdhp_score_strike'] = 0.0
            return df_strike

    def _calculate_ivsdh_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """
        Calculate IVSDH scores according to system guide specifications.
        Formula: ivsdh_value_contract = vanna_vomma_term * charm_impact_term
        Where:
        - vanna_vomma_term = (vannaxoi_contract * vommaxoi_contract) / (abs(vxoi_contract) + EPSILON)
        - charm_impact_term = (1 + (charmxoi_contract * dte_factor_for_charm))
        - dte_factor_for_charm = 1 / (1 + time_decay_sensitivity_param * dte_calc_contract)
        """
        try:
            vannaxoi_at_strike = df_strike['total_vannaxoi_at_strike'].fillna(0)
            vommaxoi_at_strike = df_strike['total_vommaxoi_at_strike'].fillna(0)
            vxoi_at_strike = df_strike['total_vxoi_at_strike'].fillna(0)
            charmxoi_at_strike = df_strike['total_charmxoi_at_strike'].fillna(0)

            if 'dte_calc' in df_strike.columns:
                dte_at_strike = df_strike['dte_calc'].fillna(30)
            else:
                dte_at_strike = pd.Series([30] * len(df_strike))

            time_decay_sensitivity = self._get_metric_config(
                'heatmap_generation_settings',
                'ivsdh_params.time_decay_sensitivity_param',
                0.1
            )

            vanna_vomma_term = (vannaxoi_at_strike * vommaxoi_at_strike) / (np.abs(vxoi_at_strike) + EPSILON)
            dte_factor_for_charm = 1.0 / (1.0 + time_decay_sensitivity * dte_at_strike)
            charm_impact_term = 1.0 + (charmxoi_at_strike * dte_factor_for_charm)
            ivsdh_scores = vanna_vomma_term * charm_impact_term

            df_strike['ivsdh_score_strike'] = ivsdh_scores

            self.logger.debug(f"IVSDH scores calculated: min={ivsdh_scores.min():.2f}, max={ivsdh_scores.max():.2f}, mean={ivsdh_scores.mean():.2f}")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating IVSDH scores: {e}", exc_info=True)
            df_strike['ivsdh_score_strike'] = 0.0
            return df_strike

    def _calculate_ugch_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate UGCH scores according to system guide specifications."""
        try:
            dxoi_at_strike = df_strike['total_dxoi_at_strike'].fillna(0)
            gxoi_at_strike = df_strike['total_gxoi_at_strike'].fillna(0)
            vxoi_at_strike = df_strike['total_vxoi_at_strike'].fillna(0)
            txoi_at_strike = df_strike['total_txoi_at_strike'].fillna(0)
            charm_at_strike = df_strike['total_charmxoi_at_strike'].fillna(0)
            vanna_at_strike = df_strike['total_vanna_at_strike'].fillna(0)
            
            def normalize_series(series):
                if series.std() > 0:
                    return (series - series.mean()) / series.std()
                else:
                    return pd.Series([0.0] * len(series))
            
            norm_dxoi = normalize_series(dxoi_at_strike)
            norm_gxoi = normalize_series(gxoi_at_strike)
            norm_vxoi = normalize_series(vxoi_at_strike)
            norm_txoi = normalize_series(txoi_at_strike)
            norm_charm = normalize_series(charm_at_strike)
            norm_vanna = normalize_series(vanna_at_strike)
            
            greek_weights = self._get_metric_config('heatmap_generation_settings', 'ugch_params.greek_weights', {
                'norm_DXOI': 1.5,
                'norm_GXOI': 2.0,
                'norm_VXOI': 1.2,
                'norm_TXOI': 0.8,
                'norm_CHARM': 0.6,
                'norm_VANNA': 1.0
            })
            
            ugch_scores = (
                greek_weights.get('norm_DXOI', 1.5) * norm_dxoi +
                greek_weights.get('norm_GXOI', 2.0) * norm_gxoi +
                greek_weights.get('norm_VXOI', 1.2) * norm_vxoi +
                greek_weights.get('norm_TXOI', 0.8) * norm_txoi +
                greek_weights.get('norm_CHARM', 0.6) * norm_charm +
                greek_weights.get('norm_VANNA', 1.0) * norm_vanna
            )
            
            df_strike['ugch_score_strike'] = ugch_scores
            
            self.logger.debug(f"UGCH scores calculated: min={ugch_scores.min():.2f}, max={ugch_scores.max():.2f}, mean={ugch_scores.mean():.2f}")
            
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating UGCH scores: {e}", exc_info=True)
            df_strike['ugch_score_strike'] = 0.0
            return df_strike
