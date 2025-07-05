import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from core_analytics_engine.eots_metrics.base_calculator import BaseCalculator, EnhancedCacheManagerV2_5
from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, MarketRegime, FlowType
from core_analytics_engine.eots_metrics.elite_regime_detector import EliteMarketRegimeDetector
from core_analytics_engine.eots_metrics.elite_volatility_surface import EliteVolatilitySurface
from core_analytics_engine.eots_metrics.elite_momentum_detector import EliteMomentumDetector
from core_analytics_engine.eots_metrics.elite_flow_classifier import EliteFlowClassifier

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class AdaptiveMetricsCalculator(BaseCalculator):
    """
    Calculates Tier 2 Adaptive Metrics: A-DAG, E-SDAG, D-TDPI, VRI 2.0.
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: EliteConfig):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config = elite_config
        self.regime_detector = EliteMarketRegimeDetector(elite_config)
        self.volatility_surface_analyzer = EliteVolatilitySurface(elite_config)
        self.momentum_detector = EliteMomentumDetector(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)
        self.flow_classifier = EliteFlowClassifier(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)

    def calculate_all_adaptive_metrics(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """
        Orchestrates the calculation of all adaptive metrics.
        """
        if len(df_strike) == 0:
            return df_strike
            
        self.logger.debug("Calculating adaptive metrics...")
        try:
            # Determine Elite Market Regime
            current_market_regime = self.regime_detector.determine_market_regime(und_data, df_strike)
            und_data['current_market_regime'] = current_market_regime.value

            # Determine Elite Volatility Regime
            volatility_regime_elite = self.volatility_surface_analyzer.determine_volatility_regime(und_data, df_strike)
            und_data['volatility_regime_elite'] = volatility_regime_elite.value

            # Calculate Momentum Acceleration Index
            momentum_acceleration_index_und = self.momentum_detector.calculate_momentum_acceleration_index(und_data)
            und_data['momentum_acceleration_index_und'] = momentum_acceleration_index_und

            # Classify Elite Flow Type
            flow_type_elite = self.flow_classifier.classify_flow_type(und_data)
            und_data['flow_type_elite'] = flow_type_elite.value

            volatility_context = self._get_volatility_context(und_data)
            dte_context = self._get_average_dte_context(df_strike)
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            
            df_strike = self._calculate_a_dag(df_strike, und_data, current_market_regime, volatility_context, dte_context)
            df_strike = self._calculate_e_sdag(df_strike, und_data, current_market_regime, volatility_context, dte_context)
            df_strike = self._calculate_d_tdpi(df_strike, und_data, current_market_regime, volatility_context, dte_context)
            df_strike = self._calculate_vri_2_0(df_strike, und_data, current_market_regime, volatility_context, dte_context)
            df_strike = self._calculate_concentration_indices(df_strike, und_data)
            df_strike = self._calculate_0dte_suite(df_strike, und_data, dte_context)
            
            self.logger.debug(f"Adaptive metrics calculation complete for {len(df_strike)} strikes.")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive metrics: {e}", exc_info=True)
            return df_strike

    def _get_volatility_context(self, und_data: Dict) -> str:
        """
        Determine volatility context for adaptive calculations.
        """
        current_iv = und_data.get('u_volatility', 0.20)
        if current_iv > 0.30:
            return 'HIGH_VOL'
        elif current_iv < 0.15:
            return 'LOW_VOL'
        else:
            return 'NORMAL_VOL'

    def _get_market_direction_bias(self, und_data: Dict) -> float:
        """
        Determine market direction bias for VRI calculations to prevent contradictory regime classifications.
        """
        try:
            current_price = und_data.get('price', 0.0)
            price_change_pct = und_data.get('price_change_pct', 0.0)
            vix_spy_divergence = und_data.get('vix_spy_price_divergence_strong_negative', False)
            spy_trend = und_data.get('spy_trend', 'sideways')

            direction_bias = 0.0
            if abs(price_change_pct) > 0.005:
                direction_bias = np.sign(price_change_pct) * min(abs(price_change_pct) * 20, 1.0)

            if vix_spy_divergence:
                direction_bias = min(direction_bias - 0.3, -0.2)

            if spy_trend == 'up':
                direction_bias = max(direction_bias, 0.2)
            elif spy_trend == 'down':
                direction_bias = min(direction_bias, -0.2)

            direction_bias = max(-1.0, min(1.0, direction_bias))
            return direction_bias
        except Exception as e:
            self.logger.warning(f"Error calculating market direction bias: {e}")
            return 0.0
    
    def _get_average_dte_context(self, df_strike: pd.DataFrame) -> str:
        """
        Determine DTE context for adaptive calculations.
        This is a placeholder - in real implementation would calculate from options data.
        """
        return 'NORMAL_DTE'
    
    def _calculate_a_dag(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """
        Calculate A-DAG (Adaptive Delta-Adjusted Gamma Exposure) per system guide.
        """
        try:
            a_dag_config = self._get_metric_config('adaptive_metric_parameters', 'a_dag_settings', {})
            base_alpha_coeffs = a_dag_config.get('base_dag_alpha_coeffs', {
                'aligned': 1.35, 'opposed': 0.65, 'neutral': 1.0
            })

            regime_multipliers = a_dag_config.get('regime_alpha_multipliers', {})
            volatility_multipliers = a_dag_config.get('volatility_context_alpha_multipliers', {})
            flow_sensitivity = a_dag_config.get('flow_sensitivity_by_regime', {})
            dte_scaling_config = a_dag_config.get('dte_gamma_flow_impact_scaling', {})

            regime_mult_data = regime_multipliers.get(market_regime, {'aligned_mult': 1.0, 'opposed_mult': 1.0})
            vol_mult_data = volatility_multipliers.get(volatility_context, {'aligned_mult': 1.0, 'opposed_mult': 1.0})

            adaptive_alpha_aligned = (base_alpha_coeffs['aligned'] *
                                    regime_mult_data.get('aligned_mult', 1.0) *
                                    vol_mult_data.get('aligned_mult', 1.0))
            adaptive_alpha_opposed = (base_alpha_coeffs['opposed'] *
                                    regime_mult_data.get('opposed_mult', 1.0) *
                                    vol_mult_data.get('opposed_mult', 1.0))
            adaptive_alpha_neutral = base_alpha_coeffs['neutral']

            gxoi_at_strike = df_strike.get('total_gxoi_at_strike', 0)
            dxoi_at_strike = df_strike.get('total_dxoi_at_strike', 0)
            net_cust_delta_flow = df_strike.get('net_cust_delta_flow_at_strike', 0)
            net_cust_gamma_flow = df_strike.get('net_cust_gamma_flow_at_strike_proxy', 0)
            
            delta_alignment = np.sign(dxoi_at_strike) * np.sign(net_cust_delta_flow)
            gamma_alignment = np.sign(gxoi_at_strike) * np.sign(net_cust_gamma_flow)
            
            combined_alignment = (delta_alignment + gamma_alignment) / 2.0
            
            alignment_type = np.where(
                combined_alignment > 0.3, 'aligned',
                np.where(combined_alignment < -0.3, 'opposed', 'neutral')
            )
            
            adaptive_alpha = np.where(
                alignment_type == 'aligned',
                adaptive_alpha_aligned,
                np.where(
                    alignment_type == 'opposed',
                    adaptive_alpha_opposed,
                    adaptive_alpha_neutral
                )
            )
             
            dte_scaling = self._get_dte_scaling_factor(dte_context)
            
            flow_alignment_ratio = np.where(
                np.abs(gxoi_at_strike) > 0,
                (net_cust_delta_flow + net_cust_gamma_flow) / (np.abs(gxoi_at_strike) + 1e-6),
                0.0
            )
            
            current_price = und_data.get('price', 0.0)
            strikes = df_strike['strike'] if 'strike' in df_strike.columns else pd.Series([current_price] * len(df_strike))
            
            directional_multiplier = np.where(strikes > current_price, -1, 1)
            
            a_dag_exposure = gxoi_at_strike * directional_multiplier * (1 + adaptive_alpha * flow_alignment_ratio) * dte_scaling
            
            use_volume_weighted = self._get_metric_config('adaptive', 'use_volume_weighted_gxoi', False)
            if use_volume_weighted:
                volume_weight = df_strike.get('total_volume_at_strike', 1.0)
                if not isinstance(volume_weight, pd.Series):
                    volume_weight = pd.Series([volume_weight] * len(df_strike))
                volume_factor = np.log1p(volume_weight) / np.log1p(volume_weight.mean() + 1e-6)
                a_dag_exposure *= volume_factor
            
            df_strike['a_dag_exposure'] = a_dag_exposure
            df_strike['a_dag_adaptive_alpha'] = adaptive_alpha
            df_strike['a_dag_flow_alignment'] = flow_alignment_ratio
            df_strike['a_dag_directional_multiplier'] = directional_multiplier
            df_strike['a_dag_strike'] = df_strike['a_dag_exposure']

            self.logger.debug(f"[A-DAG] Calculated for {len(df_strike)} strikes")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"[A-DAG] CRITICAL ERROR in A-DAG calculation: {e}", exc_info=True)
            df_strike['a_dag_exposure'] = 0.0
            df_strike['a_dag_adaptive_alpha'] = 0.0
            df_strike['a_dag_flow_alignment'] = 0.0
            df_strike['a_dag_directional_multiplier'] = 0.0
            df_strike['a_dag_strike'] = 0.0
            return df_strike
    
    def _calculate_e_sdag(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """
        Calculate E-SDAG (Enhanced Skew and Delta Adjusted Gamma Exposure) per system guide.
        """
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            e_sdag_config = self._get_metric_config('adaptive_metric_parameters', 'e_sdag_settings', {})
            use_enhanced_sgexoi = e_sdag_config.get('use_enhanced_skew_calculation_for_sgexoi', True)
            sgexoi_params = e_sdag_config.get('sgexoi_calculation_params', {})
            base_delta_weights = e_sdag_config.get('base_delta_weight_factors', {
                'e_sdag_mult': 0.5, 'e_sdag_dir': 0.6, 'e_sdag_w': 0.4, 'e_sdag_vf': 0.7
            })
            regime_multipliers = e_sdag_config.get('regime_delta_weight_multipliers', {})
            volatility_multipliers = e_sdag_config.get('volatility_delta_weight_multipliers', {})

            gxoi_at_strike = df_strike.get('total_gxoi_at_strike', 0)
            if use_enhanced_sgexoi:
                sgexoi_v2_5 = self._calculate_sgexoi_v2_5(gxoi_at_strike, und_data, sgexoi_params, dte_context)
            else:
                sgexoi_v2_5 = gxoi_at_strike

            dxoi_at_strike = df_strike.get('total_dxoi_at_strike', 0)
            dxoi_normalized = self._normalize_flow(dxoi_at_strike, 'dxoi', symbol)

            regime_mult_data = regime_multipliers.get(market_regime, {})
            vol_mult_data = volatility_multipliers.get(volatility_context, {})

            adaptive_delta_weights = {}
            for methodology in ['e_sdag_mult', 'e_sdag_dir', 'e_sdag_w', 'e_sdag_vf']:
                base_weight = base_delta_weights.get(methodology, 0.5)
                regime_mult = regime_mult_data.get(methodology, 1.0)
                vol_mult = vol_mult_data.get(methodology, 1.0)
                adaptive_delta_weights[methodology] = base_weight * regime_mult * vol_mult

            e_sdag_mult = sgexoi_v2_5 * (1 + adaptive_delta_weights['e_sdag_mult'] * dxoi_normalized)
            e_sdag_dir = sgexoi_v2_5 + (adaptive_delta_weights['e_sdag_dir'] * dxoi_at_strike)
            gamma_weight = 1.0 - adaptive_delta_weights['e_sdag_w']
            delta_weight = adaptive_delta_weights['e_sdag_w']
            e_sdag_w = gamma_weight * sgexoi_v2_5 + delta_weight * np.abs(dxoi_at_strike)
            e_sdag_vf = sgexoi_v2_5 * (1 - adaptive_delta_weights['e_sdag_vf'] * dxoi_normalized)

            df_strike['e_sdag_mult_strike'] = e_sdag_mult
            df_strike['e_sdag_dir_strike'] = e_sdag_dir
            df_strike['e_sdag_w_strike'] = e_sdag_w
            df_strike['e_sdag_vf_strike'] = e_sdag_vf

            df_strike['e_sdag_adaptive_delta_weight_mult'] = adaptive_delta_weights['e_sdag_mult']
            df_strike['e_sdag_adaptive_delta_weight_dir'] = adaptive_delta_weights['e_sdag_dir']
            df_strike['e_sdag_adaptive_delta_weight_w'] = adaptive_delta_weights['e_sdag_w']
            df_strike['e_sdag_adaptive_delta_weight_vf'] = adaptive_delta_weights['e_sdag_vf']

            self.logger.debug(f"[E-SDAG] Calculated for {len(df_strike)} strikes with adaptive weights: "
                            f"mult={adaptive_delta_weights['e_sdag_mult']:.3f}, "
                            f"dir={adaptive_delta_weights['e_sdag_dir']:.3f}, "
                            f"w={adaptive_delta_weights['e_sdag_w']:.3f}, "
                            f"vf={adaptive_delta_weights['e_sdag_vf']:.3f}")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating E-SDAG: {e}", exc_info=True)
            df_strike['e_sdag_mult_strike'] = 0.0
            df_strike['e_sdag_dir_strike'] = 0.0
            df_strike['e_sdag_w_strike'] = 0.0
            df_strike['e_sdag_vf_strike'] = 0.0
            return df_strike
    
    def _calculate_d_tdpi(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """
        Calculate D-TDPI (Dynamic Time Decay Pressure Indicator), E-CTR (Effective Call/Put Trade Ratio), and E-TDFI (Effective Time Decay Flow Index).
        """
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            charm_oi = df_strike.get('total_charmxoi_at_strike', 0)
            theta_oi = df_strike.get('total_txoi_at_strike', 0)
            net_cust_theta_flow = df_strike.get('net_cust_theta_flow_at_strike', 0)

            theta_flow_normalized = self._normalize_flow(net_cust_theta_flow, 'theta_flow', symbol)

            d_tdpi_value = charm_oi * np.sign(theta_oi) * (1 + theta_flow_normalized * 0.4)

            try:
                net_charm_flow_proxy = df_strike.get('net_cust_charm_flow_at_strike_proxy', pd.Series([0.0] * len(df_strike)))

                e_ctr_numerator = np.abs(net_charm_flow_proxy)
                e_ctr_denominator = np.abs(net_cust_theta_flow) + 1e-9
                e_ctr_value = e_ctr_numerator / e_ctr_denominator

            except Exception as e:
                self.logger.error(f"Error calculating E-CTR: {e}")
                e_ctr_value = pd.Series([0.0] * len(df_strike))

            try:
                theta_oi_at_strike = df_strike.get('total_txoi_at_strike', pd.Series([0.0] * len(df_strike)))

                theta_flow_abs = np.abs(net_cust_theta_flow)
                theta_oi_abs = np.abs(theta_oi_at_strike)

                theta_flow_max = np.maximum(theta_flow_abs.max(), 1e-9)
                theta_oi_max = np.maximum(theta_oi_abs.max(), 1e-9)

                theta_flow_normalized = theta_flow_abs / theta_flow_max
                theta_oi_normalized = theta_oi_abs / theta_oi_max

                e_tdfi_value = theta_flow_normalized / (theta_oi_normalized + 1e-9)

            except Exception as e:
                self.logger.error(f"Error calculating E-TDFI: {e}")
                e_tdfi_value = pd.Series([0.0] * len(df_strike))

            df_strike['d_tdpi_strike'] = d_tdpi_value
            df_strike['e_ctr_strike'] = e_ctr_value
            df_strike['e_tdfi_strike'] = e_tdfi_value

            self.logger.debug(f"[TIME DECAY SUITE] Calculated D-TDPI, E-CTR, E-TDFI for {len(df_strike)} strikes")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating time decay metrics (D-TDPI, E-CTR, E-TDFI): {e}", exc_info=True)
            df_strike['d_tdpi_strike'] = 0.0
            df_strike['e_ctr_strike'] = 0.0
            df_strike['e_tdfi_strike'] = 0.0
            return df_strike

    def _calculate_concentration_indices(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """
        Calculate GCI (Gamma Concentration Index) and DCI (Delta Concentration Index) at strike level.
        """
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            try:
                total_gxoi_at_strike = df_strike.get('total_gxoi_at_strike', pd.Series([0.0] * len(df_strike)))
                total_gxoi_underlying = total_gxoi_at_strike.abs().sum()

                if total_gxoi_underlying > 1e-9:
                    proportion_gamma_oi = total_gxoi_at_strike.abs() / total_gxoi_underlying
                    gci_strike = proportion_gamma_oi ** 2
                else:
                    gci_strike = pd.Series([0.0] * len(df_strike))

            except Exception as e:
                self.logger.error(f"Error calculating GCI: {e}")
                gci_strike = pd.Series([0.0] * len(df_strike))

            try:
                total_dxoi_at_strike = df_strike.get('total_dxoi_at_strike', pd.Series([0.0] * len(df_strike)))
                total_dxoi_underlying = total_dxoi_at_strike.abs().sum()

                if total_dxoi_underlying > 1e-9:
                    proportion_delta_oi = total_dxoi_at_strike.abs() / total_dxoi_underlying
                    dci_strike = proportion_delta_oi ** 2
                else:
                    dci_strike = pd.Series([0.0] * len(df_strike))

            except Exception as e:
                self.logger.error(f"Error calculating DCI: {e}")
                dci_strike = pd.Series([0.0] * len(df_strike))

            df_strike['gci_strike'] = gci_strike
            df_strike['dci_strike'] = dci_strike

            self.logger.debug(f"[CONCENTRATION INDICES] Calculated GCI and DCI for {len(df_strike)} strikes")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating concentration indices (GCI, DCI): {e}", exc_info=True)
            df_strike['gci_strike'] = 0.0
            df_strike['dci_strike'] = 0.0
            return df_strike
    
    def _calculate_vri_2_0(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """
        Calculate VRI 2.0 (Volatility Regime Indicator 2.0) - Canonical EOTS v2.5 implementation.
        """
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            vri_cfg = self.config_manager.get_setting("adaptive_metric_parameters.vri_2_0_settings", default={})
            base_gamma_coeffs = vri_cfg.get("base_vri_gamma_coeffs", {"aligned": 1.3, "opposed": 0.7, "neutral": 1.0})
            
            required_cols = [
                'total_vanna_at_strike', 'total_vxoi_at_strike', 'total_vommaxoi_at_strike',
                'net_cust_vanna_flow_proxy_at_strike', 'net_cust_vomma_flow_proxy_at_strike'
            ]
            for col in required_cols:
                if col not in df_strike.columns:
                    self.logger.warning(f"[VRI2.0] Missing required column: {col}. Filling with 0.")
                    df_strike[col] = 0.0
                df_strike[col] = pd.to_numeric(df_strike[col], errors='coerce').fillna(0.0)
            
            current_iv = und_data.get('u_volatility', 0.20) or 0.20
            front_iv = und_data.get('front_month_iv', current_iv)
            spot_iv = und_data.get('spot_iv', current_iv)
            try:
                front_iv = float(front_iv)
            except Exception:
                front_iv = current_iv
            try:
                spot_iv = float(spot_iv)
            except Exception:
                spot_iv = current_iv
            term_structure_factor = front_iv / (spot_iv + EPSILON) if spot_iv else 1.0
            enhanced_vol_context_weight = und_data.get('ivsdh_vol_context_weight', 1.0)
            try:
                enhanced_vol_context_weight = float(enhanced_vol_context_weight)
            except Exception:
                enhanced_vol_context_weight = 1.0
            
            vega_oi = pd.to_numeric(df_strike['total_vxoi_at_strike'], errors='coerce').replace(0, EPSILON)
            vomma_oi = pd.to_numeric(df_strike['total_vommaxoi_at_strike'], errors='coerce')
            enhanced_vomma_factor = np.where(vega_oi != 0, vomma_oi / (vega_oi + EPSILON), 1.0)
            enhanced_vomma_factor = enhanced_vomma_factor.astype(float)
            
            def get_gamma_coeff(row):
                regime = (market_regime or "neutral").lower()
                dte = (dte_context or "normal").lower()
                if "bull" in regime:
                    return base_gamma_coeffs.get("aligned", 1.3)
                elif "bear" in regime:
                    return base_gamma_coeffs.get("opposed", 0.7)
                else:
                    return base_gamma_coeffs.get("neutral", 1.0)
            gamma_coeffs = df_strike.apply(get_gamma_coeff, axis=1)
            if not isinstance(gamma_coeffs, pd.Series):
                gamma_coeffs = pd.Series([gamma_coeffs] * len(df_strike))
            
            vanna_oi = pd.to_numeric(df_strike['total_vanna_at_strike'], errors='coerce').replace(0, EPSILON)
            vanna_flow = pd.to_numeric(df_strike['net_cust_vanna_flow_proxy_at_strike'], errors='coerce')
            if not isinstance(vanna_flow, pd.Series) or len(vanna_flow) != len(df_strike):
                vanna_flow = pd.Series([0.0] * len(df_strike))
            vanna_flow_ratio = vanna_flow / (vanna_oi + EPSILON)
            if not isinstance(vanna_flow_ratio, pd.Series) or len(vanna_flow_ratio) != len(df_strike):
                vanna_flow_ratio = pd.Series([0.0] * len(df_strike))
            if len(vanna_flow_ratio) > 0:
                vanna_flow_ratio_mean = vanna_flow_ratio.mean()
                vanna_flow_ratio_std = vanna_flow_ratio.std() + EPSILON
                vanna_flow_ratio_norm = (vanna_flow_ratio - vanna_flow_ratio_mean) / vanna_flow_ratio_std
            else:
                vanna_flow_ratio_norm = pd.Series([0.0] * len(df_strike))
            vomma_oi = pd.to_numeric(df_strike['total_vommaxoi_at_strike'], errors='coerce').replace(0, EPSILON)
            vomma_flow = pd.to_numeric(df_strike['net_cust_vomma_flow_proxy_at_strike'], errors='coerce')
            if not isinstance(vomma_flow, pd.Series) or len(vomma_flow) != len(df_strike):
                vomma_flow = pd.Series([0.0] * len(df_strike))
            vomma_flow_ratio = vomma_flow / (vomma_oi + EPSILON)
            if not isinstance(vomma_flow_ratio, pd.Series) or len(vomma_flow_ratio) != len(df_strike):
                vomma_flow_ratio = pd.Series([0.0] * len(df_strike))
            if len(vomma_flow_ratio) > 0:
                vomma_flow_ratio_mean = vomma_flow_ratio.mean()
                vomma_flow_ratio_std = vomma_flow_ratio.std() + EPSILON
                vomma_flow_ratio_norm = (vomma_flow_ratio - vomma_flow_ratio_mean) / vomma_flow_ratio_std
            else:
                vomma_flow_ratio_norm = pd.Series([0.0] * len(df_strike))
            vega_oi = pd.to_numeric(df_strike['total_vxoi_at_strike'], errors='coerce')
            sign_vega_oi = np.sign(vega_oi)
            
            if not isinstance(vanna_oi, pd.Series):
                vanna_oi = pd.Series([vanna_oi] * len(df_strike))
            vanna_oi = pd.to_numeric(vanna_oi, errors='coerce').fillna(0).to_numpy(dtype=float, na_value=0.0)
            if not isinstance(sign_vega_oi, pd.Series):
                sign_vega_oi = pd.Series([sign_vega_oi] * len(df_strike))
            sign_vega_oi = pd.to_numeric(sign_vega_oi, errors='coerce').fillna(0).to_numpy(dtype=float, na_value=0.0)
            vanna_flow = gamma_coeffs * vanna_flow_ratio_norm.astype(float).to_numpy(dtype=float, na_value=0.0)
            one_plus_vanna_flow = 1 + vanna_flow
            vomma_flow = vomma_flow_ratio_norm.astype(float).to_numpy(dtype=float, na_value=0.0)
            vol_context = np.full(len(df_strike), float(enhanced_vol_context_weight))
            vomma_factor = np.array(enhanced_vomma_factor, dtype=float)
            term_factor = np.full(len(df_strike), float(term_structure_factor))
            vri_2_0 = (
                vanna_oi * sign_vega_oi * one_plus_vanna_flow * vomma_flow * vol_context * vomma_factor * term_factor
            )
            df_strike['vri_2_0_strike'] = vri_2_0
            
            self.logger.debug(f"[VRI2.0] Calculated for {len(df_strike)} strikes")
            return df_strike
        except Exception as e:
            self.logger.error(f"Error calculating VRI 2.0: {e}", exc_info=True)
            df_strike['vri_2_0_strike'] = 0.0
            return df_strike
    
    def _calculate_0dte_suite(self, df_strike: pd.DataFrame, und_data: Dict, dte_context: str) -> pd.DataFrame:
        is_0dte = (df_strike['dte_calc'] < 0.5) if 'dte_calc' in df_strike.columns else pd.Series([False]*len(df_strike))
        
        try:
            vannaxoi = df_strike.get('vannaxoi', pd.Series([0.0]*len(df_strike)))
            vxoi = df_strike.get('vxoi', pd.Series([0.0]*len(df_strike)))
            vannaxvolm = df_strike.get('vannaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxvolm = df_strike.get('vommaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxoi = df_strike.get('vommaxoi', pd.Series([0.0]*len(df_strike)))
            
            gamma_align_coeff = und_data.get('gamma_align_coeff', 1.0)
            skew_factor_global = und_data.get('skew_factor_global', 1.0)
            vol_trend_factor_global = und_data.get('vol_trend_factor_global', 1.0)
            max_abs_vomma_flow = np.maximum(np.abs(vommaxvolm).max(), EPSILON)
            vri_0dte_contract = (
                vannaxoi * np.sign(vxoi) *
                (1 + gamma_align_coeff * np.abs(vannaxvolm / (vannaxoi + EPSILON))) *
                (vommaxvolm / (max_abs_vomma_flow + EPSILON)) *
                skew_factor_global *
                vol_trend_factor_global
            )
            df_strike['vri_0dte'] = 0.0
            df_strike.loc[is_0dte, 'vri_0dte'] = vri_0dte_contract[is_0dte]
        except Exception as e:
            self.logger.error(f"Error calculating vri_0dte: {e}")
            df_strike['vri_0dte'] = 0.0

        try:
            vxoi = df_strike.get('vxoi', pd.Series([0.0]*len(df_strike)))
            vegas_buy = df_strike.get('vegas_buy', pd.Series([0.0]*len(df_strike)))
            vegas_sell = df_strike.get('vegas_sell', pd.Series([0.0]*len(df_strike)))

            net_cust_vega_flow_0dte = vegas_buy - vegas_sell
            if vegas_buy.sum() == 0 and vegas_sell.sum() == 0:
                net_cust_vega_flow_0dte = df_strike.get('vxvolm', pd.Series([0.0]*len(df_strike)))

            if is_0dte.sum() > 0:
                abs_net_cust_vega_flow_0dte = np.abs(net_cust_vega_flow_0dte[is_0dte])
                abs_vega_oi_0dte = np.abs(vxoi[is_0dte])

                max_abs_net_cust_vega_flow_0dte = abs_net_cust_vega_flow_0dte.max()
                max_abs_vega_oi_0dte = abs_vega_oi_0dte.max()

                if max_abs_net_cust_vega_flow_0dte > EPSILON:
                    normalized_abs_net_cust_vega_flow = abs_net_cust_vega_flow_0dte / max_abs_net_cust_vega_flow_0dte
                else:
                    normalized_abs_net_cust_vega_flow = pd.Series([0.0] * is_0dte.sum())

                if max_abs_vega_oi_0dte > EPSILON:
                    normalized_abs_vega_oi = abs_vega_oi_0dte / max_abs_vega_oi_0dte
                else:
                    normalized_abs_vega_oi = pd.Series([0.0] * is_0dte.sum())

                vfi_0dte_values = normalized_abs_net_cust_vega_flow / (normalized_abs_vega_oi + EPSILON)

                df_strike['vfi_0dte'] = 0.0
                df_strike.loc[is_0dte, 'vfi_0dte'] = vfi_0dte_values.values
            else:
                df_strike['vfi_0dte'] = 0.0
        except Exception as e:
            self.logger.error(f"Error calculating vfi_0dte: {e}")
            df_strike['vfi_0dte'] = 0.0

        try:
            vannaxvolm = df_strike.get('vannaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxvolm = df_strike.get('vommaxvolm', pd.Series([0.0]*len(df_strike)))

            abs_vanna_flow = np.abs(vannaxvolm)
            abs_vomma_flow = np.abs(vommaxvolm)

            vvr_0dte_contract = abs_vanna_flow / (abs_vomma_flow + EPSILON)

            vomma_zero_mask = abs_vomma_flow < EPSILON
            vanna_significant_mask = abs_vanna_flow > EPSILON
            extreme_vanna_dominance = vomma_zero_mask & vanna_significant_mask
            vvr_0dte_contract[extreme_vanna_dominance] = 1000.0

            both_zero_mask = (abs_vanna_flow < EPSILON) & (abs_vomma_flow < EPSILON)
            vvr_0dte_contract[both_zero_mask] = 0.0

            df_strike['vvr_0dte'] = 0.0
            df_strike.loc[is_0dte, 'vvr_0dte'] = vvr_0dte_contract[is_0dte]
        except Exception as e:
            self.logger.error(f"Error calculating vvr_0dte: {e}")
            df_strike['vvr_0dte'] = 0.0

        try:
            gci_strike = df_strike.get('gci_strike', pd.Series([0.0]*len(df_strike)))
            df_strike['gci_0dte'] = 0.0
            df_strike.loc[is_0dte, 'gci_0dte'] = gci_strike[is_0dte]
        except Exception as e:
            self.logger.error(f"Error calculating gci_0dte: {e}")
            df_strike['gci_0dte'] = 0.0

        try:
            dci_strike = df_strike.get('dci_strike', pd.Series([0.0]*len(df_strike)))
            df_strike['dci_0dte'] = 0.0
            df_strike.loc[is_0dte, 'dci_0dte'] = dci_strike[is_0dte]
        except Exception as e:
            self.logger.error(f"Error calculating dci_0dte: {e}")
            df_strike['dci_0dte'] = 0.0

        try:
            if is_0dte.sum() > 0:
                total_vanna_oi = df_strike.get('total_vannaxoi_at_strike', pd.Series([0.0]*len(df_strike))).fillna(0.0)
                vanna_oi_0dte = total_vanna_oi[is_0dte].abs()
                sum_total_abs_vanna_oi_0dte = vanna_oi_0dte.sum()

                if sum_total_abs_vanna_oi_0dte > EPSILON:
                    proportion_vanna_oi_0dte = vanna_oi_0dte / sum_total_abs_vanna_oi_0dte
                    vci_0dte_agg = (proportion_vanna_oi_0dte ** 2).sum()
                    vci_strike_proportions = proportion_vanna_oi_0dte ** 2

                    df_strike['vci_0dte_agg_value'] = 0.0
                    df_strike.loc[is_0dte.iloc[0:1].index, 'vci_0dte_agg_value'] = vci_0dte_agg
                    df_strike['vci_0dte'] = 0.0
                    df_strike.loc[is_0dte, 'vci_0dte'] = vci_strike_proportions.values
                else:
                    df_strike['vci_0dte_agg_value'] = 0.0
                    df_strike['vci_0dte'] = 0.0
            else:
                df_strike['vci_0dte_agg_value'] = 0.0
                df_strike['vci_0dte'] = 0.0
        except Exception as e:
            self.logger.error(f"Error calculating vci_0dte: {e}")
            df_strike['vci_0dte_agg_value'] = 0.0
            df_strike['vci_0dte'] = 0.0
        return df_strike

    def _calculate_sgexoi_v2_5(self, gxoi_at_strike: Union[pd.Series, np.ndarray], und_data: Dict, sgexoi_params: Dict, dte_context: str) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Enhanced Skew-Adjusted Gamma Exposure (SGEXOI_v2_5).
        """
        try:
            # Get skew adjustment parameters
            skew_sensitivity = sgexoi_params.get('skew_sensitivity', 0.3)
            term_structure_weight = sgexoi_params.get('term_structure_weight', 0.2)
            vri_integration_factor = sgexoi_params.get('vri_integration_factor', 0.15)

            # Get IV surface characteristics for skew adjustment
            current_iv = und_data.get('u_volatility', 0.20) or 0.20
            front_iv = und_data.get('front_month_iv', current_iv)
            spot_iv = und_data.get('spot_iv', current_iv)

            # Calculate term structure factor
            term_structure_factor = 1.0
            if spot_iv > 0:
                term_structure_factor = 1.0 + term_structure_weight * (front_iv / spot_iv - 1.0)

            # Calculate skew adjustment factor based on IV characteristics
            # This is a simplified implementation - full version would use detailed IV surface data
            skew_adjustment_factor = 1.0 + skew_sensitivity * (current_iv - 0.20)  # Adjust based on IV level

            # Apply VRI integration if available
            vri_factor = 1.0
            if 'vri_2_0_aggregate' in und_data:
                vri_aggregate = und_data.get('vri_2_0_aggregate', 0.0)
                vri_factor = 1.0 + vri_integration_factor * np.tanh(vri_aggregate)

            # Calculate SGEXOI_v2_5
            sgexoi_v2_5 = gxoi_at_strike * skew_adjustment_factor * term_structure_factor * vri_factor

            return sgexoi_v2_5

        except Exception as e:
            self.logger.error(f"Error calculating SGEXOI_v2_5: {e}")
            return gxoi_at_strike  # Fallback to regular GXOI
