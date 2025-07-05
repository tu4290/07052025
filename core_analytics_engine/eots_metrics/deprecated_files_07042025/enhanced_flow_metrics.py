import logging
import numpy as np
from typing import Dict, Any

from core_analytics_engine.eots_metrics.base_calculator import BaseCalculator, EnhancedCacheManagerV2_5
from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, FlowType
from core_analytics_engine.eots_metrics.elite_flow_classifier import EliteFlowClassifier
from core_analytics_engine.eots_metrics.elite_momentum_detector import EliteMomentumDetector

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class EnhancedFlowMetricsCalculator(BaseCalculator):
    """
    Calculates Tier 3 Enhanced Rolling Flow Metrics: VAPI-FA, DWFD, TW-LAF.
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: EliteConfig):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config = elite_config
        self.flow_classifier = EliteFlowClassifier(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)
        self.momentum_detector = EliteMomentumDetector(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)

    def calculate_all_enhanced_flow_metrics(self, und_data: Dict, symbol: str) -> Dict:
        """Orchestrates the calculation of all enhanced flow metrics."""
        self.logger.debug(f"Calculating enhanced flow metrics for {symbol}...")
        try:
            # Classify Elite Flow Type
            flow_type_elite = self.flow_classifier.classify_flow_type(und_data)
            und_data['flow_type_elite'] = flow_type_elite.value

            # Calculate Momentum Acceleration Index
            momentum_acceleration_index_und = self.momentum_detector.calculate_momentum_acceleration_index(und_data)
            und_data['momentum_acceleration_index_und'] = momentum_acceleration_index_und

            und_data = self._calculate_vapi_fa(und_data, symbol)
            und_data = self._calculate_dwfd(und_data, symbol)
            und_data = self._calculate_tw_laf(und_data, symbol)
            self.logger.debug(f"Enhanced flow metrics calculation complete for {symbol}.")
            return und_data
        except Exception as e:
            self.logger.error(f"Error calculating enhanced flow metrics for {symbol}: {e}", exc_info=True)
            und_data['vapi_fa_z_score_und'] = 0.0
            und_data['dwfd_z_score_und'] = 0.0
            und_data['tw_laf_z_score_und'] = 0.0
            return und_data

    def _calculate_vapi_fa(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate VAPI-FA (Volume-Adjusted Premium Intensity with Flow Acceleration)."""
        try:
            self.logger.debug(f"VAPI-FA calculation for {symbol}")
            
            # Get isolated configuration parameters
            # z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20) # Not directly used here, but in _normalize_flow
            
            net_value_flow_5m = und_data.get('net_value_flow_5m_und', und_data.get('total_nvp', und_data.get('value_bs', 0.0))) or 0.0
            net_vol_flow_5m = und_data.get('net_vol_flow_5m_und', und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0))) or 0.0
            net_vol_flow_15m = und_data.get('net_vol_flow_15m_und', net_vol_flow_5m * 2.8) or (net_vol_flow_5m * 2.8)
            current_iv = und_data.get('u_volatility', und_data.get('Current_Underlying_IV', und_data.get('implied_volatility', 0.20))) or 0.20
            
            net_value_flow_5m = float(net_value_flow_5m)
            net_vol_flow_5m = float(net_vol_flow_5m)
            net_vol_flow_15m = float(net_vol_flow_15m)
            current_iv = float(current_iv)
            
            if abs(net_vol_flow_5m) > 0.001:
                pvr_5m = net_value_flow_5m / net_vol_flow_5m
            else:
                pvr_5m = 0.0
            
            volatility_adjusted_pvr_5m = pvr_5m * current_iv
            
            flow_in_prior_5_to_10_min = (net_vol_flow_15m - net_vol_flow_5m) / 2.0
            flow_acceleration_5m = net_vol_flow_5m - flow_in_prior_5_to_10_min
            
            vapi_fa_raw = volatility_adjusted_pvr_5m * flow_acceleration_5m
            
            vapi_fa_cache = self._add_to_intraday_cache(symbol, 'vapi_fa', float(vapi_fa_raw), max_size=200)

            if len(vapi_fa_cache) >= 10:
                vapi_fa_mean = np.mean(vapi_fa_cache)
                vapi_fa_std = np.std(vapi_fa_cache)
                vapi_fa_z_score = (vapi_fa_raw - vapi_fa_mean) / max(float(vapi_fa_std), 0.001)
            else:
                vapi_fa_z_score = self._calculate_percentile_gauge_value(vapi_fa_cache, float(vapi_fa_raw))
            
            und_data['vapi_fa_raw_und'] = vapi_fa_raw
            und_data['vapi_fa_z_score_und'] = vapi_fa_z_score
            und_data['vapi_fa_pvr_5m_und'] = pvr_5m
            und_data['vapi_fa_flow_accel_5m_und'] = flow_acceleration_5m
            
            self.logger.debug(f"VAPI-FA results for {symbol}: raw={vapi_fa_raw:.2f}, z_score={vapi_fa_z_score:.2f}, intraday_cache_size={len(vapi_fa_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating VAPI-FA for {symbol}: {e}", exc_info=True)
            und_data['vapi_fa_raw_und'] = 0.0
            und_data['vapi_fa_z_score_und'] = 0.0
            und_data['vapi_fa_pvr_5m_und'] = 0.0
            und_data['vapi_fa_flow_accel_5m_und'] = 0.0
            return und_data
    
    def _calculate_dwfd(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate DWFD (Delta-Weighted Flow Divergence)."""
        try:
            # z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20)
            
            net_value_flow = und_data.get('total_nvp', und_data.get('value_bs', 0.0)) or 0.0
            net_vol_flow = und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0)) or 0.0
            
            net_value_flow = float(net_value_flow)
            net_vol_flow = float(net_vol_flow)
            
            directional_delta_flow = net_vol_flow
            
            value_cache = self._add_to_intraday_cache(symbol, 'net_value_flow', net_value_flow, max_size=200)
            vol_cache = self._add_to_intraday_cache(symbol, 'net_vol_flow', net_vol_flow, max_size=200)
            
            if len(value_cache) >= 10:
                value_mean = np.mean(value_cache)
                value_std = np.std(value_cache)
                value_z = (net_value_flow - value_mean) / max(float(value_std), 0.001)
            else:
                value_z = 0.0
            
            if len(vol_cache) >= 10:
                vol_mean = np.mean(vol_cache)
                vol_std = np.std(vol_cache)
                vol_z = (net_vol_flow - vol_mean) / max(float(vol_std), 0.001)
            else:
                vol_z = 0.0
            
            fvd = value_z - vol_z
            
            weight_factor = 0.5
            dwfd_raw = directional_delta_flow - (weight_factor * fvd)
            
            dwfd_cache = self._add_to_intraday_cache(symbol, 'dwfd', float(dwfd_raw), max_size=200)
            dwfd_z_score = self._calculate_percentile_gauge_value(dwfd_cache, float(dwfd_raw))
            
            und_data['dwfd_raw_und'] = dwfd_raw
            und_data['dwfd_z_score_und'] = dwfd_z_score
            und_data['dwfd_fvd_und'] = fvd
            
            self.logger.debug(f"DWFD results for {symbol}: raw={dwfd_raw:.2f}, z_score={dwfd_z_score:.2f}, fvd={fvd:.2f}, intraday_cache_size={len(dwfd_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating DWFD for {symbol}: {e}", exc_info=True)
            und_data['dwfd_raw_und'] = 0.0
            und_data['dwfd_z_score_und'] = 0.0
            und_data['dwfd_fvd_und'] = 0.0
            return und_data
    
    def _calculate_tw_laf(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate TW-LAF (Time-Weighted Liquidity-Adjusted Flow)."""
        try:
            # z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20)
            
            net_vol_flow_5m = und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0)) or 0.0
            net_vol_flow_15m = net_vol_flow_5m * 2.5
            net_vol_flow_30m = net_vol_flow_5m * 4.0
            
            underlying_price = und_data.get('price', 100.0) or 100.0
            
            net_vol_flow_5m = float(net_vol_flow_5m)
            net_vol_flow_15m = float(net_vol_flow_15m)
            net_vol_flow_30m = float(net_vol_flow_30m)
            underlying_price = float(underlying_price)
            
            base_spread_pct = 0.02
            normalized_spread_5m = base_spread_pct * 1.0
            normalized_spread_15m = base_spread_pct * 1.2
            normalized_spread_30m = base_spread_pct * 1.5
            
            liquidity_factor_5m = 1.0 / (normalized_spread_5m + 0.001)
            liquidity_factor_15m = 1.0 / (normalized_spread_15m + 0.001)
            liquidity_factor_30m = 1.0 / (normalized_spread_30m + 0.001)
            
            liquidity_adjusted_flow_5m = net_vol_flow_5m * liquidity_factor_5m
            liquidity_adjusted_flow_15m = net_vol_flow_15m * liquidity_factor_15m
            liquidity_adjusted_flow_30m = net_vol_flow_30m * liquidity_factor_30m
            
            weight_5m = 1.0
            weight_15m = 0.8
            weight_30m = 0.6
            
            tw_laf_raw = (weight_5m * liquidity_adjusted_flow_5m +
                         weight_15m * liquidity_adjusted_flow_15m +
                         weight_30m * liquidity_adjusted_flow_30m)
            
            tw_laf_cache = self._add_to_intraday_cache(symbol, 'tw_laf', float(tw_laf_raw), max_size=200)
            tw_laf_z_score = self._calculate_percentile_gauge_value(tw_laf_cache, float(tw_laf_raw))
            
            und_data['tw_laf_raw_und'] = tw_laf_raw
            und_data['tw_laf_z_score_und'] = tw_laf_z_score
            und_data['tw_laf_liquidity_factor_5m_und'] = liquidity_factor_5m
            und_data['tw_laf_time_weighted_sum_und'] = tw_laf_raw
            
            self.logger.debug(f"TW-LAF results for {symbol}: raw={tw_laf_raw:.2f}, z_score={tw_laf_z_score:.2f}, intraday_cache_size={len(tw_laf_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating TW-LAF for {symbol}: {e}", exc_info=True)
            und_data['tw_laf_raw_und'] = 0.0
            und_data['tw_laf_z_score_und'] = 0.0
            und_data['tw_laf_liquidity_factor_5m_und'] = 1.0
            und_data['tw_laf_time_weighted_sum_und'] = 0.0
            return und_data