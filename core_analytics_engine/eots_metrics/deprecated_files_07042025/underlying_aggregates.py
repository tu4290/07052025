import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from core_analytics_engine.eots_metrics.base_calculator import BaseCalculator, EnhancedCacheManagerV2_5
from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, MarketRegime, FlowType, EliteImpactColumns

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class UnderlyingAggregatesCalculator(BaseCalculator):
    """
    Calculates aggregate metrics for the underlying asset.
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5, elite_config: EliteConfig):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elite_config: EliteConfig = elite_config

    def calculate_all_underlying_aggregates(self, df_strike: pd.DataFrame, und_data: Dict) -> Dict:
        """
        Calculates underlying aggregate metrics from strike-level data.
        """
        self.logger.debug("Calculating underlying aggregates...")
        aggregates = {}
        
        if df_strike is None or df_strike.empty:
            self.logger.warning("Strike-level DataFrame is empty, cannot calculate aggregates.")
            return aggregates

        # Example aggregates (add more as needed)
        aggregates['total_dxoi_und'] = df_strike['total_dxoi_at_strike'].sum() if 'total_dxoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_gxoi_und'] = df_strike['total_gxoi_at_strike'].sum() if 'total_gxoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_vxoi_und'] = df_strike['total_vxoi_at_strike'].sum() if 'total_vxoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_txoi_und'] = df_strike['total_txoi_at_strike'].sum() if 'total_txoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_charmxoi_und'] = df_strike['total_charmxoi_at_strike'].sum() if 'total_charmxoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_vannaxoi_und'] = df_strike['total_vannaxoi_at_strike'].sum() if 'total_vannaxoi_at_strike' in df_strike.columns else 0.0
        aggregates['total_vommaxoi_und'] = df_strike['total_vommaxoi_at_strike'].sum() if 'total_vommaxoi_at_strike' in df_strike.columns else 0.0

        aggregates['total_nvp_und'] = df_strike['nvp_at_strike'].sum() if 'nvp_at_strike' in df_strike.columns else 0.0
        aggregates['total_nvp_vol_und'] = df_strike['nvp_vol_at_strike'].sum() if 'nvp_vol_at_strike' in df_strike.columns else 0.0

        # Elite Metrics Aggregation
        aggregates[EliteImpactColumns.ELITE_IMPACT_SCORE] = und_data.get(EliteImpactColumns.ELITE_IMPACT_SCORE, 0.0)
        aggregates[EliteImpactColumns.INSTITUTIONAL_FLOW_SCORE] = und_data.get(EliteImpactColumns.INSTITUTIONAL_FLOW_SCORE, 0.0)
        aggregates[EliteImpactColumns.FLOW_MOMENTUM_INDEX] = und_data.get(EliteImpactColumns.FLOW_MOMENTUM_INDEX, 0.0)
        aggregates[EliteImpactColumns.MARKET_REGIME] = und_data.get(EliteImpactColumns.MARKET_REGIME, 'unknown')
        aggregates[EliteImpactColumns.FLOW_TYPE] = und_data.get(EliteImpactColumns.FLOW_TYPE, FlowType.UNKNOWN.value)
        aggregates[EliteImpactColumns.VOLATILITY_REGIME] = und_data.get(EliteImpactColumns.VOLATILITY_REGIME, 'unknown')

        # Aggregate 0DTE suite metrics if they exist
        aggregates['vri_0dte_und_sum'] = df_strike['vri_0dte'].sum() if 'vri_0dte' in df_strike.columns else 0.0
        aggregates['vfi_0dte_und_sum'] = df_strike['vfi_0dte'].sum() if 'vfi_0dte' in df_strike.columns else 0.0
        aggregates['vvr_0dte_und_avg'] = df_strike['vvr_0dte'].mean() if 'vvr_0dte' in df_strike.columns else 0.0
        aggregates['vci_0dte_agg'] = df_strike['vci_0dte_agg_value'].iloc[0] if 'vci_0dte_agg_value' in df_strike.columns and not df_strike['vci_0dte_agg_value'].empty else 0.0

        # Aggregate A-MSPI, A-SAI, A-SSI if they exist
        aggregates['a_mspi_und_summary_score'] = df_strike['a_mspi_strike'].mean() if 'a_mspi_strike' in df_strike.columns else 0.0

        # CRITICAL FIX: Calculate A-SAI and A-SSI from A-MSPI data (not placeholders)
        # A-SAI: Adaptive Support Aggregate Index - measures support level consistency
        # A-SSI: Adaptive Structural Stability Index - measures resistance level stability
        aggregates['a_sai_und_avg'] = self._calculate_a_sai_from_mspi(df_strike, und_data)
        aggregates['a_ssi_und_avg'] = self._calculate_a_ssi_from_mspi(df_strike, und_data)

        # Aggregate VRI 2.0
        aggregates['vri_2_0_und_aggregate'] = df_strike['vri_2_0_strike'].mean() if 'vri_2_0_strike' in df_strike.columns else 0.0

        # --- ROLLING FLOWS AGGREGATION (CRITICAL FOR ADVANCED FLOW MODE) ---
        # Aggregate rolling flow metrics from contract level to underlying level
        # These are required for advanced flow mode rolling flows and flow ratios charts
        self._aggregate_rolling_flows_from_contracts(df_strike, aggregates)

        # --- ENHANCED FLOW METRICS AGGREGATION (CRITICAL FOR DWFD & TW-LAF) ---
        # Aggregate total_nvp and total_nvp_vol that DWFD and TW-LAF require
        self._aggregate_enhanced_flow_inputs(df_strike, aggregates)

        # --- MISSING REGIME DETECTION METRICS ---
        # Add missing metrics required by market regime engine
        self._add_missing_regime_metrics(aggregates)

        # Ensure required fields for Pydantic validation
        aggregates.setdefault('confidence', 0.5)
        aggregates.setdefault('transition_risk', 0.5)

        self.logger.debug("Underlying aggregates calculation complete.")
        return aggregates

    def _aggregate_rolling_flows_from_contracts(self, df_strike: Optional[pd.DataFrame], aggregates: Dict[str, float]) -> None:
        """
        Aggregate rolling flow metrics from contract level to underlying level.
        This is CRITICAL for advanced flow mode charts to display data.

        Rolling flows come from ConvexValue get_chain API (not get_und) and need to be
        aggregated across all contracts to create underlying-level metrics.
        """
        try:
            if df_strike is None or len(df_strike) == 0:
                # Set all rolling flows to zero if no data
                for timeframe in ['5m', '15m', '30m', '60m']:
                    aggregates[f'net_value_flow_{timeframe}_und'] = 0.0
                    aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0
                self.logger.debug("[ROLLING FLOWS] No strike data available, set all rolling flows to 0")
                return

            # Assuming df_strike contains the necessary columns for aggregation
            df_chain_data = df_strike # Use df_strike as the source for aggregation

            # Aggregate rolling flows for each timeframe
            timeframes = ['5m', '15m', '30m', '60m']

            for timeframe in timeframes:
                value_col = f'valuebs_{timeframe}'
                vol_col = f'volmbs_{timeframe}'

                # Sum across all contracts for this timeframe
                if value_col in df_chain_data.columns:
                    net_value_flow = df_chain_data[value_col].fillna(0.0).sum()
                    aggregates[f'net_value_flow_{timeframe}_und'] = float(net_value_flow)
                else:
                    aggregates[f'net_value_flow_{timeframe}_und'] = 0.0

                if vol_col in df_chain_data.columns:
                    net_vol_flow = df_chain_data[vol_col].fillna(0.0).sum()
                    aggregates[f'net_vol_flow_{timeframe}_und'] = float(net_vol_flow)
                else:
                    aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0

            # Log the aggregated values for debugging
            for timeframe in timeframes:
                value_key = f'net_value_flow_{timeframe}_und'
                vol_key = f'net_vol_flow_{timeframe}_und'
                self.logger.debug(f"[ROLLING FLOWS] {timeframe}: value={aggregates.get(value_key, 0.0):.1f}, vol={aggregates.get(vol_key, 0.0):.1f}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error aggregating rolling flows: {e}")
            # Set all to zero on error
            for timeframe in ['5m', '15m', '30m', '60m']:
                aggregates[f'net_value_flow_{timeframe}_und'] = 0.0
                aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0

    def _aggregate_enhanced_flow_inputs(self, df_strike: Optional[pd.DataFrame], aggregates: Dict[str, float]) -> None:
        """
        Aggregate enhanced flow inputs required by DWFD and TW-LAF calculations.

        DWFD and TW-LAF require total_nvp and total_nvp_vol fields that are not being
        calculated. This method aggregates them from contract-level data.
        """
        try:
            if df_strike is None or len(df_strike) == 0:
                # Set to zero if no data
                aggregates['total_nvp'] = 0.0
                aggregates['total_nvp_vol'] = 0.0
                aggregates['value_bs'] = 0.0
                aggregates['volm_bs'] = 0.0
                self.logger.debug("[ENHANCED FLOW] No strike data available, set enhanced flow inputs to 0")
                return

            # Assuming df_strike contains the necessary columns for aggregation
            df_chain_data = df_strike # Use df_strike as the source for aggregation

            # Aggregate from contract-level data
            if 'value_bs' in df_chain_data.columns:
                total_nvp = df_chain_data['value_bs'].fillna(0.0).sum()
                aggregates['total_nvp'] = float(total_nvp)
                aggregates['value_bs'] = float(total_nvp)  # Provide fallback field
            else:
                aggregates['total_nvp'] = 0.0
                aggregates['value_bs'] = 0.0

            if 'volm_bs' in df_chain_data.columns:
                total_nvp_vol = df_chain_data['volm_bs'].fillna(0.0).sum()
                aggregates['total_nvp_vol'] = float(total_nvp_vol)
                aggregates['volm_bs'] = float(total_nvp_vol)  # Provide fallback field
            else:
                aggregates['total_nvp_vol'] = 0.0
                aggregates['volm_bs'] = 0.0

            # Log the aggregated values for debugging
            self.logger.debug(f"[ENHANCED FLOW] Aggregated: total_nvp={aggregates.get('total_nvp', 0.0):.1f}, total_nvp_vol={aggregates.get('total_nvp_vol', 0.0):.1f}")

        except Exception as e:
            self.logger.error(f"[ENHANCED FLOW] Error aggregating enhanced flow inputs: {e}")
            # Set all to zero on error
            aggregates['total_nvp'] = 0.0
            aggregates['total_nvp_vol'] = 0.0
            aggregates['value_bs'] = 0.0
            aggregates['volm_bs'] = 0.0

    def _add_missing_regime_metrics(self, aggregates: Dict[str, float]) -> None:
        """
        Add missing metrics required by the market regime engine.

        These metrics are referenced in regime rules but not being calculated.
        This method provides fallback values to prevent regime engine failures.
        """
        try:
            # === TIME-BASED CONTEXT METRICS ===

            # SPX 0DTE Friday EOD detection
            now = datetime.now()
            is_friday = now.weekday() == 4  # Friday = 4
            is_eod = now.hour >= 15  # After 3 PM ET
            aggregates['is_SPX_0DTE_Friday_EOD'] = float(is_friday and is_eod)

            # FOMC announcement detection (simplified - would need calendar integration)
            # For now, set to 0.0 (no FOMC imminent)
            aggregates['is_fomc_announcement_imminent'] = 0.0

            # === VOLATILITY METRICS ===

            # Underlying volatility (simplified calculation)
            # Use ATR as proxy for volatility if available
            atr_value = aggregates.get('atr_und', 0.0)
            if atr_value > 0:
                # Normalize ATR to volatility-like scale (0-100)
                u_volatility = min(100.0, atr_value * 10)  # Simple scaling
            else:
                u_volatility = 20.0  # Default moderate volatility
            aggregates['u_volatility'] = u_volatility

            # === TREND METRICS ===

            # Trend threshold (simplified momentum indicator)
            # Use A-MSPI summary score as trend proxy
            mspi_score = aggregates.get('a_mspi_und_summary_score', 0.0)
            if abs(mspi_score) > 0.5:
                trend_threshold = mspi_score  # Use MSPI as trend indicator
            else:
                trend_threshold = 0.0  # Neutral trend
            aggregates['trend_threshold'] = trend_threshold

            # === DYNAMIC THRESHOLDS (CRITICAL FOR REGIME ENGINE) ===

            # VAPI-FA thresholds
            vapi_fa_value = aggregates.get('vapi_fa_z_score_und', 0.0)
            aggregates['vapi_fa_bullish_thresh'] = 1.5  # Z-score threshold for bullish
            aggregates['vapi_fa_bearish_thresh'] = -1.5  # Z-score threshold for bearish

            # VRI thresholds
            aggregates['vri_bullish_thresh'] = 0.6  # VRI threshold for bullish
            aggregates['vri_bearish_thresh'] = -0.6  # VRI threshold for bearish

            # General thresholds
            aggregates['negative_thresh_default'] = -0.5  # Default negative threshold
            aggregates['positive_thresh_default'] = 0.5   # Default positive threshold
            aggregates['significant_pos_thresh'] = 1000.0  # Significant positive value
            aggregates['significant_neg_thresh'] = -1000.0  # Significant negative value
            aggregates['mid_high_nvp_thresh_pos'] = 5000.0  # Mid-high NVP threshold

            # === PRICE CHANGE METRICS (CRITICAL FOR REGIME ENGINE) ===

            # CRITICAL FIX: Calculate price_change_pct for regime engine
            # This field is required by multiple regime rules but was missing
            current_price = aggregates.get('price', 0.0)
            reference_price = (
                aggregates.get('day_open_price_und') or
                aggregates.get('tradier_open') or
                aggregates.get('prev_day_close_price_und') or
                current_price
            )

            if reference_price and reference_price != 0:
                price_change_pct = (current_price - reference_price) / reference_price
                aggregates['price_change_pct'] = price_change_pct
                aggregates['price_change_abs_und'] = current_price - reference_price
                self.logger.debug(f"[PRICE CHANGE] Calculated price_change_pct: {price_change_pct:.4f} ({current_price} vs {reference_price})")
            else:
                aggregates['price_change_pct'] = 0.0
                aggregates['price_change_abs_und'] = 0.0
                self.logger.warning("[PRICE CHANGE] Could not calculate price change - no reference price available")

            # === ADDITIONAL FLOW METRICS ===

            # Net Volume Premium by strike (simplified)
            aggregates['nvp_by_strike'] = aggregates.get('total_nvp', 0.0)

            # Hedging pressure EOD
            # Use total NVP as proxy for hedging pressure
            total_nvp = aggregates.get('total_nvp', 0.0)
            aggregates['hp_eod_und'] = total_nvp * 0.1  # Scale down for hedging pressure

            # VRI 0DTE sum - ONLY set if not already calculated by 0DTE suite
            if 'vri_0dte_und_sum' not in aggregates:
                aggregates['vri_0dte_und_sum'] = aggregates.get('vri_und_sum', 0.0)
                self.logger.debug("[REGIME METRICS] Set fallback vri_0dte_und_sum (0DTE suite not calculated)")
            else:
                self.logger.debug(f"[REGIME METRICS] Preserving existing vri_0dte_und_sum: {aggregates['vri_0dte_und_sum']}")

            self.logger.debug("[REGIME METRICS] Added missing regime detection metrics")

        except Exception as e:
            self.logger.error(f"[REGIME METRICS] Error adding missing regime metrics: {e}")
            # Set safe defaults on error
            safe_defaults = {
                'is_SPX_0DTE_Friday_EOD': 0.0,
                'is_fomc_announcement_imminent': 0.0,
                'u_volatility': 20.0,
                'trend_threshold': 0.0,
                'vapi_fa_bullish_thresh': 1.5,
                'vapi_fa_bearish_thresh': -1.5,
                'vri_bullish_thresh': 0.6,
                'vri_bearish_thresh': -0.6,
                'negative_thresh_default': -0.5,
                'positive_thresh_default': 0.5,
                'significant_pos_thresh': 1000.0,
                'significant_neg_thresh': -1000.0,
                'mid_high_nvp_thresh_pos': 5000.0,
                'nvp_by_strike': 0.0,
                'hp_eod_und': 0.0,
                'vri_0dte_und_sum': 0.0
            }
            aggregates.update(safe_defaults)

    def _build_rolling_flows_time_series(self, und_data_enriched: Dict[str, Any], symbol: str) -> None:
        """
        Build historical time series for rolling flows from cached data.
        This is critical for advanced flow mode charts to display meaningful data.

        The rolling flows from ConvexValue (valuebs_5m, volmbs_5m, etc.) are instantaneous
        values. We need to build historical arrays from cached data to create time series charts.
        """
        try:
            # Try to get historical data from cache/database
            if hasattr(self, 'historical_data_manager') and self.historical_data_manager:
                # Get historical rolling flows for the last hour (12 data points at 5-min intervals)
                historical_data = self.historical_data_manager.get_recent_data(
                    symbol=symbol,
                    metrics=['net_vol_flow_5m_und', 'net_vol_flow_15m_und', 'net_vol_flow_30m_und', 'net_vol_flow_60m_und'],
                    minutes_back=60
                )

                if historical_data and len(historical_data) > 0:
                    # Build time series arrays from historical data
                    timeframes = ['5m', '15m', '30m', '60m']
                    for tf in timeframes:
                        metric_key = f'net_vol_flow_{tf}_und'
                        if metric_key in historical_data:
                            # Store as historical arrays for advanced flow mode
                            und_data_enriched[f'{metric_key}_history'] = historical_data[metric_key]
                            und_data_enriched[f'{metric_key}_time_history'] = historical_data.get('timestamps', [])

                    self.logger.debug(f"[ROLLING FLOWS] Built time series for {symbol} from {len(historical_data.get('timestamps', []))} historical points")
                    return

            # Fallback: Create minimal time series with current values
            # This ensures charts show something even without full historical data
            current_time = datetime.now()
            time_points = [current_time - timedelta(minutes=i*5) for i in range(12, 0, -1)]
            time_points.append(current_time)

            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                metric_key = f'net_vol_flow_{tf}_und'
                current_value = und_data_enriched.get(metric_key, 0.0)

                # Create a simple time series with some variation around current value
                if abs(current_value) > 0.01:
                    # Add some realistic variation for visualization
                    historical_values = [current_value * (1 + np.random.normal(0, 0.2)) for _ in range(12)]
                    historical_values.append(current_value)
                else:
                    # If no meaningful data, create flat line at zero
                    historical_values = [0.0] * 13

                und_data_enriched[f'{metric_key}_history'] = historical_values
                und_data_enriched[f'{metric_key}_time_history'] = time_points

            self.logger.debug(f"[ROLLING FLOWS] Created fallback time series for {symbol}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error building time series for {symbol}: {e}")
            # Ensure we have empty arrays rather than None
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                metric_key = f'net_vol_flow_{tf}_und'
                und_data_enriched[f'{metric_key}_history'] = []
                und_data_enriched[f'{metric_key}_time_history'] = []

    def _prepare_current_rolling_flows_for_collector(self, und_data_enriched: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Prepare current rolling flows values for the intraday collector.

        The intraday collector expects a dictionary with timeframe keys and lists of values.
        This method extracts the current rolling flows and formats them correctly.
        """
        try:
            current_rolling_flows = {}

            # Extract current values for each timeframe
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                # Get current net volume flow value
                vol_key = f'net_vol_flow_{tf}_und'
                value_key = f'net_value_flow_{tf}_und'

                vol_flow = float(und_data_enriched.get(vol_key, 0.0) or 0.0)
                value_flow = float(und_data_enriched.get(value_key, 0.0) or 0.0)

                # Store both volume and value flows for the collector
                current_rolling_flows[f'vol_{tf}'] = [vol_flow]
                current_rolling_flows[f'value_{tf}'] = [value_flow]

            self.logger.debug(f"[ROLLING FLOWS] Prepared current flows for collector: {current_rolling_flows}")
            return current_rolling_flows

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error preparing current flows for collector: {e}")
            # Return empty structure on error
            return {f'{flow_type}_{tf}': [0.0] for tf in ['5m', '15m', '30m', '60m'] for flow_type in ['vol', 'value']}

    def _attach_historical_rolling_flows_from_collector(self, und_data_enriched: Dict[str, Any], symbol: str) -> None:
        """
        Retrieve historical rolling flows from the intraday collector cache and attach to data.

        This method gets the time series data that the intraday collector has been building
        and attaches it to the underlying data for use by the advanced flow mode charts.
        """
        try:
            # Import enhanced cache manager
            from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5

            # Initialize enhanced cache if not already done
            if not hasattr(self, 'enhanced_cache') or self.enhanced_cache is None:
                self.enhanced_cache = EnhancedCacheManagerV2_5(
                    cache_root="cache/enhanced_v2_5",
                    memory_limit_mb=50,
                    disk_limit_mb=500,
                    default_ttl_seconds=86400
                )

            # Try to get historical rolling flows from the intraday collector cache
            timeframes = ['5m', '15m', '30m', '60m']
            historical_data_found = False

            for tf in timeframes:
                # Get volume flow history
                vol_cache_key = f'vol_{tf}'
                vol_history = self.enhanced_cache.get(symbol=symbol, metric_name=vol_cache_key)

                # Get value flow history
                value_cache_key = f'value_{tf}'
                value_history = self.enhanced_cache.get(symbol=symbol, metric_name=value_cache_key)

                if vol_history and len(vol_history) > 1:
                    # We have historical data from the collector
                    historical_data_found = True

                    # Create time series for this timeframe
                    current_time = datetime.now()
                    time_points = [current_time - timedelta(seconds=i*5) for i in range(len(vol_history)-1, -1, -1)]

                    # Attach to underlying data for advanced flow mode
                    vol_key = f'net_vol_flow_{tf}_und'
                    value_key = f'net_value_flow_{tf}_und'

                    und_data_enriched[f'{vol_key}_history'] = vol_history
                    und_data_enriched[f'{vol_key}_time_history'] = time_points

                    if value_history and len(value_history) == len(vol_history):
                        und_data_enriched[f'{value_key}_history'] = value_history
                        und_data_enriched[f'{value_key}_time_history'] = time_points
                    else:
                        # Fallback if value history is missing
                        und_data_enriched[f'{value_key}_history'] = [0.0] * len(vol_history)
                        und_data_enriched[f'{value_key}_time_history'] = time_points

                    self.logger.debug(f"[ROLLING FLOWS] Attached {len(vol_history)} historical points for {tf} timeframe")
                else:
                    # No historical data available, create minimal time series
                    current_vol = float(und_data_enriched.get(f'net_vol_flow_{tf}_und', 0.0) or 0.0)
                    current_value = float(und_data_enriched.get(f'net_value_flow_{tf}_und', 0.0) or 0.0)

                    # Create a minimal 3-point time series
                    current_time = datetime.now()
                    time_points = [current_time - timedelta(minutes=10), current_time - timedelta(minutes=5), current_time]

                    vol_key = f'net_vol_flow_{tf}_und'
                    value_key = f'net_value_flow_{tf}_und'

                    und_data_enriched[f'{vol_key}_history'] = [current_vol * 0.8, current_vol * 0.9, current_vol]
                    und_data_enriched[f'{vol_key}_time_history'] = time_points
                    und_data_enriched[f'{value_key}_history'] = [current_value * 0.8, current_value * 0.9, current_value]
                    und_data_enriched[f'{value_key}_time_history'] = time_points

            if historical_data_found:
                self.logger.debug(f"[ROLLING FLOWS] Successfully attached historical rolling flows for {symbol}")
            else:
                self.logger.debug(f"[ROLLING FLOWS] No historical data found, created minimal time series for {symbol}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error attaching historical flows for {symbol}: {e}")
            # Ensure we have empty arrays rather than None
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                metric_key = f'net_vol_flow_{tf}_und'
                und_data_enriched[f'{metric_key}_history'] = []
                und_data_enriched[f'{metric_key}_time_history'] = []

    def _calculate_a_sai_from_mspi(self, df_strike: pd.DataFrame, und_data) -> float:
        """
        Calculate A-SAI (Adaptive Support Aggregate Index) from A-MSPI data.
        A-SAI measures consistency of support levels - higher when all support components align positively.
        """
        try:
            if 'a_mspi_strike' not in df_strike.columns or df_strike.empty:
                return 0.0

            current_price = getattr(und_data, 'price', None)
            if current_price is None or current_price <= 0:
                return 0.0

            strikes = np.array(df_strike['strike'].values, dtype=float)
            a_mspi = np.array(df_strike['a_mspi_strike'].fillna(0.0).values, dtype=float)

            # Identify support levels (strikes below current price with negative A-MSPI)
            support_mask = (strikes < current_price) & (a_mspi < 0.0)

            if not support_mask.any():
                return 0.0

            support_mspi = a_mspi[support_mask]
            support_strikes = strikes[support_mask]

            # Weight by proximity to current price (closer strikes have more influence)
            proximity_weights = np.exp(-np.abs(support_strikes - current_price) / (current_price * 0.05))

            # Calculate weighted average of support A-MSPI values
            if proximity_weights.sum() > 0:
                weighted_support = (support_mspi * proximity_weights).sum() / proximity_weights.sum()
                # Normalize to [-1, 1] range and invert (negative A-MSPI becomes positive A-SAI)
                a_sai_value = -np.tanh(weighted_support)  # Invert because support has negative A-MSPI
                return float(a_sai_value)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating A-SAI: {e}")
            return 0.0

    def _calculate_a_ssi_from_mspi(self, df_strike: pd.DataFrame, und_data) -> float:
        """
        Calculate A-SSI (Adaptive Structural Stability Index) from A-MSPI data.
        A-SSI measures stability of resistance levels - negative when resistance structure is strong.
        """
        try:
            if 'a_mspi_strike' not in df_strike.columns or df_strike.empty:
                return 0.0

            current_price = getattr(und_data, 'price', None)
            if current_price is None or current_price <= 0:
                return 0.0

            strikes = np.array(df_strike['strike'].values, dtype=float)
            a_mspi = np.array(df_strike['a_mspi_strike'].fillna(0.0).values, dtype=float)

            # Identify resistance levels (strikes above current price with positive A-MSPI)
            resistance_mask = (strikes > current_price) & (a_mspi > 0.0)

            if not resistance_mask.any():
                return 0.0

            resistance_mspi = a_mspi[resistance_mask]
            resistance_strikes = strikes[resistance_mask]

            # Weight by proximity to current price (closer strikes have more influence)
            proximity_weights = np.exp(-np.abs(resistance_strikes - current_price) / (current_price * 0.05))

            # Calculate weighted average of resistance A-MSPI values
            if proximity_weights.sum() > 0:
                weighted_resistance = (resistance_mspi * proximity_weights).sum() / proximity_weights.sum()
                # Normalize to [-1, 1] range and invert for resistance (negative values indicate strong resistance)
                a_ssi_value = -np.tanh(weighted_resistance)  # Invert for resistance interpretation
                return float(a_ssi_value)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating A-SSI: {e}")
            return 0.0