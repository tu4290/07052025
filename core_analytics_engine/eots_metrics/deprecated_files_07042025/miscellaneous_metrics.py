import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from scipy import stats

from core_analytics_engine.eots_metrics.base_calculator import BaseCalculator, EnhancedCacheManagerV2_5 # Import EnhancedCacheManagerV2_5
from data_models.advanced_metrics import AdvancedOptionsMetricsV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class MiscellaneousMetricsCalculator(BaseCalculator):
    """
    Calculates miscellaneous metrics like ATR and Advanced Options Metrics.
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._previous_aofm = 0.0 # Initialize for AOFM calculation

    def calculate_atr(self, symbol: str, dte_max: int) -> float:
        """
        Calculates Average True Range (ATR) for the underlying symbol.
        This method relies on historical data manager.
        """
        self.logger.debug(f"Calculating ATR for {symbol}...")
        try:
            # Skip ATR calculation for futures symbols - they use different data sources
            if self._is_futures_symbol(symbol):
                self.logger.debug(f"Skipping ATR calculation for futures symbol {symbol}")
                return 0.0

            # Calculate appropriate lookback based on DTE context
            # For ATR, we need enough data for a meaningful calculation
            # Use max(dte_max, 14) to ensure minimum 14 periods for ATR but respect DTE context
            lookback_days = max(dte_max, 14)

            ohlcv_df = self.historical_data_manager.get_historical_ohlcv(symbol, lookback_days=lookback_days)
            if ohlcv_df is None or len(ohlcv_df) == 0 or len(ohlcv_df) < 2:
                self.logger.debug(f"No OHLCV data available for {symbol}, skipping ATR calculation")
                return 0.0
            
            high_low = pd.Series(ohlcv_df['high'] - ohlcv_df['low'])
            high_close = pd.Series(np.abs(ohlcv_df['high'] - ohlcv_df['close'].shift()))
            low_close = pd.Series(np.abs(ohlcv_df['low'] - ohlcv_df['close'].shift()))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(com=14, min_periods=14).mean().iloc[-1]
            return float(atr)
        except Exception as e:
            self.logger.error(f"Failed to calculate ATR for {symbol}: {e}", exc_info=True)
            return 0.0

    def calculate_advanced_options_metrics(self, options_df_raw: pd.DataFrame) -> AdvancedOptionsMetricsV2_5:
        """
        Calculate advanced options metrics for price action analysis.

        Based on "Options Contract Metrics for Price Action Analysis" document:
        1. Liquidity-Weighted Price Action Indicator (LWPAI)
        2. Volatility-Adjusted Bid/Ask Imbalance (VABAI)
        3. Aggressive Order Flow Momentum (AOFM)
        4. Liquidity-Implied Directional Bias (LIDB)

        Args:
            options_df_raw: DataFrame containing raw options contract data from ConvexValue

        Returns:
            AdvancedOptionsMetricsV2_5 containing calculated metrics
        """
        try:
            # Get configuration settings
            # Assuming config is accessible via self.config_manager
            metrics_config = self.config_manager.get_setting("ticker_context_analyzer_settings.advanced_options_metrics", {})

            if not metrics_config.get("enabled", False):
                self.logger.debug("Advanced options metrics calculation disabled in config")
                return self._get_default_advanced_metrics()

            if len(options_df_raw) == 0:
                self.logger.warning("No options data provided for advanced metrics calculation")
                return self._get_default_advanced_metrics()

            min_contracts = metrics_config.get("min_contracts_for_calculation", 10)
            if len(options_df_raw) < min_contracts:
                self.logger.warning(f"Insufficient contracts ({len(options_df_raw)}) for reliable metrics calculation (min: {min_contracts})")
                return self._get_default_advanced_metrics()

            # Initialize lists for metric values
            lwpai_values = []
            vabai_values = []
            aofm_values = []
            lidb_values = []
            spread_percentages = []
            total_liquidity = 0.0
            valid_contracts = 0

            # Configuration thresholds
            min_bid_ask_size = metrics_config.get("min_bid_ask_size", 1)
            max_spread_pct = metrics_config.get("max_spread_percentage", 5.0)
            default_iv = metrics_config.get("default_implied_volatility", 0.20)

            # Store previous AOFM for momentum calculation
            # _previous_aofm is initialized in __init__

            for _, row in options_df_raw.iterrows():
                try:
                    # Extract required fields using CORRECT ConvexValue field names
                    bid_price = float(row.get('bid', 0.0)) if pd.notna(row.get('bid')) else 0.0
                    ask_price = float(row.get('ask', 0.0)) if pd.notna(row.get('ask')) else 0.0
                    bid_size = float(row.get('bid_size', 0.0)) if pd.notna(row.get('bid_size')) else 0.0
                    ask_size = float(row.get('ask_size', 0.0)) if pd.notna(row.get('ask_size')) else 0.0
                    implied_vol = float(row.get('iv', default_iv)) if pd.notna(row.get('iv')) else default_iv
                    theo_price = float(row.get('theo', 0.0)) if pd.notna(row.get('theo')) else 0.0
                    spread = float(row.get('spread', 0.0)) if pd.notna(row.get('spread')) else (ask_price - bid_price)

                    # Data quality filters
                    if bid_price <= 0 or ask_price <= 0 or bid_size < min_bid_ask_size or ask_size < min_bid_ask_size:
                        continue

                    # Calculate spread percentage
                    mid_price = (bid_price + ask_price) / 2.0
                    if mid_price > 0:
                        spread_pct = (spread / mid_price) * 100.0
                        if spread_pct > max_spread_pct:  # Skip contracts with excessive spreads
                            continue
                        spread_percentages.append(spread_pct)

                    # 1. Liquidity-Weighted Price Action Indicator (LWPAI)
                    # Formula: ((Bid Price * Bid Size) + (Ask Price * Ask Size)) / (Bid Size + Ask Size)
                    total_size = bid_size + ask_size
                    if total_size > 0:
                        lwpai = ((bid_price * bid_size) + (ask_price * ask_size)) / total_size
                        lwpai_values.append(lwpai)
                        total_liquidity += total_size

                    # 2. Volatility-Adjusted Bid/Ask Imbalance (VABAI)
                    # Formula: ((Bid Size - Ask Size) / (Bid Size + Ask Size)) * Implied Volatility
                    if total_size > 0:
                        size_imbalance = (bid_size - ask_size) / total_size
                        vabai = size_imbalance * implied_vol
                        vabai_values.append(vabai)

                    # 3. Aggressive Order Flow Momentum (AOFM) - Current component
                    # Formula: (Ask Price * Ask Size) - (Bid Price * Bid Size)
                    aofm_component = (ask_price * ask_size) - (bid_price * bid_size)
                    aofm_values.append(aofm_component) # Collect components to average later

                    # 4. Liquidity-Implied Directional Bias (LIDB)
                    # Formula: (Bid Size / (Bid Size + Ask Size)) - 0.5
                    if total_size > 0:
                        bid_proportion = bid_size / total_size
                        lidb = bid_proportion - 0.5
                        lidb_values.append(lidb)

                    valid_contracts += 1

                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Skipping contract due to data error: {e}")
                    continue

            # Calculate final metrics
            if valid_contracts > 0:
                # Calculate averages
                avg_lwpai = np.mean(lwpai_values) if lwpai_values else 0.0
                avg_vabai = np.mean(vabai_values) if vabai_values else 0.0
                avg_lidb = np.mean(lidb_values) if lidb_values else 0.0
                
                # AOFM: Calculate momentum as change from previous
                current_aofm_sum = np.sum(aofm_values) if aofm_values else 0.0
                current_aofm = current_aofm_sum / valid_contracts if valid_contracts > 0 else 0.0
                aofm_momentum = current_aofm - self._previous_aofm
                self._previous_aofm = current_aofm  # Store for next calculation

                # Calculate supporting metrics
                avg_spread_pct = np.mean(spread_percentages) if spread_percentages else 0.0
                spread_to_vol_ratio = avg_spread_pct / (default_iv * 100) if default_iv > 0 else 0.0

                # Calculate confidence score based on data quality
                confidence_config = metrics_config.get("confidence_scoring", {})
                min_valid = confidence_config.get('min_valid_contracts', 10)
                data_quality_weight = confidence_config.get('data_quality_weight', 0.4)
                spread_quality_weight = confidence_config.get('spread_quality_weight', 0.3)
                volume_quality_weight = confidence_config.get('volume_quality_weight', 0.3)

                data_quality = min(1.0, valid_contracts / min_valid)
                spread_quality = max(0.0, 1.0 - (avg_spread_pct / max_spread_pct))
                volume_quality = min(1.0, total_liquidity / 1000.0)  # Normalize to reasonable volume

                confidence_score = (
                    data_quality * data_quality_weight +
                    spread_quality * spread_quality_weight +
                    volume_quality * volume_quality_weight
                )

                # CRITICAL FIX: Normalize metrics to -1 to +1 range for gauge display
                # Use median LWPAI as reference price for normalization
                # current_price = np.median(lwpai_values) if lwpai_values else 100.0 # Not needed for normalization logic below

                # LWPAI: Normalize using Z-score approach for better sensitivity
                if lwpai_values and len(lwpai_values) > 1:
                    lwpai_std = np.std(lwpai_values)
                    lwpai_mean = np.mean(lwpai_values) # Use mean for Z-score
                    if lwpai_std > 0:
                        lwpai_z_score = (avg_lwpai - lwpai_mean) / lwpai_std
                        lwpai_normalized = max(-1.0, min(1.0, lwpai_z_score / 3.0))  # 3-sigma normalization
                    else:
                        lwpai_normalized = 0.0
                else:
                    lwpai_normalized = 0.0

                # VABAI: Already normalized by design, but ensure range
                vabai_normalized = max(-1.0, min(1.0, avg_vabai))

                # AOFM: Normalize using percentile-based scaling for better distribution
                if total_liquidity > 0 and aofm_momentum != 0:
                    # Use a more reasonable scale factor based on typical options values
                    # For SPX options, typical AOFM values range from -10000 to +10000
                    typical_aofm_range = 5000.0  # Adjust based on historical data
                    aofm_normalized = max(-1.0, min(1.0, aofm_momentum / typical_aofm_range))
                else:
                    aofm_normalized = 0.0

                # LIDB: Scale from -0.5/+0.5 to -1.0/+1.0
                lidb_normalized = max(-1.0, min(1.0, avg_lidb * 2.0))

                metrics = AdvancedOptionsMetricsV2_5(
                    lwpai=float(lwpai_normalized),  # Convert to float
                    vabai=float(vabai_normalized),  # Convert to float
                    aofm=float(aofm_normalized),    # Convert to float
                    lidb=float(lidb_normalized),    # Convert to float
                    bid_ask_spread_percentage=float(avg_spread_pct),
                    total_liquidity_size=float(total_liquidity),
                    spread_to_volatility_ratio=float(spread_to_vol_ratio),
                    theoretical_price_deviation=0.0,  # TODO: Calculate if needed
                    valid_contracts_count=int(valid_contracts),
                    calculation_timestamp=datetime.now(),
                    confidence_score=float(confidence_score),
                    data_quality_score=float(data_quality), # Added
                    contracts_analyzed=int(valid_contracts) # Added
                )

                self.logger.debug(f"✅ Advanced options metrics calculated (RAW): LWPAI={avg_lwpai:.4f}, VABAI={avg_vabai:.4f}, AOFM={aofm_momentum:.4f}, LIDB={avg_lidb:.4f}")
                self.logger.info(f"🎯 Advanced options metrics (NORMALIZED): LWPAI={lwpai_normalized:.4f}, VABAI={vabai_normalized:.4f}, AOFM={aofm_normalized:.4f}, LIDB={lidb_normalized:.4f}, confidence={confidence_score:.3f}")
                return metrics
            else:
                self.logger.warning("No valid contracts found for advanced metrics calculation")
                return self._get_default_advanced_metrics()

        except Exception as e:
            self.logger.error(f"Error calculating advanced options metrics: {e}", exc_info=True)
            return self._get_default_advanced_metrics()

    def _get_default_advanced_metrics(self) -> AdvancedOptionsMetricsV2_5:
        """
        Return default metrics when calculation fails.
        """
        self.logger.debug("Returning default advanced metrics due to calculation failure.")
        return AdvancedOptionsMetricsV2_5(
            lwpai=0.0,
            vabai=0.0,
            aofm=0.0,
            lidb=0.0,
            bid_ask_spread_percentage=0.0,
            total_liquidity_size=0.0,
            spread_to_volatility_ratio=0.0,
            theoretical_price_deviation=0.0,
            valid_contracts_count=0,
            calculation_timestamp=datetime.now(),
            confidence_score=0.0,
            data_quality_score=0.0, # Added
            contracts_analyzed=0 # Added
        )

    def calculate_underlying_volatility_and_trend(self, symbol: str, impl_vol_atm: float, lookback_days: int = 20) -> Dict[str, float]:
        """
        Calculates historical volatility and trend strength for the underlying.
        """
        self.logger.debug(f"Calculating historical volatility and trend for {symbol}...")
        try:
            ohlcv_df = self.historical_data_manager.get_historical_ohlcv(symbol, lookback_days=lookback_days)
            if ohlcv_df is None or len(ohlcv_df) < lookback_days:
                self.logger.warning(f"Insufficient OHLCV data for {symbol} for {lookback_days} days, returning defaults.")
                return {
                    "hist_vol_20d": 0.0,
                    "impl_vol_atm": impl_vol_atm,
                    "trend_strength": 0.0
                }

            # Historical Volatility (20-day annualized)
            returns = ohlcv_df['close'].pct_change().dropna()
            hist_vol_20d = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.0

            # Trend Strength (simple linear regression slope)
            prices = ohlcv_df['close'].tail(lookback_days).values
            if len(prices) > 1:
                x = np.arange(len(prices))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
                trend_strength = slope / np.mean(prices) # Normalize slope by average price
            else:
                trend_strength = 0.0

            return {
                "hist_vol_20d": float(hist_vol_20d),
                "impl_vol_atm": float(impl_vol_atm),
                "trend_strength": float(trend_strength)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate underlying volatility and trend for {symbol}: {e}", exc_info=True)
            return {
                "hist_vol_20d": 0.0,
                "impl_vol_atm": impl_vol_atm,
                "trend_strength": 0.0
            }
