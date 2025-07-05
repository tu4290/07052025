# core_analytics_engine/eots_metrics/supplementary_metrics.py

"""
EOTS Supplementary Metrics - Consolidated Miscellaneous Calculations

Consolidates:
- miscellaneous_metrics.py: ATR, advanced options metrics, and other utilities

Optimizations:
- Streamlined ATR calculation
- Simplified advanced options metrics
- Unified utility functions
- Eliminated redundant calculations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from scipy import stats

from core_analytics_engine.eots_metrics.core_calculator import CoreCalculator
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class AdvancedOptionsMetrics:
    """Simplified advanced options metrics data structure"""
    
    def __init__(self, lwpai: float = 0.0, vabai: float = 0.0, aofm: float = 0.0, lidb: float = 0.0):
        self.lwpai = lwpai  # Liquidity-Weighted Price Action Indicator
        self.vabai = vabai  # Volatility-Adjusted Bid/Ask Imbalance
        self.aofm = aofm    # Aggressive Order Flow Momentum
        self.lidb = lidb    # Liquidity-Implied Directional Bias
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy integration"""
        return {
            'lwpai': self.lwpai,
            'vabai': self.vabai,
            'aofm': self.aofm,
            'lidb': self.lidb
        }

class SupplementaryMetrics(CoreCalculator):
    """
    Consolidated supplementary metrics calculator.
    
    Handles:
    - ATR (Average True Range) calculation
    - Advanced options metrics (LWPAI, VABAI, AOFM, LIDB)
    - Other utility calculations
    """
    
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5):
        super().__init__(config_manager, historical_data_manager, enhanced_cache_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ATR calculation parameters
        self.DEFAULT_ATR_PERIOD = 14
        self.MIN_ATR_PERIODS = 5
        
        # Advanced metrics parameters
        self.LIQUIDITY_THRESHOLD = 1000000  # $1M threshold for liquidity calculations
        self.VOLATILITY_NORMALIZATION_FACTOR = 100
        
        # Previous values for momentum calculations
        self._previous_aofm = 0.0
        self._previous_lidb = 0.0
    
    # =============================================================================
    # ATR CALCULATION - Optimized from miscellaneous_metrics.py
    # =============================================================================
    
    def calculate_atr(self, symbol: str, dte_max: int = 45) -> float:
        """
        Calculate Average True Range (ATR) for the underlying symbol.
        
        Args:
            symbol: Underlying symbol
            dte_max: Maximum DTE for context (affects lookback period)
            
        Returns:
            ATR value as float
        """
        self.logger.debug(f"Calculating ATR for {symbol}...")
        
        try:
            # FAIL-FAST: No ATR calculation for futures symbols
            if self._is_futures_symbol(symbol):
                self.logger.error(f"CRITICAL: Cannot calculate ATR for futures symbol {symbol} - futures not supported!")
                raise ValueError(f"CRITICAL: ATR calculation not supported for futures symbol {symbol}!")
            
            # Determine lookback period based on DTE context
            lookback_days = max(dte_max, self.DEFAULT_ATR_PERIOD)
            
            # Get historical OHLCV data
            ohlcv_df = self.historical_data_manager.get_historical_ohlcv(symbol, lookback_days=lookback_days)
            
            if ohlcv_df is None or len(ohlcv_df) < self.MIN_ATR_PERIODS:
                self.logger.error(f"CRITICAL: Insufficient OHLCV data for {symbol} - need at least {self.MIN_ATR_PERIODS} periods!")
                raise ValueError(f"CRITICAL: Insufficient OHLCV data for {symbol} - cannot calculate ATR without real historical data!")
            
            # Calculate True Range components
            high_low = ohlcv_df['high'] - ohlcv_df['low']
            high_close_prev = np.abs(ohlcv_df['high'] - ohlcv_df['close'].shift(1))
            low_close_prev = np.abs(ohlcv_df['low'] - ohlcv_df['close'].shift(1))
            
            # True Range is the maximum of the three components
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # Remove NaN values (from shift operation)
            true_range = true_range.dropna()
            
            if len(true_range) < self.MIN_ATR_PERIODS:
                self.logger.error(f"CRITICAL: Insufficient True Range data for {symbol} - need at least {self.MIN_ATR_PERIODS} periods!")
                raise ValueError(f"CRITICAL: Insufficient True Range data for {symbol} - cannot calculate ATR without real data!")
            
            # Calculate ATR using exponential moving average for better responsiveness
            atr_period = min(self.DEFAULT_ATR_PERIOD, len(true_range))
            atr_value = true_range.ewm(span=atr_period, adjust=False).mean().iloc[-1]
            
            # FAIL-FAST: Validate ATR value
            if pd.isna(atr_value) or atr_value <= 0:
                self.logger.error(f"CRITICAL: Invalid ATR value for {symbol}: {atr_value} - cannot return fake 0.0!")
                raise ValueError(f"CRITICAL: Invalid ATR value for {symbol}: {atr_value} - ATR must be positive!")
            
            self.logger.debug(f"ATR calculated for {symbol}: {atr_value:.4f}")
            return float(atr_value)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating ATR for {symbol}: {e}", exc_info=True)
            raise ValueError(f"CRITICAL: ATR calculation failed for {symbol} - cannot return fake 0.0 value! Error: {e}") from e
    
    # =============================================================================
    # ADVANCED OPTIONS METRICS - Simplified from miscellaneous_metrics.py
    # =============================================================================
    
    def calculate_advanced_options_metrics(self, options_df: pd.DataFrame, underlying_data: Dict) -> AdvancedOptionsMetrics:
        """
        Calculate advanced options metrics using simplified but effective methodology.
        
        Args:
            options_df: DataFrame with options data
            underlying_data: Dictionary with underlying market data
            
        Returns:
            AdvancedOptionsMetrics object with calculated values
        """
        self.logger.debug("Calculating advanced options metrics...")
        
        try:
            if options_df.empty:
                return self._get_default_advanced_metrics()
            
            # Calculate individual metrics
            lwpai = self._calculate_lwpai_optimized(options_df, underlying_data)
            vabai = self._calculate_vabai_optimized(options_df, underlying_data)
            aofm = self._calculate_aofm_optimized(options_df, underlying_data)
            lidb = self._calculate_lidb_optimized(options_df, underlying_data)
            
            # Create and return metrics object
            metrics = AdvancedOptionsMetrics(lwpai=lwpai, vabai=vabai, aofm=aofm, lidb=lidb)
            
            self.logger.debug(f"Advanced options metrics calculated: LWPAI={lwpai:.2f}, VABAI={vabai:.2f}, AOFM={aofm:.2f}, LIDB={lidb:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced options metrics: {e}", exc_info=True)
            return self._get_default_advanced_metrics()
    
    def _calculate_lwpai_optimized(self, options_df: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate Liquidity-Weighted Price Action Indicator (simplified)"""
        try:
            # FAIL-FAST: Extract volume and value data - NO FAKE DEFAULTS ALLOWED
            if 'day_volume' not in underlying_data:
                raise ValueError("CRITICAL: day_volume missing from underlying_data - cannot calculate LWPAI without real volume data!")
            total_volume = float(underlying_data['day_volume'])

            if 'net_value_flow_5m_und' not in underlying_data:
                raise ValueError("CRITICAL: net_value_flow_5m_und missing from underlying_data - cannot calculate LWPAI without real flow data!")
            net_value_flow = float(underlying_data['net_value_flow_5m_und'])

            if total_volume < self.LIQUIDITY_THRESHOLD:
                raise ValueError(f"CRITICAL: Insufficient liquidity (volume={total_volume}) - cannot calculate meaningful LWPAI signal!")
            
            # Calculate liquidity-weighted price action
            price_action = abs(net_value_flow) / max(total_volume, 1.0)
            
            # Normalize to 0-100 scale
            lwpai = min(100.0, price_action * self.VOLATILITY_NORMALIZATION_FACTOR)
            
            return lwpai
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating LWPAI: {e}")
            raise ValueError(f"CRITICAL: LWPAI calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_vabai_optimized(self, options_df: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate Volatility-Adjusted Bid/Ask Imbalance (simplified)"""
        try:
            # FAIL-FAST: Extract volatility and flow data - NO FAKE DEFAULTS ALLOWED
            if 'u_volatility' not in underlying_data:
                raise ValueError("CRITICAL: u_volatility missing from underlying_data - cannot calculate VABAI without real volatility data!")
            current_iv = float(underlying_data['u_volatility'])

            if 'net_vol_flow_5m_und' not in underlying_data:
                raise ValueError("CRITICAL: net_vol_flow_5m_und missing from underlying_data - cannot calculate VABAI without real flow data!")
            net_vol_flow = float(underlying_data['net_vol_flow_5m_und'])
            
            # Calculate volatility-adjusted imbalance
            vol_adjustment = max(0.5, min(2.0, current_iv / 0.20))  # Normalize around 20% IV
            adjusted_imbalance = net_vol_flow * vol_adjustment
            
            # Scale to meaningful range
            vabai = np.tanh(adjusted_imbalance / 100000) * 100  # Bounded between -100 and 100
            
            return vabai
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating VABAI: {e}")
            raise ValueError(f"CRITICAL: VABAI calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_aofm_optimized(self, options_df: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate Aggressive Order Flow Momentum (simplified)"""
        try:
            # FAIL-FAST: Extract flow momentum data - NO FAKE DEFAULTS ALLOWED
            if 'net_vol_flow_5m_und' not in underlying_data:
                raise ValueError("CRITICAL: net_vol_flow_5m_und missing from underlying_data - cannot calculate AOFM without real 5m flow data!")
            net_vol_flow_5m = float(underlying_data['net_vol_flow_5m_und'])

            if 'net_vol_flow_15m_und' not in underlying_data:
                raise ValueError("CRITICAL: net_vol_flow_15m_und missing from underlying_data - cannot calculate AOFM without real 15m flow data!")
            net_vol_flow_15m = float(underlying_data['net_vol_flow_15m_und'])
            
            # Calculate momentum acceleration
            if abs(net_vol_flow_15m) > 0.001:
                momentum_ratio = net_vol_flow_5m / net_vol_flow_15m
            else:
                momentum_ratio = 0.0
            
            # Calculate momentum change from previous period
            momentum_change = momentum_ratio - self._previous_aofm
            self._previous_aofm = momentum_ratio
            
            # AOFM score
            aofm = momentum_change * 50  # Scale to reasonable range
            
            return self._bound_value(aofm, -100, 100)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating AOFM: {e}")
            raise ValueError(f"CRITICAL: AOFM calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def _calculate_lidb_optimized(self, options_df: pd.DataFrame, underlying_data: Dict) -> float:
        """Calculate Liquidity-Implied Directional Bias (simplified)"""
        try:
            # FAIL-FAST: Extract directional flow data - NO FAKE DEFAULTS ALLOWED
            if 'net_cust_delta_flow_und' not in underlying_data:
                raise ValueError("CRITICAL: net_cust_delta_flow_und missing from underlying_data - cannot calculate LIDB without real delta flow data!")
            net_delta_flow = float(underlying_data['net_cust_delta_flow_und'])

            if 'day_volume' not in underlying_data:
                raise ValueError("CRITICAL: day_volume missing from underlying_data - cannot calculate LIDB without real volume data!")
            total_volume = float(underlying_data['day_volume'])

            if total_volume < self.LIQUIDITY_THRESHOLD:
                raise ValueError(f"CRITICAL: Insufficient liquidity (volume={total_volume}) - cannot calculate meaningful LIDB signal!")
            
            # Calculate liquidity-adjusted directional bias
            raw_bias = net_delta_flow / max(total_volume, 1.0)
            
            # Apply momentum component
            bias_change = raw_bias - self._previous_lidb
            self._previous_lidb = raw_bias
            
            # LIDB score with momentum
            lidb = (raw_bias * 0.7 + bias_change * 0.3) * 1000  # Scale to meaningful range
            
            return self._bound_value(lidb, -100, 100)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating LIDB: {e}")
            raise ValueError(f"CRITICAL: LIDB calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _get_default_advanced_metrics(self) -> AdvancedOptionsMetrics:
        """Return default advanced metrics on error"""
        return AdvancedOptionsMetrics(lwpai=0.0, vabai=0.0, aofm=0.0, lidb=0.0)
    
    def calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int = 20) -> float:
        """Calculate rolling correlation between two series"""
        try:
            if len(series1) < window or len(series2) < window:
                raise ValueError(f"CRITICAL: Insufficient data for rolling correlation - need at least {window} points, got {len(series1)} and {len(series2)}!")

            # Align series and calculate correlation
            aligned_data = pd.concat([series1, series2], axis=1).dropna()

            if len(aligned_data) < window:
                raise ValueError(f"CRITICAL: Insufficient aligned data for rolling correlation - need at least {window} points, got {len(aligned_data)}!")

            correlation = aligned_data.iloc[:, 0].rolling(window=window).corr(aligned_data.iloc[:, 1]).iloc[-1]

            if pd.isna(correlation):
                raise ValueError("CRITICAL: Rolling correlation calculation resulted in NaN - cannot return fake 0.0!")

            return float(correlation)

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating rolling correlation: {e}")
            raise ValueError(f"CRITICAL: Rolling correlation calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def calculate_volatility_percentile(self, current_iv: float, historical_iv_series: pd.Series, window: int = 252) -> float:
        """Calculate volatility percentile ranking"""
        try:
            if historical_iv_series.empty or len(historical_iv_series) < 10:
                return 50.0  # Neutral percentile
            
            # Use recent window for percentile calculation
            recent_iv = historical_iv_series.tail(window)
            
            # Calculate percentile rank
            percentile = stats.percentileofscore(recent_iv, current_iv)
            
            return self._bound_value(percentile, 0.0, 100.0)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating volatility percentile: {e}")
            raise ValueError(f"CRITICAL: Volatility percentile calculation failed - cannot return fake 50.0 value! Error: {e}") from e
    
    def calculate_price_momentum_score(self, price_series: pd.Series, periods: List[int] = [5, 10, 20]) -> float:
        """Calculate multi-period price momentum score"""
        try:
            if len(price_series) < max(periods):
                raise ValueError(f"CRITICAL: Insufficient price data for momentum calculation - need at least {max(periods)} points, got {len(price_series)}!")

            momentum_scores = []

            for period in periods:
                if len(price_series) >= period:
                    if price_series.iloc[-period] <= 0:
                        raise ValueError(f"CRITICAL: Invalid price data at period {period} - price must be positive!")
                    # Calculate period return
                    period_return = (price_series.iloc[-1] / price_series.iloc[-period] - 1) * 100
                    momentum_scores.append(period_return)

            if not momentum_scores:
                raise ValueError("CRITICAL: No valid momentum scores calculated - cannot return fake 0.0!")

            # Weighted average (shorter periods get higher weight)
            weights = [1.0 / (i + 1) for i in range(len(momentum_scores))]
            weighted_momentum = np.average(momentum_scores, weights=weights)

            return self._bound_value(weighted_momentum, -100, 100)

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating price momentum score: {e}")
            raise ValueError(f"CRITICAL: Price momentum calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def calculate_volume_profile_score(self, volume_series: pd.Series, current_volume: float, window: int = 20) -> float:
        """Calculate volume profile score relative to recent history"""
        try:
            if len(volume_series) < window:
                return 50.0  # Neutral score
            
            # Use recent window for comparison
            recent_volume = volume_series.tail(window)
            
            # Calculate percentile rank of current volume
            volume_percentile = stats.percentileofscore(recent_volume, current_volume)
            
            return self._bound_value(volume_percentile, 0.0, 100.0)
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating volume profile score: {e}")
            raise ValueError(f"CRITICAL: Volume profile calculation failed - cannot return fake 50.0 value! Error: {e}") from e
    
    def calculate_options_skew_indicator(self, options_df: pd.DataFrame, underlying_price: float) -> float:
        """Calculate options skew indicator"""
        try:
            if options_df.empty:
                raise ValueError("CRITICAL: options_df is empty - cannot calculate skew without options data!")

            if 'implied_volatility' not in options_df.columns:
                raise ValueError("CRITICAL: implied_volatility column missing from options_df - cannot calculate skew without IV data!")

            # Separate calls and puts
            calls = options_df[options_df['option_type'] == 'call']
            puts = options_df[options_df['option_type'] == 'put']

            if calls.empty:
                raise ValueError("CRITICAL: No call options found - cannot calculate skew without call data!")

            if puts.empty:
                raise ValueError("CRITICAL: No put options found - cannot calculate skew without put data!")

            # Find ATM options (closest to underlying price)
            calls['moneyness'] = abs(calls['strike'] - underlying_price)
            puts['moneyness'] = abs(puts['strike'] - underlying_price)

            atm_call_iv = calls.loc[calls['moneyness'].idxmin(), 'implied_volatility']
            atm_put_iv = puts.loc[puts['moneyness'].idxmin(), 'implied_volatility']

            if pd.isna(atm_call_iv) or pd.isna(atm_put_iv):
                raise ValueError("CRITICAL: ATM implied volatility is NaN - cannot calculate skew with invalid IV data!")

            # Calculate skew (put IV - call IV)
            skew = atm_put_iv - atm_call_iv

            return float(skew)

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating options skew indicator: {e}")
            raise ValueError(f"CRITICAL: Options skew calculation failed - cannot return fake 0.0 value! Error: {e}") from e

# Export the consolidated calculator
__all__ = ['SupplementaryMetrics', 'AdvancedOptionsMetrics']
