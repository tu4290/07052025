import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import signal

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models import KeyLevelsDataV2_5, KeyLevelV2_5
from data_models import ProcessedUnderlyingAggregatesV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class KeyLevelIdentifierV2_5:
    """
    Identifies and scores critical price zones (Key Levels) based on a confluence of metrics.
    """

    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.settings = self.config_manager.get_setting("key_level_settings", default={})

        # Default settings for key level identification
        self.min_conviction_for_level_reporting = self.settings.get("min_conviction_for_level_reporting", 0.5)
        self.proximity_cluster_threshold_pct = self.settings.get("proximity_cluster_threshold_pct", 0.001) # 0.1% of price
        self.conviction_weights_by_source = self.settings.get("conviction_weights_by_source", {
            "a_mspi": 0.3,
            "nvp": 0.2,
            "sgdhp": 0.25,
            "ugch": 0.25
        })
        self.logger.info("KeyLevelIdentifierV2_5 initialized.")

    def identify_and_score_key_levels(self, 
                                      df_strike: pd.DataFrame, 
                                      und_data: ProcessedUnderlyingAggregatesV2_5) -> KeyLevelsDataV2_5:
        """
        Identifies and scores key levels based on A-MSPI, NVP, SGDHP, and UGCH data.
        """
        self.logger.debug("Identifying and scoring key levels...")
        
        if df_strike.empty:
            self.logger.warning("Strike DataFrame is empty, no key levels to identify.")
            return KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=und_data.timestamp
            )

        current_price = und_data.price
        if current_price is None or current_price <= 0:
            self.logger.warning("Current underlying price is invalid, cannot identify key levels.")
            return KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=und_data.timestamp
            )

        # Ensure necessary columns exist
        required_cols = ['strike', 'a_mspi_strike', 'nvp_at_strike', 'sgdhp_score_strike', 'ugch_score_strike']
        for col in required_cols:
            if col not in df_strike.columns:
                self.logger.warning(f"Missing required column for key level identification: {col}. Skipping.")
                return KeyLevelsDataV2_5(
                    supports=[],
                    resistances=[],
                    pin_zones=[],
                    vol_triggers=[],
                    major_walls=[],
                    timestamp=und_data.timestamp
                )

        # --- Individual Source Level Identification ---
        potential_levels: List[Dict[str, Any]] = []

        # A-MSPI based levels
        if 'a_mspi_strike' in df_strike.columns:
            # Identify peaks (resistance) and troughs (support) in A-MSPI
            # Using a simple peak detection for now, can be enhanced
            a_mspi_series = df_strike['a_mspi_strike'].fillna(0)
            
            # Peaks for resistance (positive A-MSPI)
            peaks, _ = signal.find_peaks(a_mspi_series, height=a_mspi_series.quantile(0.75))
            for peak_idx in peaks:
                strike = float(df_strike.iloc[peak_idx]['strike'])
                if strike > current_price: # Only consider as resistance if above current price
                    potential_levels.append({'level': strike, 'type': 'AMSPI_Resistance', 'score': float(a_mspi_series.iloc[peak_idx]), 'source': 'a_mspi'})

            # Troughs for support (negative A-MSPI, so find peaks in -A-MSPI)
            troughs, _ = signal.find_peaks(-a_mspi_series, height=(-a_mspi_series).quantile(0.75))
            for trough_idx in troughs:
                strike = float(df_strike.iloc[trough_idx]['strike'])
                if strike < current_price: # Only consider as support if below current price
                    potential_levels.append({'level': strike, 'type': 'AMSPI_Support', 'score': float(-a_mspi_series.iloc[trough_idx]), 'source': 'a_mspi'})

        # NVP based levels (simplified: look for significant positive/negative NVP)
        if 'nvp_at_strike' in df_strike.columns:
            nvp_series = df_strike['nvp_at_strike'].fillna(0)
            nvp_threshold = self.config_manager.get_setting("key_level_settings.nvp_threshold", 1000000) # Example threshold
            
            # Positive NVP (potential support)
            positive_nvp_strikes = df_strike[nvp_series > nvp_threshold]['strike']
            for strike in positive_nvp_strikes:
                if strike < current_price: # Only consider as support if below current price
                    potential_levels.append({'level': strike, 'type': 'NVP_Support', 'score': nvp_series.loc[strike], 'source': 'nvp'})
            
            # Negative NVP (potential resistance)
            negative_nvp_strikes = df_strike[nvp_series < -nvp_threshold]['strike']
            for strike in negative_nvp_strikes:
                if strike > current_price: # Only consider as resistance if above current price
                    potential_levels.append({'level': strike, 'type': 'NVP_Resistance', 'score': abs(nvp_series.loc[strike]), 'source': 'nvp'})

        # SGDHP based levels
        if 'sgdhp_score_strike' in df_strike.columns:
            sgdhp_series = df_strike['sgdhp_score_strike'].fillna(0)
            sgdhp_threshold = self.config_manager.get_setting("key_level_settings.sgdhp_threshold", 0.5) # Example threshold

            # Positive SGDHP (potential support)
            positive_sgdhp_strikes = df_strike[sgdhp_series > sgdhp_threshold]['strike']
            for strike in positive_sgdhp_strikes:
                if strike < current_price: # Only consider as support if below current price
                    potential_levels.append({'level': strike, 'type': 'SGDHP_Support_Wall', 'score': sgdhp_series.loc[strike], 'source': 'sgdhp'})
            
            # Negative SGDHP (potential resistance)
            negative_sgdhp_strikes = df_strike[sgdhp_series < -sgdhp_threshold]['strike']
            for strike in negative_sgdhp_strikes:
                if strike > current_price: # Only consider as resistance if above current price
                    potential_levels.append({'level': strike, 'type': 'SGDHP_Resistance_Wall', 'score': abs(sgdhp_series.loc[strike]), 'source': 'sgdhp'})

        # UGCH based levels
        if 'ugch_score_strike' in df_strike.columns:
            ugch_series = df_strike['ugch_score_strike'].fillna(0)
            ugch_threshold = self.config_manager.get_setting("key_level_settings.ugch_threshold", 0.7) # Example threshold

            # Positive UGCH (potential support)
            positive_ugch_strikes = df_strike[ugch_series > ugch_threshold]['strike']
            for strike in positive_ugch_strikes:
                if strike < current_price: # Only consider as support if below current price
                    potential_levels.append({'level': strike, 'type': 'UGCH_Major_Support_Zone', 'score': ugch_series.loc[strike], 'source': 'ugch'})
            
            # Negative UGCH (potential resistance)
            negative_ugch_strikes = df_strike[ugch_series < -ugch_threshold]['strike']
            for strike in negative_ugch_strikes:
                if strike > current_price: # Only consider as resistance if above current price
                    potential_levels.append({'level': strike, 'type': 'UGCH_Major_Resistance_Zone', 'score': abs(ugch_series.loc[strike]), 'source': 'ugch'})

        # --- Consolidate and Score Levels ---
        final_levels: Dict[float, Dict[str, Any]] = {}

        for level_info in potential_levels:
            level_price = level_info['level']
            level_type = level_info['type']
            score = level_info['score']
            source = level_info['source']

            # Check for existing level within proximity
            found_existing = False
            for existing_price in list(final_levels.keys()): # Iterate over a copy of keys
                if abs(existing_price - level_price) / current_price < self.proximity_cluster_threshold_pct:
                    # Found a nearby existing level, consolidate
                    existing_level = final_levels[existing_price]
                    
                    # Update type if more specific/stronger
                    if existing_level['type'] == 'AMSPI_Support' and 'NVP_Support' in level_type: existing_level['type'] = 'NVP_Support'
                    # Add more type consolidation logic as needed

                    # Accumulate scores based on source weights
                    weight = self.conviction_weights_by_source.get(source, 0.0)
                    existing_level['raw_score'] = existing_level.get('raw_score', 0.0) + (score * weight)
                    existing_level['sources'].add(source)
                    found_existing = True
                    break
            
            if not found_existing:
                # Add new level
                new_level = {'level': level_price, 'type': level_type, 'raw_score': score * self.conviction_weights_by_source.get(source, 0.0), 'sources': {source}}
                final_levels[level_price] = new_level

        # Finalize conviction scores and filter
        supports_list: List[KeyLevelV2_5] = []
        resistances_list: List[KeyLevelV2_5] = []

        max_raw_score = max([lvl['raw_score'] for lvl in final_levels.values()]) if final_levels else 1.0

        for lvl_data in final_levels.values():
            # Normalize raw_score to 0-1 conviction
            conviction = lvl_data['raw_score'] / (max_raw_score + EPSILON)
            conviction = min(1.0, max(0.0, conviction)) # Ensure bounds

            if conviction >= self.min_conviction_for_level_reporting:
                key_level = KeyLevelV2_5(
                    level_price=lvl_data['level'],
                    level_type=lvl_data['type'],
                    conviction_score=conviction,
                    contributing_metrics=list(lvl_data['sources']),
                    source_identifier="key_level_identifier_v2_5"
                )
                if key_level.level_price < current_price: # Simple classification for now
                    supports_list.append(key_level)
                else:
                    resistances_list.append(key_level)

        # Sort by proximity to current price or by level price
        supports_list.sort(key=lambda x: abs(x.level_price - current_price))
        resistances_list.sort(key=lambda x: abs(x.level_price - current_price))

        self.logger.debug(f"Identified {len(supports_list)} support levels and {len(resistances_list)} resistance levels.")
        return KeyLevelsDataV2_5(
            supports=supports_list,
            resistances=resistances_list,
            pin_zones=[],
            vol_triggers=[],
            major_walls=[],
            timestamp=und_data.timestamp
        )

    # --- Vectorized Helper Functions (Moved from MarketIntelligenceEngine) ---
    def vectorized_gamma_analysis(
        self,
        strikes: np.ndarray,
        gammas: np.ndarray,
        volumes: np.ndarray,
        threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Vectorized analysis of gamma concentrations for key level identification.
        
        Args:
            strikes: Array of strike prices
            gammas: Array of gamma values
            volumes: Array of trading volumes
            threshold: Minimum threshold for significance
            
        Returns:
            Dictionary mapping level names to strike prices
        """
        # Calculate volume-weighted gamma
        vw_gamma = gammas * volumes
        
        # Find peaks in gamma concentration
        peaks, _ = signal.find_peaks(vw_gamma, height=np.max(vw_gamma) * threshold)
        
        # Calculate significance scores
        significance = vw_gamma[peaks] / (np.sum(vw_gamma) + EPSILON)
        
        return {
            f"gamma_level_{i}": float(strikes[peak])
            for i, peak in enumerate(peaks)
            if significance[i] > threshold
        }

    def vectorized_volume_profile(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        n_bins: int = 50,
        min_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Vectorized volume profile analysis for key level identification.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            n_bins: Number of bins for histogram
            min_threshold: Minimum threshold for significance
            
        Returns:
            Dictionary mapping level names to price levels
        """
        # Create volume profile
        hist, bins = np.histogram(prices, bins=n_bins, weights=volumes)
        
        # Find high volume nodes
        threshold = np.max(hist) * min_threshold
        significant_bins = hist > threshold
        
        # Calculate price levels for significant volume nodes
        price_levels = (bins[:-1][significant_bins] + bins[1:][significant_bins]) / 2
        
        return {
            f"volume_level_{i}": float(level)
            for i, level in enumerate(price_levels)
        }

    def vectorized_technical_analysis(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        window_size: int = 20
    ) -> Dict[str, float]:
        """
        Vectorized technical analysis for support/resistance identification.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            window_size: Size of the rolling window
            
        Returns:
            Dictionary of technical levels and indicators
        """
        # Calculate moving averages
        sma = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate Bollinger Bands
        # Ensure rolling_std calculation handles edge cases for small arrays
        if len(prices) < window_size:
            rolling_std = np.std(prices) # Use overall std if not enough data for rolling
        else:
            rolling_std = np.array([np.std(prices[i:i+window_size]) for i in range(len(prices)-window_size+1)])

        # Handle cases where rolling_std might be empty or single value
        if rolling_std.size == 0:
            rolling_std = np.array([0.0])
        elif rolling_std.size == 1:
            rolling_std = np.full_like(sma, rolling_std.item()) # Expand to match sma size
        elif len(rolling_std) < len(sma):
            # Pad or interpolate rolling_std if its length is less than sma
            # For simplicity, we'll just use the last valid value to pad
            last_val = rolling_std[-1] if rolling_std.size > 0 else 0.0
            rolling_std = np.pad(rolling_std, (0, len(sma) - len(rolling_std)), mode='constant', constant_values=last_val)

        upper_band = sma + 2 * rolling_std
        lower_band = sma - 2 * rolling_std
        
        # Identify potential support/resistance levels
        levels = {
            'sma': float(sma[-1]) if sma.size > 0 else 0.0,
            'upper_band': float(upper_band[-1]) if upper_band.size > 0 else 0.0,
            'lower_band': float(lower_band[-1]) if lower_band.size > 0 else 0.0
        }
        
        # Add recent swing points
        swings, _ = signal.find_peaks(prices, distance=window_size)
        if len(swings) > 0:
            levels['swing_high'] = float(prices[swings[-1]])
        
        swings_low, _ = signal.find_peaks(-prices, distance=window_size)
        if len(swings_low) > 0:
            levels['swing_low'] = float(prices[swings_low[-1]])
        
        return levels
