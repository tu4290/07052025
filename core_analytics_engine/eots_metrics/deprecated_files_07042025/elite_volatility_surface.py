import pandas as pd
import numpy as np
from typing import Dict, Any
from functools import lru_cache, wraps

from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, ConvexValueColumns

# Decorator for caching results
def cache_result(maxsize=128):
    """Enhanced caching decorator with configurable size"""
    def decorator(func):
        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

class EliteVolatilitySurface:
    """Advanced volatility surface modeling and analysis"""
    
    def __init__(self, config: EliteConfig):
        self.config = config
        self.surface_cache = {}
        
    @cache_result(maxsize=64)
    def calculate_skew_adjustment(self, strike: float, atm_vol: float, 
                                strike_vol: float, alpha: float = 1.0) -> float:
        """Calculate skew adjustment factor"""
        if atm_vol <= 0 or strike_vol <= 0:
            return 1.0
        
        skew_ratio = strike_vol / atm_vol
        adjustment = 1.0 + alpha * (skew_ratio - 1.0)
        return max(0.1, min(3.0, adjustment))  # Bounded adjustment
    
    def get_volatility_regime(self, options_data: pd.DataFrame) -> str:
        """Determine volatility regime"""
        if ConvexValueColumns.VOLATILITY not in options_data.columns:
            return "medium_vol"

        vol_series = pd.to_numeric(options_data[ConvexValueColumns.VOLATILITY], errors='coerce').dropna()
        if len(vol_series) == 0:
            return "medium_vol"

        vol_mean = vol_series.mean()
        vol_std = vol_series.std()

        if vol_mean > 0.4:
            return "high_vol"
        elif vol_mean < 0.15:
            return "low_vol"
        elif vol_std > 0.1:
            return "unstable"
        else:
            return "medium_vol"