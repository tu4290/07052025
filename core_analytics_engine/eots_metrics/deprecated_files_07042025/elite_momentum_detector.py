import pandas as pd
import numpy as np
import math
from typing import Dict, Any

from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig

class EliteMomentumDetector:
    """Advanced momentum and acceleration detection"""
    
    def __init__(self, config_manager=None, historical_data_manager=None, enhanced_cache_manager=None, elite_config=None):
        # Accept all harmonized arguments for orchestrator compatibility
        if elite_config is not None:
            self.config = elite_config
        elif config_manager is not None and hasattr(config_manager, 'elite_config'):
            self.config = config_manager.elite_config
        else:
            self.config = EliteConfig()
        self.momentum_cache = {}
        
    def calculate_flow_velocity(self, flow_series: pd.Series, period: int = 5) -> float:
        """Calculate flow velocity (rate of change)"""
        if len(flow_series) < period:
            return 0.0
        
        try:
            # Calculate rolling differences
            velocity = flow_series.diff(period).iloc[-1]
            if isinstance(velocity, type):
                return 0.0
            if not isinstance(velocity, (float, int)):
                return 0.0
            if pd.isna(velocity):
                return 0.0
            return float(velocity)
        except:
            return 0.0
    
    def calculate_flow_acceleration(self, flow_series: pd.Series, period: int = 5) -> float:
        """Calculate flow acceleration (rate of change of velocity)"""
        if len(flow_series) < period * 2:
            return 0.0
        
        try:
            # Calculate velocity series
            velocity_series = flow_series.diff(period)
            # Calculate acceleration as change in velocity
            acceleration = velocity_series.diff(period).iloc[-1]
            return float(acceleration) if not pd.isna(acceleration) else 0.0
        except:
            return 0.0
    
    def calculate_momentum_persistence(self, flow_series: pd.Series, threshold: float = 0.7) -> float:
        """Calculate momentum persistence score"""
        if len(flow_series) < 10:
            return 0.0
        
        try:
            # Calculate directional consistency
            changes = flow_series.diff().dropna()
            changes = changes.astype(float)
            if len(changes) == 0:
                return 0.0
            
            positive_changes = (changes > 0).sum()
            total_changes = len(changes)
            persistence = positive_changes / total_changes
            
            # Adjust for magnitude
            avg_magnitude = changes.abs().mean()
            persistence_score = persistence * min(1.0, avg_magnitude / flow_series.std())
            
            return float(persistence_score)
        except:
            return 0.0
