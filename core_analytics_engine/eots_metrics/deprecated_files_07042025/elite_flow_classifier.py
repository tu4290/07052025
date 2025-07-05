import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier # Example model

from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, FlowType, ConvexValueColumns

logger = logging.getLogger(__name__)

class EliteFlowClassifier:
    """Advanced institutional flow classification"""
    
    def __init__(self, config_manager=None, historical_data_manager=None, enhanced_cache_manager=None, elite_config=None):
        # Accept all harmonized arguments for orchestrator compatibility
        if elite_config is not None:
            self.config = elite_config
        elif config_manager is not None and hasattr(config_manager, 'elite_config'):
            self.config = config_manager.elite_config
        else:
            self.config = EliteConfig()
        self.flow_model = None # In a real scenario, this would be loaded from a pre-trained model
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_flow_features(self, options_data: pd.DataFrame) -> np.ndarray:
        """Extract features for flow classification"""
        features = []
        
        # Volume-based features
        volume_cols = [ConvexValueColumns.VOLMBS_5M, ConvexValueColumns.VOLMBS_15M, 
                      ConvexValueColumns.VOLMBS_30M, ConvexValueColumns.VOLMBS_60M]
        
        for col in volume_cols:
            if col in options_data.columns:
                series = pd.to_numeric(options_data[col], errors='coerce').dropna()
                if len(series) > 0:
                    features.extend([
                        series.abs().mean(),  # Average absolute flow
                        series.std(),         # Flow volatility
                        series.sum()          # Net flow
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        
        # Value-based features
        value_cols = [ConvexValueColumns.VALUEBS_5M, ConvexValueColumns.VALUEBS_15M,
                     ConvexValueColumns.VALUEBS_30M, ConvexValueColumns.VALUEBS_60M]
        
        for col in value_cols:
            if col in options_data.columns:
                series = pd.to_numeric(options_data[col], errors='coerce').dropna()
                if len(series) > 0:
                    features.extend([series.abs().mean(), series.sum()])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
        
        # Greek-based features (using volume-multiplied Greeks as proxies for flow)
        greek_cols = [ConvexValueColumns.GXVOLM, ConvexValueColumns.DXVOLM, ConvexValueColumns.VXVOLM]
        for col in greek_cols:
            if col in options_data.columns:
                series = pd.to_numeric(options_data[col], errors='coerce').dropna()
                if len(series) > 0:
                    features.append(series.abs().sum())
                else:
                    features.append(0)
            else:
                features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def classify_flow(self, options_data: pd.DataFrame) -> FlowType:
        """Classify flow type"""
        try:
            features = self.extract_flow_features(options_data)
            
            if not self.is_trained or self.flow_model is None:
                return self._rule_based_flow_classification(options_data)
            
            features_scaled = self.scaler.transform(features)
            if self.flow_model is not None:
                flow_idx = self.flow_model.predict(features_scaled)[0]
                flow_types = list(FlowType)
                return flow_types[min(flow_idx, len(flow_types) - 1)]
            else:
                return self._rule_based_flow_classification(options_data)
            
        except Exception as e:
            logger.warning(f"Flow classification failed: {e}, using default")
            return FlowType.UNKNOWN
    
    def _rule_based_flow_classification(self, options_data: pd.DataFrame) -> FlowType:
        """Fallback rule-based flow classification"""
        # Simple volume-based classification
        if ConvexValueColumns.VOLMBS_15M in options_data.columns:
            vol_15m = pd.to_numeric(options_data[ConvexValueColumns.VOLMBS_15M], errors='coerce').abs().sum()
            if vol_15m > 10000:
                return FlowType.INSTITUTIONAL_LARGE
            elif vol_15m > 1000:
                return FlowType.INSTITUTIONAL_SMALL
            else:
                return FlowType.RETAIL_SOPHISTICATED
        
        return FlowType.UNKNOWN