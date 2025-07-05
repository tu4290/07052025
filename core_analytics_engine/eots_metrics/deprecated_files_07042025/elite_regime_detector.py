import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from core_analytics_engine.eots_metrics.elite_definitions import EliteConfig, MarketRegime, ConvexValueColumns

logger = logging.getLogger(__name__)

class EliteMarketRegimeDetector:
    """Advanced market regime detection using machine learning"""
    
    def __init__(self, config: EliteConfig):
        self.config = config
        self.regime_model = RandomForestClassifier(n_estimators=100, random_state=42) # Example model
        self.scaler = StandardScaler()
        self.is_trained = False # In a real scenario, this would be loaded from a pre-trained model
        
    def extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification"""
        features = []
        
        # Volatility features
        if 'volatility' in market_data.columns:
            vol_series = market_data['volatility'].dropna()
            if len(vol_series) > 0:
                features.extend([
                    vol_series.mean(),
                    vol_series.std(),
                    vol_series.rolling(min(len(vol_series), 20)).mean().iloc[-1] if len(vol_series) >= 20 else vol_series.mean(),
                    vol_series.rolling(min(len(vol_series), 5)).std().iloc[-1] if len(vol_series) >= 5 else vol_series.std()
                ])
            else:
                features.extend([0.2, 0.05, 0.2, 0.05])  # Default values
        else:
            features.extend([0.2, 0.05, 0.2, 0.05])
            
        # Price momentum features
        if 'price' in market_data.columns:
            price_series = market_data['price'].dropna()
            if len(price_series) > 1:
                returns = price_series.pct_change().dropna()
                features.extend([
                    returns.mean(),
                    returns.std(),
                    (price_series.iloc[-1] / price_series.iloc[0] - 1) if len(price_series) > 0 else 0,
                    returns.rolling(min(len(returns), 10)).mean().iloc[-1] if len(returns) >= 10 else returns.mean()
                ])
            else:
                features.extend([0, 0.02, 0, 0])
        else:
            features.extend([0, 0.02, 0, 0])
            
        # Flow features
        flow_cols = [ConvexValueColumns.VOLMBS_15M, ConvexValueColumns.VALUE_BS]
        for col in flow_cols:
            if col in market_data.columns:
                series = market_data[col].dropna()
                if len(series) > 0:
                    features.extend([series.mean(), series.std()])
                else:
                    features.extend([0, 1])
            else:
                features.extend([0, 1])
                
        return np.array(features).reshape(1, -1)
    
    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            features = self.extract_regime_features(market_data)
            
            # Simple rule-based regime detection if model not trained
            if not self.is_trained:
                return self._rule_based_regime_detection(market_data)
            
            # Use trained model for regime prediction
            features_scaled = self.scaler.transform(features)
            regime_idx = self.regime_model.predict(features_scaled)[0]
            regimes = list(MarketRegime)
            return regimes[min(regime_idx, len(regimes) - 1)]
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}, using default")
            return MarketRegime.MEDIUM_VOL_RANGING
    
    def _rule_based_regime_detection(self, market_data: pd.DataFrame) -> MarketRegime:
        """Fallback rule-based regime detection"""
        # Simple volatility-based regime classification
        if 'volatility' in market_data.columns:
            vol_mean = market_data['volatility'].mean()
            if vol_mean > 0.3:
                return MarketRegime.HIGH_VOL_TRENDING
            elif vol_mean > 0.2:
                return MarketRegime.MEDIUM_VOL_RANGING
            else:
                return MarketRegime.LOW_VOL_RANGING
        
        return MarketRegime.MEDIUM_VOL_RANGING