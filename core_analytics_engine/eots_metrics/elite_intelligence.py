import logging
import numpy as np
import pandas as pd
from typing import Optional, Any
from data_models.elite_intelligence import (
    EliteConfig, EliteImpactResultsV2_5, MarketRegime, FlowType, VolatilityRegime,
    ConvexValueColumns, EliteImpactColumns
)

logger = logging.getLogger(__name__)

class EliteImpactCalculator:
    """
    Consolidated elite impact calculator combining institutional intelligence
    and advanced flow analysis
    """
    
    def __init__(self, config: Optional[EliteConfig] = None):
        """Initialize with configuration"""
        self.config = config or EliteConfig()
        self.cv_cols = ConvexValueColumns()
        self.elite_cols = EliteImpactColumns()
        
    def calculate_elite_impact_score(
        self,
        options_df: pd.DataFrame,
        underlying_data: Any
    ) -> EliteImpactResultsV2_5:
        """
        Calculate comprehensive elite impact score
        
        Args:
            options_df: DataFrame with options data
            underlying_data: Underlying data (Pydantic model or dict)
        
        Returns:
            EliteImpactResultsV2_5 with calculated scores
        """
        try:
            # Handle empty DataFrame
            if options_df.empty:
                return self._get_default_results()
            
            # Calculate components
            institutional_score = self._calculate_institutional_flow_score(options_df)
            flow_momentum = self._calculate_flow_momentum_index(options_df)
            market_regime = self._detect_market_regime(underlying_data)
            flow_type = self._classify_flow_type(options_df)
            volatility_regime = self._detect_volatility_regime(underlying_data)
            
            # Calculate confidence and transition risk
            confidence = self._calculate_confidence(options_df, underlying_data)
            transition_risk = self._calculate_transition_risk(underlying_data)
            
            # Calculate overall elite impact score
            elite_impact_score = self._calculate_overall_elite_score(
                institutional_score, flow_momentum, confidence
            )
            
            # Get additional metrics
            large_trades = self._count_large_trades(options_df)
            inst_volume_ratio = self._calculate_institutional_volume_ratio(options_df)
            smart_money = self._calculate_smart_money_indicator(options_df)
            
            return EliteImpactResultsV2_5(
                elite_impact_score_und=elite_impact_score,
                institutional_flow_score_und=institutional_score,
                flow_momentum_index_und=flow_momentum,
                market_regime_elite=market_regime.value,
                flow_type_elite=flow_type.value,
                volatility_regime_elite=volatility_regime.value,
                confidence=confidence,
                transition_risk=transition_risk,
                large_trade_count=large_trades,
                institutional_volume_ratio=inst_volume_ratio,
                smart_money_indicator=smart_money
            )
            
        except Exception as e:
            logger.error(f"Error calculating elite impact: {e}")
            return self._get_default_results()
    
    def _calculate_institutional_flow_score(self, options_df: pd.DataFrame) -> float:
        """Calculate institutional flow score based on trade characteristics"""
        if options_df.empty:
            return 50.0
        
        try:
            # Calculate volume-weighted metrics
            if self.cv_cols.VOLUME in options_df.columns:
                total_volume = options_df[self.cv_cols.VOLUME].sum()
                if total_volume > 0:
                    # Large trade volume
                    large_trades = options_df[
                        options_df[self.cv_cols.VOLUME] > self.config.large_trade_threshold
                    ]
                    large_volume_ratio = large_trades[self.cv_cols.VOLUME].sum() / total_volume
                    
                    # Greek exposure from large trades
                    if self.cv_cols.GAMMA in large_trades.columns:
                        large_gamma_ratio = abs(large_trades[self.cv_cols.GAMMA].sum()) / max(
                            abs(options_df[self.cv_cols.GAMMA].sum()), 1
                        )
                    else:
                        large_gamma_ratio = 0
                    
                    # Calculate score
                    score = (
                        large_volume_ratio * 40 +
                        large_gamma_ratio * 30 +
                        min(len(large_trades) / 10, 1) * 30
                    )
                    
                    return min(max(score * 100, 0), 100)
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating institutional flow score: {e}")
            return 50.0
    
    def _calculate_flow_momentum_index(self, options_df: pd.DataFrame) -> float:
        """Calculate flow momentum index"""
        if options_df.empty:
            return 0.0
        
        try:
            # Calculate directional flow
            if self.cv_cols.VOLUME_BS in options_df.columns:
                buy_volume = options_df[options_df[self.cv_cols.VOLUME_BS] > 0][
                    self.cv_cols.VOLUME
                ].sum()
                sell_volume = options_df[options_df[self.cv_cols.VOLUME_BS] < 0][
                    self.cv_cols.VOLUME
                ].sum()
                
                total_volume = buy_volume + sell_volume
                if total_volume > 0:
                    momentum = (buy_volume - sell_volume) / total_volume
                    return momentum
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating flow momentum: {e}")
            return 0.0
    
    def _detect_market_regime(self, underlying_data: Any) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Extract volatility and trend data
            if hasattr(underlying_data, 'u_volatility'):
                volatility = underlying_data.u_volatility
            else:
                volatility = 0.2  # Default
            
            if hasattr(underlying_data, 'trend_strength'):
                trend_strength = underlying_data.trend_strength
            else:
                trend_strength = 0.5  # Default
            
            # Classify regime
            if volatility < 0.15:
                if trend_strength > 0.7:
                    return MarketRegime.LOW_VOL_TRENDING
                else:
                    return MarketRegime.LOW_VOL_RANGING
            elif volatility < 0.25:
                if trend_strength > 0.7:
                    return MarketRegime.MEDIUM_VOL_TRENDING
                else:
                    return MarketRegime.MEDIUM_VOL_RANGING
            elif volatility < 0.40:
                if trend_strength > 0.7:
                    return MarketRegime.HIGH_VOL_TRENDING
                else:
                    return MarketRegime.HIGH_VOL_RANGING
            else:
                return MarketRegime.STRESS_REGIME
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.REGIME_UNCLEAR_OR_TRANSITIONING
    
    def _classify_flow_type(self, options_df: pd.DataFrame) -> FlowType:
        """Classify the type of flow"""
        if options_df.empty:
            return FlowType.FLOW_UNCLEAR
        
        try:
            # Analyze trade characteristics
            avg_trade_size = options_df[self.cv_cols.VOLUME].mean()
            
            if avg_trade_size < self.config.flow_size_thresholds['retail']:
                return FlowType.RETAIL_UNSOPHISTICATED
            elif avg_trade_size < self.config.flow_size_thresholds['sophisticated']:
                return FlowType.RETAIL_SOPHISTICATED
            elif avg_trade_size < self.config.flow_size_thresholds['institutional']:
                # Check for hedging patterns
                if self._is_hedging_flow(options_df):
                    return FlowType.INSTITUTIONAL_HEDGING
                else:
                    return FlowType.INSTITUTIONAL_DIRECTIONAL
            else:
                return FlowType.MARKET_MAKER_FLOW
                
        except Exception as e:
            logger.error(f"Error classifying flow type: {e}")
            return FlowType.FLOW_UNCLEAR
    
    def _detect_volatility_regime(self, underlying_data: Any) -> VolatilityRegime:
        """Detect volatility regime"""
        try:
            if hasattr(underlying_data, 'u_volatility'):
                vol = underlying_data.u_volatility
                
                if vol < self.config.volatility_thresholds['subdued']:
                    return VolatilityRegime.SUBDUED
                elif vol < self.config.volatility_thresholds['normal']:
                    return VolatilityRegime.NORMAL
                elif vol < self.config.volatility_thresholds['elevated']:
                    return VolatilityRegime.ELEVATED
                else:
                    return VolatilityRegime.EXTREME
            
            return VolatilityRegime.NORMAL
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return VolatilityRegime.NORMAL
    
    def _calculate_confidence(self, options_df: pd.DataFrame, underlying_data: Any) -> float:
        """Calculate confidence in the analysis"""
        try:
            confidence_factors = []
            
            # Data quality factor
            if not options_df.empty:
                data_completeness = 1 - options_df.isnull().sum().sum() / (
                    len(options_df) * len(options_df.columns)
                )
                confidence_factors.append(data_completeness)
            
            # Volume factor
            if self.cv_cols.VOLUME in options_df.columns:
                total_volume = options_df[self.cv_cols.VOLUME].sum()
                volume_confidence = min(total_volume / 10000, 1.0)
                confidence_factors.append(volume_confidence)
            
            # Calculate average confidence
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_transition_risk(self, underlying_data: Any) -> float:
        """Calculate risk of regime transition"""
        try:
            # Simple implementation based on volatility
            if hasattr(underlying_data, 'u_volatility'):
                vol = underlying_data.u_volatility
                # Higher volatility = higher transition risk
                return min(vol * 2, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating transition risk: {e}")
            return 0.5
    
    def _calculate_overall_elite_score(
        self,
        institutional_score: float,
        flow_momentum: float,
        confidence: float
    ) -> float:
        """Calculate overall elite impact score"""
        try:
            # Weighted combination
            base_score = (
                institutional_score * 0.6 +
                abs(flow_momentum) * 100 * 0.4
            )
            
            # Apply confidence adjustment
            return base_score * (0.5 + confidence * 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating elite score: {e}")
            return 50.0
    
    def _count_large_trades(self, options_df: pd.DataFrame) -> int:
        """Count number of large trades"""
        if options_df.empty or self.cv_cols.VOLUME not in options_df.columns:
            return 0
        
        return len(options_df[options_df[self.cv_cols.VOLUME] > self.config.large_trade_threshold])
    
    def _calculate_institutional_volume_ratio(self, options_df: pd.DataFrame) -> float:
        """Calculate ratio of institutional volume"""
        if options_df.empty or self.cv_cols.VOLUME not in options_df.columns:
            return 0.0
        
        total_volume = options_df[self.cv_cols.VOLUME].sum()
        if total_volume == 0:
            return 0.0
        
        large_volume = options_df[
            options_df[self.cv_cols.VOLUME] > self.config.large_trade_threshold
        ][self.cv_cols.VOLUME].sum()
        
        return large_volume / total_volume
    
    def _calculate_smart_money_indicator(self, options_df: pd.DataFrame) -> float:
        """Calculate smart money flow indicator"""
        if options_df.empty:
            return 0.0
        
        try:
            # Look for smart money patterns (large trades at key strikes)
            if self.cv_cols.VOLUME in options_df.columns and self.cv_cols.STRIKE in options_df.columns:
                # Find strikes with large volume
                strike_volumes = options_df.groupby(self.cv_cols.STRIKE)[
                    self.cv_cols.VOLUME
                ].sum().sort_values(ascending=False)
                
                if len(strike_volumes) > 0:
                    # Check concentration at top strikes
                    top_strikes_volume = strike_volumes.head(3).sum()
                    total_volume = strike_volumes.sum()
                    
                    if total_volume > 0:
                        concentration = top_strikes_volume / total_volume
                        return min(concentration * 100, 100)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating smart money indicator: {e}")
            return 0.0
    
    def _is_hedging_flow(self, options_df: pd.DataFrame) -> bool:
        """Check if flow appears to be hedging"""
        try:
            if self.cv_cols.OPTION_TYPE in options_df.columns:
                # Check for balanced put/call activity
                call_volume = options_df[
                    options_df[self.cv_cols.OPTION_TYPE] == 'call'
                ][self.cv_cols.VOLUME].sum()
                
                put_volume = options_df[
                    options_df[self.cv_cols.OPTION_TYPE] == 'put'
                ][self.cv_cols.VOLUME].sum()
                
                total_volume = call_volume + put_volume
                if total_volume > 0:
                    ratio = min(call_volume, put_volume) / max(call_volume, put_volume)
                    return ratio > 0.7  # Relatively balanced
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking hedging flow: {e}")
            return False
    
    def _get_default_results(self) -> EliteImpactResultsV2_5:
        """Get default results when calculation fails"""
        return EliteImpactResultsV2_5(
            elite_impact_score_und=50.0,
            institutional_flow_score_und=50.0,
            flow_momentum_index_und=0.0,
            market_regime_elite=MarketRegime.REGIME_UNCLEAR_OR_TRANSITIONING.value,
            flow_type_elite=FlowType.FLOW_UNCLEAR.value,
            volatility_regime_elite=VolatilityRegime.NORMAL.value,
            confidence=0.5,
            transition_risk=0.5,
            large_trade_count=0,
            institutional_volume_ratio=0.0,
            smart_money_indicator=0.0
        )

__all__ = ['EliteImpactCalculator']
