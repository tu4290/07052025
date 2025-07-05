"""
Market Intelligence Engine v2.5 - EOTS "Apex Predator"

A consolidated engine that combines the functionality of:
- Ticker Context Analyzer
- Signal Generator
- Recommendation Logic
- Key Level Identifier
- Trade Parameter Optimizer

This engine provides a unified interface for market analysis, signal generation,
trading recommendations, parameter optimization, and key level identification.

Author: EOTS Development Team
Version: 2.5.0
Last Updated: 2024
"""

import logging
import numpy as np
import uuid
from typing import Dict, List, Optional, Any
from typing_extensions import TypeAlias
from datetime import datetime, timedelta

from scipy import signal
from numpy.typing import NDArray

# Import Pydantic models
from data_models import (
    ProcessedDataBundleV2_5
)
from data_models.deprecated_files import SignalPayloadV2_5
from data_models import MarketRegimeState
from data_models.core_models import UnifiedIntelligenceAnalysis
from data_models.ai_ml_models import MarketPattern, PatternThresholds
from data_models.ai_ml_models import (
    HuiHuiMarketRegimeSchema,
    HuiHuiOptionsFlowSchema,
    HuiHuiSentimentSchema
)

# Import utilities
from core_analytics_engine.news_intelligence_engine_v2_5 import NewsIntelligenceEngineV2_5

# Local imports
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5, CacheLevel
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5

logger = logging.getLogger(__name__)

# Type aliases for numpy arrays
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

# Helper Functions for Vectorized Operations
def vectorized_gamma_analysis(
    strikes: FloatArray,
    gammas: FloatArray,
    volumes: FloatArray,
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
    significance = vw_gamma[peaks] / np.sum(vw_gamma)
    
    return {
        f"gamma_level_{i}": float(strikes[peak])
        for i, peak in enumerate(peaks)
        if significance[i] > threshold
    }

def vectorized_volume_profile(
    prices: FloatArray,
    volumes: FloatArray,
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
    prices: FloatArray,
    volumes: FloatArray,
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
    rolling_std = np.std([prices[i:i+window_size] for i in range(len(prices)-window_size+1)], axis=1)
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std
    
    # Identify potential support/resistance levels
    levels = {
        'sma': float(sma[-1]),
        'upper_band': float(upper_band[-1]),
        'lower_band': float(lower_band[-1])
    }
    
    # Add recent swing points
    swings = signal.find_peaks(prices, distance=window_size)[0]
    if len(swings) > 0:
        levels['swing_high'] = float(prices[swings[-1]])
    
    swings_low = signal.find_peaks(-prices, distance=window_size)[0]
    if len(swings_low) > 0:
        levels['swing_low'] = float(prices[swings_low[-1]])
    
    return levels

def vectorized_flow_analysis(
    strikes: FloatArray,
    call_volumes: FloatArray,
    put_volumes: FloatArray,
    call_deltas: FloatArray,
    put_deltas: FloatArray
) -> Dict[str, FloatArray]:
    """
    Vectorized analysis of options flow data.
    
    Args:
        strikes: Array of strike prices
        call_volumes: Array of call option volumes
        put_volumes: Array of put option volumes
        call_deltas: Array of call option deltas
        put_deltas: Array of put option deltas
        
    Returns:
        Dictionary of flow metrics arrays
    """
    # Calculate net delta flow
    net_delta_flow = (call_volumes * call_deltas) - (put_volumes * put_deltas)
    
    # Calculate volume imbalance
    volume_imbalance = call_volumes - put_volumes
    
    # Calculate delta-weighted flow
    delta_weighted_flow = net_delta_flow * np.abs(call_deltas - put_deltas)
    
    return {
        'net_delta_flow': net_delta_flow,
        'volume_imbalance': volume_imbalance,
        'delta_weighted_flow': delta_weighted_flow,
        'call_put_ratio': np.where(put_volumes > 0, call_volumes / put_volumes, np.inf)
    }

def calculate_regime_metrics(
    prices: FloatArray,
    volumes: FloatArray,
    window_size: int = 20
) -> Dict[str, float]:
    """
    Calculate market regime metrics using vectorized operations.
    
    Args:
        prices: Array of price values
        volumes: Array of volume values
        window_size: Size of the rolling window
        
    Returns:
        Dictionary of regime metrics
    """
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # Calculate volatility
    volatility = np.std(returns[-window_size:]) * np.sqrt(252)
    
    # Calculate trend strength
    trend = np.polyfit(np.arange(len(prices[-window_size:])), prices[-window_size:], 1)[0]
    trend_strength = abs(trend) / np.mean(prices[-window_size:])
    
    # Calculate volume trend
    volume_trend = np.polyfit(np.arange(len(volumes[-window_size:])), volumes[-window_size:], 1)[0]
    
    # Calculate momentum
    momentum = (prices[-1] / prices[-window_size] - 1) * 100
    
    return {
        'volatility': float(volatility),
        'trend_strength': float(trend_strength),
        'volume_trend': float(volume_trend),
        'momentum': float(momentum),
        'regime_score': float(trend_strength * (1 + np.sign(trend) * momentum/100))
    }

# PYDANTIC-FIRST: Pattern Detection Thresholds


class MarketIntelligenceEngineV2_5:
    """
    Market Intelligence Engine v2.5 - EOTS "Apex Predator"
    
    PYDANTIC-FIRST: Fully validated against EOTS schemas.
    This engine provides a unified interface for market analysis, signal generation,
    trading recommendations, parameter optimization, and key level identification.
    """
    
    def __init__(self, config_manager: ConfigManagerV2_5, metrics_calculator: Optional[MetricsCalculatorV2_5] = None, enhanced_cache: Optional[EnhancedCacheManagerV2_5] = None):
        """Initialize Market Intelligence Engine with Pydantic validation."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.config = config_manager.config.dict().get('market_intelligence_settings', {})
        
        # Initialize metrics calculator if not provided
        self.metrics_calculator = metrics_calculator or MetricsCalculatorV2_5(
            config_manager=config_manager
        )
        
        # Initialize enhanced cache manager
        self._enhanced_cache = enhanced_cache or EnhancedCacheManagerV2_5()
        
        # Initialize pattern types and thresholds
        self.pattern_types = ['VOLATILITY_EXPANSION', 'TREND_CONTINUATION', 'ACCUMULATION', 'DISTRIBUTION', 'CONSOLIDATION']
        self.pattern_thresholds = PatternThresholds(
            volatility_expansion=0.15,
            trend_continuation=0.1,
            accumulation=1.5,
            distribution=1.0,
            consolidation=0.8,
            significant_pos_thresh=0.5,
            dwfd_strong_thresh=0.7,
            moderate_confidence_thresh=0.6
        )

        # Initialize news intelligence engine
        self.news_intelligence = NewsIntelligenceEngineV2_5(config_manager=config_manager)
        
        self.logger.info("✅ Initialized MarketIntelligenceEngineV2_5 in LOCAL CACHE ONLY mode")

    def _store_pattern_longterm(self, symbol: str, pattern: MarketPattern) -> bool:
        """Store pattern in cache instead of database."""
        try:
            cache_key = f"pattern_{symbol}_{pattern.pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._enhanced_cache.set(cache_key, pattern.dict(), level=CacheLevel.LONG_TERM)
            return True
        except Exception as e:
            self.logger.error(f"Failed to store pattern in cache: {e}")
            return False

    def _get_recent_patterns(self, symbol: str, lookback_minutes: int = 30) -> List[MarketPattern]:
        """Get recent patterns from cache."""
        try:
            # Get all pattern keys for this symbol
            pattern_keys = self._enhanced_cache.get_keys(f"pattern_{symbol}_*")
            patterns = []
            
            for key in pattern_keys:
                pattern_data = self._enhanced_cache.get(key)
                if pattern_data:
                    pattern = MarketPattern(**pattern_data)
                    # Check if pattern is within lookback window
                    if (datetime.now() - pattern.timestamp) <= timedelta(minutes=lookback_minutes):
                        patterns.append(pattern)
            
            return patterns
        except Exception as e:
            self.logger.error(f"Failed to get recent patterns from cache: {e}")
            return []

    def _detect_market_patterns(
        self, 
        data_bundle: ProcessedDataBundleV2_5,
        regime_metrics: Dict[str, float],
        flow_metrics: Dict[str, FloatArray],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> List[MarketPattern]:
        """Detect market patterns using real-time analysis."""
        patterns = []
        
        try:
            # Extract market conditions
            market_conditions = {
                'volatility': regime_metrics.get('volatility', 0.0),
                'trend_strength': regime_metrics.get('trend_strength', 0.0),
                'volume_trend': regime_metrics.get('volume_trend', 0.0),
                'price_momentum': regime_metrics.get('momentum', 0.0)
            }

            # Get underlying data
            underlying_data = data_bundle.underlying_data_enriched
            
            # Extract symbol from underlying data
            symbol = underlying_data.symbol if underlying_data else "UNKNOWN"

            # Pattern detection logic
            if underlying_data and underlying_data.price:
                # Volatility Expansion Pattern
                if market_conditions['volatility'] > self.pattern_thresholds.volatility_threshold:
                    patterns.append(MarketPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type='VOLATILITY_EXPANSION',
                        symbol=symbol,
                        confidence_score=min(market_conditions['volatility'] / 2.0, 1.0),
                        eots_metrics={'volatility': market_conditions['volatility']},
                        market_conditions=market_conditions,
                        success_rate=0.0,  # Will be updated later
                        outcome=None  # Will be updated later
                    ))

                # Trend Continuation Pattern
                if market_conditions['trend_strength'] > self.pattern_thresholds.trend_threshold:
                    patterns.append(MarketPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type='TREND_CONTINUATION',
                        symbol=symbol,
                        confidence_score=min(market_conditions['trend_strength'] / 1.5, 1.0),
                        eots_metrics={'trend_strength': market_conditions['trend_strength']},
                        market_conditions=market_conditions,
                        success_rate=0.0,  # Will be updated later
                        outcome=None  # Will be updated later
                    ))

                # Accumulation/Distribution Patterns
                volume_threshold = self.pattern_thresholds.volume_threshold
                if market_conditions['volume_trend'] > volume_threshold:
                    pattern_type = 'ACCUMULATION' if market_conditions['price_momentum'] > 0 else 'DISTRIBUTION'
                    patterns.append(MarketPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=pattern_type,
                        symbol=symbol,
                        confidence_score=min(abs(market_conditions['volume_trend']) / volume_threshold, 1.0),
                        eots_metrics={
                            'volume_trend': market_conditions['volume_trend'],
                            'price_momentum': market_conditions['price_momentum']
                        },
                        market_conditions=market_conditions,
                        success_rate=0.0,  # Will be updated later
                        outcome=None  # Will be updated later
                    ))

                # Consolidation Pattern
                if (abs(market_conditions['price_momentum']) < self.pattern_thresholds.time_threshold and
                    market_conditions['volatility'] < self.pattern_thresholds.volatility_threshold):
                    patterns.append(MarketPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type='CONSOLIDATION',
                        symbol=symbol,
                        confidence_score=0.8,  # High confidence when both conditions met
                        eots_metrics={
                            'price_momentum': market_conditions['price_momentum'],
                            'volatility': market_conditions['volatility']
                        },
                        market_conditions=market_conditions,
                        success_rate=0.0,  # Will be updated later
                        outcome=None  # Will be updated later
                    ))

            # Store patterns
            for pattern in patterns:
                # Store in cache for intraday access
                self._enhanced_cache.set(
                    f"pattern_{pattern.pattern_type}_{pattern.pattern_id}",
                    pattern.dict(),
                    level=CacheLevel.INTRADAY
                )
                
                # Store in cache for long-term access
                self._store_pattern_longterm(symbol, pattern)
                
            return patterns
                
        except Exception as e:
            self.logger.error(f"Error detecting market patterns: {e}")
            return []

    async def analyze_market_data(
        self,
        data_bundle: ProcessedDataBundleV2_5,
        huihui_regime: Optional[HuiHuiMarketRegimeSchema] = None,
        huihui_flow: Optional[HuiHuiOptionsFlowSchema] = None,
        huihui_sentiment: Optional[HuiHuiSentimentSchema] = None
    ) -> UnifiedIntelligenceAnalysis:
        """
        Analyze market data using both traditional metrics and HuiHui expert insights.
        
        Args:
            data_bundle: Processed market data bundle
            huihui_regime: Optional HuiHui market regime assessment
            huihui_flow: Optional HuiHui options flow analysis
            huihui_sentiment: Optional HuiHui sentiment analysis
            
        Returns:
            UnifiedIntelligenceAnalysis containing consolidated analysis
        """
        try:
            # Extract underlying data
            underlying_data = data_bundle.underlying_data_enriched
            
            # Calculate regime metrics
            prices = np.array([underlying_data.price] * len(data_bundle.options_data_with_metrics))
            volumes = np.array([underlying_data.day_volume] * len(data_bundle.options_data_with_metrics))
            regime_metrics = calculate_regime_metrics(prices, volumes)
            
            # Determine market regime
            market_regime = self._determine_regime_state(regime_metrics, huihui_regime)
            
            # Extract options data for flow analysis
            strikes = np.array([contract.strike for contract in data_bundle.options_data_with_metrics])
            call_volumes = np.array([
                contract.volm if contract.opt_kind == "call" else 0 
                for contract in data_bundle.options_data_with_metrics
            ])
            put_volumes = np.array([
                contract.volm if contract.opt_kind == "put" else 0 
                for contract in data_bundle.options_data_with_metrics
            ])
            call_deltas = np.array([
                contract.delta_contract if contract.opt_kind == "call" else 0 
                for contract in data_bundle.options_data_with_metrics
            ])
            put_deltas = np.array([
                contract.delta_contract if contract.opt_kind == "put" else 0 
                for contract in data_bundle.options_data_with_metrics
            ])
            
            # Analyze options flow
            flow_metrics = vectorized_flow_analysis(
                strikes, call_volumes, put_volumes, call_deltas, put_deltas
            )
            
            # Identify key levels
            key_levels = {
                **vectorized_gamma_analysis(strikes, 
                    np.array([contract.gamma_contract for contract in data_bundle.options_data_with_metrics]),
                    volumes
                ),
                **vectorized_volume_profile(prices, volumes),
                **vectorized_technical_analysis(prices, volumes)
            }
            
            # Generate signals
            signals = self._generate_signals(key_levels, flow_metrics, regime_metrics, huihui_flow, huihui_sentiment)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(regime_metrics, huihui_regime, huihui_flow, huihui_sentiment)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(market_regime, signals, key_levels, huihui_regime, huihui_flow, huihui_sentiment)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(regime_metrics, flow_metrics, huihui_regime, huihui_sentiment)
            
            # Extract learning insights
            learning_insights = self._extract_learning_insights(market_regime, signals, huihui_regime, huihui_flow, huihui_sentiment)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(regime_metrics, flow_metrics, signals)
            
            # Format flow and sentiment analysis
            flow_analysis = self._format_flow_analysis(flow_metrics, huihui_flow)
            
            # Detect market patterns using real-time analysis
            market_patterns = self._detect_market_patterns(
                data_bundle, 
                regime_metrics, 
                flow_metrics, 
                huihui_regime, 
                huihui_flow, 
                huihui_sentiment
            )
            
            # Add pattern insights to learning insights
            pattern_insights = [
                f"Pattern {p.pattern_type} detected with {p.confidence_score:.2%} confidence"
                for p in market_patterns if p.confidence_score > 0.7
            ]
            learning_insights.extend(pattern_insights)
            
            # Get news intelligence insights
            news_intel = self.news_intelligence.get_real_time_intelligence_summary(data_bundle, data_bundle.underlying_data_enriched.symbol)
            
            # Enhance sentiment analysis with news intelligence
            if news_intel['intelligence_active']:
                sentiment_analysis = self._enhance_sentiment_analysis(huihui_sentiment, news_intel)
            else:
                sentiment_analysis = self._format_sentiment_analysis(huihui_sentiment)
            
            # Return unified analysis with enhanced sentiment
            return UnifiedIntelligenceAnalysis(
                symbol=underlying_data.symbol,
                timestamp=datetime.now(),
                confidence_score=confidence_score,
                market_regime_analysis=str(market_regime),
                options_flow_analysis=flow_analysis,
                sentiment_analysis=sentiment_analysis,
                strategic_recommendations=recommendations,
                risk_assessment=risk_assessment,
                learning_insights=learning_insights + [news_intel.get('diabolical_insight', '')],
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in market intelligence analysis: {str(e)}")
            raise

    def _determine_regime_state(
        self,
        regime_metrics: Dict[str, float],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema]
    ) -> MarketRegimeState:
        """
        Determine the current market regime state.
        
        Args:
            regime_metrics: Dictionary of regime metrics
            huihui_regime: Optional HuiHui regime assessment
            
        Returns:
            MarketRegimeState enum value
        """
        # If HuiHui regime is available, use it as primary input
        if huihui_regime and huihui_regime.current_regime_name:
            try:
                return MarketRegimeState(huihui_regime.current_regime_name.lower())
            except ValueError:
                logger.warning(f"Invalid HuiHui regime state: {huihui_regime.current_regime_name}")
        
        # Fallback to metric-based regime determination
        volatility = regime_metrics['volatility']
        volume_trend = regime_metrics['volume_trend']
        rsi = regime_metrics['regime_score']
        trend_strength = regime_metrics['trend_strength']
        
        # Determine regime based on metrics
        if volatility > 0.4:  # High volatility
            if trend_strength > 1.5:  # Strong trend
                if rsi > 70:
                    return MarketRegimeState.EUPHORIA
                elif rsi < 30:
                    return MarketRegimeState.PANIC
                else:
                    return MarketRegimeState.VOLATILITY_EXPANSION
            else:  # Weak trend
                return MarketRegimeState.CHOPPY
        else:  # Low volatility
            if trend_strength > 1.2:  # Strong trend
                if rsi > 60:
                    return MarketRegimeState.BULLISH_TREND
                elif rsi < 40:
                    return MarketRegimeState.BEARISH_TREND
                else:
                    return MarketRegimeState.TRENDING
            else:  # Weak trend
                if volume_trend > 0.2:
                    return MarketRegimeState.ACCUMULATION
                elif volume_trend < -0.2:
                    return MarketRegimeState.DISTRIBUTION
                else:
                    return MarketRegimeState.SIDEWAYS

    def _generate_signals(
        self,
        key_levels: Dict[str, float],
        flow_metrics: Dict[str, FloatArray],
        regime_metrics: Dict[str, float],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> List[SignalPayloadV2_5]:
        """
        Generate trading signals based on market analysis.
        
        Args:
            key_levels: Dictionary of identified key price levels
            flow_metrics: Dictionary of options flow metrics
            regime_metrics: Dictionary of regime metrics
            huihui_flow: Optional HuiHui flow analysis
            huihui_sentiment: Optional HuiHui sentiment analysis
            
        Returns:
            List of SignalPayload objects
        """
        signals = []
        
        # Generate signals from options flow
        vw_delta = np.mean(flow_metrics['net_delta_flow'])
        pc_ratio = np.mean(flow_metrics['call_put_ratio'])
        
        if huihui_flow:
            # Use HuiHui flow analysis for enhanced signal generation
            if huihui_flow.flow_type == "aggressive" and huihui_flow.directional_bias != "neutral":
                # Get current regime from metrics, ensuring it's a string
                current_regime = str(regime_metrics.get('current_regime', 'unknown'))
                
                signals.append(
                    SignalPayloadV2_5(
                        signal_id=str(uuid.uuid4()),
                        signal_name="HUIHUI_FLOW_SIGNAL",
                        symbol=huihui_flow.ticker,
                        timestamp=datetime.now(),
                        signal_type="flow",
                        direction=huihui_flow.directional_bias,
                        strength_score=float(huihui_flow.flow_intensity),
                        regime_at_signal_generation=current_regime,
                        supporting_metrics={
                            'institutional_probability': huihui_flow.institutional_probability,
                            'sophistication_score': huihui_flow.sophistication_score,
                            'market_impact_potential': huihui_flow.market_impact_potential
                        }
                    )
                )
        
        # Generate signals from technical analysis
        for level_name, level_price in key_levels.items():
            if 'gamma_level' in level_name and abs(vw_delta) > self.pattern_thresholds.significant_pos_thresh:
                signals.append(
                    SignalPayloadV2_5(
                        signal_id=str(uuid.uuid4()),
                        signal_name="GAMMA_LEVEL_SIGNAL",
                        symbol=huihui_flow.ticker if huihui_flow else "UNKNOWN",
                        timestamp=datetime.now(),
                        signal_type="technical",
                        direction="bullish" if vw_delta > 0 else "bearish",
                        strength_score=float(min(abs(vw_delta) / self.pattern_thresholds.significant_pos_thresh, 1.0)),
                        strike_impacted=level_price,
                        supporting_metrics={
                            'vw_delta': float(vw_delta),
                            'pc_ratio': float(pc_ratio),
                            'regime_metrics': regime_metrics
                        }
                    )
                )
        
        # Add sentiment-based signals if available
        if huihui_sentiment:
            if abs(huihui_sentiment.overall_sentiment_score) > self.pattern_thresholds.moderate_confidence_thresh:
                signals.append(
                    SignalPayloadV2_5(
                        signal_id=str(uuid.uuid4()),
                        signal_name="SENTIMENT_SIGNAL",
                        symbol=huihui_sentiment.ticker,
                        timestamp=datetime.now(),
                        signal_type="sentiment",
                        direction=huihui_sentiment.sentiment_direction,
                        strength_score=float(huihui_sentiment.sentiment_strength),
                        supporting_metrics={
                            'sentiment_score': huihui_sentiment.overall_sentiment_score,
                            'fear_greed_index': huihui_sentiment.fear_greed_index,
                            'crowd_psychology': huihui_sentiment.crowd_psychology_state
                        }
                    )
                )
        
        return signals

    def _calculate_confidence_score(
        self,
        regime_metrics: Dict[str, float],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> float:
        """
        Calculate confidence score based on all available metrics and HuiHui insights.
        
        Args:
            regime_metrics: Dictionary of regime metrics
            huihui_regime: Optional HuiHui regime assessment
            huihui_flow: Optional HuiHui flow analysis
            huihui_sentiment: Optional HuiHui sentiment analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        base_score = 0.5  # Start with neutral confidence
        
        # Add confidence from HuiHui experts if available
        if huihui_regime:
            base_score += huihui_regime.confidence_level * 0.2
        
        if huihui_flow:
            base_score += huihui_flow.confidence_level * 0.2
            
        if huihui_sentiment:
            base_score += huihui_sentiment.confidence_level * 0.2
            
        # Add confidence from traditional metrics
        volatility = regime_metrics.get('volatility', 0.0)
        if volatility < self.pattern_thresholds.volatility_threshold:
            base_score += 0.1
            
        # Ensure score is between 0 and 1
        # Adjust based on metric consistency
        trend_strength = regime_metrics.get('trend_strength', 0)
        volatility = regime_metrics.get('volatility', 0)
        
        # Higher confidence in strong trends and normal volatility range
        # Use volatility_threshold/2 as a proxy for normal volatility
        normal_volatility = self.pattern_thresholds.volatility_threshold / 2
        metric_score = min(1.0, (trend_strength / self.pattern_thresholds.dwfd_strong_thresh) * 
                          (1 - abs(volatility - normal_volatility) / 
                           self.pattern_thresholds.volatility_threshold))
        
        # Combine scores with more weight to HuiHui experts
        final_score = 0.7 * base_score + 0.3 * metric_score
        
        return float(min(1.0, max(0.0, final_score)))

    def _generate_recommendations(
        self,
        market_regime: MarketRegimeState,
        signals: List[SignalPayloadV2_5],
        key_levels: Dict[str, float],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> List[str]:
        """Generate strategic trading recommendations."""
        recommendations = []
        
        # Add regime-based recommendations
        if market_regime in [MarketRegimeState.VOLATILITY_EXPANSION, MarketRegimeState.PANIC]:
            recommendations.append("Consider reducing position sizes due to high volatility")
        elif market_regime == MarketRegimeState.VOLATILITY_CONTRACTION:
            recommendations.append("Watch for breakout opportunities as volatility compresses")
        
        # Add flow-based recommendations
        if huihui_flow and huihui_flow.flow_type == "aggressive":
            recommendations.append(
                f"Strong institutional flow detected: {huihui_flow.directional_bias.upper()} bias "
                f"with {huihui_flow.sophistication_score:.2f} sophistication score"
            )
        
        # Add sentiment-based recommendations
        if huihui_sentiment:
            if abs(huihui_sentiment.overall_sentiment_score) > 0.7:
                recommendations.append(
                    f"Strong {huihui_sentiment.sentiment_direction} sentiment detected "
                    f"with {huihui_sentiment.sentiment_strength:.2f} strength"
                )
        
        # Add signal-based recommendations
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)
        
        for signal_type, group in signal_groups.items():
            if len(group) > 1:
                directions = [s.direction for s in group]
                if all(d == directions[0] for d in directions):
                    recommendations.append(
                        f"Multiple {signal_type} signals showing {directions[0].upper()} bias"
                    )
        
        return recommendations

    def _generate_risk_assessment(
        self,
        regime_metrics: Dict[str, float],
        flow_metrics: Dict[str, FloatArray],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> str:
        """Generate risk assessment based on market conditions."""
        risk_factors = []
        
        # Assess volatility risk
        if regime_metrics['volatility'] > self.pattern_thresholds.volatility_threshold:
            risk_factors.append("HIGH_VOLATILITY")
        
        # Assess sentiment risk
        if huihui_sentiment:
            if huihui_sentiment.tail_risk_probability > 0.3:
                risk_factors.append("TAIL_RISK_ELEVATED")
            if huihui_sentiment.fear_greed_index > 80 or huihui_sentiment.fear_greed_index < 20:
                risk_factors.append("EXTREME_SENTIMENT")
        
        # Assess regime risk
        if huihui_regime:
            if huihui_regime.regime_stability_score < 0.3:
                risk_factors.append("UNSTABLE_REGIME")
        
        # Generate risk assessment text
        if not risk_factors:
            return "NORMAL: No significant risk factors identified"
        
        risk_level = "HIGH" if len(risk_factors) > 2 else "MODERATE"
        return f"{risk_level}: {', '.join(risk_factors)}"

    def _extract_learning_insights(
        self,
        market_regime: MarketRegimeState,
        signals: List[SignalPayloadV2_5],
        huihui_regime: Optional[HuiHuiMarketRegimeSchema],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema],
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> List[str]:
        """Extract learning insights from the analysis."""
        insights = []
        
        # Regime learning insights
        if huihui_regime and market_regime.value != huihui_regime.current_regime_name.lower():
            insights.append(
                f"Regime classification divergence: Internal ({market_regime.value}) "
                f"vs HuiHui ({huihui_regime.current_regime_name})"
            )
        
        # Flow learning insights
        if huihui_flow:
            if huihui_flow.institutional_probability > 0.8:
                insights.append(
                    f"High institutional activity detected: {huihui_flow.flow_type} "
                    f"flow with {huihui_flow.sophistication_score:.2f} sophistication"
                )
        
        # Sentiment learning insights
        if huihui_sentiment:
            if abs(huihui_sentiment.sentiment_momentum) > 0.7:
                insights.append(
                    f"Strong sentiment momentum: {huihui_sentiment.sentiment_direction} "
                    f"with {huihui_sentiment.sentiment_persistence:.2f} persistence"
                )
        
        # Signal learning insights
        signal_types = set(s.signal_type for s in signals)
        if len(signal_types) > 2:
            insights.append(
                f"Multiple signal types active: {', '.join(signal_types)}"
            )
        
        return insights

    def _calculate_performance_metrics(
        self,
        regime_metrics: Dict[str, float],
        flow_metrics: Dict[str, FloatArray],
        signals: List[SignalPayloadV2_5]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis."""
        return {
            'signal_count': len(signals),
            'average_signal_strength': np.mean([s.strength_score for s in signals]) if signals else 0.0,
            'regime_metrics': regime_metrics,
            'flow_metrics_summary': {
                'avg_vw_delta': float(np.mean(flow_metrics['net_delta_flow'])),
                'avg_pc_ratio': float(np.mean(flow_metrics['call_put_ratio'])),
                'max_vol_concentration': float(np.max(flow_metrics['volume_imbalance']))
            }
        }

    def _format_flow_analysis(
        self,
        flow_metrics: Dict[str, FloatArray],
        huihui_flow: Optional[HuiHuiOptionsFlowSchema]
    ) -> str:
        """Format options flow analysis into a readable string."""
        try:
            # Start with HuiHui flow insights if available
            if huihui_flow:
                flow_str = (
                    f"HuiHui Flow Analysis: {huihui_flow.flow_type.upper()} flow detected\n"
                    f"Direction: {huihui_flow.directional_bias}\n"
                    f"Sophistication: {huihui_flow.sophistication_score:.2f}\n"
                    f"Institutional Probability: {huihui_flow.institutional_probability:.2f}\n"
                )
                
                if huihui_flow.supporting_indicators:
                    flow_str += f"Key Insights: {', '.join(huihui_flow.supporting_indicators)}\n"
            else:
                flow_str = "Traditional Flow Analysis:\n"
            
            # Add traditional flow metrics
            if flow_metrics:
                avg_vw_delta = float(np.mean(flow_metrics['net_delta_flow']))
                avg_pc_ratio = float(np.mean(flow_metrics['call_put_ratio']))
                max_vol_conc = float(np.max(flow_metrics['volume_imbalance']))
                
                flow_str += (
                    f"Volume-Weighted Delta: {avg_vw_delta:.2f}\n"
                    f"Put-Call Ratio: {avg_pc_ratio:.2f}\n"
                    f"Max Volume Concentration: {max_vol_conc:.2f}"
                )
            
            return flow_str
            
        except Exception as e:
            logger.error(f"Error formatting flow analysis: {str(e)}")
            return "Error formatting flow analysis"

    def _format_sentiment_analysis(
        self,
        huihui_sentiment: Optional[HuiHuiSentimentSchema]
    ) -> str:
        """Format sentiment analysis into a readable string."""
        try:
            if not huihui_sentiment:
                return "No sentiment analysis available"
                
            sentiment_str = (
                f"HuiHui Sentiment Analysis:\n"
                f"Overall Sentiment: {huihui_sentiment.sentiment_direction.upper()}\n"
                f"Sentiment Strength: {huihui_sentiment.sentiment_strength:.2f}\n"
                f"Sentiment Momentum: {huihui_sentiment.sentiment_momentum:.2f}\n"
                f"Fear-Greed Index: {huihui_sentiment.fear_greed_index:.1f}\n"
            )
            
            # Add persistence and confidence metrics
            sentiment_str += (
                f"Sentiment Persistence: {huihui_sentiment.sentiment_persistence:.2f}\n"
                f"Confidence Level: {huihui_sentiment.confidence_level:.2f}\n"
            )
            
            # Add risk metrics if available
            sentiment_str += (
                f"Tail Risk Probability: {huihui_sentiment.tail_risk_probability:.2f}\n"
                f"Risk Level: {huihui_sentiment.risk_level}\n"
            )
            
            # Add key insights if available
            if huihui_sentiment.key_sentiment_drivers:
                sentiment_str += f"Key Insights: {', '.join(huihui_sentiment.key_sentiment_drivers)}"
            
            return sentiment_str
            
        except Exception as e:
            logger.error(f"Error formatting sentiment analysis: {str(e)}")
            return "Error formatting sentiment analysis"

    def _enhance_sentiment_analysis(
        self,
        huihui_sentiment: Optional[HuiHuiSentimentSchema],
        news_intel: Dict[str, Any]
    ) -> str:
        """Enhance sentiment analysis with news intelligence insights."""
        try:
            # Get base sentiment analysis
            base_sentiment = self._format_sentiment_analysis(huihui_sentiment) or "No sentiment analysis available"
            
            # Add news intelligence insights
            news_insights = (
                f"\nNews Intelligence Analysis:\n"
                f"Intelligence Score: {news_intel['intelligence_score']:.2f}\n"
                f"Sentiment Regime: {news_intel['sentiment_regime']}\n"
                f"Primary Narrative: {news_intel['primary_narrative']}\n"
                f"Confidence Level: {news_intel['confidence_level']}\n"
                f"Diabolical Insight: {news_intel['diabolical_insight']}"
            )
            
            return f"{base_sentiment}\n{news_insights}"
            
        except Exception as e:
            logger.error(f"Error enhancing sentiment analysis: {str(e)}")
            return self._format_sentiment_analysis(huihui_sentiment) or "Error in sentiment analysis"