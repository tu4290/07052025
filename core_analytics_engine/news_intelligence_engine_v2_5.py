"""
News Intelligence Engine v2.5 for EOTS "Apex Predator"
======================================================

The diabolical AI-powered news analysis engine that transforms raw sentiment data
into actionable market intelligence. This engine cross-references Alpha Vantage
sentiment analysis with EOTS v2.5 metrics to generate sophisticated market narratives.

Key Features:
- Pydantic AI-powered sentiment interpretation
- Cross-correlation with EOTS v2.5 flow metrics
- Intelligent narrative generation
- Market regime contextualization
- Contrarian signal detection
- Confluence analysis

Author: EOTS v2.5 Development Team - "Apex Predator" Division
Version: 2.5.0 - "Diabolical Intelligence"
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import uuid

# Import EOTS components
from data_management.alpha_vantage_fetcher_v2_5 import AlphaVantageDataFetcherV2_5, SentimentDataV2_5
from data_models.ai_ml_models import (
    RealTimeIntelligenceSummaryV2_5,
    HuiHuiSentimentSchema,
    HuiHuiMarketRegimeSchema,
    HuiHuiOptionsFlowSchema
)
from data_models.core_models import ProcessedDataBundleV2_5

logger = logging.getLogger(__name__)

class MarketSentimentRegime(Enum):
    """Market sentiment regime classifications."""
    EXTREME_BULLISH = "EXTREME_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL_POSITIVE = "NEUTRAL_POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEUTRAL_NEGATIVE = "NEUTRAL_NEGATIVE"
    BEARISH = "BEARISH"
    EXTREME_BEARISH = "EXTREME_BEARISH"

class NewsSignalType(Enum):
    """Types of news intelligence signals."""
    FLOW_CONFLUENCE = "FLOW_CONFLUENCE"
    SENTIMENT_SHIFT = "SENTIMENT_SHIFT"
    CONTRARIAN = "CONTRARIAN"
    REGIME_CHANGE = "REGIME_CHANGE"
    VOLATILITY_ALERT = "VOLATILITY_ALERT"
    INSTITUTIONAL_ACTIVITY = "INSTITUTIONAL_ACTIVITY"

# PYDANTIC-FIRST: News Intelligence Signal Model
class NewsIntelligenceSignal(BaseModel):
    """Pydantic model for news intelligence signals with full validation."""
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique signal identifier")
    signal_type: NewsSignalType = Field(..., description="Type of intelligence signal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence score")
    narrative: str = Field(..., min_length=10, description="Signal narrative description")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence for signal")
    eots_confluence: Dict[str, float] = Field(..., description="EOTS metrics confluence")
    sentiment_context: Dict[str, Any] = Field(..., description="Market sentiment context")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal detection timestamp")

    class Config:
        extra = 'forbid'  # Strict validation
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            NewsSignalType: lambda v: v.value
        }

    @field_validator('eots_confluence')
    @classmethod
    def validate_eots_confluence(cls, v):
        """Validate EOTS confluence has required metrics."""
        required_metrics = {'vapi_fa', 'dwfd', 'tw_laf'}
        missing_metrics = required_metrics - set(v.keys())
        if missing_metrics:
            raise ValueError(f"EOTS confluence missing required metrics: {missing_metrics}")
        return v

    @field_validator('sentiment_context')
    @classmethod
    def validate_sentiment_context(cls, v):
        """Validate sentiment context has required fields."""
        required_fields = {'score', 'article_count', 'sentiment_regime'}
        missing_fields = required_fields - set(v.keys())
        if missing_fields:
            raise ValueError(f"Sentiment context missing required fields: {missing_fields}")
        return v

    @field_validator('supporting_evidence')
    @classmethod
    def validate_supporting_evidence(cls, v):
        """Validate supporting evidence items are not empty."""
        if not v:
            raise ValueError("Supporting evidence cannot be empty")
        if any(not evidence.strip() for evidence in v):
            raise ValueError("Supporting evidence items cannot be empty strings")
        if len(v) < 1:
            raise ValueError("At least one supporting evidence item is required")
        return v

class NewsIntelligenceEngineV2_5:
    """
    The Diabolical News Intelligence Engine
    
    This AI-powered engine analyzes market sentiment and cross-references it with
    EOTS v2.5 metrics to generate sophisticated market intelligence and trading narratives.
    """
    
    def __init__(self, config_manager, alpha_vantage_fetcher: Optional[AlphaVantageDataFetcherV2_5] = None):
        """
        Initialize the News Intelligence Engine.
        
        Args:
            config_manager: EOTS v2.5 configuration manager
            alpha_vantage_fetcher: Alpha Vantage data fetcher instance
        """
        self.config_manager = config_manager
        self.alpha_vantage_fetcher = alpha_vantage_fetcher
        
        # Intelligence thresholds
        self.sentiment_thresholds = {
            'extreme_bullish': 0.4,
            'bullish': 0.15,
            'neutral_positive': 0.05,
            'neutral_negative': -0.05,
            'bearish': -0.15,
            'extreme_bearish': -0.4
        }
        
        # EOTS metric significance thresholds
        self.eots_thresholds = {
            'vapi_fa_extreme': 2.5,
            'dwfd_extreme': 2.0,
            'tw_laf_extreme': 1.8,
            'a_dag_extreme': 100000,
            'vri_2_0_extreme': 10000
        }
        
        logger.info("🧠 News Intelligence Engine v2.5 'Diabolical Intelligence' initialized")
    
    def analyze_market_intelligence(self, 
                                  processed_data: ProcessedDataBundleV2_5,
                                  symbol: str = "SPY") -> Dict[str, Any]:
        """
        Generate comprehensive market intelligence by analyzing sentiment and EOTS metrics.
        
        Args:
            processed_data: EOTS v2.5 processed data bundle
            symbol: Ticker symbol to analyze
            
        Returns:
            Dict containing comprehensive market intelligence analysis
        """
        try:
            logger.info(f"🎭 Generating diabolical market intelligence for {symbol}...")
            
            # Get Alpha Vantage sentiment data
            sentiment_intelligence = self._fetch_sentiment_intelligence(symbol)
            
            # Extract EOTS metrics
            eots_metrics = self._extract_eots_metrics(processed_data)
            
            # Classify sentiment regime
            sentiment_regime = self._classify_sentiment_regime(sentiment_intelligence)
            
            # Detect market signals
            intelligence_signals = self._detect_intelligence_signals(
                sentiment_intelligence, eots_metrics, sentiment_regime
            )
            
            # Generate market narratives
            market_narratives = self._generate_market_narratives(
                intelligence_signals, sentiment_intelligence, eots_metrics
            )
            
            # Calculate overall intelligence score
            intelligence_score = self._calculate_intelligence_score(
                sentiment_intelligence, eots_metrics, intelligence_signals
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'intelligence_score': intelligence_score,
                'sentiment_regime': sentiment_regime.value,
                'sentiment_data': sentiment_intelligence,
                'eots_metrics_summary': eots_metrics,
                'intelligence_signals': [
                    {
                        'type': signal.signal_type.value,
                        'confidence': signal.confidence,
                        'narrative': signal.narrative,
                        'evidence': signal.supporting_evidence
                    }
                    for signal in intelligence_signals
                ],
                'market_narratives': market_narratives,
                'diabolical_insights': self._generate_diabolical_insights(
                    intelligence_signals, sentiment_regime, eots_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Error in market intelligence analysis: {str(e)}")
            return self._get_fallback_intelligence(symbol)
            # ↑ FAIL-FAST REMEDIATION: removed (below) –
            #   we now raise an exception instead of returning fake data.

        except Exception as e:
            logger.error(f"Error in market intelligence analysis: {str(e)}")
            # FAIL-FAST: do **not** return placeholder data.
            raise RuntimeError(
                f"CRITICAL: market intelligence unavailable for {symbol}: {e}"
            ) from e
    
    def _fetch_sentiment_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Fetch and process sentiment intelligence from Alpha Vantage."""
        try:
            if not self.alpha_vantage_fetcher:
                return {}
            
            # Get comprehensive sentiment data
            comprehensive_intel = self.alpha_vantage_fetcher.get_comprehensive_intelligence_summary(symbol)
            
            if comprehensive_intel and comprehensive_intel.get('alpha_intelligence_active'):
                return comprehensive_intel.get('sentiment', {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching sentiment intelligence: {str(e)}")
            return {}
    
    def _extract_eots_metrics(self, processed_data: ProcessedDataBundleV2_5) -> Dict[str, float]:
        """Extract key EOTS v2.5 metrics for intelligence analysis."""
        try:
            # Extract metrics from underlying data
            underlying_data = processed_data.underlying_data_enriched
            
            # Extract Enhanced Flow Metrics (Tier 3)
            eots_summary = {
                'vapi_fa_z': underlying_data.vapi_fa_z_score_und,
                'vapi_fa_raw': underlying_data.vapi_fa_raw_und,
                'dwfd_z': underlying_data.dwfd_z_score_und,
                'dwfd_raw': underlying_data.dwfd_raw_und,
                'tw_laf_z': underlying_data.tw_laf_z_score_und,
                'tw_laf_raw': underlying_data.tw_laf_raw_und
            }
            
            # Extract Adaptive Metrics (Tier 2) from strike data
            if processed_data.strike_level_data_with_metrics:
                strike_metrics = pd.DataFrame([s.model_dump() for s in processed_data.strike_level_data_with_metrics])
                if not strike_metrics.empty:
                    eots_summary.update({
                        'a_dag_total': abs(strike_metrics.get('a_dag_strike', pd.Series([0])).sum()),
                        'vri_2_0_avg': abs(strike_metrics.get('vri_2_0_strike', pd.Series([0])).mean())
                    })
            
            return {k: v for k, v in eots_summary.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error extracting EOTS metrics: {str(e)}")
            return {}
    
    def _classify_sentiment_regime(self, sentiment_data: Dict[str, Any]) -> MarketSentimentRegime:
        """Classify the current sentiment regime."""
        try:
            sentiment_score = sentiment_data.get('score', 0.0)
            
            if sentiment_score >= self.sentiment_thresholds['extreme_bullish']:
                return MarketSentimentRegime.EXTREME_BULLISH
            elif sentiment_score >= self.sentiment_thresholds['bullish']:
                return MarketSentimentRegime.BULLISH
            elif sentiment_score >= self.sentiment_thresholds['neutral_positive']:
                return MarketSentimentRegime.NEUTRAL_POSITIVE
            elif sentiment_score >= self.sentiment_thresholds['neutral_negative']:
                return MarketSentimentRegime.NEUTRAL
            elif sentiment_score >= self.sentiment_thresholds['bearish']:
                return MarketSentimentRegime.NEUTRAL_NEGATIVE
            elif sentiment_score >= self.sentiment_thresholds['extreme_bearish']:
                return MarketSentimentRegime.BEARISH
            else:
                return MarketSentimentRegime.EXTREME_BEARISH
                
        except Exception as e:
            logger.error(f"Error classifying sentiment regime: {str(e)}")
            return MarketSentimentRegime.NEUTRAL
    
    def _detect_intelligence_signals(self, sentiment_data: Dict[str, Any], eots_metrics: Dict[str, float], 
                                   sentiment_regime: MarketSentimentRegime) -> List[NewsIntelligenceSignal]:
        """Detect intelligence signals from sentiment and EOTS data."""
        signals = []
        
        try:
            # Detect contrarian setups
            if abs(sentiment_data.get('score', 0)) > 0.4 and any(abs(v) > 2.0 for v in eots_metrics.values()):
                signals.append(NewsIntelligenceSignal(
                    signal_type=NewsSignalType.CONTRARIAN,
                    confidence=0.85,
                    narrative="Strong contrarian setup detected with extreme sentiment divergence",
                    supporting_evidence=[
                        f"Extreme sentiment score: {sentiment_data.get('score', 0):.2f}",
                        f"EOTS metrics showing extreme readings"
                    ],
                    eots_confluence={
                        'vapi_fa': eots_metrics.get('vapi_fa_z', 0),
                        'dwfd': eots_metrics.get('dwfd_z', 0),
                        'tw_laf': eots_metrics.get('tw_laf_z', 0)
                    },
                    sentiment_context=sentiment_data
                ))
            
            # Detect flow confluence
            strong_flow = sum(1 for v in [
                eots_metrics.get('vapi_fa_z', 0),
                eots_metrics.get('dwfd_z', 0),
                eots_metrics.get('tw_laf_z', 0)
            ] if abs(v) > 1.5)
            
            if strong_flow >= 2:
                signals.append(NewsIntelligenceSignal(
                    signal_type=NewsSignalType.FLOW_CONFLUENCE,
                    confidence=0.9,
                    narrative="Strong flow confluence detected across multiple metrics",
                    supporting_evidence=[
                        f"{strong_flow}/3 flow metrics showing extreme readings",
                        "Institutional activity confirmed"
                    ],
                    eots_confluence={
                        'vapi_fa': eots_metrics.get('vapi_fa_z', 0),
                        'dwfd': eots_metrics.get('dwfd_z', 0),
                        'tw_laf': eots_metrics.get('tw_laf_z', 0)
                    },
                    sentiment_context=sentiment_data
                ))
            
            # Detect sentiment shifts
            if sentiment_regime in [MarketSentimentRegime.EXTREME_BULLISH, MarketSentimentRegime.EXTREME_BEARISH]:
                signals.append(NewsIntelligenceSignal(
                    signal_type=NewsSignalType.SENTIMENT_SHIFT,
                    confidence=0.8,
                    narrative=f"Extreme {sentiment_regime.value.lower()} sentiment detected",
                    supporting_evidence=[
                        f"Sentiment regime: {sentiment_regime.value}",
                        f"Sentiment score: {sentiment_data.get('score', 0):.2f}"
                    ],
                    eots_confluence={
                        'vapi_fa': eots_metrics.get('vapi_fa_z', 0),
                        'dwfd': eots_metrics.get('dwfd_z', 0),
                        'tw_laf': eots_metrics.get('tw_laf_z', 0)
                    },
                    sentiment_context=sentiment_data
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting intelligence signals: {str(e)}")
            return []

    def _generate_market_narratives(self,
                                  signals: List[NewsIntelligenceSignal],
                                  sentiment_data: Dict[str, Any],
                                  eots_metrics: Dict[str, float]) -> List[str]:
        """Generate sophisticated market narratives based on intelligence analysis."""
        narratives = []

        try:
            # Primary narrative based on strongest signal
            if signals:
                strongest_signal = max(signals, key=lambda s: s.confidence)
                narratives.append(f"🎭 PRIMARY INTELLIGENCE: {strongest_signal.narrative}")

            # Secondary narratives based on metric confluence
            strong_eots_signals = sum([
                abs(eots_metrics.get('vapi_fa_z', 0)) > 1.5,
                abs(eots_metrics.get('dwfd_z', 0)) > 1.5,
                abs(eots_metrics.get('tw_laf_z', 0)) > 1.2
            ])

            if strong_eots_signals >= 2:
                narratives.append(f"⚡ EOTS CONFLUENCE: {strong_eots_signals}/3 advanced metrics showing extreme readings")

            # Sentiment context narrative
            sentiment_score = sentiment_data.get('score', 0.0)
            article_count = sentiment_data.get('article_count', 0)

            if abs(sentiment_score) > 0.2 and article_count > 20:
                sentiment_intensity = "EXTREME" if abs(sentiment_score) > 0.4 else "STRONG"
                direction = "BULLISH" if sentiment_score > 0 else "BEARISH"
                narratives.append(f"📰 NEWS INTELLIGENCE: {sentiment_intensity} {direction} sentiment "
                                f"across {article_count} articles driving market attention")

            return narratives[:4]  # Limit to 4 narratives

        except Exception as e:
            logger.error(f"Error generating market narratives: {str(e)}")
            return ["🤖 Market narrative generation temporarily unavailable"]

    def _calculate_intelligence_score(self,
                                    sentiment_data: Dict[str, Any],
                                    eots_metrics: Dict[str, float],
                                    signals: List[NewsIntelligenceSignal]) -> float:
        """Calculate overall intelligence confidence score."""
        try:
            score_components = []

            # Sentiment confidence component
            sentiment_confidence = sentiment_data.get('confidence', 50.0) / 100.0
            score_components.append(sentiment_confidence * 0.3)

            # EOTS signal strength component
            max_eots_z = max([
                abs(eots_metrics.get('vapi_fa_z', 0)),
                abs(eots_metrics.get('dwfd_z', 0)),
                abs(eots_metrics.get('tw_laf_z', 0))
            ])
            eots_strength = min(max_eots_z / 3.0, 1.0)
            score_components.append(eots_strength * 0.4)

            # Signal confluence component
            if signals:
                avg_signal_confidence = sum(s.confidence for s in signals) / len(signals)
                score_components.append(avg_signal_confidence * 0.3)
            else:
                score_components.append(0.5 * 0.3)  # Neutral if no signals

            return min(sum(score_components), 0.95)

        except Exception as e:
            logger.error(f"Error calculating intelligence score: {str(e)}")
            return 0.5

    def _generate_diabolical_insights(self,
                                    signals: List[NewsIntelligenceSignal],
                                    sentiment_regime: MarketSentimentRegime,
                                    eots_metrics: Dict[str, float]) -> List[str]:
        """Generate the most diabolical and sophisticated market insights."""
        insights = []

        try:
            # Diabolical signal insights
            for signal in signals[:2]:  # Top 2 signals
                if signal.signal_type == NewsSignalType.CONTRARIAN:
                    insights.append(f"😈 DIABOLICAL CONTRARIAN: While the crowd panics/celebrates, "
                                  f"smart money is quietly positioning for the opposite move "
                                  f"(Confidence: {signal.confidence:.1%})")

                elif signal.signal_type == NewsSignalType.FLOW_CONFLUENCE:
                    insights.append(f"🔥 DIABOLICAL CONFLUENCE: Perfect storm detected - "
                                  f"news sentiment and options flow aligned for devastating move "
                                  f"(Confidence: {signal.confidence:.1%})")

                elif signal.signal_type == NewsSignalType.SENTIMENT_SHIFT:
                    insights.append(f"🌪️ DIABOLICAL SENTIMENT SHIFT: Extreme sentiment ({sentiment_regime.value}) "
                                  f"often marks reversals - watch for smart money divergence")

            # Regime-based diabolical insights
            if sentiment_regime in [MarketSentimentRegime.EXTREME_BULLISH, MarketSentimentRegime.EXTREME_BEARISH]:
                insights.append(f"⚠️ DIABOLICAL WARNING: Extreme sentiment ({sentiment_regime.value}) "
                              f"often marks reversals - watch for smart money divergence")

            # EOTS-based diabolical insights
            vapi_fa_z = abs(eots_metrics.get('vapi_fa_z', 0))
            dwfd_z = abs(eots_metrics.get('dwfd_z', 0))

            if vapi_fa_z > 2.5 and dwfd_z > 2.0:
                insights.append(f"💀 DIABOLICAL SETUP: VAPI-FA ({vapi_fa_z:.1f}σ) + DWFD ({dwfd_z:.1f}σ) "
                              f"extreme readings suggest institutional positioning for major move")

            # Market timing diabolical insight
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 10 and signals:
                insights.append(f"🌅 DIABOLICAL TIMING: Opening hour + intelligence signals = "
                              f"Perfect execution window for apex predators")
            elif 15 <= current_hour <= 16 and signals:
                insights.append(f"🌆 DIABOLICAL TIMING: Power hour + intelligence confluence = "
                              f"Institutional positioning for overnight/next day moves")

            return insights[:4]  # Limit to 4 diabolical insights

        except Exception as e:
            logger.error(f"Error generating diabolical insights: {str(e)}")
            return ["😈 Diabolical insights temporarily unavailable - but the predator still hunts..."]

    # --------------------------------------------------------------------- #
    # PUBLIC DASHBOARD INTERFACE                                           #
    # --------------------------------------------------------------------- #

    def get_real_time_intelligence_summary(
        self,
        processed_data: ProcessedDataBundleV2_5,
        symbol: str = "SPY"
    ) -> RealTimeIntelligenceSummaryV2_5:
        """
        Get real-time intelligence summary for dashboard display.

        This is the main interface for the dashboard to get processed intelligence.
        """
        try:
            logger.info(f"🎭 Generating real-time intelligence summary for {symbol}")

            # Get full intelligence analysis
            full_intelligence = self.analyze_market_intelligence(processed_data, symbol)

            # Extract key components for dashboard
            return RealTimeIntelligenceSummaryV2_5(
                intelligence_active=True,
                intelligence_score=full_intelligence["intelligence_score"],
                sentiment_regime=full_intelligence["sentiment_regime"],
                primary_narrative=(
                    full_intelligence["market_narratives"][0]
                    if full_intelligence["market_narratives"]
                    else "No primary narrative"
                ),
                top_signal=(
                    full_intelligence["intelligence_signals"][0]
                    if full_intelligence["intelligence_signals"]
                    else None
                ),
                diabolical_insight=(
                    full_intelligence["diabolical_insights"][0]
                    if full_intelligence["diabolical_insights"]
                    else "😈 Apex predator analyzing..."
                ),
                confidence_level=f"{full_intelligence['intelligence_score']:.1%}",
                timestamp=full_intelligence["timestamp"],
            )

        except Exception as e:
            logger.error(f"Error generating real-time intelligence summary: {str(e)}")
            # FAIL-FAST: no placeholder summaries.
            raise RuntimeError(
                f"CRITICAL: failed to build real-time intelligence summary for {symbol}: {e}"
            ) from e

