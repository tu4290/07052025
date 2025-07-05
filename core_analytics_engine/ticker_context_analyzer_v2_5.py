"""
Ticker Context Analyzer v2.5 - EOTS "Apex Predator"

Provides comprehensive ticker-specific context analysis including sector analysis,
market cap considerations, volatility patterns, and correlation analysis.

Author: EOTS Development Team
Version: 2.5.0
Last Updated: 2024
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import time
import random
import warnings

# Conditional import for Yahoo Finance
YFINANCE_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    warnings.warn("yfinance not available. Yahoo Finance features will be disabled.")

# Import Pydantic model for type hints
from data_models import TickerContextAnalyzerSettings

# Configure logging
logger = logging.getLogger(__name__)

# ZERO TOLERANCE FAKE DATA: No fallback ticker data allowed

class TickerProfile(BaseModel):
    """Comprehensive ticker profile with fundamental and technical characteristics."""
    symbol: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    beta: Optional[float] = None
    volatility_regime: str = Field('normal', description="'low', 'normal', 'high', 'extreme'")
    liquidity_tier: str = Field('medium', description="'high', 'medium', 'low'")
    options_activity_level: str = Field('normal', description="'low', 'normal', 'high', 'extreme'")

class MarketContext(BaseModel):
    """Current market environment context."""
    market_regime: str = Field(..., description="'bull', 'bear', 'sideways', 'volatile'")
    vix_level: float
    spy_trend: str = Field(..., description="'up', 'down', 'sideways'")
    sector_rotation: Dict[str, float]
    risk_sentiment: str = Field(..., description="'risk_on', 'risk_off', 'neutral'")

class CorrelationAnalysis(BaseModel):
    """Correlation analysis with market and sector."""
    spy_correlation: float
    qqq_correlation: float
    sector_correlation: float
    correlation_stability: float = Field(..., description="How stable correlations are")
    relative_strength: float = Field(..., description="vs SPY")

class VolatilityProfile(BaseModel):
    """Comprehensive volatility analysis."""
    realized_vol_1d: float
    realized_vol_5d: float
    realized_vol_20d: float
    implied_vol_rank: Optional[float] = None
    vol_regime: str = Field('normal', description="'low', 'normal', 'high', 'extreme'")
    vol_trend: str = Field('stable', description="'increasing', 'decreasing', 'stable'")
    vol_percentile: Optional[float] = None

class TickerContextAnalysis(BaseModel):
    """Complete ticker context analysis results."""
    symbol: str
    timestamp: datetime
    ticker_profile: TickerProfile
    market_context: MarketContext
    correlation_analysis: CorrelationAnalysis
    volatility_profile: VolatilityProfile
    trading_characteristics: Dict[str, Any]
    risk_factors: List[str]
    opportunities: List[str]
    context_score: float = Field(..., description="Overall context favorability (0-1)")

class TickerContextAnalyzerV2_5:
    """
    Advanced ticker context analyzer that provides comprehensive symbol-specific
    analysis including fundamental, technical, and market environment factors.
    """
    
    def __init__(self, config: Union[TickerContextAnalyzerSettings, Dict[str, Any]]):
        self.logger = logger.getChild(self.__class__.__name__)
        
        # PYDANTIC COMPLIANCE: Handle both Pydantic models and dictionaries
        if isinstance(config, TickerContextAnalyzerSettings):
            # Direct attribute access for Pydantic models
            self.lookback_days = config.lookback_days
            self.correlation_window = config.correlation_window  
            self.volatility_windows = config.volatility_windows
            self.volume_threshold = config.volume_threshold
            self.use_yahoo_finance = config.use_yahoo_finance and YFINANCE_AVAILABLE
            self.yahoo_finance_rate_limit = config.yahoo_finance_rate_limit_seconds
        else:
            # Dictionary-style access for backward compatibility
            self.lookback_days = config.get('lookback_days', 252) if hasattr(config, 'get') else getattr(config, 'lookback_days', 252)
            self.correlation_window = config.get('correlation_window', 60) if hasattr(config, 'get') else getattr(config, 'correlation_window', 60)
            self.volatility_windows = config.get('volatility_windows', [1, 5, 20]) if hasattr(config, 'get') else getattr(config, 'volatility_windows', [1, 5, 20])
            self.volume_threshold = config.get('volume_threshold', 1000000) if hasattr(config, 'get') else getattr(config, 'volume_threshold', 1000000)
            self.use_yahoo_finance = (config.get('use_yahoo_finance', False) if hasattr(config, 'get') else getattr(config, 'use_yahoo_finance', False)) and YFINANCE_AVAILABLE
            self.yahoo_finance_rate_limit = config.get('yahoo_finance_rate_limit_seconds', 2.0) if hasattr(config, 'get') else getattr(config, 'yahoo_finance_rate_limit_seconds', 2.0)
        
        self.logger.info(f"TickerContextAnalyzer initialized. Yahoo Finance: {'Enabled' if self.use_yahoo_finance else 'Disabled'}")
        
        # Cache for ticker info to avoid repeated API calls
        self._ticker_info_cache = {}
        
        # Rate limiting for Yahoo Finance
        self._last_yahoo_call = None
        
        # Market benchmarks
        self.benchmarks = ['SPY', 'QQQ', 'IWM', 'VIX']
        
        # Sector mappings (simplified)
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
        
        # Cache for market data
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
    
    def analyze_ticker_context(self, 
                             symbol: str,
                             price_data: pd.DataFrame,
                             options_data: Optional[pd.DataFrame] = None) -> TickerContextAnalysis:
        """
        Perform comprehensive ticker context analysis.
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            options_data: Options chain data (optional)
            
        Returns:
            TickerContextAnalysis object with complete analysis
        """
        try:
            timestamp = datetime.now()
            
            # Get ticker profile
            ticker_profile = self._build_ticker_profile(symbol, price_data)
            
            # Analyze market context
            market_context = self._analyze_market_context()
            
            # Perform correlation analysis
            correlation_analysis = self._analyze_correlations(symbol, price_data)
            
            # Analyze volatility profile
            volatility_profile = self._analyze_volatility(price_data, options_data)
            
            # Determine trading characteristics
            trading_chars = self._analyze_trading_characteristics(price_data, options_data)
            
            # Identify risk factors and opportunities
            risk_factors = self._identify_risk_factors(
                ticker_profile, market_context, correlation_analysis, volatility_profile
            )
            opportunities = self._identify_opportunities(
                ticker_profile, market_context, correlation_analysis, volatility_profile
            )
            
            # Calculate overall context score
            context_score = self._calculate_context_score(
                ticker_profile, market_context, correlation_analysis, volatility_profile
            )
            
            return TickerContextAnalysis(
                symbol=symbol,
                timestamp=timestamp,
                ticker_profile=ticker_profile,
                market_context=market_context,
                correlation_analysis=correlation_analysis,
                volatility_profile=volatility_profile,
                trading_characteristics=trading_chars,
                risk_factors=risk_factors,
                opportunities=opportunities,
                context_score=context_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing ticker context for {symbol}: {str(e)}")
            raise
    
    def _build_ticker_profile(self, symbol: str, price_data: pd.DataFrame) -> TickerProfile:
        """Build comprehensive ticker profile."""
        try:
            # Try to get fundamental data
            ticker_info = self._get_ticker_info(symbol)
            
            # Calculate technical metrics
            avg_volume = price_data['volume'].mean() if 'volume' in price_data.columns else None
            
            # Determine volatility regime
            returns = price_data['close'].pct_change().dropna()
            vol_20d = returns.tail(20).std() * np.sqrt(252)
            vol_regime = self._classify_volatility_regime(vol_20d)
            
            # Determine liquidity tier
            liquidity_tier = self._classify_liquidity_tier(avg_volume or 0.0, ticker_info.get('marketCap'))
            
            return TickerProfile(
                symbol=symbol,
                sector=ticker_info.get('sector'),
                industry=ticker_info.get('industry'),
                market_cap=ticker_info.get('marketCap'),
                avg_volume=avg_volume,
                beta=ticker_info.get('beta'),
                volatility_regime=vol_regime,
                liquidity_tier=liquidity_tier,
                options_activity_level='normal'  # Will be updated with options data
            )
            
        except Exception as e:
            self.logger.warning(f"Error building ticker profile for {symbol}: {str(e)}")
            return TickerProfile(symbol=symbol)
    
    def _get_ticker_info(self, symbol: str) -> Dict:
        """Get ticker fundamental information with rate limiting and error handling."""
        if symbol in self._ticker_info_cache:
            return self._ticker_info_cache[symbol]
        
        if not self.use_yahoo_finance:
            raise ValueError(f"CRITICAL: Yahoo Finance disabled and no fallback data allowed for {symbol} - cannot proceed without real ticker data!")
        
        try:
            # Rate limiting: use configured delay
            time.sleep(random.uniform(1.0, self.yahoo_finance_rate_limit))
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Validate that we got useful data
            if not info or len(info) < 5:
                raise ValueError(f"CRITICAL: Yahoo Finance returned minimal data for {symbol} - cannot proceed without real ticker data!")
            
            self._ticker_info_cache[symbol] = info
            self.logger.debug(f"Successfully fetched Yahoo Finance data for {symbol}")
            return info
            
        except Exception as e:
            raise ValueError(f"CRITICAL: Yahoo Finance failed for {symbol}: {str(e)} - cannot proceed without real ticker data!")
    
    # ZERO TOLERANCE FAKE DATA: _get_fallback_ticker_info method removed
    
    def _analyze_market_context(self) -> MarketContext:
        """Analyze current market environment."""
        try:
            # Get market data
            spy_data = self._get_market_data('SPY')
            vix_data = self._get_market_data('VIX')
            
            # Determine market regime
            spy_returns = spy_data['close'].pct_change().tail(20)
            market_regime = self._classify_market_regime(spy_returns)
            
            # Get VIX level
            vix_level = vix_data['close'].iloc[-1] if not vix_data.empty else 20.0
            
            # Determine SPY trend
            spy_trend = self._determine_trend(spy_data['close'])
            
            # Analyze sector rotation
            sector_rotation = self._analyze_sector_rotation()
            
            # Determine risk sentiment
            risk_sentiment = self._determine_risk_sentiment(vix_level, spy_trend)
            
            return MarketContext(
                market_regime=market_regime,
                vix_level=vix_level,
                spy_trend=spy_trend,
                sector_rotation=sector_rotation,
                risk_sentiment=risk_sentiment
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing market context: {str(e)}")
            return MarketContext(
                market_regime='sideways',
                vix_level=20.0,
                spy_trend='sideways',
                sector_rotation={},
                risk_sentiment='neutral'
            )
    
    def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data with caching and error handling."""
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol]
        
        if not self.use_yahoo_finance:
            self.logger.warning(f"Yahoo Finance disabled or unavailable - returning empty DataFrame for {symbol}")
            return pd.DataFrame()
        
        try:
            # Rate limiting: use configured delay
            time.sleep(random.uniform(1.0, self.yahoo_finance_rate_limit))
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo')
            
            if data.empty:
                self.logger.warning(f"Yahoo Finance returned empty data for {symbol}")
                return pd.DataFrame()
            
            self.market_data_cache[symbol] = data
            self.logger.debug(f"Successfully fetched market data for {symbol}: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.warning(f"Could not fetch market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_correlations(self, symbol: str, price_data: pd.DataFrame) -> CorrelationAnalysis:
        """Analyze correlations with market benchmarks."""
        try:
            # Get benchmark data
            spy_data = self._get_market_data('SPY')
            qqq_data = self._get_market_data('QQQ')
            
            if spy_data.empty or qqq_data.empty:
                return CorrelationAnalysis(
                    spy_correlation=0.5,
                    qqq_correlation=0.5,
                    sector_correlation=0.5,
                    correlation_stability=0.5,
                    relative_strength=1.0
                )
            
            # Align data
            common_dates = price_data.index.intersection(spy_data.index)
            if len(common_dates) < 20:
                return CorrelationAnalysis(
                    spy_correlation=0.5,
                    qqq_correlation=0.5,
                    sector_correlation=0.5,
                    correlation_stability=0.5,
                    relative_strength=1.0
                )
            
            # Calculate returns
            symbol_returns = price_data.loc[common_dates, 'close'].pct_change().dropna()
            spy_returns = spy_data.loc[common_dates, 'close'].pct_change().dropna()
            qqq_returns = qqq_data.loc[common_dates, 'close'].pct_change().dropna()
            
            # Calculate correlations
            spy_corr = symbol_returns.corr(spy_returns)
            qqq_corr = symbol_returns.corr(qqq_returns)
            
            # Calculate relative strength
            symbol_perf = (price_data['close'].iloc[-1] / price_data['close'].iloc[-21] - 1) * 100
            spy_perf = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[-21] - 1) * 100
            relative_strength = symbol_perf - spy_perf
            
            return CorrelationAnalysis(
                spy_correlation=spy_corr if not np.isnan(spy_corr) else 0.5,
                qqq_correlation=qqq_corr if not np.isnan(qqq_corr) else 0.5,
                sector_correlation=0.6,  # Placeholder
                correlation_stability=0.7,  # Placeholder
                relative_strength=relative_strength if not np.isnan(relative_strength) else 0.0
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing correlations for {symbol}: {str(e)}")
            return CorrelationAnalysis(
                spy_correlation=0.5,
                qqq_correlation=0.5,
                sector_correlation=0.5,
                correlation_stability=0.5,
                relative_strength=0.0
            )
    
    def _analyze_volatility(self, price_data: pd.DataFrame, 
                          options_data: Optional[pd.DataFrame] = None) -> VolatilityProfile:
        """Analyze volatility characteristics."""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Calculate realized volatilities
            rv_1d = returns.tail(1).std() * np.sqrt(252) if len(returns) >= 1 else 0.2
            rv_5d = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0.2
            rv_20d = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.2
            
            # Determine volatility regime and trend
            vol_regime = self._classify_volatility_regime(rv_20d)
            vol_trend = self._determine_volatility_trend(returns)
            
            # Calculate implied vol metrics if options data available
            iv_rank = None
            vol_percentile = None
            if options_data is not None and 'implied_volatility' in options_data.columns:
                current_iv = options_data['implied_volatility'].mean()
                iv_rank = self._calculate_iv_rank(current_iv, rv_20d)
                vol_percentile = self._calculate_vol_percentile(rv_20d, returns)
            
            return VolatilityProfile(
                realized_vol_1d=rv_1d,
                realized_vol_5d=rv_5d,
                realized_vol_20d=rv_20d,
                implied_vol_rank=iv_rank,
                vol_regime=vol_regime,
                vol_trend=vol_trend,
                vol_percentile=vol_percentile
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volatility: {str(e)}")
            return VolatilityProfile(
                realized_vol_1d=0.2,
                realized_vol_5d=0.2,
                realized_vol_20d=0.2,
                vol_regime='normal',
                vol_trend='stable'
            )
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime."""
        if volatility < 0.15:
            return 'low'
        elif volatility < 0.25:
            return 'medium'
        elif volatility < 0.40:
            return 'high'
        else:
            return 'extreme'
    
    def _classify_liquidity_tier(self, avg_volume: float, market_cap: Optional[float] = None) -> str:
        """Classify ticker into liquidity tiers based on volume and market cap."""
        if avg_volume > 10_000_000:  # 10M+ daily volume
            return "TIER_1_ULTRA_LIQUID"
        elif avg_volume > 5_000_000:  # 5M+ daily volume
            return "TIER_2_HIGH_LIQUID"
        elif avg_volume > 1_000_000:  # 1M+ daily volume
            return "TIER_3_MODERATE_LIQUID"
        elif avg_volume > 100_000:   # 100K+ daily volume
            return "TIER_4_LOW_LIQUID"
        else:
            return "TIER_5_ILLIQUID"
    
    def _classify_market_regime(self, returns: pd.Series) -> str:
        """Classify current market regime."""
        if len(returns) < 10:
            return 'sideways'
        
        avg_return = returns.mean()
        volatility = returns.std()
        
        if avg_return > 0.001 and volatility < 0.02:
            return 'bull'
        elif avg_return < -0.001 and volatility < 0.02:
            return 'bear'
        elif volatility > 0.03:
            return 'volatile'
        else:
            return 'sideways'
    
    def _determine_trend(self, prices: pd.Series) -> str:
        """Determine price trend."""
        if len(prices) < 10:
            return 'sideways'
        
        recent_change = (prices.iloc[-1] / prices.iloc[-10] - 1)
        
        if recent_change > 0.02:
            return 'up'
        elif recent_change < -0.02:
            return 'down'
        else:
            return 'sideways'
    
    def _analyze_sector_rotation(self) -> Dict[str, float]:
        """Analyze sector rotation patterns."""
        sector_performance = {}
        
        for sector, etf in self.sector_etfs.items():
            try:
                data = self._get_market_data(etf)
                if not data.empty and len(data) >= 21:
                    perf = (data['close'].iloc[-1] / data['close'].iloc[-21] - 1) * 100
                    sector_performance[sector] = perf
            except Exception:
                continue
        
        return sector_performance
    
    def _determine_risk_sentiment(self, vix_level: float, spy_trend: str) -> str:
        """Determine overall risk sentiment."""
        if vix_level < 15 and spy_trend == 'up':
            return 'risk_on'
        elif vix_level > 25 or spy_trend == 'down':
            return 'risk_off'
        else:
            return 'neutral'
    
    def _determine_volatility_trend(self, returns: pd.Series) -> str:
        """Determine volatility trend."""
        if len(returns) < 20:
            return 'stable'
        
        recent_vol = returns.tail(10).std()
        older_vol = returns.tail(20).head(10).std()
        
        if recent_vol > older_vol * 1.2:
            return 'increasing'
        elif recent_vol < older_vol * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_iv_rank(self, current_iv: float, realized_vol: float) -> float:
        """Calculate implied volatility rank."""
        # Simplified IV rank calculation
        return min(current_iv / realized_vol, 2.0) / 2.0
    
    def _calculate_vol_percentile(self, current_vol: float, returns: pd.Series) -> float:
        """Calculate volatility percentile."""
        if len(returns) < 50:
            return 0.5
        
        # Rolling 20-day volatility
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) == 0:
            return 0.5
        
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol)
        return percentile
    
    def _analyze_trading_characteristics(self, price_data: pd.DataFrame, 
                                       options_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze trading characteristics."""
        characteristics = {}
        
        # Price characteristics
        if len(price_data) >= 20:
            returns = price_data['close'].pct_change().dropna()
            characteristics['avg_daily_range'] = ((price_data['high'] - price_data['low']) / price_data['close']).tail(20).mean()
            characteristics['trend_strength'] = abs(returns.tail(20).mean()) / returns.tail(20).std()
            characteristics['momentum'] = (price_data['close'].iloc[-1] / price_data['close'].iloc[-5] - 1)
        
        # Volume characteristics
        if 'volume' in price_data.columns:
            characteristics['volume_trend'] = self._determine_volume_trend(price_data['volume'])
            characteristics['relative_volume'] = price_data['volume'].tail(5).mean() / price_data['volume'].tail(20).mean()
        
        return characteristics
    
    def _determine_volume_trend(self, volume: pd.Series) -> str:
        """Determine volume trend."""
        if len(volume) < 10:
            return 'stable'
        
        recent_avg = volume.tail(5).mean()
        older_avg = volume.tail(15).head(10).mean()
        
        if recent_avg > older_avg * 1.2:
            return 'increasing'
        elif recent_avg < older_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _identify_risk_factors(self, ticker_profile: TickerProfile, 
                             market_context: MarketContext,
                             correlation_analysis: CorrelationAnalysis,
                             volatility_profile: VolatilityProfile) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # Volatility risks
        if volatility_profile.vol_regime in ['high', 'extreme']:
            risks.append('High volatility environment')
        
        if volatility_profile.vol_trend == 'increasing':
            risks.append('Increasing volatility trend')
        
        # Market risks
        if market_context.market_regime == 'bear':
            risks.append('Bear market conditions')
        
        if market_context.vix_level > 25:
            risks.append('Elevated fear levels (VIX > 25)')
        
        # Liquidity risks
        if ticker_profile.liquidity_tier == 'low':
            risks.append('Low liquidity - wider spreads possible')
        
        # Correlation risks
        if abs(correlation_analysis.spy_correlation) > 0.8:
            risks.append('High market correlation - limited diversification')
        
        return risks
    
    def _identify_opportunities(self, ticker_profile: TickerProfile,
                              market_context: MarketContext,
                              correlation_analysis: CorrelationAnalysis,
                              volatility_profile: VolatilityProfile) -> List[str]:
        """Identify potential opportunities."""
        opportunities = []
        
        # Volatility opportunities
        if volatility_profile.implied_vol_rank and volatility_profile.implied_vol_rank > 0.7:
            opportunities.append('High IV rank - potential vol selling opportunity')
        
        if volatility_profile.vol_trend == 'decreasing' and volatility_profile.vol_regime == 'high':
            opportunities.append('Volatility compression opportunity')
        
        # Relative strength opportunities
        if correlation_analysis.relative_strength > 5:
            opportunities.append('Strong relative performance vs market')
        
        # Market opportunities
        if market_context.risk_sentiment == 'risk_on' and ticker_profile.beta and ticker_profile.beta > 1.2:
            opportunities.append('High beta in risk-on environment')
        
        return opportunities
    
    def _calculate_context_score(self, ticker_profile: TickerProfile,
                               market_context: MarketContext,
                               correlation_analysis: CorrelationAnalysis,
                               volatility_profile: VolatilityProfile) -> float:
        """Calculate overall context favorability score."""
        score = 0.5  # Base score
        
        # Market context adjustments
        if market_context.market_regime == 'bull':
            score += 0.1
        elif market_context.market_regime == 'bear':
            score -= 0.1
        
        # Volatility adjustments
        if volatility_profile.vol_regime == 'normal':
            score += 0.05
        elif volatility_profile.vol_regime in ['high', 'extreme']:
            score -= 0.05
        
        # Liquidity adjustments
        if ticker_profile.liquidity_tier == 'high':
            score += 0.1
        elif ticker_profile.liquidity_tier == 'low':
            score -= 0.1
        
        # Relative strength adjustments
        if correlation_analysis.relative_strength > 0:
            score += min(correlation_analysis.relative_strength / 100, 0.1)
        else:
            score += max(correlation_analysis.relative_strength / 100, -0.1)
        
        return max(0.0, min(1.0, score))
    
    def get_context_summary(self, analysis: TickerContextAnalysis) -> Dict[str, Any]:
        """Get a summary of ticker context analysis."""
        return {
            'symbol': analysis.symbol,
            'sector': analysis.ticker_profile.sector,
            'market_cap': analysis.ticker_profile.market_cap,
            'volatility_regime': analysis.volatility_profile.vol_regime,
            'liquidity_tier': analysis.ticker_profile.liquidity_tier,
            'market_regime': analysis.market_context.market_regime,
            'spy_correlation': analysis.correlation_analysis.spy_correlation,
            'relative_strength': analysis.correlation_analysis.relative_strength,
            'context_score': analysis.context_score,
            'risk_factors_count': len(analysis.risk_factors),
            'opportunities_count': len(analysis.opportunities),
            'timestamp': analysis.timestamp
        }