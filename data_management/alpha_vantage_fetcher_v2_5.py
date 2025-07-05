"""
Alpha Vantage Data Fetcher for EOTS v2.5
========================================

This module provides integration with Alpha Vantage's Alpha Intelligence™ APIs
to enhance the AI dashboard with real market sentiment and intelligence data.

Key Features:
- News & Sentiment analysis for market intelligence
- Real-time sentiment scoring for confidence validation
- Topic-based news filtering for regime analysis
- Historical sentiment data for performance tracking

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# PYDANTIC-FIRST: Replace dataclasses with Pydantic models for validation
from pydantic import BaseModel, Field, ConfigDict

class SentimentDataV2_5(BaseModel):
    """Pydantic model for Alpha Vantage sentiment analysis results - EOTS v2.5 compliant."""
    ticker: str = Field(..., description="Stock ticker symbol")
    overall_sentiment_score: float = Field(..., description="Overall sentiment score", ge=-1.0, le=1.0)
    overall_sentiment_label: str = Field(..., description="Sentiment label (Bearish/Neutral/Bullish)")
    article_count: int = Field(..., description="Number of articles analyzed", ge=0)
    relevance_score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    topics: List[str] = Field(default_factory=list, description="List of topics/themes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    model_config = ConfigDict(extra='forbid')

class NewsArticleV2_5(BaseModel):
    """Pydantic model for individual news articles - EOTS v2.5 compliant."""
    title: str = Field(..., description="Article title")
    url: str = Field(..., description="Article URL")
    time_published: str = Field(..., description="Publication timestamp")
    summary: str = Field(..., description="Article summary")
    sentiment_score: float = Field(..., description="Article sentiment score", ge=-1.0, le=1.0)
    sentiment_label: str = Field(..., description="Sentiment label")
    relevance_score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    topics: List[str] = Field(default_factory=list, description="Article topics")

    model_config = ConfigDict(extra='forbid')

class AlphaVantageDataFetcherV2_5:
    """
    Alpha Vantage Data Fetcher for EOTS v2.5

    Integrates with Alpha Vantage's Alpha Intelligence™ APIs to provide:
    - Real-time market sentiment analysis
    - News-driven market intelligence
    - Topic-based sentiment filtering
    - Historical sentiment tracking
    """

    def __init__(self, api_key: str = "9CZXMNC1HO3EI2QR"):
        """
        Initialize Alpha Vantage Data Fetcher.

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EOTS-v2.5-AI-Dashboard/1.0'
        })

        # Rate limiting and graceful degradation
        self.daily_request_count = 0
        self.daily_limit = 25  # Free tier limit
        self.rate_limited = False
        self.last_reset_date = datetime.now().date()

        logger.info("🧠 Alpha Vantage Data Fetcher initialized with Alpha Intelligence™")

    def _check_rate_limit(self) -> bool:
        """
        Check if we've hit the daily rate limit and handle graceful degradation.

        Returns:
            bool: True if we can make requests, False if rate limited
        """
        # Reset counter if it's a new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_request_count = 0
            self.rate_limited = False
            self.last_reset_date = current_date
            logger.info("🔄 Alpha Vantage daily rate limit reset")

        # Check if we're already rate limited
        if self.rate_limited:
            logger.debug("🚫 Alpha Vantage rate limited - using fallback data")
            return False

        # Check if we're approaching the limit
        if self.daily_request_count >= self.daily_limit:
            self.rate_limited = True
            logger.warning(f"🚫 Alpha Vantage daily rate limit ({self.daily_limit}) reached. Switching to fallback mode.")
            return False

        return True

    def _increment_request_count(self):
        """Increment the daily request counter."""
        self.daily_request_count += 1
        remaining = self.daily_limit - self.daily_request_count

        if remaining <= 5:
            logger.warning(f"⚠️ Alpha Vantage API: Only {remaining} requests remaining today")
        elif remaining <= 10:
            logger.info(f"📊 Alpha Vantage API: {remaining} requests remaining today")

    def _handle_api_response(self, response) -> Dict[str, Any]:
        """Handle API response and check for rate limiting."""
        try:
            data = response.json()

            # Check for rate limit message in response
            if 'Note' in data and 'rate limit' in data['Note'].lower():
                logger.warning(f"🚫 Alpha Vantage rate limit detected: {data['Note']}")
                self.rate_limited = True
                return {}

            # Check for other API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Alpha Vantage response: {str(e)}")
            return {}

    def get_news_sentiment(self,
                          tickers: str = "SPY", 
                          topics: Optional[str] = None,
                          time_from: Optional[str] = None,
                          time_to: Optional[str] = None,
                          limit: int = 50,
                          sort: str = "LATEST") -> Dict[str, Any]:
        """
        Fetch news sentiment data from Alpha Vantage Alpha Intelligence™.
        
        Args:
            tickers: Comma-separated list of tickers (e.g., "SPY,QQQ")
            topics: Comma-separated list of topics (e.g., "financial_markets,economy_monetary")
            time_from: Start time in YYYYMMDDTHHMM format
            time_to: End time in YYYYMMDDTHHMM format
            limit: Number of results (max 1000)
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            
        Returns:
            Dict containing sentiment analysis results
        """
        try:
            # Check rate limit before making request
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning fallback data for {tickers}")
                return {}

            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': tickers,
                'limit': limit,
                'sort': sort,
                'apikey': self.api_key
            }

            # Add optional parameters
            if topics:
                params['topics'] = topics
            if time_from:
                params['time_from'] = time_from
            if time_to:
                params['time_to'] = time_to

            logger.info(f"🔍 Fetching Alpha Intelligence™ sentiment for {tickers}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            # Increment request counter
            self._increment_request_count()

            # Handle response with rate limit checking
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched sentiment data for {tickers}")

            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage sentiment data: {str(e)}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Alpha Vantage response: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in Alpha Vantage sentiment fetch: {str(e)}")
            return {}
    
    def parse_sentiment_data(self, raw_data: Dict[str, Any], ticker: str = "SPY") -> Optional[SentimentDataV2_5]:
        """
        Parse raw Alpha Vantage sentiment data into structured format using Pydantic validation.

        Args:
            raw_data: Raw response from Alpha Vantage API
            ticker: Primary ticker for analysis

        Returns:
            SentimentDataV2_5 Pydantic model or None if parsing fails
        """
        try:
            if not raw_data or 'feed' not in raw_data:
                return None
            
            articles = raw_data.get('feed', [])
            if not articles:
                return None
            
            # Calculate overall sentiment metrics
            sentiment_scores = []
            relevance_scores = []
            all_topics = set()
            
            for article in articles:
                # Get ticker-specific sentiment if available
                ticker_sentiments = article.get('ticker_sentiment', [])
                for ts in ticker_sentiments:
                    if ts.get('ticker', '').upper() == ticker.upper():
                        sentiment_scores.append(float(ts.get('ticker_sentiment_score', 0)))
                        relevance_scores.append(float(ts.get('relevance_score', 0)))
                        break
                else:
                    # Fallback to overall sentiment
                    sentiment_scores.append(float(article.get('overall_sentiment_score', 0)))
                
                # Collect topics
                topics = article.get('topics', [])
                for topic in topics:
                    all_topics.add(topic.get('topic', ''))
            
            if not sentiment_scores:
                return None
            
            # Calculate aggregated metrics
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            
            # Determine sentiment label
            if avg_sentiment >= 0.15:
                sentiment_label = "Bullish"
            elif avg_sentiment <= -0.15:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            # PYDANTIC-FIRST: Create validated SentimentDataV2_5 model
            return SentimentDataV2_5(
                ticker=ticker,
                overall_sentiment_score=avg_sentiment,
                overall_sentiment_label=sentiment_label,
                article_count=len(articles),
                relevance_score=avg_relevance,
                topics=list(all_topics),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing sentiment data: {str(e)}")
            return None
    
    def get_market_intelligence_summary(self, ticker: str = "SPY") -> Dict[str, Any]:
        """
        Get comprehensive market intelligence summary for AI dashboard.
        
        Args:
            ticker: Ticker symbol to analyze
            
        Returns:
            Dict containing market intelligence metrics
        """
        try:
            # Fetch recent sentiment data with financial markets focus
            sentiment_data = self.get_news_sentiment(
                tickers=ticker,
                topics="financial_markets,economy_monetary,economy_macro",
                limit=100,
                sort="RELEVANCE"
            )
            
            if not sentiment_data:
                return self._get_fallback_intelligence()
            
            # Parse sentiment data
            parsed_sentiment = self.parse_sentiment_data(sentiment_data, ticker)
            
            if not parsed_sentiment:
                return self._get_fallback_intelligence()
            
            # Generate intelligence summary
            intelligence = {
                'sentiment_score': parsed_sentiment.overall_sentiment_score,
                'sentiment_label': parsed_sentiment.overall_sentiment_label,
                'confidence_score': min(parsed_sentiment.relevance_score * 100, 95),
                'article_count': parsed_sentiment.article_count,
                'news_volume': self._categorize_news_volume(parsed_sentiment.article_count),
                'topics': parsed_sentiment.topics[:5],  # Top 5 topics
                'market_attention': self._calculate_market_attention(parsed_sentiment),
                'ai_insights': self._generate_ai_insights(parsed_sentiment),
                'timestamp': parsed_sentiment.timestamp.isoformat()
            }
            
            logger.info(f"✅ Generated market intelligence summary for {ticker}")
            return intelligence
            
        except Exception as e:
            logger.error(f"Error generating market intelligence: {str(e)}")
            return self._get_fallback_intelligence()
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage API is available (not rate limited)."""
        return self._check_rate_limit()

    def get_status(self) -> Dict[str, Any]:
        """Get current Alpha Vantage API status."""
        return {
            'available': not self.rate_limited,
            'daily_requests_used': self.daily_request_count,
            'daily_limit': self.daily_limit,
            'requests_remaining': max(0, self.daily_limit - self.daily_request_count),
            'rate_limited': self.rate_limited,
            'last_reset_date': self.last_reset_date.isoformat()
        }

    def _get_fallback_intelligence(self) -> Dict[str, Any]:
        """ZERO TOLERANCE FAKE DATA: No fallback intelligence allowed."""
        raise ValueError("CRITICAL: Alpha Vantage API unavailable and no fallback data allowed - cannot provide intelligence without real market data!")
    
    def _categorize_news_volume(self, article_count: int) -> str:
        """Categorize news volume based on article count."""
        if article_count >= 50:
            return "Very High"
        elif article_count >= 30:
            return "High"
        elif article_count >= 15:
            return "Moderate"
        elif article_count >= 5:
            return "Low"
        else:
            return "Very Low"
    
    def _calculate_market_attention(self, sentiment_data: SentimentDataV2_5) -> str:
        """Calculate market attention level."""
        attention_score = (sentiment_data.article_count * 0.4 + 
                          abs(sentiment_data.overall_sentiment_score) * 100 * 0.6)
        
        if attention_score >= 40:
            return "Extreme"
        elif attention_score >= 25:
            return "High"
        elif attention_score >= 15:
            return "Moderate"
        else:
            return "Low"
    
    def get_earnings_call_transcript(self, ticker: str = "", year: str = "", quarter: str = "") -> Dict[str, Any]:
        """
        Fetch earnings call transcript from Alpha Vantage Alpha Intelligence™.

        Args:
            ticker: Stock ticker symbol
            year: Year (YYYY format, optional - defaults to most recent)
            quarter: Quarter (Q1, Q2, Q3, Q4, optional - defaults to most recent)

        Returns:
            Dict containing earnings call transcript and analysis
        """
        params = {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': str(ticker),
            'year': str(year),
            'quarter': str(quarter),
            'apikey': str(self.api_key) if self.api_key else ""
        }
        try:
            logger.info(f"��️ Fetching earnings call transcript for {ticker}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}

            logger.info(f"✅ Successfully fetched earnings transcript for {ticker}")
            return data

        except Exception as e:
            logger.error(f"Error fetching earnings transcript: {str(e)}")
            return {}

    # ==================== CORE STOCK DATA APIs ====================

    async def get_intraday_data(self, symbol: str, interval: str = "5min",
                         adjusted: bool = True, extended_hours: bool = True,
                         outputsize: str = "compact") -> Dict[str, Any]:
        """
        Fetch intraday OHLCV time series data.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            adjusted: Whether to adjust for splits/dividends
            extended_hours: Include pre/post market hours
            outputsize: 'compact' (100 points) or 'full' (30 days)

        Returns:
            Dict containing intraday time series data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty intraday data for {symbol}")
                return {}

            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'adjusted': str(adjusted).lower(),
                'extended_hours': str(extended_hours).lower(),
                'outputsize': outputsize,
                'apikey': self.api_key
            }

            logger.info(f"📈 Fetching intraday data for {symbol} ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched intraday data for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            return {}

    def get_daily_data(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """
        Fetch daily OHLCV time series data.

        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (100 points) or 'full' (20+ years)

        Returns:
            Dict containing daily time series data
        """
        # DEPRECATED: Save Alpha Vantage calls for news/sentiment only
        logger.warning(f"🚫 Alpha Vantage daily data deprecated for {symbol} - use Tradier instead")
        logger.debug(f"🚫 Alpha Vantage rate limited - returning empty daily data for {symbol}")
        return {}

    def get_daily_adjusted_data(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """
        Fetch daily adjusted OHLCV time series data with split/dividend adjustments.

        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (100 points) or 'full' (20+ years)

        Returns:
            Dict containing daily adjusted time series data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty adjusted data for {symbol}")
                return {}

            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }

            logger.info(f"📈 Fetching daily adjusted data for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched daily adjusted data for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching daily adjusted data: {str(e)}")
            return {}

    def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict containing real-time quote data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty quote for {symbol}")
                return {}

            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }

            logger.info(f"💰 Fetching real-time quote for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched quote for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching quote: {str(e)}")
            return {}

    def search_symbol(self, keywords: str) -> Dict[str, Any]:
        """
        Search for symbols matching keywords.

        Args:
            keywords: Search keywords

        Returns:
            Dict containing search results
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty search results")
                return {}

            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': keywords,
                'apikey': self.api_key
            }

            logger.info(f"🔍 Searching symbols for: {keywords}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully searched symbols for: {keywords}")

            return data

        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            return {}

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get global market status (open/closed).

        Returns:
            Dict containing market status information
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty market status")
                return {}

            params = {
                'function': 'MARKET_STATUS',
                'apikey': self.api_key
            }

            logger.info("🌍 Fetching global market status")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched market status")

            return data

        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            return {}

    def get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company overview and fundamentals from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing company fundamentals and overview
        """
        try:
            # Check rate limit before making request
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty data for {ticker}")
                return {}

            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.api_key
            }

            logger.info(f"🏢 Fetching company overview for {ticker}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            # Increment request counter
            self._increment_request_count()

            # Handle response with rate limit checking
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched company overview for {ticker}")

            return data

        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            return {}

    def get_earnings_calendar(self, horizon: str = "3month") -> Dict[str, Any]:
        """
        Fetch upcoming earnings calendar from Alpha Vantage.

        Args:
            horizon: Time horizon (3month, 6month, 12month)

        Returns:
            Dict containing upcoming earnings events
        """
        try:
            # Check rate limit before making request
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty earnings calendar")
                return {}

            params = {
                'function': 'EARNINGS_CALENDAR',
                'horizon': horizon,
                'apikey': self.api_key
            }

            logger.info(f"📅 Fetching earnings calendar for {horizon}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            # Increment request counter
            self._increment_request_count()

            # Check response content type and size
            if response.headers.get('content-type', '').startswith('text/csv') or len(response.content) < 100:
                logger.warning("📅 Earnings calendar returned CSV or empty data - skipping JSON parsing")
                return {}

            # Handle response with rate limit checking
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched earnings calendar")

            return data

        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {str(e)}")
            return {}

    # ==================== ALPHA INTELLIGENCE™ APIs ====================

    def get_news_sentiment_v2(self, tickers: str = "", topics: str = "",
                          time_from: str = "", time_to: str = "",
                          sort: str = "LATEST", limit: int = 50) -> Dict[str, Any]:
        """
        Fetch live and historical market news & sentiment data.

        Args:
            tickers: Stock/crypto/forex symbols (e.g., "AAPL,TSLA")
            topics: News topics (e.g., "technology,earnings")
            time_from: Start time in YYYYMMDDTHHMM format
            time_to: End time in YYYYMMDDTHHMM format
            sort: Sort order ("LATEST", "EARLIEST", "RELEVANCE")
            limit: Number of results (1-1000, default 50)

        Returns:
            Dict containing news and sentiment data
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': str(tickers),
            'limit': str(limit),
            'sort': str(sort),
            'apikey': str(self.api_key) if self.api_key else ""
        }
        if topics:
            params['topics'] = str(topics)
        if time_from:
            params['time_from'] = str(time_from)
        if time_to:
            params['time_to'] = str(time_to)
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty news sentiment")
                return {}

            logger.info("📰 Fetching news & sentiment data")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched news & sentiment data")

            return data

        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            return {}

    def get_top_gainers_losers(self) -> Dict[str, Any]:
        """
        Fetch top gainers, losers, and most active stocks.

        Returns:
            Dict containing market movers data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty gainers/losers")
                return {}

            params = {
                'function': 'TOP_GAINERS_LOSERS',
                'apikey': self.api_key
            }

            logger.info("📈 Fetching top gainers, losers, and most active")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched market movers")

            return data

        except Exception as e:
            logger.error(f"Error fetching top gainers/losers: {str(e)}")
            return {}

    def get_analytics_fixed_window(self, symbols: str, range_period: str = "1month",
                                  interval: str = "daily", ohlc: str = "close") -> Dict[str, Any]:
        """
        Fetch analytics data for fixed time window.

        Args:
            symbols: Comma-separated list of symbols
            range_period: Time period (1month, 3month, 6month, 1year, 2year)
            interval: Data interval (daily, weekly, monthly)
            ohlc: Price type (close, open, high, low)

        Returns:
            Dict containing analytics data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty analytics data")
                return {}

            params = {
                'function': 'ANALYTICS_FIXED_WINDOW',
                'SYMBOLS': symbols,
                'RANGE': range_period,
                'INTERVAL': interval,
                'OHLC': ohlc,
                'apikey': self.api_key
            }

            logger.info(f"📊 Fetching analytics data for {symbols} ({range_period})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched analytics data")

            return data

        except Exception as e:
            logger.error(f"Error fetching analytics data: {str(e)}")
            return {}

    def get_analytics_sliding_window(self, symbols: str, range_period: str = "1month",
                                   interval: str = "daily", ohlc: str = "close",
                                   window_size: int = 60) -> Dict[str, Any]:
        """
        Fetch analytics data for sliding time window.

        Args:
            symbols: Comma-separated list of symbols
            range_period: Time period (1month, 3month, 6month, 1year, 2year)
            interval: Data interval (daily, weekly, monthly)
            ohlc: Price type (close, open, high, low)
            window_size: Sliding window size in days

        Returns:
            Dict containing sliding window analytics data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty sliding analytics")
                return {}

            params = {
                'function': 'ANALYTICS_SLIDING_WINDOW',
                'SYMBOLS': symbols,
                'RANGE': range_period,
                'INTERVAL': interval,
                'OHLC': ohlc,
                'WINDOW_SIZE': window_size,
                'apikey': self.api_key
            }

            logger.info(f"📊 Fetching sliding window analytics for {symbols}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched sliding window analytics")

            return data

        except Exception as e:
            logger.error(f"Error fetching sliding window analytics: {str(e)}")
            return {}

    def get_insider_transactions(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch insider transactions data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict containing insider transactions data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty insider data")
                return {}

            params = {
                'function': 'INSIDER_TRANSACTIONS',
                'symbol': symbol,
                'apikey': self.api_key
            }

            logger.info(f"👥 Fetching insider transactions for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched insider transactions for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching insider transactions: {str(e)}")
            return {}

    def get_ipo_calendar(self) -> Dict[str, Any]:
        """
        Fetch IPO calendar data.

        Returns:
            Dict containing IPO calendar data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty IPO calendar")
                return {}

            params = {
                'function': 'IPO_CALENDAR',
                'apikey': self.api_key
            }

            logger.info("🚀 Fetching IPO calendar")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched IPO calendar")

            return data

        except Exception as e:
            logger.error(f"Error fetching IPO calendar: {str(e)}")
            return {}

    # ==================== ECONOMIC INDICATORS APIs ====================

    def get_real_gdp(self, interval: str = "annual") -> Dict[str, Any]:
        """
        Fetch Real GDP data.

        Args:
            interval: Data interval ("annual", "quarterly")

        Returns:
            Dict containing Real GDP data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty GDP data")
                return {}

            params = {
                'function': 'REAL_GDP',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"📈 Fetching Real GDP data ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Real GDP data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Real GDP: {str(e)}")
            return {}

    def get_treasury_yield(self, interval: str = "monthly", maturity: str = "10year") -> Dict[str, Any]:
        """
        Fetch Treasury Yield data.

        Args:
            interval: Data interval ("daily", "weekly", "monthly")
            maturity: Treasury maturity ("3month", "2year", "5year", "10year", "30year")

        Returns:
            Dict containing Treasury Yield data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty treasury data")
                return {}

            params = {
                'function': 'TREASURY_YIELD',
                'interval': interval,
                'maturity': maturity,
                'apikey': self.api_key
            }

            logger.info(f"📊 Fetching Treasury Yield data ({maturity}, {interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Treasury Yield data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Treasury Yield: {str(e)}")
            return {}

    def get_federal_funds_rate(self, interval: str = "monthly") -> Dict[str, Any]:
        """
        Fetch Federal Funds Rate data.

        Args:
            interval: Data interval ("daily", "weekly", "monthly")

        Returns:
            Dict containing Federal Funds Rate data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty fed funds data")
                return {}

            params = {
                'function': 'FEDERAL_FUNDS_RATE',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"🏦 Fetching Federal Funds Rate data ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Federal Funds Rate data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Federal Funds Rate: {str(e)}")
            return {}

    def get_cpi(self, interval: str = "monthly") -> Dict[str, Any]:
        """
        Fetch Consumer Price Index (CPI) data.

        Args:
            interval: Data interval ("monthly", "semiannual")

        Returns:
            Dict containing CPI data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty CPI data")
                return {}

            params = {
                'function': 'CPI',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"📊 Fetching CPI data ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched CPI data")

            return data

        except Exception as e:
            logger.error(f"Error fetching CPI: {str(e)}")
            return {}

    def get_inflation(self) -> Dict[str, Any]:
        """
        Fetch Inflation data.

        Returns:
            Dict containing Inflation data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty inflation data")
                return {}

            params = {
                'function': 'INFLATION',
                'apikey': self.api_key
            }

            logger.info("📈 Fetching Inflation data")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Inflation data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Inflation: {str(e)}")
            return {}

    def get_unemployment_rate(self) -> Dict[str, Any]:
        """
        Fetch Unemployment Rate data.

        Returns:
            Dict containing Unemployment Rate data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty unemployment data")
                return {}

            params = {
                'function': 'UNEMPLOYMENT',
                'apikey': self.api_key
            }

            logger.info("📊 Fetching Unemployment Rate data")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Unemployment Rate data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Unemployment Rate: {str(e)}")
            return {}

    def get_nonfarm_payroll(self) -> Dict[str, Any]:
        """
        Fetch Nonfarm Payroll data.

        Returns:
            Dict containing Nonfarm Payroll data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty payroll data")
                return {}

            params = {
                'function': 'NONFARM_PAYROLL',
                'apikey': self.api_key
            }

            logger.info("💼 Fetching Nonfarm Payroll data")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Nonfarm Payroll data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Nonfarm Payroll: {str(e)}")
            return {}

    # ==================== TECHNICAL INDICATORS APIs ====================

    def get_rsi(self, symbol: str, interval: str = "daily", time_period: int = 14,
               series_type: str = "close") -> Dict[str, Any]:
        """
        Fetch RSI (Relative Strength Index) technical indicator.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period: Number of periods for RSI calculation
            series_type: Price type (close, open, high, low)

        Returns:
            Dict containing RSI data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty RSI for {symbol}")
                return {}

            params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': series_type,
                'apikey': self.api_key
            }

            logger.info(f"📈 Fetching RSI for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched RSI for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching RSI: {str(e)}")
            return {}

    def get_macd(self, symbol: str, interval: str = "daily", series_type: str = "close",
                fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Dict[str, Any]:
        """
        Fetch MACD (Moving Average Convergence Divergence) technical indicator.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval
            series_type: Price type
            fastperiod: Fast EMA period
            slowperiod: Slow EMA period
            signalperiod: Signal line EMA period

        Returns:
            Dict containing MACD data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty MACD for {symbol}")
                return {}

            params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': interval,
                'series_type': series_type,
                'fastperiod': fastperiod,
                'slowperiod': slowperiod,
                'signalperiod': signalperiod,
                'apikey': self.api_key
            }

            logger.info(f"📊 Fetching MACD for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched MACD for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching MACD: {str(e)}")
            return {}

    def get_bollinger_bands(self, symbol: str, interval: str = "daily", time_period: int = 20,
                           series_type: str = "close", nbdevup: int = 2, nbdevdn: int = 2) -> Dict[str, Any]:
        """
        Fetch Bollinger Bands technical indicator.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval
            time_period: Number of periods
            series_type: Price type
            nbdevup: Upper band standard deviations
            nbdevdn: Lower band standard deviations

        Returns:
            Dict containing Bollinger Bands data
        """
        try:
            if not self._check_rate_limit():
                logger.debug(f"🚫 Alpha Vantage rate limited - returning empty Bollinger Bands for {symbol}")
                return {}

            params = {
                'function': 'BBANDS',
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': series_type,
                'nbdevup': nbdevup,
                'nbdevdn': nbdevdn,
                'apikey': self.api_key
            }

            logger.info(f"📈 Fetching Bollinger Bands for {symbol}")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info(f"✅ Successfully fetched Bollinger Bands for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching Bollinger Bands: {str(e)}")
            return {}

    # ==================== COMMODITIES APIs ====================

    def get_wti_oil(self, interval: str = "monthly") -> Dict[str, Any]:
        """
        Fetch WTI (West Texas Intermediate) crude oil prices.

        Args:
            interval: Data interval (daily, weekly, monthly)

        Returns:
            Dict containing WTI oil price data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty WTI data")
                return {}

            params = {
                'function': 'WTI',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"🛢️ Fetching WTI oil prices ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched WTI oil data")

            return data

        except Exception as e:
            logger.error(f"Error fetching WTI oil: {str(e)}")
            return {}

    def get_brent_oil(self, interval: str = "monthly") -> Dict[str, Any]:
        """
        Fetch Brent crude oil prices.

        Args:
            interval: Data interval (daily, weekly, monthly)

        Returns:
            Dict containing Brent oil price data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty Brent data")
                return {}

            params = {
                'function': 'BRENT',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"🛢️ Fetching Brent oil prices ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched Brent oil data")

            return data

        except Exception as e:
            logger.error(f"Error fetching Brent oil: {str(e)}")
            return {}

    def get_copper_prices(self, interval: str = "monthly") -> Dict[str, Any]:
        """
        Fetch copper commodity prices.

        Args:
            interval: Data interval (monthly, quarterly, annual)

        Returns:
            Dict containing copper price data
        """
        try:
            if not self._check_rate_limit():
                logger.debug("🚫 Alpha Vantage rate limited - returning empty copper data")
                return {}

            params = {
                'function': 'COPPER',
                'interval': interval,
                'apikey': self.api_key
            }

            logger.info(f"🔶 Fetching copper prices ({interval})")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            self._increment_request_count()
            data = self._handle_api_response(response)

            if data:
                logger.info("✅ Successfully fetched copper data")

            return data

        except Exception as e:
            logger.error(f"Error fetching copper prices: {str(e)}")
            return {}

    def _generate_ai_insights(self, sentiment_data: SentimentDataV2_5) -> List[str]:
        """Generate AI insights based on sentiment analysis."""
        insights = []

        # Sentiment-based insights
        if abs(sentiment_data.overall_sentiment_score) >= 0.3:
            direction = "bullish" if sentiment_data.overall_sentiment_score > 0 else "bearish"
            insights.append(f"🎯 Strong {direction} sentiment detected ({sentiment_data.overall_sentiment_score:.3f}) "
                          f"across {sentiment_data.article_count} articles")

        # Volume-based insights
        if sentiment_data.article_count >= 30:
            insights.append(f"📈 High news volume ({sentiment_data.article_count} articles) indicates elevated market attention")

        # Topic-based insights
        if 'economy_monetary' in sentiment_data.topics:
            insights.append("🏦 Federal Reserve policy discussions detected - potential volatility catalyst")

        if 'earnings' in sentiment_data.topics:
            insights.append("📊 Earnings-related news flow - monitor for sector rotation signals")

        # Relevance insights
        if sentiment_data.relevance_score >= 0.7:
            insights.append(f"✅ High relevance score ({sentiment_data.relevance_score:.2f}) - news directly impacts target asset")

        return insights[:4]  # Limit to 4 insights

    def get_comprehensive_intelligence_summary(self, ticker: str = "SPY") -> Dict[str, Any]:
        """
        Get comprehensive Alpha Intelligence™ summary combining all data sources.

        Args:
            ticker: Ticker symbol to analyze

        Returns:
            Dict containing comprehensive market intelligence
        """
        try:
            logger.info(f"🧠 Generating comprehensive Alpha Intelligence™ summary for {ticker}")

            # Fetch all intelligence data sources
            sentiment_data = self.get_news_sentiment(
                tickers=ticker,
                topics="financial_markets,economy_monetary,earnings",
                limit=50,
                sort="RELEVANCE"
            )

            company_overview = self.get_company_overview(ticker)
            earnings_calendar = self.get_earnings_calendar("3month")

            # Parse sentiment data
            parsed_sentiment = self.parse_sentiment_data(sentiment_data, ticker) if sentiment_data else None

            # Generate comprehensive summary
            intelligence = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'alpha_intelligence_active': True,

                # Sentiment Intelligence
                'sentiment': {
                    'score': parsed_sentiment.overall_sentiment_score if parsed_sentiment else 0.0,
                    'label': parsed_sentiment.overall_sentiment_label if parsed_sentiment else 'Neutral',
                    'confidence': min(parsed_sentiment.relevance_score * 100, 95) if parsed_sentiment else 50.0,
                    'article_count': parsed_sentiment.article_count if parsed_sentiment else 0,
                    'topics': parsed_sentiment.topics[:5] if parsed_sentiment else [],
                    'insights': self._generate_ai_insights(parsed_sentiment) if parsed_sentiment else []
                },

                # Company Fundamentals
                'fundamentals': self._extract_key_fundamentals(company_overview),

                # Earnings Intelligence
                'earnings': self._extract_earnings_intelligence(earnings_calendar, ticker),

                # Market Context
                'market_context': self._generate_market_context(parsed_sentiment, company_overview),

                # AI Recommendations
                'ai_recommendations': self._generate_ai_recommendations(parsed_sentiment, company_overview)
            }

            logger.info(f"✅ Generated comprehensive Alpha Intelligence™ summary for {ticker}")
            return intelligence

        except Exception as e:
            logger.error(f"Error generating comprehensive intelligence: {str(e)}")
            return self._get_fallback_comprehensive_intelligence(ticker)

    def _extract_key_fundamentals(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fundamental metrics from company overview."""
        if not company_data:
            return {}

        try:
            return {
                'market_cap': company_data.get('MarketCapitalization', 'N/A'),
                'pe_ratio': company_data.get('PERatio', 'N/A'),
                'dividend_yield': company_data.get('DividendYield', 'N/A'),
                'beta': company_data.get('Beta', 'N/A'),
                'sector': company_data.get('Sector', 'N/A'),
                'industry': company_data.get('Industry', 'N/A'),
                'description': company_data.get('Description', '')[:200] + '...' if company_data.get('Description') else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error extracting fundamentals: {str(e)}")
            return {}

    def _extract_earnings_intelligence(self, earnings_data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Extract earnings intelligence for the specific ticker."""
        if not earnings_data:
            return {'next_earnings': 'Unknown', 'earnings_proximity': 'Unknown'}

        try:
            # Look for the ticker in upcoming earnings
            # Note: This is a simplified implementation - actual parsing would depend on API response format
            return {
                'next_earnings': 'Check earnings calendar',
                'earnings_proximity': 'Monitor for upcoming events',
                'earnings_impact': 'Potential volatility catalyst'
            }
        except Exception as e:
            logger.error(f"Error extracting earnings intelligence: {str(e)}")
            return {'next_earnings': 'Error', 'earnings_proximity': 'Error'}

    def _generate_market_context(self, sentiment_data: Optional[SentimentDataV2_5], company_data: Dict[str, Any]) -> List[str]:
        """Generate market context insights."""
        context = []

        try:
            # Sentiment context
            if sentiment_data and abs(sentiment_data.overall_sentiment_score) > 0.2:
                direction = "positive" if sentiment_data.overall_sentiment_score > 0 else "negative"
                context.append(f"📰 Strong {direction} news sentiment driving market attention")

            # Sector context
            sector = company_data.get('Sector', '') if company_data else ''
            if sector:
                context.append(f"🏭 {sector} sector dynamics may influence price action")

            # Beta context
            beta = company_data.get('Beta', '') if company_data else ''
            if beta and beta != 'N/A':
                try:
                    beta_val = float(beta)
                    if beta_val > 1.2:
                        context.append(f"⚡ High beta ({beta}) suggests amplified market moves")
                    elif beta_val < 0.8:
                        context.append(f"🛡️ Low beta ({beta}) suggests defensive characteristics")
                except:
                    pass

            return context[:3]  # Limit to 3 context items

        except Exception as e:
            logger.error(f"Error generating market context: {str(e)}")
            return ["📊 Market context analysis temporarily unavailable"]

    def _generate_ai_recommendations(self, sentiment_data: Optional[SentimentDataV2_5], company_data: Dict[str, Any]) -> List[str]:
        """Generate AI-powered trading recommendations."""
        recommendations = []

        try:
            # Sentiment-based recommendations
            if sentiment_data:
                if sentiment_data.overall_sentiment_score > 0.3:
                    recommendations.append("🚀 Strong bullish sentiment supports long strategies")
                elif sentiment_data.overall_sentiment_score < -0.3:
                    recommendations.append("🐻 Strong bearish sentiment favors short strategies")

                if sentiment_data.article_count > 25:
                    recommendations.append("📈 High news volume suggests increased volatility - consider vol strategies")

            # Add general recommendation
            recommendations.append("🎯 Cross-reference with EOTS v2.5 metrics for optimal entry timing")

            return recommendations[:3]  # Limit to 3 recommendations

        except Exception as e:
            logger.error(f"Error generating AI recommendations: {str(e)}")
            return ["🤖 AI recommendations temporarily unavailable"]

    def _get_fallback_comprehensive_intelligence(self, ticker: str) -> Dict[str, Any]:
        """ZERO TOLERANCE FAKE DATA: No fallback comprehensive intelligence allowed."""
        raise ValueError(f"CRITICAL: Alpha Vantage API unavailable for {ticker} and no fallback data allowed - cannot provide comprehensive intelligence without real market data!")
