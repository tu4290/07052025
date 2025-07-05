# data_management/tradier_data_fetcher_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL V2.5.3 IMPLEMENTATION (UNABRIDGED)

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import ValidationError
from dotenv import load_dotenv

from utils.config_manager_v2_5 import ConfigManagerV2_5
from utils.async_resilience_v2_5 import async_retry

# The schema is not strictly needed here as this fetcher now returns a dict,
# but it's good practice to keep the import for context.
from data_models import RawUnderlyingDataCombinedV2_5, RawOptionsContractV2_5
from data_models.historical_data_models import OHLCVItem, HistoricalDataResponse # Updated import

load_dotenv()  # This will load variables from .env into os.environ

logger = logging.getLogger(__name__)

class TradierDataFetcherV2_5:
    """
    Asynchronous data fetcher for the Tradier API, precisely aligned
    with the unabridged canonical parameter lists.
    """
    def __init__(self, config_manager: ConfigManagerV2_5):

        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        # Use Pydantic v2 model access instead of get_setting method
        try:
            data_fetcher_settings = self.config_manager.config.data_fetcher_settings
            self.api_key = data_fetcher_settings.tradier_api_key
            self.account_id = data_fetcher_settings.tradier_account_id
            self.timeout = getattr(data_fetcher_settings, 'timeout_seconds', 30.0)
        except AttributeError as e:
            self.logger.error(f"❌ Failed to access data fetcher settings: {e}")
            self.api_key = None
            self.account_id = None
            self.timeout = 30.0
        if not self.api_key or self.api_key.startswith('${'):
            self.logger.warning("Tradier API key not found or is an environment variable placeholder. API calls will fail.")
            self.api_key = None
        else:
            self.base_url = "https://api.tradier.com/v1" # Using production API
            self.headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger.info("TradierDataFetcherV2_5 initialized.")



    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None

    def _parse_chain_response(self, json_payload: Dict[str, Any]) -> List[RawOptionsContractV2_5]:
        """Parses the options/chain response into a list of Pydantic models."""
        parsed_contracts = []
        options = json_payload.get('options', {}).get('option', [])
        
        if not isinstance(options, list):
            self.logger.error(f"Expected 'options.option' in chain response to be a list, got {type(options)}.")
            return []

        for contract_data in options:
            try:
                # Convert Tradier's snake_case to our camelCase format
                contract_dict = {
                    'contract_symbol': contract_data.get('symbol'),
                    'strike': float(contract_data.get('strike', 0)),
                    'opt_kind': contract_data.get('option_type', '').lower(),
                    'expiration_date': datetime.strptime(contract_data.get('expiration_date', ''), '%Y-%m-%d').date(),
                    'price': float(contract_data.get('last', 0)),
                    'volatility': float(contract_data.get('greeks', {}).get('smv_vol', 0)),
                    'multiplier': 100,  # Standard for US options
                    'oi': int(contract_data.get('open_interest', 0)),
                    'delta': float(contract_data.get('greeks', {}).get('delta', 0)),
                    'gamma': float(contract_data.get('greeks', {}).get('gamma', 0)),
                    'theta': float(contract_data.get('greeks', {}).get('theta', 0)),
                    'vega': float(contract_data.get('greeks', {}).get('vega', 0))
                }

                validated_contract = RawOptionsContractV2_5(**contract_dict)
                parsed_contracts.append(validated_contract)
            except (ValidationError, ValueError) as e:
                self.logger.warning(f"Skipping contract due to validation error: {e}. Data: {contract_data}")
        
        self.logger.info(f"Successfully parsed {len(parsed_contracts)} contracts from options/chain.")
        return parsed_contracts

    def _parse_underlying_response(self, json_payload: Dict[str, Any]) -> Optional[RawUnderlyingDataCombinedV2_5]:
        """Parses the markets/quotes response into a single Pydantic model."""
        quotes = json_payload.get('quotes', {}).get('quote', {})
        if not quotes:
            self.logger.error("Quotes response is missing or empty.")
            return None
            
        try:
            und_dict = {
                'symbol': quotes.get('symbol'),
                'timestamp': datetime.now(),  # Tradier doesn't provide timestamp in quotes
                'price': float(quotes.get('last', 0)),
                'volatility': float(quotes.get('volatility', 0)),
                'day_volume': int(quotes.get('volume', 0))
            }
            
            return RawUnderlyingDataCombinedV2_5(**und_dict)
        except (ValidationError, ValueError) as e:
            self.logger.error(f"Pydantic validation failed for Tradier underlying data: {e}", exc_info=True)
            return None

    @async_retry(max_attempts=3, backoff_factor=0.7)
    async def _fetch_raw_data(self, session: aiohttp.ClientSession, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Private helper to perform a single resilient GET request."""
        if not session:
            raise RuntimeError("No active session available")
        self.logger.debug(f"Fetching from {self.base_url}/{endpoint}...")
        try:
            async with session.get(f"{self.base_url}/{endpoint}", params=params, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
        except asyncio.TimeoutError:
            self.logger.error(f"Request to {endpoint} timed out after {self.timeout} seconds")
            raise
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error for {endpoint}: {e}")
            raise

    async def fetch_underlying_quote(self, symbol: str) -> Optional[RawUnderlyingDataCombinedV2_5]:
        """Fetches the underlying quote data for a given symbol."""
        
        # Make API call
        if not self.session:
            raise RuntimeError("No active session available. Use async with context manager.")
            
        try:
            underlying_data = await self._fetch_raw_data(
                self.session,
                "markets/quotes",
                {'symbols': symbol}
            )
            return self._parse_underlying_response(underlying_data)
        except Exception as e:
            self.logger.error(f"Error fetching underlying quote for {symbol}: {e}", exc_info=True)
            return None

    async def fetch_raw_quote_data(self, symbol: str) -> Optional[Dict[str, Any]]: # Return type kept as Dict for now, pending further investigation
        """Fetches raw quote data for a given symbol (used by market regime engine for VIX data)."""
        
        # Make API call
        if not self.session:
            raise RuntimeError("No active session available. Use async with context manager.")
            
        try:
            underlying_data = await self._fetch_raw_data(
                self.session,
                "markets/quotes",
                {'symbols': symbol}
            )
            return underlying_data
        except Exception as e:
            self.logger.error(f"Error fetching raw quote data for {symbol}: {e}", exc_info=True)
            return None

    async def fetch_historical_data(self, symbol: str, days: int = 30) -> Optional[HistoricalDataResponse]: # Changed return type
        """Fetches historical OHLCV data for technical analysis."""
        if not self.session:
            raise RuntimeError("No active session available. Use async with context manager.")
            
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'symbol': symbol,
                'interval': 'daily',
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            historical_data = await self._fetch_raw_data(
                self.session,
                "markets/history",
                params
            )
            
            # Parse historical data into standard OHLCV format
            if 'history' in historical_data and 'day' in historical_data['history']:
                days_data = historical_data['history']['day']
                if not isinstance(days_data, list):
                    days_data = [days_data]
                
                # Convert to standard format with proper column names
                ohlcv_data = []
                for day in days_data:
                    ohlcv_data.append({
                        'date': day.get('date'),
                        'open': float(day.get('open', 0)),
                        'high': float(day.get('high', 0)),
                        'low': float(day.get('low', 0)),
                        'close': float(day.get('close', 0)),
                        'volume': int(day.get('volume', 0))
                    })
                
                # Validate and create Pydantic model
                try:
                    ohlcv_items = [OHLCVItem(**item) for item in ohlcv_data]
                    return HistoricalDataResponse(
                        symbol=symbol,
                        data=ohlcv_items,
                        columns=['date', 'open', 'high', 'low', 'close', 'volume'] # Kept for now
                    )
                except ValidationError as ve:
                    self.logger.error(f"Pydantic validation error for historical data {symbol}: {ve}")
                    return None
            else:
                self.logger.warning(f"No historical data found for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
            return None

    async def fetch_chain_and_underlying(self, symbol: str) -> Tuple[Optional[List[RawOptionsContractV2_5]], Optional[RawUnderlyingDataCombinedV2_5]]:
        """Concurrently fetches data from options/chain and markets/quotes."""
        
        # Make API calls
        if not self.session:
            raise RuntimeError("No active session available. Use async with context manager.")
            
        # Get expiration dates for the chain request
        expirations = await self._fetch_raw_data(
            self.session,
            "markets/options/expirations",
            {'symbol': symbol, 'includeAllRoots': 'true'}
        )
        
        if not expirations or 'expirations' not in expirations:
            self.logger.error("Failed to fetch expiration dates")
            return None, None
            
        expiration_dates = expirations['expirations']['date']
        if not isinstance(expiration_dates, list):
            expiration_dates = [expiration_dates]
            
        # Use the next 4 expiration dates
        expiration_dates = expiration_dates[:4]
        
        chain_params = {
            'symbol': symbol,
            'expiration': ','.join(expiration_dates),
            'greeks': 'true'
        }
        
        und_params = {'symbols': symbol}

        chain_task = self._fetch_raw_data(self.session, "markets/options/chains", chain_params)
        und_task = self._fetch_raw_data(self.session, "markets/quotes", und_params)
        
        results = await asyncio.gather(chain_task, und_task, return_exceptions=True)

        chain_result, und_result = results
        
        parsed_contracts = None
        if isinstance(chain_result, dict):
            parsed_contracts = self._parse_chain_response(chain_result)
        elif isinstance(chain_result, Exception):
            self.logger.error(f"Failed to fetch Tradier chain data: {chain_result}", exc_info=True)

        parsed_underlying = None
        if isinstance(und_result, dict):
            parsed_underlying = self._parse_underlying_response(und_result)
        elif isinstance(und_result, Exception):
            self.logger.error(f"Failed to fetch Tradier underlying data: {und_result}", exc_info=True)
            
        return parsed_contracts, parsed_underlying

    # Note: Other methods like fetch_options_chain from the old file are removed,
    # as that responsibility now lies with the ConvexValue fetcher. This module
    # now has a single, clear responsibility.