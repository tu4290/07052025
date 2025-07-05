# data_management/historical_data_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED

import logging
from typing import Dict, Any, Optional, List
from datetime import date, timedelta
import pandas as pd
import builtins

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_storage_v2_5 import HistoricalDataStorageV2_5

logger = logging.getLogger(__name__)

def _get_db_manager() -> DatabaseManagerV2_5:
    db_manager = getattr(builtins, 'db_manager', None)
    if db_manager is None:
        raise RuntimeError("Global db_manager is not initialized. Make sure the runner script sets it up.")
    return db_manager

class HistoricalDataManagerV2_5:
    """
    Manages the retrieval and storage of historical market data and metrics.
    Provides methods to fetch OHLCV and custom metrics for rolling analytics,
    and to store daily EOTS metrics for archival.
    """
    def __init__(self, config_manager: ConfigManagerV2_5, db_manager: Optional[DatabaseManagerV2_5] = None):
        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.db_manager = db_manager or _get_db_manager()
        if not isinstance(self.db_manager, DatabaseManagerV2_5):
            self.logger.critical("FATAL: Invalid db_manager object provided.")
            raise TypeError("db_manager must be an instance of DatabaseManagerV2_5")
        self.logger.info("HistoricalDataManagerV2_5 initialized with live database access.")

        # Instantiate HistoricalDataStorageV2_5 for storage operations
        self.data_storage = HistoricalDataStorageV2_5(self.db_manager)

    def store_regime_snapshot(self, symbol: str, regime_data: dict) -> bool:
        return self.data_storage.store_daily_regime_snapshot(symbol, regime_data)

    def get_historical_metric(self, symbol: str, metric_name: str, lookback_days: int) -> Optional[pd.Series]:
        """
        Fetches a historical metric series for a symbol over the specified lookback window.
        Args:
            symbol (str): The ticker symbol.
            metric_name (str): The metric/column name to fetch.
            lookback_days (int): Number of days to look back.
        Returns:
            Optional[pd.Series]: Series indexed by date, or None if not found.
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)
            series = self.db_manager.query_metric(
                table_name="daily_eots_metrics",
                metric_name=metric_name,
                start_date=start_date,
                end_date=end_date
            )
            if series is not None:
                self.logger.info(f"Fetched {len(series)} rows for {symbol} {metric_name} ({start_date} to {end_date})")
            else:
                self.logger.warning(f"No data found for {symbol} {metric_name} ({start_date} to {end_date})")
            return series
        except Exception as e:
            self.logger.error(f"Error fetching historical metric for {symbol} {metric_name}: {e}")
            return None

    def get_historical_ohlcv(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data for a symbol over the specified lookback window.
        Args:
            symbol (str): The ticker symbol.
            lookback_days (int): Number of days to look back.
        Returns:
            Optional[pd.DataFrame]: DataFrame of OHLCV data, or None if not found.
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)
            self.logger.info(f"🔍 Querying OHLCV data for {symbol} from {start_date} to {end_date} (checking metrics schema first, then public)")

            df = self.db_manager.query_ohlcv(
                table_name="daily_ohlcv",
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                # Filter for the specific symbol
                symbol_df = df[df['symbol'] == symbol]
                if not symbol_df.empty:
                    self.logger.info(f"✅ Successfully fetched {len(symbol_df)} OHLCV rows for {symbol} ({start_date} to {end_date})")
                    return symbol_df
                else:
                    self.logger.warning(f"⚠️ OHLCV table found but no data for symbol {symbol} ({start_date} to {end_date})")
                    return None
            else:
                self.logger.warning(f"❌ No OHLCV data found in any schema for date range ({start_date} to {end_date})")
                return None

        except Exception as e:
            self.logger.error(f"💥 Error fetching historical OHLCV for {symbol}: {e}")
            return None

    def store_daily_eots_metrics(self, symbol: str, metric_date: date, metrics_data: Dict[str, Any]) -> None:
        """
        Stores daily EOTS metrics for a symbol/date into the database.
        Args:
            symbol (str): The ticker symbol.
            metric_date (date): The date for the metrics.
            metrics_data (Dict[str, Any]): The metrics to store (column:value pairs).
        """
        try:
            record = {"symbol": symbol, "date": metric_date}
            record.update(metrics_data)
            self.db_manager.insert_record("daily_eots_metrics", record)
            self.logger.info(f"Stored daily EOTS metrics for {symbol} on {metric_date}.")
        except Exception as e:
            self.logger.error(f"Error storing daily EOTS metrics for {symbol} on {metric_date}: {e}")

    def get_recent_data(self, symbol: str, metrics: Optional[List[str]] = None, minutes_back: int = 30, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetches recent data for a symbol over the specified lookback window.
        This is a convenience method that combines OHLCV and metrics data.

        Args:
            symbol (str): The ticker symbol
            metrics (Optional[List[str]]): List of specific metrics to fetch. If None, fetches default metrics.
            minutes_back (int): Number of minutes to look back for intraday data
            lookback_days (int): Number of days to look back for historical data

        Returns:
            Optional[pd.DataFrame]: Combined DataFrame with recent data, or None if not found
        """
        try:
            # Get OHLCV data
            ohlcv_df = self.get_historical_ohlcv(symbol, lookback_days)
            if ohlcv_df is None:
                self.logger.warning(f"No OHLCV data found for {symbol}")
                return None

            # Get metrics data
            metrics_df = pd.DataFrame()
            default_metrics = ['vri_3_composite', 'flow_intensity_score', 'regime_stability_score']
            metrics_to_fetch = metrics if metrics is not None else default_metrics

            for metric in metrics_to_fetch:
                series = self.get_historical_metric(symbol, metric, lookback_days)
                if series is not None:
                    metrics_df[metric] = series

            # Combine data if metrics exist
            if not metrics_df.empty:
                combined_df = pd.merge(
                    ohlcv_df,
                    metrics_df,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                return combined_df

            return ohlcv_df

        except Exception as e:
            self.logger.error(f"Error getting recent data for {symbol}: {e}")
            return None