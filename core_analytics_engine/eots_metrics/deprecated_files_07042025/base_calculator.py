import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set, TypeVar
from datetime import datetime, date
from pydantic import BaseModel, Field
from collections import deque

# Import necessary schemas from data_models
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5, CacheLevel # Import EnhancedCacheManager

logger = logging.getLogger(__name__)
EPSILON = 1e-9

T = TypeVar('T')

# Pydantic Models for internal state and cache (moved from metrics_calculator_v2_5.py)
class MetricCalculationState(BaseModel):
    """Pydantic model for metric calculation state tracking"""
    current_symbol: Optional[str] = Field(None, description="Current symbol being processed")
    calculation_timestamp: Optional[datetime] = Field(None, description="Timestamp of last calculation")
    metrics_completed: Set[str] = Field(default_factory=set, description="Set of completed metrics")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")

    def update_state(self, **kwargs):
        """Update state attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_state(self, key: str) -> Any:
        """Get state attribute value"""
        return getattr(self, key)

    def get(self, key: str, default: Optional[T] = None) -> Union[Any, T]:
        """Dictionary-like get with default value"""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to state attributes"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Dictionary-like setting of cache attributes"""
        if hasattr(self, key):
            setattr(self, key, value)

class MetricCache(BaseModel):
    """Pydantic model for individual metric cache"""
    data: Dict[str, Any] = Field(default_factory=dict)

class MetricCacheConfig(BaseModel):
    """Pydantic model for metric cache configuration"""
    vapi_fa: MetricCache = Field(default_factory=MetricCache)
    dwfd: MetricCache = Field(default_factory=MetricCache)
    tw_laf: MetricCache = Field(default_factory=MetricCache)
    a_dag: MetricCache = Field(default_factory=MetricCache)
    e_sdag: MetricCache = Field(default_factory=MetricCache)
    d_tdpi: MetricCache = Field(default_factory=MetricCache)
    vri_2_0: MetricCache = Field(default_factory=MetricCache)
    heatmap: MetricCache = Field(default_factory=MetricCache)
    normalization: MetricCache = Field(default_factory=MetricCache)

    def get_cache(self, metric_name: str) -> Dict[str, Any]:
        """Get cache for specific metric"""
        if hasattr(self, metric_name):
            return getattr(self, metric_name).data
        return {}

    def set_cache(self, metric_name: str, data: Dict[str, Any]):
        """Set cache for specific metric"""
        if hasattr(self, metric_name):
            cache = getattr(self, metric_name)
            cache.data = data

    def has_metric(self, metric_name: str) -> bool:
        """Check if metric exists"""
        return hasattr(self, metric_name)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Dictionary-like access to cache data"""
        if hasattr(self, key):
            return getattr(self, key).data
        return {}

    def __setitem__(self, key: str, value: Dict[str, Any]):
        """Dictionary-like setting of cache data"""
        if hasattr(self, key):
            cache = getattr(self, key)
            cache.data = value

    def __contains__(self, key: str) -> bool:
        """Support for 'in' operator"""
        return hasattr(self, key)


class BaseCalculator:
    """
    Base class for metric calculations, containing common utilities,
    data conversion, caching, and validation methods.
    """
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5):
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager
        self.enhanced_cache = enhanced_cache_manager # Store the enhanced cache manager instance
        self._metric_caches = MetricCacheConfig()
        self._calculation_state = MetricCalculationState()
        self.current_trading_date = datetime.now().date()

    def _convert_numpy_value(self, val: Any) -> Any:
        """Convert numpy types to Python types"""
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val.tolist()
        elif isinstance(val, pd.Series):
            return self._convert_numpy_value(val.to_numpy())
        elif isinstance(val, pd.DataFrame):
            return val.to_dict('records')
        return val

    def _convert_dataframe_to_pydantic_models(self, df: Optional[pd.DataFrame], model_type: BaseModel) -> List[BaseModel]:
        """
        Convert DataFrame to list of Pydantic models of a specified type.
        Generalized from _convert_dataframe_to_strike_metrics and _convert_dataframe_to_contract_metrics.
        """
        if df is None or df.empty:
            return []
        
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.to_numpy()
                record[col] = self._convert_numpy_value(val)
            try:
                records.append(model_type(**record))
            except Exception as e:
                self.logger.error(f"Failed to create {model_type.__name__} from record: {e}. Record: {record}")
                continue
        return records

    def _serialize_dataframe_for_redis(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to Redis-serializable format by handling timestamps and other non-JSON types.
        """
        if df is None or len(df) == 0:
            return []

        records = df.to_dict('records')
        serializable_records = []
        for record in records:
            serializable_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    serializable_record[key] = None
                elif isinstance(value, (pd.Timestamp, datetime, date)):
                    serializable_record[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_record[key] = float(value) if not np.isnan(value) else None
                elif isinstance(value, np.ndarray):
                    serializable_record[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    serializable_record[key] = value
                else:
                    serializable_record[key] = str(value)
            serializable_records.append(serializable_record)
        return serializable_records

    def _serialize_underlying_data_for_redis(self, und_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert underlying data dictionary to Redis-serializable format.
        """
        serializable_data = {}
        for key, value in und_data.items():
            if pd.isna(value) if hasattr(pd, 'isna') and not isinstance(value, (list, dict)) else False:
                serializable_data[key] = None
            elif isinstance(value, (pd.Timestamp, datetime, date)):
                serializable_data[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
            elif isinstance(value, (np.integer, np.floating)):
                serializable_data[key] = float(value) if not np.isnan(value) else None
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, list):
                serializable_list = []
                for item in value:
                    if isinstance(item, (pd.Timestamp, datetime, date)):
                        serializable_list.append(item.isoformat() if hasattr(item, 'isoformat') else str(item))
                    elif isinstance(item, (np.integer, np.floating)):
                        serializable_list.append(float(item) if not np.isnan(item) else None)
                    elif isinstance(item, (int, float, str, bool, type(None))):
                        serializable_list.append(item)
                    else:
                        serializable_list.append(str(item))
                serializable_data[key] = serializable_list
            elif isinstance(value, dict):
                serializable_data[key] = self._serialize_underlying_data_for_redis(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)
        return serializable_data

    def _get_isolated_cache(self, metric_name: str, symbol: str, cache_type: str = 'history') -> Dict[str, Any]:
        """Get isolated cache for a specific metric and symbol."""
        cache_key = f"{metric_name}_{symbol}_{cache_type}"
        if cache_key not in self._metric_caches:
            self._metric_caches[cache_key] = {}
        return self._metric_caches[cache_key]

    def _store_metric_data(self, metric_name: str, symbol: str, data: Any, cache_type: str = 'history') -> None:
        """Store metric data in isolated cache."""
        cache = self._get_isolated_cache(metric_name, symbol, cache_type)
        cache_key = f"{metric_name}_{cache_type}"
        cache[cache_key] = data

    def _get_metric_data(self, metric_name: str, symbol: str, cache_type: str = 'history') -> List[Any]:
        """Retrieve metric data from isolated cache."""
        cache = self._get_isolated_cache(metric_name, symbol, cache_type)
        cache_key = f"{metric_name}_{cache_type}"
        return cache.get(cache_key, [])

    def _validate_metric_bounds(self, metric_name: str, value: float, bounds: Tuple[float, float] = (-10.0, 10.0)) -> bool:
        """Validate metric values are within reasonable bounds to prevent interference."""
        try:
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                self.logger.warning(f"Invalid {metric_name} value: {value}")
                return False
            
            if value < bounds[0] or value > bounds[1]:
                self.logger.warning(f"{metric_name} value {value} outside bounds {bounds}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {metric_name}: {e}")
            return False
    
    def _check_metric_dependencies(self, metric_group: str) -> bool:
        """Check if metric dependencies are satisfied before calculation."""
        try:
            # This dependency graph will be managed by the orchestrator, but keeping the method for now
            # as it might be used by individual metric calculators for internal checks.
            # The actual _metric_dependencies dict will be in the orchestrator.
            return True # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking dependencies for {metric_group}: {e}")
            return False
    
    def _mark_metric_completed(self, metric_group: str) -> None:
        """Mark metric group as completed."""
        # This state will be managed by the orchestrator.
        self.logger.debug(f"Metric group {metric_group} completed (placeholder)")
    
    def _get_metric_config(self, metric_group: str, config_key: str, default_value: Any = None) -> Any:
        """Get configuration value for a specific metric group and key."""
        try:
            # This will be refactored to use a centralized config access pattern
            # passed from the main orchestrator. For now, direct config_manager access.
            if metric_group == 'enhanced_flow':
                settings = self.config_manager.get_setting("enhanced_flow_metric_settings", default={})
                if hasattr(settings, config_key):
                    return getattr(settings, config_key)
                return default_value
            elif metric_group == 'adaptive':
                settings = self.config_manager.get_setting("adaptive_metric_parameters", default={})
                if hasattr(settings, config_key):
                    return getattr(settings, config_key)
                return default_value
            elif metric_group == 'heatmap_generation_settings': # Added for heatmap config access
                settings = self.config_manager.get_setting("heatmap_generation_settings", default={})
                if hasattr(settings, config_key):
                    return getattr(settings, config_key)
                return default_value
            else:
                return default_value
        except Exception as e:
            self.logger.warning(f"Error getting config {config_key} for {metric_group}: {e}")
            return default_value
    
    def _validate_aggregates(self, aggregates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize aggregate metrics before applying.
        """
        validated = {}
        
        for key, value in aggregates.items():
            try:
                if pd.isna(value) or np.isinf(value):
                    validated[key] = 0.0
                    self.logger.warning(f"Invalid aggregate value for {key}, setting to 0.0")
                elif isinstance(value, (int, float)):
                    if 'ratio' in key.lower() or 'factor' in key.lower():
                        validated[key] = max(-10.0, min(10.0, float(value)))
                    elif 'concentration' in key.lower() or 'index' in key.lower():
                        validated[key] = max(0.0, min(1.0, float(value)))
                    else:
                        validated[key] = float(value)
                else:
                    validated[key] = value
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error validating aggregate {key}: {e}, setting to 0.0")
                validated[key] = 0.0
        
        return validated
    
    def _perform_final_validation(self, df_strike: Optional[pd.DataFrame], und_data: Dict[str, Any]) -> None:
        """
        Perform final validation on calculated metrics.
        """
        try:
            symbol = und_data.get('symbol', 'UNKNOWN')
            is_futures = self._is_futures_symbol(symbol) # Assuming this is a helper method

            if df_strike is not None and len(df_strike) > 0:
                numeric_cols = df_strike.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df_strike[col].isna().sum() > 0:
                        if is_futures:
                            self.logger.debug(f"Found NaN values in {col} for futures symbol {symbol}, filling with 0")
                        else:
                            self.logger.warning(f"Found NaN values in {col}, filling with 0")
                        df_strike[col] = df_strike[col].fillna(0)

            for key, value in und_data.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    if is_futures:
                        self.logger.debug(f"Invalid value for {key} in futures symbol {symbol}: {value}, setting to 0")
                    else:
                        self.logger.warning(f"Invalid value for {key}: {value}, setting to 0")
                    und_data[key] = 0.0

        except Exception as e:
            self.logger.error(f"Error in final validation: {e}", exc_info=True)

    def _is_futures_symbol(self, symbol: str) -> bool:
        """
        Determines if a symbol is a futures symbol (placeholder).
        # Placeholder: In a real system, this would check a list of known futures symbols
        # or use a more sophisticated method.
        """
        return symbol.upper().startswith('ES=F') or symbol.upper().startswith('NQ=F') # Example for S&P 500 E-mini futures

    def _add_to_intraday_cache(self, symbol: str, metric_name: str, value: float, max_size: int = 200) -> deque:
        """
        Adds a value to an intraday cache for a specific metric and symbol.
        """
        cache_key = f"intraday_cache_{symbol}_{metric_name}"
        if cache_key not in self._metric_caches.normalization.data:
            self._metric_caches.normalization.data[cache_key] = deque(maxlen=max_size)
        
        cache = self._metric_caches.normalization.data[cache_key]
        
        # If cache is empty (new ticker), seed it with baseline values
        if not cache: # Check if deque is empty
            try:
                seeded_values = self._seed_new_ticker_cache(symbol, metric_name, value)
                cache.extend(seeded_values)
            except ValueError as ve:
                self.logger.warning(f"Could not seed cache for {symbol} {metric_name}: {ve}. Starting with current value.")
                cache.append(value)
        else:
            cache.append(value)
        
        # The deque automatically handles max_size, no need for manual truncation
        
        return cache

    def _seed_new_ticker_cache(self, symbol: str, metric_name: str, current_value: float) -> List[float]:
        """
        Fail-fast when no historical cache data is available for new ticker.
        Attempts to load baseline values from existing cache files for similar metrics/symbols.
        """
        baseline_values = []
        
        # Use enhanced_cache to load historical data
        historical_data_from_cache = self.enhanced_cache.get(symbol=symbol, metric_name=metric_name, tags=["historical", "intraday_seed"])
        if historical_data_from_cache and len(historical_data_from_cache) >= 10:
            baseline_values = historical_data_from_cache[-10:]
            self.logger.debug(f"Seeded {symbol} {metric_name} cache with {len(baseline_values)} values from enhanced cache")
        
        if not baseline_values:
            raise ValueError(f"No historical cache data available for {symbol} {metric_name}. Cannot generate baseline values without real market data.")
        
        baseline_values.append(current_value)
        
        # Save the seeded cache (this will use the enhanced cache system)
        self._save_intraday_cache(symbol, metric_name, baseline_values)
        
        return baseline_values

    def _calculate_percentile_gauge_value(self, data_cache: deque, current_value: float) -> float:
        """
        Calculates a percentile-based gauge value (0-1 scale) for a given value
        relative to its historical cache.
        """
        if not data_cache or len(data_cache) < 5: # Need at least 5 data points for meaningful percentile
            return 0.0 # Cannot calculate meaningful percentile

        data = np.array(list(data_cache))
        
        # Calculate percentiles
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)

        if current_value >= p90:
            return 1.0
        elif current_value <= p10:
            return -1.0
        elif current_value > p50:
            return (current_value - p50) / (p90 - p50 + EPSILON) * 0.5
        else: # current_value <= p50
            return (current_value - p50) / (p50 - p10 + EPSILON) * 0.5

    def _normalize_flow(self, flow_series: Union[pd.Series, np.ndarray], metric_name: str, symbol: str) -> Union[pd.Series, np.ndarray]:
        """
        Normalizes a flow series using Z-score based on its intraday cache.
        If insufficient history, returns the raw series.
        """
        if isinstance(flow_series, pd.Series):
            flow_array = flow_series.to_numpy()
        else:
            flow_array = np.asarray(flow_series)

        if flow_array.size == 0:
            return np.array([0.0]) # Return a single 0.0 if empty

        # Use a single value for normalization if the series is constant
        if np.all(flow_array == flow_array[0]):
            return np.full_like(flow_array, 0.0) # Already normalized if constant

        # Add all values from the current series to the cache
        for val in flow_array:
            self._add_to_intraday_cache(symbol, metric_name, float(val))
        
        cache_key = f"intraday_cache_{symbol}_{metric_name}"
        cache = self._metric_caches.normalization.data.get(cache_key, deque())

        if len(cache) < 10: # Need sufficient history for Z-score
            return flow_series # Return original if not enough history

        cache_array = np.array(list(cache))
        mean = np.mean(cache_array)
        std = np.std(cache_array)

        if std < EPSILON: # Avoid division by zero if std is very small
            return np.full_like(flow_array, 0.0)

        normalized_flow = (flow_array - mean) / std
        
        if isinstance(flow_series, pd.Series):
            return pd.Series(normalized_flow, index=flow_series.index)
        return normalized_flow

    def _get_dte_scaling_factor(self, dte_context: str) -> float:
        """Placeholder for DTE scaling factor based on context."""
        # This would be more sophisticated, potentially using config or a lookup table
        if dte_context == '0DTE':
            return 1.5
        elif dte_context == 'SHORT_DTE': # e.g., 1-5 DTE
            return 1.2
        else:
            return 1.0

    def sanitize_symbol(self, symbol: str) -> str:
        """
        Sanitize a ticker symbol for safe use in file paths and cache keys.
        Replaces '/' and ':' with '_'.
        """
        return symbol.replace('/', '_').replace(':', '_')

    def _load_intraday_cache(self, symbol: str, metric_name: str) -> List[float]:
        """
        Load intraday cache using enhanced cache system.
        """
        try:
            cached_data = self.enhanced_cache.get(symbol=symbol, metric_name=metric_name, tags=[f"intraday_{self.current_trading_date.isoformat()}", "metrics_calculator"])
            if cached_data is not None:
                return cached_data if isinstance(cached_data, list) else [cached_data]
        except Exception as e:
            self.logger.warning(f"Enhanced cache error for {symbol}_{metric_name}: {e}. Returning empty list.")

        return []

    def _save_intraday_cache(self, symbol: str, metric_name: str, values: List[float]) -> None:
        """
        Save intraday cache using enhanced cache system.
        """
        try:
            # Determine cache level based on data size (example logic)
            data_size_mb = len(str(values)) / (1024 * 1024)
            cache_level = CacheLevel.COMPRESSED if data_size_mb > 1.0 else CacheLevel.MEMORY

            success = self.enhanced_cache.put(
                symbol=symbol,
                metric_name=metric_name,
                data=values,
                cache_level=cache_level,
                tags=[f"intraday_{self.current_trading_date.isoformat()}", "metrics_calculator"]
            )

            if success:
                self.logger.debug(f"Saved {symbol}_{metric_name} to enhanced cache (level: {cache_level})")
            else:
                self.logger.warning(f"Failed to save {symbol}_{metric_name} to enhanced cache")

        except Exception as e:
            self.logger.error(f"Error saving intraday cache for {symbol}_{metric_name}: {e}")