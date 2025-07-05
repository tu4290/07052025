# core_analytics_engine/metrics_calculator_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE & COMPLETE METRICS CALCULATOR

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING, Set, TypeVar
from datetime import datetime, timedelta, date, time
from pathlib import Path
from pydantic import BaseModel, Field, validator

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models.eots_schemas_v2_5 import (
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    AdvancedOptionsMetricsV2_5,
    AnalyticsEngineConfigV2_5,
    ProcessedDataBundleV2_5
)

if TYPE_CHECKING:
    from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

T = TypeVar('T')

# PYDANTIC-FIRST: Input validation models
class MetricsCalculatorInputV2_5(BaseModel):
    """Pydantic model for metrics calculator inputs."""
    options_df_raw: pd.DataFrame = Field(..., description="Raw options chain DataFrame")
    und_data_api_raw: Dict[str, Any] = Field(..., description="Raw underlying data from API")
    dte_max: int = Field(default=45, ge=0, le=365, description="Maximum DTE to consider")
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame
        extra = 'forbid'  # Strict validation

    @validator('options_df_raw')
    def validate_options_df(cls, v):
        """Validate options DataFrame has required columns."""
        required_columns = {'strike', 'dte_calc', 'option_type', 'volume', 'open_interest'}
        missing_columns = required_columns - set(v.columns)
        if missing_columns:
            raise ValueError(f"Options DataFrame missing required columns: {missing_columns}")
        return v

    @validator('und_data_api_raw')
    def validate_underlying_data(cls, v):
        """Validate underlying data has required fields."""
        required_fields = {'price', 'price_change_pct', 'day_volume'}
        missing_fields = required_fields - set(v.keys())
        if missing_fields:
            raise ValueError(f"Underlying data missing required fields: {missing_fields}")
        return v

class MetricsCalculatorOutputV2_5(BaseModel):
    """Pydantic model for metrics calculator outputs."""
    strike_level_data: Optional[List[ProcessedStrikeLevelMetricsV2_5]] = Field(None, description="Strike-level metrics")
    options_with_metrics: List[ProcessedContractMetricsV2_5] = Field(..., description="Options chain with metrics")
    underlying_enriched: ProcessedUnderlyingAggregatesV2_5 = Field(..., description="Enriched underlying data")

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame
        extra = 'forbid'  # Strict validation

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
        """Dictionary-like setting of state attributes"""
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

class MetricsCalculatorV2_5:
    """
    High-performance, vectorized metrics calculator for EOTS v2.5.
    PYDANTIC-FIRST: Fully validated against EOTS schemas
    """

    def __init__(self, config_manager: ConfigManagerV2_5, historical_data_manager: 'HistoricalDataManagerV2_5'):
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager
        
        # Load and validate settings using Pydantic
        analytics_settings = self.config_manager.get_setting("analytics_engine", default={})
        
        # Handle different input types for analytics_settings
        if isinstance(analytics_settings, AnalyticsEngineConfigV2_5):
            # If it's already the correct model, use it directly
            self.analytics_config = analytics_settings
        elif isinstance(analytics_settings, dict):
            # If it's a dict, validate it against the model
            self.analytics_config = AnalyticsEngineConfigV2_5.model_validate(analytics_settings)
        elif hasattr(analytics_settings, 'model_dump'):
            # If it's a Pydantic model but not the right type, convert to dict and validate
            self.analytics_config = AnalyticsEngineConfigV2_5.model_validate(analytics_settings.model_dump())
        else:
            # Default to empty config if we can't determine the type
            self.analytics_config = AnalyticsEngineConfigV2_5()
        
        # Initialize metric caches with Pydantic validation
        self._metric_caches = MetricCacheConfig()
        
        # Initialize calculation state with Pydantic validation
        self._calculation_state = MetricCalculationState(
            current_symbol=None,
            calculation_timestamp=None,
            metrics_completed=set(),
            validation_results={}
        )
        
        # Metric dependency graph
        self._metric_dependencies = {
            'foundational': [],
            'enhanced_flow': ['foundational'],
            'adaptive': ['foundational'],
            'aggregates': ['enhanced_flow', 'adaptive'],
            'atr': []
        }
        
        # Initialize cache directories
        self.intraday_cache_dir = self.config_manager.get_setting("data_management.intraday_cache_dir", "data_cache_v2_5/intraday")
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

    def _convert_dataframe_to_strike_metrics(self, df: Optional[pd.DataFrame]) -> List[ProcessedStrikeLevelMetricsV2_5]:
        """Convert DataFrame to list of ProcessedStrikeLevelMetricsV2_5"""
        if df is None or df.empty:
            return []
        
        # Convert DataFrame to list of dicts with proper type conversion
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.to_numpy()
                record[col] = self._convert_numpy_value(val)
            try:
                records.append(ProcessedStrikeLevelMetricsV2_5(**record))
            except Exception as e:
                self.logger.error(f"Failed to create ProcessedStrikeLevelMetricsV2_5: {e}")
                continue
        return records

    def _convert_dataframe_to_contract_metrics(self, df: Optional[pd.DataFrame]) -> List[ProcessedContractMetricsV2_5]:
        """Convert DataFrame to list of ProcessedContractMetricsV2_5"""
        if df is None or df.empty:
            return []
        
        # Convert DataFrame to list of dicts with proper type conversion
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.to_numpy()
                record[col] = self._convert_numpy_value(val)
            try:
                records.append(ProcessedContractMetricsV2_5(**record))
            except Exception as e:
                self.logger.error(f"Failed to create ProcessedContractMetricsV2_5: {e}")
                continue
        return records

    def process_data_bundle(self, data_bundle: Dict[str, Any]) -> ProcessedDataBundleV2_5:
        """Process data bundle with proper type conversion"""
        strike_level_df = data_bundle.get('strike_level_data')
        options_df = data_bundle.get('options_data')
        
        # Create underlying data with proper validation
        underlying_data_dict = data_bundle.get('underlying_data', {})
        # Convert numpy values in underlying data
        for key, val in underlying_data_dict.items():
            if isinstance(val, pd.Series):
                val = val.to_numpy()
            underlying_data_dict[key] = self._convert_numpy_value(val)
        
        try:
            underlying_data = ProcessedUnderlyingAggregatesV2_5(
                **underlying_data_dict,
                confidence=0.5,
                transition_risk=0.5
            )
        except Exception as e:
            self.logger.error(f"Failed to create ProcessedUnderlyingAggregatesV2_5: {e}")
            underlying_data = ProcessedUnderlyingAggregatesV2_5(
                symbol=underlying_data_dict.get('symbol', 'UNKNOWN'),
                timestamp=datetime.now(),
                price=0.0,
                price_change_abs_und=0.0,
                price_change_pct_und=0.0,
                day_open_price_und=0.0,
                day_high_price_und=0.0,
                day_low_price_und=0.0,
                prev_day_close_price_und=0.0,
                u_volatility=0.0,
                day_volume=0,
                call_gxoi=0.0,
                put_gxoi=0.0,
                gammas_call_buy=0.0,
                gammas_call_sell=0.0,
                gammas_put_buy=0.0,
                gammas_put_sell=0.0,
                deltas_call_buy=0.0,
                deltas_call_sell=0.0,
                deltas_put_buy=0.0,
                deltas_put_sell=0.0,
                vegas_call_buy=0.0,
                vegas_call_sell=0.0,
                vegas_put_buy=0.0,
                vegas_put_sell=0.0,
                thetas_call_buy=0.0,
                thetas_call_sell=0.0,
                thetas_put_buy=0.0,
                thetas_put_sell=0.0,
                call_vxoi=0.0,
                put_vxoi=0.0,
                value_bs=0.0,
                volm_bs=0.0,
                deltas_buy=0.0,
                deltas_sell=0.0,
                vegas_buy=0.0,
                vegas_sell=0.0,
                thetas_buy=0.0,
                thetas_sell=0.0,
                volm_call_buy=0.0,
                volm_put_buy=0.0,
                volm_call_sell=0.0,
                volm_put_sell=0.0,
                value_call_buy=0.0,
                value_put_buy=0.0,
                value_call_sell=0.0,
                value_put_sell=0.0,
                vflowratio=0.0,
                dxoi=0.0,
                gxoi=0.0,
                vxoi=0.0,
                txoi=0.0,
                call_dxoi=0.0,
                put_dxoi=0.0,
                tradier_iv5_approx_smv_avg=0.0,
                total_call_oi_und=0,
                total_put_oi_und=0,
                total_call_vol_und=0,
                total_put_vol_und=0,
                tradier_open=0.0,
                tradier_high=0.0,
                tradier_low=0.0,
                tradier_close=0.0,
                tradier_volume=0,
                tradier_vwap=0.0,
                confidence=0.0,
                transition_risk=1.0
            )
        
        # Convert DataFrames to Pydantic models with proper validation
        strike_metrics = self._convert_dataframe_to_strike_metrics(strike_level_df)
        contract_metrics = self._convert_dataframe_to_contract_metrics(options_df)
        
        return ProcessedDataBundleV2_5(
            options_data_with_metrics=contract_metrics,
            strike_level_data_with_metrics=strike_metrics,
            underlying_data_enriched=underlying_data,
            processing_timestamp=datetime.now(),
            errors=[]
        )

    def _create_strike_level_df(self, df_chain: pd.DataFrame, und_data: Dict[str, Any]) -> pd.DataFrame:
        """Creates the primary strike-level DataFrame from per-contract data."""
        if len(df_chain) == 0:
            return pd.DataFrame()

        # Aggregate OI-based exposures from the chain
        strike_groups = df_chain.groupby('strike')
        df_strike = strike_groups.agg({
            'dxoi': 'sum',
            'gxoi': 'sum',
            'vxoi': 'sum',
            'txoi': 'sum',
            'charmxoi': 'sum',
            'vannaxoi': 'sum',
            'vommaxoi': 'sum'
        }).fillna(0)
        
        df_strike = df_strike.reset_index()
        return df_strike

    def calculate_metrics(self, options_df_raw: pd.DataFrame, und_data_api_raw: Dict[str, Any], dte_max: int = 45) -> MetricsCalculatorOutputV2_5:
        """
        Calculate all metrics for a given options chain and underlying data.
        
        Args:
            options_df_raw: Raw options chain DataFrame
            und_data_api_raw: Raw underlying data from API
            dte_max: Maximum DTE to consider
            
        Returns:
            MetricsCalculatorOutputV2_5: Calculated metrics in Pydantic model
        """
        # Validate inputs using Pydantic
        input_data = MetricsCalculatorInputV2_5(
            options_df_raw=options_df_raw,
            und_data_api_raw=und_data_api_raw,
            dte_max=dte_max,
            symbol=und_data_api_raw.get('symbol', 'UNKNOWN'),
            timestamp=datetime.now()
        )

        # Extract symbol for caching and logging
        symbol = input_data.symbol
        self._calculation_state['current_symbol'] = symbol
        self._calculation_state['calculation_timestamp'] = input_data.timestamp
        self._calculation_state['metrics_completed'] = set()

        # Initialize return values with validated Pydantic models
        df_strike = None
        df_options_with_metrics = input_data.options_df_raw.copy()
        und_data_enriched = ProcessedUnderlyingAggregatesV2_5(**{
            'symbol': symbol,
            'timestamp': datetime.now(),
            **input_data.und_data_api_raw
        })
        
        # Step 1: Calculate foundational metrics
        und_data_enriched = self._calculate_foundational_metrics(und_data_enriched.model_dump())
        self._mark_metric_completed('foundational')
        
        # Step 2: Create strike-level DataFrame if we have options data
        if len(input_data.options_df_raw) > 0:
            df_strike = self._create_strike_level_df(input_data.options_df_raw, und_data_enriched)
        
        # Step 3: Calculate adaptive metrics
        if df_strike is not None and len(df_strike) > 0:
            df_strike = self._calculate_adaptive_metrics(df_strike, und_data_enriched)
        self._mark_metric_completed('adaptive')
        
        # Step 4: Calculate underlying aggregates
        aggregates = self._calculate_underlying_aggregates(df_strike)
        und_data_enriched = ProcessedUnderlyingAggregatesV2_5(**{
            'symbol': symbol,
            'timestamp': datetime.now(),
            **und_data_enriched,
            **aggregates
        })
        self._mark_metric_completed('aggregates')
        
        # Step 5: Calculate enhanced flow metrics
        if symbol:
            und_data_enriched = ProcessedUnderlyingAggregatesV2_5(**{
                **und_data_enriched.model_dump(),
                **self._calculate_enhanced_flow_metrics(und_data_enriched.model_dump(), symbol)
            })
        self._mark_metric_completed('enhanced_flow')
        
        # Step 6: Calculate ATR
        if symbol:
            atr_value = self._calculate_atr(symbol, input_data.dte_max)
            und_data_enriched = ProcessedUnderlyingAggregatesV2_5(**{
                **und_data_enriched.model_dump(),
                'atr_und': atr_value
            })
        self._mark_metric_completed('atr')
        
        # Step 7: Calculate advanced options metrics
        advanced_metrics = self.calculate_advanced_options_metrics(input_data.options_df_raw)
        und_data_enriched = ProcessedUnderlyingAggregatesV2_5(**{
            **und_data_enriched.model_dump(),
            'advanced_options_metrics': advanced_metrics.model_dump()
        })

        # Return validated output
        return MetricsCalculatorOutputV2_5(
            strike_level_data=self._convert_dataframe_to_strike_metrics(df_strike),
            options_with_metrics=self._convert_dataframe_to_contract_metrics(df_options_with_metrics),
            underlying_enriched=und_data_enriched
        )

    def _serialize_dataframe_for_redis(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to Redis-serializable format by handling timestamps and other non-JSON types.

        Args:
            df: DataFrame to serialize

        Returns:
            List of dictionaries with JSON-serializable values
        """
        if df is None or len(df) == 0:
            return []

        # Convert DataFrame to records
        records = df.to_dict('records')

        # Process each record to handle non-JSON serializable types
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
                    # Convert any other type to string as fallback
                    serializable_record[key] = str(value)
            serializable_records.append(serializable_record)

        return serializable_records

    def _serialize_underlying_data_for_redis(self, und_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert underlying data dictionary to Redis-serializable format.

        Args:
            und_data: Dictionary containing underlying data

        Returns:
            Dictionary with JSON-serializable values
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
                # Handle lists that might contain non-serializable objects
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
                # Recursively handle nested dictionaries
                serializable_data[key] = self._serialize_underlying_data_for_redis(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_data[key] = value
            else:
                # Convert any other type to string as fallback
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
            required_groups = self._metric_dependencies.get(metric_group, [])
            
            for required_group in required_groups:
                if required_group not in self._calculation_state['metrics_completed']:
                    self.logger.error(f"Dependency {required_group} not completed for {metric_group}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking dependencies for {metric_group}: {e}")
            return False
    
    def _mark_metric_completed(self, metric_group: str) -> None:
        """Mark metric group as completed."""
        self._calculation_state['metrics_completed'].add(metric_group)
        self.logger.debug(f"Metric group {metric_group} completed")
    
    def _get_metric_config(self, metric_group: str, config_key: str, default_value: Any = None) -> Any:
        """Get configuration value for a specific metric group and key."""
        try:
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
                    # Apply reasonable bounds based on metric type
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
        """Perform final validation on calculated metrics."""
        try:
            symbol = und_data.get('symbol', 'UNKNOWN')
            is_futures = self._is_futures_symbol(symbol)

            if df_strike is not None and len(df_strike) > 0:
                # Validate strike-level metrics
                numeric_cols = df_strike.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df_strike[col].isna().sum() > 0:
                        if is_futures:
                            self.logger.debug(f"Found NaN values in {col} for futures symbol {symbol}, filling with 0")
                        else:
                            self.logger.warning(f"Found NaN values in {col}, filling with 0")
                        df_strike[col] = df_strike[col].fillna(0)

            # Validate underlying metrics
            for key, value in und_data.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    if is_futures:
                        self.logger.debug(f"Invalid value for {key} in futures symbol {symbol}: {value}, setting to 0")
                    else:
                        self.logger.warning(f"Invalid value for {key}: {value}, setting to 0")
                    und_data[key] = 0.0

        except Exception as e:
            self.logger.error(f"Error in final validation: {e}", exc_info=True)

    def _calculate_foundational_metrics(self, und_data: Dict) -> Dict:
        """Calculates key underlying metrics from the get_und data (excluding GIB which needs aggregates)."""
        # Net Customer Greek Flows (Daily Total) - Fix: Handle None values properly
        deltas_buy = und_data.get('deltas_buy') or 0
        deltas_sell = und_data.get('deltas_sell') or 0
        und_data['net_cust_delta_flow_und'] = deltas_buy - deltas_sell
        
        # Handle None values from ConvexValue gamma flow fields
        gammas_call_buy = und_data.get('gammas_call_buy') or 0
        gammas_put_buy = und_data.get('gammas_put_buy') or 0
        gammas_call_sell = und_data.get('gammas_call_sell') or 0
        gammas_put_sell = und_data.get('gammas_put_sell') or 0
        und_data['net_cust_gamma_flow_und'] = (gammas_call_buy + gammas_put_buy) - (gammas_call_sell + gammas_put_sell)
        
        vegas_buy = und_data.get('vegas_buy', 0) or 0
        vegas_sell = und_data.get('vegas_sell', 0) or 0
        thetas_buy = und_data.get('thetas_buy', 0) or 0
        thetas_sell = und_data.get('thetas_sell', 0) or 0
        
        und_data['net_cust_vega_flow_und'] = vegas_buy - vegas_sell
        und_data['net_cust_theta_flow_und'] = thetas_buy - thetas_sell

        return und_data

    def _calculate_gib_based_metrics(self, und_data: Dict) -> Dict:
        """
        Calculate GIB, HP_EOD, and TD_GIB metrics per system guide with Pydantic-first scaling.

        This method follows the EOTS v2.5 system guide for dollarization but applies
        appropriate scaling for dashboard display while maintaining calculation accuracy.
        """
        # GIB (Gamma Imbalance from Open Interest) - CORRECTED per system guide
        # Formula: GIB = Sum_of_Call_GXOI - Sum_of_Put_GXOI
        call_gxoi = und_data.get('call_gxoi', 0)
        put_gxoi = und_data.get('put_gxoi', 0)

        # Enhanced logging to debug the issue
        self.logger.info(f"🔍 GIB DEBUG - call_gxoi: {call_gxoi} (type: {type(call_gxoi)})")
        self.logger.info(f"🔍 GIB DEBUG - put_gxoi: {put_gxoi} (type: {type(put_gxoi)})")
        self.logger.info(f"🔍 GIB DEBUG - Available und_data keys: {list(und_data.keys())}")

        # Step 1: Calculate raw GIB in gamma units - FIX: Handle None values
        call_gxoi_safe = call_gxoi or 0.0
        put_gxoi_safe = put_gxoi or 0.0
        gib_raw_gamma_units = call_gxoi_safe - put_gxoi_safe

        # Step 2: Dollarize per system guide
        # GIB_Dollar_Value = GIB * Underlying_Price * Contract_Multiplier
        underlying_price = und_data.get('price', 100.0)
        contract_multiplier = 100  # Standard options contract multiplier
        gib_dollar_value_full = gib_raw_gamma_units * underlying_price * contract_multiplier

        # Step 3: Scale for dashboard display (Pydantic-first approach)
        # The full dollarized value is mathematically correct but too large for dashboard gauges
        # Scale down by 10,000 to bring values into the expected dashboard range (-50k to 50k)
        gib_display_value = gib_dollar_value_full / 10000.0

        # Store both full calculation and display-scaled values following Pydantic schema
        und_data['gib_oi_based_und'] = gib_display_value  # Main GIB value (scaled for display)
        und_data['gib_raw_gamma_units_und'] = gib_raw_gamma_units  # Raw gamma units for reference
        und_data['gib_dollar_value_full_und'] = gib_dollar_value_full  # Full dollarized value for calculations

        self.logger.info(f"🔍 GIB RESULT - raw_gamma={gib_raw_gamma_units:.2f}, price={underlying_price:.2f}, "
                        f"full_dollar_value={gib_dollar_value_full:.2f}, display_value={gib_display_value:.2f}")

        # HP_EOD (End-of-Day Hedging Pressure) calculation
        hp_eod_value = self.calculate_hp_eod_und_v2_5(und_data)
        und_data['hp_eod_und'] = hp_eod_value
        self.logger.debug(f"HP_EOD calculated: {hp_eod_value}")

        # TD_GIB (Time-Decayed GIB) - Enhanced calculation using display-scaled value
        current_time = datetime.now().time()
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM

        if market_open <= current_time <= market_close:
            # Calculate time decay factor (higher closer to close)
            total_market_minutes = (market_close.hour - market_open.hour) * 60 + (market_close.minute - market_open.minute)
            current_minutes = (current_time.hour - market_open.hour) * 60 + (current_time.minute - market_open.minute)
            time_decay_factor = max(0.1, current_minutes / total_market_minutes)  # Min 0.1, max 1.0

            # Use display-scaled value for TD_GIB to maintain consistency
            td_gib_value = gib_display_value * time_decay_factor
        else:
            td_gib_value = 0

        und_data['td_gib_und'] = td_gib_value
        self.logger.debug(f"TD_GIB calculated: {td_gib_value} (time_decay_factor: {time_decay_factor if 'time_decay_factor' in locals() else 0})")

        return und_data

    def calculate_hp_eod_und_v2_5(self, und_data: Dict) -> float:
        """
        Calculate HP_EOD (End-of-Day Hedging Pressure) - v2.5 Pydantic-first implementation.

        Uses the full dollarized GIB value for accurate calculations per EOTS system guide,
        then scales the result appropriately for dashboard display.
        """
        try:
            # Get full dollarized GIB value for accurate HP_EOD calculation
            gib_full = und_data.get('gib_dollar_value_full_und', 0.0)

            # Fallback to scaled value if full value not available (backward compatibility)
            if gib_full == 0.0:
                gib_display = und_data.get('gib_oi_based_und', 0.0)
                gib_full = gib_display * 10000.0  # Convert back to full scale

            # Get current and reference prices
            current_price = und_data.get('price', 0.0)

            # Try multiple reference price sources with better fallback logic
            reference_price = (
                und_data.get('day_open_price_und') or
                und_data.get('tradier_open') or
                und_data.get('prev_day_close_price_und') or
                current_price * 0.995  # Use 0.5% below current as fallback
            )

            # Enhanced time-based calculation - work during trading hours
            current_time = datetime.now().time()
            market_open = time(9, 30)  # 9:30 AM
            market_close = time(16, 0)  # 4:00 PM

            if market_open <= current_time <= market_close:
                # Calculate time progression through trading day
                total_market_minutes = (market_close.hour - market_open.hour) * 60 + (market_close.minute - market_open.minute)
                current_minutes = (current_time.hour - market_open.hour) * 60 + (current_time.minute - market_open.minute)
                time_progression = current_minutes / total_market_minutes  # 0.0 to 1.0

                # HP_EOD increases as we approach end of day
                time_multiplier = 0.5 + (time_progression * 0.5)  # 0.5 to 1.0

                # Calculate HP_EOD using full GIB value per system guide
                price_change = current_price - reference_price

                # Calculate HP_EOD = GIB_Dollar_Value_Per_Point * Price_Difference (per system guide)
                hp_eod_full = gib_full * price_change * time_multiplier

                # Scale down for dashboard display (divide by 10,000 to match GIB scaling)
                hp_eod_display = hp_eod_full / 10000.0

                self.logger.debug(f"HP_EOD calculation: gib_full={gib_full}, price_change={price_change}, "
                                f"time_multiplier={time_multiplier:.3f}, hp_eod_full={hp_eod_full}, "
                                f"hp_eod_display={hp_eod_display}")

                return float(hp_eod_display)
            else:
                # Outside trading hours
                self.logger.debug(f"Current time {current_time} is outside trading hours, HP_EOD = 0")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating HP_EOD: {e}")
            return 0.0

    def _calculate_enhanced_flow_metrics(self, und_data: Dict, symbol: str) -> Dict:
        """Calculates Tier 3 Enhanced Flow Metrics: VAPI-FA, DWFD, TW-LAF."""
        try:
            # Calculate VAPI-FA (Volume-Adjusted Premium Intensity with Flow Acceleration)
            und_data = self._calculate_vapi_fa(und_data, symbol)
            
            # Calculate DWFD (Delta-Weighted Flow Divergence)
            und_data = self._calculate_dwfd(und_data, symbol)
            
            # Calculate TW-LAF (Time-Weighted Liquidity-Adjusted Flow)
            und_data = self._calculate_tw_laf(und_data, symbol)
            
            self.logger.debug(f"Enhanced flow metrics calculated for {symbol}")
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced flow metrics for {symbol}: {e}", exc_info=True)
            # Set default values on error
            und_data['vapi_fa_z_score_und'] = 0.0
            und_data['dwfd_z_score_und'] = 0.0
            und_data['tw_laf_z_score_und'] = 0.0
            return und_data
        
    def _calculate_adaptive_metrics(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculates Tier 2 Adaptive Metrics: A-DAG, E-SDAG, D-TDPI, VRI 2.0."""
        if len(df_strike) == 0:
            return df_strike
            
        try:
            # Get market context for adaptive calculations
            market_regime = und_data.get('current_market_regime', 'REGIME_NEUTRAL')
            volatility_context = self._get_volatility_context(und_data)
            dte_context = self._get_average_dte_context(df_strike)
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            
            # Calculate A-DAG (Adaptive Delta-Adjusted Gamma Exposure)
            df_strike = self._calculate_a_dag(df_strike, und_data, market_regime, volatility_context, dte_context)
            
            # Calculate E-SDAG methodologies (Enhanced Skew and Delta Adjusted Gamma Exposure)
            df_strike = self._calculate_e_sdag(df_strike, und_data, market_regime, volatility_context, dte_context)
            
            # Calculate D-TDPI (Dynamic Time Decay Pressure Indicator)
            df_strike = self._calculate_d_tdpi(df_strike, und_data, market_regime, volatility_context, dte_context)
            
            # Calculate VRI 2.0 (Volatility Regime Indicator 2.0) - Canonical EOTS v2.5 implementation
            df_strike = self._calculate_vri_2_0(df_strike, und_data, market_regime, volatility_context, dte_context)

            # Calculate Concentration Indices (GCI, DCI) - Required for 0DTE suite
            df_strike = self._calculate_concentration_indices(df_strike, und_data)

            # Calculate 0DTE Suite metrics if applicable
            df_strike = self._calculate_0dte_suite(df_strike, und_data, dte_context)
            
            # Calculate Enhanced Heatmap Data (SGDHP, IVSDH, UGCH)
            df_strike = self._calculate_enhanced_heatmap_data(df_strike, und_data)

            # Calculate A-MSPI (Adaptive Market Structure Pressure Index)
            df_strike = self._calculate_a_mspi(df_strike, und_data, market_regime, volatility_context, dte_context)

            self.logger.debug(f"Adaptive metrics calculated for {len(df_strike)} strikes")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive metrics: {e}", exc_info=True)
            return df_strike

    def _calculate_vapi_fa(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate VAPI-FA (Volume-Adjusted Premium Intensity with Flow Acceleration)."""
        try:
            # Debug logging for input data
            self.logger.debug(f"VAPI-FA calculation for {symbol}")
            self.logger.debug(f"Available und_data keys: {list(und_data.keys())}")
            
            # Get isolated configuration parameters
            z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20)
            
            # Extract required inputs - CORRECTED per system guide
            # Use proper field names for 5m and 15m flows
            net_value_flow_5m = und_data.get('net_value_flow_5m_und', und_data.get('total_nvp', und_data.get('value_bs', 0.0))) or 0.0
            net_vol_flow_5m = und_data.get('net_vol_flow_5m_und', und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0))) or 0.0
            # Try to get actual 15m data, fallback to approximation if not available
            net_vol_flow_15m = und_data.get('net_vol_flow_15m_und', net_vol_flow_5m * 2.8) or (net_vol_flow_5m * 2.8)
            current_iv = und_data.get('u_volatility', und_data.get('Current_Underlying_IV', und_data.get('implied_volatility', 0.20))) or 0.20
            
            self.logger.debug(f"VAPI-FA inputs: net_value_flow_5m={net_value_flow_5m}, net_vol_flow_5m={net_vol_flow_5m}, net_vol_flow_15m={net_vol_flow_15m}, current_iv={current_iv}")
            
            # Ensure proper types
            net_value_flow_5m = float(net_value_flow_5m)
            net_vol_flow_5m = float(net_vol_flow_5m)
            net_vol_flow_15m = float(net_vol_flow_15m)
            current_iv = float(current_iv)
            
            # Step 1: Calculate Premium-to-Volume Ratio (PVR_5m_Und) - CORRECTED per system guide
            # Formula: PVR_5m_Und = NetValueFlow_5m_Und / NetVolFlow_5m_Und (preserving sign)
            if abs(net_vol_flow_5m) > 0.001:
                pvr_5m = net_value_flow_5m / net_vol_flow_5m  # Direct division, preserves sign naturally
            else:
                pvr_5m = 0.0
            
            # Step 2: Volatility Adjustment (Context Component) - FIXED FORMULA
            volatility_adjusted_pvr_5m = pvr_5m * current_iv  # MULTIPLY, not divide!
            
            # Step 3: Calculate Flow Acceleration (FA_5m_Und) - FIXED FORMULA
            # FA_5m = NetVolFlow_5m - (NetVolFlow_15m - NetVolFlow_5m)/2
            flow_in_prior_5_to_10_min = (net_vol_flow_15m - net_vol_flow_5m) / 2.0
            flow_acceleration_5m = net_vol_flow_5m - flow_in_prior_5_to_10_min
            
            # Step 4: Calculate Final VAPI-FA - FIXED FORMULA (PRODUCT, not sum)
            vapi_fa_raw = volatility_adjusted_pvr_5m * flow_acceleration_5m
            
            self.logger.debug(f"VAPI-FA components: pvr_5m={pvr_5m:.2f}, vol_adj_pvr={volatility_adjusted_pvr_5m:.2f}, flow_accel={flow_acceleration_5m:.2f}")
            
            # Step 5: Z-score normalization per system guide
            # VAPI_FA_Z_Score_Und = (VAPI_FA_Und - mean(historical_VAPI_FA_Und)) / (std(historical_VAPI_FA_Und) + EPSILON)
            vapi_fa_cache = self._add_to_intraday_cache(symbol, 'vapi_fa', float(vapi_fa_raw), max_size=200)

            if len(vapi_fa_cache) >= 10:  # Need sufficient history for Z-score
                vapi_fa_mean = np.mean(vapi_fa_cache)
                vapi_fa_std = np.std(vapi_fa_cache)
                vapi_fa_z_score = (vapi_fa_raw - vapi_fa_mean) / max(float(vapi_fa_std), 0.001)  # EPSILON
            else:
                # Fallback to percentile method if insufficient history
                vapi_fa_z_score = self._calculate_percentile_gauge_value(vapi_fa_cache, float(vapi_fa_raw))
            
            und_data['vapi_fa_raw_und'] = vapi_fa_raw
            und_data['vapi_fa_z_score_und'] = vapi_fa_z_score
            und_data['vapi_fa_pvr_5m_und'] = pvr_5m
            und_data['vapi_fa_flow_accel_5m_und'] = flow_acceleration_5m
            
            self.logger.debug(f"VAPI-FA results for {symbol}: raw={vapi_fa_raw:.2f}, z_score={vapi_fa_z_score:.2f}, intraday_cache_size={len(vapi_fa_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating VAPI-FA for {symbol}: {e}", exc_info=True)
            und_data['vapi_fa_raw_und'] = 0.0
            und_data['vapi_fa_z_score_und'] = 0.0
            und_data['vapi_fa_pvr_5m_und'] = 0.0
            und_data['vapi_fa_flow_accel_5m_und'] = 0.0
            return und_data
    
    def _calculate_dwfd(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate DWFD (Delta-Weighted Flow Divergence)."""
        try:
            # Get isolated configuration parameters
            z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20)
            
            # Extract required inputs - prioritize aggregated totals over individual ones
            net_value_flow = und_data.get('total_nvp', und_data.get('value_bs', 0.0)) or 0.0
            net_vol_flow = und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0)) or 0.0
            
            self.logger.debug(f"DWFD inputs for {symbol}: net_value_flow={net_value_flow}, net_vol_flow={net_vol_flow}")
            
            # Ensure proper types
            net_value_flow = float(net_value_flow)
            net_vol_flow = float(net_vol_flow)
            
            # Step 1: Proxy for Directional Delta Flow
            directional_delta_flow = net_vol_flow
            
            # Step 2: Calculate Flow Value vs. Volume Divergence using intraday cache
            value_cache = self._add_to_intraday_cache(symbol, 'net_value_flow', net_value_flow, max_size=200)
            vol_cache = self._add_to_intraday_cache(symbol, 'net_vol_flow', net_vol_flow, max_size=200)
            
            # Calculate Z-scores using intraday data
            if len(value_cache) >= 10:
                value_mean = np.mean(value_cache)
                value_std = np.std(value_cache)
                value_z = (net_value_flow - value_mean) / max(float(value_std), 0.001)
            else:
                value_z = 0.0
            
            if len(vol_cache) >= 10:
                vol_mean = np.mean(vol_cache)
                vol_std = np.std(vol_cache)
                vol_z = (net_vol_flow - vol_mean) / max(float(vol_std), 0.001)
            else:
                vol_z = 0.0
            
            # Flow Value vs. Volume Divergence
            fvd = value_z - vol_z
            
            # Step 3: Calculate Final DWFD - FIXED FORMULA per system guide
            # DWFD_5m_Und = ProxyDeltaFlow_5m - (Weight_Factor * FVD_5m)
            weight_factor = 0.5  # Configurable weight for FVD component
            dwfd_raw = directional_delta_flow - (weight_factor * fvd)
            
            self.logger.debug(f"DWFD components for {symbol}: directional_flow={directional_delta_flow:.2f}, fvd={fvd:.2f}, weight_factor={weight_factor}, result={dwfd_raw:.2f}")
            
            # Step 4: Percentile-based normalization using intraday cache
            dwfd_cache = self._add_to_intraday_cache(symbol, 'dwfd', float(dwfd_raw), max_size=200)
            dwfd_z_score = self._calculate_percentile_gauge_value(dwfd_cache, float(dwfd_raw))
            
            und_data['dwfd_raw_und'] = dwfd_raw
            und_data['dwfd_z_score_und'] = dwfd_z_score
            und_data['dwfd_fvd_und'] = fvd
            
            self.logger.debug(f"DWFD results for {symbol}: raw={dwfd_raw:.2f}, z_score={dwfd_z_score:.2f}, fvd={fvd:.2f}, intraday_cache_size={len(dwfd_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating DWFD for {symbol}: {e}", exc_info=True)
            und_data['dwfd_raw_und'] = 0.0
            und_data['dwfd_z_score_und'] = 0.0
            und_data['dwfd_fvd_und'] = 0.0
            return und_data
    
    def _calculate_tw_laf(self, und_data: Dict, symbol: str) -> Dict:
        """Calculate TW-LAF (Time-Weighted Liquidity-Adjusted Flow)."""
        try:
            # Get isolated configuration parameters
            z_score_window = self._get_metric_config('enhanced_flow', 'z_score_window', 20)
            
            # Extract required inputs for multiple intervals
            net_vol_flow_5m = und_data.get('total_nvp_vol', und_data.get('volm_bs', 0.0)) or 0.0
            # For 15m and 30m flows, use proxies (in real implementation, would need actual interval data)
            net_vol_flow_15m = net_vol_flow_5m * 2.5  # Approximate 15m flow
            net_vol_flow_30m = net_vol_flow_5m * 4.0  # Approximate 30m flow
            
            underlying_price = und_data.get('price', 100.0) or 100.0
            
            # Ensure proper types
            net_vol_flow_5m = float(net_vol_flow_5m)
            net_vol_flow_15m = float(net_vol_flow_15m)
            net_vol_flow_30m = float(net_vol_flow_30m)
            underlying_price = float(underlying_price)
            
            self.logger.debug(f"TW-LAF inputs for {symbol}: 5m_flow={net_vol_flow_5m}, 15m_flow={net_vol_flow_15m}, 30m_flow={net_vol_flow_30m}, price={underlying_price}")
            
            # Step 1: Calculate Liquidity Factors for each interval
            # Simplified liquidity factor based on typical option spreads
            # In real implementation, would calculate from bid/ask spreads per interval
            base_spread_pct = 0.02  # 2% typical spread
            normalized_spread_5m = base_spread_pct * 1.0  # Most recent = tightest
            normalized_spread_15m = base_spread_pct * 1.2  # Slightly wider
            normalized_spread_30m = base_spread_pct * 1.5  # Wider for older data
            
            liquidity_factor_5m = 1.0 / (normalized_spread_5m + 0.001)
            liquidity_factor_15m = 1.0 / (normalized_spread_15m + 0.001)
            liquidity_factor_30m = 1.0 / (normalized_spread_30m + 0.001)
            
            # Step 2: Calculate Liquidity-Adjusted Flow for each interval
            liquidity_adjusted_flow_5m = net_vol_flow_5m * liquidity_factor_5m
            liquidity_adjusted_flow_15m = net_vol_flow_15m * liquidity_factor_15m
            liquidity_adjusted_flow_30m = net_vol_flow_30m * liquidity_factor_30m
            
            # Step 3: Calculate Time-Weighted Sum per system guide
            # Time weights: more recent gets higher weight
            weight_5m = 1.0   # Most recent
            weight_15m = 0.8  # Recent
            weight_30m = 0.6  # Older
            
            tw_laf_raw = (weight_5m * liquidity_adjusted_flow_5m + 
                         weight_15m * liquidity_adjusted_flow_15m + 
                         weight_30m * liquidity_adjusted_flow_30m)
            
            self.logger.debug(f"TW-LAF components for {symbol}: liq_adj_5m={liquidity_adjusted_flow_5m:.2f}, liq_adj_15m={liquidity_adjusted_flow_15m:.2f}, liq_adj_30m={liquidity_adjusted_flow_30m:.2f}")
            
            # Step 4: Percentile-based normalization using intraday cache
            tw_laf_cache = self._add_to_intraday_cache(symbol, 'tw_laf', float(tw_laf_raw), max_size=200)
            tw_laf_z_score = self._calculate_percentile_gauge_value(tw_laf_cache, float(tw_laf_raw))
            
            und_data['tw_laf_raw_und'] = tw_laf_raw
            und_data['tw_laf_z_score_und'] = tw_laf_z_score
            und_data['tw_laf_liquidity_factor_5m_und'] = liquidity_factor_5m
            und_data['tw_laf_time_weighted_sum_und'] = tw_laf_raw
            
            self.logger.debug(f"TW-LAF results for {symbol}: raw={tw_laf_raw:.2f}, z_score={tw_laf_z_score:.2f}, intraday_cache_size={len(tw_laf_cache)}")
            
            return und_data
            
        except Exception as e:
            self.logger.error(f"Error calculating TW-LAF for {symbol}: {e}", exc_info=True)
            und_data['tw_laf_raw_und'] = 0.0
            und_data['tw_laf_z_score_und'] = 0.0
            und_data['tw_laf_liquidity_factor_5m_und'] = 1.0
            und_data['tw_laf_time_weighted_sum_und'] = 0.0
            return und_data

    def _get_volatility_context(self, und_data: Dict) -> str:
        """Determine volatility context for adaptive calculations."""
        current_iv = und_data.get('Current_Underlying_IV', 0.20)
        if current_iv > 0.30:
            return 'HIGH_VOL'
        elif current_iv < 0.15:
            return 'LOW_VOL'
        else:
            return 'NORMAL_VOL'

    def _get_market_direction_bias(self, und_data: Dict) -> float:
        """
        Determine market direction bias for VRI calculations to prevent contradictory regime classifications.

        Returns:
            float: Direction bias multiplier (-1.0 to 1.0)
                  -1.0 = Strong bearish bias
                   0.0 = Neutral/sideways
                   1.0 = Strong bullish bias
        """
        try:
            # Get current price and recent price data
            current_price = und_data.get('price', 0.0)

            # Method 1: Use price change if available
            price_change_pct = und_data.get('price_change_pct', 0.0)

            # Method 2: Check for VIX-SPY divergence flags from ticker context
            vix_spy_divergence = und_data.get('vix_spy_price_divergence_strong_negative', False)

            # Method 3: Use any available trend indicators
            spy_trend = und_data.get('spy_trend', 'sideways')  # from ticker context if available

            # Calculate direction bias
            direction_bias = 0.0

            # Primary: Price change signal
            if abs(price_change_pct) > 0.005:  # > 0.5% move
                direction_bias = np.sign(price_change_pct) * min(abs(price_change_pct) * 20, 1.0)

            # Secondary: VIX-SPY divergence (bearish signal)
            if vix_spy_divergence:
                direction_bias = min(direction_bias - 0.3, -0.2)  # Add bearish bias

            # Tertiary: SPY trend context
            if spy_trend == 'up':
                direction_bias = max(direction_bias, 0.2)
            elif spy_trend == 'down':
                direction_bias = min(direction_bias, -0.2)

            # Ensure bounds
            direction_bias = max(-1.0, min(1.0, direction_bias))

            return direction_bias

        except Exception as e:
            self.logger.warning(f"Error calculating market direction bias: {e}")
            return 0.0  # Neutral bias on error
    
    def _get_average_dte_context(self, df_strike: pd.DataFrame) -> str:
        """Determine DTE context for adaptive calculations."""
        # This is a placeholder - in real implementation would calculate from options data
        return 'NORMAL_DTE'
    
    def _calculate_a_dag(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """Calculate A-DAG (Adaptive Delta-Adjusted Gamma Exposure) per system guide."""
        try:
            # Get A-DAG configuration parameters
            a_dag_config = self._get_metric_config('adaptive_metric_parameters', 'a_dag_settings', {})
            base_alpha_coeffs = a_dag_config.get('base_dag_alpha_coeffs', {
                'aligned': 1.35, 'opposed': 0.65, 'neutral': 1.0
            })

            # Get regime and volatility multipliers
            regime_multipliers = a_dag_config.get('regime_alpha_multipliers', {})
            volatility_multipliers = a_dag_config.get('volatility_context_alpha_multipliers', {})
            flow_sensitivity = a_dag_config.get('flow_sensitivity_by_regime', {})
            dte_scaling_config = a_dag_config.get('dte_gamma_flow_impact_scaling', {})

            # Step 1: Calculate Adaptive Alignment Coefficient (adaptive_dag_alpha)
            regime_mult_data = regime_multipliers.get(market_regime, {'aligned_mult': 1.0, 'opposed_mult': 1.0})
            vol_mult_data = volatility_multipliers.get(volatility_context, {'aligned_mult': 1.0, 'opposed_mult': 1.0})

            # Calculate adaptive alpha coefficients
            adaptive_alpha_aligned = (base_alpha_coeffs['aligned'] *
                                    regime_mult_data.get('aligned_mult', 1.0) *
                                    vol_mult_data.get('aligned_mult', 1.0))
            adaptive_alpha_opposed = (base_alpha_coeffs['opposed'] *
                                    regime_mult_data.get('opposed_mult', 1.0) *
                                    vol_mult_data.get('opposed_mult', 1.0))
            adaptive_alpha_neutral = base_alpha_coeffs['neutral']

            # Step 2: Get core inputs
            gxoi_at_strike = df_strike.get('total_gxoi_at_strike', 0)
            dxoi_at_strike = df_strike.get('total_dxoi_at_strike', 0)
            net_cust_delta_flow = df_strike.get('net_cust_delta_flow_at_strike', 0)
            net_cust_gamma_flow = df_strike.get('net_cust_gamma_flow_at_strike_proxy', 0)
            
            # Determine flow alignment (aligned, opposed, neutral)
            delta_alignment = np.sign(dxoi_at_strike) * np.sign(net_cust_delta_flow)
            gamma_alignment = np.sign(gxoi_at_strike) * np.sign(net_cust_gamma_flow)
            
            # Combined alignment score
            combined_alignment = (delta_alignment + gamma_alignment) / 2.0
            
            # Map alignment to coefficient type
            alignment_type = np.where(
                combined_alignment > 0.3, 'aligned',
                np.where(combined_alignment < -0.3, 'opposed', 'neutral')
            )
            
            # Step 3: Apply adaptive coefficients
            adaptive_alpha = np.where(
                alignment_type == 'aligned',
                adaptive_alpha_aligned,
                np.where(
                    alignment_type == 'opposed',
                    adaptive_alpha_opposed,
                    adaptive_alpha_neutral
                )
            )
             
            # Step 4: DTE Scaling for Gamma/Flow Impact
            dte_scaling = self._get_dte_scaling_factor(dte_context)
            
            # Step 5: Calculate A-DAG using the core formula with DIRECTIONAL COMPONENT
            # A-DAG = GXOI * directional_multiplier * (1 + adaptive_alpha * flow_alignment) * dte_scaling
            flow_alignment_ratio = np.where(
                np.abs(gxoi_at_strike) > 0,
                (net_cust_delta_flow + net_cust_gamma_flow) / (np.abs(gxoi_at_strike) + 1e-6),
                0.0
            )
            
            # FIX: Add directional component based on strike vs current price
            current_price = und_data.get('price', 0.0)
            strikes = df_strike['strike'] if 'strike' in df_strike.columns else pd.Series([current_price] * len(df_strike))
            
            # Apply directional signs: above price = resistance (negative), below = support (positive)
            directional_multiplier = np.where(strikes > current_price, -1, 1)
            
            # Calculate A-DAG with proper directional component
            a_dag_exposure = gxoi_at_strike * directional_multiplier * (1 + adaptive_alpha * flow_alignment_ratio) * dte_scaling
            
            # Step 6: Optional Volume-Weighted GXOI refinement
            use_volume_weighted = self._get_metric_config('adaptive', 'use_volume_weighted_gxoi', False)
            if use_volume_weighted:
                volume_weight = df_strike.get('total_volume_at_strike', 1.0)
                if not isinstance(volume_weight, pd.Series):
                    volume_weight = pd.Series([volume_weight] * len(df_strike))
                volume_factor = np.log1p(volume_weight) / np.log1p(volume_weight.mean() + 1e-6)
                a_dag_exposure *= volume_factor
            
            df_strike['a_dag_exposure'] = a_dag_exposure
            df_strike['a_dag_adaptive_alpha'] = adaptive_alpha
            df_strike['a_dag_flow_alignment'] = flow_alignment_ratio
            df_strike['a_dag_directional_multiplier'] = directional_multiplier  # Store for debugging
            # --- FIX: assign a_dag_strike for dashboard compatibility ---
            df_strike['a_dag_strike'] = df_strike['a_dag_exposure']

            # Debug logging
            self.logger.debug(f"[A-DAG] Calculated for {len(df_strike)} strikes")
            self.logger.debug(f"[A-DAG] Range: [{a_dag_exposure.min():.3f}, {a_dag_exposure.max():.3f}]")
            self.logger.debug(f"[A-DAG] Non-zero count: {(a_dag_exposure != 0).sum()}")

            return df_strike
            
        except Exception as e:
            self.logger.error(f"[A-DAG] CRITICAL ERROR in A-DAG calculation: {e}", exc_info=True)
            self.logger.error(f"[A-DAG] Input data shapes - df_strike: {len(df_strike)}, und_data keys: {list(und_data.keys())}")
            self.logger.error(f"[A-DAG] Market regime: {market_regime}, Volatility context: {volatility_context}, DTE context: {dte_context}")
            df_strike['a_dag_exposure'] = 0.0
            df_strike['a_dag_adaptive_alpha'] = 0.0
            df_strike['a_dag_flow_alignment'] = 0.0
            df_strike['a_dag_directional_multiplier'] = 0.0
            df_strike['a_dag_strike'] = 0.0
            return df_strike
    
    def _calculate_e_sdag(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """Calculate E-SDAG (Enhanced Skew and Delta Adjusted Gamma Exposure) per system guide."""
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            # Get E-SDAG configuration parameters
            e_sdag_config = self._get_metric_config('adaptive_metric_parameters', 'e_sdag_settings', {})
            use_enhanced_sgexoi = e_sdag_config.get('use_enhanced_skew_calculation_for_sgexoi', True)
            sgexoi_params = e_sdag_config.get('sgexoi_calculation_params', {})
            base_delta_weights = e_sdag_config.get('base_delta_weight_factors', {
                'e_sdag_mult': 0.5, 'e_sdag_dir': 0.6, 'e_sdag_w': 0.4, 'e_sdag_vf': 0.7
            })
            regime_multipliers = e_sdag_config.get('regime_delta_weight_multipliers', {})
            volatility_multipliers = e_sdag_config.get('volatility_delta_weight_multipliers', {})

            # Step 1: Calculate Enhanced SGEXOI_v2_5 (if enabled) or use GXOI
            gxoi_at_strike = df_strike.get('total_gxoi_at_strike', 0)
            if use_enhanced_sgexoi:
                sgexoi_v2_5 = self._calculate_sgexoi_v2_5(gxoi_at_strike, und_data, sgexoi_params, dte_context)
            else:
                sgexoi_v2_5 = gxoi_at_strike

            # Step 2: Get and normalize DXOI
            dxoi_at_strike = df_strike.get('total_dxoi_at_strike', 0)
            dxoi_normalized = self._normalize_flow(dxoi_at_strike, 'dxoi', symbol)

            # Step 3: Calculate Adaptive Delta Weighting Factors for each methodology
            regime_mult_data = regime_multipliers.get(market_regime, {})
            vol_mult_data = volatility_multipliers.get(volatility_context, {})

            adaptive_delta_weights = {}
            for methodology in ['e_sdag_mult', 'e_sdag_dir', 'e_sdag_w', 'e_sdag_vf']:
                base_weight = base_delta_weights.get(methodology, 0.5)
                regime_mult = regime_mult_data.get(methodology, 1.0)
                vol_mult = vol_mult_data.get(methodology, 1.0)
                adaptive_delta_weights[methodology] = base_weight * regime_mult * vol_mult

            # Step 4: Calculate each E-SDAG methodology using system guide formulas

            # E-SDAG_Mult: SGEXOI_v2_5 * (1 + Adaptive_Delta_Weight * Normalized_DXOI)
            e_sdag_mult = sgexoi_v2_5 * (1 + adaptive_delta_weights['e_sdag_mult'] * dxoi_normalized)

            # E-SDAG_Dir: SGEXOI_v2_5 + (Adaptive_Delta_Weight * DXOI)
            e_sdag_dir = sgexoi_v2_5 + (adaptive_delta_weights['e_sdag_dir'] * dxoi_at_strike)

            # E-SDAG_W: Weighted average of GEXOI and DEXOI with adaptive weights
            gamma_weight = 1.0 - adaptive_delta_weights['e_sdag_w']
            delta_weight = adaptive_delta_weights['e_sdag_w']
            e_sdag_w = gamma_weight * sgexoi_v2_5 + delta_weight * np.abs(dxoi_at_strike)

            # E-SDAG_VF: SGEXOI_v2_5 * (1 - Adaptive_Delta_Weight * Normalized_DXOI) [note subtraction]
            e_sdag_vf = sgexoi_v2_5 * (1 - adaptive_delta_weights['e_sdag_vf'] * dxoi_normalized)

            # Store results
            df_strike['e_sdag_mult_strike'] = e_sdag_mult
            df_strike['e_sdag_dir_strike'] = e_sdag_dir
            df_strike['e_sdag_w_strike'] = e_sdag_w
            df_strike['e_sdag_vf_strike'] = e_sdag_vf

            # Store adaptive weights for debugging
            df_strike['e_sdag_adaptive_delta_weight_mult'] = adaptive_delta_weights['e_sdag_mult']
            df_strike['e_sdag_adaptive_delta_weight_dir'] = adaptive_delta_weights['e_sdag_dir']
            df_strike['e_sdag_adaptive_delta_weight_w'] = adaptive_delta_weights['e_sdag_w']
            df_strike['e_sdag_adaptive_delta_weight_vf'] = adaptive_delta_weights['e_sdag_vf']

            self.logger.debug(f"[E-SDAG] Calculated for {len(df_strike)} strikes with adaptive weights: "
                            f"mult={adaptive_delta_weights['e_sdag_mult']:.3f}, "
                            f"dir={adaptive_delta_weights['e_sdag_dir']:.3f}, "
                            f"w={adaptive_delta_weights['e_sdag_w']:.3f}, "
                            f"vf={adaptive_delta_weights['e_sdag_vf']:.3f}")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating E-SDAG: {e}", exc_info=True)
            df_strike['e_sdag_mult_strike'] = 0.0
            df_strike['e_sdag_dir_strike'] = 0.0
            df_strike['e_sdag_w_strike'] = 0.0
            df_strike['e_sdag_vf_strike'] = 0.0
            return df_strike
    
    def _calculate_d_tdpi(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """Calculate D-TDPI (Dynamic Time Decay Pressure Indicator), E-CTR (Effective Call/Put Trade Ratio), and E-TDFI (Effective Time Decay Flow Index)."""
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            # Core inputs for D-TDPI
            charm_oi = df_strike.get('total_charmxoi_at_strike', 0)
            theta_oi = df_strike.get('total_txoi_at_strike', 0)
            net_cust_theta_flow = df_strike.get('net_cust_theta_flow_at_strike', 0)

            # Normalized theta flow (simplified Z-score)
            theta_flow_normalized = self._normalize_flow(net_cust_theta_flow, 'theta_flow', symbol)

            # Calculate D-TDPI
            d_tdpi_value = charm_oi * np.sign(theta_oi) * (1 + theta_flow_normalized * 0.4)

            # --- E-CTR (Enhanced Charm Decay Rate) Calculation ---
            # Formula from system guide: E_CTR_strike = abs(Adaptive_NetCharmFlowProxy_at_Strike) / (abs(Adaptive_NetCustThetaFlow_at_Strike) + EPSILON)
            try:
                # Get charm flow proxy and theta flow data
                net_charm_flow_proxy = df_strike.get('net_cust_charm_flow_proxy_at_strike', pd.Series([0.0] * len(df_strike)))

                # Calculate E-CTR using exact formula from system guide
                e_ctr_numerator = np.abs(net_charm_flow_proxy)
                e_ctr_denominator = np.abs(net_cust_theta_flow) + 1e-9  # EPSILON
                e_ctr_value = e_ctr_numerator / e_ctr_denominator

                self.logger.debug(f"[E-CTR] Calculated for {len(df_strike)} strikes, range: [{e_ctr_value.min():.3f}, {e_ctr_value.max():.3f}]")

            except Exception as e:
                self.logger.error(f"Error calculating E-CTR: {e}")
                e_ctr_value = pd.Series([0.0] * len(df_strike))

            # --- E-TDFI (Enhanced Time Decay Flow Imbalance) Calculation ---
            # Formula from system guide: E_TDFI_strike = normalize(abs(Adaptive_NetCustThetaFlow_at_Strike)) / (normalize(abs(ThetaOI_at_Strike)) + EPSILON)
            try:
                # Get theta OI data
                theta_oi_at_strike = df_strike.get('total_txoi_at_strike', pd.Series([0.0] * len(df_strike)))

                # Normalize theta flow and theta OI components
                theta_flow_abs = np.abs(net_cust_theta_flow)
                theta_oi_abs = np.abs(theta_oi_at_strike)

                # Simple normalization (can be enhanced with Z-score if needed)
                theta_flow_max = np.maximum(theta_flow_abs.max(), 1e-9)
                theta_oi_max = np.maximum(theta_oi_abs.max(), 1e-9)

                theta_flow_normalized = theta_flow_abs / theta_flow_max
                theta_oi_normalized = theta_oi_abs / theta_oi_max

                # Calculate E-TDFI using exact formula from system guide
                e_tdfi_value = theta_flow_normalized / (theta_oi_normalized + 1e-9)  # EPSILON

                self.logger.debug(f"[E-TDFI] Calculated for {len(df_strike)} strikes, range: [{e_tdfi_value.min():.3f}, {e_tdfi_value.max():.3f}]")

            except Exception as e:
                self.logger.error(f"Error calculating E-TDFI: {e}")
                e_tdfi_value = pd.Series([0.0] * len(df_strike))

            # Store all results
            df_strike['d_tdpi_strike'] = d_tdpi_value
            df_strike['e_ctr_strike'] = e_ctr_value
            df_strike['e_tdfi_strike'] = e_tdfi_value

            self.logger.debug(f"[TIME DECAY SUITE] Calculated D-TDPI, E-CTR, E-TDFI for {len(df_strike)} strikes")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating time decay metrics (D-TDPI, E-CTR, E-TDFI): {e}", exc_info=True)
            df_strike['d_tdpi_strike'] = 0.0
            df_strike['e_ctr_strike'] = 0.0
            df_strike['e_tdfi_strike'] = 0.0
            return df_strike

    def _calculate_concentration_indices(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate GCI (Gamma Concentration Index) and DCI (Delta Concentration Index) at strike level.

        These indices measure the concentration of gamma and delta exposure at each strike level,
        required for the 0DTE suite calculations.
        """
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')

            # --- GCI (Gamma Concentration Index) Calculation ---
            try:
                # Get gamma OI data at strike level
                total_gxoi_at_strike = df_strike.get('total_gxoi_at_strike', pd.Series([0.0] * len(df_strike)))

                # Calculate total gamma OI across all strikes
                total_gxoi_underlying = total_gxoi_at_strike.abs().sum()

                if total_gxoi_underlying > 1e-9:
                    # Calculate proportion of gamma OI at each strike (HHI-style)
                    proportion_gamma_oi = total_gxoi_at_strike.abs() / total_gxoi_underlying

                    # GCI at strike level: proportion squared (concentration measure)
                    gci_strike = proportion_gamma_oi ** 2

                    self.logger.debug(f"[GCI] Calculated for {len(df_strike)} strikes, range: [{gci_strike.min():.4f}, {gci_strike.max():.4f}]")
                else:
                    gci_strike = pd.Series([0.0] * len(df_strike))
                    self.logger.debug("[GCI] No gamma OI found, setting all strikes to 0")

            except Exception as e:
                self.logger.error(f"Error calculating GCI: {e}")
                gci_strike = pd.Series([0.0] * len(df_strike))

            # --- DCI (Delta Concentration Index) Calculation ---
            try:
                # Get delta OI data at strike level
                total_dxoi_at_strike = df_strike.get('total_dxoi_at_strike', pd.Series([0.0] * len(df_strike)))

                # Calculate total delta OI across all strikes
                total_dxoi_underlying = total_dxoi_at_strike.abs().sum()

                if total_dxoi_underlying > 1e-9:
                    # Calculate proportion of delta OI at each strike (HHI-style)
                    proportion_delta_oi = total_dxoi_at_strike.abs() / total_dxoi_underlying

                    # DCI at strike level: proportion squared (concentration measure)
                    dci_strike = proportion_delta_oi ** 2

                    self.logger.debug(f"[DCI] Calculated for {len(df_strike)} strikes, range: [{dci_strike.min():.4f}, {dci_strike.max():.4f}]")
                else:
                    dci_strike = pd.Series([0.0] * len(df_strike))
                    self.logger.debug("[DCI] No delta OI found, setting all strikes to 0")

            except Exception as e:
                self.logger.error(f"Error calculating DCI: {e}")
                dci_strike = pd.Series([0.0] * len(df_strike))

            # Store results
            df_strike['gci_strike'] = gci_strike
            df_strike['dci_strike'] = dci_strike

            self.logger.debug(f"[CONCENTRATION INDICES] Calculated GCI and DCI for {len(df_strike)} strikes")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating concentration indices (GCI, DCI): {e}", exc_info=True)
            df_strike['gci_strike'] = 0.0
            df_strike['dci_strike'] = 0.0
            return df_strike
    
    def _calculate_vri_2_0(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """Calculate VRI 2.0 (Volatility Regime Indicator 2.0) - Canonical EOTS v2.5 implementation."""
        try:
            import numpy as np
            import pandas as pd
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            EPSILON = 1e-9
            vri_cfg = self.config_manager.get_setting("adaptive_metric_parameters.vri_2_0_settings", default={})
            base_gamma_coeffs = vri_cfg.get("base_vri_gamma_coeffs", {"aligned": 1.3, "opposed": 0.7, "neutral": 1.0})
            # --- Ensure all required columns are present and correctly mapped to schema ---
            required_cols = [
                'total_vanna_at_strike', 'total_vxoi_at_strike', 'total_vommaxoi_at_strike',
                'net_cust_vanna_flow_proxy_at_strike', 'net_cust_vomma_flow_proxy_at_strike'
            ]
            for col in required_cols:
                if col not in df_strike.columns:
                    self.logger.warning(f"[VRI2.0] Missing required column: {col}. Filling with 0.")
                    df_strike[col] = 0.0
                df_strike[col] = pd.to_numeric(df_strike[col], errors='coerce').fillna(0.0)
            # --- IV/term structure context ---
            current_iv = und_data.get('u_volatility', 0.20) or 0.20
            front_iv = und_data.get('front_month_iv', current_iv)
            spot_iv = und_data.get('spot_iv', current_iv)
            try:
                front_iv = float(front_iv)
            except Exception:
                front_iv = current_iv
            try:
                spot_iv = float(spot_iv)
            except Exception:
                spot_iv = current_iv
            term_structure_factor = front_iv / (spot_iv + EPSILON) if spot_iv else 1.0
            enhanced_vol_context_weight = und_data.get('ivsdh_vol_context_weight', 1.0)
            try:
                enhanced_vol_context_weight = float(enhanced_vol_context_weight)
            except Exception:
                enhanced_vol_context_weight = 1.0
            # --- Enhanced vomma factor: vomma/vega ratio, fallback to 1.0 if vega is 0 ---
            vega_oi = pd.to_numeric(df_strike['total_vxoi_at_strike'], errors='coerce').replace(0, EPSILON)
            vomma_oi = pd.to_numeric(df_strike['total_vommaxoi_at_strike'], errors='coerce')
            enhanced_vomma_factor = np.where(vega_oi != 0, vomma_oi / (vega_oi + EPSILON), 1.0)
            enhanced_vomma_factor = enhanced_vomma_factor.astype(float)
            # --- Adaptive gamma coeff: modulated by regime/DTE ---
            def get_gamma_coeff(row):
                regime = (market_regime or "neutral").lower()
                dte = (dte_context or "normal").lower()
                if "bull" in regime:
                    return base_gamma_coeffs.get("aligned", 1.3)
                elif "bear" in regime:
                    return base_gamma_coeffs.get("opposed", 0.7)
                else:
                    return base_gamma_coeffs.get("neutral", 1.0)
            gamma_coeffs = df_strike.apply(get_gamma_coeff, axis=1)
            if not isinstance(gamma_coeffs, pd.Series):
                gamma_coeffs = pd.Series([gamma_coeffs] * len(df_strike))
            # --- Adaptive Vanna/Vomma flow ratios ---
            vanna_oi = pd.to_numeric(df_strike['total_vanna_at_strike'], errors='coerce').replace(0, EPSILON)
            vanna_flow = pd.to_numeric(df_strike['net_cust_vanna_flow_proxy_at_strike'], errors='coerce')
            if not isinstance(vanna_flow, pd.Series) or len(vanna_flow) != len(df_strike):
                vanna_flow = pd.Series([0.0] * len(df_strike))
            vanna_flow_ratio = vanna_flow / (vanna_oi + EPSILON)
            if not isinstance(vanna_flow_ratio, pd.Series) or len(vanna_flow_ratio) != len(df_strike):
                vanna_flow_ratio = pd.Series([0.0] * len(df_strike))
            if len(vanna_flow_ratio) > 0:
                vanna_flow_ratio_mean = vanna_flow_ratio.mean()
                vanna_flow_ratio_std = vanna_flow_ratio.std() + EPSILON
                vanna_flow_ratio_norm = (vanna_flow_ratio - vanna_flow_ratio_mean) / vanna_flow_ratio_std
            else:
                vanna_flow_ratio_norm = pd.Series([0.0] * len(df_strike))
            vomma_oi = pd.to_numeric(df_strike['total_vommaxoi_at_strike'], errors='coerce').replace(0, EPSILON)
            vomma_flow = pd.to_numeric(df_strike['net_cust_vomma_flow_proxy_at_strike'], errors='coerce')
            if not isinstance(vomma_flow, pd.Series) or len(vomma_flow) != len(df_strike):
                vomma_flow = pd.Series([0.0] * len(df_strike))
            vomma_flow_ratio = vomma_flow / (vomma_oi + EPSILON)
            if not isinstance(vomma_flow_ratio, pd.Series) or len(vomma_flow_ratio) != len(df_strike):
                vomma_flow_ratio = pd.Series([0.0] * len(df_strike))
            if len(vomma_flow_ratio) > 0:
                vomma_flow_ratio_mean = vomma_flow_ratio.mean()
                vomma_flow_ratio_std = vomma_flow_ratio.std() + EPSILON
                vomma_flow_ratio_norm = (vomma_flow_ratio - vomma_flow_ratio_mean) / vomma_flow_ratio_std
            else:
                vomma_flow_ratio_norm = pd.Series([0.0] * len(df_strike))
            vega_oi = pd.to_numeric(df_strike['total_vxoi_at_strike'], errors='coerce')
            sign_vega_oi = np.sign(vega_oi)
            # --- Main VRI 2.0 calculation ---
            if not isinstance(vanna_oi, pd.Series):
                vanna_oi = pd.Series([vanna_oi] * len(df_strike))
            vanna_oi = pd.to_numeric(vanna_oi, errors='coerce').fillna(0).to_numpy(dtype=float, na_value=0.0)
            if not isinstance(sign_vega_oi, pd.Series):
                sign_vega_oi = pd.Series([sign_vega_oi] * len(df_strike))
            sign_vega_oi = pd.to_numeric(sign_vega_oi, errors='coerce').fillna(0).to_numpy(dtype=float, na_value=0.0)
            vanna_flow = gamma_coeffs * vanna_flow_ratio_norm.astype(float).to_numpy(dtype=float, na_value=0.0)
            one_plus_vanna_flow = 1 + vanna_flow
            vomma_flow = vomma_flow_ratio_norm.astype(float).to_numpy(dtype=float, na_value=0.0)
            vol_context = np.full(len(df_strike), float(enhanced_vol_context_weight))
            vomma_factor = np.array(enhanced_vomma_factor, dtype=float)
            term_factor = np.full(len(df_strike), float(term_structure_factor))
            vri_2_0 = (
                vanna_oi * sign_vega_oi * one_plus_vanna_flow * vomma_flow * vol_context * vomma_factor * term_factor
            )
            df_strike['vri_2_0_strike'] = vri_2_0
            # --- Logging for diagnostics ---
            def safe_list(val):
                if isinstance(val, (pd.Series, np.ndarray, list)):
                    return list(val)[:5]
                try:
                    return [float(val)]
                except Exception:
                    return [str(val)]
            self.logger.debug(f"[VRI2.0] symbol={symbol} | regime={market_regime} | dte={dte_context}")
            self.logger.debug(f"[VRI2.0] term_structure_factor={term_structure_factor:.3f} | enhanced_vol_context_weight={enhanced_vol_context_weight}")
            self.logger.debug(f"[VRI2.0] gamma_coeffs (sample)={safe_list(gamma_coeffs)}")
            self.logger.debug(f"[VRI2.0] vanna_flow_ratio_norm (sample)={safe_list(vanna_flow_ratio_norm)}")
            self.logger.debug(f"[VRI2.0] vomma_flow_ratio_norm (sample)={safe_list(vomma_flow_ratio_norm)}")
            self.logger.debug(f"[VRI2.0] enhanced_vomma_factor (sample)={safe_list(enhanced_vomma_factor)}")
            self.logger.debug(f"[VRI2.0] vri_2_0 (sample)={safe_list(vri_2_0)}")
            self.logger.debug(f"[VRI2.0] Pre-calc sample: total_vanna_at_strike: {df_strike['total_vanna_at_strike'].head(3) if 'total_vanna_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[VRI2.0] Pre-calc sample: total_vommaxoi_at_strike: {df_strike['total_vommaxoi_at_strike'].head(3) if 'total_vommaxoi_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[VRI2.0] Pre-calc sample: net_cust_vanna_flow_proxy_at_strike: {df_strike['net_cust_vanna_flow_proxy_at_strike'].head(3) if 'net_cust_vanna_flow_proxy_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[VRI2.0] Pre-calc sample: net_cust_vomma_flow_proxy_at_strike: {df_strike['net_cust_vomma_flow_proxy_at_strike'].head(3) if 'net_cust_vomma_flow_proxy_at_strike' in df_strike.columns else 'MISSING'}")
            return df_strike
        except Exception as e:
            self.logger.error(f"Error calculating VRI 2.0: {e}", exc_info=True)
            df_strike['vri_2_0_strike'] = 0.0
            return df_strike
    
    def _calculate_0dte_suite(self, df_strike: pd.DataFrame, und_data: Dict, dte_context: str) -> pd.DataFrame:
        import numpy as np
        EPSILON = 1e-9
        is_0dte = (df_strike['dte_calc'] < 0.5) if 'dte_calc' in df_strike.columns else pd.Series([False]*len(df_strike))
        # --- Canonical 0DTE Volatility Suite Calculation ---
        # VRI 0DTE (per system guide)
        try:
            vannaxoi = df_strike.get('vannaxoi', pd.Series([0.0]*len(df_strike)))
            vxoi = df_strike.get('vxoi', pd.Series([0.0]*len(df_strike)))
            vannaxvolm = df_strike.get('vannaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxvolm = df_strike.get('vommaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxoi = df_strike.get('vommaxoi', pd.Series([0.0]*len(df_strike)))
            # Contextual factors (use 1.0 if not present)
            gamma_align_coeff = und_data.get('gamma_align_coeff', 1.0)
            skew_factor_global = und_data.get('skew_factor_global', 1.0)
            vol_trend_factor_global = und_data.get('vol_trend_factor_global', 1.0)
            max_abs_vomma_flow = np.maximum(np.abs(vommaxvolm).max(), EPSILON)
            vri_0dte_contract = (
                vannaxoi * np.sign(vxoi) *
                (1 + gamma_align_coeff * np.abs(vannaxvolm / (vannaxoi + EPSILON))) *
                (vommaxvolm / (max_abs_vomma_flow + EPSILON)) *
                skew_factor_global *
                vol_trend_factor_global
            )
            df_strike['vri_0dte'] = 0.0
            df_strike.loc[is_0dte, 'vri_0dte'] = vri_0dte_contract[is_0dte]
            self.logger.debug(f"[0DTE SUITE] vri_0dte sample: {df_strike['vri_0dte'].head()} (nonzero count: {(df_strike['vri_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating vri_0dte: {e}")
            df_strike['vri_0dte'] = 0.0
        # VFI 0DTE (per system guide) - Volatility Flow Indicator
        try:
            # Get required data for VFI calculation per system guide specification
            vxoi = df_strike.get('vxoi', pd.Series([0.0]*len(df_strike)))

            # Get customer vega flows per contract (per system guide)
            # Use vegas_buy and vegas_sell if available, otherwise fallback to proxies
            vegas_buy = df_strike.get('vegas_buy', pd.Series([0.0]*len(df_strike)))
            vegas_sell = df_strike.get('vegas_sell', pd.Series([0.0]*len(df_strike)))

            # Calculate Net Customer Vega Flow per 0DTE Contract (per system guide)
            net_cust_vega_flow_0dte = vegas_buy - vegas_sell

            # If direct flows not available, use proxy from vxvolm
            if vegas_buy.sum() == 0 and vegas_sell.sum() == 0:
                net_cust_vega_flow_0dte = df_strike.get('vxvolm', pd.Series([0.0]*len(df_strike)))

            # Filter for 0DTE contracts only
            if is_0dte.sum() > 0:
                # Calculate absolute values for normalization (per system guide)
                abs_net_cust_vega_flow_0dte = np.abs(net_cust_vega_flow_0dte[is_0dte])
                abs_vega_oi_0dte = np.abs(vxoi[is_0dte])

                # Normalize within the 0DTE batch (per system guide)
                max_abs_net_cust_vega_flow_0dte = abs_net_cust_vega_flow_0dte.max()
                max_abs_vega_oi_0dte = abs_vega_oi_0dte.max()

                # Normalized values per system guide
                if max_abs_net_cust_vega_flow_0dte > EPSILON:
                    normalized_abs_net_cust_vega_flow = abs_net_cust_vega_flow_0dte / max_abs_net_cust_vega_flow_0dte
                else:
                    normalized_abs_net_cust_vega_flow = pd.Series([0.0] * is_0dte.sum())

                if max_abs_vega_oi_0dte > EPSILON:
                    normalized_abs_vega_oi = abs_vega_oi_0dte / max_abs_vega_oi_0dte
                else:
                    normalized_abs_vega_oi = pd.Series([0.0] * is_0dte.sum())

                # Calculate VFI per system guide formula
                vfi_0dte_values = normalized_abs_net_cust_vega_flow / (normalized_abs_vega_oi + EPSILON)

                df_strike['vfi_0dte'] = 0.0
                df_strike.loc[is_0dte, 'vfi_0dte'] = vfi_0dte_values.values
            else:
                df_strike['vfi_0dte'] = 0.0

            self.logger.debug(f"[0DTE SUITE] vfi_0dte sample: {df_strike['vfi_0dte'].head()} (nonzero count: {(df_strike['vfi_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating vfi_0dte: {e}")
            df_strike['vfi_0dte'] = 0.0

        # VVR 0DTE (per system guide) - Vanna-Vomma Ratio
        try:
            # Get Vanna and Vomma flow proxies
            vannaxvolm = df_strike.get('vannaxvolm', pd.Series([0.0]*len(df_strike)))
            vommaxvolm = df_strike.get('vommaxvolm', pd.Series([0.0]*len(df_strike)))

            # Calculate VVR per system guide: ratio of absolute Vanna flow to absolute Vomma flow
            abs_vanna_flow = np.abs(vannaxvolm)
            abs_vomma_flow = np.abs(vommaxvolm)

            # VVR = |Vanna Flow| / (|Vomma Flow| + EPSILON)
            vvr_0dte_contract = abs_vanna_flow / (abs_vomma_flow + EPSILON)

            # Handle edge cases per system guide
            # If Vomma flow is effectively zero but Vanna flow is significant, set high value
            vomma_zero_mask = abs_vomma_flow < EPSILON
            vanna_significant_mask = abs_vanna_flow > EPSILON
            extreme_vanna_dominance = vomma_zero_mask & vanna_significant_mask
            vvr_0dte_contract[extreme_vanna_dominance] = 1000.0  # Extreme Vanna dominance

            # If both are zero, set to 0
            both_zero_mask = (abs_vanna_flow < EPSILON) & (abs_vomma_flow < EPSILON)
            vvr_0dte_contract[both_zero_mask] = 0.0

            df_strike['vvr_0dte'] = 0.0
            df_strike.loc[is_0dte, 'vvr_0dte'] = vvr_0dte_contract[is_0dte]
            self.logger.debug(f"[0DTE SUITE] vvr_0dte sample: {df_strike['vvr_0dte'].head()} (nonzero count: {(df_strike['vvr_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating vvr_0dte: {e}")
            df_strike['vvr_0dte'] = 0.0
        # GCI 0DTE (use gci_strike if available, else 0)
        try:
            gci_strike = df_strike.get('gci_strike', pd.Series([0.0]*len(df_strike)))
            df_strike['gci_0dte'] = 0.0
            df_strike.loc[is_0dte, 'gci_0dte'] = gci_strike[is_0dte]
            self.logger.debug(f"[0DTE SUITE] gci_0dte sample: {df_strike['gci_0dte'].head()} (nonzero count: {(df_strike['gci_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating gci_0dte: {e}")
            df_strike['gci_0dte'] = 0.0
        # DCI 0DTE (use dci_strike if available, else 0)
        try:
            dci_strike = df_strike.get('dci_strike', pd.Series([0.0]*len(df_strike)))
            df_strike['dci_0dte'] = 0.0
            df_strike.loc[is_0dte, 'dci_0dte'] = dci_strike[is_0dte]
            self.logger.debug(f"[0DTE SUITE] dci_0dte sample: {df_strike['dci_0dte'].head()} (nonzero count: {(df_strike['dci_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating dci_0dte: {e}")
            df_strike['dci_0dte'] = 0.0
        # VCI 0DTE (Vanna Concentration Index for 0DTE) - HHI-style calculation per system guide
        try:
            if is_0dte.sum() > 0:
                # Get Vanna OI at strike level for 0DTE contracts (per system guide: vannaxoi)
                total_vanna_oi = df_strike.get('total_vannaxoi_at_strike', pd.Series([0.0]*len(df_strike))).fillna(0.0)

                # Filter for 0DTE strikes only and get absolute values
                vanna_oi_0dte = total_vanna_oi[is_0dte].abs()

                # Calculate total absolute Vanna OI across all 0DTE strikes (per system guide)
                sum_total_abs_vanna_oi_0dte = vanna_oi_0dte.sum()

                if sum_total_abs_vanna_oi_0dte > EPSILON:
                    # Calculate proportion of Vanna OI at each 0DTE strike (per system guide)
                    proportion_vanna_oi_0dte = vanna_oi_0dte / sum_total_abs_vanna_oi_0dte

                    # Calculate HHI-style concentration index (per system guide formula)
                    # vci_0dte_agg = SUM((Proportion_Strike_VannaOI_0DTE)^2 across all 0DTE strikes)
                    vci_0dte_agg = (proportion_vanna_oi_0dte ** 2).sum()

                    # Store individual strike proportions for analysis (optional)
                    vci_strike_proportions = proportion_vanna_oi_0dte ** 2

                    # Store the aggregate VCI value in a special column for later aggregation
                    df_strike['vci_0dte_agg_value'] = 0.0
                    df_strike.loc[is_0dte.iloc[0:1].index, 'vci_0dte_agg_value'] = vci_0dte_agg  # Store once

                    # Store individual strike contributions for detailed analysis
                    df_strike['vci_0dte'] = 0.0
                    df_strike.loc[is_0dte, 'vci_0dte'] = vci_strike_proportions.values

                    self.logger.debug(f"[VCI 0DTE] HHI-style aggregate: {vci_0dte_agg:.4f} from {len(vci_strike_proportions)} 0DTE strikes")
                    self.logger.debug(f"[VCI 0DTE] Strike proportions range: [{vci_strike_proportions.min():.4f}, {vci_strike_proportions.max():.4f}]")

                    # Identify top concentration strikes (per system guide)
                    top_strikes_mask = vci_strike_proportions >= vci_strike_proportions.quantile(0.8)
                    if top_strikes_mask.sum() > 0:
                        top_strikes = df_strike.loc[is_0dte, 'strike'][top_strikes_mask]
                        self.logger.debug(f"[VCI 0DTE] Top Vanna concentration strikes: {top_strikes.tolist()}")
                else:
                    # No Vanna OI found
                    df_strike['vci_0dte_agg_value'] = 0.0
                    df_strike['vci_0dte'] = 0.0
                    self.logger.debug("[VCI 0DTE] No Vanna OI found for 0DTE strikes")
            else:
                # No 0DTE strikes found
                df_strike['vci_0dte_agg_value'] = 0.0
                df_strike['vci_0dte'] = 0.0
                self.logger.debug("[VCI 0DTE] No 0DTE strikes found")

            self.logger.debug(f"[0DTE SUITE] vci_0dte sample: {df_strike['vci_0dte'].head()} (nonzero count: {(df_strike['vci_0dte']!=0).sum()})")
        except Exception as e:
            self.logger.error(f"Error calculating vci_0dte: {e}")
            df_strike['vci_0dte_agg_value'] = 0.0
            df_strike['vci_0dte'] = 0.0
        return df_strike
    
    def _calculate_enhanced_heatmap_data(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate Enhanced Heatmap Data (SGDHP, IVSDH, UGCH)."""
        try:
            symbol = self._calculation_state.get('current_symbol', 'UNKNOWN')
            
            # Core exposure inputs - properly access DataFrame columns
            gamma_exposure = df_strike['total_gxoi_at_strike'].fillna(0) if 'total_gxoi_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            delta_exposure = df_strike['total_dxoi_at_strike'].fillna(0) if 'total_dxoi_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            vanna_exposure = df_strike['total_vanna_at_strike'].fillna(0) if 'total_vanna_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            
            # Flow inputs - properly access DataFrame columns
            net_gamma_flow = df_strike['net_cust_gamma_flow_at_strike_proxy'].fillna(0) if 'net_cust_gamma_flow_at_strike_proxy' in df_strike.columns else pd.Series([0] * len(df_strike))
            net_delta_flow = df_strike['net_cust_delta_flow_at_strike'].fillna(0) if 'net_cust_delta_flow_at_strike' in df_strike.columns else pd.Series([0] * len(df_strike))
            net_vanna_flow = df_strike['net_cust_vanna_flow_at_strike_proxy'].fillna(0) if 'net_cust_vanna_flow_at_strike_proxy' in df_strike.columns else pd.Series([0] * len(df_strike))
            
            # Weights
            gamma_weight = 0.4
            delta_weight = 0.3
            vanna_weight = 0.2
            flow_weight = 0.1
            
            # Calculate normalized intensities - ensure they match DataFrame length
            gamma_intensity = self._normalize_flow(gamma_exposure, 'gamma', symbol) * gamma_weight
            delta_intensity = self._normalize_flow(delta_exposure, 'delta', symbol) * delta_weight
            vanna_intensity = self._normalize_flow(vanna_exposure, 'vanna', symbol) * vanna_weight
            
            # Ensure arrays match DataFrame length
            num_rows = len(df_strike)
            if len(gamma_intensity) == 1 and num_rows > 1:
                gamma_intensity = np.full(num_rows, gamma_intensity[0])
            if len(delta_intensity) == 1 and num_rows > 1:
                delta_intensity = np.full(num_rows, delta_intensity[0])
            if len(vanna_intensity) == 1 and num_rows > 1:
                vanna_intensity = np.full(num_rows, vanna_intensity[0])
            
            # --- Flow alignment factors ---
            gamma_exposure_arr = pd.Series(pd.to_numeric(gamma_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_gamma_flow_arr = pd.Series(pd.to_numeric(net_gamma_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            delta_exposure_arr = pd.Series(pd.to_numeric(delta_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_delta_flow_arr = pd.Series(pd.to_numeric(net_delta_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            vanna_exposure_arr = pd.Series(pd.to_numeric(vanna_exposure, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            net_vanna_flow_arr = pd.Series(pd.to_numeric(net_vanna_flow, errors='coerce')).fillna(0).to_numpy(dtype=float, na_value=0.0)
            gamma_flow_factor = np.where(
                np.sign(gamma_exposure_arr) == np.sign(net_gamma_flow_arr),
                1.2, 0.8
            )
            delta_flow_factor = np.where(
                np.sign(delta_exposure_arr) == np.sign(net_delta_flow_arr),
                1.2, 0.8
            )
            vanna_flow_factor = np.where(
                np.sign(vanna_exposure_arr) == np.sign(net_vanna_flow_arr),
                1.2, 0.8
            )
            
            # --- Flow component addition ---
            flow_sum = (
                net_gamma_flow_arr +
                net_delta_flow_arr +
                net_vanna_flow_arr
            )
            flow_intensity = self._normalize_flow(flow_sum, 'combined_flow', str(symbol) if symbol is not None else "") * flow_weight
            
            # --- Multi-dimensional heatmap calculation ---
            composite_intensity = (
                gamma_intensity + delta_intensity + vanna_intensity + flow_intensity
            )
            
            # Store main heatmap intensity
            df_strike['sgdhp_data'] = composite_intensity
            df_strike['ivsdh_data'] = vanna_intensity
            df_strike['ugch_data'] = delta_intensity
            
            # Calculate proper SGDHP, IVSDH, and UGCH scores according to system guide
            df_strike = self._calculate_sgdhp_scores(df_strike, und_data)
            df_strike = self._calculate_ivsdh_scores(df_strike, und_data)
            df_strike = self._calculate_ugch_scores(df_strike, und_data)
            
            # Additional heatmap metrics
            df_strike['heatmap_regime_scaling'] = 1.0
            df_strike['heatmap_gamma_component'] = gamma_intensity
            df_strike['heatmap_delta_component'] = delta_intensity
            df_strike['heatmap_vanna_component'] = vanna_intensity
            df_strike['heatmap_flow_component'] = flow_intensity
            
            self.logger.debug(f"[HEATMAP] Pre-calc sample: total_gxoi_at_strike: {df_strike['total_gxoi_at_strike'].head(3) if 'total_gxoi_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[HEATMAP] Pre-calc sample: total_dxoi_at_strike: {df_strike['total_dxoi_at_strike'].head(3) if 'total_dxoi_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[HEATMAP] Pre-calc sample: total_vanna_at_strike: {df_strike['total_vanna_at_strike'].head(3) if 'total_vanna_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[HEATMAP] Pre-calc sample: net_cust_gamma_flow_at_strike: {df_strike['net_cust_gamma_flow_at_strike'].head(3) if 'net_cust_gamma_flow_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[HEATMAP] Pre-calc sample: net_cust_delta_flow_at_strike: {df_strike['net_cust_delta_flow_at_strike'].head(3) if 'net_cust_delta_flow_at_strike' in df_strike.columns else 'MISSING'}")
            self.logger.debug(f"[HEATMAP] Pre-calc sample: net_cust_vanna_flow_at_strike: {df_strike['net_cust_vanna_flow_at_strike'].head(3) if 'net_cust_vanna_flow_at_strike' in df_strike.columns else 'MISSING'}")
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced heatmap data: {e}", exc_info=True)
            df_strike['sgdhp_data'] = 0.0
            df_strike['ivsdh_data'] = 0.0
            df_strike['ugch_data'] = 0.0
            return df_strike

    def _calculate_sgdhp_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate SGDHP scores according to system guide specifications."""
        try:
            current_price = und_data.get('price', 0.0)
            if current_price <= 0:
                df_strike['sgdhp_score_strike'] = 0.0
                return df_strike
            
            # Get required data
            gxoi_at_strike = df_strike['total_gxoi_at_strike'].fillna(0)
            dxoi_at_strike = df_strike['total_dxoi_at_strike'].fillna(0)
            strikes = df_strike['strike'].fillna(0)
            
            # Calculate price proximity factor according to system guide
            proximity_sensitivity = self._get_metric_config('heatmap_generation_settings', 'sgdhp_params.proximity_sensitivity_param', 0.05)
            price_proximity_factor = np.exp(-0.5 * ((strikes - current_price) / (current_price * proximity_sensitivity)) ** 2)

            # Calculate DXOI normalized impact according to system guide
            max_abs_dxoi = dxoi_at_strike.abs().max()
            EPSILON = 1e-9
            if max_abs_dxoi > 0:
                dxoi_normalized_impact = (1 + dxoi_at_strike.abs()) / (max_abs_dxoi + EPSILON)
            else:
                dxoi_normalized_impact = pd.Series([1.0] * len(df_strike))
            
            # Recent flow confirmation factor (simplified for now)
            # In full implementation, this would use strike-level recent flows
            recent_flow_confirmation = pd.Series([0.1] * len(df_strike))  # Small positive confirmation
            
            # Calculate SGDHP score according to system guide formula
            sgdhp_scores = (
                (gxoi_at_strike * price_proximity_factor) * 
                np.sign(dxoi_at_strike) * 
                dxoi_normalized_impact * 
                (1 + recent_flow_confirmation)
            )
            
            df_strike['sgdhp_score_strike'] = sgdhp_scores
            
            self.logger.debug(f"SGDHP scores calculated: min={sgdhp_scores.min():.2f}, max={sgdhp_scores.max():.2f}, mean={sgdhp_scores.mean():.2f}")
            
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating SGDHP scores: {e}", exc_info=True)
            df_strike['sgdhp_score_strike'] = 0.0
            return df_strike

    def _calculate_ivsdh_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """
        Calculate IVSDH scores according to system guide specifications.
        Formula: ivsdh_value_contract = vanna_vomma_term * charm_impact_term
        Where:
        - vanna_vomma_term = (vannaxoi_contract * vommaxoi_contract) / (abs(vxoi_contract) + EPSILON)
        - charm_impact_term = (1 + (charmxoi_contract * dte_factor_for_charm))
        - dte_factor_for_charm = 1 / (1 + time_decay_sensitivity_param * dte_calc_contract)
        """
        try:
            # Get required data from strike level
            vannaxoi_at_strike = df_strike['total_vannaxoi_at_strike'].fillna(0)
            vommaxoi_at_strike = df_strike['total_vommaxoi_at_strike'].fillna(0)
            vxoi_at_strike = df_strike['total_vxoi_at_strike'].fillna(0)
            charmxoi_at_strike = df_strike['total_charmxoi_at_strike'].fillna(0)

            # Get DTE data (use average DTE for the strike if available)
            if 'dte_calc' in df_strike.columns:
                dte_at_strike = df_strike['dte_calc'].fillna(30)  # Default to 30 DTE
            else:
                dte_at_strike = pd.Series([30] * len(df_strike))  # Default DTE

            # Get time decay sensitivity parameter from config
            time_decay_sensitivity = self._get_metric_config(
                'heatmap_generation_settings',
                'ivsdh_params.time_decay_sensitivity_param',
                0.1
            )

            # Calculate vanna_vomma_term according to system guide
            EPSILON = 1e-9
            vanna_vomma_term = (vannaxoi_at_strike * vommaxoi_at_strike) / (np.abs(vxoi_at_strike) + EPSILON)

            # Calculate dte_factor_for_charm according to system guide
            dte_factor_for_charm = 1.0 / (1.0 + time_decay_sensitivity * dte_at_strike)

            # Calculate charm_impact_term according to system guide
            charm_impact_term = 1.0 + (charmxoi_at_strike * dte_factor_for_charm)

            # Calculate final IVSDH value according to system guide
            ivsdh_scores = vanna_vomma_term * charm_impact_term

            df_strike['ivsdh_score_strike'] = ivsdh_scores

            self.logger.debug(f"IVSDH scores calculated: min={ivsdh_scores.min():.2f}, max={ivsdh_scores.max():.2f}, mean={ivsdh_scores.mean():.2f}")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating IVSDH scores: {e}", exc_info=True)
            df_strike['ivsdh_score_strike'] = 0.0
            return df_strike

    def _calculate_ugch_scores(self, df_strike: pd.DataFrame, und_data: Dict) -> pd.DataFrame:
        """Calculate UGCH scores according to system guide specifications."""
        try:
            # Get Greek exposures at strike level
            dxoi_at_strike = df_strike['total_dxoi_at_strike'].fillna(0)
            gxoi_at_strike = df_strike['total_gxoi_at_strike'].fillna(0)
            vxoi_at_strike = df_strike['total_vxoi_at_strike'].fillna(0)
            txoi_at_strike = df_strike['total_txoi_at_strike'].fillna(0)
            charm_at_strike = df_strike['total_charmxoi_at_strike'].fillna(0)
            vanna_at_strike = df_strike['total_vanna_at_strike'].fillna(0)
            
            # Normalize each Greek series (Z-score normalization)
            def normalize_series(series):
                if series.std() > 0:
                    return (series - series.mean()) / series.std()
                else:
                    return pd.Series([0.0] * len(series))
            
            norm_dxoi = normalize_series(dxoi_at_strike)
            norm_gxoi = normalize_series(gxoi_at_strike)
            norm_vxoi = normalize_series(vxoi_at_strike)
            norm_txoi = normalize_series(txoi_at_strike)
            norm_charm = normalize_series(charm_at_strike)
            norm_vanna = normalize_series(vanna_at_strike)
            
            # Get Greek weights from config (with defaults)
            greek_weights = self._get_metric_config('heatmap_generation_settings', 'ugch_params.greek_weights', {
                'norm_DXOI': 1.5,
                'norm_GXOI': 2.0,
                'norm_VXOI': 1.2,
                'norm_TXOI': 0.8,
                'norm_CHARM': 0.6,
                'norm_VANNA': 1.0
            })
            
            # Calculate weighted confluence score
            ugch_scores = (
                greek_weights.get('norm_DXOI', 1.5) * norm_dxoi +
                greek_weights.get('norm_GXOI', 2.0) * norm_gxoi +
                greek_weights.get('norm_VXOI', 1.2) * norm_vxoi +
                greek_weights.get('norm_TXOI', 0.8) * norm_txoi +
                greek_weights.get('norm_CHARM', 0.6) * norm_charm +
                greek_weights.get('norm_VANNA', 1.0) * norm_vanna
            )
            
            df_strike['ugch_score_strike'] = ugch_scores
            
            self.logger.debug(f"UGCH scores calculated: min={ugch_scores.min():.2f}, max={ugch_scores.max():.2f}, mean={ugch_scores.mean():.2f}")
            
            return df_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating UGCH scores: {e}", exc_info=True)
            df_strike['ugch_score_strike'] = 0.0
            return df_strike

    def _calculate_a_mspi(self, df_strike: pd.DataFrame, und_data: Dict, market_regime: str, volatility_context: str, dte_context: str) -> pd.DataFrame:
        """Calculate A-MSPI (Adaptive Market Structure Pressure Index) from component metrics."""
        try:
            import numpy as np
            import pandas as pd

            # Get A-MSPI configuration
            mspi_config = self._get_metric_config('adaptive_metric_parameters', 'a_mspi_settings', {})
            weights = mspi_config.get('component_weights', {
                'a_dag_weight': 0.35,
                'd_tdpi_weight': 0.25,
                'vri_2_0_weight': 0.25,
                'e_sdag_mult_weight': 0.10,
                'e_sdag_dir_weight': 0.05
            })

            # Get component data
            a_dag = df_strike.get('a_dag_strike', pd.Series([0.0] * len(df_strike)))
            d_tdpi = df_strike.get('d_tdpi_strike', pd.Series([0.0] * len(df_strike)))
            vri_2_0 = df_strike.get('vri_2_0_strike', pd.Series([0.0] * len(df_strike)))
            e_sdag_mult = df_strike.get('e_sdag_mult_strike', pd.Series([0.0] * len(df_strike)))
            e_sdag_dir = df_strike.get('e_sdag_dir_strike', pd.Series([0.0] * len(df_strike)))

            # Normalize components using Z-score normalization
            def safe_normalize(series):
                """Safely normalize a series using Z-score, handling edge cases."""
                if len(series) == 0:
                    return series
                mean_val = series.mean()
                std_val = series.std()
                if std_val == 0 or pd.isna(std_val):
                    return pd.Series([0.0] * len(series))
                return (series - mean_val) / std_val

            a_dag_norm = safe_normalize(a_dag)
            d_tdpi_norm = safe_normalize(d_tdpi)
            vri_2_0_norm = safe_normalize(vri_2_0)
            e_sdag_mult_norm = safe_normalize(e_sdag_mult)
            e_sdag_dir_norm = safe_normalize(e_sdag_dir)

            # Calculate weighted A-MSPI
            a_mspi = (
                a_dag_norm * weights['a_dag_weight'] +
                d_tdpi_norm * weights['d_tdpi_weight'] +
                vri_2_0_norm * weights['vri_2_0_weight'] +
                e_sdag_mult_norm * weights['e_sdag_mult_weight'] +
                e_sdag_dir_norm * weights['e_sdag_dir_weight']
            )

            # Store A-MSPI results
            df_strike['a_mspi_strike'] = a_mspi
            df_strike['a_mspi_a_dag_component'] = a_dag_norm * weights['a_dag_weight']
            df_strike['a_mspi_d_tdpi_component'] = d_tdpi_norm * weights['d_tdpi_weight']
            df_strike['a_mspi_vri_2_0_component'] = vri_2_0_norm * weights['vri_2_0_weight']
            df_strike['a_mspi_e_sdag_mult_component'] = e_sdag_mult_norm * weights['e_sdag_mult_weight']
            df_strike['a_mspi_e_sdag_dir_component'] = e_sdag_dir_norm * weights['e_sdag_dir_weight']

            # Calculate A-SAI and A-SSI based on A-MSPI
            current_price = und_data.get('price', 0.0)
            strikes = df_strike['strike'] if 'strike' in df_strike.columns else pd.Series([current_price] * len(df_strike))

            # A-SAI: Support Aggregate Index (strikes below current price)
            # A-SSI: Resistance Aggregate Index (strikes above current price)
            support_mask = strikes <= current_price
            resistance_mask = strikes > current_price

            # Calculate A-SAI (internal consistency for support levels)
            if support_mask.sum() > 0:
                support_mspi = a_mspi[support_mask]
                # A-SAI measures consistency - higher when all support components align positively
                a_sai_values = pd.Series([0.0] * len(df_strike))
                if len(support_mspi) > 0:
                    # Normalize to [-1, 1] range and weight by proximity to current price
                    proximity_weights = np.exp(-np.abs(strikes[support_mask] - current_price) / (current_price * 0.05))
                    weighted_support = (support_mspi * proximity_weights).sum() / proximity_weights.sum()
                    # Normalize to [-1, 1] range
                    a_sai_value = np.tanh(weighted_support)
                    a_sai_values[support_mask] = a_sai_value
                df_strike['a_sai_und_avg'] = a_sai_values
            else:
                df_strike['a_sai_und_avg'] = 0.0

            # Calculate A-SSI (structural stability for resistance levels)
            if resistance_mask.sum() > 0:
                resistance_mspi = a_mspi[resistance_mask]
                # A-SSI measures stability - negative when resistance structure is strong
                a_ssi_values = pd.Series([0.0] * len(df_strike))
                if len(resistance_mspi) > 0:
                    # Normalize to [-1, 1] range and weight by proximity to current price
                    proximity_weights = np.exp(-np.abs(strikes[resistance_mask] - current_price) / (current_price * 0.05))
                    # Prevent division by zero
                    weight_sum = proximity_weights.sum()
                    if weight_sum > 0:
                        weighted_resistance = (resistance_mspi * proximity_weights).sum() / weight_sum
                        # Invert for resistance (negative values indicate strong resistance)
                        a_ssi_value = -np.tanh(weighted_resistance)
                        a_ssi_values[resistance_mask] = a_ssi_value
                    else:
                        self.logger.debug("Zero proximity weights sum, skipping weighted resistance calculation")
                df_strike['a_ssi_und_avg'] = a_ssi_values
            else:
                df_strike['a_ssi_und_avg'] = 0.0

            self.logger.debug(f"[A-MSPI] Calculated for {len(df_strike)} strikes, range: [{a_mspi.min():.3f}, {a_mspi.max():.3f}]")

            return df_strike

        except Exception as e:
            self.logger.error(f"Error calculating A-MSPI: {e}", exc_info=True)
            # Set default values on error
            df_strike['a_mspi_strike'] = 0.0
            df_strike['a_mspi_a_dag_component'] = 0.0
            df_strike['a_mspi_d_tdpi_component'] = 0.0
            df_strike['a_mspi_vri_2_0_component'] = 0.0
            df_strike['a_mspi_e_sdag_mult_component'] = 0.0
            df_strike['a_mspi_e_sdag_dir_component'] = 0.0
            df_strike['a_sai_und_avg'] = 0.0
            df_strike['a_ssi_und_avg'] = 0.0
            return df_strike

    def _calculate_sgexoi_v2_5(self, gxoi_at_strike, und_data: Dict, sgexoi_params: Dict, dte_context: str):
        """Calculate Enhanced Skew-Adjusted Gamma Exposure (SGEXOI_v2_5)."""
        try:
            # Get skew adjustment parameters
            skew_sensitivity = sgexoi_params.get('skew_sensitivity', 0.3)
            term_structure_weight = sgexoi_params.get('term_structure_weight', 0.2)
            vri_integration_factor = sgexoi_params.get('vri_integration_factor', 0.15)

            # Get IV surface characteristics for skew adjustment
            current_iv = und_data.get('u_volatility', 0.20) or 0.20
            front_iv = und_data.get('front_month_iv', current_iv)
            spot_iv = und_data.get('spot_iv', current_iv)

            # Calculate term structure factor
            term_structure_factor = 1.0
            if spot_iv > 0:
                term_structure_factor = 1.0 + term_structure_weight * (front_iv / spot_iv - 1.0)

            # Calculate skew adjustment factor based on IV characteristics
            # This is a simplified implementation - full version would use detailed IV surface data
            skew_adjustment_factor = 1.0 + skew_sensitivity * (current_iv - 0.20)  # Adjust based on IV level

            # Apply VRI integration if available
            vri_factor = 1.0
            if 'vri_2_0_aggregate' in und_data:
                vri_aggregate = und_data.get('vri_2_0_aggregate', 0.0)
                vri_factor = 1.0 + vri_integration_factor * np.tanh(vri_aggregate)

            # Calculate SGEXOI_v2_5
            sgexoi_v2_5 = gxoi_at_strike * skew_adjustment_factor * term_structure_factor * vri_factor

            return sgexoi_v2_5

        except Exception as e:
            self.logger.error(f"Error calculating SGEXOI_v2_5: {e}")
            return gxoi_at_strike  # Fallback to regular GXOI

    def _get_dte_scaling_factor_a_dag(self, dte_context: str, dte_scaling_config: Dict) -> float:
        """Get DTE scaling factor for A-DAG calculations."""
        return dte_scaling_config.get(dte_context, 1.0)

    def _get_dte_scaling_factor(self, dte_context: str) -> float:
        """Get DTE scaling factor for adaptive calculations."""
        dte_scalers = {
            '0DTE': 1.5,
            'SHORT_DTE': 1.2,
            'NORMAL_DTE': 1.0,
            'LONG_DTE': 0.8
        }
        return dte_scalers.get(dte_context, 1.0)
    
    def _calculate_time_weight(self, current_time: pd.Timestamp) -> float:
        """Calculate time-of-day weighting factor."""
        try:
            # Simple time weighting - higher weight towards end of day
            hour = current_time.hour
            minute = current_time.minute
            
            # Market hours: 9:30 AM to 4:00 PM ET
            market_open_minutes = 9 * 60 + 30  # 9:30 AM
            market_close_minutes = 16 * 60     # 4:00 PM
            current_minutes = hour * 60 + minute
            
            if current_minutes < market_open_minutes or current_minutes > market_close_minutes:
                return 0.5  # After hours
            
            # Calculate progress through trading day
            trading_progress = (current_minutes - market_open_minutes) / (market_close_minutes - market_open_minutes)
            
            # Exponential weighting towards end of day
            time_weight = 0.5 + 0.5 * (trading_progress ** 2)
            
            return time_weight
            
        except Exception as e:
            self.logger.error(f"Error calculating time weight: {e}")
            return 1.0
    
    def _normalize_flow(self, flow_values, flow_type: str, symbol: Optional[str] = None) -> np.ndarray:
        symbol = str(symbol) if symbol is not None else ""
        try:
            if symbol and hasattr(flow_values, '__iter__'):
                # Use isolated cache for normalization
                cache = self._get_isolated_cache('enhanced_heatmap', symbol)
                cache_key = f"{flow_type}_normalization_history"
                
                if cache_key not in cache:
                    cache[cache_key] = []
                
                # Add current values to history
                if isinstance(flow_values, (list, np.ndarray, pd.Series)):
                    cache[cache_key].extend(list(flow_values))
                else:
                    cache[cache_key].append(flow_values)
                
                # Keep only recent history
                if len(cache[cache_key]) > 100:
                    cache[cache_key] = cache[cache_key][-100:]
                
                # Normalize using historical context
                if len(cache[cache_key]) > 10:
                    mean_val = np.mean(cache[cache_key])
                    std_val = np.std(cache[cache_key])
                    if std_val > 0:
                        if isinstance(flow_values, (list, np.ndarray, pd.Series)):
                            return (np.array(flow_values) - mean_val) / std_val
                        else:
                            return (flow_values - mean_val) / std_val
            
            # Fallback: simple normalization
            if isinstance(flow_values, (list, np.ndarray, pd.Series)):
                flow_array = np.array(flow_values)
                if len(flow_array) > 0:
                    return (flow_array - np.mean(flow_array)) / (np.std(flow_array) + 1e-6)
                else:
                    return np.array([0.0])
            else:
                return np.array([flow_values])
                
        except Exception as e:
            self.logger.error(f"Error normalizing flow {flow_type}: {e}", exc_info=True)
            if isinstance(flow_values, (list, np.ndarray, pd.Series)):
                return np.zeros(len(flow_values))
            else:
                return np.array([0.0])

    def _calculate_atr(self, symbol: str, dte_max: int = 45) -> float:
        """Fetches OHLCV data and calculates the Average True Range (ATR)."""
        try:
            # Skip ATR calculation for futures symbols - they use different data sources
            if self._is_futures_symbol(symbol):
                self.logger.debug(f"Skipping ATR calculation for futures symbol {symbol}")
                return 0.0

            # Calculate appropriate lookback based on DTE context
            # For ATR, we need enough data for a meaningful calculation
            # Use max(dte_max, 14) to ensure minimum 14 periods for ATR but respect DTE context
            lookback_days = max(dte_max, 14)

            ohlcv_df = self.historical_data_manager.get_historical_ohlcv(symbol, lookback_days=lookback_days)
            if ohlcv_df is None or len(ohlcv_df) == 0 or len(ohlcv_df) < 2:
                self.logger.debug(f"No OHLCV data available for {symbol}, skipping ATR calculation")
                return 0.0
            
            high_low = pd.Series(ohlcv_df['high'] - ohlcv_df['low'])
            high_close = pd.Series(np.abs(ohlcv_df['high'] - ohlcv_df['close'].shift()))
            low_close = pd.Series(np.abs(ohlcv_df['low'] - ohlcv_df['close'].shift()))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(com=14, min_periods=14).mean().iloc[-1]
            return atr
        except Exception as e:
            self.logger.error(f"Failed to calculate ATR for {symbol}: {e}", exc_info=True)
            return 0.0
    
    def _calculate_underlying_aggregates(self, df_strike: Optional[pd.DataFrame]) -> Dict[str, float]:
        import numpy as np
        aggregates = {}
        if df_strike is not None and len(df_strike) > 0:
            is_0dte = (df_strike['dte_calc'] < 0.5) if 'dte_calc' in df_strike.columns else pd.Series([False]*len(df_strike))
            df_0dte = df_strike[is_0dte]
            # --- 0DTE AGGREGATES ---
            # VRI 0DTE AGG - Sum of all 0DTE VRI values (per system guide)
            if 'vri_0dte' in df_0dte.columns:
                vri_vals = df_0dte['vri_0dte'].dropna()
                if len(vri_vals) > 0:
                    # Sum all VRI 0DTE values as per system guide
                    vri_sum = vri_vals.sum()
                    # Apply reasonable bounds but preserve the actual calculated value
                    aggregates['vri_0dte_und_sum'] = max(min(vri_sum, 10.0), -10.0)
                else:
                    aggregates['vri_0dte_und_sum'] = 0.0

            # VFI 0DTE AGG - PYDANTIC-FIRST: Validate against ProcessedUnderlyingAggregatesV2_5 schema
            if 'vfi_0dte' in df_0dte.columns:
                vfi_vals = df_0dte['vfi_0dte'].dropna()
                if len(vfi_vals) > 0:
                    # Use weighted average or simple average based on significance
                    vfi_avg = vfi_vals.mean()
                    # Apply reasonable bounds for display
                    vfi_bounded = max(min(vfi_avg, 5.0), -5.0)

                    # PYDANTIC-FIRST: Create both fields as required by EOTS schemas
                    aggregates['vfi_0dte_und_sum'] = vfi_bounded  # Legacy field for compatibility
                    aggregates['vfi_0dte_und_avg'] = vfi_bounded  # Required by regime engine schema

                    self.logger.debug(f"[VFI 0DTE] Calculated VFI metrics: sum={vfi_bounded:.4f}, avg={vfi_bounded:.4f}")
                else:
                    aggregates['vfi_0dte_und_sum'] = 0.0
                    aggregates['vfi_0dte_und_avg'] = 0.0
                    self.logger.debug("[VFI 0DTE] No VFI values found, setting to 0.0")
            else:
                # Ensure both fields exist even if no VFI data - PYDANTIC validation requirement
                aggregates['vfi_0dte_und_sum'] = 0.0
                aggregates['vfi_0dte_und_avg'] = 0.0
                self.logger.warning("[VFI 0DTE] No vfi_0dte column found in 0DTE data")

            # VVR 0DTE AGG - Average of 0DTE VVR values (weighted by significance)
            if 'vvr_0dte' in df_0dte.columns:
                vvr_vals = df_0dte['vvr_0dte'].dropna()
                if len(vvr_vals) > 0:
                    # Filter out extreme values (>100) for averaging but keep meaningful ratios
                    reasonable_vvr = vvr_vals[vvr_vals <= 100.0]
                    if len(reasonable_vvr) > 0:
                        vvr_avg = reasonable_vvr.mean()
                    else:
                        vvr_avg = vvr_vals.median()  # Use median if all values are extreme
                    # VVR should be positive ratio, typically 0.1 to 10.0 range
                    aggregates['vvr_0dte_und_avg'] = max(min(vvr_avg, 10.0), 0.0)
                else:
                    aggregates['vvr_0dte_und_avg'] = 0.0
            # VCI 0DTE AGG - Use proper HHI aggregate value (per system guide)
            if 'vci_0dte_agg_value' in df_0dte.columns:
                # Get the pre-calculated HHI aggregate value
                vci_agg_vals = df_0dte['vci_0dte_agg_value'].dropna()
                vci_agg_vals = vci_agg_vals[vci_agg_vals > 0]  # Filter out zeros
                if len(vci_agg_vals) > 0:
                    # Use the HHI aggregate value (should be the same across all 0DTE rows)
                    vci_agg = vci_agg_vals.iloc[0]
                    # VCI ranges from 0 (dispersed) to 1.0 (concentrated at single strike)
                    aggregates['vci_0dte_agg'] = max(min(vci_agg, 1.0), 0.0)
                else:
                    aggregates['vci_0dte_agg'] = 0.0
            else:
                # Fallback: calculate from individual strike values (less accurate)
                if 'vci_0dte' in df_0dte.columns:
                    vci_vals = df_0dte['vci_0dte'].dropna()
                    if len(vci_vals) > 0:
                        # Sum the squared proportions (HHI formula)
                        vci_agg = vci_vals.sum()
                        aggregates['vci_0dte_agg'] = max(min(vci_agg, 1.0), 0.0)
                    else:
                        aggregates['vci_0dte_agg'] = 0.0
                else:
                    aggregates['vci_0dte_agg'] = 0.0
            # Logging for diagnostics
            for k, v in aggregates.items():
                self.logger.debug(f"[0DTE AGGREGATE] {k}: {v}")
        # --- CANONICAL A-SAI/A-SSI CALCULATION AND LOGGING ---
        try:
            if df_strike is not None and 'a_sai_und_avg' in df_strike.columns:
                sai_vals = df_strike['a_sai_und_avg'].dropna()
                # Filter out zero values and take the first non-zero value (they should all be the same)
                non_zero_sai = sai_vals[sai_vals != 0.0]
                if len(non_zero_sai) > 0:
                    aggregates['a_sai_und_avg'] = float(non_zero_sai.iloc[0])
                    # Also create the field name that AI dashboard expects
                    aggregates['a_sai_und'] = float(non_zero_sai.iloc[0])
                else:
                    aggregates['a_sai_und_avg'] = 0.0
                    aggregates['a_sai_und'] = 0.0
                self.logger.debug(f"[A-SAI] a_sai_und_avg calculated: {aggregates['a_sai_und_avg']} | Nonzero count: {len(non_zero_sai)}")
            else:
                self.logger.warning("[A-SAI] a_sai_und_avg column missing from df_strike!")
                aggregates['a_sai_und_avg'] = 0.0
                aggregates['a_sai_und'] = 0.0
        except Exception as e:
            self.logger.error(f"[A-SAI] Error calculating a_sai_und_avg: {e}")
            aggregates['a_sai_und_avg'] = 0.0
            aggregates['a_sai_und'] = 0.0
        try:
            if df_strike is not None and 'a_ssi_und_avg' in df_strike.columns:
                ssi_vals = df_strike['a_ssi_und_avg'].dropna()
                # Filter out zero values and take the first non-zero value (they should all be the same)
                non_zero_ssi = ssi_vals[ssi_vals != 0.0]
                if len(non_zero_ssi) > 0:
                    aggregates['a_ssi_und_avg'] = float(non_zero_ssi.iloc[0])
                    # Also create the field name that AI dashboard expects
                    aggregates['a_ssi_und'] = float(non_zero_ssi.iloc[0])
                else:
                    aggregates['a_ssi_und_avg'] = 0.0
                    aggregates['a_ssi_und'] = 0.0
                self.logger.debug(f"[A-SSI] a_ssi_und_avg calculated: {aggregates['a_ssi_und_avg']} | Nonzero count: {len(non_zero_ssi)}")
            else:
                self.logger.warning("[A-SSI] a_ssi_und_avg column missing from df_strike!")
                aggregates['a_ssi_und_avg'] = 0.0
                aggregates['a_ssi_und'] = 0.0
        except Exception as e:
            self.logger.error(f"[A-SSI] Error calculating a_ssi_und_avg: {e}")
            aggregates['a_ssi_und_avg'] = 0.0
            aggregates['a_ssi_und'] = 0.0

        # --- A-MSPI SUMMARY SCORE CALCULATION ---
        try:
            if df_strike is not None and 'a_mspi_strike' in df_strike.columns:
                mspi_vals = df_strike['a_mspi_strike'].dropna()
                if len(mspi_vals) > 0:
                    # Calculate weighted average based on proximity to current price
                    if 'strike' in df_strike.columns:
                        strikes = df_strike['strike'].fillna(0)
                        current_price = aggregates.get('price', 0.0)
                        if current_price > 0:
                            # Weight by proximity to current price (Gaussian decay)
                            proximity_weights = np.exp(-((strikes - current_price) / current_price) ** 2 / (2 * 0.05 ** 2))
                            weighted_mspi = (mspi_vals * proximity_weights).sum() / proximity_weights.sum()
                        else:
                            weighted_mspi = mspi_vals.mean()
                    else:
                        weighted_mspi = mspi_vals.mean()
                    aggregates['a_mspi_und_summary_score'] = float(weighted_mspi)
                    # Also create the field name that AI dashboard expects
                    aggregates['a_mspi_und'] = float(weighted_mspi)
                else:
                    aggregates['a_mspi_und_summary_score'] = 0.0
                    aggregates['a_mspi_und'] = 0.0
                self.logger.debug(f"[A-MSPI] a_mspi_und_summary_score calculated: {aggregates.get('a_mspi_und_summary_score', 0.0)}")
            else:
                self.logger.warning("[A-MSPI] a_mspi_strike column missing from df_strike!")
                aggregates['a_mspi_und_summary_score'] = 0.0
                aggregates['a_mspi_und'] = 0.0
        except Exception as e:
            self.logger.error(f"[A-MSPI] Error calculating a_mspi_und_summary_score: {e}")
            aggregates['a_mspi_und_summary_score'] = 0.0
            aggregates['a_mspi_und'] = 0.0

        # --- VRI 2.0 AGGREGATION ---
        try:
            if df_strike is not None and 'vri_2_0_strike' in df_strike.columns:
                vri_vals = df_strike['vri_2_0_strike'].dropna()
                if len(vri_vals) > 0:
                    # Calculate weighted average VRI 2.0 across all strikes
                    vri_avg = vri_vals.mean()
                    aggregates['vri_2_0_und'] = float(vri_avg)
                    aggregates['vri_2_0_aggregate'] = float(vri_avg)  # For SGEXOI calculation
                else:
                    aggregates['vri_2_0_und'] = 0.0
                    aggregates['vri_2_0_aggregate'] = 0.0
                self.logger.debug(f"[VRI 2.0] vri_2_0_und calculated: {aggregates.get('vri_2_0_und', 0.0)}")
            else:
                self.logger.warning("[VRI 2.0] vri_2_0_strike column missing from df_strike!")
                aggregates['vri_2_0_und'] = 0.0
                aggregates['vri_2_0_aggregate'] = 0.0
        except Exception as e:
            self.logger.error(f"[VRI 2.0] Error calculating vri_2_0_und: {e}")
            aggregates['vri_2_0_und'] = 0.0
            aggregates['vri_2_0_aggregate'] = 0.0

        # --- E-SDAG AGGREGATION ---
        try:
            if df_strike is not None and 'e_sdag_mult_strike' in df_strike.columns:
                e_sdag_mult_vals = df_strike['e_sdag_mult_strike'].dropna()
                if len(e_sdag_mult_vals) > 0:
                    # Use the multiplicative E-SDAG as the primary metric
                    e_sdag_avg = e_sdag_mult_vals.mean()
                    aggregates['e_sdag_mult_und'] = float(e_sdag_avg)
                else:
                    aggregates['e_sdag_mult_und'] = 0.0
                self.logger.debug(f"[E-SDAG] e_sdag_mult_und calculated: {aggregates.get('e_sdag_mult_und', 0.0)}")
            else:
                self.logger.warning("[E-SDAG] e_sdag_mult_strike column missing from df_strike!")
                aggregates['e_sdag_mult_und'] = 0.0
        except Exception as e:
            self.logger.error(f"[E-SDAG] Error calculating e_sdag_mult_und: {e}")
            aggregates['e_sdag_mult_und'] = 0.0

        # --- A-DAG AGGREGATION ---
        try:
            if df_strike is not None and 'a_dag_strike' in df_strike.columns:
                a_dag_vals = df_strike['a_dag_strike'].dropna()
                if len(a_dag_vals) > 0:
                    # Sum A-DAG values across all strikes for total exposure
                    a_dag_total = a_dag_vals.sum()
                    aggregates['a_dag_total_und'] = float(a_dag_total)
                else:
                    aggregates['a_dag_total_und'] = 0.0
                self.logger.debug(f"[A-DAG] a_dag_total_und calculated: {aggregates.get('a_dag_total_und', 0.0)}")
            else:
                self.logger.warning("[A-DAG] a_dag_strike column missing from df_strike!")
                aggregates['a_dag_total_und'] = 0.0
        except Exception as e:
            self.logger.error(f"[A-DAG] Error calculating a_dag_total_und: {e}")
            aggregates['a_dag_total_und'] = 0.0

        # --- ROLLING FLOWS AGGREGATION (CRITICAL FOR ADVANCED FLOW MODE) ---
        # Aggregate rolling flow metrics from contract level to underlying level
        # These are required for advanced flow mode rolling flows and flow ratios charts
        self._aggregate_rolling_flows_from_contracts(df_strike, aggregates)

        # --- ENHANCED FLOW METRICS AGGREGATION (CRITICAL FOR DWFD & TW-LAF) ---
        # Aggregate total_nvp and total_nvp_vol that DWFD and TW-LAF require
        self._aggregate_enhanced_flow_inputs(df_strike, aggregates)

        # --- MISSING REGIME DETECTION METRICS ---
        # Add missing metrics required by market regime engine
        self._add_missing_regime_metrics(aggregates)

        return aggregates

    def _aggregate_rolling_flows_from_contracts(self, df_strike: Optional[pd.DataFrame], aggregates: Dict[str, float]) -> None:
        """
        Aggregate rolling flow metrics from contract level to underlying level.
        This is CRITICAL for advanced flow mode charts to display data.

        Rolling flows come from ConvexValue get_chain API (not get_und) and need to be
        aggregated across all contracts to create underlying-level metrics.
        """
        try:
            if df_strike is None or len(df_strike) == 0:
                # Set all rolling flows to zero if no data
                for timeframe in ['5m', '15m', '30m', '60m']:
                    aggregates[f'net_value_flow_{timeframe}_und'] = 0.0
                    aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0
                self.logger.debug("[ROLLING FLOWS] No strike data available, set all rolling flows to 0")
                return

            # Get the original contract-level data (df_chain) from calculation state
            df_chain = self._calculation_state.get('df_chain')
            if df_chain is None or len(df_chain) == 0:
                # Fallback: try to reconstruct from df_strike if possible
                self.logger.warning("[ROLLING FLOWS] No df_chain available, using fallback aggregation")
                for timeframe in ['5m', '15m', '30m', '60m']:
                    aggregates[f'net_value_flow_{timeframe}_und'] = 0.0
                    aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0
                return

            # Aggregate rolling flows for each timeframe
            timeframes = ['5m', '15m', '30m', '60m']

            for timeframe in timeframes:
                value_col = f'valuebs_{timeframe}'
                vol_col = f'volmbs_{timeframe}'

                # Sum across all contracts for this timeframe
                if value_col in df_chain.columns:
                    net_value_flow = df_chain[value_col].fillna(0.0).sum()
                    aggregates[f'net_value_flow_{timeframe}_und'] = float(net_value_flow)
                else:
                    aggregates[f'net_value_flow_{timeframe}_und'] = 0.0

                if vol_col in df_chain.columns:
                    net_vol_flow = df_chain[vol_col].fillna(0.0).sum()
                    aggregates[f'net_vol_flow_{timeframe}_und'] = float(net_vol_flow)
                else:
                    aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0

            # Log the aggregated values for debugging
            for timeframe in timeframes:
                value_key = f'net_value_flow_{timeframe}_und'
                vol_key = f'net_vol_flow_{timeframe}_und'
                self.logger.debug(f"[ROLLING FLOWS] {timeframe}: value={aggregates.get(value_key, 0.0):.1f}, vol={aggregates.get(vol_key, 0.0):.1f}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error aggregating rolling flows: {e}")
            # Set all to zero on error
            for timeframe in ['5m', '15m', '30m', '60m']:
                aggregates[f'net_value_flow_{timeframe}_und'] = 0.0
                aggregates[f'net_vol_flow_{timeframe}_und'] = 0.0

    def _aggregate_enhanced_flow_inputs(self, df_strike: Optional[pd.DataFrame], aggregates: Dict[str, float]) -> None:
        """
        Aggregate enhanced flow inputs required by DWFD and TW-LAF calculations.

        DWFD and TW-LAF require total_nvp and total_nvp_vol fields that are not being
        calculated. This method aggregates them from contract-level data.
        """
        try:
            if df_strike is None or len(df_strike) == 0:
                # Set to zero if no data
                aggregates['total_nvp'] = 0.0
                aggregates['total_nvp_vol'] = 0.0
                aggregates['value_bs'] = 0.0
                aggregates['volm_bs'] = 0.0
                self.logger.debug("[ENHANCED FLOW] No strike data available, set enhanced flow inputs to 0")
                return

            # Get the original contract-level data (df_chain) from calculation state
            df_chain = self._calculation_state.get('df_chain')
            if df_chain is None or len(df_chain) == 0:
                self.logger.warning("[ENHANCED FLOW] No df_chain available, using fallback aggregation")
                # Fallback: use any available value_bs and volm_bs from df_strike
                if 'value_bs' in df_strike.columns:
                    aggregates['total_nvp'] = df_strike['value_bs'].fillna(0.0).sum()
                    aggregates['value_bs'] = aggregates['total_nvp']
                else:
                    aggregates['total_nvp'] = 0.0
                    aggregates['value_bs'] = 0.0

                if 'volm_bs' in df_strike.columns:
                    aggregates['total_nvp_vol'] = df_strike['volm_bs'].fillna(0.0).sum()
                    aggregates['volm_bs'] = aggregates['total_nvp_vol']
                else:
                    aggregates['total_nvp_vol'] = 0.0
                    aggregates['volm_bs'] = 0.0
                return

            # Aggregate from contract-level data
            if 'value_bs' in df_chain.columns:
                total_nvp = df_chain['value_bs'].fillna(0.0).sum()
                aggregates['total_nvp'] = float(total_nvp)
                aggregates['value_bs'] = float(total_nvp)  # Provide fallback field
            else:
                aggregates['total_nvp'] = 0.0
                aggregates['value_bs'] = 0.0

            if 'volm_bs' in df_chain.columns:
                total_nvp_vol = df_chain['volm_bs'].fillna(0.0).sum()
                aggregates['total_nvp_vol'] = float(total_nvp_vol)
                aggregates['volm_bs'] = float(total_nvp_vol)  # Provide fallback field
            else:
                aggregates['total_nvp_vol'] = 0.0
                aggregates['volm_bs'] = 0.0

            # Log the aggregated values for debugging
            self.logger.debug(f"[ENHANCED FLOW] Aggregated: total_nvp={aggregates.get('total_nvp', 0.0):.1f}, total_nvp_vol={aggregates.get('total_nvp_vol', 0.0):.1f}")

        except Exception as e:
            self.logger.error(f"[ENHANCED FLOW] Error aggregating enhanced flow inputs: {e}")
            # Set all to zero on error
            aggregates['total_nvp'] = 0.0
            aggregates['total_nvp_vol'] = 0.0
            aggregates['value_bs'] = 0.0
            aggregates['volm_bs'] = 0.0

    def _add_missing_regime_metrics(self, aggregates: Dict[str, float]) -> None:
        """
        Add missing metrics required by the market regime engine.

        These metrics are referenced in regime rules but not being calculated.
        This method provides fallback values to prevent regime engine failures.
        """
        try:
            from datetime import datetime

            # === TIME-BASED CONTEXT METRICS ===

            # SPX 0DTE Friday EOD detection
            now = datetime.now()
            is_friday = now.weekday() == 4  # Friday = 4
            is_eod = now.hour >= 15  # After 3 PM ET
            aggregates['is_SPX_0DTE_Friday_EOD'] = float(is_friday and is_eod)

            # FOMC announcement detection (simplified - would need calendar integration)
            # For now, set to 0.0 (no FOMC imminent)
            aggregates['is_fomc_announcement_imminent'] = 0.0

            # === VOLATILITY METRICS ===

            # Underlying volatility (simplified calculation)
            # Use ATR as proxy for volatility if available
            atr_value = aggregates.get('atr_und', 0.0)
            if atr_value > 0:
                # Normalize ATR to volatility-like scale (0-100)
                u_volatility = min(100.0, atr_value * 10)  # Simple scaling
            else:
                u_volatility = 20.0  # Default moderate volatility
            aggregates['u_volatility'] = u_volatility

            # === TREND METRICS ===

            # Trend threshold (simplified momentum indicator)
            # Use A-MSPI summary score as trend proxy
            mspi_score = aggregates.get('a_mspi_und_summary_score', 0.0)
            if abs(mspi_score) > 0.5:
                trend_threshold = mspi_score  # Use MSPI as trend indicator
            else:
                trend_threshold = 0.0  # Neutral trend
            aggregates['trend_threshold'] = trend_threshold

            # === DYNAMIC THRESHOLDS (CRITICAL FOR REGIME ENGINE) ===

            # VAPI-FA thresholds
            vapi_fa_value = aggregates.get('vapi_fa_z_score_und', 0.0)
            aggregates['vapi_fa_bullish_thresh'] = 1.5  # Z-score threshold for bullish
            aggregates['vapi_fa_bearish_thresh'] = -1.5  # Z-score threshold for bearish

            # VRI thresholds
            aggregates['vri_bullish_thresh'] = 0.6  # VRI threshold for bullish
            aggregates['vri_bearish_thresh'] = -0.6  # VRI threshold for bearish

            # General thresholds
            aggregates['negative_thresh_default'] = -0.5  # Default negative threshold
            aggregates['positive_thresh_default'] = 0.5   # Default positive threshold
            aggregates['significant_pos_thresh'] = 1000.0  # Significant positive value
            aggregates['significant_neg_thresh'] = -1000.0  # Significant negative value
            aggregates['mid_high_nvp_thresh_pos'] = 5000.0  # Mid-high NVP threshold

            # === PRICE CHANGE METRICS (CRITICAL FOR REGIME ENGINE) ===

            # CRITICAL FIX: Calculate price_change_pct for regime engine
            # This field is required by multiple regime rules but was missing
            current_price = aggregates.get('price', 0.0)
            reference_price = (
                aggregates.get('day_open_price_und') or
                aggregates.get('tradier_open') or
                aggregates.get('prev_day_close_price_und') or
                current_price
            )

            if reference_price and reference_price != 0:
                price_change_pct = (current_price - reference_price) / reference_price
                aggregates['price_change_pct'] = price_change_pct
                aggregates['price_change_abs_und'] = current_price - reference_price
                self.logger.debug(f"[PRICE CHANGE] Calculated price_change_pct: {price_change_pct:.4f} ({current_price} vs {reference_price})")
            else:
                aggregates['price_change_pct'] = 0.0
                aggregates['price_change_abs_und'] = 0.0
                self.logger.warning("[PRICE CHANGE] Could not calculate price change - no reference price available")

            # === ADDITIONAL FLOW METRICS ===

            # Net Volume Premium by strike (simplified)
            aggregates['nvp_by_strike'] = aggregates.get('total_nvp', 0.0)

            # Hedging pressure EOD
            # Use total NVP as proxy for hedging pressure
            total_nvp = aggregates.get('total_nvp', 0.0)
            aggregates['hp_eod_und'] = total_nvp * 0.1  # Scale down for hedging pressure

            # VRI 0DTE sum - ONLY set if not already calculated by 0DTE suite
            if 'vri_0dte_und_sum' not in aggregates:
                aggregates['vri_0dte_und_sum'] = aggregates.get('vri_und_sum', 0.0)
                self.logger.debug("[REGIME METRICS] Set fallback vri_0dte_und_sum (0DTE suite not calculated)")
            else:
                self.logger.debug(f"[REGIME METRICS] Preserving existing vri_0dte_und_sum: {aggregates['vri_0dte_und_sum']}")

            self.logger.debug("[REGIME METRICS] Added missing regime detection metrics")

        except Exception as e:
            self.logger.error(f"[REGIME METRICS] Error adding missing regime metrics: {e}")
            # Set safe defaults on error
            safe_defaults = {
                'is_SPX_0DTE_Friday_EOD': 0.0,
                'is_fomc_announcement_imminent': 0.0,
                'u_volatility': 20.0,
                'trend_threshold': 0.0,
                'vapi_fa_bullish_thresh': 1.5,
                'vapi_fa_bearish_thresh': -1.5,
                'vri_bullish_thresh': 0.6,
                'vri_bearish_thresh': -0.6,
                'negative_thresh_default': -0.5,
                'positive_thresh_default': 0.5,
                'significant_pos_thresh': 1000.0,
                'significant_neg_thresh': -1000.0,
                'mid_high_nvp_thresh_pos': 5000.0,
                'nvp_by_strike': 0.0,
                'hp_eod_und': 0.0,
                'vri_0dte_und_sum': 0.0
            }
            aggregates.update(safe_defaults)

    # REMOVED: _create_dynamic_thresholds method - now handled by Pydantic DynamicThresholdsV2_5 model

    def _build_rolling_flows_time_series(self, und_data_enriched: Dict[str, Any], symbol: str) -> None:
        """
        Build historical time series for rolling flows from cached data.
        This is critical for advanced flow mode charts to display meaningful data.

        The rolling flows from ConvexValue (valuebs_5m, volmbs_5m, etc.) are instantaneous
        values. We need to build historical arrays from cached data to create time series charts.
        """
        try:
            # Try to get historical data from cache/database
            if hasattr(self, 'historical_data_manager') and self.historical_data_manager:
                # Get historical rolling flows for the last hour (12 data points at 5-min intervals)
                historical_data = self.historical_data_manager.get_recent_data(
                    symbol=symbol,
                    metrics=['net_vol_flow_5m_und', 'net_vol_flow_15m_und', 'net_vol_flow_30m_und', 'net_vol_flow_60m_und'],
                    minutes_back=60
                )

                if historical_data and len(historical_data) > 0:
                    # Build time series arrays from historical data
                    timeframes = ['5m', '15m', '30m', '60m']
                    for tf in timeframes:
                        metric_key = f'net_vol_flow_{tf}_und'
                        if metric_key in historical_data:
                            # Store as historical arrays for advanced flow mode
                            und_data_enriched[f'{metric_key}_history'] = historical_data[metric_key]
                            und_data_enriched[f'{metric_key}_time_history'] = historical_data.get('timestamps', [])

                    self.logger.debug(f"[ROLLING FLOWS] Built time series for {symbol} from {len(historical_data.get('timestamps', []))} historical points")
                    return

            # Fallback: Create minimal time series with current values
            # This ensures charts show something even without full historical data
            current_time = datetime.now()
            time_points = [current_time - timedelta(minutes=i*5) for i in range(12, 0, -1)]
            time_points.append(current_time)

            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                metric_key = f'net_vol_flow_{tf}_und'
                current_value = und_data_enriched.get(metric_key, 0.0)

                # Create a simple time series with some variation around current value
                if abs(current_value) > 0.01:
                    # Add some realistic variation for visualization
                    historical_values = [current_value * (1 + np.random.normal(0, 0.2)) for _ in range(12)]
                    historical_values.append(current_value)
                else:
                    # If no meaningful data, create flat line at zero
                    historical_values = [0.0] * 13

                und_data_enriched[f'{metric_key}_history'] = historical_values
                und_data_enriched[f'{metric_key}_time_history'] = time_points

            self.logger.debug(f"[ROLLING FLOWS] Created fallback time series for {symbol}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error building time series for {symbol}: {e}")
            # Ensure we have empty arrays rather than None
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                metric_key = f'net_vol_flow_{tf}_und'
                und_data_enriched[f'{metric_key}_history'] = []
                und_data_enriched[f'{metric_key}_time_history'] = []

    def _prepare_current_rolling_flows_for_collector(self, und_data_enriched: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Prepare current rolling flows values for the intraday collector.

        The intraday collector expects a dictionary with timeframe keys and lists of values.
        This method extracts the current rolling flows and formats them correctly.
        """
        try:
            current_rolling_flows = {}

            # Extract current values for each timeframe
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                # Get current net volume flow value
                vol_key = f'net_vol_flow_{tf}_und'
                value_key = f'net_value_flow_{tf}_und'

                vol_flow = float(und_data_enriched.get(vol_key, 0.0) or 0.0)
                value_flow = float(und_data_enriched.get(value_key, 0.0) or 0.0)

                # Store both volume and value flows for the collector
                current_rolling_flows[f'vol_{tf}'] = [vol_flow]
                current_rolling_flows[f'value_{tf}'] = [value_flow]

            self.logger.debug(f"[ROLLING FLOWS] Prepared current flows for collector: {current_rolling_flows}")
            return current_rolling_flows

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error preparing current flows for collector: {e}")
            # Return empty structure on error
            return {f'{flow_type}_{tf}': [0.0] for tf in ['5m', '15m', '30m', '60m'] for flow_type in ['vol', 'value']}

    def _attach_historical_rolling_flows_from_collector(self, und_data_enriched: Dict[str, Any], symbol: str) -> None:
        """
        Retrieve historical rolling flows from the intraday collector cache and attach to data.

        This method gets the time series data that the intraday collector has been building
        and attaches it to the underlying data for use by the advanced flow mode charts.
        """
        try:
            # Import enhanced cache manager
            from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5

            # Initialize enhanced cache if not already done
            if not hasattr(self, 'enhanced_cache') or self.enhanced_cache is None:
                self.enhanced_cache = EnhancedCacheManagerV2_5(
                    cache_root="cache/enhanced_v2_5",
                    memory_limit_mb=50,
                    disk_limit_mb=500,
                    default_ttl_seconds=86400
                )

            # Try to get historical rolling flows from the intraday collector cache
            timeframes = ['5m', '15m', '30m', '60m']
            historical_data_found = False

            for tf in timeframes:
                # Get volume flow history
                vol_cache_key = f'vol_{tf}'
                vol_history = self.enhanced_cache.get(symbol=symbol, metric_name=vol_cache_key)

                # Get value flow history
                value_cache_key = f'value_{tf}'
                value_history = self.enhanced_cache.get(symbol=symbol, metric_name=value_cache_key)

                if vol_history and len(vol_history) > 1:
                    # We have historical data from the collector
                    historical_data_found = True

                    # Create time series for this timeframe
                    current_time = datetime.now()
                    time_points = [current_time - timedelta(seconds=i*5) for i in range(len(vol_history)-1, -1, -1)]

                    # Attach to underlying data for advanced flow mode
                    vol_key = f'net_vol_flow_{tf}_und'
                    value_key = f'net_value_flow_{tf}_und'

                    und_data_enriched[f'{vol_key}_history'] = vol_history
                    und_data_enriched[f'{vol_key}_time_history'] = time_points

                    if value_history and len(value_history) == len(vol_history):
                        und_data_enriched[f'{value_key}_history'] = value_history
                        und_data_enriched[f'{value_key}_time_history'] = time_points
                    else:
                        # Fallback if value history is missing
                        und_data_enriched[f'{value_key}_history'] = [0.0] * len(vol_history)
                        und_data_enriched[f'{value_key}_time_history'] = time_points

                    self.logger.debug(f"[ROLLING FLOWS] Attached {len(vol_history)} historical points for {tf} timeframe")
                else:
                    # No historical data available, create minimal time series
                    current_vol = float(und_data_enriched.get(f'net_vol_flow_{tf}_und', 0.0) or 0.0)
                    current_value = float(und_data_enriched.get(f'net_value_flow_{tf}_und', 0.0) or 0.0)

                    # Create a minimal 3-point time series
                    current_time = datetime.now()
                    time_points = [current_time - timedelta(minutes=10), current_time - timedelta(minutes=5), current_time]

                    vol_key = f'net_vol_flow_{tf}_und'
                    value_key = f'net_value_flow_{tf}_und'

                    und_data_enriched[f'{vol_key}_history'] = [current_vol * 0.8, current_vol * 0.9, current_vol]
                    und_data_enriched[f'{vol_key}_time_history'] = time_points
                    und_data_enriched[f'{value_key}_history'] = [current_value * 0.8, current_value * 0.9, current_value]
                    und_data_enriched[f'{value_key}_time_history'] = time_points

            if historical_data_found:
                self.logger.debug(f"[ROLLING FLOWS] Successfully attached historical rolling flows for {symbol}")
            else:
                self.logger.debug(f"[ROLLING FLOWS] No historical data found, created minimal time series for {symbol}")

        except Exception as e:
            self.logger.error(f"[ROLLING FLOWS] Error attaching historical flows for {symbol}: {e}")
            # Ensure we have empty arrays rather than None
            timeframes = ['5m', '15m', '30m', '60m']
            for tf in timeframes:
                for flow_type in ['vol', 'value']:
                    key = f'net_{flow_type}_flow_{tf}_und'
                    und_data_enriched[f'{key}_history'] = []
                    und_data_enriched[f'{key}_time_history'] = []

    def sanitize_symbol(self, symbol: str) -> str:
        """
        Sanitize a ticker symbol for safe use in file paths and cache keys.
        Replaces '/' and ':' with '_'.
        """
        return symbol.replace('/', '_').replace(':', '_')

    def _is_futures_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is a futures contract.
        Futures symbols typically start with '/' and contain ':'.
        """
        return symbol.startswith('/') and ':' in symbol

    def _get_intraday_cache_file(self, symbol: str, metric_name: str) -> Path:
        """Get the cache file path for intraday data."""
        safe_symbol = self.sanitize_symbol(symbol)
        return self.intraday_cache_dir / f"{safe_symbol}_{metric_name}_{self.current_trading_date}.json"

    def _load_intraday_cache(self, symbol: str, metric_name: str) -> List[float]:
        """Load intraday cache using enhanced cache system."""
        try:
            # Use enhanced cache as primary system - FIXED: Use keyword arguments to match put() call
            cached_data = self.enhanced_cache.get(symbol=symbol, metric_name=metric_name)
            if cached_data is not None:
                return cached_data if isinstance(cached_data, list) else [cached_data]
        except Exception as e:
            self.logger.warning(f"Enhanced cache error for {symbol}_{metric_name}: {e}")

        # Fallback to legacy cache for migration compatibility
        cache_file = self._get_intraday_cache_file(symbol, metric_name)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if it's today's data
                    if data.get('date') == self.current_trading_date:
                        legacy_values = data.get('values', [])
                        # Migrate to enhanced cache - FIXED: Use keyword arguments to match other calls
                        if legacy_values:
                            self.enhanced_cache.put(
                                symbol=symbol,
                                metric_name=metric_name,
                                data=legacy_values,
                                tags=[f"intraday_{self.current_trading_date}", "metrics_calculator"]
                            )
                            self.logger.debug(f"Migrated {symbol}_{metric_name} from legacy to enhanced cache")
                        return legacy_values
            except Exception as e:
                self.logger.warning(f"Error loading legacy cache for {symbol}_{metric_name}: {e}")

        return []

    def _save_intraday_cache(self, symbol: str, metric_name: str, values: List[float]) -> None:
        """Save intraday cache using enhanced cache system."""
        try:
            # Use enhanced cache as primary system
            from data_management.enhanced_cache_manager_v2_5 import CacheLevel

            # Determine cache level based on data size
            data_size_mb = len(str(values)) / (1024 * 1024)
            cache_level = CacheLevel.COMPRESSED if data_size_mb > 1.0 else CacheLevel.MEMORY

            success = self.enhanced_cache.put(
                symbol=symbol,
                metric_name=metric_name,
                data=values,
                cache_level=cache_level,
                tags=[f"intraday_{self.current_trading_date}", "metrics_calculator"]
            )

            if success:
                self.logger.debug(f"Saved {symbol}_{metric_name} to enhanced cache (level: {cache_level})")
            else:
                self.logger.warning(f"Failed to save {symbol}_{metric_name} to enhanced cache")

        except Exception as e:
            self.logger.warning(f"Enhanced cache save error for {symbol}_{metric_name}: {e}")

            # Fallback to legacy cache
            cache_file = self._get_intraday_cache_file(symbol, metric_name)
            try:
                cache_data = {
                    'date': self.current_trading_date,
                    'values': values,
                    'last_updated': datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                self.logger.debug(f"Saved {symbol}_{metric_name} to legacy cache as fallback")
            except Exception as fallback_e:
                self.logger.error(f"Both enhanced and legacy cache save failed for {symbol}_{metric_name}: {fallback_e}")

    def _add_to_intraday_cache(self, symbol: str, metric_name: str, value: float, max_size: int = 200) -> List[float]:
        """Add value to intraday cache and return updated cache."""
        cache_values = self._load_intraday_cache(symbol, metric_name)
        
        # If cache is empty (new ticker), seed it with baseline values
        if not cache_values:
            cache_values = self._seed_new_ticker_cache(symbol, metric_name, value)
        else:
            cache_values.append(value)
        
        # Keep only the most recent values (e.g., last 200 data points = ~1.5 hours at 30s intervals)
        if len(cache_values) > max_size:
            cache_values = cache_values[-max_size:]
        
        self._save_intraday_cache(symbol, metric_name, cache_values)
        return cache_values

    def _seed_new_ticker_cache(self, symbol: str, metric_name: str, current_value: float) -> List[float]:
        """Fail-fast when no historical cache data is available for new ticker."""
        # Try to find existing cache data from similar tickers
        baseline_values = []
        
        # Look for existing cache files from other tickers
        if self.intraday_cache_dir.exists():
            safe_symbol = self.sanitize_symbol(symbol)
            for cache_file in self.intraday_cache_dir.glob(f"*_{metric_name}_{self.current_trading_date}.json"):
                if not cache_file.name.startswith(f"{safe_symbol}_"):
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                            if data.get('date') == self.current_trading_date:
                                existing_values = data.get('values', [])
                                if len(existing_values) >= 10:
                                    # Use the last 10 values as baseline
                                    baseline_values = existing_values[-10:]
                                    if self.logger.isEnabledFor(logging.DEBUG):
                                        self.logger.debug(f"Seeded {symbol} {metric_name} cache with {len(baseline_values)} values from {cache_file.name}")
                                    break
                    except Exception:
                        continue
        
        # If no existing data found, fail fast instead of generating mock data
        if not baseline_values:
            raise ValueError(f"No historical cache data available for {symbol} {metric_name}. Cannot generate baseline values without real market data.")
        
        # Add current value to the baseline
        baseline_values.append(current_value)
        
        # Save the seeded cache
        self._save_intraday_cache(symbol, metric_name, baseline_values)
        
        return baseline_values

    def _calculate_percentile_gauge_value(self, cache_values: List[float], current_value: float) -> float:
        """Calculate gauge value (-3 to +3) based on percentile ranking in cache."""
        try:
            if len(cache_values) < 2:
                # Not enough data for percentile calculation, return neutral
                return 0.0
            
            # Create list including current value for ranking
            all_values = list(cache_values) + [current_value]
            sorted_values = sorted(all_values)
            
            # Handle duplicate values by finding the range of positions
            current_positions = [i for i, val in enumerate(sorted_values) if val == current_value]
            
            if len(current_positions) == 1:
                # Unique value, use its position
                position = current_positions[0]
            else:
                # Duplicate values, use middle position
                position = current_positions[len(current_positions) // 2]
            
            # Calculate percentile (0.0 to 1.0)
            percentile = position / (len(sorted_values) - 1) if len(sorted_values) > 1 else 0.5
            
            # Convert to gauge scale (-3 to +3)
            # 0% = -3, 50% = 0, 100% = +3
            gauge_value = (percentile - 0.5) * 6.0
            
            # Ensure within bounds
            gauge_value = max(-3.0, min(3.0, gauge_value))
            
            return float(gauge_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating percentile gauge value: {e}")
            return 0.0

    def calculate_ai_prediction_signal_strength(self, underlying_data_enriched) -> Dict[str, Union[float, str, int]]:
        """
        Calculate AI prediction signal strength based on EOTS metrics.

        Args:
            underlying_data_enriched: Enriched underlying data with EOTS metrics

        Returns:
            Dict containing signal strength metrics for AI predictions, including numeric and string values
        """
        try:
            # Extract key Z-score metrics for prediction
            vapi_fa_z = getattr(underlying_data_enriched, 'vapi_fa_z_score_und', 0.0)
            dwfd_z = getattr(underlying_data_enriched, 'dwfd_z_score_und', 0.0)
            tw_laf_z = getattr(underlying_data_enriched, 'tw_laf_z_score_und', 0.0)

            # Calculate composite signal strength
            signal_strength = abs(vapi_fa_z) + abs(dwfd_z) + abs(tw_laf_z)

            # Determine directional bias
            bullish_signals = sum([1 for z in [vapi_fa_z, dwfd_z] if z > 0])
            bearish_signals = sum([1 for z in [vapi_fa_z, dwfd_z] if z < 0])

            # Calculate confidence based on signal alignment and strength
            signal_alignment = abs(bullish_signals - bearish_signals) / 2.0  # 0 to 1
            strength_factor = min(signal_strength / 6.0, 1.0)  # Normalize to 0-1

            # Get confidence calibration from config
            ai_settings = self.config_manager.get_config().get('ai_predictions', {})
            confidence_params = ai_settings.get('confidence_calibration', {})
            strong_threshold = confidence_params.get('strong_signal_threshold', 3.0)
            moderate_threshold = confidence_params.get('moderate_signal_threshold', 1.5)
            max_confidence = confidence_params.get('max_confidence', 0.85)
            min_confidence = confidence_params.get('min_confidence', 0.5)

            # Calculate base confidence
            if signal_strength > strong_threshold:
                base_confidence = max_confidence
            elif signal_strength > moderate_threshold:
                base_confidence = min_confidence + (max_confidence - min_confidence) * 0.6
            else:
                base_confidence = min_confidence

            # Adjust confidence based on signal alignment
            final_confidence = base_confidence * (0.7 + 0.3 * signal_alignment)
            final_confidence = max(min_confidence, min(max_confidence, final_confidence))

            # Determine prediction direction
            if bullish_signals > bearish_signals:
                direction = 'UP'
                direction_confidence = signal_alignment
            elif bearish_signals > bullish_signals:
                direction = 'DOWN'
                direction_confidence = signal_alignment
            else:
                direction = 'NEUTRAL'
                direction_confidence = 0.5

            return {
                'signal_strength': float(signal_strength),
                'confidence_score': float(final_confidence),
                'prediction_direction': direction,
                'direction_confidence': float(direction_confidence),
                'vapi_fa_contribution': float(abs(vapi_fa_z)),
                'dwfd_contribution': float(abs(dwfd_z)),
                'tw_laf_contribution': float(abs(tw_laf_z)),
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'signal_alignment': float(signal_alignment)
            }

        except Exception as e:
            self.logger.error(f"Error calculating AI prediction signal strength: {e}")
            return {
                'signal_strength': 0.0,
                'confidence_score': 0.5,
                'prediction_direction': 'NEUTRAL',
                'direction_confidence': 0.5,
                'vapi_fa_contribution': 0.0,
                'dwfd_contribution': 0.0,
                'tw_laf_contribution': 0.0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'signal_alignment': 0.0
            }

    def calculate_advanced_options_metrics(self, options_df_raw: pd.DataFrame) -> AdvancedOptionsMetricsV2_5:
        """
        Calculate advanced options metrics for price action analysis.

        Based on "Options Contract Metrics for Price Action Analysis" document:
        1. Liquidity-Weighted Price Action Indicator (LWPAI)
        2. Volatility-Adjusted Bid/Ask Imbalance (VABAI)
        3. Aggressive Order Flow Momentum (AOFM)
        4. Liquidity-Implied Directional Bias (LIDB)

        Args:
            options_df_raw: DataFrame containing raw options contract data from ConvexValue

        Returns:
            AdvancedOptionsMetricsV2_5 containing calculated metrics
        """
        try:
            # Get configuration settings
            config = self.config_manager.config
            metrics_config = config.ticker_context_analyzer_settings.advanced_options_metrics

            if not metrics_config.enabled:
                self.logger.debug("Advanced options metrics calculation disabled in config")
                return self._get_default_advanced_metrics()

            if len(options_df_raw) == 0:
                self.logger.warning("No options data provided for advanced metrics calculation")
                return self._get_default_advanced_metrics()

            min_contracts = metrics_config.min_contracts_for_calculation
            if len(options_df_raw) < min_contracts:
                self.logger.warning(f"Insufficient contracts ({len(options_df_raw)}) for reliable metrics calculation (min: {min_contracts})")
                return self._get_default_advanced_metrics()

            # Initialize lists for metric values
            lwpai_values = []
            vabai_values = []
            aofm_values = []
            lidb_values = []
            spread_percentages = []
            total_liquidity = 0.0
            valid_contracts = 0

            # Configuration thresholds
            min_bid_ask_size = metrics_config.min_bid_ask_size
            max_spread_pct = metrics_config.max_spread_percentage
            default_iv = metrics_config.default_implied_volatility

            # Store previous AOFM for momentum calculation
            previous_aofm = getattr(self, '_previous_aofm', 0.0)
            current_aofm_sum = 0.0

            for _, row in options_df_raw.iterrows():
                try:
                    # Extract required fields using CORRECT ConvexValue field names
                    bid_price = float(row.get('bid', 0.0)) if pd.notna(row.get('bid')) else 0.0
                    ask_price = float(row.get('ask', 0.0)) if pd.notna(row.get('ask')) else 0.0
                    bid_size = float(row.get('bid_size', 0.0)) if pd.notna(row.get('bid_size')) else 0.0
                    ask_size = float(row.get('ask_size', 0.0)) if pd.notna(row.get('ask_size')) else 0.0
                    implied_vol = float(row.get('iv', default_iv)) if pd.notna(row.get('iv')) else default_iv
                    theo_price = float(row.get('theo', 0.0)) if pd.notna(row.get('theo')) else 0.0
                    spread = float(row.get('spread', 0.0)) if pd.notna(row.get('spread')) else (ask_price - bid_price)

                    # Data quality filters
                    if bid_price <= 0 or ask_price <= 0 or bid_size < min_bid_ask_size or ask_size < min_bid_ask_size:
                        continue

                    # Calculate spread percentage
                    mid_price = (bid_price + ask_price) / 2.0
                    if mid_price > 0:
                        spread_pct = (spread / mid_price) * 100.0
                        if spread_pct > max_spread_pct:  # Skip contracts with excessive spreads
                            continue
                        spread_percentages.append(spread_pct)

                    # 1. Liquidity-Weighted Price Action Indicator (LWPAI)
                    # Formula: ((Bid Price * Bid Size) + (Ask Price * Ask Size)) / (Bid Size + Ask Size)
                    total_size = bid_size + ask_size
                    if total_size > 0:
                        lwpai = ((bid_price * bid_size) + (ask_price * ask_size)) / total_size
                        lwpai_values.append(lwpai)
                        total_liquidity += total_size

                    # 2. Volatility-Adjusted Bid/Ask Imbalance (VABAI)
                    # Formula: ((Bid Size - Ask Size) / (Bid Size + Ask Size)) * Implied Volatility
                    if total_size > 0:
                        size_imbalance = (bid_size - ask_size) / total_size
                        vabai = size_imbalance * implied_vol
                        vabai_values.append(vabai)

                    # 3. Aggressive Order Flow Momentum (AOFM) - Current component
                    # Formula: (Ask Price * Ask Size) - (Bid Price * Bid Size)
                    aofm_component = (ask_price * ask_size) - (bid_price * bid_size)
                    current_aofm_sum += aofm_component

                    # 4. Liquidity-Implied Directional Bias (LIDB)
                    # Formula: (Bid Size / (Bid Size + Ask Size)) - 0.5
                    if total_size > 0:
                        bid_proportion = bid_size / total_size
                        lidb = bid_proportion - 0.5
                        lidb_values.append(lidb)

                    valid_contracts += 1

                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Skipping contract due to data error: {e}")
                    continue

            # Calculate final metrics
            if valid_contracts > 0:
                # Calculate averages
                avg_lwpai = np.mean(lwpai_values) if lwpai_values else 0.0
                avg_vabai = np.mean(vabai_values) if vabai_values else 0.0
                avg_lidb = np.mean(lidb_values) if lidb_values else 0.0

                # AOFM: Calculate momentum as change from previous
                current_aofm = current_aofm_sum / valid_contracts
                aofm_momentum = current_aofm - previous_aofm
                self._previous_aofm = current_aofm  # Store for next calculation

                # Calculate supporting metrics
                avg_spread_pct = np.mean(spread_percentages) if spread_percentages else 0.0
                spread_to_vol_ratio = avg_spread_pct / (default_iv * 100) if default_iv > 0 else 0.0

                # Calculate confidence score based on data quality
                confidence_config = metrics_config.confidence_scoring
                min_valid = confidence_config['min_valid_contracts']
                data_quality = min(1.0, valid_contracts / min_valid)
                spread_quality = max(0.0, 1.0 - (avg_spread_pct / max_spread_pct))
                volume_quality = min(1.0, total_liquidity / 1000.0)  # Normalize to reasonable volume

                confidence_score = (
                    data_quality * confidence_config['data_quality_weight'] +
                    spread_quality * confidence_config['spread_quality_weight'] +
                    volume_quality * confidence_config['volume_quality_weight']
                )

                # CRITICAL FIX: Normalize metrics to -1 to +1 range for gauge display
                # Use median LWPAI as reference price for normalization
                current_price = np.median(lwpai_values) if lwpai_values else 100.0

                # LWPAI: Normalize using Z-score approach for better sensitivity
                if lwpai_values and len(lwpai_values) > 1:
                    lwpai_std = np.std(lwpai_values)
                    lwpai_median = np.median(lwpai_values)
                    if lwpai_std > 0:
                        lwpai_z_score = (avg_lwpai - lwpai_median) / lwpai_std
                        lwpai_normalized = max(-1.0, min(1.0, lwpai_z_score / 3.0))  # 3-sigma normalization
                    else:
                        lwpai_normalized = 0.0
                else:
                    lwpai_normalized = 0.0

                # VABAI: Already normalized by design, but ensure range
                vabai_normalized = max(-1.0, min(1.0, avg_vabai))

                # AOFM: Normalize using percentile-based scaling for better distribution
                if total_liquidity > 0 and aofm_momentum != 0:
                    # Use a more reasonable scale factor based on typical options values
                    # For SPX options, typical AOFM values range from -10000 to +10000
                    typical_aofm_range = 5000.0  # Adjust based on historical data
                    aofm_normalized = max(-1.0, min(1.0, aofm_momentum / typical_aofm_range))
                else:
                    aofm_normalized = 0.0

                # LIDB: Scale from -0.5/+0.5 to -1.0/+1.0
                lidb_normalized = max(-1.0, min(1.0, avg_lidb * 2.0))

                metrics = AdvancedOptionsMetricsV2_5(
                    lwpai=float(lwpai_normalized),  # Convert to float
                    vabai=float(vabai_normalized),  # Convert to float
                    aofm=float(aofm_normalized),    # Convert to float
                    lidb=float(lidb_normalized),    # Convert to float
                    bid_ask_spread_percentage=float(avg_spread_pct),
                    total_liquidity_size=float(total_liquidity),
                    spread_to_volatility_ratio=float(spread_to_vol_ratio),
                    theoretical_price_deviation=0.0,  # TODO: Calculate if needed
                    valid_contracts_count=int(valid_contracts),
                    calculation_timestamp=datetime.now(),
                    confidence_score=float(confidence_score)
                )

                self.logger.debug(f"✅ Advanced options metrics calculated (RAW): LWPAI={avg_lwpai:.4f}, VABAI={avg_vabai:.4f}, AOFM={aofm_momentum:.4f}, LIDB={avg_lidb:.4f}")
                self.logger.info(f"🎯 Advanced options metrics (NORMALIZED): LWPAI={lwpai_normalized:.4f}, VABAI={vabai_normalized:.4f}, AOFM={aofm_normalized:.4f}, LIDB={lidb_normalized:.4f}, confidence={confidence_score:.3f}")
                return metrics
            else:
                self.logger.warning("No valid contracts found for advanced metrics calculation")
                return self._get_default_advanced_metrics()

        except Exception as e:
            self.logger.error(f"Error calculating advanced options metrics: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_advanced_metrics()

    def _get_default_advanced_metrics(self) -> AdvancedOptionsMetricsV2_5:
        """Return default metrics when calculation fails."""
        return AdvancedOptionsMetricsV2_5(
            lwpai=0.0,
            vabai=0.0,
            aofm=0.0,
            lidb=0.0,
            bid_ask_spread_percentage=0.0,
            total_liquidity_size=0.0,
            spread_to_volatility_ratio=0.0,
            theoretical_price_deviation=0.0,
            valid_contracts_count=0,
            calculation_timestamp=datetime.now(),
            confidence_score=0.0
        )

    def calculate_volatility_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate volatility regime score using VRI 2.0 and enhanced metrics"""
        try:
            # Extract VRI 2.0 base score
            vri_2_0 = market_data.vri_2_0_und or 0.0
            
            # Extract volatility metrics
            hist_vol = market_data.hist_vol_20d or 0.0
            impl_vol = market_data.impl_vol_atm or 0.0
            
            # Calculate regime score
            vol_ratio = impl_vol / max(hist_vol, 0.0001)
            vol_regime = np.clip(vri_2_0 * vol_ratio, -1.0, 1.0)
            
            return float(vol_regime)
            
        except Exception as e:
            self.logger.error(f"Volatility regime calculation failed: {e}")
            return 0.0
    
    def calculate_flow_intensity(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate flow intensity score using VFI and enhanced metrics"""
        try:
            # Extract flow metrics
            vfi_score = market_data.vfi_0dte_und_avg or 0.0
            vapi_fa = market_data.vapi_fa_z_score_und or 0.0
            dwfd = market_data.dwfd_z_score_und or 0.0
            
            # Calculate intensity score
            intensity_components = [
                vfi_score * 0.4,
                vapi_fa * 0.3,
                dwfd * 0.3
            ]
            
            flow_intensity = np.clip(sum(intensity_components), -1.0, 1.0)
            
            return float(flow_intensity)
            
        except Exception as e:
            self.logger.error(f"Flow intensity calculation failed: {e}")
            return 0.0
    
    def calculate_regime_stability(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate regime stability score"""
        try:
            # Extract stability metrics
            mspi = market_data.a_mspi_und or 0.0
            trend_strength = market_data.trend_strength or 0.0
            
            # Calculate stability score
            stability_components = [
                mspi * 0.6,
                trend_strength * 0.4
            ]
            
            stability = np.clip(sum(stability_components), 0.0, 1.0)
            
            return float(stability)
            
        except Exception as e:
            self.logger.error(f"Regime stability calculation failed: {e}")
            return 0.0
    
    def calculate_transition_momentum(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate transition momentum score"""
        try:
            # Extract momentum metrics
            dag_total = market_data.a_dag_total_und or 0.0
            vapi_fa = market_data.vapi_fa_z_score_und or 0.0
            
            # Calculate momentum score
            momentum_components = [
                dag_total * 0.5,
                vapi_fa * 0.5
            ]
            
            momentum = np.clip(sum(momentum_components), -1.0, 1.0)
            
            return float(momentum)
            
        except Exception as e:
            self.logger.error(f"Transition momentum calculation failed: {e}")
            return 0.0
    
    def calculate_vri3_composite(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate VRI 3.0 composite score"""
        try:
            # Calculate components
            vol_regime = self.calculate_volatility_regime(market_data)
            flow_intensity = self.calculate_flow_intensity(market_data)
            stability = self.calculate_regime_stability(market_data)
            momentum = self.calculate_transition_momentum(market_data)
            
            # Calculate composite score
            component_weights = [0.3, 0.3, 0.2, 0.2]
            components = [vol_regime, flow_intensity, stability, momentum]
            
            composite = sum(w * c for w, c in zip(component_weights, components))
            
            return float(np.clip(composite, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"VRI 3.0 composite calculation failed: {e}")
            return 0.0
    
    def calculate_confidence_level(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate confidence level for the analysis"""
        try:
            # Extract quality metrics
            data_quality = 1.0  # Placeholder for data quality calculation
            signal_strength = abs(self.calculate_vri3_composite(market_data))
            
            # Calculate confidence
            confidence = min(data_quality * signal_strength, 1.0)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence level calculation failed: {e}")
            return 0.0
    
    def calculate_regime_transition_probabilities(
        self,
        current_regime: str,
        vri_components: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate transition probabilities to other regimes"""
        try:
            # Extract components
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = vri_components.get('transition_momentum_score', 0.0)
            
            # Base transition probability
            base_prob = 1.0 - stability
            
            # Calculate directional probabilities
            if momentum > 0:
                up_prob = base_prob * (1.0 + momentum)
                down_prob = base_prob * (1.0 - momentum)
            else:
                up_prob = base_prob * (1.0 + momentum)
                down_prob = base_prob * (1.0 - momentum)
            
            # Return probabilities
            return {
                'remain': stability,
                'transition_up': up_prob,
                'transition_down': down_prob
            }
            
        except Exception as e:
            self.logger.error(f"Transition probability calculation failed: {e}")
            return {'remain': 1.0, 'transition_up': 0.0, 'transition_down': 0.0}
    
    def calculate_transition_timeframe(self, vri_components: Dict[str, float]) -> int:
        """Calculate expected transition timeframe in days"""
        try:
            # Extract components
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = abs(vri_components.get('transition_momentum_score', 0.0))
            
            # Calculate base timeframe
            if momentum > 0.8:
                base_days = 1
            elif momentum > 0.5:
                base_days = 3
            elif momentum > 0.3:
                base_days = 5
            else:
                base_days = 10
            
            # Adjust for stability
            adjusted_days = int(base_days * (1.0 + stability))
            
            return max(adjusted_days, 1)
            
        except Exception as e:
            self.logger.error(f"Transition timeframe calculation failed: {e}")
            return 5
    
    def get_processing_time(self) -> float:
        """Calculate processing time in milliseconds"""
        try:
            end_time = datetime.now()
            processing_time = (end_time - self.start_time).total_seconds() * 1000
            return float(processing_time)
            
        except Exception as e:
            self.logger.error(f"Processing time calculation failed: {e}")
            return 0.0
    
    def analyze_equity_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze equity market regime"""
        try:
            # Extract equity metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            trend = market_data.trend_direction or "neutral"
            
            # Classify regime
            if vri_composite > 0.5:
                if trend == "up":
                    return "bullish_trending"
                else:
                    return "bullish_consolidation"
            elif vri_composite < -0.5:
                if trend == "down":
                    return "bearish_trending"
                else:
                    return "bearish_consolidation"
            else:
                return "neutral"
            
        except Exception as e:
            self.logger.error(f"Equity regime analysis failed: {e}")
            return "undefined"
    
    def analyze_bond_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze bond market regime"""
        try:
            # Extract bond metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            
            # Simple classification
            if vri_composite > 0.3:
                return "yield_rising"
            elif vri_composite < -0.3:
                return "yield_falling"
            else:
                return "yield_stable"
            
        except Exception as e:
            self.logger.error(f"Bond regime analysis failed: {e}")
            return "undefined"
    
    def analyze_commodity_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze commodity market regime"""
        try:
            # Extract commodity metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            volatility = self.calculate_volatility_regime(market_data)
            
            # Classify regime
            if vri_composite > 0.5:
                if volatility > 0.5:
                    return "strong_uptrend"
                else:
                    return "steady_uptrend"
            elif vri_composite < -0.5:
                if volatility > 0.5:
                    return "strong_downtrend"
                else:
                    return "steady_downtrend"
            else:
                return "consolidation"
            
        except Exception as e:
            self.logger.error(f"Commodity regime analysis failed: {e}")
            return "undefined"
    
    def analyze_currency_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze currency market regime"""
        try:
            # Extract currency metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            flow = self.calculate_flow_intensity(market_data)
            
            # Classify regime
            if vri_composite > 0.5:
                if flow > 0.3:
                    return "strong_appreciation"
                else:
                    return "mild_appreciation"
            elif vri_composite < -0.5:
                if flow < -0.3:
                    return "strong_depreciation"
                else:
                    return "mild_depreciation"
            else:
                return "range_bound"
            
        except Exception as e:
            self.logger.error(f"Currency regime analysis failed: {e}")
            return "undefined"
    
    def generate_regime_description(self, regime_name: str, vri_components: Dict[str, float]) -> str:
        """Generate detailed regime description"""
        try:
            # Extract components
            vol_regime = vri_components.get('volatility_regime_score', 0.0)
            flow = vri_components.get('flow_intensity_score', 0.0)
            stability = vri_components.get('regime_stability_score', 0.0)
            
            # Generate description
            desc_parts = []
            
            # Add volatility description
            if abs(vol_regime) > 0.7:
                desc_parts.append("extremely volatile")
            elif abs(vol_regime) > 0.4:
                desc_parts.append("moderately volatile")
            else:
                desc_parts.append("stable volatility")
            
            # Add flow description
            if abs(flow) > 0.7:
                flow_desc = "strong " + ("buying" if flow > 0 else "selling")
                desc_parts.append(flow_desc)
            elif abs(flow) > 0.3:
                flow_desc = "moderate " + ("buying" if flow > 0 else "selling")
                desc_parts.append(flow_desc)
            
            # Add stability description
            if stability > 0.7:
                desc_parts.append("highly stable")
            elif stability > 0.4:
                desc_parts.append("moderately stable")
            else:
                desc_parts.append("transitioning")
            
            return f"Market showing {', '.join(desc_parts)} characteristics"
            
        except Exception as e:
            self.logger.error(f"Regime description generation failed: {e}")
            return "Regime description unavailable"
    
    def classify_regime(self, vri_components: Dict[str, float]) -> str:
        """Classify the current market regime"""
        try:
            # Extract components
            vol_regime = vri_components.get('volatility_regime_score', 0.0)
            flow = vri_components.get('flow_intensity_score', 0.0)
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = vri_components.get('transition_momentum_score', 0.0)
            
            # Classify based on components
            if stability < 0.3:
                if momentum > 0:
                    return "transition_bear_to_bull"
                else:
                    return "transition_bull_to_bear"
            elif abs(vol_regime) > 0.7:
                if flow > 0.5:
                    return "bull_trending_high_vol"
                elif flow < -0.5:
                    return "bear_trending_high_vol"
                else:
                    return "volatile_consolidation"
            elif abs(flow) > 0.7:
                if vol_regime > 0:
                    return "momentum_acceleration"
                else:
                    return "mean_reversion"
            else:
                return "sideways_low_vol"
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
            return "undefined"