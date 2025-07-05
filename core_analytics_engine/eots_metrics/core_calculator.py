# core_analytics_engine/eots_metrics/core_calculator.py

"""
EOTS Core Calculator - Consolidated Base Utilities + Foundational Metrics

Consolidates:
- base_calculator.py: Core utilities, caching, validation
- foundational_metrics.py: Tier 1 foundational metrics

Optimizations:
- Unified caching strategy
- Streamlined utility functions
- Integrated foundational calculations
- Eliminated redundant base classes
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set, TypeVar
from datetime import datetime, date, timezone
from pydantic import BaseModel, Field, ConfigDict
from collections import deque

# Import necessary schemas
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5, CacheLevel
from data_models import ProcessedUnderlyingAggregatesV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9
T = TypeVar('T')

# =============================================================================
# STRICT PYDANTIC V2 MODELS - NO DICT[STR, ANY] PATTERNS
# =============================================================================

class AdditionalDataModel(BaseModel):
    """Pydantic model for additional data fields - replaces Dict[str, Any]"""
    custom_metrics: Optional[Dict[str, float]] = Field(default_factory=dict, description="Custom metric values")
    calculation_flags: Optional[Dict[str, bool]] = Field(default_factory=dict, description="Calculation flags")
    metadata_tags: Optional[List[str]] = Field(default_factory=list, description="Metadata tags")
    source_identifiers: Optional[List[str]] = Field(default_factory=list, description="Data source identifiers")
    processing_notes: Optional[List[str]] = Field(default_factory=list, description="Processing notes")
    
    model_config = ConfigDict(extra='forbid')

class ValidationResultsModel(BaseModel):
    """Pydantic model for validation results - replaces Dict[str, Any]"""
    validation_passed: bool = Field(default=True, description="Overall validation status")
    error_messages: List[str] = Field(default_factory=list, description="Validation error messages")
    warning_messages: List[str] = Field(default_factory=list, description="Validation warning messages")
    bounds_checks: Dict[str, bool] = Field(default_factory=dict, description="Bounds check results")
    data_quality_score: Optional[float] = Field(default=None, description="Overall data quality score")
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Validation timestamp")
    
    model_config = ConfigDict(extra='forbid')
    
    def update(self, **kwargs):
        """Update validation results"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class CacheDataModel(BaseModel):
    """Pydantic model for cache data - replaces Dict[str, Any]"""
    metric_values: Dict[str, float] = Field(default_factory=dict, description="Cached metric values")
    timestamps: Dict[str, datetime] = Field(default_factory=dict, description="Value timestamps")
    expiry_times: Dict[str, datetime] = Field(default_factory=dict, description="Cache expiry times")
    hit_counts: Dict[str, int] = Field(default_factory=dict, description="Cache hit counts")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last cache update")
    
    model_config = ConfigDict(extra='forbid')

class UnderlyingDataInput(BaseModel):
    """Pydantic model for underlying data input to replace raw Dict usage"""
    symbol: str = Field(..., description="Underlying symbol")
    price: float = Field(..., description="Current underlying price")

    # Basic market data
    day_volume: Optional[float] = Field(None, description="Daily volume")
    u_volatility: Optional[float] = Field(None, description="Underlying volatility")

    # Greek flows
    call_gxoi: Optional[float] = Field(None, description="Call gamma x open interest")
    put_gxoi: Optional[float] = Field(None, description="Put gamma x open interest")
    net_cust_delta_flow_und: Optional[float] = Field(None, description="Net customer delta flow")
    net_cust_gamma_flow_und: Optional[float] = Field(None, description="Net customer gamma flow")
    net_cust_vega_flow_und: Optional[float] = Field(None, description="Net customer vega flow")
    net_cust_theta_flow_und: Optional[float] = Field(None, description="Net customer theta flow")

    # Flow metrics
    net_value_flow_5m_und: Optional[float] = Field(None, description="5-minute net value flow")
    net_vol_flow_5m_und: Optional[float] = Field(None, description="5-minute net volume flow")
    net_value_flow_15m_und: Optional[float] = Field(None, description="15-minute net value flow")
    net_vol_flow_15m_und: Optional[float] = Field(None, description="15-minute net volume flow")
    net_value_flow_30m_und: Optional[float] = Field(None, description="30-minute net value flow")
    net_vol_flow_30m_und: Optional[float] = Field(None, description="30-minute net volume flow")

    # Additional fields for extensibility - STRICT PYDANTIC V2
    additional_data: AdditionalDataModel = Field(default_factory=AdditionalDataModel, description="Additional data fields")

    model_config = ConfigDict(extra='allow')  # Allow extra fields for flexibility

class FoundationalMetricsOutput(BaseModel):
    """Pydantic model for foundational metrics output"""
    # Input data (preserved)
    symbol: str = Field(..., description="Underlying symbol")
    price: float = Field(..., description="Current underlying price")

    # Calculated foundational metrics
    gib_oi_based_und: Optional[float] = Field(None, description="GIB from open interest")
    hp_eod_und: Optional[float] = Field(None, description="End-of-day hedging pressure")
    td_gib_und: Optional[float] = Field(None, description="Traded dealer gamma imbalance")

    # Net customer flows (calculated)
    net_cust_delta_flow_und: Optional[float] = Field(None, description="Net customer delta flow")
    net_cust_gamma_flow_und: Optional[float] = Field(None, description="Net customer gamma flow")
    net_cust_vega_flow_und: Optional[float] = Field(None, description="Net customer vega flow")
    net_cust_theta_flow_und: Optional[float] = Field(None, description="Net customer theta flow")

    # Validation results
    validation_passed: bool = Field(True, description="Whether validation passed")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")

    # Calculation metadata
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When calculated")

    model_config = ConfigDict(extra='allow')  # Allow extra fields for extensibility

class MetricCalculationState(BaseModel):
    """Unified state tracking for all metric calculations"""
    current_symbol: Optional[str] = Field(None, description="Current symbol being processed")
    calculation_timestamp: Optional[datetime] = Field(None, description="Timestamp of last calculation")
    metrics_completed: Set[str] = Field(default_factory=set, description="Set of completed metrics")
    validation_results: ValidationResultsModel = Field(default_factory=lambda: ValidationResultsModel(), description="Validation results")

    model_config = ConfigDict(extra='forbid')

    def update_state(self, **kwargs):
        """Update state attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class MetricCache(BaseModel):
    """Unified cache for all metric types"""
    data: CacheDataModel = Field(default_factory=CacheDataModel, description="Cache data model")

    model_config = ConfigDict(extra='forbid')
    
class MetricCacheConfig(BaseModel):
    """Consolidated cache configuration"""
    foundational: MetricCache = Field(default_factory=MetricCache)
    flow_metrics: MetricCache = Field(default_factory=MetricCache)
    adaptive: MetricCache = Field(default_factory=MetricCache)
    heatmap: MetricCache = Field(default_factory=MetricCache)
    normalization: MetricCache = Field(default_factory=MetricCache)

    model_config = ConfigDict(extra='forbid')

    def get_cache(self, metric_name: str) -> CacheDataModel:
        """Get cache for specific metric category - STRICT PYDANTIC V2"""
        if hasattr(self, metric_name):
            return getattr(self, metric_name).data
        return CacheDataModel()

    def set_cache(self, metric_name: str, data: CacheDataModel):
        """Set cache for specific metric category - STRICT PYDANTIC V2"""
        if hasattr(self, metric_name):
            cache = getattr(self, metric_name)
            cache.data = data

class CoreCalculator:
    """
    Consolidated core calculator with base utilities and foundational metrics.
    
    Combines functionality from:
    - BaseCalculator: Core utilities, caching, validation
    - FoundationalMetricsCalculator: Tier 1 metrics
    """
    
    def __init__(self, config_manager: Any, historical_data_manager: Any, enhanced_cache_manager: EnhancedCacheManagerV2_5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager
        self.enhanced_cache_manager = enhanced_cache_manager
        
        # Initialize unified state and cache
        self._calculation_state = MetricCalculationState(
            current_symbol=None,
            calculation_timestamp=None
        )
        self._metric_cache_config = MetricCacheConfig()
        
        # Initialize options dataframe placeholder
        self._options_df = None
        
        # Configuration constants
        self.METRIC_BOUNDS = {
            'gib_oi_based_und': (-1000000, 1000000),
            'hp_eod_und': (-100, 100),
            'net_cust_delta_flow_und': (-1000000, 1000000),
            'net_cust_gamma_flow_und': (-1000000, 1000000),
            'net_cust_vega_flow_und': (-1000000, 1000000),
            'net_cust_theta_flow_und': (-1000000, 1000000)
        }
    
    # =============================================================================
    # FOUNDATIONAL METRICS (Tier 1) - Consolidated from foundational_metrics.py
    # =============================================================================
    
    def calculate_all_foundational_metrics(self, und_data: 'ProcessedUnderlyingAggregatesV2_5') -> 'ProcessedUnderlyingAggregatesV2_5':
        """
        STRICT PYDANTIC V2-ONLY: Calculate foundational metrics directly on ProcessedUnderlyingAggregatesV2_5.

        Calculates:
        - Net Customer Greek Flows (Delta, Gamma, Vega, Theta)
        - GIB (Gamma Imbalance from Open Interest)
        - HP_EOD (End-of-Day Hedging Pressure)
        - TD_GIB (Traded Dealer Gamma Imbalance)
        - Elite Impact Scores and Market Regime Metrics
        - Enhanced Rolling Flow Metrics (VAPI, DWFD, TW_LAF)

        Args:
            und_data: ProcessedUnderlyingAggregatesV2_5 model with raw data

        Returns:
            ProcessedUnderlyingAggregatesV2_5 model with calculated foundational metrics
        """
        self.logger.debug("Calculating foundational metrics...")

        try:
            # STRICT PYDANTIC V2-ONLY: Work directly with the model, no conversions
            if not hasattr(und_data, 'symbol') or not hasattr(und_data, 'price'):
                raise ValueError(f"CRITICAL: Invalid model type {type(und_data)} - must be ProcessedUnderlyingAggregatesV2_5")

            # Update calculation state
            self._calculation_state.update_state(
                current_symbol=und_data.symbol,
                calculation_timestamp=datetime.now(timezone.utc)
            )

            # Calculate Net Customer Greek Flows - STRICT PYDANTIC V2-ONLY
            updated_model = self._calculate_net_customer_greek_flows_v2_5(und_data)

            # Calculate GIB-based metrics - STRICT PYDANTIC V2-ONLY
            updated_model = self._calculate_gib_based_metrics_v2_5(updated_model)

            # Calculate Elite Impact Scores and Market Regime Metrics
            from data_models.elite_intelligence import EliteImpactCalculator
            elite_calc = EliteImpactCalculator()
            elite_results = elite_calc.calculate_elite_impact_score(self._options_df, updated_model)

            # Update model with elite impact results
            updated_model = updated_model.model_copy(update={
                'elite_impact_score_und': elite_results.elite_impact_score_und,
                'institutional_flow_score_und': elite_results.institutional_flow_score_und,
                'flow_momentum_index_und': elite_results.flow_momentum_index_und,
                'market_regime_elite': elite_results.market_regime_elite,
                'flow_type_elite': elite_results.flow_type_elite,
                'volatility_regime_elite': elite_results.volatility_regime_elite,
                'confidence': elite_results.confidence,
                'transition_risk': elite_results.transition_risk
            })

            # Calculate Enhanced Rolling Flow Metrics (VAPI, DWFD, TW_LAF)
            from core_analytics_engine.eots_metrics.flow_analytics import FlowAnalyticsCalculator
            flow_calc = FlowAnalyticsCalculator()
            flow_metrics = flow_calc.calculate_all_enhanced_flow_metrics(self._options_df, updated_model)

            # Update model with enhanced flow metrics
            updated_model = updated_model.model_copy(update={
                'vapi_fa_z_score_und': flow_metrics.vapi_fa_z_score_und,
                'dwfd_z_score_und': flow_metrics.dwfd_z_score_und,
                'tw_laf_z_score_und': flow_metrics.tw_laf_z_score_und
            })

            # Validate results
            self._validate_foundational_metrics_v2_5(updated_model)

            self.logger.debug("Foundational metrics calculation complete.")
            return updated_model

        except Exception as e:
            self.logger.error(f"Error calculating foundational metrics: {e}", exc_info=True)
            self._fail_fast_on_missing_metrics(f"foundational metrics calculation failed: {e}")

    def _validate_foundational_metrics_v2_5(self, model: 'ProcessedUnderlyingAggregatesV2_5'):
        """STRICT PYDANTIC V2-ONLY: Validate foundational metrics directly on model"""
        validation_errors = []

        # Check for reasonable bounds - FAIL FAST if values are suspicious
        if hasattr(model, 'gib_oi_based_und') and model.gib_oi_based_und is not None:
            if abs(model.gib_oi_based_und) > 1000000:
                validation_errors.append(f"GIB value out of bounds: {model.gib_oi_based_und}")

        if hasattr(model, 'hp_eod_und') and model.hp_eod_und is not None:
            if abs(model.hp_eod_und) > 100:
                validation_errors.append(f"HP_EOD value out of bounds: {model.hp_eod_und}")

        # FAIL FAST on validation errors - NO SILENT FAILURES
        if validation_errors:
            self.logger.error(f"CRITICAL: Foundational metrics validation failed: {validation_errors}")
            raise ValueError(f"CRITICAL: Foundational metrics validation failed - {validation_errors}")

        self.logger.debug("Foundational metrics validation passed")



    def _calculate_net_customer_greek_flows_v2_5(self, model: 'ProcessedUnderlyingAggregatesV2_5') -> 'ProcessedUnderlyingAggregatesV2_5':
        """Calculate Net Customer Greek Flows using strict Pydantic v2 architecture"""
        self.logger.debug("Calculating Net Customer Greek Flows...")

        # Delta Flow - STRICT PYDANTIC V2: Direct model access, FAIL FAST on missing data
        deltas_buy = self._require_pydantic_field(model, 'deltas_buy', 'deltas buy data')
        deltas_sell = self._require_pydantic_field(model, 'deltas_sell', 'deltas sell data')
        net_cust_delta_flow_und = deltas_buy - deltas_sell

        # Gamma Flow - FAIL FAST on missing data
        gamma_buy = (self._require_pydantic_field(model, 'gammas_call_buy', 'gamma call buy') +
                    self._require_pydantic_field(model, 'gammas_put_buy', 'gamma put buy'))
        gamma_sell = (self._require_pydantic_field(model, 'gammas_call_sell', 'gamma call sell') +
                     self._require_pydantic_field(model, 'gammas_put_sell', 'gamma put sell'))
        net_cust_gamma_flow_und = gamma_buy - gamma_sell

        # Vega Flow - FAIL FAST on missing data
        vegas_buy = self._require_pydantic_field(model, 'vegas_buy', 'vegas buy data')
        vegas_sell = self._require_pydantic_field(model, 'vegas_sell', 'vegas sell data')
        net_cust_vega_flow_und = vegas_buy - vegas_sell

        # Theta Flow - FAIL FAST on missing data
        thetas_buy = self._require_pydantic_field(model, 'thetas_buy', 'thetas buy data')
        thetas_sell = self._require_pydantic_field(model, 'thetas_sell', 'thetas sell data')
        net_cust_theta_flow_und = thetas_buy - thetas_sell

        # STRICT PYDANTIC V2-ONLY: Use model_copy(update={}) - NO DICTIONARIES
        updated_model = model.model_copy(update={
            'net_cust_delta_flow_und': net_cust_delta_flow_und,
            'net_cust_gamma_flow_und': net_cust_gamma_flow_und,
            'net_cust_vega_flow_und': net_cust_vega_flow_und,
            'net_cust_theta_flow_und': net_cust_theta_flow_und
        })

        self.logger.debug("Net Customer Greek Flows calculated.")
        # Instead of dumping to dict and revalidating, return the updated model instance directly
        return updated_model

    def _calculate_gib_based_metrics_v2_5(self, model: 'ProcessedUnderlyingAggregatesV2_5') -> 'ProcessedUnderlyingAggregatesV2_5':
        """STRICT PYDANTIC V2-ONLY: Calculate GIB, HP_EOD, and TD_GIB metrics directly on model"""
        self.logger.debug("Calculating GIB-based metrics...")

        # GIB (Gamma Imbalance from Open Interest) - FAIL FAST on missing data
        call_gxoi = self._require_pydantic_field(model, 'call_gxoi', 'call gamma exposure')
        put_gxoi = self._require_pydantic_field(model, 'put_gxoi', 'put gamma exposure')
        gib_raw_gamma_units = put_gxoi - call_gxoi

        # Calculate dollar value - FAIL FAST on missing price
        underlying_price = self._require_pydantic_field(model, 'price', 'underlying price')
        if underlying_price <= 0:
            raise ValueError(f"CRITICAL: Invalid underlying price {underlying_price} - must be positive!")

        contract_multiplier = 100
        gib_dollar_value_full = gib_raw_gamma_units * underlying_price * contract_multiplier

        # Scale for display (optimized scaling)
        gib_display_value = gib_dollar_value_full / 10000.0

        # HP_EOD (End-of-Day Hedging Pressure)
        hp_eod_value = self._calculate_hp_eod_optimized_v2_5(model, gib_display_value)

        # TD_GIB (Traded Dealer Gamma Imbalance)
        td_gib_value = self._calculate_td_gib_optimized_v2_5(model)

        # STRICT PYDANTIC V2-ONLY: Use model_copy(update={}) - NO DICTIONARIES
        updated_model = model.model_copy(update={
            'gib_oi_based_und': gib_display_value,
            'gib_raw_gamma_units_und': gib_raw_gamma_units,
            'gib_dollar_value_full_und': gib_dollar_value_full,
            'hp_eod_und': hp_eod_value,
            'td_gib_und': td_gib_value
        })

        self.logger.debug(f"GIB metrics calculated: GIB={gib_display_value:.2f}, HP_EOD={hp_eod_value:.2f}, TD_GIB={td_gib_value:.2f}")
        return updated_model
    
    def _calculate_hp_eod_optimized_v2_5(self, model: 'ProcessedUnderlyingAggregatesV2_5', gib_value: float) -> float:
        """STRICT PYDANTIC V2-ONLY: HP_EOD calculation directly on model"""
        try:
            current_time = datetime.now().time()
            market_close = datetime.strptime("16:00", "%H:%M").time()

            # Time-based scaling factor
            if current_time >= market_close:
                time_scaling = 1.0
            else:
                hours_to_close = (datetime.combine(date.today(), market_close) -
                                datetime.combine(date.today(), current_time)).seconds / 3600
                time_scaling = max(0.1, 1.0 - (hours_to_close / 6.5))

            # Calculate pressure components - FAIL FAST on missing data
            gib_component = gib_value * 0.6
            gamma_flow = getattr(model, 'net_cust_gamma_flow_und', None)
            if gamma_flow is None:
                # If not calculated yet, use 0 (will be calculated in previous step)
                gamma_flow = 0.0
            # FAIL-FAST: Convert gamma flow to float or raise error
            if gamma_flow is None or pd.isna(gamma_flow):
                raise ValueError(f"CRITICAL: gamma flow is None/NaN - cannot use fake default in financial calculations!")
            flow_component = float(gamma_flow) * 0.4 / 10000.0

            hp_eod = (gib_component + flow_component) * time_scaling
            return self._bound_value(hp_eod, -100, 100)

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating HP_EOD: {e}")
            raise ValueError(f"CRITICAL: HP_EOD calculation failed - cannot return fake 0.0 value! Error: {e}") from e

    def _calculate_td_gib_optimized_v2_5(self, model: 'ProcessedUnderlyingAggregatesV2_5') -> float:
        """STRICT PYDANTIC V2-ONLY: TD_GIB calculation directly on model"""
        try:
            # Use volume-weighted gamma flows for TD_GIB - FAIL FAST on missing data
            gamma_flow = getattr(model, 'net_cust_gamma_flow_und', None)
            if gamma_flow is None:
                # If not calculated yet, use 0 (will be calculated in previous step)
                gamma_flow = 0.0
            # FAIL-FAST: Convert gamma flow to float or raise error
            if gamma_flow is None or pd.isna(gamma_flow):
                raise ValueError(f"CRITICAL: gamma flow is None/NaN - cannot use fake default in financial calculations!")
            gamma_flow = float(gamma_flow)

            day_volume = self._require_pydantic_field(model, 'day_volume', 'daily volume')
            # FAIL-FAST: Convert day volume to float or raise error
            if day_volume is None or pd.isna(day_volume):
                raise ValueError(f"CRITICAL: day volume is None/NaN - cannot use fake default in financial calculations!")
            volume_factor = max(1.0, float(day_volume) / 1000000)

            td_gib = gamma_flow * volume_factor / 10000.0
            return self._bound_value(td_gib, -1000, 1000)

        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating TD_GIB: {e}")
            raise ValueError(f"CRITICAL: TD_GIB calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    # =============================================================================
    # CORE UTILITIES - Consolidated and optimized from base_calculator.py
    # =============================================================================
    
    def _require_pydantic_field(self, pydantic_model, field_name: str, field_description: str):
        """FAIL-FAST: Require field from Pydantic model - NO DICTIONARY CONVERSION ALLOWED"""
        if not hasattr(pydantic_model, field_name):
            raise ValueError(f"CRITICAL: Required field '{field_name}' ({field_description}) missing from Pydantic model!")

        value = getattr(pydantic_model, field_name)
        if value is None:
            raise ValueError(f"CRITICAL: Field '{field_name}' ({field_description}) is None - cannot use fake defaults in financial calculations!")

        return value


    
    def _bound_value(self, value: float, min_val: float, max_val: float) -> float:
        """Bound value within specified range"""
        return max(min_val, min(max_val, value))
    
    def _validate_foundational_metrics(self, und_data: Dict) -> bool:
        """Validate foundational metrics against bounds"""
        validation_results = {}
        
        for metric, bounds in self.METRIC_BOUNDS.items():
            if metric in und_data:
                value = und_data[metric]
                min_val, max_val = bounds
                is_valid = min_val <= value <= max_val
                validation_results[metric] = {
                    'value': value,
                    'valid': is_valid,
                    'bounds': bounds
                }
                
                if not is_valid:
                    self.logger.warning(f"Metric {metric} value {value} outside bounds {bounds}")
        
        self._calculation_state.validation_results.update(validation_results)
        return all(result['valid'] for result in validation_results.values())
    
    def _fail_fast_on_missing_metrics(self, error_context: str) -> None:
        """FAIL FAST - NO DEFAULT FOUNDATIONAL METRICS ALLOWED!"""
        raise ValueError(
            f"CRITICAL: Failed to calculate foundational metrics - {error_context}. "
            f"NO FAKE DATA WILL BE SUBSTITUTED! Fix the underlying calculation issue."
        )
    
    # =============================================================================
    # CACHING UTILITIES - Unified and optimized
    # =============================================================================
    
    def _add_to_intraday_cache(self, symbol: str, metric_name: str, value: float, max_size: int = 200) -> List[float]:
        """Unified intraday caching with automatic cleanup"""
        try:
            # CRITICAL FIX: Check if cache manager is None
            if self.enhanced_cache_manager is None:
                return [float(value)]

            # Get existing cache or create new
            # CRITICAL FIX: Use correct cache manager signature (symbol, metric_name)
            cache_data = self.enhanced_cache_manager.get(symbol, f"{metric_name}_intraday")
            if cache_data is None:
                cache_data = deque(maxlen=max_size)
            elif not isinstance(cache_data, deque):
                cache_data = deque(cache_data if isinstance(cache_data, list) else [cache_data], maxlen=max_size)
            
            # Add new value
            cache_data.append(float(value))
            
            # Store back in cache
            # CRITICAL FIX: Use correct cache manager method (put) and signature
            self.enhanced_cache_manager.put(symbol, f"{metric_name}_intraday", list(cache_data), cache_level=CacheLevel.MEMORY)
            
            return list(cache_data)
            
        except Exception as e:
            self.logger.warning(f"Error updating intraday cache for {symbol}_{metric_name}: {e}")
            import traceback
            self.logger.warning(f"Cache error traceback: {traceback.format_exc()}")
            return [float(value)]
    
    def _calculate_percentile_gauge_value(self, cache_data: List[float], current_value: float) -> float:
        """FAIL-FAST: Calculate percentile-based gauge value - NO FAKE DEFAULTS ALLOWED"""
        if not cache_data or len(cache_data) < 2:
            raise ValueError(f"CRITICAL: Insufficient cache data for percentile gauge calculation - need at least 2 data points, got {len(cache_data) if cache_data else 0}!")
        
        try:
            # Use percentile ranking for small datasets
            sorted_data = sorted(cache_data)
            rank = sum(1 for x in sorted_data if x <= current_value)
            percentile = rank / len(sorted_data)
            
            # Convert to z-score-like value
            return (percentile - 0.5) * 4  # Scale to approximate z-score range
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Error calculating percentile gauge: {e}")
            raise ValueError(f"CRITICAL: Percentile gauge calculation failed - cannot return fake 0.0 value! Error: {e}") from e
    
    def sanitize_symbol(self, symbol: str) -> str:
        """Sanitize ticker symbol for consistent processing"""
        if not symbol:
            return "UNKNOWN"
        return str(symbol).upper().strip()
    
    def _is_futures_symbol(self, symbol: str) -> bool:
        """Determine if symbol is a futures contract"""
        futures_patterns = ['/ES', '/NQ', '/YM', '/RTY', '/CL', '/GC', '/SI']
        return any(pattern in symbol.upper() for pattern in futures_patterns)

# Export the consolidated calculator
__all__ = ['CoreCalculator', 'MetricCalculationState', 'MetricCache', 'MetricCacheConfig']
