"""
Core Data Models for EOTS v2.5

Consolidated from: base_types.py, system_schemas.py, raw_data.py,
processed_data.py, bundle_schemas.py, context_schemas.py
"""

# Standard library imports
from typing import Any, TypeVar, Generic, List, Optional, Dict, Union, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum

# Third-party imports
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, GetJsonSchemaHandler, GetCoreSchemaHandler, field_validator, model_validator, FieldValidationInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

class EOTSBaseModel(BaseModel):
    """Base model for all EOTS Pydantic models, providing common configurations."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class MarketRegimeState(str, Enum):
    """Market regime states for the expert system.

    Attributes:
        BULLISH_TREND: Sustained upward price movement
        BEARISH_TREND: Sustained downward price movement
        SIDEWAYS: No clear trend, price oscillating in a range
        VOLATILITY_EXPANSION: Increasing price volatility
        VOLATILITY_CONTRACTION: Decreasing price volatility
        BULLISH_REVERSAL: Potential reversal from downtrend to uptrend
        BEARISH_REVERSAL: Potential reversal from uptrend to downtrend
        DISTRIBUTION: Smart money distributing positions
        ACCUMULATION: Smart money accumulating positions
        CAPITULATION: Panic selling
        EUPHORIA: Extreme bullish sentiment
        PANIC: Extreme bearish sentiment
        CONSOLIDATION: Price moving in a tight range
        BREAKOUT: Price breaking out of a range
        BREAKDOWN: Price breaking down from a range
        CHOPPY: Erratic price action
        TRENDING: Clear directional movement
        RANGE_BOUND: Price contained within support/resistance
        UNDEFINED: Default/unknown state
    """
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    SIDEWAYS = "sideways"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"
    CAPITULATION = "capitulation"
    EUPHORIA = "euphoria"
    PANIC = "panic"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CHOPPY = "choppy"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    UNDEFINED = "undefined"


class DataFrameSchema(Generic[TypeVar('T')]):
    """
    Pydantic v2 custom type for pandas DataFrames with schema validation.
    
    Usage:
        class MyModel(BaseModel):
            df: DataFrameSchema[MyRowModel]  # Validates each row against MyRowModel
    
    Attributes:
        schema: The Pydantic model to validate each row against
        strict: If True, raises ValidationError if any rows fail validation
    """
    def __class_getitem__(cls, item):
        return type(f'DataFrameSchema[{item.__name__}]', (cls,), {'__schema__': item})
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = cls.__dict__.get('__schema__')
        if schema is None:
            raise ValueError("DataFrameSchema must be parameterized with a row type")
            
        def validate(v: Any) -> pd.DataFrame:
            if not isinstance(v, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(v).__name__}")
                
            # Convert each row to the schema and back to catch validation errors
            try:
                rows = [schema.model_validate(row.to_dict()) for _, row in v.iterrows()]
                return pd.DataFrame([row.model_dump() for row in rows])
            except Exception as e:
                raise ValueError(f"DataFrame validation failed: {str(e)}")
        
        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda df: df.to_dict('records'),
                when_used='json',
            ),
        )
    
    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = cls.__dict__.get('__schema__')
        return {
            'type': 'array',
            'items': handler(schema.__pydantic_core_schema__) if hasattr(schema, '__pydantic_core_schema__') else {},
            'title': f'DataFrame[{schema.__name__ if hasattr(schema, "__name__") else str(schema)}]',
        }


# Backward compatible type alias that can be used when full schema validation isn't needed
PandasDataFrame = pd.DataFrame

# Note: This file can be expanded with other common simple type aliases
# or very basic shared Pydantic models if they arise in the future.
# For now, it primarily serves to define PandasDataFrame for type hinting.

# Example of another base type if needed in the future:
# class Identifier(BaseModel):
#     id: str = Field(..., description="A unique identifier.")
#     system: Optional[str] = Field(None, description="The system or namespace of the identifier.")

"""
Base type definitions used across various EOTS v2.5 data models.
This module centralizes common type aliases or very simple base Pydantic models
to promote consistency and ease of maintenance.
"""


# =============================================================================
# SYSTEM STATE AND HEALTH MODELS (from system_schemas.py)
# =============================================================================


class SystemComponentStatuses(BaseModel):
    """Status of individual system components - FAIL FAST ON MISSING STATUS."""
    database: str = Field(..., description="Database status - REQUIRED (e.g., 'connected', 'disconnected', 'error')")
    api_connections: str = Field(..., description="API connections status - REQUIRED (e.g., 'active', 'failed', 'partial')")
    data_processing: str = Field(..., description="Data processing status - REQUIRED (e.g., 'running', 'stopped', 'error')")

    @field_validator('database', 'api_connections', 'data_processing')
    @classmethod
    def validate_no_unknown_status(cls, v: str, info: FieldValidationInfo) -> str:
        """CRITICAL: Reject 'unknown' status values that indicate missing real system data."""
        if v.lower() in ['unknown', 'n/a', 'na', '']:
            raise ValueError(f"CRITICAL: {info.field_name} status cannot be 'unknown' - this indicates missing real system status!")
        return v

    model_config = ConfigDict(extra='forbid')


class SystemStateV2_5(BaseModel):
    """Represents the overall operational state of the EOTS v2.5 system."""
    is_running: bool = Field(..., description="True if the system is actively running, False otherwise.")
    current_mode: str = Field(..., description="Current operational mode (e.g., 'operational', 'maintenance', 'diagnostic').")
    active_processes: List[str] = Field(..., min_items=1, description="List of currently active system processes or modules - REQUIRED and must not be empty")
    status_message: str = Field(..., min_length=1, description="A human-readable message describing the current system status - REQUIRED")
    last_heartbeat: Optional[str] = Field(None, description="Timestamp of the last successful system heartbeat - optional")
    errors: List[str] = Field(..., description="List of recent critical errors or warnings - REQUIRED (empty list if no errors)")

    @field_validator('active_processes')
    @classmethod
    def validate_active_processes_not_empty(cls, v: List[str]) -> List[str]:
        """CRITICAL: Ensure active processes list is not empty - empty list indicates system not running."""
        if not v:
            raise ValueError("CRITICAL: active_processes cannot be empty - this indicates system is not running!")
        return v

    model_config = ConfigDict(extra='forbid')

class AISystemHealthV2_5(BaseModel):
    """Comprehensive Pydantic model for AI system health monitoring - FAIL FAST ON MISSING HEALTH DATA."""
    # Database connectivity - REQUIRED
    database_connected: bool = Field(..., description="Database connection status - REQUIRED")
    ai_tables_available: bool = Field(..., description="AI tables availability status - REQUIRED")

    # Component health - REQUIRED
    predictions_manager_healthy: bool = Field(..., description="AI Predictions Manager health - REQUIRED")
    learning_system_healthy: bool = Field(..., description="AI Learning System health - REQUIRED")
    adaptation_engine_healthy: bool = Field(..., description="AI Adaptation Engine health - REQUIRED")

    # Performance metrics - REQUIRED
    overall_health_score: float = Field(..., ge=0.0, le=1.0, description="Overall system health score - REQUIRED")
    response_time_ms: float = Field(..., ge=0.0, description="System response time in milliseconds - REQUIRED")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="System error rate - REQUIRED")

    # Status details - REQUIRED
    status_message: str = Field(..., min_length=1, description="Current status message - REQUIRED")
    component_status: SystemComponentStatuses = Field(..., description="Detailed status of system components - REQUIRED")
    last_checked: datetime = Field(..., description="Last health check timestamp - REQUIRED")

    @field_validator('status_message')
    @classmethod
    def validate_no_generic_status(cls, v: str) -> str:
        """CRITICAL: Reject generic status messages that indicate missing real system data."""
        generic_messages = ['system initializing', 'unknown', 'n/a', 'default', 'placeholder']
        if v.lower().strip() in generic_messages:
            raise ValueError(f"CRITICAL: Status message '{v}' is generic - provide real system status!")
        return v

    model_config = ConfigDict(
        extra='allow',
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

# =============================================================================
# RAW DATA MODELS (from raw_data.py)
# =============================================================================
"""
Pydantic models for raw, unprocessed data as fetched from external sources
before significant EOTS v2.5 processing. These schemas are designed to
closely mirror the structure of the source APIs (e.g., ConvexValue)
and include `Config.extra = 'allow'` to accommodate potential new fields
from the API without breaking the system.
"""


# --- Canonical Parameter Lists from ConvexValue ---
# For reference and ensuring Raw models are comprehensive.
# These lists help map schema fields back to their expected source in the CV API.
UNDERLYING_REQUIRED_PARAMS_CV = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]

OPTIONS_CHAIN_REQUIRED_PARAMS_CV = [
    "price", "volatility", "multiplier", "oi", "delta", "gamma", "theta", "vega",
    "vanna", "vomma", "charm", "dxoi", "gxoi", "vxoi", "txoi", "vannaxoi",
    "vommaxoi", "charmxoi", "dxvolm", "gxvolm", "vxvolm", "txvolm", "vannaxvolm",
    "vommaxvolm", "charmxvolm", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "gammas_buy", "gammas_sell", "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell",
    "valuebs_5m", "volmbs_5m", "valuebs_15m", "volmbs_15m",
    "valuebs_30m", "volmbs_30m", "valuebs_60m", "volmbs_60m",
    "volm", "volm_buy", "volm_sell", "value_buy", "value_sell"
]
# End Canonical Parameter Lists


class RawOptionsContractV2_5(BaseModel):
    """
    Represents the raw, unprocessed data for a single options contract as fetched
    directly from the primary data source (e.g., ConvexValue API `get_chain`).
    It serves as the foundational data structure for an individual option contract
    before any cleaning, transformation, or metric calculation.
    """
    contract_symbol: str = Field(..., description="Unique identifier for the option contract.", min_length=1)
    strike: float = Field(..., description="Strike price of the option.", gt=0.0)

    @field_validator('strike')
    @classmethod
    def validate_strike_not_fake(cls, v: float) -> float:
        """CRITICAL: Reject suspicious strike prices that indicate fake data."""
        # Reject extremely low strikes that are likely fake
        if v < 0.01:
            raise ValueError(f"CRITICAL: Strike price {v} is suspiciously low - likely fake data!")
        return v
    opt_kind: str = Field(..., description="Type of option, typically 'call' or 'put'.")
    dte_calc: float = Field(..., description="Calculated Days To Expiration for the contract.", ge=0.0)

    # Fields corresponding to OPTIONS_CHAIN_REQUIRED_PARAMS_CV from ConvexValue
    open_interest: Optional[float] = Field(None, description="Open interest for the contract (Source: CV 'oi').")
    iv: Optional[float] = Field(None, description="Implied Volatility for the contract (Source: CV 'volatility').")
    raw_price: Optional[float] = Field(None, description="Raw price of the option contract (Source: CV 'price').")
    delta_contract: Optional[float] = Field(None, description="Delta per contract (Source: CV 'delta').")
    gamma_contract: Optional[float] = Field(None, description="Gamma per contract (Source: CV 'gamma').")
    theta_contract: Optional[float] = Field(None, description="Theta per contract (Source: CV 'theta').")
    vega_contract: Optional[float] = Field(None, description="Vega per contract (Source: CV 'vega').")
    rho_contract: Optional[float] = Field(None, description="Rho per contract (Standard Greek, may not be in all CV responses).")
    vanna_contract: Optional[float] = Field(None, description="Vanna per contract (Source: CV 'vanna').")
    vomma_contract: Optional[float] = Field(None, description="Vomma per contract (Source: CV 'vomma').")
    charm_contract: Optional[float] = Field(None, description="Charm per contract (Source: CV 'charm').")

    # Greeks OI (Open Interest based Greeks)
    dxoi: Optional[float] = Field(None, description="Delta Open Interest Exposure (Source: CV 'dxoi').")
    gxoi: Optional[float] = Field(None, description="Gamma Open Interest Exposure (Source: CV 'gxoi').")
    vxoi: Optional[float] = Field(None, description="Vega Open Interest Exposure (Source: CV 'vxoi').")
    txoi: Optional[float] = Field(None, description="Theta Open Interest Exposure (Source: CV 'txoi').")
    vannaxoi: Optional[float] = Field(None, description="Vanna Open Interest Exposure (Source: CV 'vannaxoi').")
    vommaxoi: Optional[float] = Field(None, description="Vomma Open Interest Exposure (Source: CV 'vommaxoi').")
    charmxoi: Optional[float] = Field(None, description="Charm Open Interest Exposure (Source: CV 'charmxoi').")

    # Greek-Volume Proxies
    dxvolm: Optional[Any] = Field(None, description="Delta-Weighted Volume (Source: CV 'dxvolm'). Type can vary based on source.")
    gxvolm: Optional[Any] = Field(None, description="Gamma-Weighted Volume (Source: CV 'gxvolm'). Type can vary.")
    vxvolm: Optional[Any] = Field(None, description="Vega-Weighted Volume (Source: CV 'vxvolm'). Type can vary.")
    txvolm: Optional[Any] = Field(None, description="Theta-Weighted Volume (Source: CV 'txvolm'). Type can vary.")
    vannaxvolm: Optional[Any] = Field(None, description="Vanna-Weighted Volume (Source: CV 'vannaxvolm'). Type can vary.")
    vommaxvolm: Optional[Any] = Field(None, description="Vomma-Weighted Volume (Source: CV 'vommaxvolm'). Type can vary.")
    charmxvolm: Optional[Any] = Field(None, description="Charm-Weighted Volume (Source: CV 'charmxvolm'). Type can vary.")

    # Transaction Data
    value_bs: Optional[float] = Field(None, description="Day Sum of Buy Value minus Sell Value Traded (Source: CV 'value_bs').")
    volm_bs: Optional[float] = Field(None, description="Volume of Buys minus Sells for the day (Source: CV 'volm_bs').")
    volm: Optional[float] = Field(None, description="Total daily volume for the contract (Source: CV 'volm').")

    # Rolling Flows
    valuebs_5m: Optional[float] = Field(None, description="Net signed value traded in the last 5 minutes (Source: CV 'valuebs_5m').")
    volmbs_5m: Optional[float] = Field(None, description="Net signed volume traded in the last 5 minutes (Source: CV 'volmbs_5m').")
    valuebs_15m: Optional[float] = Field(None, description="Net signed value traded in the last 15 minutes (Source: CV 'valuebs_15m').")
    volmbs_15m: Optional[float] = Field(None, description="Net signed volume traded in the last 15 minutes (Source: CV 'volmbs_15m').")
    valuebs_30m: Optional[float] = Field(None, description="Net signed value traded in the last 30 minutes (Source: CV 'valuebs_30m').")
    volmbs_30m: Optional[float] = Field(None, description="Net signed volume traded in the last 30 minutes (Source: CV 'volmbs_30m').")
    valuebs_60m: Optional[float] = Field(None, description="Net signed value traded in the last 60 minutes (Source: CV 'valuebs_60m').")
    volmbs_60m: Optional[float] = Field(None, description="Net signed volume traded in the last 60 minutes (Source: CV 'volmbs_60m').")

    # Bid/Ask for liquidity calculations
    bid_price: Optional[float] = Field(None, description="Current bid price of the option.")
    ask_price: Optional[float] = Field(None, description="Current ask price of the option.")
    mid_price: Optional[float] = Field(None, description="Calculated midpoint price of the option (bid/ask).")

    # New fields from OPTIONS_CHAIN_REQUIRED_PARAMS_CV not previously explicitly listed
    multiplier: Optional[float] = Field(None, description="Option contract multiplier, e.g., 100 (Source: CV 'multiplier').")
    deltas_buy: Optional[Any] = Field(None, description="Aggregated delta of buy orders (Source: CV 'deltas_buy'). Type can vary.")
    deltas_sell: Optional[Any] = Field(None, description="Aggregated delta of sell orders (Source: CV 'deltas_sell'). Type can vary.")
    gammas_buy: Optional[Any] = Field(None, description="Aggregated gamma of buy orders (Source: CV 'gammas_buy'). Type can vary.")
    gammas_sell: Optional[Any] = Field(None, description="Aggregated gamma of sell orders (Source: CV 'gammas_sell'). Type can vary.")
    vegas_buy: Optional[Any] = Field(None, description="Aggregated vega of buy orders (Source: CV 'vegas_buy'). Type can vary.")
    vegas_sell: Optional[Any] = Field(None, description="Aggregated vega of sell orders (Source: CV 'vegas_sell'). Type can vary.")
    thetas_buy: Optional[Any] = Field(None, description="Aggregated theta of buy orders (Source: CV 'thetas_buy'). Type can vary.")
    thetas_sell: Optional[Any] = Field(None, description="Aggregated theta of sell orders (Source: CV 'thetas_sell'). Type can vary.")
    volm_buy: Optional[Any] = Field(None, description="Total buy volume (Source: CV 'volm_buy'). Type can vary.")
    volm_sell: Optional[Any] = Field(None, description="Total sell volume (Source: CV 'volm_sell'). Type can vary.")
    value_buy: Optional[Any] = Field(None, description="Total value of buy orders (Source: CV 'value_buy'). Type can vary.")
    value_sell: Optional[Any] = Field(None, description="Total value of sell orders (Source: CV 'value_sell'). Type can vary.")

    model_config = ConfigDict(extra='allow')  # Accommodate potential new fields from API

    @field_validator('opt_kind')
    @classmethod
    def validate_opt_kind(cls, v):
        """Validate option type is either 'call' or 'put'."""
        if v.lower() not in ['call', 'put']:
            raise ValueError("opt_kind must be either 'call' or 'put'")
        return v.lower()

    @field_validator('iv')
    @classmethod
    def validate_implied_volatility(cls, v):
        """Validate implied volatility is reasonable."""
        if v is not None and (v < 0.0 or v > 10.0):
            raise ValueError("Implied volatility must be between 0.0 and 10.0")
        return v

    @field_validator('delta_contract')
    @classmethod
    def validate_delta(cls, v):
        """Validate delta is within reasonable bounds."""
        if v is not None and (v < -1.0 or v > 1.0):
            raise ValueError("Delta must be between -1.0 and 1.0")
        return v

    @field_validator('gamma_contract', 'vega_contract')
    @classmethod
    def validate_positive_greeks(cls, v):
        """Validate gamma and vega are non-negative."""
        if v is not None and v < 0.0:
            raise ValueError(f"{cls.__name__} must be non-negative")
        return v


class RawUnderlyingDataV2_5(BaseModel):
    """
    Represents raw, unprocessed data for the underlying asset (e.g., stock, index)
    primarily from a source like ConvexValue `get_und` endpoint. This serves as
    the initial container for underlying-specific market data before enrichment
    from other sources or metric calculation.
    """
    symbol: str = Field(..., description="Ticker symbol of the underlying asset.", min_length=1)
    timestamp: datetime = Field(..., description="Timestamp of when this underlying data was fetched or is valid for.")
    price: float = Field(..., description="Current market price of the underlying asset - REQUIRED", gt=0.0)

    @field_validator('price')
    @classmethod
    def validate_price_not_fake(cls, v: float) -> float:
        """CRITICAL: Reject fake or suspicious price values."""
        if v <= 0.0:
            raise ValueError("CRITICAL: Price must be positive - zero or negative prices indicate fake data!")
        if v < 0.01:
            raise ValueError(f"CRITICAL: Price {v} is suspiciously low - likely fake data!")
        return v
    price_change_abs_und: Optional[float] = Field(None, description="Absolute price change of the underlying for the current trading session.")
    price_change_pct_und: Optional[float] = Field(None, description="Percentage price change of the underlying for the current trading session.")

    # OHLC data, typically from a secondary source like Tradier if not in primary underlying feed
    day_open_price_und: Optional[float] = Field(None, description="Daily open price of the underlying from primary/secondary source.")
    day_high_price_und: Optional[float] = Field(None, description="Daily high price of the underlying from primary/secondary source.")
    day_low_price_und: Optional[float] = Field(None, description="Daily low price of the underlying from primary/secondary source.")
    prev_day_close_price_und: Optional[float] = Field(None, description="Previous trading day's closing price of the underlying from primary/secondary source.")

    # Fields from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV
    u_volatility: Optional[float] = Field(None, description="General Implied Volatility for the underlying asset (Source: CV 'volatility').")
    day_volume: Optional[Any] = Field(None, description="Total daily volume for the underlying asset (Source: CV 'day_volume'). Type can vary.")
    call_gxoi: Optional[Any] = Field(None, description="Aggregate Gamma Open Interest Exposure for call options (Source: CV 'call_gxoi'). Type can vary.")
    put_gxoi: Optional[Any] = Field(None, description="Aggregate Gamma Open Interest Exposure for put options (Source: CV 'put_gxoi'). Type can vary.")
    gammas_call_buy: Optional[Any] = Field(None, description="Aggregated gamma of call buy orders (Source: CV 'gammas_call_buy'). Type can vary.")
    gammas_call_sell: Optional[Any] = Field(None, description="Aggregated gamma of call sell orders (Source: CV 'gammas_call_sell'). Type can vary.")
    gammas_put_buy: Optional[Any] = Field(None, description="Aggregated gamma of put buy orders (Source: CV 'gammas_put_buy'). Type can vary.")
    gammas_put_sell: Optional[Any] = Field(None, description="Aggregated gamma of put sell orders (Source: CV 'gammas_put_sell'). Type can vary.")
    deltas_call_buy: Optional[Any] = Field(None, description="Aggregated delta of call buy orders (Source: CV 'deltas_call_buy'). Type can vary.")
    deltas_call_sell: Optional[Any] = Field(None, description="Aggregated delta of call sell orders (Source: CV 'deltas_call_sell'). Type can vary.")
    deltas_put_buy: Optional[Any] = Field(None, description="Aggregated delta of put buy orders (Source: CV 'deltas_put_buy'). Type can vary.")
    deltas_put_sell: Optional[Any] = Field(None, description="Aggregated delta of put sell orders (Source: CV 'deltas_put_sell'). Type can vary.")
    vegas_call_buy: Optional[Any] = Field(None, description="Aggregated vega of call buy orders (Source: CV 'vegas_call_buy'). Type can vary.")
    vegas_call_sell: Optional[Any] = Field(None, description="Aggregated vega of call sell orders (Source: CV 'vegas_call_sell'). Type can vary.")
    vegas_put_buy: Optional[Any] = Field(None, description="Aggregated vega of put buy orders (Source: CV 'vegas_put_buy'). Type can vary.")
    vegas_put_sell: Optional[Any] = Field(None, description="Aggregated vega of put sell orders (Source: CV 'vegas_put_sell'). Type can vary.")
    thetas_call_buy: Optional[Any] = Field(None, description="Aggregated theta of call buy orders (Source: CV 'thetas_call_buy'). Type can vary.")
    thetas_call_sell: Optional[Any] = Field(None, description="Aggregated theta of call sell orders (Source: CV 'thetas_call_sell'). Type can vary.")
    thetas_put_buy: Optional[Any] = Field(None, description="Aggregated theta of put buy orders (Source: CV 'thetas_put_buy'). Type can vary.")
    thetas_put_sell: Optional[Any] = Field(None, description="Aggregated theta of put sell orders (Source: CV 'thetas_put_sell'). Type can vary.")
    call_vxoi: Optional[Any] = Field(None, description="Aggregate Vega Open Interest Exposure for call options (Source: CV 'call_vxoi'). Type can vary.")
    put_vxoi: Optional[Any] = Field(None, description="Aggregate Vega Open Interest Exposure for put options (Source: CV 'put_vxoi'). Type can vary.")
    value_bs: Optional[Any] = Field(None, description="Overall net signed value traded for the underlying's options (Source: CV 'value_bs'). Type can vary.")
    volm_bs: Optional[Any] = Field(None, description="Overall net signed volume traded for the underlying's options (Source: CV 'volm_bs'). Type can vary.")
    deltas_buy: Optional[Any] = Field(None, description="Overall aggregated delta of buy orders for the underlying's options (Source: CV 'deltas_buy'). Type can vary.")
    deltas_sell: Optional[Any] = Field(None, description="Overall aggregated delta of sell orders for the underlying's options (Source: CV 'deltas_sell'). Type can vary.")
    vegas_buy: Optional[Any] = Field(None, description="Overall aggregated vega of buy orders (Source: CV 'vegas_buy'). Type can vary.")
    vegas_sell: Optional[Any] = Field(None, description="Overall aggregated vega of sell orders (Source: CV 'vegas_sell'). Type can vary.")
    thetas_buy: Optional[Any] = Field(None, description="Overall aggregated theta of buy orders (Source: CV 'thetas_buy'). Type can vary.")
    thetas_sell: Optional[Any] = Field(None, description="Overall aggregated theta of sell orders (Source: CV 'thetas_sell'). Type can vary.")
    volm_call_buy: Optional[Any] = Field(None, description="Total buy volume for call options (Source: CV 'volm_call_buy'). Type can vary.")
    volm_put_buy: Optional[Any] = Field(None, description="Total buy volume for put options (Source: CV 'volm_put_buy'). Type can vary.")
    volm_call_sell: Optional[Any] = Field(None, description="Total sell volume for call options (Source: CV 'volm_call_sell'). Type can vary.")
    volm_put_sell: Optional[Any] = Field(None, description="Total sell volume for put options (Source: CV 'volm_put_sell'). Type can vary.")
    value_call_buy: Optional[Any] = Field(None, description="Total value of call buy orders (Source: CV 'value_call_buy'). Type can vary.")
    value_put_buy: Optional[Any] = Field(None, description="Total value of put buy orders (Source: CV 'value_put_buy'). Type can vary.")
    value_call_sell: Optional[Any] = Field(None, description="Total value of call sell orders (Source: CV 'value_call_sell'). Type can vary.")
    value_put_sell: Optional[Any] = Field(None, description="Total value of put sell orders (Source: CV 'value_put_sell'). Type can vary.")
    vflowratio: Optional[Any] = Field(None, description="Ratio of Vanna flow to Vega flow (Source: CV 'vflowratio'). Type can vary.")
    dxoi: Optional[Any] = Field(None, description="Overall Delta Open Interest Exposure for the underlying (Source: CV 'dxoi'). Type can vary.")
    gxoi: Optional[Any] = Field(None, description="Overall Gamma Open Interest Exposure for the underlying (Source: CV 'gxoi'). Type can vary.")
    vxoi: Optional[Any] = Field(None, description="Overall Vega Open Interest Exposure for the underlying (Source: CV 'vxoi'). Type can vary.")
    txoi: Optional[Any] = Field(None, description="Overall Theta Open Interest Exposure for the underlying (Source: CV 'txoi'). Type can vary.")
    call_dxoi: Optional[Any] = Field(None, description="Aggregate Delta Open Interest Exposure for call options (Source: CV 'call_dxoi'). Type can vary.")
    put_dxoi: Optional[Any] = Field(None, description="Aggregate Delta Open Interest Exposure for put options (Source: CV 'put_dxoi'). Type can vary.")

    # Other pre-existing fields that might be populated from various sources
    tradier_iv5_approx_smv_avg: Optional[float] = Field(None, description="Tradier IV5 approximation (SMV_VOL based).")
    total_call_oi_und: Optional[float] = Field(None, description="Total call Open Interest for the underlying (may be summed from chain or from source).")
    total_put_oi_und: Optional[float] = Field(None, description="Total put Open Interest for the underlying.")
    total_call_vol_und: Optional[float] = Field(None, description="Total call volume for the underlying.")
    total_put_vol_und: Optional[float] = Field(None, description="Total put volume for the underlying.")

    model_config = ConfigDict(extra='allow')  # Accommodate potential new fields from API


class RawUnderlyingDataCombinedV2_5(RawUnderlyingDataV2_5):
    """
    A consolidated data structure that combines raw underlying data from the primary
    source (e.g., ConvexValue `get_und`) with supplementary underlying data,
    typically OHLCV (Open, High, Low, Close, Volume) and VWAP (Volume Weighted Average Price),
    from a secondary source like Tradier. This ensures all necessary raw underlying
    information is available in a single object for downstream processing.
    """
    tradier_open: Optional[float] = Field(None, description="Tradier daily open price for the underlying.")
    tradier_high: Optional[float] = Field(None, description="Tradier daily high price for the underlying.")
    tradier_low: Optional[float] = Field(None, description="Tradier daily low price for the underlying.")
    tradier_close: Optional[float] = Field(None, description="Tradier daily close price (typically previous day's close if fetched mid-day).")
    tradier_volume: Optional[float] = Field(None, description="Tradier daily volume for the underlying.")
    tradier_vwap: Optional[float] = Field(None, description="Tradier daily Volume Weighted Average Price (VWAP) for the underlying.")

    model_config = ConfigDict(extra='allow')  # Accommodate fields from both sources


class UnprocessedDataBundleV2_5(BaseModel):
    """
    Serves as a container for all raw data fetched at the beginning of an analysis cycle,
    before any significant processing or metric calculation by EOTS v2.5 occurs.
    It bundles the list of raw options contracts and the combined raw underlying data,
    along with metadata about the fetch operation (timestamp, errors).
    """
    options_contracts: List[RawOptionsContractV2_5] = Field(default_factory=list, description="List of all raw options contracts fetched for the current analysis cycle.")
    underlying_data: RawUnderlyingDataCombinedV2_5 = Field(..., description="The combined raw data for the underlying asset from various sources.")
    fetch_timestamp: datetime = Field(..., description="Timestamp indicating when the data fetching process for this bundle was completed.")
    errors: List[str] = Field(default_factory=list, description="A list to store any error messages encountered during data fetching.")

    model_config = ConfigDict(
        extra='forbid',  # This bundle is internally constructed, should not have extra fields.
        arbitrary_types_allowed=True  # To allow underlying_data which might contain complex types if not fully parsed to RawUnderlyingDataCombinedV2_5 initially
    )


# =============================================================================
# PROCESSED DATA MODELS (from processed_data.py)
# =============================================================================
"""
Pydantic models for data that has undergone initial processing and metric
calculation by EOTS v2.5. These schemas represent enriched versions of
raw data, including calculated metrics at contract, strike, and underlying levels.
"""

# Import from trading_market_models to avoid circular imports
if TYPE_CHECKING:
    from .trading_market_models import TickerContextDictV2_5, DynamicThresholdsV2_5
else:
    # Runtime imports to avoid circular dependencies
    TickerContextDictV2_5 = Any
    DynamicThresholdsV2_5 = Any


class ProcessedContractMetricsV2_5(RawOptionsContractV2_5):
    """
    Represents an individual option contract after initial, contract-level
    metric calculations have been performed by MetricsCalculatorV2_5.
    It extends RawOptionsContractV2_5 with specific metrics,
    particularly those relevant to the 0DTE suite.
    """
    # 0DTE Suite metrics calculated per contract
    vri_0dte_contract: Optional[float] = Field(None, description="Calculated 0DTE Volatility Regime Indicator for this specific contract.")
    vfi_0dte_contract: Optional[float] = Field(None, description="Calculated 0DTE Volatility Flow Indicator for this contract.")
    vvr_0dte_contract: Optional[float] = Field(None, description="Calculated 0DTE Vanna-Vomma Ratio for this contract.")
    # Note: Other per-contract calculated metrics can be added here.

    model_config = ConfigDict(extra='forbid')  # Stricter, as this is processed internal data


class ProcessedStrikeLevelMetricsV2_5(BaseModel):
    """
    Consolidates all relevant Open Interest exposures, net customer Greek flows,
    transactional pressures (NVP), and calculated adaptive/structural metrics
    at each individual strike price. This forms the basis for identifying key
    levels and understanding structural market dynamics.
    """
    strike: float = Field(..., description="The strike price this data pertains to.")

    # Aggregated OI-weighted Greeks
    total_dxoi_at_strike: Optional[float] = Field(None, description="Total Delta Open Interest Exposure at this strike.")
    total_gxoi_at_strike: Optional[float] = Field(None, description="Total Gamma Open Interest Exposure at this strike.")
    total_vxoi_at_strike: Optional[float] = Field(None, description="Total Vega Open Interest Exposure at this strike.")
    total_txoi_at_strike: Optional[float] = Field(None, description="Total Theta Open Interest Exposure at this strike.")
    total_charmxoi_at_strike: Optional[float] = Field(None, description="Total Charm Open Interest Exposure at this strike.")
    total_vannaxoi_at_strike: Optional[float] = Field(None, description="Total Vanna Open Interest Exposure at this strike.")
    total_vommaxoi_at_strike: Optional[float] = Field(None, description="Total Vomma Open Interest Exposure at this strike.")

    # Net Customer Greek Flows (Daily Total at Strike)
    net_cust_delta_flow_at_strike: Optional[float] = Field(None, description="Net customer-initiated Delta flow at this strike.")
    net_cust_gamma_flow_at_strike: Optional[float] = Field(None, description="Net customer-initiated Gamma flow at this strike.")
    net_cust_vega_flow_at_strike: Optional[float] = Field(None, description="Net customer-initiated Vega flow at this strike.")
    net_cust_theta_flow_at_strike: Optional[float] = Field(None, description="Net customer-initiated Theta flow at this strike.")
    net_cust_charm_flow_proxy_at_strike: Optional[float] = Field(None, description="Net customer-initiated Charm flow (volume proxy) at this strike.")
    net_cust_vanna_flow_proxy_at_strike: Optional[float] = Field(None, description="Net customer-initiated Vanna flow (volume proxy) at this strike.")

    # Transactional Pressures
    nvp_at_strike: Optional[float] = Field(None, description="Net Value Pressure (signed premium) at this strike.")
    nvp_vol_at_strike: Optional[float] = Field(None, description="Net Volume Pressure (signed contracts) at this strike.")

    # Adaptive Metrics (Tier 2)
    a_dag_strike: Optional[float] = Field(None, description="Adaptive Delta Adjusted Gamma Exposure at this strike.")
    e_sdag_mult_strike: Optional[float] = Field(None, description="Enhanced SDAG (Multiplicative) at this strike.")
    e_sdag_dir_strike: Optional[float] = Field(None, description="Enhanced SDAG (Directional) at this strike.")
    e_sdag_w_strike: Optional[float] = Field(None, description="Enhanced SDAG (Weighted) at this strike.")
    e_sdag_vf_strike: Optional[float] = Field(None, description="Enhanced SDAG (Volatility-Focused) at this strike, signals Volatility Trigger.")
    d_tdpi_strike: Optional[float] = Field(None, description="Dynamic Time Decay Pressure Indicator at this strike.")
    e_ctr_strike: Optional[float] = Field(None, description="Enhanced Charm Decay Rate (derived from D-TDPI components) at this strike.")
    e_tdfi_strike: Optional[float] = Field(None, description="Enhanced Time Decay Flow Imbalance (derived from D-TDPI components) at this strike.")
    vri_2_0_strike: Optional[float] = Field(None, description="Volatility Regime Indicator Version 2.0 at this strike.")
    e_vvr_sens_strike: Optional[float] = Field(None, description="Enhanced Vanna-Vomma Ratio (Sensitivity version from VRI 2.0) at this strike.")
    e_vfi_sens_strike: Optional[float] = Field(None, description="Enhanced Volatility Flow Indicator (Sensitivity version from VRI 2.0) at this strike.")

    # Other Structural/Flow Metrics
    arfi_strike: Optional[float] = Field(None, description="Average Relative Flow Index calculated for this strike.")

    # Enhanced Heatmap Data Scores
    sgdhp_score_strike: Optional[float] = Field(None, description="Super Gamma-Delta Hedging Pressure score for this strike.")
    ugch_score_strike: Optional[float] = Field(None, description="Ultimate Greek Confluence score for this strike.")
    # ivsdh_score_strike: Optional[float] = Field(None, description="Integrated Volatility Surface Dynamics score (if aggregated to strike).")

    # Elite Impact Metrics (Strike-level)
    elite_impact_score: Optional[float] = Field(None, description="Master composite elite impact score for this strike.")
    sdag_consensus: Optional[float] = Field(None, description="Consensus SDAG score across all methodologies for this strike.")
    dag_consensus: Optional[float] = Field(None, description="Consensus DAG score across all methodologies for this strike.")
    prediction_confidence: Optional[float] = Field(None, description="Confidence level (0-1) for the elite impact prediction at this strike.")
    signal_strength: Optional[float] = Field(None, description="Magnitude of the elite impact signal (0-1) at this strike.")
    strike_magnetism_index: Optional[float] = Field(None, description="Gamma wall strength / Strike Magnetism Index for this strike.")
    volatility_pressure_index: Optional[float] = Field(None, description="Volatility Pressure Index for this strike.")
    cross_exp_gamma_surface: Optional[float] = Field(None, description="Cross-expiration gamma surface value for this strike.")
    expiration_transition_factor: Optional[float] = Field(None, description="Expiration transition factor for this strike.")
    regime_adjusted_gamma: Optional[float] = Field(None, description="Regime-adjusted gamma impact for this strike.")
    regime_adjusted_delta: Optional[float] = Field(None, description="Regime-adjusted delta impact for this strike.")
    regime_adjusted_vega: Optional[float] = Field(None, description="Regime-adjusted vega impact for this strike.")
    vanna_impact_raw: Optional[float] = Field(None, description="Raw Vanna impact for this strike.")
    vomma_impact_raw: Optional[float] = Field(None, description="Raw Vomma impact for this strike.")
    charm_impact_raw: Optional[float] = Field(None, description="Raw Charm impact for this strike.")

    model_config = ConfigDict(extra='allow')  # Allow extra fields for flexible configuration and testing

    @field_validator('prediction_confidence', 'signal_strength')
    @classmethod
    def validate_confidence_scores(cls, v):
        """Validate confidence and signal strength scores are between 0 and 1."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Confidence and signal strength scores must be between 0.0 and 1.0")
        return v

    @field_validator('strike')
    @classmethod
    def validate_strike_positive(cls, v):
        """Validate strike price is positive."""
        if v <= 0.0:
            raise ValueError("Strike price must be positive")
        return v


class ProcessedUnderlyingAggregatesV2_5(RawUnderlyingDataCombinedV2_5):
    """
    Represents the fully processed and enriched data for the underlying asset for a
    given analysis cycle. Extends RawUnderlyingDataCombinedV2_5 with all calculated
    aggregate underlying-level metrics, the classified market regime, ticker context,
    dynamic threshold information, and regime analysis confidence/transition risk.
    This is a key input for high-level system components.
    """
    # Foundational Aggregate Metrics (Tier 1) - CRITICAL FINANCIAL DATA
    gib_oi_based_und: float = Field(..., description="Gamma Imbalance from Open Interest for the underlying - REQUIRED")
    td_gib_und: float = Field(..., description="Traded Dealer Gamma Imbalance for the underlying - REQUIRED")
    hp_eod_und: float = Field(..., description="End-of-Day Hedging Pressure for the underlying - REQUIRED")
    net_cust_delta_flow_und: float = Field(..., description="Net daily customer-initiated Delta flow for the underlying - REQUIRED")
    net_cust_gamma_flow_und: float = Field(..., description="Net daily customer-initiated Gamma flow for the underlying - REQUIRED")
    net_cust_vega_flow_und: float = Field(..., description="Net daily customer-initiated Vega flow for the underlying - REQUIRED")
    net_cust_theta_flow_und: float = Field(..., description="Net daily customer-initiated Theta flow for the underlying - REQUIRED")

    @field_validator('gib_oi_based_und', 'td_gib_und', 'hp_eod_und', 'net_cust_delta_flow_und',
                     'net_cust_gamma_flow_und', 'net_cust_vega_flow_und', 'net_cust_theta_flow_und')
    @classmethod
    def validate_foundational_metrics_not_fake(cls, v: float, info: FieldValidationInfo) -> float:
        """CRITICAL: Validate foundational metrics are real financial data, not placeholders."""
        # Allow negative values for flows (short positions), but reject suspicious patterns
        if v == 0.0:
            # Zero is suspicious for foundational metrics - could indicate missing calculation
            import warnings
            warnings.warn(f"WARNING: {info.field_name} is exactly 0.0 - verify this is real market data, not missing calculation!")
        return v

    # Standard Rolling Flows (Underlying Level)
    net_value_flow_5m_und: Optional[float] = Field(None, description="Net signed value traded in underlying's options over last 5 mins.")
    net_vol_flow_5m_und: Optional[float] = Field(None, description="Net signed volume traded in underlying's options over last 5 mins.")
    net_value_flow_15m_und: Optional[float] = Field(None, description="Net signed value traded in underlying's options over last 15 mins.")
    net_vol_flow_15m_und: Optional[float] = Field(None, description="Net signed volume traded in underlying's options over last 15 mins.")
    net_value_flow_30m_und: Optional[float] = Field(None, description="Net signed value traded in underlying's options over last 30 mins.")
    net_vol_flow_30m_und: Optional[float] = Field(None, description="Net signed volume traded in underlying's options over last 30 mins.")
    net_value_flow_60m_und: Optional[float] = Field(None, description="Net signed value traded in underlying's options over last 60 mins.")
    net_vol_flow_60m_und: Optional[float] = Field(None, description="Net signed volume traded in underlying's options over last 60 mins.")

    # 0DTE Suite Aggregates
    vri_0dte_und_sum: Optional[float] = Field(None, description="Sum of per-contract vri_0dte for 0DTE options.")
    vfi_0dte_und_sum: Optional[float] = Field(None, description="Sum of per-contract vfi_0dte for 0DTE options.")
    vvr_0dte_und_avg: Optional[float] = Field(None, description="Average per-contract vvr_0dte for 0DTE options.")
    vci_0dte_agg: Optional[float] = Field(None, description="Vanna Concentration Index (HHI-style) for 0DTE options.")

    # Other Aggregated Structural/Flow Metrics
    arfi_overall_und_avg: Optional[float] = Field(None, description="Overall Average Relative Flow Index for the underlying.")
    a_mspi_und_summary_score: Optional[float] = Field(None, description="Aggregate summary score from Adaptive MSPI components.")
    a_sai_und_avg: Optional[float] = Field(None, description="Average Adaptive Structure Alignment Index.")
    a_ssi_und_avg: Optional[float] = Field(None, description="Average Adaptive Structure Stability Index.")
    vri_2_0_und_aggregate: Optional[float] = Field(None, description="Aggregate Volatility Regime Indicator Version 2.0 score for the underlying.")

    # Enhanced Rolling Flow Metrics (Tier 3) - Z-Scores - CRITICAL TRADING SIGNALS
    vapi_fa_z_score_und: float = Field(..., description="Z-Score of Volatility-Adjusted Premium Intensity with Flow Acceleration - REQUIRED")
    dwfd_z_score_und: float = Field(..., description="Z-Score of Delta-Weighted Flow Divergence - REQUIRED")
    tw_laf_z_score_und: float = Field(..., description="Z-Score of Time-Weighted Liquidity-Adjusted Flow - REQUIRED")

    @field_validator('vapi_fa_z_score_und', 'dwfd_z_score_und', 'tw_laf_z_score_und')
    @classmethod
    def validate_zscore_metrics_not_fake(cls, v: float, info: FieldValidationInfo) -> float:
        """CRITICAL: Validate Z-score metrics are real trading signals, not fake data."""
        # Z-scores should typically be within reasonable bounds
        if abs(v) > 10.0:
            import warnings
            warnings.warn(f"WARNING: {info.field_name} = {v} is extremely high - verify this is real market data!")
        if v == 0.0:
            import warnings
            warnings.warn(f"WARNING: {info.field_name} is exactly 0.0 - verify this is real Z-score, not missing calculation!")
        return v

    # Enhanced Heatmap Data (Surface data might be complex)
    ivsdh_surface_data: Optional[PandasDataFrame] = Field(None, description="Data structure for the Integrated Volatility Surface Dynamics heatmap (often a DataFrame).")
    # SGDHP and UGCH are typically strike-level scores, but an aggregate summary might be here if needed.

    # Contextual & System State
    current_market_regime_v2_5: Optional[str] = Field(None, description="The classified market regime string for the current cycle.")
    ticker_context_dict_v2_5: Optional[TickerContextDictV2_5] = Field(None, description="Contextual information specific to the ticker for this cycle.")
    atr_und: Optional[float] = Field(None, description="Calculated Average True Range for the underlying.")
    hist_vol_20d: Optional[float] = Field(None, description="Historical volatility over 20 days for the underlying.")
    impl_vol_atm: Optional[float] = Field(None, description="Implied volatility at the money for the underlying.")
    trend_strength: Optional[float] = Field(None, description="Trend strength of the underlying.")
    trend_direction: Optional[str] = Field(None, description="Direction of the underlying trend (e.g., 'up', 'down', 'neutral').")
    dynamic_thresholds: Optional[DynamicThresholdsV2_5] = Field(None, description="Resolved dynamic thresholds used in this analysis cycle.")

    # Elite Impact Metrics (Underlying-level) - CRITICAL TRADING INTELLIGENCE
    elite_impact_score_und: float = Field(..., ge=0.0, le=100.0, description="Master composite elite impact score for the underlying - REQUIRED")
    institutional_flow_score_und: float = Field(..., ge=0.0, le=100.0, description="Institutional flow score for the underlying - REQUIRED")
    flow_momentum_index_und: float = Field(..., ge=-100.0, le=100.0, description="Flow momentum index for the underlying - REQUIRED")
    market_regime_elite: str = Field(..., min_length=1, description="Elite classified market regime - REQUIRED")
    flow_type_elite: str = Field(..., min_length=1, description="Elite classified flow type - REQUIRED")
    volatility_regime_elite: str = Field(..., min_length=1, description="Elite classified volatility regime - REQUIRED")

    @field_validator('elite_impact_score_und', 'institutional_flow_score_und')
    @classmethod
    def validate_elite_scores_not_fake(cls, v: float, info: FieldValidationInfo) -> float:
        """CRITICAL: Validate elite scores are real trading intelligence, not fake data."""
        if v == 0.0:
            raise ValueError(f"CRITICAL: {info.field_name} cannot be 0.0 - this indicates missing elite intelligence calculation!")
        return v

    @field_validator('market_regime_elite', 'flow_type_elite', 'volatility_regime_elite')
    @classmethod
    def validate_elite_classifications_not_fake(cls, v: str, info: FieldValidationInfo) -> str:
        """CRITICAL: Validate elite classifications are real analysis, not placeholders."""
        fake_values = ['unknown', 'default', 'placeholder', 'n/a', 'none', 'normal', '']
        if v.lower().strip() in fake_values:
            raise ValueError(f"CRITICAL: {info.field_name} '{v}' is a placeholder - provide real elite classification!")
        return v
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level for regime analysis (0-1, required)")
    transition_risk: float = Field(..., ge=0.0, le=1.0, description="Transition risk score for regime analysis (0-1, required)")

    # --- Time-Series/History Fields for Dashboard Panels ---
    # TIER 3: These are legitimately optional (visualization-only, no history on first run)
    vapifa_zscore_history: Optional[List[float]] = Field(None, description="History of VAPI FA Z-Score values for time-series charts.")
    vapifa_time_history: Optional[List[str]] = Field(None, description="Timestamps corresponding to vapifa_zscore_history.")
    dwfd_zscore_history: Optional[List[float]] = Field(None, description="History of DWFD Z-Score values for time-series charts.")
    dwfd_time_history: Optional[List[str]] = Field(None, description="Timestamps corresponding to dwfd_zscore_history.")
    twlaf_zscore_history: Optional[List[float]] = Field(None, description="History of TW LAF Z-Score values for time-series charts.")
    twlaf_time_history: Optional[List[str]] = Field(None, description="Timestamps corresponding to twlaf_zscore_history.")
    rolling_flows: Optional[List[float]] = Field(None, description="History of rolling flow values for the underlying.")
    rolling_flows_time: Optional[List[str]] = Field(None, description="Timestamps corresponding to rolling_flows.")
    nvp_by_strike: Optional[List[float]] = Field(None, description="Net Value Pressure by strike for heatmap/structure charts.")
    nvp_vol_by_strike: Optional[List[float]] = Field(None, description="Net Volume Pressure by strike for heatmap/structure charts.")
    strikes: Optional[List[float]] = Field(None, description="Strike prices corresponding to nvp_by_strike and nvp_vol_by_strike.")
    greek_flows: Optional[List[float]] = Field(None, description="History of aggregate Greek flows for the underlying.")
    greek_flows_time: Optional[List[str]] = Field(None, description="Timestamps corresponding to greek_flows.")
    flow_ratios: Optional[List[float]] = Field(None, description="History of flow ratio values for the underlying.")
    flow_ratios_time: Optional[List[str]] = Field(None, description="Timestamps corresponding to flow_ratios.")

    @model_validator(mode='after')
    def validate_time_series_data_consistency(self) -> 'ProcessedUnderlyingAggregatesV2_5':
        """TIER 3: Validate time-series history data consistency (lower priority - visualization only)."""
        # Validate paired time-series data consistency
        time_series_pairs = [
            (self.vapifa_zscore_history, self.vapifa_time_history, "vapifa"),
            (self.dwfd_zscore_history, self.dwfd_time_history, "dwfd"),
            (self.twlaf_zscore_history, self.twlaf_time_history, "twlaf"),
            (self.rolling_flows, self.rolling_flows_time, "rolling_flows"),
            (self.greek_flows, self.greek_flows_time, "greek_flows"),
            (self.flow_ratios, self.flow_ratios_time, "flow_ratios")
        ]

        for data_list, time_list, name in time_series_pairs:
            if data_list is not None and time_list is not None:
                if len(data_list) != len(time_list):
                    import warnings
                    warnings.warn(f"WARNING: {name} data and time arrays have mismatched lengths - this may cause chart display issues!")
            elif (data_list is None) != (time_list is None):
                import warnings
                warnings.warn(f"WARNING: {name} has data but no timestamps (or vice versa) - this may cause chart display issues!")

        # Validate strike-level data consistency
        if self.nvp_by_strike is not None and self.strikes is not None:
            if len(self.nvp_by_strike) != len(self.strikes):
                import warnings
                warnings.warn("WARNING: nvp_by_strike and strikes arrays have mismatched lengths - this may cause heatmap display issues!")

        if self.nvp_vol_by_strike is not None and self.strikes is not None:
            if len(self.nvp_vol_by_strike) != len(self.strikes):
                import warnings
                warnings.warn("WARNING: nvp_vol_by_strike and strikes arrays have mismatched lengths - this may cause heatmap display issues!")

        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # For PandasDataFrame
        extra='allow'  # Allow flexible configuration
    )


class KeyLevelV2_5(BaseModel):
    """
    Represents a single identified key price level (e.g., support, resistance,
    pin zone, volatility trigger, major wall). It includes the level's price,
    type, a conviction score, and the metrics that contributed to its identification.
    """
    level_price: float = Field(..., description="The price of the identified key level.", gt=0.0)
    level_type: str = Field(..., description="The type of key level (e.g., 'Support', 'Resistance', 'PinZone', 'VolTrigger', 'MajorWall').", min_length=1)
    conviction_score: float = Field(..., description="A score (e.g., 0.0 to 1.0 or 1-5) indicating the system's confidence in this level's significance, based on confluence and source strength.", ge=0.0, le=5.0)
    contributing_metrics: List[str] = Field(default_factory=list, description="A list of metric names or source identifiers that flagged or support this level (e.g., ['A-MSPI', 'NVP_Peak', 'SGDHP_Zone']).")
    source_identifier: Optional[str] = Field(None, description="An optional string to identify the primary method or specific source that generated this level (e.g., 'A-MSPI_Daily', 'UGCH_Strike_4500').")

    model_config = ConfigDict(extra='forbid')  # Internal model

    @field_validator('level_type')
    @classmethod
    def validate_level_type(cls, v):
        """Validate level type is one of the allowed types."""
        allowed_types = ['Support', 'Resistance', 'PinZone', 'VolTrigger', 'MajorWall', 'GammaWall', 'DeltaWall']
        if v not in allowed_types:
            raise ValueError(f"level_type must be one of {allowed_types}")
        return v


class KeyLevelsDataV2_5(BaseModel):
    """
    A container object that aggregates all identified key levels for a given
    analysis cycle, categorized by type. It provides a structured overview
    of critical price zones identified by KeyLevelIdentifierV2_5.
    """
    supports: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified support levels.")
    resistances: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified resistance levels.")
    pin_zones: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified potential pinning zones (often from D-TDPI + vci_0dte).")
    vol_triggers: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified volatility trigger levels (often from E-SDAG_VF or IVSDH data).")
    major_walls: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified major structural walls (often from strong SGDHP/UGCH scores).")
    timestamp: datetime = Field(..., description="Timestamp of when this set of key levels was generated.")

    model_config = ConfigDict(extra='forbid')  # Internal model

    @field_validator('supports', 'resistances', 'pin_zones', 'vol_triggers', 'major_walls')
    @classmethod
    def validate_key_levels_lists(cls, v):
        """Validate that key level lists don't contain duplicate prices."""
        if len(v) > 1:
            prices = [level.level_price for level in v]
            if len(prices) != len(set(prices)):
                raise ValueError("Key levels list contains duplicate prices")
        return v


class ProcessedDataBundleV2_5(BaseModel):
    """
    Represents the fully processed data state for an EOTS v2.5 analysis cycle.
    It contains all calculated metrics at the contract, strike, and underlying levels,
    serving as the primary input for higher-level analytical components like the
    Market Regime Engine, Signal Generator, and Adaptive Trade Idea Framework (ATIF).
    """
    options_data_with_metrics: List[ProcessedContractMetricsV2_5] = Field(..., min_items=1, description="List of option contracts with their calculated per-contract metrics - REQUIRED and must not be empty")
    strike_level_data_with_metrics: List[ProcessedStrikeLevelMetricsV2_5] = Field(..., min_items=1, description="List of strike-level aggregations and calculated metrics - REQUIRED and must not be empty")
    underlying_data_enriched: ProcessedUnderlyingAggregatesV2_5 = Field(..., description="The fully processed data for the underlying asset - REQUIRED")
    processing_timestamp: datetime = Field(..., description="Timestamp indicating when the data processing was completed - REQUIRED")
    errors: List[str] = Field(..., description="List of errors encountered during processing - REQUIRED (empty list if no errors)")

    @field_validator('options_data_with_metrics', 'strike_level_data_with_metrics')
    @classmethod
    def validate_data_lists_not_empty(cls, v: List, info: FieldValidationInfo) -> List:
        """CRITICAL: Ensure data lists are not empty - empty lists indicate missing market data."""
        if not v:
            raise ValueError(f"CRITICAL: {info.field_name} cannot be empty - this indicates missing market data processing!")
        return v

    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Due to ProcessedUnderlyingAggregatesV2_5 containing PandasDataFrame
    )

# =============================================================================
# BUNDLE SCHEMAS (from bundle_schemas.py)
# =============================================================================
"""
Pydantic models for top-level data bundles used in EOTS v2.5,
primarily for packaging comprehensive analysis results for consumption
by the dashboard or other system outputs.
"""

# Import necessary schemas from other new modules
# from processed_data import ProcessedDataBundleV2_5
# from signal_level_schemas import SignalPayloadV2_5, KeyLevelsDataV2_5
# from atif_schemas import ATIFStrategyDirectivePayloadV2_5 # For ATIF's pre-TPO directives
# from recommendation_schemas import ActiveRecommendationPayloadV2_5

# Import dependencies or use TYPE_CHECKING for forward references
# from raw_data import RawOptionsContractV2_5

# Import these from their proper modules to avoid circular dependencies
if TYPE_CHECKING:
    from .trading_market_models import (
        SignalPayloadV2_5,
        KeyLevelsDataV2_5,
        ATIFStrategyDirectivePayloadV2_5,
        ActiveRecommendationPayloadV2_5
    )

class UnprocessedDataBundleV2_5(BaseModel):
    """Container for all raw/unprocessed market data before analytics and metrics are applied."""
    options_contracts: List[RawOptionsContractV2_5] = Field(default_factory=list, description="Raw options contracts fetched from data provider.")
    underlying_data: RawUnderlyingDataCombinedV2_5 = Field(..., description="Raw underlying asset data.")
    fetch_timestamp: datetime = Field(..., description="Timestamp when data was fetched.")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during data fetch.")
    model_config = ConfigDict(extra='forbid')


class FinalAnalysisBundleV2_5(BaseModel):
    """
    The comprehensive, top-level data structure that encapsulates all analytical
    outputs for a single symbol from one full EOTS v2.5 analysis cycle.
    This bundle is the primary data product consumed by the dashboard for
    visualization and by any external systems that need the complete analytical picture.
    """
    processed_data_bundle: ProcessedDataBundleV2_5 = Field(..., description="Contains all metric-enriched data: options, strike-level, and underlying aggregates (which includes regime and ticker context).")
    scored_signals_v2_5: Dict[str, List[Any]] = Field(..., min_items=1, description="Dictionary of all scored raw signals generated during the cycle - REQUIRED and must not be empty")
    key_levels_data_v2_5: Any = Field(..., description="All identified key support, resistance, pin, and trigger levels - REQUIRED")

    atif_recommendations_v2_5: List[Any] = Field(..., description="List of new strategic directives generated by ATIF - REQUIRED (empty list if none)")
    active_recommendations_v2_5: List[Any] = Field(..., description="List of all active trade recommendations - REQUIRED (empty list if none)")

    bundle_timestamp: datetime = Field(..., description="Timestamp of when this final analysis bundle was created - REQUIRED")
    target_symbol: str = Field(..., min_length=1, description="The ticker symbol this bundle pertains to - REQUIRED")
    system_status_messages: List[str] = Field(..., description="System-level status messages, warnings, or errors - REQUIRED (empty list if none)")

    # Data freshness information for tiered weekend analysis
    data_freshness: Optional[Dict[str, Any]] = Field(default=None, description="Data freshness classification and metadata for tiered analysis")

    @field_validator('scored_signals_v2_5')
    @classmethod
    def validate_signals_not_empty(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """CRITICAL: Ensure signals dictionary is not empty - empty dict indicates missing signal generation."""
        if not v:
            raise ValueError("CRITICAL: scored_signals_v2_5 cannot be empty - this indicates missing signal generation!")
        return v

    @field_validator('target_symbol')
    @classmethod
    def validate_symbol_not_placeholder(cls, v: str) -> str:
        """CRITICAL: Ensure target symbol is not a placeholder."""
        placeholder_symbols = ['symbol', 'ticker', 'default', 'placeholder', 'unknown', '']
        if v.lower().strip() in placeholder_symbols:
            raise ValueError(f"CRITICAL: target_symbol '{v}' is a placeholder - provide real trading symbol!")
        return v.upper()

    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Necessary because ProcessedDataBundleV2_5 contains ProcessedUnderlyingAggregatesV2_5, which holds a PandasDataFrame for ivsdh_surface_data.
    )


class UnifiedIntelligenceAnalysis(BaseModel):
    """Unified intelligence analysis combining all AI systems."""
    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Analysis timestamp")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    market_regime_analysis: str = Field(description="Market regime analysis")
    options_flow_analysis: str = Field(description="Options flow analysis")
    sentiment_analysis: str = Field(description="Sentiment analysis")
    strategic_recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    risk_assessment: str = Field(description="Risk assessment")
    learning_insights: List[str] = Field(default_factory=list, description="Learning insights")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# ADVANCED METRICS MODELS (from advanced_metrics.py)
# =============================================================================
"""
Pydantic models for advanced, derived options metrics.
"""

class AdvancedOptionsMetricsV2_5(BaseModel):
    """
    Advanced options metrics for price action analysis based on liquidity and volatility.

    These metrics are derived from the "Options Contract Metrics for Price Action Analysis"
    document and provide sophisticated insights into market dynamics.
    """
    lwpai: Optional[float] = None  # Liquidity-Weighted Price Action Indicator
    vabai: Optional[float] = None  # Volatility-Adjusted Bid/Ask Imbalance
    aofm: Optional[float] = None   # Aggressive Order Flow Momentum
    lidb: Optional[float] = None   # Liquidity-Implied Directional Bias

    # Supporting metrics
    bid_ask_spread_percentage: Optional[float] = None
    total_liquidity_size: Optional[float] = None
    spread_to_volatility_ratio: Optional[float] = None
    theoretical_price_deviation: Optional[float] = None

    # Metadata
    valid_contracts_count: Optional[int] = None
    calculation_timestamp: Optional[datetime] = None
    confidence_score: Optional[float] = None  # 0-1 based on data quality
    data_quality_score: Optional[float] = None  # Additional data quality metric
    contracts_analyzed: Optional[int] = None  # Number of contracts analyzed


class TradeParametersV2_5(BaseModel):
    """
    Encapsulates the precise, executable parameters for a single leg of an
    options trade, as determined by the TradeParameterOptimizerV2_5.
    This model holds all necessary details for one specific option contract
    that is part of a broader recommended strategy.
    """
    option_symbol: str = Field(..., description="The full symbol of the specific option contract (e.g., 'SPY231215C00450000').")
    option_type: str = Field(..., description="The type of the option, typically 'call' or 'put'.") # Consider Enum: Literal["call", "put"]
    strike: float = Field(..., description="Strike price of this specific option leg.")
    expiration_str: str = Field(..., description="Expiration date of this option leg in a standardized string format (e.g., 'YYYY-MM-DD').")
    entry_price_suggested: float = Field(..., ge=0, description="The suggested entry price for this specific option leg (premium).")
    stop_loss_price: float = Field(..., ge=0, description="The calculated stop-loss price for this option leg (premium).")
    target_1_price: float = Field(..., ge=0, description="The first profit target price for this option leg (premium).")
    target_2_price: Optional[float] = Field(None, ge=0, description="An optional second profit target price (premium).")
    target_3_price: Optional[float] = Field(None, ge=0, description="An optional third profit target price (premium).")
    target_rationale: str = Field(..., description="Brief rationale explaining how these parameters were derived (e.g., based on ATR, key levels).")

    class Config:
        extra = 'forbid'


class ActiveRecommendationPayloadV2_5(BaseModel):
    """
    Represents a fully formulated and parameterized trade recommendation that is
    currently active or has been recently managed by the EOTS v2.5 system.
    This is the primary data structure for display and tracking of trade ideas.
    """
    recommendation_id: str = Field(..., description="Unique identifier for the recommendation.")
    symbol: str = Field(..., description="Ticker symbol of the underlying asset.")
    timestamp_issued: datetime = Field(..., description="Timestamp when the recommendation was first fully parameterized and made active.")
    strategy_type: str = Field(..., description="The type of options strategy (e.g., 'LongCall', 'BullPutSpread', 'ShortIronCondor').")
    selected_option_details: List[TradeParametersV2_5] = Field(default_factory=list, description="A list of TradeParametersV2_5 objects, each detailing an individual option leg involved in the strategy.")
    trade_bias: str = Field(..., description="The directional or volatility bias of the trade (e.g., 'Bullish', 'Bearish', 'NeutralVol').")

    # Initial parameters as set by TPO
    entry_price_initial: float = Field(..., description="The suggested entry premium for the overall strategy as initially set by TPO.")
    stop_loss_initial: float = Field(..., description="The initial stop-loss premium or underlying price level for the overall strategy.")
    target_1_initial: float = Field(..., description="Initial first profit target premium or underlying price level.")
    target_2_initial: Optional[float] = Field(None, description="Initial second profit target.")
    target_3_initial: Optional[float] = Field(None, description="Initial third profit target.")
    target_rationale: str = Field(..., description="Rationale from TPO on how initial targets/stops were set for the overall strategy.")

    # Fields for tracking execution and current state
    entry_price_actual: Optional[float] = Field(None, description="Actual fill price if the trade is executed and tracked.")
    stop_loss_current: float = Field(..., description="Current, potentially adjusted by ATIF, stop-loss level for the overall strategy.")
    target_1_current: float = Field(..., description="Current, potentially adjusted, first profit target.")
    target_2_current: Optional[float] = Field(None, description="Current second profit target.")
    target_3_current: Optional[float] = Field(None, description="Current third profit target.")

    status: str = Field(..., description="Current status of the recommendation (e.g., 'ACTIVE_NEW_NO_TSL', 'ACTIVE_ADJUSTED_T1_HIT', 'EXITED_TARGET_1', 'EXITED_STOPLOSS', 'CANCELLED').")
    status_update_reason: Optional[str] = Field(None, description="Reason for the latest status change (often from an ATIF management directive or SL/TP hit).")
    timestamp_last_update: datetime = Field(..., description="Timestamp of the most recent update to this recommendation.")

    # Tracking and performance fields
    pnl_current: Optional[float] = Field(None, description="Current profit/loss of the position if tracked.")
    pnl_percentage: Optional[float] = Field(None, description="Current profit/loss as a percentage.")

    # Additional metadata
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence score of the recommendation (0-100).")
    risk_reward_ratio: float = Field(..., description="Risk/reward ratio of the trade.")

    class Config:
        extra = 'forbid' 
