"""
Pydantic models for raw, unprocessed data as fetched from external sources
before significant EOTS v2.5 processing. These schemas are designed to
closely mirror the structure of the source APIs (e.g., ConvexValue)
and include `Config.extra = 'allow'` to accommodate potential new fields
from the API without breaking the system.
"""
from pydantic import BaseModel, Field
from typing import List, Any, Optional
from datetime import datetime


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
    contract_symbol: str = Field(..., description="Unique identifier for the option contract.")
    strike: float = Field(..., description="Strike price of the option.")
    opt_kind: str = Field(..., description="Type of option, typically 'call' or 'put'.")
    dte_calc: float = Field(..., description="Calculated Days To Expiration for the contract.")

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

    class Config:
        extra = 'allow' # Accommodate potential new fields from API


class RawUnderlyingDataV2_5(BaseModel):
    """
    Represents raw, unprocessed data for the underlying asset (e.g., stock, index)
    primarily from a source like ConvexValue `get_und` endpoint. This serves as
    the initial container for underlying-specific market data before enrichment
    from other sources or metric calculation.
    """
    symbol: str = Field(..., description="Ticker symbol of the underlying asset.")
    timestamp: datetime = Field(..., description="Timestamp of when this underlying data was fetched or is valid for.")
    price: Optional[float] = Field(None, description="Current market price of the underlying asset (Source: CV 'price').")
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

    class Config:
        extra = 'allow' # Accommodate potential new fields from API


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

    class Config:
        extra = 'allow' # Accommodate fields from both sources


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

    class Config:
        extra = 'forbid' # This bundle is internally constructed, should not have extra fields.
        arbitrary_types_allowed = True # To allow underlying_data which might contain complex types if not fully parsed to RawUnderlyingDataCombinedV2_5 initially