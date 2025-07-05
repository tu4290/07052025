"""
Trading & Market Models for EOTS v2.5

Consolidated from: context_schemas.py, signal_level_schemas.py,
recommendation_schemas.py, atif_schemas.py
"""

# Standard library imports
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, FieldValidationInfo

class AdvancedOptionsMetrics(BaseModel):
    """Advanced options metrics for comprehensive analysis - FAIL FAST ON MISSING DATA."""
    implied_volatility_rank: float = Field(..., ge=0.0, le=100.0, description="IV rank percentile - REQUIRED")
    implied_volatility_percentile: float = Field(..., ge=0.0, le=100.0, description="IV percentile - REQUIRED")
    historical_volatility: float = Field(..., gt=0.0, description="Historical volatility - REQUIRED and must be positive")
    volatility_skew: float = Field(..., description="Volatility skew metric - REQUIRED")
    term_structure_slope: float = Field(..., description="Term structure slope - REQUIRED")
    put_call_ratio: float = Field(..., gt=0.0, description="Put/call ratio - REQUIRED and must be positive")
    max_pain: Optional[float] = Field(None, gt=0.0, description="Max pain level - optional but must be positive if provided")
    gamma_exposure: float = Field(..., description="Net gamma exposure - REQUIRED (can be negative for short gamma)")
    vanna_exposure: float = Field(..., description="Net vanna exposure - REQUIRED (can be negative)")
    charm_exposure: float = Field(..., description="Net charm exposure - REQUIRED (can be negative)")

    @field_validator('historical_volatility', 'implied_volatility_rank', 'implied_volatility_percentile')
    @classmethod
    def validate_no_zero_volatility(cls, v: float, info: FieldValidationInfo) -> float:
        """CRITICAL: Reject zero volatility values that indicate missing/fake data."""
        if v == 0.0:
            raise ValueError(f"CRITICAL: {info.field_name} cannot be 0.0 - this indicates missing real market data!")
        return v

class SupportingMetrics(BaseModel):
    """Supporting metrics for trade analysis and validation - NO EMPTY DICTIONARIES ALLOWED."""
    volume_profile: Optional[Dict[str, float]] = Field(None, description="Volume profile data - None if not available")
    liquidity_metrics: Optional[Dict[str, float]] = Field(None, description="Liquidity assessment metrics - None if not available")
    market_microstructure: Optional[Dict[str, Any]] = Field(None, description="Market microstructure data - None if not available")
    correlation_metrics: Optional[Dict[str, float]] = Field(None, description="Correlation with other assets - None if not available")
    momentum_indicators: Optional[Dict[str, float]] = Field(None, description="Momentum indicators - None if not available")
    mean_reversion_indicators: Optional[Dict[str, float]] = Field(None, description="Mean reversion indicators - None if not available")
    volatility_indicators: Optional[Dict[str, float]] = Field(None, description="Volatility indicators - None if not available")
    sentiment_indicators: Optional[Dict[str, float]] = Field(None, description="Market sentiment indicators - None if not available")

    @model_validator(mode='after')
    def validate_at_least_one_metric(self) -> 'SupportingMetrics':
        """CRITICAL: Ensure at least one supporting metric is provided - no completely empty metrics allowed."""
        all_metrics = [
            self.volume_profile, self.liquidity_metrics, self.market_microstructure,
            self.correlation_metrics, self.momentum_indicators, self.mean_reversion_indicators,
            self.volatility_indicators, self.sentiment_indicators
        ]

        if all(metric is None or (isinstance(metric, dict) and len(metric) == 0) for metric in all_metrics):
            raise ValueError("CRITICAL: At least one supporting metric must be provided - empty metrics indicate missing data!")

        return self


# =============================================================================
# FROM context_schemas.py
# =============================================================================
"""
Pydantic models defining contextual information used across the EOTS v2.5 system.
These schemas help in tailoring analysis based on ticker-specific characteristics,
market events, or time-based conditions.
"""


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

class TickerContextDictV2_5(BaseModel):
    """
    Holds various contextual flags and states specific to the ticker being analyzed,
    generated by TickerContextAnalyzerV2_5. This information is used by other
    system components (MRE, Metrics Calculator, ATIF) to adapt their logic.
    """
    is_0dte: Optional[bool] = Field(None, description="True if current day is a 0 DTE (Days To Expiration) day for the active symbol.")
    is_1dte: Optional[bool] = Field(None, description="True if current day is a 1 DTE day for the active symbol.")
    is_spx_mwf_expiry_type: Optional[bool] = Field(None, description="Flags specific SPX Monday/Wednesday/Friday expiration types.")
    is_spy_eom_expiry: Optional[bool] = Field(None, description="Flags if it's an SPY End-of-Month expiration.")
    is_quad_witching_week_flag: Optional[bool] = Field(None, description="Flags if the current week is a quadruple witching week.")
    days_to_nearest_0dte: Optional[int] = Field(None, description="Number of calendar days to the nearest 0DTE event for the symbol.")
    days_to_monthly_opex: Optional[int] = Field(None, description="Number of calendar days to the next standard monthly options expiration.")

    # Event-based context
    is_fomc_meeting_day: Optional[bool] = Field(None, description="True if the current day is an FOMC meeting day.")
    is_fomc_announcement_imminent: Optional[bool] = Field(None, description="True if an FOMC announcement is expected shortly (e.g., within a specific time window).")
    post_fomc_drift_period_active: Optional[bool] = Field(None, description="True if within the typical post-FOMC announcement drift period.")

    # Behavioral pattern flags (examples)
    vix_spy_price_divergence_strong_negative: Optional[bool] = Field(None, description="Example: True if VIX is up strongly while SPY is also up, indicating unusual divergence.")
    
    # Intraday session context
    active_intraday_session: Optional[str] = Field(None, description="Current intraday session (e.g., 'OPENING_RUSH', 'LUNCH_LULL', 'POWER_HOUR', 'EOD_AUCTION').")
    is_near_auction_period: Optional[bool] = Field(None, description="True if current time is near market open or close auction periods.")

    # General ticker characteristics
    ticker_liquidity_profile_flag: Optional[str] = Field(None, description="General liquidity assessment for the ticker (e.g., 'High', 'Medium', 'Low', 'Illiquid').")
    ticker_volatility_state_flag: Optional[str] = Field(None, description="Assessment of the ticker's current volatility character (e.g., 'IV_HIGH_RV_LOW', 'IV_CRUSH_IMMINENT').")
    
    # Earnings context (for equities)
    earnings_approaching_flag: Optional[bool] = Field(None, description="True if an earnings announcement is imminent for the stock (e.g., within a week).")
    days_to_earnings: Optional[int] = Field(None, description="Number of calendar days to the next scheduled earnings announcement.")

    # Fields from old schema
    is_SPX_0DTE_Friday_EOD: Optional[bool] = None
    a_mspi_und_summary_score: Optional[float] = None
    nvp_by_strike: Optional[Dict[str, float]] = None
    hp_eod_und: Optional[float] = None
    trend_threshold: Optional[float] = None
    advanced_options_metrics: Optional[AdvancedOptionsMetrics] = None

    model_config = ConfigDict(extra='forbid')  # Internal model, structure should be strictly defined


class TimeOfDayDefinitions(BaseModel):
    """
    Defines critical time points for market operations and EOTS v2.5 internal logic,
    such as determining intraday sessions or when to perform EOD calculations.
    These are typically loaded from system configuration.
    """
    # FAIL-FAST ARCHITECTURE: All timing parameters are REQUIRED from configuration - no hardcoded market hours!
    market_open: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="Market open time in HH:MM:SS format - REQUIRED from config")
    market_close: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="Market close time in HH:MM:SS format - REQUIRED from config")
    pre_market_start: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="Pre-market start time in HH:MM:SS format - REQUIRED from config")
    after_hours_end: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="After hours end time in HH:MM:SS format - REQUIRED from config")
    eod_pressure_calc_time: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="Time to trigger end-of-day pressure calculations - REQUIRED from config")

    @model_validator(mode='after')
    def validate_market_timing_consistency(self) -> 'TimeOfDayDefinitions':
        """CRITICAL: Validate market timing consistency and reject invalid schedules."""
        from datetime import datetime, time

        try:
            # Parse all times to validate format
            pre_market = datetime.strptime(self.pre_market_start, "%H:%M:%S").time()
            market_open = datetime.strptime(self.market_open, "%H:%M:%S").time()
            eod_calc = datetime.strptime(self.eod_pressure_calc_time, "%H:%M:%S").time()
            market_close = datetime.strptime(self.market_close, "%H:%M:%S").time()
            after_hours = datetime.strptime(self.after_hours_end, "%H:%M:%S").time()

            # Validate logical order
            if not (pre_market < market_open < eod_calc < market_close < after_hours):
                raise ValueError("CRITICAL: Market timing sequence is invalid - times must be in logical order!")

        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("CRITICAL: Invalid time format - must be HH:MM:SS!")
            raise

        return self
    # Add other specific time definitions as needed, e.g., for intraday session boundaries
    # opening_rush_end: str = Field(default="10:15:00", description="End of 'Opening Rush' session.")
    # lunch_lull_start: str = Field(default="12:00:00", description="Start of 'Lunch Lull' session.")
    # lunch_lull_end: str = Field(default="13:30:00", description="End of 'Lunch Lull' session.")
    # power_hour_start: str = Field(default="15:00:00", description="Start of 'Power Hour' session.")

    model_config = ConfigDict(extra='forbid')  # Configuration model, structure should be strictly defined


# --- PORTED: DynamicThresholdsV2_5 (from deprecated_files/eots_schemas_v2_5.py) ---
class DynamicThresholdsV2_5(BaseModel):
    """Dynamic thresholds for market regime detection and signal generation.
    This model defines configurable thresholds used throughout the system for:
    - Market regime classification
    - Signal generation and filtering
    - Risk management
    - Data quality assessment
    All thresholds are designed to be dynamically adjustable based on market conditions.
    """
    # FAIL-FAST ARCHITECTURE: All thresholds are REQUIRED from configuration - no hardcoded trading parameters!
    vapi_fa_bullish_thresh: float = Field(..., gt=0.0, description="Z-score threshold for bullish VAPI-FA signals - REQUIRED from config")
    vapi_fa_bearish_thresh: float = Field(..., lt=0.0, description="Z-score threshold for bearish VAPI-FA signals - REQUIRED from config")
    vri_bullish_thresh: float = Field(..., gt=0.0, lt=1.0, description="VRI threshold for bullish regime classification - REQUIRED from config")
    vri_bearish_thresh: float = Field(..., gt=-1.0, lt=0.0, description="VRI threshold for bearish regime classification - REQUIRED from config")
    negative_thresh_default: float = Field(..., lt=0.0, description="Default negative threshold for signal classification - REQUIRED from config")
    positive_thresh_default: float = Field(..., gt=0.0, description="Default positive threshold for signal classification - REQUIRED from config")
    significant_pos_thresh: float = Field(..., gt=0, description="Threshold for significant positive values - REQUIRED from config")
    significant_neg_thresh: float = Field(..., lt=0, description="Threshold for significant negative values - REQUIRED from config")
    mid_high_nvp_thresh_pos: float = Field(..., gt=0, description="Mid-high threshold for Net Vega Position - REQUIRED from config")
    high_nvp_thresh_pos: float = Field(..., gt=0, description="High threshold for Net Vega Position - REQUIRED from config")
    dwfd_strong_thresh: float = Field(..., gt=0.0, description="Threshold for strong DWFD signal - REQUIRED from config")
    tw_laf_strong_thresh: float = Field(..., gt=0.0, description="Threshold for strong TW-LAF signal - REQUIRED from config")
    volatility_expansion_thresh: float = Field(..., gt=0.0, le=1.0, description="Threshold for volatility expansion detection - REQUIRED from config")
    hedging_pressure_thresh: float = Field(..., gt=0, description="Threshold for significant hedging pressure - REQUIRED from config")
    high_confidence_thresh: float = Field(..., gt=0.0, le=1.0, description="Minimum score for high confidence classification - REQUIRED from config")
    moderate_confidence_thresh: float = Field(..., gt=0.0, le=1.0, description="Minimum score for moderate confidence classification - REQUIRED from config")
    data_quality_thresh: float = Field(..., gt=0.0, le=1.0, description="Minimum data quality score for analytics - REQUIRED from config")

    @model_validator(mode='after')
    def validate_threshold_consistency(self) -> 'DynamicThresholdsV2_5':
        """CRITICAL: Validate threshold consistency and reject suspicious values."""
        # Validate VRI thresholds are properly ordered
        if self.vri_bearish_thresh >= self.vri_bullish_thresh:
            raise ValueError("CRITICAL: VRI bearish threshold must be less than bullish threshold!")

        # Validate confidence thresholds are properly ordered
        if self.moderate_confidence_thresh >= self.high_confidence_thresh:
            raise ValueError("CRITICAL: Moderate confidence threshold must be less than high confidence threshold!")

        # Validate NVP thresholds are properly ordered
        if self.mid_high_nvp_thresh_pos >= self.high_nvp_thresh_pos:
            raise ValueError("CRITICAL: Mid-high NVP threshold must be less than high NVP threshold!")

        return self

    model_config = ConfigDict(extra='forbid')

    @field_validator('vri_bullish_thresh', 'vri_bearish_thresh', mode='before')
    @classmethod
    def validate_vri_thresholds(cls, v: float, info: FieldValidationInfo) -> float:
        if info.field_name == 'vri_bullish_thresh' and not (0.0 < v < 1.0):
            raise ValueError('vri_bullish_thresh must be between 0 and 1')
        if info.field_name == 'vri_bearish_thresh' and not (-1.0 < v < 0.0):
            raise ValueError('vri_bearish_thresh must be between -1 and 0')
        return v

    @field_validator('high_confidence_thresh', 'moderate_confidence_thresh', 'data_quality_thresh', mode='before')
    @classmethod
    def validate_probability_thresholds(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError('Thresholds must be between 0 and 1')
        return v

# --- END PORT ---


# =============================================================================
# FROM signal_level_schemas.py
# =============================================================================
"""
Pydantic models for representing generated trading signals and identified
key price levels within the EOTS v2.5 system.
"""

class SignalPayloadV2_5(BaseModel):
    """
    Represents a single, discrete trading signal generated by SignalGeneratorV2_5.
    It encapsulates all relevant information about a specific signal event,
    including its nature, strength, and the context in which it occurred.
    These payloads are primary inputs to the Adaptive Trade Idea Framework (ATIF).
    """
    signal_id: str = Field(..., description="A unique identifier for this specific signal instance (e.g., UUID).", min_length=1)
    signal_name: str = Field(..., description="A human-readable name for the signal (e.g., 'VAPI_FA_Bullish_Surge', 'A_MSPI_Support_Confirmed').", min_length=1)
    symbol: str = Field(..., description="The ticker symbol for which the signal was generated.", min_length=1)
    timestamp: datetime = Field(..., description="Timestamp of when the signal was generated.")
    signal_type: str = Field(..., description="Categorization of the signal (e.g., 'Directional', 'Volatility', 'Structural', 'Flow_Divergence', 'Warning').", min_length=1)
    direction: Optional[str] = Field(None, description="If applicable, the directional bias of the signal (e.g., 'Bullish', 'Bearish', 'Neutral').")
    strength_score: float = Field(..., ge=-5.0, le=5.0, description="A numerical score (e.g., -1.0 to +1.0 or 0 to 1.0) indicating the intensity or confidence of this raw signal, generated by SignalGeneratorV2_5 before ATIF weighting.")
    strike_impacted: Optional[float] = Field(None, description="The specific strike price most relevant to this signal, if applicable (e.g., for a structural signal).", gt=0.0)
    regime_at_signal_generation: Optional[str] = Field(None, description="The market regime active when this signal was triggered.")
    supporting_metrics: SupportingMetrics = Field(default_factory=SupportingMetrics, description="Key metric values and their states that contributed to triggering this signal, providing context for ATIF evaluation.")

    model_config = ConfigDict(extra='forbid')  # Internal model, structure should be strictly defined

    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        """Validate direction is one of the allowed values."""
        if v is not None and v not in ['Bullish', 'Bearish', 'Neutral']:
            raise ValueError("direction must be 'Bullish', 'Bearish', or 'Neutral'")
        return v

    @field_validator('signal_type')
    @classmethod
    def validate_signal_type(cls, v):
        """Validate signal type is one of the allowed categories."""
        allowed_types = ['Directional', 'Volatility', 'Structural', 'Flow_Divergence', 'Warning', 'Momentum', 'Mean_Reversion']
        if v not in allowed_types:
            raise ValueError(f"signal_type must be one of {allowed_types}")
        return v


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


# =============================================================================
# FROM recommendation_schemas.py
# =============================================================================
"""
Pydantic models defining the structure of fully parameterized trade
recommendations within the EOTS v2.5 system, as well as the parameters
for individual option legs.
"""

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

    model_config = ConfigDict(extra='forbid')


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
    
    # Context at issuance
    atif_conviction_score_at_issuance: float = Field(..., description="ATIF's final conviction score (e.g., 0-5) when the idea was formed.")
    triggering_signals_summary: Optional[str] = Field(None, description="A summary of the key signals that led to this recommendation.")
    regime_at_issuance: str = Field(..., description="The market regime active when the recommendation was issued.")
    
    # Outcome details (populated upon closure)
    exit_timestamp: Optional[datetime] = Field(None, description="Timestamp of when the trade was exited.")
    exit_price: Optional[float] = Field(None, description="Actual exit premium or underlying price for the overall strategy.")
    pnl_percentage: Optional[float] = Field(None, description="Profit/Loss percentage for the trade.")
    pnl_absolute: Optional[float] = Field(None, description="Absolute Profit/Loss for the trade.")
    exit_reason: Optional[str] = Field(None, description="Reason for trade exit (e.g., 'TargetHit', 'StopLossHit', 'ATIF_Directive_RegimeChange').")

    model_config = ConfigDict(extra='allow')  # Allow flexible configuration
    # arbitrary_types_allowed = True # Not needed if selected_option_details is List[TradeParametersV2_5]


# =============================================================================
# FROM atif_schemas.py
# =============================================================================
"""
Pydantic models specifically related to the internal workings and directive
outputs of the Adaptive Trade Idea Framework (ATIF) in EOTS v2.5.
These schemas define data structures for situational assessment, strategic
directives before parameterization, and management actions for active trades.
"""

# Forward reference if TickerContextDictV2_5 is in another file and imported via __init__
# context_schemas import TickerContextDictV2_5 # Example if needed

class ATIFSituationalAssessmentProfileV2_5(BaseModel):
    """
    Represents ATIF's initial, holistic assessment of the current market situation.
    FAIL-FAST ARCHITECTURE: All assessment scores are REQUIRED - no fake trading signals allowed!
    """
    bullish_assessment_score: float = Field(..., ge=-5.0, le=5.0, description="ATIF's aggregated weighted score for a bullish outlook - REQUIRED")
    bearish_assessment_score: float = Field(..., ge=-5.0, le=5.0, description="ATIF's aggregated weighted score for a bearish outlook - REQUIRED")
    vol_expansion_score: float = Field(..., ge=0.0, le=5.0, description="ATIF's assessment of volatility expansion likelihood - REQUIRED")
    vol_contraction_score: float = Field(..., ge=0.0, le=5.0, description="ATIF's assessment of volatility contraction likelihood - REQUIRED")
    mean_reversion_likelihood: float = Field(..., ge=0.0, le=1.0, description="ATIF's mean reversion probability assessment - REQUIRED")
    timestamp: datetime = Field(..., description="Timestamp of when this assessment profile was generated by ATIF.")

    @model_validator(mode='after')
    def validate_assessment_consistency(self) -> 'ATIFSituationalAssessmentProfileV2_5':
        """CRITICAL: Validate that assessment scores are consistent and not fake data."""
        # Check for suspicious patterns that indicate fake data
        all_scores = [
            self.bullish_assessment_score, self.bearish_assessment_score,
            self.vol_expansion_score, self.vol_contraction_score, self.mean_reversion_likelihood
        ]

        # Reject if all scores are exactly 0.0 (indicates fake data)
        if all(score == 0.0 for score in all_scores):
            raise ValueError("CRITICAL: All assessment scores are 0.0 - this indicates fake trading data!")

        return self

    model_config = ConfigDict(extra='forbid')


class ATIFStrategyDirectivePayloadV2_5(BaseModel):
    """
    Represents a strategic directive formulated by ATIF when it identifies a
    potential trading opportunity with sufficient conviction. This payload specifies
    the type of options strategy, target DTE/delta ranges, and ATIF's conviction,
    serving as input to the TradeParameterOptimizerV2_5.
    """
    selected_strategy_type: str = Field(..., description="The category of options strategy selected by ATIF (e.g., 'LongCall', 'BullPutSpread', 'ShortIronCondor').")
    target_dte_min: int = Field(..., ge=0, description="Minimum Days-To-Expiration for the desired option(s).")
    target_dte_max: int = Field(..., ge=0, description="Maximum Days-To-Expiration for the desired option(s).")
    target_delta_long_leg_min: Optional[float] = Field(None, description="Target minimum delta for the long leg(s) of the strategy (e.g., 0.3 for 30 delta).")
    target_delta_long_leg_max: Optional[float] = Field(None, description="Target maximum delta for the long leg(s) of the strategy (e.g., 0.7 for 70 delta).")
    target_delta_short_leg_min: Optional[float] = Field(None, description="Target minimum delta for the short leg(s) of a spread (absolute value, e.g., 0.1 for 10 delta).")
    target_delta_short_leg_max: Optional[float] = Field(None, description="Target maximum delta for the short leg(s) of a spread (absolute value, e.g., 0.4 for 40 delta).")
    underlying_price_at_decision: float = Field(..., description="The price of the underlying asset when ATIF made this strategic decision.")
    final_conviction_score_from_atif: float = Field(..., description="ATIF's final conviction score for this strategic idea (e.g., on a 0-5 scale).")
    supportive_rationale_components: Dict[str, Any] = Field(default_factory=dict, description="Key data points, signals, or reasons supporting this directive.")
    assessment_profile: ATIFSituationalAssessmentProfileV2_5 = Field(..., description="The detailed situational assessment that led to this directive.")

    model_config = ConfigDict(extra='forbid')


class ATIFManagementDirectiveV2_5(BaseModel):
    """
    Represents a specific management action directive issued by ATIF for an
    existing, active trade recommendation. It instructs the ITSOrchestratorV2_5
    on how to modify an active trade based on evolving market conditions.
    """
    recommendation_id: str = Field(..., description="The unique ID of the active recommendation this directive applies to.")
    action: str = Field(..., description="The specific management action to take (e.g., 'EXIT', 'ADJUST_STOPLOSS', 'ADJUST_TARGET_1', 'PARTIAL_PROFIT_TAKE', 'HOLD').")
    reason: str = Field(..., description="A human-readable reason for the directive (e.g., 'RegimeInvalidation_BearishShift', 'ATR_TrailingStop_Activated').")
    new_stop_loss: Optional[float] = Field(None, description="The new stop-loss price (option premium or underlying), if the action is to adjust it.")
    new_target_1: Optional[float] = Field(None, description="The new first profit target price, if applicable.")
    new_target_2: Optional[float] = Field(None, description="The new second profit target price, if applicable.")
    exit_price_type: Optional[str] = Field(None, description="Suggested execution type for an exit (e.g., 'Market', 'Limit').")
    percentage_to_manage: Optional[float] = Field(None, ge=0, le=1, description="For partial actions (e.g., PARTIAL_PROFIT_TAKE), the percentage of the position to affect (0.0 to 1.0).")

    model_config = ConfigDict(extra='forbid')

# Imports that might be needed by the classes below - will be resolved by data_models/__init__.py
# from .ai_ml_models import AIPredictionV2_5 # Example if it were structured this way
# Need to ensure datetime.timezone is available
from datetime import timezone as dt_timezone # Alias to avoid conflict if datetime.timezone is used elsewhere

class ConsolidatedAnalysisRequest(BaseModel):
    """Unified request model for complete trade intelligence analysis."""
    ticker: str = Field(..., description="Trading symbol")
    analysis_type: str = Field(default="comprehensive", description="Analysis depth")
    include_predictions: bool = Field(default=True, description="Generate AI predictions")
    include_optimization: bool = Field(default=True, description="Optimize trade parameters")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    time_horizon: str = Field(default="short_term", description="Trade time horizon")
    position_size: Optional[float] = Field(None, description="Desired position size")

    model_config = ConfigDict(extra='forbid')

class SuperiorTradeIntelligence(BaseModel):
    """Superior consolidated trade intelligence result."""
    ticker: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt_timezone.utc))

    ai_analysis: Dict[str, Any] = Field(..., description="HuiHui AI expert analysis")
    # The following will be imported from data_models once __init__ is set up
    # For now, using Any to avoid direct import error before __init__ is fully set.
    # Actual type should be List[AIPredictionV2_5]
    predictions: List[Any] = Field(default_factory=list, description="AI predictions. Type: List[AIPredictionV2_5]")

    strategy_directive: ATIFStrategyDirectivePayloadV2_5 = Field(..., description="Strategy recommendation")
    optimized_parameters: ActiveRecommendationPayloadV2_5 = Field(..., description="Optimized trade parameters")

    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance analytics")
    learning_insights: List[str] = Field(default_factory=list, description="Learning insights")

    overall_confidence: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    conviction_score: float = Field(..., description="Trade conviction score", ge=0.0, le=5.0)

    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            'version': '2.5',
            'generated_at': datetime.now(dt_timezone.utc).isoformat(),
            'adaptive_framework': True,
            'system': 'ATIF Engine v2.5'
        },
        description="System metadata and generation context"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

# Imports that might be needed by the classes below - will be resolved by data_models/__init__.py
# from .ai_ml_models import AIPredictionV2_5 # Example if it were structured this way

class ConsolidatedAnalysisRequest(BaseModel):
    """Unified request model for complete trade intelligence analysis."""
    ticker: str = Field(..., description="Trading symbol")
    analysis_type: str = Field(default="comprehensive", description="Analysis depth")
    include_predictions: bool = Field(default=True, description="Generate AI predictions")
    include_optimization: bool = Field(default=True, description="Optimize trade parameters")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    time_horizon: str = Field(default="short_term", description="Trade time horizon")
    position_size: Optional[float] = Field(None, description="Desired position size")

    model_config = ConfigDict(extra='forbid')

class SuperiorTradeIntelligence(BaseModel):
    """Superior consolidated trade intelligence result."""
    ticker: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc)) # Corrected

    # AI Intelligence
    ai_analysis: Dict[str, Any] = Field(..., description="HuiHui AI expert analysis")
    # The following will be imported from data_models once __init__ is set up
    predictions: List[Any] = Field(default_factory=list, description="AI predictions. Type: List[AIPredictionV2_5]")

    # Strategy Intelligence
    strategy_directive: ATIFStrategyDirectivePayloadV2_5 = Field(..., description="Strategy recommendation")
    optimized_parameters: ActiveRecommendationPayloadV2_5 = Field(..., description="Optimized trade parameters")

    # Performance Intelligence
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance analytics")
    learning_insights: List[str] = Field(default_factory=list, description="Learning insights")

    # Confidence Metrics
    overall_confidence: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    conviction_score: float = Field(..., description="Trade conviction score", ge=0.0, le=5.0)

    # System Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            'version': '2.5',
            'generated_at': datetime.now(datetime.timezone.utc).isoformat(), # Corrected
            'adaptive_framework': True,
            'system': 'ATIF Engine v2.5'
        },
        description="System metadata and generation context"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )