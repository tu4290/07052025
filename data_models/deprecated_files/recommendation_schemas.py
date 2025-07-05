"""
Pydantic models defining the structure of fully parameterized trade
recommendations within the EOTS v2.5 system, as well as the parameters
for individual option legs.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

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

    class Config:
        extra = 'forbid' # Internal model, structure should be strictly defined
        # arbitrary_types_allowed = True # Not needed if selected_option_details is List[TradeParametersV2_5]