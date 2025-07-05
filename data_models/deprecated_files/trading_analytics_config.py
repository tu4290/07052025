"""
Trading & Analytics Configuration Models for EOTS v2.5

This module contains trading parameters, performance tracking, signal configurations,
and analytics engine settings.

Extracted from configuration_models.py for better modularity.
"""

# Standard library imports
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# DATA PROCESSOR SETTINGS
# =============================================================================

class DataProcessorFactors(BaseModel):
    """Various numerical factors used in metric calculations."""
    tdpi_gaussian_width: Optional[float] = Field(None, description="Gaussian width for TDPI calculations")
    flow_smoothing_factor: Optional[float] = Field(None, description="Factor for smoothing flow calculations")
    volatility_adjustment_factor: Optional[float] = Field(None, description="Factor for volatility adjustments")
    momentum_decay_factor: Optional[float] = Field(None, description="Decay factor for momentum calculations")
    regime_transition_sensitivity: Optional[float] = Field(None, description="Sensitivity for regime transitions")

    # Additional config fields from config file
    volume_factor: Optional[float] = Field(1.0, description="Volume factor for calculations")
    price_factor: Optional[float] = Field(1.0, description="Price factor for calculations")
    volatility_factor: Optional[float] = Field(1.0, description="Volatility factor for calculations")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_config = ConfigDict(extra='allow')

class IVContextParameters(BaseModel):
    """Parameters for IV contextualization."""
    vol_trend_avg_days: Optional[int] = Field(None, description="Days for volatility trend averaging")
    iv_rank_lookback_days: Optional[int] = Field(None, description="Lookback days for IV rank calculation")
    iv_percentile_window: Optional[int] = Field(None, description="Window for IV percentile calculation")
    term_structure_analysis_enabled: Optional[bool] = Field(None, description="Enable term structure analysis")
    skew_analysis_enabled: Optional[bool] = Field(None, description="Enable skew analysis")

    # Additional config fields from config file
    iv_threshold: Optional[float] = Field(0.25, description="IV threshold for analysis")
    iv_lookback_days: Optional[int] = Field(30, description="Lookback days for IV analysis")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_config = ConfigDict(extra='allow')

class DataProcessorSettings(BaseModel):
    """Settings for the Data Processor module."""
    enabled: bool = Field(True, description="Enable/disable data processing.")
    factors: DataProcessorFactors = Field(default_factory=DataProcessorFactors, description="Numerical factors for calculations.")
    iv_context: IVContextParameters = Field(default_factory=IVContextParameters, description="IV contextualization parameters.")
    iv_context_parameters: Optional[IVContextParameters] = Field(default_factory=IVContextParameters, description="IV contextualization parameters (alternative field name).")
    max_data_age_seconds: int = Field(300, ge=10, description="Maximum age of data to process (in seconds).")
    batch_size: int = Field(100, ge=1, description="Batch size for processing operations.")
    parallel_processing: bool = Field(True, description="Enable parallel processing where applicable.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'factors': self.factors.to_dict(),
            'iv_context': self.iv_context.to_dict(),
            'iv_context_parameters': self.iv_context_parameters.to_dict() if self.iv_context_parameters else None,
            'max_data_age_seconds': self.max_data_age_seconds,
            'batch_size': self.batch_size,
            'parallel_processing': self.parallel_processing
        }

    model_config = ConfigDict(extra='allow')


# =============================================================================
# MARKET REGIME ENGINE SETTINGS
# =============================================================================

class RegimeRuleConditions(BaseModel):
    """Conditions for a specific market regime rule."""
    vix_threshold: Optional[float] = Field(None, description="VIX threshold for regime")
    flow_alignment_threshold: Optional[float] = Field(None, description="Flow alignment threshold")
    volatility_regime: Optional[str] = Field(None, description="Volatility regime condition")
    momentum_condition: Optional[str] = Field(None, description="Momentum condition")
    structure_condition: Optional[str] = Field(None, description="Structure condition")
    confidence_threshold: Optional[float] = Field(None, description="Confidence threshold")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}
    
    model_config = ConfigDict(extra='allow')

class RegimeRules(BaseModel):
    """Dictionary of rules defining conditions for each market regime."""
    # Basic regime types
    BULLISH_MOMENTUM: Optional[RegimeRuleConditions] = Field(None, description="Rules for bullish momentum regime")
    BEARISH_MOMENTUM: Optional[RegimeRuleConditions] = Field(None, description="Rules for bearish momentum regime")
    CONSOLIDATION: Optional[RegimeRuleConditions] = Field(None, description="Rules for consolidation regime")
    HIGH_VOLATILITY: Optional[RegimeRuleConditions] = Field(None, description="Rules for high volatility regime")
    LOW_VOLATILITY: Optional[RegimeRuleConditions] = Field(None, description="Rules for low volatility regime")
    REGIME_UNCLEAR_OR_TRANSITIONING: Optional[RegimeRuleConditions] = Field(None, description="Rules for unclear/transitioning regime")
    
    # Specific advanced regime types from config
    REGIME_SPX_0DTE_FRIDAY_EOD_VANNA_CASCADE_POTENTIAL_BULLISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for SPX 0DTE Friday EOD Vanna cascade bullish regime")
    REGIME_SPY_PRE_FOMC_VOL_COMPRESSION_WITH_DWFD_ACCUMULATION: Optional[RegimeRuleConditions] = Field(None, description="Rules for SPY pre-FOMC vol compression with DWFD accumulation regime")
    REGIME_HIGH_VAPI_FA_BULLISH_MOMENTUM_UNIVERSAL: Optional[RegimeRuleConditions] = Field(None, description="Rules for high VAPI-FA bullish momentum universal regime")
    REGIME_ADAPTIVE_STRUCTURE_BREAKDOWN_WITH_DWFD_CONFIRMATION_BEARISH_UNIVERSAL: Optional[RegimeRuleConditions] = Field(None, description="Rules for adaptive structure breakdown with DWFD confirmation bearish universal regime")
    REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for vol expansion imminent VRI 0DTE bullish regime")
    REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for vol expansion imminent VRI 0DTE bearish regime")
    REGIME_NVP_STRONG_BUY_IMBALANCE_AT_KEY_STRIKE: Optional[RegimeRuleConditions] = Field(None, description="Rules for NVP strong buy imbalance at key strike regime")
    REGIME_EOD_HEDGING_PRESSURE_BUY: Optional[RegimeRuleConditions] = Field(None, description="Rules for EOD hedging pressure buy regime")
    REGIME_EOD_HEDGING_PRESSURE_SELL: Optional[RegimeRuleConditions] = Field(None, description="Rules for EOD hedging pressure sell regime")
    REGIME_SIDEWAYS_MARKET: Optional[RegimeRuleConditions] = Field(None, description="Rules for sideways market regime")
    
    # REMOVED: to_dict() method - Use model_dump() instead for Pydantic v2 compliance
    # This enforces strict Pydantic v2 architecture with no dictionary fallbacks
    
    # Additional config fields from config file
    bullish_conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Bullish regime conditions")
    bearish_conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Bearish regime conditions")
    neutral_conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Neutral regime conditions")

    model_config = ConfigDict(extra='allow')

class MarketRegimeEngineSettings(BaseModel):
    """Settings for the Market Regime Engine module."""
    enabled: bool = Field(True, description="Enable/disable market regime analysis.")
    regime_rules: RegimeRules = Field(default_factory=RegimeRules, description="Rules defining market regimes.")
    regime_update_interval_seconds: int = Field(60, ge=10, description="How often to update regime analysis (in seconds).")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence for regime classification.")
    regime_transition_smoothing: bool = Field(True, description="Enable smoothing for regime transitions.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'regime_rules': self.regime_rules.to_dict(),
            'regime_update_interval_seconds': self.regime_update_interval_seconds,
            'confidence_threshold': self.confidence_threshold,
            'regime_transition_smoothing': self.regime_transition_smoothing
        }
    
    model_config = ConfigDict(extra='allow')


# =============================================================================
# ENHANCED FLOW METRIC SETTINGS
# =============================================================================

class VAPIFAParameters(BaseModel):
    """Parameters specific to VAPI-FA calculation."""
    primary_flow_interval: Optional[str] = Field(None, description="Primary flow interval for VAPI-FA")
    iv_source_key: Optional[str] = Field(None, description="IV source key for calculations")
    flow_acceleration_window: Optional[int] = Field(None, description="Window for flow acceleration")
    volatility_adjustment_enabled: Optional[bool] = Field(None, description="Enable volatility adjustment")

    # Additional config fields from config file
    flow_threshold: Optional[float] = Field(0.5, description="Flow threshold for VAPI-FA")
    flow_window: Optional[int] = Field(10, description="Flow window for calculations")
    smoothing_factor: Optional[float] = Field(0.3, description="Smoothing factor for VAPI-FA")
    volume_weight: Optional[float] = Field(0.4, description="Volume weight in calculations")
    premium_weight: Optional[float] = Field(0.6, description="Premium weight in calculations")
    acceleration_lookback: Optional[int] = Field(5, description="Acceleration lookback period")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_config = ConfigDict(extra='allow')

class DWFDParameters(BaseModel):
    """Parameters specific to DWFD calculation."""
    flow_interval: Optional[str] = Field(None, description="Flow interval for DWFD")
    fvd_weight_factor: Optional[float] = Field(None, description="FVD weight factor")
    divergence_threshold: Optional[float] = Field(None, description="Divergence threshold")
    smoothing_factor: Optional[float] = Field(None, description="Smoothing factor")

    # Additional config fields from config file
    flow_threshold: Optional[float] = Field(0.5, description="Flow threshold for DWFD")
    flow_window: Optional[int] = Field(10, description="Flow window for calculations")
    delta_weight_factor: Optional[float] = Field(1.2, description="Delta weight factor")
    divergence_sensitivity: Optional[float] = Field(1.5, description="Divergence sensitivity")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_config = ConfigDict(extra='allow')

class TWLAFParameters(BaseModel):
    """Parameters specific to TW-LAF calculation."""
    time_weights_for_intervals: Optional[List[float]] = Field(None, description="Time weights for intervals")
    spread_calculation_params: Optional[Dict[str, float]] = Field(None, description="Spread calculation parameters")
    liquidity_adjustment_factor: Optional[float] = Field(None, description="Liquidity adjustment factor")
    flow_normalization_method: Optional[str] = Field(None, description="Flow normalization method")

    # Additional config fields from config file
    flow_threshold: Optional[float] = Field(0.5, description="Flow threshold for TW-LAF")
    flow_window: Optional[int] = Field(10, description="Flow window for calculations")
    smoothing_factor: Optional[float] = Field(0.3, description="Smoothing factor for TW-LAF")
    time_weight_decay: Optional[float] = Field(0.95, description="Time weight decay factor")
    liquidity_adjustment: Optional[float] = Field(0.8, description="Liquidity adjustment factor")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_config = ConfigDict(extra='allow')

class EnhancedFlowMetricSettings(BaseModel):
    """Settings for Enhanced Flow Metrics module."""
    enabled: bool = Field(True, description="Enable/disable enhanced flow metrics.")
    vapi_fa: VAPIFAParameters = Field(default_factory=VAPIFAParameters, description="VAPI-FA specific parameters.")
    dwfd: DWFDParameters = Field(default_factory=DWFDParameters, description="DWFD specific parameters.")
    tw_laf: TWLAFParameters = Field(default_factory=TWLAFParameters, description="TW-LAF specific parameters.")
    calculation_interval_seconds: int = Field(30, ge=5, description="Interval for flow metric calculations.")
    historical_lookback_periods: int = Field(20, ge=1, description="Number of periods to look back for calculations.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'vapi_fa': self.vapi_fa.to_dict(),
            'dwfd': self.dwfd.to_dict(),
            'tw_laf': self.tw_laf.to_dict(),
            'calculation_interval_seconds': self.calculation_interval_seconds,
            'historical_lookback_periods': self.historical_lookback_periods
        }
    
    model_config = ConfigDict(extra='allow')


# =============================================================================
# SIGNAL INTEGRATION SETTINGS
# =============================================================================

class SignalIntegrationParameters(BaseModel):
    """Parameters for signal integration."""
    base_signal_weights: Dict[str, float] = Field(default_factory=dict, description="Base signal weights")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for signals")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    class Config:
        extra = 'forbid'


# =============================================================================
# TRADING STRATEGY SETTINGS
# =============================================================================

class StrategySpecificRule(BaseModel):
    """Strategy-specific trading rule."""
    rule_name: str = Field(..., description="Name of the rule")
    rule_type: str = Field(..., description="Type of rule (entry, exit, risk)")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Rule conditions")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    enabled: bool = Field(default=True, description="Whether rule is enabled")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class IntelligentRecommendationManagementRules(BaseModel):
    """Rules for intelligent recommendation management."""
    exit_rules: Dict[str, Any] = Field(default_factory=dict, description="Exit rules")
    position_sizing_rules: Dict[str, Any] = Field(default_factory=dict, description="Position sizing rules")
    risk_management_rules: Dict[str, Any] = Field(default_factory=dict, description="Risk management rules")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class ContractSelectionFilters(BaseModel):
    """Filters for contract selection."""
    min_volume: int = Field(default=100, description="Minimum volume requirement")
    min_open_interest: int = Field(default=50, description="Minimum open interest requirement")
    max_bid_ask_spread: float = Field(default=0.05, description="Maximum bid-ask spread")
    dte_range: tuple = Field(default=(7, 45), description="Days to expiration range")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class StopLossCalculationRules(BaseModel):
    """Rules for stop loss calculation."""
    default_stop_pct: float = Field(default=0.1, description="Default stop loss percentage")
    volatility_adjusted: bool = Field(default=True, description="Use volatility-adjusted stops")
    max_stop_pct: float = Field(default=0.25, description="Maximum stop loss percentage")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class ProfitTargetCalculationRules(BaseModel):
    """Rules for profit target calculation."""
    default_target_pct: float = Field(default=0.2, description="Default profit target percentage")
    risk_reward_ratio: float = Field(default=2.0, description="Risk-reward ratio")
    dynamic_targets: bool = Field(default=True, description="Use dynamic profit targets")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'


# =============================================================================
# PERFORMANCE TRACKING SETTINGS
# =============================================================================

class PerformanceMetadata(BaseModel):
    """Metadata for performance tracking."""
    tracking_start_date: Optional[datetime] = Field(None, description="Start date for performance tracking")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol for comparison")
    performance_attribution: bool = Field(default=True, description="Enable performance attribution")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class StrategyParameters(BaseModel):
    """Parameters for trading strategies."""
    strategy_name: str = Field(..., description="Name of the strategy")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    risk_limits: Dict[str, float] = Field(default_factory=dict, description="Risk limits")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class PerformanceTrackerSettingsV2_5(BaseModel):
    """Settings for the Performance Tracker module."""
    enabled: bool = Field(True, description="Enable/disable performance tracking.")
    track_paper_trades: bool = Field(True, description="Track paper/simulated trades.")
    track_live_trades: bool = Field(False, description="Track live trades (requires broker integration).")
    performance_calculation_interval_seconds: int = Field(300, ge=60, description="Interval for performance calculations.")
    metadata: PerformanceMetadata = Field(default_factory=PerformanceMetadata, description="Performance tracking metadata.")
    strategy_params: List[StrategyParameters] = Field(default_factory=list, description="Strategy parameters for tracking.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'track_paper_trades': self.track_paper_trades,
            'track_live_trades': self.track_live_trades,
            'performance_calculation_interval_seconds': self.performance_calculation_interval_seconds,
            'metadata': self.metadata.to_dict(),
            'strategy_params': [sp.to_dict() for sp in self.strategy_params],
            'tracking_interval_seconds': self.tracking_interval_seconds,
            'performance_metadata': self.performance_metadata,
            'performance_data_directory': self.performance_data_directory,
            'historical_window_days': self.historical_window_days,
            'weight_smoothing_factor': self.weight_smoothing_factor,
            'min_sample_size': self.min_sample_size,
            'confidence_threshold': self.confidence_threshold,
            'update_interval_seconds': self.update_interval_seconds
        }

    # Additional config fields from config file
    tracking_interval_seconds: Optional[int] = Field(60, description="Tracking interval in seconds")
    performance_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metadata configuration")
    performance_data_directory: Optional[str] = Field("data_cache_v2_5/performance_data_store", description="Performance data directory")
    historical_window_days: Optional[int] = Field(365, description="Historical window in days")
    weight_smoothing_factor: Optional[float] = Field(0.1, description="Weight smoothing factor")
    min_sample_size: Optional[int] = Field(10, description="Minimum sample size")
    confidence_threshold: Optional[float] = Field(0.75, description="Confidence threshold")
    update_interval_seconds: Optional[int] = Field(3600, description="Update interval in seconds")

    model_config = ConfigDict(extra='allow')


# =============================================================================
# ADAPTIVE METRIC PARAMETERS
# =============================================================================

class TickerSpecificParameters(BaseModel):
    """Ticker-specific parameters for metrics."""
    volatility_adjustment: float = Field(default=1.0, description="Volatility adjustment factor")
    liquidity_adjustment: float = Field(default=1.0, description="Liquidity adjustment factor")
    sector_adjustment: float = Field(default=1.0, description="Sector-specific adjustment")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    class Config:
        extra = 'forbid'

class AdaptiveMetricParameters(BaseModel):
    """Parameters for adaptive metric calculations."""
    adaptation_enabled: bool = Field(True, description="Enable adaptive parameter adjustment.")
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="Learning rate for parameter adaptation.")
    adaptation_window_days: int = Field(30, ge=7, description="Window for adaptation calculations.")
    ticker_specific: Dict[str, TickerSpecificParameters] = Field(default_factory=dict, description="Ticker-specific parameters.")

    # Additional config fields from config file
    a_dag_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A-DAG settings")
    e_sdag_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="E-SDAG settings")
    d_tdpi_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="D-TDPI settings")
    vri_2_0_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="VRI 2.0 settings")
    enhanced_heatmap_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Enhanced heatmap settings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'adaptation_enabled': self.adaptation_enabled,
            'learning_rate': self.learning_rate,
            'adaptation_window_days': self.adaptation_window_days,
            'ticker_specific': {k: v.to_dict() for k, v in self.ticker_specific.items()},
            'a_dag_settings': self.a_dag_settings,
            'e_sdag_settings': self.e_sdag_settings,
            'd_tdpi_settings': self.d_tdpi_settings,
            'vri_2_0_settings': self.vri_2_0_settings,
            'enhanced_heatmap_settings': self.enhanced_heatmap_settings
        }

    model_config = ConfigDict(extra='allow')

class ConvictionMappingParameters(BaseModel):
    """Parameters for conviction mapping."""
    conviction_thresholds: Dict[str, float] = Field(default_factory=dict, description="Conviction thresholds")
    mapping_function: str = Field(default="linear", description="Conviction mapping function")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    model_config = ConfigDict(extra='allow')