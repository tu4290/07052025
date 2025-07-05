"""
Pydantic models for data that has undergone initial processing and metric
calculation by EOTS v2.5. These schemas represent enriched versions of
raw data, including calculated metrics at contract, strike, and underlying levels.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .processed_data_config_schemas import DynamicThresholds

from .raw_data import RawOptionsContractV2_5, RawUnderlyingDataCombinedV2_5
from .base_types import PandasDataFrame

# Forward reference for TickerContextDictV2_5 if it's in a separate file later
# For now, assuming it might be co-located or imported via __init__
try:
    from .context_schemas import TickerContextDictV2_5
except ImportError:
    TickerContextDictV2_5 = Any # Fallback if context_schemas not yet created

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core_analytics_engine.eots_metrics.elite_definitions import MarketRegime, FlowType


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

    model_config = ConfigDict(extra='forbid') # Stricter, as this is processed internal data


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

    model_config = ConfigDict(extra='forbid') # Stricter, as this is processed internal data


class ProcessedUnderlyingAggregatesV2_5(RawUnderlyingDataCombinedV2_5):
    """
    Represents the fully processed and enriched data for the underlying asset for a
    given analysis cycle. Extends RawUnderlyingDataCombinedV2_5 with all calculated
    aggregate underlying-level metrics, the classified market regime, ticker context,
    dynamic threshold information, and regime analysis confidence/transition risk.
    This is a key input for high-level system components.
    """
    # Foundational Aggregate Metrics (Tier 1)
    gib_oi_based_und: Optional[float] = Field(None, description="Gamma Imbalance from Open Interest for the underlying.")
    td_gib_und: Optional[float] = Field(None, description="Traded Dealer Gamma Imbalance for the underlying.")
    hp_eod_und: Optional[float] = Field(None, description="End-of-Day Hedging Pressure for the underlying.")
    net_cust_delta_flow_und: Optional[float] = Field(None, description="Net daily customer-initiated Delta flow for the underlying.")
    net_cust_gamma_flow_und: Optional[float] = Field(None, description="Net daily customer-initiated Gamma flow for the underlying.")
    net_cust_vega_flow_und: Optional[float] = Field(None, description="Net daily customer-initiated Vega flow for the underlying.")
    net_cust_theta_flow_und: Optional[float] = Field(None, description="Net daily customer-initiated Theta flow for the underlying.")

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

    # Enhanced Rolling Flow Metrics (Tier 3) - Z-Scores
    vapi_fa_z_score_und: Optional[float] = Field(None, description="Z-Score of Volatility-Adjusted Premium Intensity with Flow Acceleration.")
    dwfd_z_score_und: Optional[float] = Field(None, description="Z-Score of Delta-Weighted Flow Divergence.")
    tw_laf_z_score_und: Optional[float] = Field(None, description="Z-Score of Time-Weighted Liquidity-Adjusted Flow.")

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
    dynamic_thresholds: Optional[DynamicThresholds] = Field(None, description="Resolved dynamic thresholds used in this analysis cycle.")

    # Elite Impact Metrics (Underlying-level)
    elite_impact_score_und: Optional[float] = Field(None, description="Master composite elite impact score for the underlying.")
    institutional_flow_score_und: Optional[float] = Field(None, description="Institutional flow score for the underlying.")
    flow_momentum_index_und: Optional[float] = Field(None, description="Flow momentum index for the underlying.")
    market_regime_elite: Optional[str] = Field(None, description="Elite classified market regime (string representation of enum).")
    flow_type_elite: Optional[str] = Field(None, description="Elite classified flow type (string representation of enum).")
    volatility_regime_elite: Optional[str] = Field(None, description="Elite classified volatility regime (string representation of enum).")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level for regime analysis (0-1, required)")
    transition_risk: float = Field(..., ge=0.0, le=1.0, description="Transition risk score for regime analysis (0-1, required)")
    
    # --- Time-Series/History Fields for Dashboard Panels ---
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid') # For PandasDataFrame & Stricter internal model


class ProcessedDataBundleV2_5(BaseModel):
    """
    Represents the fully processed data state for an EOTS v2.5 analysis cycle.
    It contains all calculated metrics at the contract, strike, and underlying levels,
    serving as the primary input for higher-level analytical components like the
    Market Regime Engine, Signal Generator, and Adaptive Trade Idea Framework (ATIF).
    """
    options_data_with_metrics: List[ProcessedContractMetricsV2_5] = Field(default_factory=list, description="List of option contracts with their calculated per-contract metrics.")
    strike_level_data_with_metrics: List[ProcessedStrikeLevelMetricsV2_5] = Field(default_factory=list, description="List of strike-level aggregations and calculated metrics.")
    underlying_data_enriched: ProcessedUnderlyingAggregatesV2_5 = Field(..., description="The fully processed data for the underlying asset, including all aggregate metrics and contextual information.")
    processing_timestamp: datetime = Field(..., description="Timestamp indicating when the data processing and metric calculation for this bundle were completed.")
    errors: List[str] = Field(default_factory=list, description="List to store any errors encountered during processing or metric calculation.")

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True) # Due to ProcessedUnderlyingAggregatesV2_5 containing PandasDataFrame