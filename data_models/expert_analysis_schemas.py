
# data_models/expert_analysis_schemas.py
# Centralized Pydantic models for HuiHui AI Expert Analysis requests and responses.
# This file is created to break circular dependencies and centralize schema definitions.

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- Request and Response Base Models ---

class ExpertAnalysisRequest(BaseModel):
    """
    Standardized request model for all HuiHui AI Expert analysis tasks.
    """
    analysis_id: str = Field(..., description="Unique identifier for the analysis request.")
    symbol: str = Field(..., description="Trading symbol for which analysis is requested.")
    data_bundle: Any = Field(..., description="The processed data bundle (ProcessedDataBundleV2_5) for analysis.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the request.")
    # Add any other common request parameters here

class ExpertAnalysisResponse(BaseModel):
    """
    Standardized response model for all HuiHui AI Expert analysis tasks.
    """
    analysis_summary: str = Field(..., description="A concise summary of the expert's analysis.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Expert's confidence in its analysis (0.0 to 1.0).")
    details: Any = Field(..., description="Specific, detailed analysis results from the expert (e.g., MarketRegimeAnalysisDetails).")
    processing_time_ms: float = Field(..., ge=0.0, description="Time taken by the expert to process the request in milliseconds.")
    expert_id: str = Field(..., description="Identifier of the expert that generated this response.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the response.")
    # Add any other common response parameters here

# --- Specific Expert Analysis Details Models ---

class MarketRegimeAnalysisDetails(BaseModel):
    """
    Detailed analysis results from the Market Regime Expert.
    """
    vri_score: float = Field(..., description="VRI 3.0 composite score.")
    regime_id: str = Field(..., description="Identified market regime ID.")
    regime_name: str = Field(..., description="Human-readable name of the market regime.")
    transition_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of a regime transition.")
    volatility_level: str = Field(..., description="Current volatility level (e.g., 'LOW', 'HIGH').")
    trend_direction: str = Field(..., description="Current trend direction (e.g., 'BULLISH', 'BEARISH', 'NEUTRAL').")
    supporting_indicators: List[str] = Field(default_factory=list, description="List of indicators supporting the regime classification.")
    # Add other specific fields for Market Regime Expert here

class OptionsFlowAnalysisDetails(BaseModel):
    """
    Detailed analysis results from the Options Flow Expert.
    """
    vapi_fa_score: float = Field(..., description="VAPI-FA Z-score.")
    dwfd_score: float = Field(..., description="DWFD Z-score.")
    flow_type: str = Field(..., description="Classified options flow type (e.g., 'institutional_accumulation').")
    flow_intensity: str = Field(..., description="Intensity of the options flow.")
    institutional_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of institutional participation.")
    gamma_metrics: List[float] = Field(default_factory=list, description="Key gamma metrics (e.g., net gamma exposure, hedging pressure).")
    # Add other specific fields for Options Flow Expert here

class SentimentAnalysisDetails(BaseModel):
    """
    Detailed analysis results from the Sentiment Expert.
    """
    overall_sentiment_score: float = Field(..., description="Overall market sentiment score (-1.0 to 1.0).")
    sentiment_direction: str = Field(..., description="Direction of sentiment (e.g., 'bullish', 'bearish', 'neutral').")
    sentiment_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the sentiment.")
    fear_greed_index: float = Field(..., ge=0.0, le=100.0, description="Fear & Greed Index.")
    sentiment_drivers: List[str] = Field(default_factory=list, description="Key drivers of the current sentiment.")
    # Add other specific fields for Sentiment Expert here

# --- Models moved from options_flow_expert.py ---

class SDAGAnalysis(BaseModel):
    """PYDANTIC-FIRST: SDAG (Skew and Delta Adjusted GEX) analysis results"""
    multiplicative_sdag: float = Field(..., description="Multiplicative SDAG methodology")
    directional_sdag: float = Field(..., description="Directional SDAG methodology")
    weighted_sdag: float = Field(..., description="Weighted SDAG methodology")
    volatility_focused_sdag: float = Field(..., description="Volatility-focused SDAG methodology")
    sdag_consensus_score: float = Field(..., description="Consensus score across all methodologies")
    sdag_confidence: float = Field(..., description="Confidence in SDAG analysis")
    skew_adjustment_factor: float = Field(default=1.0, description="Skew adjustment factor")
    delta_adjustment_factor: float = Field(default=1.0, description="Delta adjustment factor")
    gamma_exposure_raw: float = Field(default=0.0, description="Raw gamma exposure")
    calculation_timestamp: datetime = Field(default_factory=datetime.now)
    contracts_analyzed: int = Field(default=0, description="Number of contracts analyzed")
    model_config = ConfigDict(extra='forbid')

class DAGAnalysis(BaseModel):
    """PYDANTIC-FIRST: DAG (Delta Adjusted Gamma Exposure) analysis results"""
    multiplicative_dag: float = Field(..., description="Multiplicative DAG approach")
    additive_dag: float = Field(..., description="Additive DAG approach")
    weighted_dag: float = Field(..., description="Weighted DAG approach")
    consensus_dag: float = Field(..., description="Consensus DAG methodology")
    dag_consensus_score: float = Field(..., description="Consensus score across all methodologies")
    dag_confidence: float = Field(..., description="Confidence in DAG analysis")
    delta_exposure_total: float = Field(default=0.0, description="Total delta exposure")
    gamma_exposure_adjusted: float = Field(default=0.0, description="Gamma exposure adjusted")
    dealer_positioning_score: float = Field(default=0.0, description="Dealer positioning score")
    calculation_timestamp: datetime = Field(default_factory=datetime.now)
    strikes_analyzed: int = Field(default=0, description="Number of strikes analyzed")
    model_config = ConfigDict(extra='forbid')

class AdvancedFlowAnalytics(BaseModel):
    """PYDANTIC-FIRST: Advanced flow analytics (VAPI-FA, DWFD, TW-LAF)"""
    vapi_fa_raw: float = Field(..., description="Raw VAPI-FA value")
    vapi_fa_z_score: float = Field(..., description="VAPI-FA Z-score normalized")
    vapi_fa_percentile: float = Field(..., description="VAPI-FA percentile ranking")
    dwfd_raw: float = Field(..., description="Raw DWFD value")
    dwfd_z_score: float = Field(..., description="DWFD Z-score normalized")
    dwfd_institutional_score: float = Field(..., description="Institutional flow detection score")
    tw_laf_raw: float = Field(..., description="Raw TW-LAF value")
    tw_laf_z_score: float = Field(..., description="TW-LAF Z-score normalized")
    tw_laf_momentum_score: float = Field(..., description="Flow momentum score")
    flow_intensity_composite: float = Field(..., description="Composite flow intensity score")
    flow_direction_confidence: float = Field(..., description="Flow direction confidence")
    institutional_probability: float = Field(..., description="Probability of institutional flow")
    calculation_timestamp: datetime = Field(default_factory=datetime.now)
    data_quality_score: float = Field(default=1.0, description="Data quality score")
    model_config = ConfigDict(extra='forbid')

class FlowClassification(BaseModel):
    """PYDANTIC-FIRST: Flow classification and intelligence"""
    flow_type: str = Field(..., description="Primary flow type")
    flow_subtype: str = Field(..., description="Flow subtype")
    flow_intensity: str = Field(..., description="Flow intensity level")
    institutional_probability: float = Field(..., description="Institutional participant probability")
    retail_probability: float = Field(..., description="Retail participant probability")
    dealer_probability: float = Field(..., description="Dealer participant probability")
    directional_bias: str = Field(..., description="Directional bias (bullish/bearish/neutral)")
    time_sensitivity: str = Field(..., description="Time sensitivity (urgent/normal/patient)")
    size_classification: str = Field(..., description="Size classification (small/medium/large/block)")
    sophistication_score: float = Field(..., description="Flow sophistication score")
    information_content: float = Field(..., description="Information content score")
    market_impact_potential: float = Field(..., description="Potential market impact")
    supporting_indicators: List[str] = Field(default_factory=list, description="Supporting indicators")
    confidence_factors: List[str] = Field(default_factory=list, description="Confidence factors")
    model_config = ConfigDict(extra='forbid')

class GammaDynamicsAnalysis(BaseModel):
    """PYDANTIC-FIRST: Gamma dynamics and dealer positioning analysis"""
    total_gamma_exposure: float = Field(..., description="Total gamma exposure")
    call_gamma_exposure: float = Field(..., description="Call gamma exposure")
    put_gamma_exposure: float = Field(..., description="Put gamma exposure")
    net_gamma_exposure: float = Field(..., description="Net gamma exposure")
    dealer_gamma_position: float = Field(..., description="Estimated dealer gamma position")
    dealer_hedging_pressure: float = Field(..., description="Dealer hedging pressure")
    gamma_squeeze_probability: float = Field(..., description="Gamma squeeze probability")
    gamma_acceleration: float = Field(..., description="Gamma acceleration")
    gamma_momentum: float = Field(..., description="Gamma momentum")
    gamma_stability: float = Field(..., description="Gamma stability score")
    upside_gamma_impact: float = Field(..., description="Upside gamma impact")
    downside_gamma_impact: float = Field(..., description="Downside gamma impact")
    gamma_neutral_level: Optional[float] = Field(None, description="Gamma neutral price level")
    model_config = ConfigDict(extra='forbid')
