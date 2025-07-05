import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

class MarketRegime(Enum):
    """Market regime classifications for dynamic adaptation"""
    LOW_VOL_TRENDING = "low_vol_trending"
    LOW_VOL_RANGING = "low_vol_ranging"
    MEDIUM_VOL_TRENDING = "medium_vol_trending"
    MEDIUM_VOL_RANGING = "medium_vol_ranging"
    HIGH_VOL_TRENDING = "high_vol_trending"
    HIGH_VOL_RANGING = "high_vol_ranging"
    STRESS_REGIME = "stress_regime"
    EXPIRATION_REGIME = "expiration_regime"
    REGIME_UNCLEAR_OR_TRANSITIONING = "REGIME_UNCLEAR_OR_TRANSITIONING"

class FlowType(Enum):
    """Flow classification types for institutional intelligence"""
    RETAIL_UNSOPHISTICATED = "retail_unsophisticated"
    RETAIL_SOPHISTICATED = "retail_sophisticated"
    INSTITUTIONAL_SMALL = "institutional_small"
    INSTITUTIONAL_LARGE = "institutional_large"
    HEDGE_FUND = "hedge_fund"
    MARKET_MAKER = "market_maker"
    UNKNOWN = "unknown"

class EliteConfig(BaseModel):
    """Pydantic model for elite impact calculation configuration."""
    regime_detection_enabled: bool = Field(default=True, description="Enable dynamic regime adaptation")
    enable_elite_regime_detection: bool = Field(default=True, description="Enable elite-driven market regime detection")
    regime_lookback_periods: Dict[str, int] = Field(default_factory=lambda: {'short': 20, 'medium': 60, 'long': 252}, description="Lookback periods for regime detection")
    cross_expiration_enabled: bool = Field(default=True, description="Enable cross-expiration modeling")
    expiration_decay_lambda: float = Field(default=0.1, description="Decay lambda for expiration modeling")
    max_expirations_tracked: int = Field(default=12, description="Maximum expirations tracked")
    flow_classification_enabled: bool = Field(default=True, description="Enable institutional flow intelligence")
    institutional_threshold_percentile: float = Field(default=95.0, description="Percentile for institutional flow threshold")
    flow_momentum_periods: List[int] = Field(default_factory=lambda: [5, 15, 30, 60], description="Periods for flow momentum analysis")
    volatility_surface_enabled: bool = Field(default=True, description="Enable volatility surface integration")
    skew_adjustment_alpha: float = Field(default=1.0, description="Alpha for skew adjustment")
    surface_stability_threshold: float = Field(default=0.15, description="Threshold for volatility surface stability")
    momentum_detection_enabled: bool = Field(default=True, description="Enable momentum-acceleration detection")
    acceleration_threshold_multiplier: float = Field(default=2.0, description="Multiplier for acceleration threshold")
    momentum_persistence_threshold: float = Field(default=0.7, description="Threshold for momentum persistence")
    enable_caching: bool = Field(default=True, description="Enable caching for performance optimization")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    enable_sdag_calculation: bool = Field(default=True, description="Enable SDAG calculation")
    enable_dag_calculation: bool = Field(default=True, description="Enable DAG calculation")
    enable_advanced_greeks: bool = Field(default=True, description="Enable advanced Greeks calculation")
    enable_flow_clustering: bool = Field(default=True, description="Enable flow clustering")

class ConvexValueColumns:
    """Comprehensive ConvexValue column definitions"""
    # Basic option parameters
    OPT_KIND = 'opt_kind'
    STRIKE = 'strike'
    EXPIRATION = 'expiration'
    EXPIRATION_TS = 'expiration_ts'
    
    # Greeks
    DELTA = 'delta'
    GAMMA = 'gamma'
    THETA = 'theta'
    VEGA = 'vega'
    RHO = 'rho'
    VANNA = 'vanna'
    VOMMA = 'vomma'
    CHARM = 'charm'
    
    # Open Interest multiplied metrics
    DXOI = 'dxoi'  # Delta x Open Interest
    GXOI = 'gxoi'  # Gamma x Open Interest
    VXOI = 'vxoi'  # Vega x Open Interest
    TXOI = 'txoi'  # Theta x Open Interest
    VANNAXOI = 'vannaxoi'  # Vanna x Open Interest
    VOMMAXOI = 'vommaxoi'  # Vomma x Open Interest
    CHARMXOI = 'charmxoi'  # Charm x Open Interest
    
    # Volume multiplied metrics
    DXVOLM = 'dxvolm'  # Delta x Volume
    GXVOLM = 'gxvolm'  # Gamma x Volume
    VXVOLM = 'vxvolm'  # Vega x Volume
    TXVOLM = 'txvolm'    # Theta x Volume
    VANNAXVOLM = 'vannaxvolm'  # Vanna x Volume
    VOMMAXVOLM = 'vommaxvolm'  # Vomma x Volume
    CHARMXVOLM = 'charmxvolm'  # Charm x Volume
    
    # Flow metrics
    VALUE_BS = 'value_bs'  # Buy Value - Sell Value
    VOLM_BS = 'volm_bs'    # Buy Volume - Sell Volume
    
    # Multi-timeframe flow metrics
    VOLMBS_5M = 'volmbs_5m'
    VOLMBS_15M = 'volmbs_15m'
    VOLMBS_30M = 'volmbs_30m'
    VOLMBS_60M = 'volmbs_60m'
    
    VALUEBS_5M = 'valuebs_5m'
    VALUEBS_15M = 'valuebs_15m'
    VALUEBS_30M = 'valuebs_30m'
    VALUEBS_60M = 'valuebs_60m'
    
    # Call/Put specific metrics
    CALL_GXOI = 'call_gxoi'
    CALL_DXOI = 'call_dxoi'
    PUT_GXOI = 'put_gxoi'
    PUT_DXOI = 'put_dxoi'
    
    # Advanced flow metrics
    FLOWNET = 'flownet'  # Net flow calculation
    VFLOWRATIO = 'vflowratio'  # Volume flow ratio
    PUT_CALL_RATIO = 'put_call_ratio'
    
    # Volatility metrics
    VOLATILITY = 'volatility'
    FRONT_VOLATILITY = 'front_volatility'
    BACK_VOLATILITY = 'back_volatility'
    
    # Open Interest
    OI = 'oi'
    OI_CH = 'oi_ch'

class EliteImpactColumns:
    """Elite impact calculation output columns"""
    # Basic impact metrics
    DELTA_IMPACT_RAW = 'delta_impact_raw'
    GAMMA_IMPACT_RAW = 'gamma_impact_raw'
    VEGA_IMPACT_RAW = 'vega_impact_raw'
    THETA_IMPACT_RAW = 'theta_impact_raw'
    
    # Advanced impact metrics
    VANNA_IMPACT_RAW = 'vanna_impact_raw'
    VOMMA_IMPACT_RAW = 'vomma_impact_raw'
    CHARM_IMPACT_RAW = 'charm_impact_raw'
    
    # Elite composite metrics
    SDAG_MULTIPLICATIVE = 'sdag_multiplicative'
    SDAG_DIRECTIONAL = 'sdag_directional'
    SDAG_WEIGHTED = 'sdag_weighted'
    SDAG_VOLATILITY_FOCUSED = 'sdag_volatility_focused'
    SDAG_CONSENSUS = 'sdag_consensus'
    
    DAG_MULTIPLICATIVE = 'dag_multiplicative'
    DAG_DIRECTIONAL = 'dag_directional'
    DAG_WEIGHTED = 'dag_weighted'
    DAG_VOLATILITY_FOCUSED = 'dag_volatility_focused'
    DAG_CONSENSUS = 'dag_consensus'
    
    # Market structure metrics
    STRIKE_MAGNETISM_INDEX = 'strike_magnetism_index'
    VOLATILITY_PRESSURE_INDEX = 'volatility_pressure_index'
    FLOW_MOMENTUM_INDEX = 'flow_momentum_index'
    INSTITUTIONAL_FLOW_SCORE = 'institutional_flow_score'
    
    # Regime-adjusted metrics
    REGIME_ADJUSTED_GAMMA = 'regime_adjusted_gamma'
    REGIME_ADJUSTED_DELTA = 'regime_adjusted_delta'
    REGIME_ADJUSTED_VEGA = 'regime_adjusted_vega'
    
    # Cross-expiration metrics
    CROSS_EXP_GAMMA_SURFACE = 'cross_exp_gamma_surface'
    EXPIRATION_TRANSITION_FACTOR = 'expiration_transition_factor'
    
    # Momentum metrics
    FLOW_VELOCITY_5M = 'flow_velocity_5m'
    FLOW_VELOCITY_15M = 'flow_velocity_15m'
    FLOW_ACCELERATION = 'flow_acceleration'
    MOMENTUM_PERSISTENCE = 'momentum_persistence'
    
    # Classification outputs
    MARKET_REGIME = 'market_regime'
    FLOW_TYPE = 'flow_type'
    VOLATILITY_REGIME = 'volatility_regime'
    
    # Elite performance metrics
    ELITE_IMPACT_SCORE = 'elite_impact_score'
    PREDICTION_CONFIDENCE = 'prediction_confidence'
    SIGNAL_STRENGTH = 'signal_strength'
