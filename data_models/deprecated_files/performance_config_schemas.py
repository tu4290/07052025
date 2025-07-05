"""Performance configuration schemas for the Elite Options Trading System v2.5.

This module defines Pydantic models to replace Dict[str, Any] patterns in performance
tracking and analytics, providing type safety and validation.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class PerformanceMetadata(BaseModel):
    """Metadata for performance metrics with context-specific information."""
    source: Optional[str] = Field(None, description="Source of the metric")
    environment: Optional[str] = Field(None, description="Environment where metric was collected")
    version: Optional[str] = Field(None, description="Version of the component")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    custom_fields: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom metadata fields")
    model_config = ConfigDict(extra='forbid')

class StrategyParameters(BaseModel):
    """Strategy parameters used in backtesting."""
    lookback_period: Optional[int] = Field(None, description="Lookback period for analysis")
    risk_threshold: Optional[float] = Field(None, description="Risk threshold parameter")
    position_size: Optional[float] = Field(None, description="Position sizing parameter")
    stop_loss_pct: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, description="Take profit percentage")
    max_positions: Optional[int] = Field(None, description="Maximum number of positions")
    rebalance_frequency: Optional[str] = Field(None, description="Rebalancing frequency")
    custom_params: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom strategy parameters")
    model_config = ConfigDict(extra='forbid')

class PerformanceSummary(BaseModel):
    """Aggregated summary statistics for performance reports."""
    total_trades: Optional[int] = Field(None, description="Total number of trades")
    winning_trades: Optional[int] = Field(None, description="Number of winning trades")
    losing_trades: Optional[int] = Field(None, description="Number of losing trades")
    avg_return_pct: Optional[float] = Field(None, description="Average return percentage")
    avg_win_pct: Optional[float] = Field(None, description="Average winning trade percentage")
    avg_loss_pct: Optional[float] = Field(None, description="Average losing trade percentage")
    largest_win_pct: Optional[float] = Field(None, description="Largest winning trade percentage")
    largest_loss_pct: Optional[float] = Field(None, description="Largest losing trade percentage")
    consecutive_wins: Optional[int] = Field(None, description="Maximum consecutive wins")
    consecutive_losses: Optional[int] = Field(None, description="Maximum consecutive losses")
    avg_trade_duration_hours: Optional[float] = Field(None, description="Average trade duration in hours")
    total_fees: Optional[float] = Field(None, description="Total fees paid")
    net_profit: Optional[float] = Field(None, description="Net profit after fees")
    roi_pct: Optional[float] = Field(None, description="Return on investment percentage")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio (annual return / max drawdown)")
    recovery_factor: Optional[float] = Field(None, description="Recovery factor")
    expectancy: Optional[float] = Field(None, description="Mathematical expectancy")
    kelly_criterion: Optional[float] = Field(None, description="Kelly criterion percentage")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    cvar_95: Optional[float] = Field(None, description="Conditional Value at Risk (95% confidence)")
    custom_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom summary metrics")
    model_config = ConfigDict(extra='forbid')

class SystemHealthMetrics(BaseModel):
    """System health and operational metrics."""
    uptime_hours: Optional[float] = Field(None, description="System uptime in hours")
    error_count: Optional[int] = Field(None, description="Number of errors")
    warning_count: Optional[int] = Field(None, description="Number of warnings")
    memory_peak_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
    cpu_peak_pct: Optional[float] = Field(None, description="Peak CPU usage percentage")
    disk_io_mb: Optional[float] = Field(None, description="Disk I/O in MB")
    network_io_mb: Optional[float] = Field(None, description="Network I/O in MB")
    cache_hit_rate_pct: Optional[float] = Field(None, description="Cache hit rate percentage")
    database_connections: Optional[int] = Field(None, description="Number of database connections")
    api_calls_count: Optional[int] = Field(None, description="Number of API calls made")
    response_time_avg_ms: Optional[float] = Field(None, description="Average response time in milliseconds")
    throughput_per_second: Optional[float] = Field(None, description="Throughput per second")
    custom_health_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom health metrics")
    model_config = ConfigDict(extra='forbid')

class RiskMetrics(BaseModel):
    """Risk assessment and management metrics."""
    portfolio_var: Optional[float] = Field(None, description="Portfolio Value at Risk")
    portfolio_cvar: Optional[float] = Field(None, description="Portfolio Conditional Value at Risk")
    beta: Optional[float] = Field(None, description="Portfolio beta")
    alpha: Optional[float] = Field(None, description="Portfolio alpha")
    correlation_spy: Optional[float] = Field(None, description="Correlation with SPY")
    correlation_vix: Optional[float] = Field(None, description="Correlation with VIX")
    downside_deviation: Optional[float] = Field(None, description="Downside deviation")
    tracking_error: Optional[float] = Field(None, description="Tracking error")
    information_ratio: Optional[float] = Field(None, description="Information ratio")
    treynor_ratio: Optional[float] = Field(None, description="Treynor ratio")
    custom_risk_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom risk metrics")
    model_config = ConfigDict(extra='forbid')

class MarketConditions(BaseModel):
    """Market conditions during performance measurement."""
    market_regime: Optional[str] = Field(None, description="Market regime (bull, bear, sideways)")
    volatility_regime: Optional[str] = Field(None, description="Volatility regime (low, medium, high)")
    spy_return_pct: Optional[float] = Field(None, description="SPY return percentage")
    vix_level: Optional[float] = Field(None, description="VIX level")
    interest_rate_pct: Optional[float] = Field(None, description="Risk-free interest rate")
    sector_rotation: Optional[str] = Field(None, description="Sector rotation pattern")
    earnings_season: Optional[bool] = Field(None, description="Whether it's earnings season")
    fomc_week: Optional[bool] = Field(None, description="Whether it's FOMC week")
    expiration_week: Optional[bool] = Field(None, description="Whether it's options expiration week")
    custom_conditions: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Custom market conditions")
    model_config = ConfigDict(extra='forbid')