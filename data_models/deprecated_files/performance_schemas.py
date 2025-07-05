"""
Performance tracking and analytics models for the Elite Options Trading System v2.5.

This module defines Pydantic models for tracking system performance, backtesting results,
and execution metrics in a consistent, type-safe manner.
"""
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from .performance_config_schemas import (
    PerformanceMetadata, StrategyParameters, PerformanceSummary,
    SystemHealthMetrics, RiskMetrics, MarketConditions
)

class PerformanceInterval(str, Enum):
    """Time intervals for performance metrics aggregation."""
    MINUTE_1 = "1m"; MINUTE_5 = "5m"; MINUTE_15 = "15m"; MINUTE_30 = "30m"
    HOUR_1 = "1h"; HOUR_4 = "4h"; HOUR_12 = "12h"; DAILY = "1d"; WEEKLY = "1w"; MONTHLY = "1M"

class PerformanceMetricType(str, Enum):
    """Types of performance metrics that can be tracked."""
    LATENCY = "latency"; THROUGHPUT = "throughput"; MEMORY = "memory"; CPU = "cpu"; NETWORK = "network"
    CACHE_HIT = "cache_hit"; CACHE_MISS = "cache_miss"; ERROR_RATE = "error_rate"; SUCCESS_RATE = "success_rate"
    ORDER_EXECUTION = "order_execution"; DATA_QUALITY = "data_quality"; BACKTEST_RETURN = "backtest_return"
    SHARPE_RATIO = "sharpe_ratio"; MAX_DRAWDOWN = "max_drawdown"; WIN_RATE = "win_rate"; PROFIT_FACTOR = "profit_factor"

class PerformanceMetricV2_5(BaseModel):
    """Base model for performance metrics with common fields and validation."""
    metric_type: PerformanceMetricType = Field(..., description="Type of performance metric being recorded.")
    component: str = Field(..., description="Component or subsystem this metric applies to.")
    value: float = Field(..., description="Numeric value of the metric.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this metric was recorded (UTC).")
    interval: PerformanceInterval = Field(PerformanceInterval.MINUTE_1, description="Aggregation interval for this metric.")
    metadata: PerformanceMetadata = Field(default_factory=PerformanceMetadata, description="Additional context-specific metadata.")
    model_config = ConfigDict(extra='forbid')

    @field_validator('value')
    def validate_value(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Value must be numeric")
        if v < 0 and cls.metric_type not in [PerformanceMetricType.ERROR_RATE, PerformanceMetricType.MAX_DRAWDOWN]:
            raise ValueError(f"Negative values not allowed for {cls.metric_type}")
        return float(v)

class SystemPerformanceV2_5(BaseModel):
    """System-level performance metrics and health indicators."""
    cpu_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current CPU usage percentage (0-100).")
    memory_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current memory usage percentage (0-100).")
    disk_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current disk usage percentage (0-100).")
    network_latency_ms: float = Field(..., ge=0.0, description="Average network latency in milliseconds.")
    active_processes: int = Field(..., ge=0, description="Number of active processes.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this snapshot was taken (UTC).")
    model_config = ConfigDict(extra='forbid')

class BacktestPerformanceV2_5(BaseModel):
    """Comprehensive backtest performance metrics and statistics."""
    strategy_name: str = Field(..., description="Name or identifier of the backtested strategy.")
    start_date: datetime = Field(..., description="Backtest start date (UTC).")
    end_date: datetime = Field(..., description="Backtest end date (UTC).")
    total_return_pct: float = Field(..., description="Total return percentage over the backtest period.")
    annualized_return_pct: float = Field(..., description="Annualized return percentage.")
    annualized_volatility_pct: float = Field(..., ge=0.0, description="Annualized volatility percentage.")
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return metric (higher is better).")
    sortino_ratio: Optional[float] = Field(None, description="Risk-adjusted return focusing on downside volatility.")
    max_drawdown_pct: float = Field(..., le=0.0, description="Maximum peak-to-trough decline (negative percentage).")
    win_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of winning trades.")
    profit_factor: float = Field(..., ge=0.0, description="Gross profit divided by gross loss.")
    total_trades: int = Field(..., ge=0, description="Total number of trades executed.")
    avg_trade_duration: timedelta = Field(..., description="Average duration of trades.")
    params: StrategyParameters = Field(default_factory=StrategyParameters, description="Strategy parameters used in this backtest.")
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_dates(self) -> 'BacktestPerformanceV2_5':
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        return self

class ExecutionMetricsV2_5(BaseModel):
    """Detailed metrics for trade execution quality and performance."""
    order_id: str = Field(..., description="Unique identifier for the order.")
    symbol: str = Field(..., description="Traded symbol.")
    order_type: str = Field(..., description="Type of order (e.g., 'market', 'limit').")
    side: str = Field(..., description="'buy' or 'sell'.")
    quantity: float = Field(..., gt=0, description="Number of contracts/shares.")
    target_price: float = Field(..., gt=0, description="Target or limit price.")
    avg_fill_price: float = Field(..., gt=0, description="Average fill price.")
    slippage: float = Field(..., description="Difference between target and actual fill price.")
    slippage_pct: float = Field(..., description="Slippage as percentage of target price.")
    execution_time_ms: int = Field(..., ge=0, description="Time to execute the order in milliseconds.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the order was executed (UTC).")
    model_config = ConfigDict(extra='forbid')

class PerformanceReportV2_5(BaseModel):
    """Comprehensive performance report combining multiple metrics and analyses."""
    report_id: str = Field(..., description="Unique identifier for this report.")
    start_time: datetime = Field(..., description="Start of the reporting period (UTC).")
    end_time: datetime = Field(..., description="End of the reporting period (UTC).")
    system_metrics: List[SystemPerformanceV2_5] = Field(default_factory=list, description="System performance snapshots.")
    backtest_results: List[BacktestPerformanceV2_5] = Field(default_factory=list, description="Backtest performance results.")
    execution_metrics: List[ExecutionMetricsV2_5] = Field(default_factory=list, description="Trade execution metrics.")
    custom_metrics: Dict[str, List[PerformanceMetricV2_5]] = Field(default_factory=dict, description="Additional custom metrics.")
    summary: PerformanceSummary = Field(default_factory=PerformanceSummary, description="Aggregated summary statistics.")
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_report_period(self) -> 'PerformanceReportV2_5':
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self