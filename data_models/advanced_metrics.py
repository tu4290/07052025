
"""
Pydantic models for advanced, derived options metrics.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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
