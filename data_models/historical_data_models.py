from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class OHLCVItem(BaseModel):
    """Represents a single OHLCV (Open, High, Low, Close, Volume) data point."""
    date: str = Field(..., description="Date of the OHLCV data in YYYY-MM-DD format.")
    open: float = Field(..., description="Opening price.")
    high: float = Field(..., description="Highest price.")
    low: float = Field(..., description="Lowest price.")
    close: float = Field(..., description="Closing price.")
    volume: int = Field(..., description="Trading volume.")

    model_config = ConfigDict(extra='forbid')

class HistoricalDataResponse(BaseModel):
    """Represents a response containing historical OHLCV data for a symbol."""
    symbol: str = Field(..., description="The ticker symbol.")
    data: List[OHLCVItem] = Field(..., description="List of OHLCV data points.")
    columns: List[str] = Field(..., description="List of column names in the data.")

    model_config = ConfigDict(extra='forbid')