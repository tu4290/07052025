from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, List, Dict, Any

class DashboardDefaults(BaseModel):
    """Default dashboard settings."""
    symbol: str = Field("SPY", description="Default ticker symbol.")
    dte_min: int = Field(0, description="Default minimum DTE.")
    dte_max: int = Field(5, description="Default maximum DTE.")
    price_range_percent: int = Field(5, description="Default price range percentage.")
    update_interval_seconds: int = Field(30, description="Default refresh interval in seconds.")

    model_config = ConfigDict(extra='forbid')

class DashboardModeSettings(BaseModel):
    """Configuration for a single dashboard mode."""
    label: str = Field(..., description="Label for the navigation link.")
    module_name: str = Field(..., description="Module name for the mode's layout.")
    dynamic_thresholds: Optional[Any] = Field(None, description="Dynamic thresholds specific to this mode.")

    model_config = ConfigDict(extra='forbid')

class DashboardModeCollection(BaseModel):
    """A collection of dashboard mode settings."""
    main: DashboardModeSettings = Field(..., description="Main dashboard mode.")
    flow: Optional[DashboardModeSettings] = Field(None, description="Flow analysis mode.")
    volatility: Optional[DashboardModeSettings] = Field(None, description="Volatility analysis mode.")
    structure: Optional[DashboardModeSettings] = Field(None, description="Market structure mode.")
    timedecay: Optional[DashboardModeSettings] = Field(None, description="Time decay analysis mode.")
    advanced: Optional[DashboardModeSettings] = Field(None, description="Advanced analysis mode.")
    ai: Optional[DashboardModeSettings] = Field(None, description="AI insights mode.")

    model_config = ConfigDict(extra='forbid')

class DashboardServerConfig(BaseModel):
    """Configuration for the Dash server and display settings."""
    host: str = Field("127.0.0.1", description="Host address for the Dash server.")
    port: int = Field(8050, description="Port for the Dash server.")
    debug: bool = Field(True, description="Enable Dash debug mode.")
    dev_tools_hot_reload: bool = Field(True, description="Enable hot reloading for development.")
    assets_folder: str = Field("assets", description="Folder for static assets.")
    update_interval_seconds: int = Field(5, description="Interval for dashboard updates in seconds.")
    enable_background_callbacks: bool = Field(True, description="Enable background callbacks for long-running tasks.")
    suppress_callback_exceptions: bool = Field(True, description="Suppress callback exceptions for development.")
    title: str = Field("EOTS v2.5 Dashboard", description="Title of the dashboard.")
    external_stylesheets: List[str] = Field(default_factory=list, description="List of external CSS stylesheets.")
    external_scripts: List[str] = Field(default_factory=list, description="List of external JavaScript scripts.")
    defaults: DashboardDefaults = Field(default_factory=DashboardDefaults, description="Default dashboard settings.")
    modes_detail_config: DashboardModeCollection = Field(default_factory=DashboardModeCollection, description="Detailed configuration for each dashboard mode.")

    model_config = ConfigDict(extra='forbid')