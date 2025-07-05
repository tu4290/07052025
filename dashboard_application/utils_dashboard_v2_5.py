# dashboard_application/utils_dashboard_v2_5.py
# EOTS v2.5 - PYDANTIC-FIRST DASHBOARD UTILITY MODULE

import logging
from typing import Any, Optional, Union, Dict
from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
from dateutil import parser as date_parser
from pydantic import BaseModel, Field

# EOTS v2.5 Imports
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models.deprecated_files.core_system_config import MainDashboardDisplaySettings, DashboardModeSettings

# --- Module-level logger ---
logger = logging.getLogger(__name__)

# --- Pydantic Models for Dashboard Utilities ---

class DashboardThemeConfig(BaseModel):
    """Dashboard theme configuration"""
    plotly_template: str = "plotly_dark"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S %Z"
    show_chart_timestamps: bool = True
    default_figure_height: int = 400
    
    class Config:
        extra = 'forbid'

class AnnotationConfig(BaseModel):
    """Annotation positioning and styling"""
    x: float = 0.5
    y: float = -0.15
    xref: str = "paper"
    yref: str = "paper"
    xanchor: str = "center"
    yanchor: str = "top"
    showarrow: bool = False
    font_size: int = 9
    font_color: str = "grey"
    
    class Config:
        extra = 'forbid'

class LineStyleConfig(BaseModel):
    """Line style configuration"""
    color: str = "rgba(255, 255, 255, 0.5)"
    width: int = 1
    dash: str = "dash"
    
    class Config:
        extra = 'forbid'

class DashboardUtilsConfig(BaseModel):
    """Main configuration for dashboard utilities"""
    theme: DashboardThemeConfig = Field(default_factory=lambda: DashboardThemeConfig())
    annotation: AnnotationConfig = Field(default_factory=lambda: AnnotationConfig())
    line_style: LineStyleConfig = Field(default_factory=lambda: LineStyleConfig())
    
    class Config:
        extra = 'forbid'

# --- Main Dashboard Utils Class ---

class DashboardUtils:
    """Dashboard utilities with Pydantic configuration integration"""
    
    def __init__(self, config_manager: Optional[ConfigManagerV2_5] = None):
        self.config_manager = config_manager or ConfigManagerV2_5()
        self.config = self._load_config()
        logger.info(f"Dashboard utilities initialized with theme: {self.config.theme.plotly_template}")
    
    def _load_config(self) -> DashboardUtilsConfig:
        """Load configuration from EOTS config manager"""
        try:
            # Get visualization settings from config
            viz_settings = self.config_manager.config.visualization_settings.model_dump()
            
            if isinstance(viz_settings, dict):
                dashboard_config = viz_settings.get("dashboard", {})
                
                # Build theme config
                theme_config = DashboardThemeConfig(
                    plotly_template=dashboard_config.get("theme", "plotly_dark"),
                    timestamp_format=dashboard_config.get("timestamp_format", "%Y-%m-%d %H:%M:%S %Z"),
                    show_chart_timestamps=dashboard_config.get("show_chart_timestamps", True),
                    default_figure_height=dashboard_config.get("height", 400)
                )
                
                return DashboardUtilsConfig(theme=theme_config)
            
        except Exception as e:
            logger.warning(f"Failed to load EOTS config, using defaults: {e}")
        
        return DashboardUtilsConfig()
    
    def create_empty_figure(
        self, 
        title: str, 
        height: Optional[int] = None, 
        reason: str = "No Data Available"
    ) -> go.Figure:
        """Create a standardized empty Plotly figure"""
        
        fig_height = height or self.config.theme.default_figure_height
        
        fig = go.Figure()
        fig.update_layout(
            title_text=title,
            height=fig_height,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            annotations=[{
                'text': reason,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16, 'color': '#888888'}
            }],
            template=self.config.theme.plotly_template
        )
        
        # Apply custom dark theme styling
        self.apply_dark_theme_template(fig)
        
        return fig
    
    def apply_dark_theme_template(self, fig: go.Figure) -> go.Figure:
        """Apply the custom dark theme template to match the dark theme dashboard"""
        fig.update_layout(CUSTOM_PLOTLY_TEMPLATE['layout'])
        return fig
    
    def add_timestamp_annotation(
        self, 
        fig: go.Figure, 
        timestamp: Union[datetime, str, float, None]
    ) -> go.Figure:
        """Add a standardized timestamp annotation to a figure"""
        
        if not self.config.theme.show_chart_timestamps or timestamp is None:
            return fig
        
        ts_str = self._format_timestamp(timestamp)
        if not ts_str:
            return fig
        
        annotation_config = self.config.annotation
        fig.add_annotation(
            text=f"Updated: {ts_str}",
            x=annotation_config.x,
            y=annotation_config.y,
            xref=annotation_config.xref,
            yref=annotation_config.yref,
            showarrow=annotation_config.showarrow,
            xanchor=annotation_config.xanchor,
            yanchor=annotation_config.yanchor,
            font={
                'size': annotation_config.font_size,
                'color': annotation_config.font_color
            }
        )
        
        return fig
    
    def add_price_line(
        self, 
        fig: go.Figure, 
        price: Optional[float], 
        orientation: str = 'vertical', 
        color: Optional[str] = None,
        width: Optional[int] = None,
        dash: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Add a price line to a figure"""
        if price is None or not pd.notna(price):
            return fig
        line_style = self.config.line_style
        # Only use valid properties for Plotly line
        line_config = dict(
            color=color or line_style.color,
            width=width if width is not None else line_style.width,
            dash=dash or line_style.dash
        )
        # Remove any invalid keys from kwargs
        for k in list(kwargs.keys()):
            if k not in ("color", "width", "dash"):
                kwargs.pop(k)
        line_config.update(kwargs)
        if orientation == 'vertical':
            fig.add_vline(x=price, line=line_config)
        elif orientation == 'horizontal':
            fig.add_hline(y=price, line=line_config)
        else:
            logger.warning(f"Invalid orientation '{orientation}'. Use 'vertical' or 'horizontal'.")
        return fig
    
    def _format_timestamp(self, timestamp: Union[datetime, str, float, None]) -> str:
        """Format timestamp according to configuration"""
        if timestamp is None:
            return ""
        
        try:
            ts_dt: Optional[datetime] = None
            
            if isinstance(timestamp, (int, float)):
                if timestamp > 1e12:  # Likely milliseconds
                    ts_dt = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
                else:  # Assume seconds
                    ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            elif isinstance(timestamp, str):
                ts_dt = date_parser.parse(timestamp)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            elif isinstance(timestamp, datetime):
                ts_dt = timestamp
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            
            if ts_dt:
                formatted = ts_dt.strftime(self.config.theme.timestamp_format)
                if '%Z' in self.config.theme.timestamp_format and not ts_dt.strftime('%Z'):
                    formatted = ts_dt.strftime(self.config.theme.timestamp_format.replace(' %Z', '')) + " UTC"
                return formatted
                
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"Could not format timestamp '{str(timestamp)[:100]}': {e}")
            return str(timestamp)[:50] if timestamp else ""
        
        return ""
    
    def get_dashboard_mode_config(self, mode_key: str) -> Optional[DashboardModeSettings]:
        """Get configuration for a specific dashboard mode"""
        try:
            modes_config = self.config_manager.get_setting(
                "visualization_settings.dashboard.modes_detail_config"
            )
            if not hasattr(modes_config, mode_key):
                logger.warning(f"Mode '{mode_key}' not found in configuration")
                return None
            mode_data = getattr(modes_config, mode_key)
            if not isinstance(mode_data, DashboardModeSettings):
                raise TypeError(f"mode_data for '{mode_key}' must be DashboardModeSettings, got {type(mode_data)}")
            return mode_data
        except Exception as e:
            logger.error(f"Error getting dashboard mode config for '{mode_key}': {e}")
            return None
    
    def get_main_dashboard_settings(self) -> MainDashboardDisplaySettings:
        """Get main dashboard display settings"""
        try:
            settings_dict = self.config_manager.get_setting(
                "visualization_settings.dashboard.main_dashboard_settings",
                default={}
            )
            return MainDashboardDisplaySettings(**settings_dict)
            
        except Exception as e:
            logger.error(f"Error getting main dashboard settings: {e}")
            return MainDashboardDisplaySettings()
    
    def apply_standard_layout(
        self, 
        fig: go.Figure, 
        title: Optional[str] = None, 
        height: Optional[int] = None
    ) -> go.Figure:
        """Apply standard layout settings to a figure"""
        
        layout_updates: Dict[str, Any] = {
            'template': self.config.theme.plotly_template
        }
        
        if title:
            layout_updates['title_text'] = title
        
        if height:
            layout_updates['height'] = height
        elif self.config.theme.default_figure_height:
            layout_updates['height'] = self.config.theme.default_figure_height
        
        fig.update_layout(**layout_updates)
        return fig

    def add_bottom_right_timestamp_annotation(self, fig: go.Figure, timestamp: Union[datetime, str, float, None]) -> go.Figure:
        """Add a timestamp annotation at the bottom right of the chart."""
        if timestamp is None:
            return fig
        ts_str = self._format_timestamp(timestamp)
        if not ts_str:
            return fig
        fig.add_annotation(
            text=f"Updated: {ts_str}",
            x=1.0,
            y=-0.18,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font={
                'size': 9,
                'color': 'grey'
            }
        )
        return fig

# --- Global Instance ---
_dashboard_utils: Optional[DashboardUtils] = None

def initialize_dashboard_utils(config_manager: Optional[ConfigManagerV2_5] = None) -> DashboardUtils:
    """Initialize the global dashboard utilities"""
    global _dashboard_utils
    _dashboard_utils = DashboardUtils(config_manager=config_manager)
    return _dashboard_utils

def get_dashboard_utils() -> DashboardUtils:
    """Get the global dashboard utilities instance"""
    global _dashboard_utils
    if _dashboard_utils is None:
        _dashboard_utils = initialize_dashboard_utils()
    return _dashboard_utils

# --- Convenience Functions (for backward compatibility) ---

def create_empty_figure(title: str, height: Optional[int] = None, reason: str = "No Data Available") -> go.Figure:
    """Create a standardized empty Plotly figure (convenience function)"""
    utils = get_dashboard_utils()
    return utils.create_empty_figure(title=title, height=height, reason=reason)

def add_timestamp_annotation(fig: go.Figure, timestamp: Union[datetime, str, float, None]) -> go.Figure:
    """Add a standardized timestamp annotation (convenience function)"""
    utils = get_dashboard_utils()
    return utils.add_timestamp_annotation(fig, timestamp)

def add_price_line(fig: go.Figure, price: Optional[float], orientation: str = 'vertical', **kwargs) -> go.Figure:
    """Add a price line to a figure (convenience function)"""
    utils = get_dashboard_utils()
    return utils.add_price_line(fig, price, orientation, **kwargs)

def add_bottom_right_timestamp_annotation(fig: go.Figure, timestamp: Union[datetime, str, float, None]) -> go.Figure:
    """Add a timestamp annotation at the bottom right of the chart."""
    utils = get_dashboard_utils()
    return utils.add_bottom_right_timestamp_annotation(fig, timestamp)

def apply_dark_theme_template(fig: go.Figure) -> go.Figure:
    """Apply the custom dark theme template to match the dark theme dashboard."""
    utils = get_dashboard_utils()
    return utils.apply_dark_theme_template(fig)

# --- Enhanced Plotly Template Configuration ---
# Custom template that matches the dark theme dashboard
CUSTOM_PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparent background
        'plot_bgcolor': 'rgba(0,0,0,0)',   # Transparent plot area
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'color': '#E8E8E8',  # --text-primary
            'size': 12
        },
        'title': {
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'color': '#E8E8E8',  # --text-primary
                'size': 16
            }
        },
        'xaxis': {
            'gridcolor': '#2A2A2A',    # --border-primary
            'gridwidth': 0.5,
            'griddash': 'dot',
            'linecolor': '#888888',     # --text-muted
            'linewidth': 1,
            'tickcolor': '#888888',     # --text-muted
            'tickfont': {
                'family': 'JetBrains Mono, Consolas, monospace',
                'color': '#888888',     # --text-muted
                'size': 12
            },
            'title': {
                'font': {
                    'family': 'Inter, sans-serif',
                    'color': '#B8B8B8',     # --text-secondary
                    'size': 14
                }
            },
            'showspikes': False,       # Disable spike lines for cleaner hover
            'spikemode': 'across',     # Spike mode when enabled
            'spikesnap': 'cursor'      # Snap spikes to cursor
        },
        'yaxis': {
            'gridcolor': '#2A2A2A',    # --border-primary
            'gridwidth': 0.5,
            'griddash': 'dot',
            'linecolor': '#888888',     # --text-muted
            'linewidth': 1,
            'tickcolor': '#888888',     # --text-muted
            'tickfont': {
                'family': 'JetBrains Mono, Consolas, monospace',
                'color': '#888888',     # --text-muted
                'size': 12
            },
            'title': {
                'font': {
                    'family': 'Inter, sans-serif',
                    'color': '#B8B8B8',     # --text-secondary
                    'size': 14
                }
            },
            'showspikes': False,       # Disable spike lines for cleaner hover
            'spikemode': 'across',     # Spike mode when enabled
            'spikesnap': 'cursor'      # Snap spikes to cursor
        },
        'legend': {
            'font': {
                'family': 'Inter, sans-serif',
                'color': '#B8B8B8',     # --text-secondary
                'size': 12
            },
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(0,0,0,0)'
        },
        'hovermode': 'x unified',      # Unified hover mode to reduce obstruction
        'hoverdistance': 20,           # Distance threshold for hover activation
        'spikedistance': 15,           # Distance for spike lines
        'dragmode': 'pan',             # Default interaction mode
        'selectdirection': 'd',         # Selection direction for better UX (diagonal)
        'hoverlabel': {
            'bgcolor': '#1E1E1E',       # --bg-elevated
            'bordercolor': '#2A2A2A',   # --border-primary
            'font': {
                'family': 'Inter, sans-serif',
                'color': '#E8E8E8',     # --text-primary
                'size': 12
            },
            'namelength': -1,           # Show full names in tooltips
            'align': 'left'             # Align tooltip text to left
        },
        'hoversubplots': 'axis',       # Constrain hover to subplot area
        'hoverlabel_align': 'auto',    # Auto-align tooltips to avoid obstruction
        'colorway': [
            '#4A9EFF',  # --accent-primary (soft blue)
            '#FFB84A',  # --accent-secondary (muted amber)
            '#8B5CF6',  # --accent-tertiary (soft violet)
            '#10B981',  # --positive (emerald)
            '#EF4444',  # --negative (refined red)
            '#6B7280',  # --neutral (gray)
            '#5BADFF',  # --accent-primary-hover
            '#FFC866',  # --accent-secondary-hover
            '#A78BFA',  # --accent-tertiary-hover
        ],
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 60}
    }
}

# --- Backward Compatibility Constants ---
PLOTLY_TEMPLATE = "plotly_dark"

def apply_universal_filtering_hardwired(data: Any) -> Any:
    """Universal filtering function for dashboard data processing"""
    # Placeholder implementation - this function provides universal filtering
    # for all dashboard modes to ensure consistent data processing
    logger.debug("Applying universal filtering to data")
    return data