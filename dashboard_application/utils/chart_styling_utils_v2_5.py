# dashboard_application/utils/chart_styling_utils_v2_5.py
# EOTS v2.5 - UNIFIED CHART STYLING UTILITIES
# CRITICAL FIX: Centralized chart styling to eliminate duplicate functions across all dashboard modes

from typing import Optional, Tuple, Dict, Any
from data_models.dashboard_ui_models import GraphStyleConfig
import logging

logger = logging.getLogger(__name__)

def create_pydantic_graph_style(
    height: str = "400px", 
    width: str = "100%", 
    min_height: str = "350px", 
    max_height: Optional[str] = None,
    enable_mode_bar: bool = False,
    enable_logo: bool = False,
    responsive: bool = True,
    container_aware: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    CRITICAL FIX: Unified Pydantic v2 compliant graph styling for ALL dashboard modes.
    
    This function eliminates the duplicate _create_pydantic_graph_style functions
    across Flow, Volatility, Time Decay, Structure, and Main dashboard modes.
    
    Args:
        height: Graph height (default: "400px")
        width: Graph width (default: "100%")
        min_height: Minimum height (default: "350px")
        max_height: Maximum height (default: None for unlimited)
        enable_mode_bar: Whether to show Plotly mode bar (default: False)
        enable_logo: Whether to show Plotly logo (default: False)
        responsive: Enable responsive behavior (default: True)
        container_aware: Enable container-aware sizing (default: True)
    
    Returns:
        Tuple of (style_dict, config_dict) for Dash Graph components
    """
    try:
        graph_style_config = GraphStyleConfig(
            width=width,
            height=height,
            min_height=min_height,
            max_height=max_height,
            display_mode_bar=enable_mode_bar,
            display_logo=enable_logo,
            responsive=responsive,
            container_aware=container_aware,
            overflow="visible"
        )
        
        return graph_style_config.to_dash_style(), graph_style_config.to_dash_config()
        
    except Exception as e:
        logger.error(f"Error creating Pydantic graph style: {e}", exc_info=True)
        # Fallback to basic styling if GraphStyleConfig fails
        return (
            {
                'width': width,
                'height': height,
                'minHeight': min_height,
                'overflow': 'visible',
                'display': 'flex',
                'flexDirection': 'column',
                'boxSizing': 'border-box'
            },
            {
                'displayModeBar': enable_mode_bar,
                'displaylogo': enable_logo,
                'responsive': responsive
            }
        )

def create_heatmap_style(
    calculated_height: Optional[int] = None,
    enable_autosize: bool = True,
    min_height: str = "400px"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    CRITICAL FIX: Specialized styling for heatmap charts that prevents cutoffs.
    
    This function provides heatmap-specific styling that addresses the chart
    cutoff issues identified in Flow Mode and other dashboard modes.
    
    Args:
        calculated_height: Calculated height in pixels (optional)
        enable_autosize: Whether to enable autosize (default: True)
        min_height: Minimum height (default: "400px")
    
    Returns:
        Tuple of (style_dict, config_dict) for Dash Graph components
    """
    if enable_autosize:
        # For autosize, use flexible height without restrictive max
        return create_pydantic_graph_style(
            height="auto",
            min_height=min_height,
            max_height=None,  # No max height restriction for heatmaps
            responsive=True,
            container_aware=True
        )
    else:
        # For fixed size, use calculated height with reasonable bounds
        height_px = f"{calculated_height}px" if calculated_height else "600px"
        return create_pydantic_graph_style(
            height=height_px,
            min_height=min_height,
            max_height=None,  # No max height restriction for heatmaps
            responsive=True,
            container_aware=True
        )

def create_gauge_style(
    height: str = "300px",
    width: str = "100%"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    CRITICAL FIX: Specialized styling for gauge charts.
    
    Args:
        height: Gauge height (default: "300px")
        width: Gauge width (default: "100%")
    
    Returns:
        Tuple of (style_dict, config_dict) for Dash Graph components
    """
    return create_pydantic_graph_style(
        height=height,
        width=width,
        min_height="250px",
        max_height=None,
        responsive=True,
        container_aware=True
    )

def create_line_chart_style(
    height: str = "400px",
    width: str = "100%"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    CRITICAL FIX: Specialized styling for line charts.
    
    Args:
        height: Chart height (default: "400px")
        width: Chart width (default: "100%")
    
    Returns:
        Tuple of (style_dict, config_dict) for Dash Graph components
    """
    return create_pydantic_graph_style(
        height=height,
        width=width,
        min_height="350px",
        max_height=None,
        responsive=True,
        container_aware=True
    )

def create_bar_chart_style(
    height: str = "400px",
    width: str = "100%"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    CRITICAL FIX: Specialized styling for bar charts.
    
    Args:
        height: Chart height (default: "400px")
        width: Chart width (default: "100%")
    
    Returns:
        Tuple of (style_dict, config_dict) for Dash Graph components
    """
    return create_pydantic_graph_style(
        height=height,
        width=width,
        min_height="350px",
        max_height=None,
        responsive=True,
        container_aware=True
    )

# CRITICAL FIX: Standard CSS class names for consistent chart containers
CHART_CONTAINER_CLASSES = "chart-container plotly-chart-container"

# CRITICAL FIX: Standard chart configuration for all modes
STANDARD_CHART_CONFIG = {
    'displayModeBar': False,
    'displaylogo': False,
    'responsive': True
}

# CRITICAL FIX: Standard chart style for all modes
STANDARD_CHART_STYLE = {
    'width': '100%',
    'height': 'auto',
    'minHeight': '350px',
    'overflow': 'visible',
    'display': 'flex',
    'flexDirection': 'column',
    'boxSizing': 'border-box'
}
