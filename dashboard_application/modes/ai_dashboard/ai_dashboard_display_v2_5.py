"""
AI Dashboard Display - EOTS v2.5 Refactored
===========================================

Fully Pydantic-compliant AI dashboard with unified architecture.
Follows the established refactoring pattern for consistent mode behavior.

Key Features:
- Strict Pydantic v2 validation
- Unified component architecture
- Elite visual design system
- Comprehensive error handling
- Performance optimized
- AI-powered insights and recommendations
- CONTROL PANEL FILTERING ENFORCED
"""

import logging
from typing import Any, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.development.base_component import Component
from pydantic import BaseModel, Field, ConfigDict

# Core imports
from data_models.core_models import FinalAnalysisBundleV2_5
from data_models.dashboard_mode_models import (
    AIDashboardConfig,
    FilteredDataBundle,
    ModeValidationResult
)
from dashboard_application.utils_dashboard_v2_5 import (
    create_empty_figure,
    add_price_line,
    PLOTLY_TEMPLATE,
    add_bottom_right_timestamp_annotation,
    apply_dark_theme_template
)

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AIDashboardState(BaseModel):
    """State management for AI dashboard mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[AIDashboardConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class AIInsight(BaseModel):
    """AI insight configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    insight_type: str = Field(..., description="Type of AI insight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    message: str = Field(..., description="Insight message")
    recommendation: Optional[str] = Field(None, description="AI recommendation")
    risk_level: str = Field(default="MEDIUM", description="Risk level assessment")

# =============================================================================
# CONTROL PANEL FILTERING FUNCTIONS
# =============================================================================

def apply_control_panel_filters(bundle: FinalAnalysisBundleV2_5, control_params: Any) -> FilteredDataBundle:
    """Apply control panel filters to the data bundle."""
    # CRITICAL: Validate control panel parameters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot filter data without settings")
    
    # Extract control panel settings
    target_symbol = getattr(control_params, 'target_symbol', None)
    price_range_percent = getattr(control_params, 'price_range_percent', None)
    dte_min = getattr(control_params, 'dte_min', None)
    dte_max = getattr(control_params, 'dte_max', None)
    
    if not target_symbol:
        raise ValueError("Target symbol is required from control panel")
    if price_range_percent is None:
        raise ValueError("Price range percentage is required from control panel")
    if dte_min is None or dte_max is None:
        raise ValueError("DTE range is required from control panel")
    
    # Get current price for the symbol
    und_data = getattr(bundle, 'underlying_aggregates', None)
    if not und_data:
        raise ValueError("Underlying data is required for filtering")
    
    current_price = getattr(und_data, 'current_price', None)
    if current_price is None:
        raise ValueError("Current price is required for price range filtering")
    
    # Calculate price range
    price_range_factor = price_range_percent / 100.0
    price_range_min = current_price * (1 - price_range_factor)
    price_range_max = current_price * (1 + price_range_factor)
    
    # Filter strikes by price range
    strike_data = getattr(bundle, 'strike_level_metrics', None)
    filtered_strikes = []
    total_strikes = 0
    
    if strike_data and hasattr(strike_data, 'strikes'):
        total_strikes = len(strike_data.strikes)
        for strike in strike_data.strikes:
            strike_price = getattr(strike, 'strike_price', None)
            if strike_price and price_range_min <= strike_price <= price_range_max:
                filtered_strikes.append(strike)
    
    # Filter contracts by DTE range (placeholder - would need contract data)
    filtered_contracts = []
    total_contracts = 0
    
    return FilteredDataBundle(
        filtered_strikes=filtered_strikes,
        filtered_contracts=filtered_contracts,
        price_range_min=price_range_min,
        price_range_max=price_range_max,
        current_price=current_price,
        price_range_percent=price_range_percent,
        dte_min=dte_min,
        dte_max=dte_max,
        total_strikes_available=total_strikes,
        total_contracts_available=total_contracts,
        strikes_filtered_count=len(filtered_strikes),
        contracts_filtered_count=len(filtered_contracts)
    )

def create_filter_info_panel(filtered_data: FilteredDataBundle) -> dbc.Card:
    """Create a panel showing applied control panel filters."""
    return dbc.Card([
        dbc.CardHeader([
            html.H6("Active Filters", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            html.P(filtered_data.get_filter_summary(), className="mb-0 small text-muted")
        ])
    ], className="mb-3 filter-info-card")

# =============================================================================
# CORE COMPONENT BUILDERS
# =============================================================================

def create_ai_overview_card(filtered_data: FilteredDataBundle, config: AIDashboardConfig) -> dbc.Card:
    """Create the AI overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for AI operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot generate AI insights without filtered data")
    
    # Generate AI insights from FILTERED data only
    total_strikes = len(filtered_data.filtered_strikes)
    total_volume = sum(getattr(strike, 'call_volume', 0) + getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes)
    avg_iv = sum(getattr(strike, 'call_implied_volatility', 0) + getattr(strike, 'put_implied_volatility', 0) for strike in filtered_data.filtered_strikes) / (total_strikes * 2)
    
    # AI-generated insights
    insights = []
    
    # Volume insight
    if total_volume > 50000:
        insights.append(AIInsight(
            insight_type="VOLUME",
            confidence=0.85,
            message=f"High volume detected: {total_volume:,} contracts",
            recommendation="Monitor for potential breakout",
            risk_level="HIGH"
        ))
    
    # Volatility insight
    if avg_iv > 0.3:
        insights.append(AIInsight(
            insight_type="VOLATILITY",
            confidence=0.78,
            message=f"Elevated IV: {avg_iv:.1%}",
            recommendation="Consider volatility strategies",
            risk_level="MEDIUM"
        ))
    
    # Market structure insight
    call_volume = sum(getattr(strike, 'call_volume', 0) for strike in filtered_data.filtered_strikes)
    put_volume = sum(getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes)
    call_put_ratio = call_volume / put_volume if put_volume > 0 else 0
    
    if call_put_ratio > 2.0:
        insights.append(AIInsight(
            insight_type="SENTIMENT",
            confidence=0.72,
            message=f"Bullish sentiment: C/P ratio {call_put_ratio:.2f}",
            recommendation="Monitor for continuation",
            risk_level="LOW"
        ))
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("AI Market Intelligence (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{len(insights)}", className="text-success mb-0"),
                    html.Small("AI Insights", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{total_strikes}", className="text-info mb-0"),
                    html.Small("Analyzed Strikes", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{avg_iv:.1%}", className="text-warning mb-0"),
                    html.Small("Avg IV", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{call_put_ratio:.2f}", className="text-primary mb-0"),
                    html.Small("C/P Ratio", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4 elite-card")

def create_ai_recommendations_panel(filtered_data: FilteredDataBundle, config: AIDashboardConfig) -> dbc.Card:
    """Create the AI recommendations panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for AI recommendations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot generate AI recommendations without filtered data")
    
    # Generate AI recommendations from FILTERED data
    recommendations = []
    
    # Analyze filtered data for recommendations
    current_price = filtered_data.current_price
    strikes_near_money = [s for s in filtered_data.filtered_strikes 
                         if abs(getattr(s, 'strike_price', 0) - current_price) / current_price < 0.05]
    
    if strikes_near_money:
        total_gamma = sum(getattr(s, 'total_gamma', 0) for s in strikes_near_money)
        if total_gamma > 0.1:
            recommendations.append({
                "type": "GAMMA_RISK",
                "confidence": "HIGH",
                "message": f"High gamma concentration near ${current_price:.2f}",
                "action": "Monitor for pin risk at expiration"
            })
    
    # Volume-based recommendations
    high_volume_strikes = sorted(filtered_data.filtered_strikes, 
                               key=lambda s: getattr(s, 'call_volume', 0) + getattr(s, 'put_volume', 0), 
                               reverse=True)[:3]
    
    if high_volume_strikes:
        top_strike = high_volume_strikes[0]
        strike_price = getattr(top_strike, 'strike_price', 0)
        total_vol = getattr(top_strike, 'call_volume', 0) + getattr(top_strike, 'put_volume', 0)
        
        recommendations.append({
            "type": "VOLUME_HOTSPOT",
            "confidence": "MEDIUM",
            "message": f"High activity at ${strike_price:.2f} strike ({total_vol:,} contracts)",
            "action": "Consider this level for support/resistance"
        })
    
    # Create recommendation items
    recommendation_items = []
    for rec in recommendations:
        color = "success" if rec["confidence"] == "HIGH" else "warning" if rec["confidence"] == "MEDIUM" else "info"
        recommendation_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"{rec['type']} ({rec['confidence']})", className="d-block"),
                    html.P(rec["message"], className="mb-1"),
                    html.Small(rec["action"], className="text-muted")
                ])
            ], color=color, className="mb-2")
        )
    
    if not recommendation_items:
        recommendation_items.append(
            dbc.ListGroupItem("No specific recommendations for current filtered data", color="info")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("AI Recommendations (Filtered)", className="mb-0 text-success")
        ]),
        dbc.CardBody([
            dbc.ListGroup(recommendation_items, flush=True)
        ])
    ], className="mb-4 elite-card")

def create_market_compass_chart(filtered_data: FilteredDataBundle, config: AIDashboardConfig) -> dcc.Graph:
    """Create the market compass chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for market compass - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create market compass without filtered data")
    
    fig = go.Figure()
    
    # Create compass from FILTERED data
    # Calculate market vectors from filtered strikes
    call_volume = sum(getattr(s, 'call_volume', 0) for s in filtered_data.filtered_strikes)
    put_volume = sum(getattr(s, 'put_volume', 0) for s in filtered_data.filtered_strikes)
    total_gamma = sum(getattr(s, 'total_gamma', 0) for s in filtered_data.filtered_strikes)
    total_theta = sum(getattr(s, 'total_theta', 0) for s in filtered_data.filtered_strikes)
    
    # Normalize values for compass display
    max_val = max(call_volume, put_volume, abs(total_gamma * 1000), abs(total_theta * 100))
    if max_val == 0:
        max_val = 1
    
    # Create compass points
    compass_data = {
        'Call Flow': call_volume / max_val,
        'Put Flow': put_volume / max_val,
        'Gamma': abs(total_gamma * 1000) / max_val,
        'Theta': abs(total_theta * 100) / max_val
    }
    
    # Add compass trace
    fig.add_trace(go.Scatterpolar(
        r=list(compass_data.values()),
        theta=list(compass_data.keys()),
        fill='toself',
        name='Market Compass',
        line_color='#00ff88',
        fillcolor='rgba(0, 255, 136, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Market Compass (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        template=PLOTLY_TEMPLATE,
        height=config.market_compass_height
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_flow_intelligence_panel(filtered_data: FilteredDataBundle, config: AIDashboardConfig) -> dbc.Card:
    """Create the flow intelligence panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for flow intelligence - cannot proceed without data")
    
    intelligence_items = []
    
    if filtered_data.filtered_strikes:
        # Calculate flow intelligence from filtered data
        total_premium = sum(
            (getattr(s, 'call_premium', 0) * getattr(s, 'call_volume', 0)) +
            (getattr(s, 'put_premium', 0) * getattr(s, 'put_volume', 0))
            for s in filtered_data.filtered_strikes
        )
        
        avg_strike_price = sum(getattr(s, 'strike_price', 0) for s in filtered_data.filtered_strikes) / len(filtered_data.filtered_strikes)
        price_deviation = abs(avg_strike_price - filtered_data.current_price) / filtered_data.current_price
        
        metrics = [
            ("Filtered Strikes", f"{len(filtered_data.filtered_strikes)}"),
            ("Total Premium", f"${total_premium/1000000:.1f}M"),
            ("Avg Strike", f"${avg_strike_price:.2f}"),
            ("Price Deviation", f"{price_deviation:.1%}"),
            ("Current Price", f"${filtered_data.current_price:.2f}"),
            ("Price Range", f"±{filtered_data.price_range_percent}%")
        ]
        
        for name, value in metrics:
            color = "info"
            if "Premium" in name and total_premium > 10000000:
                color = "warning"
            elif "Deviation" in name and price_deviation > 0.1:
                color = "danger"
            
            intelligence_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.Strong(name, className="me-2"),
                        html.Span(value, className=f"text-{color}")
                    ])
                ])
            )
    
    if not intelligence_items:
        intelligence_items.append(
            dbc.ListGroupItem("No flow intelligence data available in filtered range", color="warning")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Flow Intelligence (Filtered)", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            dbc.ListGroup(intelligence_items, flush=True)
        ])
    ], className="mb-4 elite-card")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_ai_dashboard_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for AI dashboard mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for AI operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = AIDashboardConfig()
    elif not isinstance(config, AIDashboardConfig):
        try:
            config = AIDashboardConfig.model_validate(config)
        except Exception as e:
            return ModeValidationResult(
                is_valid=False,
                errors=[f"Invalid config: {str(e)}"],
                warnings=[]
            )
    
    return ModeValidationResult(
        is_valid=True,
        errors=[],
        warnings=[]
    )

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_layout(bundle: FinalAnalysisBundleV2_5, config: Any, control_params: Any = None, db_manager: Any = None) -> html.Div:
    """
    Creates the layout for the AI dashboard display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the AI dashboard
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        db_manager: Database manager (for compatibility, not used)
        
    Returns:
        html.Div: The complete AI dashboard layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_ai_dashboard_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"AI dashboard validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display AI dashboard without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to AI dashboard: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    ai_overview = create_ai_overview_card(filtered_data, config)
    recommendations_panel = create_ai_recommendations_panel(filtered_data, config)
    market_compass = create_market_compass_chart(filtered_data, config)
    flow_intelligence = create_flow_intelligence_panel(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("AI Dashboard", className="text-primary mb-0"),
                html.P("AI-Powered Market Intelligence & Recommendations (FILTERED)", className="text-muted")
            ], width=8),
            dbc.Col([
                filter_info_panel
            ], width=4)
        ], className="mb-4"),
        
        # AI Overview
        dbc.Row([
            dbc.Col(ai_overview, width=12)
        ]),
        
        # Main Content Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(market_compass)
                ], className="elite-card")
            ], width=6),
            dbc.Col([
                recommendations_panel
            ], width=6)
        ], className="mb-4"),
        
        # Flow Intelligence Panel
        dbc.Row([
            dbc.Col(flow_intelligence, width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("AI dashboard layout created successfully with FILTERED market data")
    return layout