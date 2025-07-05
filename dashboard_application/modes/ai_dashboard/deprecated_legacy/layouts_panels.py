"""
AI Hub Panels Module - Individual Panel Components v2.5
======================================================

This module contains individual panel creation functions for the AI Hub:
- Trade recommendations panel
- Market analysis panel  
- Legendary market compass panel
- Pydantic-first validation throughout

Author: EOTS v2.5 Development Team
Version: 2.5.1 (Modular)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from dash import dcc, html
import plotly.graph_objects as go

# EOTS Schema imports - Pydantic-first validation
from data_models import FinalAnalysisBundleV2_5
from data_models import ProcessedDataBundleV2_5, ProcessedUnderlyingAggregatesV2_5
from data_models import ActiveRecommendationPayloadV2_5

# Import existing components - preserve dependencies
from .components import (
    AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS,
    create_placeholder_card, get_card_style, create_clickable_title_with_info
)
from .visualizations import create_legendary_market_compass  # CRITICAL FIX: Import missing function

import compliance_decorators_v2_5

logger = logging.getLogger(__name__)

@compliance_decorators_v2_5.track_compliance("trade_recommendations_panel", "Trade Recommendations Panel")
def create_trade_recommendations_panel(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str
) -> html.Div:
    """
    Create trade recommendations panel with Pydantic-first validation.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        ai_settings: AI configuration settings
        symbol: Trading symbol
        
    Returns:
        html.Div: Trade recommendations panel
    """
    try:
        # Extract ATIF recommendations using Pydantic model access
        atif_recs = bundle_data.atif_recommendations_v2_5 or []
        
        # Extract processed data for additional context
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None
        
        recommendations = []
        
        # Process ATIF recommendations with Pydantic validation
        for rec in atif_recs[:3]:  # Show top 3 recommendations
            conviction = getattr(rec, 'conviction_score', 0.0) or 0.0
            strategy_type = getattr(rec, 'strategy_type', 'Unknown')
            rationale = getattr(rec, 'rationale', 'No rationale provided')
            
            # Determine recommendation styling based on conviction
            if conviction >= 0.8:
                rec_style = "success"
                rec_icon = "📈"
            elif conviction >= 0.6:
                rec_style = "warning"
                rec_icon = "⚡"
            else:
                rec_style = "info"
                rec_icon = "💡"
            
            recommendations.append(
                html.Div([
                    html.Div([
                        html.Span(rec_icon, className="recommendation-icon"),
                        html.Strong(f"{strategy_type}", className="recommendation-title"),
                        html.Span(f"{conviction:.0%}", className=f"badge badge-{rec_style} ml-2")
                    ], className="recommendation-header"),
                    
                    html.P(rationale[:100] + "..." if len(rationale) > 100 else rationale,
                           className="recommendation-text"),
                    
                    html.Div([
                        html.Small(f"Risk: {getattr(rec, 'risk_level', 'Moderate')}", 
                                 className="text-muted mr-3"),
                        html.Small(f"Timeframe: {getattr(rec, 'timeframe', 'Intraday')}", 
                                 className="text-muted")
                    ], className="recommendation-meta")
                    
                ], className="recommendation-item mb-2", style={
                    "border": f"1px solid {AI_COLORS.get(rec_style, AI_COLORS['info'])}",
                    "borderRadius": AI_EFFECTS['border_radius_sm'],
                    "padding": AI_SPACING['sm'],
                    "background": f"rgba({AI_COLORS.get(rec_style, AI_COLORS['info'])}, 0.1)"
                })
            )
        
        # If no recommendations, show placeholder
        if not recommendations:
            recommendations = [
                html.Div([
                    html.P("🤖 AI is analyzing current market conditions...", 
                           className="text-muted text-center"),
                    html.Small("Recommendations will appear when high-conviction setups are detected.",
                              className="text-muted text-center d-block")
                ], className="p-3")
            ]
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🎯 AI Trade Recommendations",
                    "trade_recommendations",
                    "AI-generated trade ideas with conviction scoring and risk assessment"
                )
            ], className="panel-header"),
            
            html.Div(recommendations, className="recommendations-container")
            
        ], style=get_card_style('primary'))
        
    except Exception as e:
        logger.error(f"Error creating trade recommendations panel: {str(e)}")
        return create_placeholder_card("🎯 Trade Recommendations", f"Error: {str(e)}")

@compliance_decorators_v2_5.track_compliance("market_analysis_panel", "Market Analysis Panel")
def create_market_analysis_panel(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str
) -> html.Div:
    """
    Create market analysis panel with Pydantic-first validation.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        ai_settings: AI configuration settings
        symbol: Trading symbol
        
    Returns:
        html.Div: Market analysis panel
    """
    try:
        # Extract data using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None
        
        if not enriched_data:
            return create_placeholder_card("🧠 Market Analysis", "No analysis data available")
        
        # Extract key metrics with safe access
        vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
        dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
        tw_laf = getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0
        vri_2_0 = getattr(enriched_data, 'vri_2_0_und', 0.0) or 0.0
        
        # Determine market state based on metrics
        if vapi_fa > 1.5 and dwfd > 1.0:
            market_state = "🔥 Strong Bullish Flow"
            state_color = AI_COLORS['success']
        elif vapi_fa < -1.5 and dwfd < -1.0:
            market_state = "🐻 Strong Bearish Flow"
            state_color = AI_COLORS['danger']
        elif abs(vri_2_0) > 10000:
            market_state = "⚡ High Volatility Environment"
            state_color = AI_COLORS['warning']
        else:
            market_state = "📊 Neutral/Consolidating"
            state_color = AI_COLORS['info']
        
        # Generate key insights
        insights = []
        
        if abs(vapi_fa) > 2.0:
            insights.append(f"• Extreme VAPI-FA reading ({vapi_fa:.2f}) indicates unusual premium flow")
        
        if abs(dwfd) > 1.5:
            direction = "institutional buying" if dwfd > 0 else "institutional selling"
            insights.append(f"• DWFD signals strong {direction} pressure")
        
        if abs(tw_laf) > 1.5:
            insights.append(f"• TW-LAF shows sustained flow conviction ({tw_laf:.2f})")
        
        if abs(vri_2_0) > 15000:
            insights.append(f"• Elevated VRI 2.0 ({vri_2_0:.0f}) suggests volatility expansion")
        
        if not insights:
            insights = ["• Market showing balanced flow dynamics", "• No extreme readings detected"]
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🧠 Market Analysis",
                    "market_analysis",
                    "Real-time market state analysis using EOTS metrics"
                )
            ], className="panel-header"),
            
            html.Div([
                # Current market state
                html.Div([
                    html.H6("Current State:", className="analysis-label"),
                    html.Span(market_state, style={"color": state_color, "fontWeight": "bold"})
                ], className="mb-3"),
                
                # Key insights
                html.Div([
                    html.H6("Key Insights:", className="analysis-label"),
                    html.Div([
                        html.P(insight, className="insight-item") for insight in insights
                    ])
                ], className="mb-3"),
                
                # Risk factors
                html.Div([
                    html.H6("Risk Factors:", className="analysis-label"),
                    html.P("• Monitor for regime transitions", className="risk-item"),
                    html.P("• Watch key support/resistance levels", className="risk-item")
                ])
                
            ], className="analysis-content")
            
        ], style=get_card_style('analysis'))
        
    except Exception as e:
        logger.error(f"Error creating market analysis panel: {str(e)}")
        return create_placeholder_card("🧠 Market Analysis", f"Error: {str(e)}")

@compliance_decorators_v2_5.track_compliance("legendary_market_compass_panel", "Legendary Market Compass Panel")
def create_legendary_market_compass_panel(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str
) -> html.Div:
    """
    Create the legendary market compass panel with Pydantic-first validation.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        ai_settings: AI configuration settings
        symbol: Trading symbol
        
    Returns:
        html.Div: Legendary market compass panel
    """
    try:
        # Extract data using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        enriched_data = processed_data.underlying_data_enriched if processed_data else None
        
        if not enriched_data:
            return create_placeholder_card("🧭 Market Compass", "No compass data available")

        # Extract key metrics for compass display (CRITICAL FIX: Move before function call)
        metrics_data = {}
        if enriched_data:
            metrics_data = {
                'VAPI-FA': getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0,
                'DWFD': getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0,
                'TW-LAF': getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0,
                'VRI 2.0': (getattr(enriched_data, 'vri_2_0_und', 0.0) or 0.0) / 10000,  # Normalize
                'A-DAG': (getattr(enriched_data, 'a_dag_total_und', 0.0) or 0.0) / 100000,  # Normalize
                'GIB': (getattr(enriched_data, 'gib_oi_based_und', 0.0) or 0.0) / 100000   # Normalize
            }

        # Create legendary market compass visualization (CRITICAL FIX: Add after metrics_data is populated)
        compass_figure = create_legendary_market_compass(metrics_data, symbol)

        # Calculate overall compass strength
        compass_strength = sum(abs(v) for v in metrics_data.values()) / len(metrics_data) if metrics_data else 0
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    f"🧭 Market Compass - {symbol}",
                    "market_compass",
                    "12-dimensional market intelligence radar with multi-timeframe analysis"
                )
            ], className="panel-header"),
            
            html.Div([
                # Main compass visualization
                dcc.Graph(
                    figure=compass_figure,
                    config={'displayModeBar': False},
                    style={"height": "300px"}
                ),
                
                # Compass metrics summary
                html.Div([
                    html.Div([
                        html.Small("Compass Strength:", className="metric-label"),
                        html.Strong(f"{compass_strength:.2f}", className="metric-value")
                    ], className="compass-metric"),
                    
                    html.Div([
                        html.Small("Active Dimensions:", className="metric-label"),
                        html.Strong(f"{sum(1 for v in metrics_data.values() if abs(v) > 0.5)}/6", 
                                   className="metric-value")
                    ], className="compass-metric")
                ], className="compass-summary d-flex justify-content-between")
                
            ], className="compass-content")
            
        ], style=get_card_style('primary'))
        
    except Exception as e:
        logger.error(f"Error creating market compass panel: {str(e)}")
        return create_placeholder_card("🧭 Market Compass", f"Error: {str(e)}")

