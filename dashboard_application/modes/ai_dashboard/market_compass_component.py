# dashboard_application/modes/ai_dashboard/market_compass_component.py
"""
Legendary Market Compass Component for EOTS AI Hub v2.5
========================================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module has been ENHANCED to support the full LEGENDARY MARKET COMPASS vision:
- 12 dimensions instead of 4
- Multi-timeframe analysis
- Pattern detection
- Smooth animations
- Interactive features
- Real-time updates via API

The core flow is:
- `create_market_compass_model_from_analysis()`: Transforms orchestrator data into a `MarketCompassModel`.
- `_create_compass_figure()`: Takes the `MarketCompassModel` and generates an enhanced Plotly figure.
- `create_legendary_market_compass_panel()`: Assembles the final Dash component for the UI.

Author: EOTS v2.5 Development Team
Version: 3.0.0 - LEGENDARY EDITION
"""

import logging
import math
from typing import Optional, List, Dict, Any
from datetime import datetime

import plotly.graph_objects as go
from dash import dcc, html
import numpy as np

# EOTS Pydantic Models: The single source of truth for all data structures
from data_models import (
    MOEUnifiedResponseV2_5,
    MarketCompassModel,
    MarketCompassSegment,
    PanelConfigModel,
    ComponentStatus,
    PanelType,
    CompassTheme,
    MarketRegimeAnalysisDetails,
    OptionsFlowAnalysisDetails,
    SentimentAnalysisDetails,
    ProcessedDataBundleV2_5
)

# Import shared components and styling constants
from .components import (
    create_placeholder_card,
    AI_COLORS,
    AI_TYPOGRAPHY,
    get_unified_text_style
)

# Import our enhanced compass engines
from .compass_metrics_engine import CompassMetricsEngine, CompassDimension
from .legendary_compass_component import LegendaryMarketCompass

logger = logging.getLogger(__name__)

# Initialize the engines
metrics_engine = CompassMetricsEngine()
legendary_compass = LegendaryMarketCompass()

# --- Core Data Transformation Logic ---

def create_market_compass_model_from_analysis(
    analysis_bundle: Optional[MOEUnifiedResponseV2_5],
    symbol: str,
    processed_data: Optional[ProcessedDataBundleV2_5] = None
) -> MarketCompassModel:
    """
    ENHANCED: Transforms the raw MOEUnifiedResponseV2_5 and ProcessedDataBundleV2_5 
    into a strictly validated MarketCompassModel with 12 dimensions.

    Args:
        analysis_bundle: The unified response from the HuiHui experts.
        symbol: The target trading symbol.
        processed_data: The processed data bundle with flow and microstructure data.

    Returns:
        A fully populated and validated MarketCompassModel.
    """
    panel_config = PanelConfigModel(
        id=f"market-compass-{symbol}",
        title="Legendary Market Compass",
        panel_type=PanelType.MARKET_COMPASS
    )

    if not analysis_bundle or not analysis_bundle.expert_responses:
        logger.warning(f"Compass: No analysis bundle or expert responses for {symbol}. Returning loading state.")
        return MarketCompassModel(
            panel_config=panel_config,
            segments=[],
            overall_directional_bias=0,
            bias_label="Neutral",
            tactical_summary="Awaiting analysis data..."
        )

    try:
        # If we have processed data, calculate all 12 dimensions
        if processed_data:
            # Calculate all 12 dimensions using the metrics engine
            dimensions = metrics_engine.calculate_all_dimensions(processed_data)
            
            # Detect patterns
            patterns = metrics_engine.detect_confluence_patterns(dimensions)
            
            # Convert dimensions to segments for backward compatibility
            segments = []
            for dim in dimensions[:6]:  # Use first 6 for the basic model
                segments.append(MarketCompassSegment(
                    label=dim.label,
                    score=dim.value,
                    color=dim.color,
                    description=dim.description,
                    tactical_advice=f"Monitor {dim.name} - currently at {dim.value:.0%}"
                ))
            
            # Calculate overall bias from all dimensions
            flow_dims = [d for d in dimensions if d.category == 'flow']
            flow_bias = np.mean([d.value for d in flow_dims]) - 0.5
            
            vol_dims = [d for d in dimensions if d.category == 'volatility']
            vol_level = np.mean([d.value for d in vol_dims])
            
            # Determine overall bias
            if flow_bias > 0.2:
                overall_bias = min(1.0, flow_bias * 2)
                bias_label = "Strong Bullish" if overall_bias > 0.6 else "Bullish"
            elif flow_bias < -0.2:
                overall_bias = max(-1.0, flow_bias * 2)
                bias_label = "Strong Bearish" if overall_bias < -0.6 else "Bearish"
            else:
                overall_bias = flow_bias
                bias_label = "Neutral"
            
            # Generate tactical summary with patterns
            tactical_summary = f"Market is {bias_label.lower()} with {vol_level:.0%} volatility. "
            if patterns:
                top_pattern = patterns[0]
                tactical_summary += f"Pattern detected: {top_pattern['name']} - {top_pattern['action']}"
            
        else:
            # Fallback to basic 4-dimension calculation from analysis bundle
            # Extract data from the three core experts
            regime_details = _get_expert_details(analysis_bundle, "market_regime", MarketRegimeAnalysisDetails)
            flow_details = _get_expert_details(analysis_bundle, "options_flow", OptionsFlowAnalysisDetails)
            sentiment_details = _get_expert_details(analysis_bundle, "sentiment", SentimentAnalysisDetails)

            # --- Calculate Scores for Each Segment (0.0 to 1.0) ---
            regime_score = regime_details.vri_score / 100.0 if regime_details else 0.0
            flow_score = flow_details.vapi_fa_score / 100.0 if flow_details else 0.0
            sentiment_score = (sentiment_details.overall_sentiment_score + 1) / 2 if sentiment_details else 0.5
            volatility_score = _calculate_volatility_score(regime_details)

            scores = {
                "regime": regime_score,
                "flow": flow_score,
                "sentiment": sentiment_score,
                "volatility": volatility_score
            }

            # --- Calculate Overall Directional Bias (-1.0 to 1.0) ---
            overall_bias = _calculate_directional_bias(scores, regime_details, sentiment_details)
            bias_label = _get_bias_label(overall_bias)

            # --- Create Compass Segments ---
            segments = [
                MarketCompassSegment(
                    label="Market Regime",
                    score=scores["regime"],
                    color=AI_COLORS['primary'],
                    description=f"Regime: {regime_details.regime_name if regime_details else 'N/A'}",
                    tactical_advice=_get_regime_advice(regime_details)
                ),
                MarketCompassSegment(
                    label="Options Flow",
                    score=scores["flow"],
                    color=AI_COLORS['secondary'],
                    description=f"Flow Type: {flow_details.flow_type if flow_details else 'N/A'}",
                    tactical_advice=_get_flow_advice(flow_details)
                ),
                MarketCompassSegment(
                    label="Sentiment",
                    score=scores["sentiment"],
                    color=AI_COLORS['accent'],
                    description=f"Direction: {sentiment_details.sentiment_direction if sentiment_details else 'N/A'}",
                    tactical_advice=_get_sentiment_advice(sentiment_details)
                ),
                MarketCompassSegment(
                    label="Volatility",
                    score=scores["volatility"],
                    color=AI_COLORS['success'],
                    description=f"Level: {regime_details.volatility_level if regime_details else 'N/A'}",
                    tactical_advice=_get_volatility_advice(regime_details)
                )
            ]

            # --- Generate Tactical Summary ---
            tactical_summary = f"Overall bias is {bias_label.lower()}. {_get_overall_advice(overall_bias, scores)}"

        panel_config.status = ComponentStatus.OK
        panel_config.last_updated = datetime.now()

        return MarketCompassModel(
            panel_config=panel_config,
            segments=segments,
            overall_directional_bias=overall_bias,
            bias_label=bias_label,
            tactical_summary=tactical_summary
        )

    except Exception as e:
        logger.error(f"Compass: Failed to create MarketCompassModel for {symbol}. Error: {e}", exc_info=True)
        panel_config.status = ComponentStatus.ERROR
        return MarketCompassModel(
            panel_config=panel_config,
            segments=[],
            overall_directional_bias=0,
            bias_label="Error",
            tactical_summary=f"Failed to process analysis data. Please check logs."
        )

# --- Enhanced Visualization Logic ---

def _create_compass_figure(model: MarketCompassModel, processed_data: Optional[ProcessedDataBundleV2_5] = None) -> go.Figure:
    """
    ENHANCED: Creates the Plotly figure for the Market Compass.
    If processed_data is available, creates the full 12-dimension compass.
    Otherwise, falls back to the basic 4-dimension view.
    """
    if processed_data:
        # Use the enhanced legendary compass with 12 dimensions
        try:
            dimensions = metrics_engine.calculate_all_dimensions(processed_data)
            patterns = metrics_engine.detect_confluence_patterns(dimensions)
            return legendary_compass.create_enhanced_compass_figure(dimensions, patterns)
        except Exception as e:
            logger.warning(f"Failed to create enhanced compass, falling back to basic: {e}")
    
    # Fallback to basic compass
    if not model.segments:
        # Return a blank figure if there's no data
        return go.Figure().update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text="No Data", showarrow=False, font=dict(color="white"))]
        )

    r_values = [seg.score * 100 for seg in model.segments]
    theta_values = [seg.label for seg in model.segments]
    colors = [seg.color for seg in model.segments]
    hover_text = [f"{seg.description}<br>{seg.tactical_advice}" for seg in model.segments]

    fig = go.Figure()

    # Add the compass segments with enhanced styling
    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        mode='lines+markers',
        fill='toself',
        marker=dict(
            color=colors, 
            size=15,  # Larger markers
            symbol='diamond',
            line=dict(color='white', width=2)  # White border
        ),
        line=dict(color=AI_COLORS['primary'], width=4),  # Thicker line
        hoverinfo='text',
        text=hover_text,
        fillcolor='rgba(0, 212, 255, 0.2)',
        name='Current State'
    ))

    # Add animated pulse effect for high values
    high_value_indices = [i for i, v in enumerate(r_values) if v > 80]
    if high_value_indices:
        pulse_r = [r_values[i] * 1.1 if i in high_value_indices else r_values[i] for i in range(len(r_values))]
        fig.add_trace(go.Scatterpolar(
            r=pulse_r,
            theta=theta_values,
            mode='markers',
            marker=dict(
                color='rgba(255, 0, 0, 0.3)',
                size=25,
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Configure layout with enhanced styling
    fig.update_layout(
        height=450,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=AI_COLORS['primary'],
            borderwidth=1,
            font=dict(color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0.3)',  # Slight background
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticksuffix='%',
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.4)',
                tickfont=dict(color='white', size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color=AI_COLORS['text']),
                rotation=90,
                direction="clockwise",
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.4)'
            )
        ),
        font=dict(
            family=AI_TYPOGRAPHY.get('font_family', 'Arial'),
            color=AI_COLORS['text']
        ),
        margin=dict(l=60, r=120, t=80, b=60),
        # Enhanced central annotation
        annotations=[
            dict(
                text=f"<b>{model.bias_label}</b><br><span style='font-size:14px'>{model.overall_directional_bias:+.0%}</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=24, color=AI_COLORS['primary']),
                bgcolor="rgba(0,0,0,0.7)",
                borderpad=8,
                bordercolor=AI_COLORS['primary'],
                borderwidth=2,
                opacity=0.9
            )
        ]
    )

    return fig


# --- Main Component Assembly ---

def create_legendary_market_compass_panel(
    compass_model: Optional[MarketCompassModel],
    processed_data: Optional[ProcessedDataBundleV2_5] = None,
    symbol: str = "SPY"
) -> html.Div:
    """
    ENHANCED: Assembles the complete Market Compass panel component for the AI Hub layout.
    Now supports both basic and legendary 12-dimension modes.

    Args:
        compass_model: A validated MarketCompassModel containing all necessary data.
        processed_data: Optional processed data bundle for enhanced 12-dimension view.
        symbol: The trading symbol.

    Returns:
        A Dash html.Div containing the compass visualization or a placeholder.
    """
    # If we have processed data, use the full legendary compass
    if processed_data:
        try:
            return legendary_compass.create_compass_panel(processed_data, symbol)
        except Exception as e:
            logger.warning(f"Failed to create legendary compass panel, falling back to basic: {e}")
    
    # Fallback to basic compass
    if not compass_model or compass_model.panel_config.status in [ComponentStatus.LOADING, ComponentStatus.UNKNOWN]:
        return create_placeholder_card("Market Compass", "Awaiting analysis data...")
    
    if compass_model.panel_config.status == ComponentStatus.ERROR:
        return create_placeholder_card("Market Compass", compass_model.tactical_summary)

    try:
        figure = _create_compass_figure(compass_model, processed_data)
        
        # Assemble the component with enhanced styling
        return html.Div([
            html.H4(
                "🧭 " + compass_model.panel_config.title,
                className="card-title",
                style={
                    **get_unified_text_style("title").model_dump(exclude_none=True),
                    'textAlign': 'center',
                    'marginBottom': '20px'
                }
            ),
            dcc.Graph(
                id=f"{compass_model.panel_config.id}-graph",
                figure=figure,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'market_compass_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                    }
                }
            ),
            html.Div([
                html.P(
                    compass_model.tactical_summary,
                    className="text-muted mt-3",
                    style={
                        **get_unified_text_style("muted").model_dump(exclude_none=True),
                        'textAlign': 'center',
                        'fontSize': '14px',
                        'fontStyle': 'italic'
                    }
                ),
                # Add update timestamp
                html.P(
                    f"Updated: {compass_model.panel_config.last_updated.strftime('%H:%M:%S') if compass_model.panel_config.last_updated else 'N/A'}",
                    style={
                        'textAlign': 'center',
                        'fontSize': '12px',
                        'color': '#808080',
                        'marginTop': '10px'
                    }
                )
            ])
        ], style={
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'padding': '20px',
            'borderRadius': '10px',
            'border': f'1px solid {AI_COLORS["primary"]}40'  # Semi-transparent border
        })
    except Exception as e:
        logger.error(f"Compass Panel: Failed to render component for {compass_model.panel_config.id}. Error: {e}", exc_info=True)
        return create_placeholder_card("Market Compass", "An error occurred during visualization.")


# --- Helper Functions (kept for backward compatibility) ---

def _get_expert_details(bundle, expert_id, model_class):
    """Safely extracts and validates expert details from the bundle."""
    for resp in bundle.expert_responses:
        if resp.expert_id == expert_id and resp.success:
            # The details can be a dict, so we must validate it into our Pydantic model
            return model_class.model_validate(resp.response_data.details)
    return None

def _calculate_volatility_score(regime: Optional[MarketRegimeAnalysisDetails]) -> float:
    if not regime: return 0.5
    level_map = {"LOW": 0.2, "MODERATE": 0.5, "HIGH": 0.8, "EXTREME": 1.0}
    return level_map.get(regime.volatility_level, 0.5)

def _calculate_directional_bias(scores: dict, regime: Optional[MarketRegimeAnalysisDetails], sentiment: Optional[SentimentAnalysisDetails]) -> float:
    """Calculates a weighted directional bias from -1 (bearish) to 1 (bullish)."""
    if not regime or not sentiment: return 0.0
    
    regime_bias = 0
    if regime.trend_direction == "BULLISH": regime_bias = 1
    elif regime.trend_direction == "BEARISH": regime_bias = -1
    
    sentiment_bias = sentiment.overall_sentiment_score # Already -1 to 1
    
    # Weighted average: giving regime and sentiment higher weight
    bias = (scores["regime"] * regime_bias * 0.4) + \
           (scores["sentiment"] * sentiment_bias * 0.4) + \
           (scores["flow"] * (1 if regime_bias >= 0 else -1) * 0.2) # Flow bias depends on regime
    
    return max(-1.0, min(1.0, bias))

def _get_bias_label(bias: float) -> str:
    if bias > 0.6: return "Strong Bullish"
    if bias > 0.2: return "Bullish"
    if bias < -0.6: return "Strong Bearish"
    if bias < -0.2: return "Bearish"
    return "Neutral"

def _get_regime_advice(d: Optional[MarketRegimeAnalysisDetails]) -> str:
    if not d: return "Regime data unavailable."
    return f"Trend is {d.trend_direction.lower()}. Consider strategies aligned with {d.regime_name.lower()} conditions."

def _get_flow_advice(d: Optional[OptionsFlowAnalysisDetails]) -> str:
    if not d: return "Flow data unavailable."
    return f"Institutional probability is {d.institutional_probability:.0%}. Monitor {d.flow_type.lower()} flow for confirmation."

def _get_sentiment_advice(d: Optional[SentimentAnalysisDetails]) -> str:
    if not d: return "Sentiment data unavailable."
    return f"Sentiment is {d.sentiment_direction.lower()} with {d.sentiment_strength:.0%} strength. F&G at {d.fear_greed_index}."

def _get_volatility_advice(d: Optional[MarketRegimeAnalysisDetails]) -> str:
    if not d: return "Volatility data unavailable."
    return f"Volatility is {d.volatility_level.lower()}. Expect { 'range-bound' if d.volatility_level in ['LOW', 'MODERATE'] else 'wider price swings'}."

def _get_overall_advice(bias: float, scores: dict) -> str:
    if abs(bias) < 0.2:
        return "Market is neutral; focus on range-bound strategies or wait for clearer signals."
    if bias > 0:
        return "Bias is bullish. Look for long opportunities, especially if flow and sentiment align."
    else:
        return "Bias is bearish. Consider protective or short strategies, confirming with flow dynamics."