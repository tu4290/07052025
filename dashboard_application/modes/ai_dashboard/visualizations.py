"""
AI Dashboard Visualizations Module for EOTS v2.5
================================================

This module contains all chart and graph creation functions for the AI dashboard,
refactored to be fully Pydantic-first compliant. All functions accept validated
Pydantic models instead of dictionaries, ensuring type safety and adherence to
the ZERO DICT ACCEPTANCE policy.

Author: EOTS v2.5 Development Team (Refactored)
Version: 2.5.2
"""

import logging
from typing import Optional
import plotly.graph_objects as go
from pydantic import BaseModel, confloat

# Import Pydantic models for chart data and configuration
from data_models import MarketCompassModel, GaugeConfigModel

# Import styling constants
from .components import AI_COLORS

logger = logging.getLogger(__name__)


# Pydantic model for the regime transition gauge configuration
class RegimeTransitionGaugeConfig(BaseModel):
    """Configuration model for the regime transition gauge."""
    transition_prob: confloat(ge=0.0, le=1.0)
    regime_confidence: confloat(ge=0.0, le=1.0)


def create_legendary_market_compass(compass_model: MarketCompassModel) -> go.Figure:
    """
    Creates the Legendary Market Compass from a validated Pydantic model.
    This function is purely for rendering and contains no business logic.

    Args:
        compass_model (MarketCompassModel): A validated Pydantic model containing all data for the compass.

    Returns:
        go.Figure: A Plotly graph object representing the Market Compass.
    """
    try:
        if not compass_model or not compass_model.segments:
            logger.warning("No compass model or segments provided. Returning empty figure.")
            return go.Figure()

        # Extract data directly from the validated Pydantic model
        labels = [seg.label for seg in compass_model.segments]
        scores = [seg.score * 100 for seg in compass_model.segments]  # Scale to 0-100 for display
        colors = [seg.color for seg in compass_model.segments]
        hover_text = [
            f"<b>{seg.label}</b><br>Score: {seg.score:.2f}<br><i>{seg.tactical_advice}</i>"
            for seg in compass_model.segments
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=labels,
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.2)',
            line=dict(color=AI_COLORS['primary'], width=3),
            marker=dict(color=colors, size=10, symbol='diamond'),
            name='Market Compass',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text
        ))

        # Layout and Styling
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0, 0, 0, 0.1)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10, color='white'),
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)',
                    ticksuffix='%'
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, color='white', family='Arial Black'),
                    gridcolor='rgba(255, 255, 255, 0.3)',
                    linecolor='rgba(255, 255, 255, 0.5)'
                )
            ),
            title=dict(
                text=f'🧭 {compass_model.panel_config.title.upper()}',
                x=0.5, y=0.95, font=dict(size=20, color='white', family='Arial Black')
            ),
            annotations=[
                dict(
                    text=f"<b>{compass_model.bias_label}</b>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=24, color=AI_COLORS['primary']),
                    bgcolor="rgba(0,0,0,0.5)", borderpad=4,
                    bordercolor=AI_COLORS['primary'], borderwidth=1, opacity=0.8
                )
            ],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating Legendary Market Compass: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 0.95)', plot_bgcolor='rgba(0, 0, 0, 0)',
            height=500,
            annotations=[dict(text="Compass Unavailable", showarrow=False,
                              font=dict(color=AI_COLORS['danger'], size=16))]
        )
        return fig


def create_metric_gauge(config: GaugeConfigModel) -> go.Figure:
    """
    Creates a standardized, styled gauge from a Pydantic configuration model.

    Args:
        config (GaugeConfigModel): A Pydantic model containing all gauge parameters.

    Returns:
        go.Figure: A Plotly graph object representing the gauge.
    """
    try:
        # Determine color based on how extreme the value is
        if abs(config.value) >= 2.0: color = AI_COLORS['danger']
        elif abs(config.value) >= 1.5: color = AI_COLORS['warning']
        elif abs(config.value) >= 1.0: color = AI_COLORS['primary']
        else: color = AI_COLORS['success']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=config.value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': config.title, 'font': {'size': 12, 'color': 'white'}},
            number={'font': {'size': 16, 'color': color}},
            gauge={
                'axis': {'range': [config.range_min, config.range_max], 'tickcolor': 'white', 'tickfont': {'size': 8}},
                'bar': {'color': color, 'thickness': 0.7},
                'bgcolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255, 255, 255, 0.3)',
                'steps': [
                    {'range': [config.range_min, -1], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [-1, 1], 'color': 'rgba(107, 207, 127, 0.2)'},
                    {'range': [1, config.range_max], 'color': 'rgba(255, 71, 87, 0.2)'}
                ]
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'}, height=config.height, margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating metric gauge for {config.title}: {e}", exc_info=True)
        return go.Figure()


def create_regime_transition_gauge(config: RegimeTransitionGaugeConfig) -> go.Figure:
    """
    Creates a gauge visualizing regime transition probability from a Pydantic model.

    Args:
        config (RegimeTransitionGaugeConfig): A Pydantic model with transition probability.

    Returns:
        go.Figure: A Plotly graph object representing the transition gauge.
    """
    try:
        if config.transition_prob >= 0.7: gauge_color, gauge_level = AI_COLORS['danger'], "High Risk"
        elif config.transition_prob >= 0.4: gauge_color, gauge_level = AI_COLORS['warning'], "Moderate Risk"
        else: gauge_color, gauge_level = AI_COLORS['success'], "Low Risk"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=config.transition_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Transition Risk: {gauge_level}", 'font': {'size': 12, 'color': 'white'}},
            number={'font': {'size': 18, 'color': gauge_color}, 'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'size': 9}},
                'bar': {'color': gauge_color, 'thickness': 0.7},
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(107, 207, 127, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 70 # High risk threshold
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'}, height=150, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating regime transition gauge: {e}", exc_info=True)
        return go.Figure()
