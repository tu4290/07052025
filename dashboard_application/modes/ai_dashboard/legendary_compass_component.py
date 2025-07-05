"""
ENHANCED LEGENDARY MARKET COMPASS COMPONENT
===========================================

Advanced Plotly-based compass with:
- 12 dimensions
- Multi-timeframe support
- Smooth animations
- Pattern detection
- Interactive features
- Real-time updates via API polling
"""

import logging
import math
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, callback_context
import dash
from dash.dependencies import Input, Output, State
import numpy as np

# EOTS Pydantic Models
from data_models import (
    MOEUnifiedResponseV2_5,
    MarketCompassModel,
    MarketCompassSegment,
    PanelConfigModel,
    ComponentStatus,
    PanelType,
    CompassTheme
)

# Import our new metrics engine
from .compass_metrics_engine import CompassMetricsEngine, CompassDimension

# Import shared components
from .components import (
    create_placeholder_card,
    AI_COLORS,
    AI_TYPOGRAPHY,
    get_unified_text_style
)

logger = logging.getLogger(__name__)

class LegendaryMarketCompass:
    """
    Enhanced Market Compass with 12 dimensions and advanced features.
    Uses Plotly for visualization with smooth animations and interactivity.
    """
    
    def __init__(self):
        self.metrics_engine = CompassMetricsEngine()
        self.animation_frame_duration = 500  # ms
        self.history_length = 100  # Keep last 100 states for playback
        self.compass_history = []
        
        # Timeframe configurations
        self.timeframes = {
            '5m': {'opacity': 0.8, 'width': 3, 'scale': 1.0},
            '15m': {'opacity': 0.6, 'width': 4, 'scale': 0.95},
            '1h': {'opacity': 0.4, 'width': 5, 'scale': 0.90},
            '4h': {'opacity': 0.2, 'width': 6, 'scale': 0.85}
        }
        
        # Alert thresholds
        self.extreme_threshold = 0.95
        self.confluence_threshold = 0.8
        
    def create_enhanced_compass_figure(
        self,
        dimensions: List[CompassDimension],
        patterns: List[Dict[str, Any]],
        timeframe: str = 'all'
    ) -> go.Figure:
        """
        Create an enhanced Plotly figure with 12 dimensions and advanced features.
        """
        fig = go.Figure()
        
        # Prepare data for plotting
        labels = [d.label for d in dimensions]
        
        # Add trace for each timeframe or combined view
        if timeframe == 'all':
            # Multi-timeframe view with translucent layers
            for tf, config in self.timeframes.items():
                # Simulate different values for each timeframe (in production, get from API)
                values = [d.value * 100 * config['scale'] for d in dimensions]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the shape
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor=f'rgba(0, 212, 255, {config["opacity"] * 0.3})',
                    line=dict(
                        color=self._get_timeframe_color(tf),
                        width=config['width']
                    ),
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        symbol='diamond',
                        color=[self._get_alert_color(v/100) for v in values]
                    ),
                    name=tf,
                    text=[f"{d.description}<br>Value: {d.value:.2f}<br>Raw: {d.raw_value:.2f}" 
                          for d in dimensions],
                    hoverinfo='text+name'
                ))
        else:
            # Single timeframe view
            values = [d.value * 100 for d in dimensions]
            colors = [d.color for d in dimensions]
            
            # Check for extreme values
            extreme_indices = [i for i, v in enumerate(values) if v > self.extreme_threshold * 100]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                fillcolor='rgba(0, 212, 255, 0.2)',
                line=dict(color=AI_COLORS['primary'], width=4),
                mode='lines+markers+text',
                marker=dict(
                    size=[20 if i in extreme_indices else 12 for i in range(len(values))],
                    symbol='diamond',
                    color=colors,
                    line=dict(
                        color=['red' if i in extreme_indices else 'white' for i in range(len(values))],
                        width=[3 if i in extreme_indices else 1 for i in range(len(values))]
                    )
                ),
                text=['⚡' if i in extreme_indices else '' for i in range(len(values))],
                textposition='top center',
                name='Current',
                customdata=dimensions,
                hovertemplate='<b>%{customdata.label}</b><br>' +
                             '%{customdata.description}<br>' +
                             'Value: %{r:.1f}%<br>' +
                             'Raw: %{customdata.raw_value:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Configure layout for professional appearance
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                x=1.1,
                y=1,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor=AI_COLORS['primary'],
                borderwidth=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='rgba(0,0,0,0.3)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    ticksuffix='%',
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.4)',
                    tickfont=dict(color='white', size=10),
                    layer='below traces'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='white', family='monospace'),
                    rotation=90,
                    direction="clockwise",
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.4)'
                )
            ),
            font=dict(
                family=AI_TYPOGRAPHY.get('font_family', 'Arial'),
                color='white'
            ),
            margin=dict(l=80, r=150, t=100, b=80),
            
            # Title with pattern detection
            title=dict(
                text=self._create_title_with_patterns(patterns),
                font=dict(size=20, color=AI_COLORS['primary']),
                x=0.5,
                xanchor='center'
            ),
            
            # Annotations for center display and alerts
            annotations=self._create_annotations(dimensions, patterns)
        )
        
        # Add animation frames for smooth transitions
        if self.compass_history:
            frames = self._create_animation_frames()
            fig.frames = frames
            
            # Add play button
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.1,
                    'y': -0.15,
                    'buttons': [
                        {
                            'label': '▶ Play History',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': self.animation_frame_duration},
                                'transition': {'duration': 300},
                                'fromcurrent': True,
                                'mode': 'immediate'
                            }]
                        },
                        {
                            'label': '⏸ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0},
                                'transition': {'duration': 0},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }]
            )
        
        return fig
    
    def _get_timeframe_color(self, timeframe: str) -> str:
        """Get color for each timeframe."""
        colors = {
            '5m': '#00ff00',   # Green
            '15m': '#00ffff',  # Cyan
            '1h': '#ff00ff',   # Magenta
            '4h': '#ffff00'    # Yellow
        }
        return colors.get(timeframe, AI_COLORS['primary'])
    
    def _get_alert_color(self, value: float) -> str:
        """Get color based on value intensity."""
        if value > 0.9:
            return '#ff0000'  # Red for extreme
        elif value > 0.7:
            return '#ff8000'  # Orange for high
        elif value > 0.3:
            return '#ffff00'  # Yellow for medium
        else:
            return '#00ff00'  # Green for low
    
    def _create_title_with_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Create dynamic title with detected patterns."""
        if patterns:
            top_pattern = max(patterns, key=lambda p: p['strength'])
            return f"🧭 LEGENDARY MARKET COMPASS<br><span style='font-size:14px;color:#ff8000;'>⚡ {top_pattern['name']}: {top_pattern['description']}</span>"
        return "🧭 LEGENDARY MARKET COMPASS"
    
    def _create_annotations(
        self, 
        dimensions: List[CompassDimension],
        patterns: List[Dict[str, Any]]
    ) -> List[dict]:
        """Create annotations for center display and alerts."""
        annotations = []
        
        # Calculate overall market state
        avg_value = np.mean([d.value for d in dimensions])
        volatility_dims = [d for d in dimensions if d.category == 'volatility']
        flow_dims = [d for d in dimensions if d.category == 'flow']
        
        volatility_avg = np.mean([d.value for d in volatility_dims])
        flow_avg = np.mean([d.value for d in flow_dims])
        
        # Determine market state
        if avg_value > 0.7:
            state = "🔥 EXTREME"
            color = "#ff0000"
        elif avg_value > 0.5:
            state = "⚡ ACTIVE"
            color = "#ff8000"
        elif avg_value > 0.3:
            state = "📊 NORMAL"
            color = "#00ff00"
        else:
            state = "😴 QUIET"
            color = "#808080"
        
        # Center annotation with market state
        annotations.append({
            'text': f"<b>{state}</b><br><span style='font-size:12px'>Vol: {volatility_avg:.0%} | Flow: {flow_avg:.0%}</span>",
            'x': 0.5,
            'y': 0.5,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': dict(size=24, color=color),
            'bgcolor': 'rgba(0,0,0,0.7)',
            'borderpad': 10,
            'bordercolor': color,
            'borderwidth': 2
        })
        
        # Add pattern alerts
        for i, pattern in enumerate(patterns[:3]):  # Show top 3 patterns
            annotations.append({
                'text': f"⚡ {pattern['action']}",
                'x': 0.02,
                'y': 0.98 - (i * 0.08),
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': dict(size=12, color='#ff8000'),
                'bgcolor': 'rgba(0,0,0,0.5)',
                'borderpad': 4,
                'align': 'left'
            })
        
        # Add extreme value alerts
        extreme_dims = [d for d in dimensions if d.value > self.extreme_threshold]
        if extreme_dims:
            alert_text = "🚨 EXTREME: " + ", ".join([d.label for d in extreme_dims[:3]])
            annotations.append({
                'text': alert_text,
                'x': 0.5,
                'y': -0.1,
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': dict(size=14, color='#ff0000'),
                'bgcolor': 'rgba(255,0,0,0.2)',
                'borderpad': 6
            })
        
        return annotations
    
    def _create_animation_frames(self) -> List[go.Frame]:
        """Create animation frames from historical data."""
        frames = []
        
        for i, hist_data in enumerate(self.compass_history[-20:]):  # Last 20 states
            values = [d.value * 100 for d in hist_data['dimensions']]
            
            frame = go.Frame(
                data=[go.Scatterpolar(
                    r=values + [values[0]],
                    theta=[d.label for d in hist_data['dimensions']] + [hist_data['dimensions'][0].label]
                )],
                name=str(i),
                traces=[0]
            )
            frames.append(frame)
        
        return frames
    
    def update_compass_history(self, dimensions: List[CompassDimension], patterns: List[Dict[str, Any]]):
        """Store historical states for playback."""
        self.compass_history.append({
            'timestamp': datetime.now(),
            'dimensions': dimensions.copy(),
            'patterns': patterns.copy()
        })
        
        # Keep only recent history
        if len(self.compass_history) > self.history_length:
            self.compass_history.pop(0)
    
    def create_compass_panel(
        self,
        data_bundle: Optional[Any] = None,
        symbol: str = "SPY"
    ) -> html.Div:
        """
        Create the complete compass panel with enhanced features.
        """
        if not data_bundle:
            return create_placeholder_card(
                "🧭 Legendary Market Compass",
                "Awaiting market data..."
            )
        
        try:
            # Calculate all 12 dimensions
            dimensions = self.metrics_engine.calculate_all_dimensions(data_bundle)
            
            # Detect patterns
            patterns = self.metrics_engine.detect_confluence_patterns(dimensions)
            
            # Update history
            self.update_compass_history(dimensions, patterns)
            
            # Create figure
            figure = self.create_enhanced_compass_figure(dimensions, patterns)
            
            # Build panel
            return html.Div([
                # Header with controls
                html.Div([
                    html.H3(
                        "🧭 LEGENDARY MARKET COMPASS",
                        style={
                            'color': AI_COLORS['primary'],
                            'marginBottom': '10px',
                            'fontFamily': 'monospace'
                        }
                    ),
                    
                    # Timeframe selector
                    html.Div([
                        html.Label("Timeframe: ", style={'color': 'white', 'marginRight': '10px'}),
                        dcc.RadioItems(
                            id='compass-timeframe-selector',
                            options=[
                                {'label': 'All Layers', 'value': 'all'},
                                {'label': '5 Min', 'value': '5m'},
                                {'label': '15 Min', 'value': '15m'},
                                {'label': '1 Hour', 'value': '1h'},
                                {'label': '4 Hour', 'value': '4h'}
                            ],
                            value='all',
                            inline=True,
                            style={'color': 'white'}
                        )
                    ], style={'marginBottom': '20px'})
                ], style={'textAlign': 'center'}),
                
                # Compass visualization
                dcc.Graph(
                    id='legendary-compass-graph',
                    figure=figure,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'compass_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                        }
                    }
                ),
                
                # Pattern insights
                html.Div([
                    html.H4("🎯 Detected Patterns", style={'color': AI_COLORS['accent']}),
                    html.Div([
                        html.Div([
                            html.Span(f"⚡ {p['name']}", style={
                                'color': '#ff8000',
                                'fontWeight': 'bold',
                                'marginRight': '10px'
                            }),
                            html.Span(p['action'], style={'color': 'white'}),
                            html.Span(f" ({p['strength']:.0%})", style={
                                'color': '#808080',
                                'marginLeft': '10px'
                            })
                        ], style={'marginBottom': '5px'})
                        for p in patterns[:5]  # Show top 5 patterns
                    ])
                ], style={
                    'backgroundColor': 'rgba(0,0,0,0.5)',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'marginTop': '20px'
                }) if patterns else None,
                
                # Real-time update indicator
                html.Div([
                    html.Span("🟢", style={'color': '#00ff00', 'marginRight': '5px'}),
                    html.Span(
                        f"Live Data • Updated {datetime.now().strftime('%H:%M:%S')}",
                        style={'color': '#808080', 'fontSize': '12px'}
                    )
                ], style={'textAlign': 'center', 'marginTop': '10px'})
            ])
            
        except Exception as e:
            logger.error(f"Error creating legendary compass: {e}", exc_info=True)
            return create_placeholder_card(
                "🧭 Legendary Market Compass",
                f"Error: {str(e)}"
            )