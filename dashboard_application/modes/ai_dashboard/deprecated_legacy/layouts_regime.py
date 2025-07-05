"""
AI Hub Regime Module - Persistent Market Regime MOE v2.5
========================================================

This module contains the persistent Market Regime MOE that operates system-wide:
- PersistentMarketRegimeMOE class for continuous monitoring
- Regime display components for control panel integration
- Cross-system regime intelligence sharing

Author: EOTS v2.5 Development Team
Version: 2.5.1 (Modular)
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from dash import dcc, html
import plotly.graph_objects as go

# EOTS Schema imports - Pydantic-first validation
from data_models import FinalAnalysisBundleV2_5
from data_models import ProcessedDataBundleV2_5, ProcessedUnderlyingAggregatesV2_5

# Import existing components - preserve dependencies
from .components import (
    AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS,
    get_card_style
)

import compliance_decorators_v2_5

logger = logging.getLogger(__name__)

class PersistentMarketRegimeMOE:
    """
    Persistent Market Regime MOE that operates system-wide.
    
    This MOE continuously monitors market conditions and provides
    regime intelligence across all system modes and components.
    """
    
    def __init__(self):
        self.current_regime = "ANALYZING"
        self.confidence = 0.0
        self.transition_risk = 0.0
        self.last_update = datetime.now()
        self.regime_history = []
        self.is_monitoring = False
        
        # Regime classification thresholds
        self.regime_thresholds = {
            'bull_trending': {'vapi_fa': 1.5, 'dwfd': 1.0, 'vri_2_0': 5000},
            'bear_trending': {'vapi_fa': -1.5, 'dwfd': -1.0, 'vri_2_0': 5000},
            'high_volatility': {'vri_2_0': 15000},
            'sideways': {'vapi_fa': 0.5, 'dwfd': 0.5, 'vri_2_0': 3000}
        }
    
    def update_with_bundle_data(self, bundle_data: FinalAnalysisBundleV2_5) -> None:
        """Update regime analysis with new bundle data."""
        try:
            if not isinstance(bundle_data, FinalAnalysisBundleV2_5):
                logger.warning("Invalid bundle data for regime update")
                return
            
            # Extract data using Pydantic model access
            processed_data = bundle_data.processed_data_bundle
            if not processed_data or not processed_data.underlying_data_enriched:
                logger.warning("No enriched data available for regime analysis")
                return
            
            enriched_data = processed_data.underlying_data_enriched
            
            # Extract key metrics for regime analysis
            vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
            dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
            tw_laf = getattr(enriched_data, 'tw_laf_z_score_und', 0.0) or 0.0
            vri_2_0 = getattr(enriched_data, 'vri_2_0_und', 0.0) or 0.0
            
            # Analyze current regime
            regime_analysis = self._analyze_regime(vapi_fa, dwfd, tw_laf, vri_2_0)
            
            # Update regime state
            self.current_regime = regime_analysis['regime']
            self.confidence = regime_analysis['confidence']
            self.transition_risk = regime_analysis['transition_risk']
            self.last_update = datetime.now()
            
            # Add to history
            self.regime_history.append({
                'timestamp': self.last_update,
                'regime': self.current_regime,
                'confidence': self.confidence,
                'metrics': {'vapi_fa': vapi_fa, 'dwfd': dwfd, 'tw_laf': tw_laf, 'vri_2_0': vri_2_0}
            })
            
            # Keep only last 100 entries
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"Regime updated: {self.current_regime} (confidence: {self.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error updating regime with bundle data: {str(e)}")
    
    def _analyze_regime(self, vapi_fa: float, dwfd: float, tw_laf: float, vri_2_0: float) -> Dict[str, Any]:
        """Analyze current market regime based on metrics."""
        try:
            # High volatility check (overrides other regimes)
            if abs(vri_2_0) > self.regime_thresholds['high_volatility']['vri_2_0']:
                return {
                    'regime': 'HIGH_VOLATILITY',
                    'confidence': min(abs(vri_2_0) / 20000, 1.0),
                    'transition_risk': 0.8
                }
            
            # Bull trending check
            if (vapi_fa > self.regime_thresholds['bull_trending']['vapi_fa'] and 
                dwfd > self.regime_thresholds['bull_trending']['dwfd']):
                confidence = min((vapi_fa + dwfd) / 4.0, 1.0)
                return {
                    'regime': 'BULL_TRENDING',
                    'confidence': confidence,
                    'transition_risk': max(0.1, 1.0 - confidence)
                }
            
            # Bear trending check
            if (vapi_fa < self.regime_thresholds['bear_trending']['vapi_fa'] and 
                dwfd < self.regime_thresholds['bear_trending']['dwfd']):
                confidence = min(abs(vapi_fa + dwfd) / 4.0, 1.0)
                return {
                    'regime': 'BEAR_TRENDING',
                    'confidence': confidence,
                    'transition_risk': max(0.1, 1.0 - confidence)
                }
            
            # Sideways/consolidating (default)
            signal_strength = abs(vapi_fa) + abs(dwfd) + abs(tw_laf)
            if signal_strength < 1.5:
                return {
                    'regime': 'SIDEWAYS_CONSOLIDATING',
                    'confidence': max(0.3, 1.0 - signal_strength / 3.0),
                    'transition_risk': min(0.7, signal_strength / 2.0)
                }
            
            # Transitioning state
            return {
                'regime': 'TRANSITIONING',
                'confidence': 0.4,
                'transition_risk': 0.9
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime: {str(e)}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'transition_risk': 1.0
            }
    
    def get_regime_display_data(self) -> Dict[str, Any]:
        """Get data for regime display components."""
        try:
            # Determine display properties
            regime_colors = {
                'BULL_TRENDING': AI_COLORS['success'],
                'BEAR_TRENDING': AI_COLORS['danger'],
                'HIGH_VOLATILITY': AI_COLORS['warning'],
                'SIDEWAYS_CONSOLIDATING': AI_COLORS['info'],
                'TRANSITIONING': AI_COLORS['warning'],
                'UNKNOWN': AI_COLORS['muted']
            }
            
            regime_icons = {
                'BULL_TRENDING': '📈',
                'BEAR_TRENDING': '📉',
                'HIGH_VOLATILITY': '⚡',
                'SIDEWAYS_CONSOLIDATING': '📊',
                'TRANSITIONING': '🔄',
                'UNKNOWN': '❓'
            }
            
            # Calculate pulse rate based on transition risk
            pulse_rate = max(0.5, min(2.0, self.transition_risk * 2))
            
            return {
                'regime': self.current_regime,
                'confidence': self.confidence,
                'transition_risk': self.transition_risk,
                'color': regime_colors.get(self.current_regime, AI_COLORS['muted']),
                'icon': regime_icons.get(self.current_regime, '❓'),
                'pulse_rate': pulse_rate,
                'last_update': self.last_update,
                'display_name': self._get_display_name()
            }
            
        except Exception as e:
            logger.error(f"Error getting regime display data: {str(e)}")
            return {
                'regime': 'ERROR',
                'confidence': 0.0,
                'transition_risk': 1.0,
                'color': AI_COLORS['danger'],
                'icon': '⚠️',
                'pulse_rate': 1.0,
                'last_update': datetime.now(),
                'display_name': 'Error'
            }
    
    def _get_display_name(self) -> str:
        """Get human-readable display name for current regime."""
        display_names = {
            'BULL_TRENDING': 'Bull Trending',
            'BEAR_TRENDING': 'Bear Trending',
            'HIGH_VOLATILITY': 'High Volatility',
            'SIDEWAYS_CONSOLIDATING': 'Sideways',
            'TRANSITIONING': 'Transitioning',
            'UNKNOWN': 'Analyzing...'
        }
        return display_names.get(self.current_regime, 'Unknown')
    
    async def start_continuous_monitoring(self):
        """Start continuous regime monitoring (for future implementation)."""
        self.is_monitoring = True
        logger.info("Persistent Market Regime MOE monitoring started")
        
        # This would contain the actual continuous monitoring loop
        # For now, it's a placeholder for future implementation
        while self.is_monitoring:
            await asyncio.sleep(15)  # Update every 15 seconds
            # In real implementation, this would fetch new data and update regime
    
    def stop_monitoring(self):
        """Stop continuous regime monitoring."""
        self.is_monitoring = False
        logger.info("Persistent Market Regime MOE monitoring stopped")

@compliance_decorators_v2_5.track_compliance("persistent_regime_display", "Persistent Regime Display")
def create_persistent_regime_display(regime_data: Dict[str, Any], symbol: str) -> html.Div:
    """
    Create persistent regime display for control panel.
    
    Args:
        regime_data: Regime display data from MOE
        symbol: Trading symbol
        
    Returns:
        html.Div: Persistent regime display component
    """
    try:
        return html.Div([
            # Regime icon and name
            html.Div([
                html.Span(regime_data['icon'], className="regime-icon mr-2"),
                html.Span("REGIME:", className="regime-label mr-2"),
                html.Strong(regime_data['display_name'], className="regime-name", style={
                    "color": regime_data['color']
                }),
                html.Span(
                    f"{regime_data['confidence']:.0%}" if regime_data.get('confidence') is not None else "N/A",
                    className="regime-confidence ml-2",
                    style={"color": regime_data['color']}
                ),
                html.Span(
                    f"{regime_data['transition_risk']:.0%}" if regime_data.get('transition_risk') is not None else "N/A",
                    className="regime-transition-risk ml-2",
                    style={"color": regime_data['color']}
                ),
            ], className="regime-main d-flex align-items-center"),
            
            # Pulse indicator
            html.Div(className="regime-pulse", style={
                "width": "8px",
                "height": "8px",
                "borderRadius": "50%",
                "backgroundColor": regime_data['color'],
                "animation": f"pulse {regime_data['pulse_rate']}s infinite"
            })
            
        ], className="persistent-regime-display d-flex align-items-center", style={
            "background": f"rgba({regime_data['color']}, 0.1)",
            "border": f"1px solid {regime_data['color']}",
            "borderRadius": AI_EFFECTS['border_radius_sm'],
            "padding": f"{AI_SPACING['sm']} {AI_SPACING['md']}",
            "backdropFilter": AI_EFFECTS['backdrop_blur']
        })
        
    except Exception as e:
        logger.error(f"Error creating persistent regime display: {str(e)}")
        return html.Div("Regime display unavailable")

def create_regime_transition_visualization(regime_moe: PersistentMarketRegimeMOE) -> go.Figure:
    """Create regime transition probability visualization."""
    try:
        # Get recent regime history
        history = regime_moe.regime_history[-20:] if regime_moe.regime_history else []
        
        if not history:
            return go.Figure()
        
        timestamps = [entry['timestamp'] for entry in history]
        transition_risks = [entry.get('transition_risk', 0) for entry in history]
        
        fig = go.Figure()
        
        # Add transition risk line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=transition_risks,
            mode='lines+markers',
            name='Transition Risk',
            line=dict(color=AI_COLORS['warning'], width=2),
            marker=dict(size=4)
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color=AI_COLORS['danger'], 
                     annotation_text="High Risk")
        fig.add_hline(y=0.4, line_dash="dash", line_color=AI_COLORS['warning'], 
                     annotation_text="Moderate Risk")
        
        fig.update_layout(
            title="Regime Transition Risk",
            xaxis_title="Time",
            yaxis_title="Transition Probability",
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white'),
            height=200,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.1)')
        fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.1)', range=[0, 1])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating regime transition visualization: {str(e)}")
        return go.Figure()

