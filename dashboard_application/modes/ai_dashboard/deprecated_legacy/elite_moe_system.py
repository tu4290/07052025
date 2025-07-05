"""
ELITE MOE SYSTEM - Complete Professional Implementation
======================================================

All-in-one file with stunning aesthetics and professional-grade compass.
This replaces all the scattered files with one elite system.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

class EliteMOE:
    """
    Elite MOE System with stunning professional aesthetics.
    Combines hybrid AI intelligence with beautiful visualizations.
    """
    
    def __init__(self):
        # Elite color schemes
        self.elite_colors = {
            'RED': '#FF3B30',           # Volatility explosion
            'GREEN': '#30D158',         # Trend continuation  
            'BLUE': '#007AFF',          # Premium selling
            'PURPLE': '#AF52DE',        # EOD hedging
            'ORANGE': '#FF9500',        # Danger/caution
            'CYAN': '#5AC8FA',          # Flow convergence
            'YELLOW': '#FFCC02',        # Volatility squeeze
            'ELECTRIC_BLUE': '#00D4FF', # Gamma storm
            'BRIGHT_GREEN': '#32D74B',  # Momentum explosion
            'ORANGE_RED': '#FF6B35',    # Smart money divergence
            'GRAY': '#8E8E93'           # Neutral
        }
        
        # Professional gradients
        self.gradients = {
            'RED': ['#FF3B30', '#FF6B6B', '#FF8E8E'],
            'GREEN': ['#30D158', '#5DE283', '#8AE8A8'],
            'BLUE': ['#007AFF', '#4DA3FF', '#80C7FF'],
            'PURPLE': ['#AF52DE', '#C478E8', '#D99EF2'],
            'ELECTRIC_BLUE': ['#00D4FF', '#33E0FF', '#66ECFF']
        }
        
        # Foundation rules (your trading expertise)
        self.foundation_rules = {
            'VOLATILE_DOWN_IGNITION': {
                'conditions': {
                    'VAPI-FA': {'min': 0.5, 'max': 1.2, 'weight': 0.3},
                    'DWFD': {'min': 1.5, 'max': 2.5, 'weight': 0.4},
                    'VRI 2.0': {'min': -3.0, 'max': -2.0, 'weight': 0.3}
                },
                'output': {
                    'signal': 'Ignition Point: Volatile Down',
                    'action': 'Prepare for volatility expansion',
                    'color': 'RED',
                    'shape_focus': 'VRI 2.0',
                    'bulge_intensity': 1.8,
                    'confidence': 0.92
                }
            },
            'MOMENTUM_EXPLOSION': {
                'conditions': {
                    'MSPI': {'min': 1.8, 'max': 3.5, 'weight': 0.4},
                    'A-DAG': {'min': 1.5, 'max': 3.0, 'weight': 0.3},
                    'AOFM': {'min': 1.2, 'max': 2.8, 'weight': 0.3}
                },
                'output': {
                    'signal': 'Momentum Explosion Imminent',
                    'action': 'Aggressive trend following',
                    'color': 'BRIGHT_GREEN',
                    'shape_focus': 'MSPI',
                    'bulge_intensity': 2.0,
                    'confidence': 0.88
                }
            },
            'EOD_GAMMA_HEDGING': {
                'conditions': {
                    'GIB': {'min': 2.5, 'max': 4.5, 'weight': 0.5},
                    'TDPI': {'min': 1.5, 'max': 3.2, 'weight': 0.3},
                    'time_factor': {'min': 14, 'max': 16, 'weight': 0.2}
                },
                'output': {
                    'signal': 'EOD Hedging Opportunity',
                    'action': 'Position for end-of-day flows',
                    'color': 'PURPLE',
                    'shape_focus': 'GIB',
                    'bulge_intensity': 1.9,
                    'confidence': 0.85
                }
            },
            'PREMIUM_CRUSH_SETUP': {
                'conditions': {
                    'VRI 2.0': {'min': -1.0, 'max': 0.8, 'weight': 0.4},
                    'VAPI-FA': {'min': -0.6, 'max': 0.6, 'weight': 0.3},
                    'GIB': {'min': -1.2, 'max': 1.2, 'weight': 0.3}
                },
                'output': {
                    'signal': 'Premium Selling Environment',
                    'action': 'Sell options premium',
                    'color': 'BLUE',
                    'shape_focus': 'VRI 2.0',
                    'bulge_intensity': 1.4,
                    'confidence': 0.78
                }
            },
            'GAMMA_STORM_WARNING': {
                'conditions': {
                    'GIB': {'min': 3.2, 'max': 5.0, 'weight': 0.4},
                    'A-DAG': {'min': 2.8, 'max': 4.5, 'weight': 0.3},
                    'VRI 2.0': {'min': 2.5, 'max': 4.0, 'weight': 0.3}
                },
                'output': {
                    'signal': 'Gamma Storm Detected',
                    'action': 'Extreme caution - reduce size',
                    'color': 'ELECTRIC_BLUE',
                    'shape_focus': 'GIB',
                    'bulge_intensity': 2.2,
                    'confidence': 0.95
                }
            }
        }
        
        # Learning layer
        self.discovered_patterns = {}
        self.pattern_performance = {}
    
    def analyze_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Main analysis function that returns everything needed for visualization.
        """
        # Check foundation rules first
        foundation_result = self._check_foundation_rules(metrics)
        
        if foundation_result:
            return self._format_analysis_result(foundation_result, metrics, 'FOUNDATION')
        
        # Check learning patterns
        learning_result = self._check_learning_patterns(metrics)
        
        if learning_result:
            return self._format_analysis_result(learning_result, metrics, 'LEARNING')
        
        # Default neutral state
        neutral_result = {
            'output': {
                'signal': 'Market Monitoring',
                'action': 'Continue analysis',
                'color': 'GRAY',
                'shape_focus': 'VAPI-FA',
                'bulge_intensity': 1.0,
                'confidence': 0.4
            }
        }
        
        return self._format_analysis_result(neutral_result, metrics, 'NEUTRAL')
    
    def _check_foundation_rules(self, metrics: Dict[str, float]):
        """Check foundation rules with sophisticated matching."""
        current_hour = datetime.now().hour
        
        for rule_name, rule in self.foundation_rules.items():
            score = 0
            total_weight = 0
            conditions_met = 0
            
            for condition_name, condition in rule['conditions'].items():
                weight = condition['weight']
                total_weight += weight
                
                if condition_name == 'time_factor':
                    if condition['min'] <= current_hour <= condition['max']:
                        score += weight
                        conditions_met += 1
                elif condition_name in metrics:
                    value = metrics[condition_name]
                    min_val, max_val = condition['min'], condition['max']
                    
                    if min_val <= value <= max_val:
                        score += weight
                        conditions_met += 1
                    elif min_val - 0.8 <= value <= max_val + 0.8:
                        score += weight * 0.6
                        conditions_met += 0.6
            
            # Require strong confluence
            if total_weight > 0:
                confidence_score = score / total_weight
                if confidence_score > 0.65 and conditions_met >= 1.5:
                    adjusted_rule = rule.copy()
                    adjusted_rule['output']['confidence'] *= confidence_score
                    return adjusted_rule
        
        return None
    
    def _check_learning_patterns(self, metrics: Dict[str, float]):
        """Advanced learning pattern detection."""
        values = list(metrics.values())
        
        # Statistical analysis
        mean_abs = np.mean([abs(v) for v in values])
        std_dev = np.std(values)
        max_val = max([abs(v) for v in values])
        
        # High volatility cluster
        if mean_abs > 2.2 and std_dev > 1.2:
            focus_metric = max(metrics.items(), key=lambda x: abs(x[1]))[0]
            return {
                'output': {
                    'signal': f'High Volatility Cluster (AI)',
                    'action': 'Monitor for breakout',
                    'color': 'YELLOW',
                    'shape_focus': focus_metric,
                    'bulge_intensity': min(mean_abs / 1.5, 2.5),
                    'confidence': min(mean_abs / 3.5, 0.82)
                }
            }
        
        # Low activity environment
        elif mean_abs < 0.9 and std_dev < 0.6:
            return {
                'output': {
                    'signal': 'Low Activity Environment (AI)',
                    'action': 'Consider premium strategies',
                    'color': 'CYAN',
                    'shape_focus': 'VRI 2.0',
                    'bulge_intensity': 1.2,
                    'confidence': min(1.2 - mean_abs, 0.75)
                }
            }
        
        return None
    
    def _format_analysis_result(self, rule_result, metrics, source_type):
        """Format analysis result for compass visualization."""
        output = rule_result['output']
        
        # Calculate shape deformation
        focus_metric = output['shape_focus']
        bulge_intensity = output['bulge_intensity']
        
        deformation_factors = {}
        for metric in metrics.keys():
            if metric == focus_metric:
                deformation_factors[metric] = bulge_intensity
            else:
                # Compress others proportionally
                deformation_factors[metric] = max(0.4, 1.0 / (bulge_intensity * 0.8))
        
        return {
            'signal': output['signal'],
            'action': output['action'],
            'color': output['color'],
            'confidence': output['confidence'],
            'shape_focus': focus_metric,
            'bulge_intensity': bulge_intensity,
            'deformation_factors': deformation_factors,
            'source_type': source_type,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
    
    def create_elite_compass(self, analysis_result: Dict[str, Any]) -> go.Figure:
        """
        Create a stunning, professional-grade compass visualization.
        This is where the magic happens - elite aesthetics!
        """
        metrics = analysis_result['metrics']
        color = analysis_result['color']
        deformation_factors = analysis_result['deformation_factors']
        signal = analysis_result['signal']
        confidence = analysis_result['confidence']
        
        # Create figure with dark professional theme
        fig = go.Figure()
        
        # Elite color scheme
        primary_color = self.elite_colors[color]
        gradient_colors = self.gradients.get(color, [primary_color, primary_color, primary_color])
        
        # Prepare data for hexagon
        metrics_names = list(metrics.keys())[:6]  # Limit to 6 for hexagon
        values = []
        
        for name in metrics_names:
            base_value = abs(metrics[name])
            deformed_value = base_value * deformation_factors.get(name, 1.0)
            values.append(deformed_value)
        
        # Close the hexagon
        theta_labels = metrics_names + [metrics_names[0]]
        r_values = values + [values[0]]
        
        # Create multiple layers for depth effect
        
        # Layer 1: Outer glow effect
        fig.add_trace(go.Scatterpolar(
            r=[v * 1.15 for v in r_values],
            theta=theta_labels,
            fill='toself',
            fillcolor=f'rgba({int(primary_color[1:3], 16)}, {int(primary_color[3:5], 16)}, {int(primary_color[5:7], 16)}, 0.15)',
            line=dict(color='rgba(255, 255, 255, 0.1)', width=1),
            name='Glow',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Layer 2: Main compass shape with gradient effect
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_labels,
            fill='toself',
            fillcolor=f'rgba({int(primary_color[1:3], 16)}, {int(primary_color[3:5], 16)}, {int(primary_color[5:7], 16)}, 0.7)',
            line=dict(color=primary_color, width=3),
            name=f'{signal}',
            hovertemplate='<b>%{theta}</b><br>' +
                         'Value: %{r:.2f}<br>' +
                         f'Signal: {signal}<br>' +
                         f'Confidence: {confidence:.1%}<br>' +
                         '<extra></extra>'
        ))
        
        # Layer 3: Inner core for depth
        fig.add_trace(go.Scatterpolar(
            r=[v * 0.3 for v in r_values],
            theta=theta_labels,
            fill='toself',
            fillcolor=f'rgba({int(primary_color[1:3], 16)}, {int(primary_color[3:5], 16)}, {int(primary_color[5:7], 16)}, 0.9)',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
            name='Core',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add focus indicator (star on the emphasized metric)
        focus_metric = analysis_result['shape_focus']
        if focus_metric in metrics_names:
            focus_index = metrics_names.index(focus_metric)
            focus_value = r_values[focus_index]
            
            fig.add_trace(go.Scatterpolar(
                r=[focus_value * 1.1],
                theta=[focus_metric],
                mode='markers',
                marker=dict(
                    size=25,
                    color='white',
                    symbol='star',
                    line=dict(width=3, color=primary_color)
                ),
                name=f'Focus: {focus_metric}',
                showlegend=False,
                hovertemplate=f'<b>🎯 FOCUS POINT</b><br>' +
                             f'Metric: {focus_metric}<br>' +
                             f'Emphasis: {deformation_factors.get(focus_metric, 1.0):.1f}x<br>' +
                             '<extra></extra>'
            ))
        
        # Professional layout with elite styling
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0, 0, 0, 0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, max(r_values) * 1.3],
                    tickfont=dict(size=11, color='rgba(255, 255, 255, 0.8)', family='SF Pro Display'),
                    gridcolor='rgba(255, 255, 255, 0.15)',
                    linecolor='rgba(255, 255, 255, 0.2)',
                    tickcolor='rgba(255, 255, 255, 0.3)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color='white', family='SF Pro Display', weight='bold'),
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)',
                    rotation=90,
                    direction='clockwise'
                )
            ),
            title=dict(
                text=f'<b>🧭 ELITE MOE COMPASS</b><br>' +
                     f'<span style="font-size:16px; color:{primary_color};">{signal}</span><br>' +
                     f'<span style="font-size:14px; color:rgba(255,255,255,0.8);">Confidence: {confidence:.1%} | Source: {analysis_result["source_type"]}</span>',
                x=0.5,
                y=0.95,
                font=dict(size=20, color='white', family='SF Pro Display'),
                xanchor='center'
            ),
            paper_bgcolor='rgba(15, 23, 42, 0.95)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white', family='SF Pro Display'),
            height=700,
            width=700,
            margin=dict(l=50, r=50, t=120, b=50),
            
            # Add subtle animations
            transition=dict(
                duration=800,
                easing='cubic-in-out'
            )
        )
        
        # Add professional annotations
        fig.add_annotation(
            text=f'<b>{analysis_result["action"]}</b>',
            x=0.5,
            y=0.02,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=14, color=primary_color, family='SF Pro Display'),
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor=primary_color,
            borderwidth=1,
            borderpad=8
        )
        
        return fig

def demonstrate_elite_system():
    """Demonstrate the complete elite MOE system."""
    
    # Test scenarios
    scenarios = {
        "Volatile Down Ignition": {
            'VAPI-FA': 0.88,
            'DWFD': 1.96,
            'VRI 2.0': -2.53,
            'GIB': 1.2,
            'MSPI': 0.5,
            'A-DAG': 0.3
        },
        "Momentum Explosion": {
            'MSPI': 2.3,
            'A-DAG': 1.9,
            'AOFM': 1.7,
            'VAPI-FA': 1.8,
            'GIB': 1.4,
            'VRI 2.0': 1.2
        },
        "Gamma Storm": {
            'GIB': 3.8,
            'A-DAG': 3.2,
            'VRI 2.0': 2.9,
            'VAPI-FA': 1.5,
            'MSPI': 1.8,
            'AOFM': 2.1
        }
    }
    
    elite_moe = EliteMOE()
    
    print("🏆 ELITE MOE SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    for i, (scenario_name, metrics) in enumerate(scenarios.items()):
        print(f"\n📊 SCENARIO {i+1}: {scenario_name}")
        print("-" * 40)
        
        # Analyze metrics
        analysis = elite_moe.analyze_metrics(metrics)
        
        print(f"🎯 SIGNAL: {analysis['signal']}")
        print(f"🎬 ACTION: {analysis['action']}")
        print(f"🎨 COLOR: {analysis['color']}")
        print(f"🎪 CONFIDENCE: {analysis['confidence']:.1%}")
        print(f"🔍 SOURCE: {analysis['source_type']}")
        print(f"⭐ FOCUS: {analysis['shape_focus']}")
        print(f"📈 BULGE: {analysis['bulge_intensity']:.1f}x")
        
        # Create elite compass
        fig = elite_moe.create_elite_compass(analysis)
        
        # Save files
        filename = f"elite_compass_{scenario_name.lower().replace(' ', '_')}"
        fig.write_html(f"/home/ubuntu/{filename}.html")
        fig.write_image(f"/home/ubuntu/{filename}.png", width=700, height=700, scale=2)
        
        print(f"💾 SAVED: {filename}.html and {filename}.png")
    
    print(f"\n🏆 ELITE MOE SYSTEM COMPLETE!")
    print("All compass visualizations saved with professional aesthetics!")

if __name__ == "__main__":
    demonstrate_elite_system()

