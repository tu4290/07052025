# dashboard_application/components/ai_intelligence_panel_v2_5.py
# EOTS v2.5 - AI INTELLIGENCE DASHBOARD PANEL

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

def create_unified_ai_intelligence_panel(
    symbol: str,
    final_analysis_bundle,
    config,
    timestamp: Optional[datetime] = None,
    unified_ai_orchestrator=None
) -> html.Div:
    """
    Creates the Unified AI Intelligence Panel - The "Apex Predator" brain visualization
    showing insights from ALL EOTS components synthesized by AI.
    """

    # Generate unified intelligence if orchestrator is available
    unified_intelligence = None
    if unified_ai_orchestrator:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            unified_intelligence = loop.run_until_complete(
                unified_ai_orchestrator.generate_unified_intelligence(symbol, final_analysis_bundle)
            )
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to generate unified intelligence: {e}")

    # Create the panel
    return _create_unified_intelligence_display(symbol, unified_intelligence, final_analysis_bundle, config, timestamp)

def create_ai_intelligence_panel(
    symbol: str,
    ai_enhanced_data: Dict[str, Any],
    config,
    timestamp: Optional[datetime] = None
) -> html.Div:
    """
    Creates an advanced AI intelligence panel showing insights from both
    Supabase (live data) and MCP (AI learning patterns).
    """
    
    # Extract data sources
    live_recs = ai_enhanced_data.get('live_recommendations', [])
    ai_patterns = ai_enhanced_data.get('ai_learning_patterns', [])
    cross_market = ai_enhanced_data.get('cross_market_insights', [])
    data_sources = ai_enhanced_data.get('data_sources', {})
    
    # Create status indicators
    status_indicators = _create_data_source_status(data_sources)
    
    # Create AI patterns visualization
    patterns_viz = _create_ai_patterns_chart(ai_patterns)
    
    # Create cross-market correlations
    correlations_viz = _create_correlations_chart(cross_market, symbol)
    
    # Create live recommendations summary
    live_recs_summary = _create_live_recommendations_summary(live_recs)
    
    # Create AI insights summary
    ai_insights_summary = _create_ai_insights_summary(ai_patterns, cross_market)
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-brain me-2"),
                    "🧠 AI Intelligence Hub",
                    html.Small(" (Supabase + MCP)", className="text-muted ms-2")
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Status indicators row
                dbc.Row([
                    dbc.Col([
                        html.H6("📊 Data Sources Status", className="mb-2"),
                        status_indicators
                    ], md=12, className="mb-3")
                ]),
                
                # Main content row
                dbc.Row([
                    # Left column - Live data and AI insights
                    dbc.Col([
                        # Live recommendations
                        html.H6("🚀 Live Recommendations", className="mb-2"),
                        live_recs_summary,
                        html.Hr(),
                        
                        # AI insights summary
                        html.H6("🧠 AI Intelligence Summary", className="mb-2"),
                        ai_insights_summary
                    ], md=6),
                    
                    # Right column - Visualizations
                    dbc.Col([
                        # AI patterns chart
                        html.H6("📈 AI Learning Patterns", className="mb-2"),
                        patterns_viz,
                        html.Hr(className="my-3"),
                        
                        # Cross-market correlations
                        html.H6("🔗 Cross-Market Intelligence", className="mb-2"),
                        correlations_viz
                    ], md=6)
                ])
            ])
        ], className="elite-card fade-in-up")
    ])

def _create_data_source_status(data_sources: Dict[str, bool]) -> html.Div:
    """Create status indicators for data sources."""
    
    indicators = []
    
    # Supabase status
    supabase_color = "success" if data_sources.get('supabase_available', False) else "danger"
    supabase_icon = "fas fa-check-circle" if data_sources.get('supabase_available', False) else "fas fa-times-circle"
    indicators.append(
        dbc.Badge([
            html.I(className=f"{supabase_icon} me-1"),
            "Supabase Live Data"
        ], color=supabase_color, className="me-2")
    )
    
    # MCP patterns status
    mcp_color = "success" if data_sources.get('mcp_patterns_available', False) else "warning"
    mcp_icon = "fas fa-brain" if data_sources.get('mcp_patterns_available', False) else "fas fa-exclamation-triangle"
    indicators.append(
        dbc.Badge([
            html.I(className=f"{mcp_icon} me-1"),
            "MCP AI Patterns"
        ], color=mcp_color, className="me-2")
    )
    
    # Cross-market status
    cross_color = "success" if data_sources.get('cross_market_available', False) else "secondary"
    cross_icon = "fas fa-link" if data_sources.get('cross_market_available', False) else "fas fa-unlink"
    indicators.append(
        dbc.Badge([
            html.I(className=f"{cross_icon} me-1"),
            "Cross-Market Data"
        ], color=cross_color, className="me-2")
    )
    
    return html.Div(indicators)

def _create_ai_patterns_chart(ai_patterns: List[Dict[str, Any]]) -> dcc.Graph:
    """Create visualization of AI learning patterns."""
    
    if not ai_patterns:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No AI patterns available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            ).update_layout(
                height=200,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            ),
            config={'displayModeBar': False}
        )
    
    # Prepare data for visualization
    pattern_names = [p['pattern_name'][:20] + '...' if len(p['pattern_name']) > 20 else p['pattern_name'] for p in ai_patterns[:5]]
    success_rates = [p['success_rate'] * 100 for p in ai_patterns[:5]]
    confidence_scores = [p['confidence_score'] * 100 for p in ai_patterns[:5]]
    sample_sizes = [p['sample_size'] for p in ai_patterns[:5]]
    
    # Create bubble chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=success_rates,
        y=confidence_scores,
        mode='markers+text',
        marker=dict(
            size=[min(max(s, 10), 50) for s in sample_sizes],  # Scale bubble size
            color=success_rates,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Success Rate %")
        ),
        text=pattern_names,
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>" +
                      "Success Rate: %{x:.1f}%<br>" +
                      "Confidence: %{y:.1f}%<br>" +
                      "Sample Size: %{marker.size}<br>" +
                      "<extra></extra>"
    ))
    
    fig.update_layout(
        title="AI Learning Patterns (Success vs Confidence)",
        xaxis_title="Success Rate (%)",
        yaxis_title="Confidence Score (%)",
        height=250,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def _create_correlations_chart(cross_market: List[Dict[str, Any]], symbol: str) -> dcc.Graph:
    """Create cross-market correlations visualization."""
    
    if not cross_market:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No cross-market data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            ).update_layout(
                height=200,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            ),
            config={'displayModeBar': False}
        )
    
    # Prepare correlation data
    symbols = []
    correlations = []
    colors = []
    
    for corr in cross_market[:5]:
        other_symbol = corr['correlated_symbol'] if corr['primary_symbol'] == symbol else corr['primary_symbol']
        symbols.append(other_symbol)
        corr_value = corr['correlation_coefficient']
        correlations.append(corr_value)
        colors.append('green' if corr_value > 0 else 'red')
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=correlations,
            marker_color=colors,
            text=[f"{c:.2f}" for c in correlations],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                          "Correlation: %{y:.3f}<br>" +
                          "<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=f"Cross-Market Correlations with {symbol}",
        xaxis_title="Correlated Symbols",
        yaxis_title="Correlation Coefficient",
        height=250,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=40),
        yaxis=dict(range=[-1, 1])
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def _create_live_recommendations_summary(live_recs: List[Dict[str, Any]]) -> html.Div:
    """Create summary of live recommendations."""
    
    if not live_recs:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "No live recommendations available"
            ], color="info", className="mb-0")
        ])
    
    summary_items = []
    for rec in live_recs[:3]:  # Show top 3
        conviction = rec.get('conviction_score', 0)
        strategy = rec.get('strategy_type', 'Unknown')
        
        # Color based on conviction
        if conviction > 0.8:
            color = "success"
            icon = "fas fa-star"
        elif conviction > 0.6:
            color = "warning"
            icon = "fas fa-star-half-alt"
        else:
            color = "secondary"
            icon = "far fa-star"
        
        summary_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.I(className=f"{icon} me-2"),
                    html.Strong(strategy),
                    dbc.Badge(f"{conviction:.1%}", color=color, className="ms-2")
                ])
            ])
        )
    
    return html.Div([
        dbc.ListGroup(summary_items, flush=True)
    ])

def _create_ai_insights_summary(ai_patterns: List[Dict[str, Any]], cross_market: List[Dict[str, Any]]) -> html.Div:
    """Create AI insights summary."""
    
    insights = []
    
    # AI patterns insight
    if ai_patterns:
        best_pattern = max(ai_patterns, key=lambda x: x['success_rate'] * x['confidence_score'])
        insights.append(
            dbc.Alert([
                html.I(className="fas fa-brain me-2"),
                html.Strong("Best AI Pattern: "),
                f"{best_pattern['pattern_name']} ({best_pattern['success_rate']:.1%} success)"
            ], color="info", className="mb-2")
        )
    
    # Cross-market insight
    if cross_market:
        strongest_corr = max(cross_market, key=lambda x: abs(x['correlation_coefficient']))
        corr_type = "positive" if strongest_corr['correlation_coefficient'] > 0 else "negative"
        insights.append(
            dbc.Alert([
                html.I(className="fas fa-link me-2"),
                html.Strong("Strongest Correlation: "),
                f"{strongest_corr['correlated_symbol']} ({corr_type}, {strongest_corr['correlation_coefficient']:.2f})"
            ], color="secondary", className="mb-2")
        )
    
    if not insights:
        insights.append(
            dbc.Alert([
                html.I(className="fas fa-search me-2"),
                "Building AI intelligence database..."
            ], color="light", className="mb-0")
        )
    
    return html.Div(insights)

def _create_unified_intelligence_display(
    symbol: str,
    unified_intelligence,
    final_analysis_bundle,
    config,
    timestamp: Optional[datetime] = None
) -> html.Div:
    """Create the unified AI intelligence display - The Apex Predator brain."""

    if not unified_intelligence:
        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-brain me-2"),
                        "🧠 Unified AI Intelligence",
                        html.Small(" (Apex Predator)", className="text-muted ms-2")
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Unified AI Intelligence is initializing or unavailable. ",
                        "The system is analyzing all EOTS components to provide comprehensive insights."
                    ], color="warning")
                ])
            ], className="elite-card fade-in-up")
        ])

    # Create AI intelligence visualization
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-brain me-2"),
                    "🧠 Unified AI Intelligence",
                    html.Small(" (Apex Predator)", className="text-muted ms-2")
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Overall assessment
                dbc.Row([
                    dbc.Col([
                        html.H5("🎯 AI Market Assessment", className="mb-2"),
                        dbc.Alert([
                            html.H6(unified_intelligence.overall_market_assessment, className="mb-2"),
                            html.P(unified_intelligence.reasoning[:200] + "...", className="mb-0")
                        ], color="primary" if unified_intelligence.ai_conviction_score > 0.7 else "secondary")
                    ], md=12, className="mb-3")
                ]),

                # Key metrics row
                dbc.Row([
                    dbc.Col([
                        html.H6("📊 AI Conviction", className="mb-1"),
                        html.H4(f"{unified_intelligence.ai_conviction_score:.1%}",
                                className="text-success" if unified_intelligence.ai_conviction_score > 0.7 else "text-warning")
                    ], md=3),
                    dbc.Col([
                        html.H6("🎯 Regime Confidence", className="mb-1"),
                        html.H4(f"{unified_intelligence.regime_confidence:.1%}",
                                className="text-info")
                    ], md=3),
                    dbc.Col([
                        html.H6("🔍 Focus Area", className="mb-1"),
                        html.H6(unified_intelligence.recommended_focus, className="text-primary")
                    ], md=6)
                ], className="mb-3"),

                # Opportunities and risks
                dbc.Row([
                    dbc.Col([
                        html.H6("🚀 Key Opportunities", className="mb-2"),
                        html.Ul([
                            html.Li(opp) for opp in unified_intelligence.key_opportunities[:3]
                        ], className="mb-0")
                    ], md=6),
                    dbc.Col([
                        html.H6("⚠️ Risk Warnings", className="mb-2"),
                        html.Ul([
                            html.Li(risk) for risk in unified_intelligence.risk_warnings[:3]
                        ], className="mb-0")
                    ], md=6)
                ], className="mb-3"),

                # Component synergies
                html.H6("🔗 Component Synergies", className="mb-2"),
                html.Div([
                    dbc.Badge(synergy, color="info", className="me-2 mb-1")
                    for synergy in unified_intelligence.component_synergies[:5]
                ]),

                # Analysis summaries
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("📈 Flow Analysis", className="mb-1"),
                        html.P(unified_intelligence.flow_analysis_summary[:100] + "...",
                               className="small text-muted")
                    ], md=6),
                    dbc.Col([
                        html.H6("🏗️ Structural Analysis", className="mb-1"),
                        html.P(unified_intelligence.structural_analysis_summary[:100] + "...",
                               className="small text-muted")
                    ], md=6)
                ])
            ])
        ], className="elite-card fade-in-up")
    ])
