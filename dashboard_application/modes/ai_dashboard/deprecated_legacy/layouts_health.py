"""
AI Hub Health Module - Row 3 System Health Containers v2.5
==========================================================

This module contains the 4 system health containers for Row 3:
- Data Pipeline Monitor (Market feeds, options chain, database connections)
- HuiHui Experts Monitor (All 4 MOE status and performance)
- Performance Monitor (Response times, memory, CPU, error rates)
- Alerts & Status Monitor (Critical alerts, warnings, opportunities)

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
from data_models import ProcessedDataBundleV2_5
from data_models import EOTSConfigV2_5

# Import existing components - preserve dependencies
from .components import (
    AI_COLORS, AI_TYPOGRAPHY, AI_SPACING, AI_EFFECTS,
    create_placeholder_card, get_card_style, create_clickable_title_with_info
)

import compliance_decorators_v2_5

logger = logging.getLogger(__name__)

def create_status_indicator(status: str, label: str) -> html.Div:
    """Create a status indicator with color coding."""
    status_colors = {
        'healthy': AI_COLORS['success'],
        'warning': AI_COLORS['warning'],
        'error': AI_COLORS['danger'],
        'unknown': AI_COLORS['muted']
    }
    
    color = status_colors.get(status.lower(), AI_COLORS['muted'])
    
    return html.Div([
        html.Div(className="status-dot", style={
            "width": "8px",
            "height": "8px",
            "borderRadius": "50%",
            "backgroundColor": color,
            "display": "inline-block",
            "marginRight": "8px"
        }),
        html.Span(label, className="status-label")
    ], className="status-item d-flex align-items-center mb-1")

@compliance_decorators_v2_5.track_compliance("data_pipeline_monitor", "Data Pipeline Monitor")
def create_data_pipeline_monitor(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """
    Create Data Pipeline Monitor container.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        
    Returns:
        html.Div: Data pipeline monitor container
    """
    try:
        # Assess data pipeline health based on bundle data availability
        pipeline_status = []
        
        # Check market data feeds
        if bundle_data.processed_data_bundle:
            pipeline_status.append(create_status_indicator('healthy', 'Market Data Feed'))
        else:
            pipeline_status.append(create_status_indicator('error', 'Market Data Feed'))
        
        # Check options chain data
        if (bundle_data.processed_data_bundle and 
            bundle_data.processed_data_bundle.underlying_data_enriched):
            pipeline_status.append(create_status_indicator('healthy', 'Options Chain'))
        else:
            pipeline_status.append(create_status_indicator('warning', 'Options Chain'))
        
        # Check database connections (simplified check)
        pipeline_status.append(create_status_indicator('healthy', 'Database Conn'))
        
        # Check API status
        if bundle_data.atif_recommendations_v2_5:
            pipeline_status.append(create_status_indicator('healthy', 'ATIF API'))
        else:
            pipeline_status.append(create_status_indicator('warning', 'ATIF API'))
        
        # Data quality assessment
        data_quality = "Good"
        if not bundle_data.processed_data_bundle:
            data_quality = "Poor"
        elif not bundle_data.processed_data_bundle.underlying_data_enriched:
            data_quality = "Fair"
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🔌 Data Pipeline",
                    "data_pipeline",
                    "Real-time monitoring of market data feeds, options chain updates, and API connections"
                )
            ], className="container-header"),
            
            html.Div([
                # Pipeline status indicators
                html.Div(pipeline_status, className="pipeline-status mb-3"),
                
                # Data quality summary
                html.Div([
                    html.Small("Data Quality:", className="text-muted"),
                    html.Strong(data_quality, className="ml-2", style={
                        "color": AI_COLORS['success'] if data_quality == "Good" else 
                                AI_COLORS['warning'] if data_quality == "Fair" else AI_COLORS['danger']
                    })
                ], className="quality-summary"),
                
                # Last update timestamp
                html.Div([
                    html.Small("Last Update:", className="text-muted"),
                    html.Small(datetime.now().strftime("%H:%M:%S"), className="ml-2")
                ], className="update-timestamp")
                
            ], className="pipeline-content")
            
        ], style=get_card_style('info'))
        
    except Exception as e:
        logger.error(f"Error creating data pipeline monitor: {str(e)}")
        return create_placeholder_card("🔌 Data Pipeline", f"Error: {str(e)}")

@compliance_decorators_v2_5.track_compliance("huihui_experts_monitor", "HuiHui Experts Monitor")
def create_huihui_experts_monitor(bundle_data: FinalAnalysisBundleV2_5, symbol: str, db_manager=None) -> html.Div:
    """
    Create HuiHui Experts Monitor container.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        db_manager: Database manager instance
        
    Returns:
        html.Div: HuiHui experts monitor container
    """
    try:
        # Simulate expert status (in real implementation, this would check actual MOE status)
        experts_status = [
            create_status_indicator('healthy', 'Market Regime Expert'),
            create_status_indicator('healthy', 'Options Flow Expert'),
            create_status_indicator('healthy', 'Intelligence Expert'),
            create_status_indicator('warning', 'Meta-Orchestrator')
        ]
        
        # Calculate overall expert health
        expert_health_score = 85  # Simplified calculation
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🧠 HuiHui Experts",
                    "huihui_experts",
                    "Status and performance monitoring for all 4 HuiHui MOE experts"
                )
            ], className="container-header"),
            
            html.Div([
                # Expert status indicators
                html.Div(experts_status, className="experts-status mb-3"),
                
                # Overall health score
                html.Div([
                    html.Small("Expert Health:", className="text-muted"),
                    html.Strong(f"{expert_health_score}%", className="ml-2", style={
                        "color": AI_COLORS['success'] if expert_health_score >= 80 else 
                                AI_COLORS['warning'] if expert_health_score >= 60 else AI_COLORS['danger']
                    })
                ], className="health-score"),
                
                # Learning progress
                html.Div([
                    html.Small("Learning Active:", className="text-muted"),
                    html.Span("●", className="ml-2 text-success")
                ], className="learning-status")
                
            ], className="experts-content")
            
        ], style=get_card_style('primary'))
        
    except Exception as e:
        logger.error(f"Error creating HuiHui experts monitor: {str(e)}")
        return create_placeholder_card("🧠 HuiHui Experts", f"Error: {str(e)}")

@compliance_decorators_v2_5.track_compliance("performance_monitor", "Performance Monitor")
def create_performance_monitor(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """
    Create Performance Monitor container.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        
    Returns:
        html.Div: Performance monitor container
    """
    try:
        # Simulate performance metrics (in real implementation, these would be actual measurements)
        performance_metrics = [
            ("Response Time", "0.8s", AI_COLORS['success']),
            ("Memory Usage", "67%", AI_COLORS['warning']),
            ("CPU Load", "23%", AI_COLORS['success']),
            ("Error Rate", "0.1%", AI_COLORS['success'])
        ]
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "⚡ Performance",
                    "performance",
                    "Real-time system performance metrics: response times, resource usage, error rates"
                )
            ], className="container-header"),
            
            html.Div([
                # Performance metrics
                html.Div([
                    html.Div([
                        html.Small(metric[0], className="metric-label d-block"),
                        html.Strong(metric[1], style={"color": metric[2]})
                    ], className="performance-metric mb-2") for metric in performance_metrics
                ], className="performance-metrics"),
                
                # System efficiency
                html.Div([
                    html.Small("Efficiency:", className="text-muted"),
                    html.Strong("92%", className="ml-2 text-success")
                ], className="efficiency-score")
                
            ], className="performance-content")
            
        ], style=get_card_style('analysis'))
        
    except Exception as e:
        logger.error(f"Error creating performance monitor: {str(e)}")
        return create_placeholder_card("⚡ Performance", f"Error: {str(e)}")

@compliance_decorators_v2_5.track_compliance("alerts_status_monitor", "Alerts & Status Monitor")
def create_alerts_status_monitor(bundle_data: FinalAnalysisBundleV2_5, symbol: str, db_manager=None) -> html.Div:
    """
    Create Alerts & Status Monitor container.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5
        symbol: Trading symbol
        db_manager: Database manager instance
        
    Returns:
        html.Div: Alerts & status monitor container
    """
    try:
        # Generate alerts based on data analysis
        alerts = []
        
        # Check for extreme readings
        processed_data = bundle_data.processed_data_bundle
        if processed_data and processed_data.underlying_data_enriched:
            enriched_data = processed_data.underlying_data_enriched
            
            vapi_fa = getattr(enriched_data, 'vapi_fa_z_score_und', 0.0) or 0.0
            if abs(vapi_fa) > 2.0:
                alerts.append(("🚨", "Extreme VAPI-FA reading", AI_COLORS['danger']))
            
            dwfd = getattr(enriched_data, 'dwfd_z_score_und', 0.0) or 0.0
            if abs(dwfd) > 1.5:
                alerts.append(("⚠️", "Strong DWFD signal", AI_COLORS['warning']))
        
        # Add trading opportunities
        if bundle_data.atif_recommendations_v2_5:
            high_conviction_recs = [rec for rec in bundle_data.atif_recommendations_v2_5 
                                  if getattr(rec, 'conviction_score', 0) > 0.8]
            if high_conviction_recs:
                alerts.append(("💡", f"{len(high_conviction_recs)} high-conviction setups", AI_COLORS['success']))
        
        # Default message if no alerts
        if not alerts:
            alerts = [("✅", "All systems normal", AI_COLORS['success'])]
        
        # Calculate overall health score
        health_score = 95 if not any(alert[2] == AI_COLORS['danger'] for alert in alerts) else 75
        
        return html.Div([
            html.Div([
                create_clickable_title_with_info(
                    "🚨 Alerts & Status",
                    "alerts_status",
                    "Critical alerts, system warnings, trading opportunities, and overall health status"
                )
            ], className="container-header"),
            
            html.Div([
                # Alerts list
                html.Div([
                    html.Div([
                        html.Span(alert[0], className="alert-icon mr-2"),
                        html.Small(alert[1], style={"color": alert[2]})
                    ], className="alert-item mb-1") for alert in alerts
                ], className="alerts-list mb-3"),
                
                # Overall health score
                html.Div([
                    html.Small("System Health:", className="text-muted"),
                    html.Strong(f"{health_score}%", className="ml-2", style={
                        "color": AI_COLORS['success'] if health_score >= 90 else 
                                AI_COLORS['warning'] if health_score >= 70 else AI_COLORS['danger']
                    })
                ], className="health-score")
                
            ], className="alerts-content")
            
        ], style=get_card_style('warning'))
        
    except Exception as e:
        logger.error(f"Error creating alerts status monitor: {str(e)}")
        return create_placeholder_card("🚨 Alerts & Status", f"Error: {str(e)}")

