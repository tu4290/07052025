"""
Monitoring Dashboard for EOTS v2.5

Provides a unified view of system metrics using Prometheus and Grafana.
"""

from typing import Dict, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Handles the monitoring dashboard configuration and setup."""
    
    def __init__(self, project_root: str):
        """Initialize the monitoring dashboard.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
    def get_dashboard_json(self) -> Dict[str, Any]:
        """Return the Grafana dashboard configuration as a dictionary."""
        return {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "target": {
                            "limit": 100,
                            "matchAny": False,
                            "tags": [],
                            "type": "dashboard"
                        },
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "id": 1,
            "links": [],
            "liveNow": True,
            "panels": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": "${DS_PROMETHEUS}"
                    },
                    "fieldConfig": {
                        "defaults": {
                            "color": {
                                "mode": "palette-classic"
                            },
                            "custom": {
                                "axisCenteredZero": False,
                                "axisColorMode": "text",
                                "axisLabel": "",
                                "axisPlacement": "auto",
                                "drawStyle": "line",
                                "fillOpacity": 10,
                                "gradientMode": "none",
                                "lineInterpolation": "linear",
                                "lineWidth": 2,
                                "pointSize": 5,
                                "showPoints": "auto"
                            }
                        }
                    },
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 0
                    },
                    "id": 1,
                    "options": {
                        "legend": {
                            "displayMode": "table",
                            "placement": "bottom",
                            "showLegend": True
                        },
                        "tooltip": {
                            "mode": "single",
                            "sort": "none"
                        }
                    },
                    "targets": [
                        {
                            "datasource": {
                                "type": "prometheus",
                                "uid": "${DS_PROMETHEUS}"
                            },
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{handler}} - {{status_code}}",
                            "refId": "A"
                        }
                    ],
                    "title": "HTTP Request Rate",
                    "type": "timeseries"
                }
            ],
            "refresh": "5s",
            "schemaVersion": 36,
            "style": "dark",
            "tags": ["eots", "monitoring"],
            "templating": {
                "list": []
            },
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "timepicker": {},
            "timezone": "browser",
            "title": "EOTS System Monitoring",
            "version": 1,
            "weekStart": ""
        }
    
    def save_dashboard_json(self, filename: str = "eots_monitoring_dashboard.json") -> Path:
        """Save the dashboard JSON to a file.
        
        Args:
            filename: Name of the file to save
            
        Returns:
            Path: Path to the saved file
        """
        dashboard_path = self.config_dir / filename
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_dashboard_json(), f, indent=2)
        return dashboard_path
    
    def get_import_instructions(self) -> str:
        """Return instructions for importing the dashboard into Grafana."""
        return """
        To import this dashboard into Grafana:
        
        1. Open Grafana at http://localhost:3000
        2. Log in (default: admin/admin)
        3. Click '+' > Import
        4. Upload the dashboard JSON file or paste its contents
        5. Select Prometheus as the data source
        6. Click 'Import'
        """


def setup_monitoring_dashboard(project_root: str) -> Dict[str, Any]:
    """Set up and return the monitoring dashboard configuration.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        dict: Dashboard configuration
    """
    dashboard = MonitoringDashboard(project_root)
    dashboard_path = dashboard.save_dashboard_json()
    logger.info(f"Saved monitoring dashboard to: {dashboard_path}")
    logger.info(dashboard.get_import_instructions())
    return dashboard.get_dashboard_json()
