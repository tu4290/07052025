"""
Monitoring stack management for EOTS v2.5
Handles Prometheus and Grafana containers and dashboards.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import requests

logger = logging.getLogger(__name__)


class GrafanaAPI:
    """Handles Grafana API interactions."""
    
    def __init__(self, base_url: str = "http://localhost:3000", 
                 username: str = "admin", password: str = "admin"):
        """Initialize the Grafana API client."""
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password)
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def is_ready(self, timeout: int = 60) -> bool:
        """Check if Grafana is ready to accept requests."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/api/health",
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except (requests.RequestException, ConnectionError):
                pass
            time.sleep(1)
        return False
    
    def create_datasource(self, name: str, url: str) -> bool:
        """Create a Prometheus data source in Grafana."""
        data = {
            "name": name,
            "type": "prometheus",
            "url": url,
            "access": "proxy",
            "isDefault": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/datasources",
                headers=self.headers,
                auth=self.auth,
                json=data,
                timeout=10
            )
            
            if response.status_code in (200, 409):  # 409 means already exists
                logger.info("Grafana data source '%s' is ready", name)
                return True
                
            logger.error("Failed to create Grafana data source: %s", response.text)
            return False
            
        except requests.RequestException:
            logger.exception("Error creating Grafana data source")
            return False
    
    def create_dashboard(self, dashboard: Dict[str, Any]) -> bool:
        """Create or update a dashboard in Grafana."""
        data = {
            "dashboard": dashboard,
            "overwrite": True,
            "message": "Updated by EOTS monitoring",
            "folderId": 0
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/dashboards/db",
                headers=self.headers,
                auth=self.auth,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Dashboard created/updated successfully")
                return True
                
            logger.error("Failed to create dashboard: %s", response.text)
            return False
            
        except requests.RequestException:
            logger.exception("Error creating dashboard")
            return False


class MonitoringStack:
    """Manages the monitoring stack (Prometheus + Grafana)."""

    def __init__(self, project_root: str) -> None:
        """Initialize the monitoring stack.
        
        Args:
            project_root: Path to the project root directory.
        """
        self.project_root = Path(project_root)
        self.compose_file = self.project_root / 'docker-compose.monitoring.yml'
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        self._ensure_compose_file()

    def _ensure_compose_file(self) -> None:
        """Ensure the monitoring compose file exists."""
        if not self.compose_file.exists():
            self._create_default_compose_file()

    def _create_default_compose_file(self) -> None:
        """Create a default docker-compose file for monitoring."""
        content = """version: '3'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: eots-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: eots-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
"""
        with open(self.compose_file, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info("Created monitoring compose file at %s", self.compose_file)

    def start(self) -> bool:
        """Start the monitoring stack and set up dashboards.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Ensure prometheus.yml exists
            prometheus_yml = self.project_root / 'prometheus.yml'
            if not prometheus_yml.exists():
                self._create_default_prometheus_config()

            # Start the stack
            cmd = ["docker-compose", "-f", str(self.compose_file), "up", "-d"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error("Failed to start monitoring stack: %s", result.stderr)
                return False

            logger.info(
                "Monitoring stack started. Grafana: %s, Prometheus: %s",
                self.grafana_url,
                self.prometheus_url
            )
            
            # Wait for Grafana to be ready
            grafana = GrafanaAPI()
            if not grafana.is_ready():
                logger.error("Grafana did not become ready in time")
                return False
                
            # Set up data source
            if not grafana.create_datasource("Prometheus", "http://prometheus:9090"):
                logger.error("Failed to set up Grafana data source")
                return False
                
            # Import dashboard
            from dashboard_application.modes.monitoring_dashboard_v2_5 import setup_monitoring_dashboard
            dashboard_config = setup_monitoring_dashboard(str(self.project_root))
            
            if not grafana.create_dashboard(dashboard_config):
                logger.error("Failed to set up Grafana dashboard")
                return False
                
            logger.info("Grafana dashboard is ready at %s", self.grafana_url)
            return True

        except Exception:
            logger.exception("Error starting monitoring stack")
            return False

    def stop(self) -> bool:
        """Stop the monitoring stack.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "down"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error("Failed to stop monitoring stack: %s", result.stderr)
                return False

            logger.info("Monitoring stack stopped")
            return True

        except Exception:
            logger.exception("Error stopping monitoring stack")
            return False

    def _create_default_prometheus_config(self) -> None:
        """Create a default Prometheus configuration."""
        content = """global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'eots'
    static_configs:
      - targets: ['host.docker.internal:8000']  # For Windows/Mac

  # Add more scrape configs as needed
  # - job_name: 'node'
  #   static_configs:
  #     - targets: ['node-exporter:9100']
"""
        config_path = self.project_root / 'prometheus.yml'
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info("Created default Prometheus config at %s", config_path)


def setup_monitoring(project_root: str) -> bool:
    """Set up and start the monitoring stack.
    
    Args:
        project_root: Path to the project root directory.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        monitoring = MonitoringStack(project_root)
        return monitoring.start()
    except Exception:
        logger.exception("Failed to set up monitoring")
        return False
