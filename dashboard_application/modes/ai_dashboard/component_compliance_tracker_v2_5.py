"""Component Compliance Tracker v2.5

This module provides compliance tracking functionality for EOTS v2.5 components.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Enumeration of data source types for compliance tracking."""
    BUNDLE_DATA = "bundle_data"
    PROCESSED_DATA = "processed_data"
    AI_SETTINGS = "ai_settings"
    SYMBOL = "symbol"
    OTHER = "other"

class ComponentComplianceTracker:
    """Tracks compliance for dashboard components."""
    
    def __init__(self):
        self.tracked_components = {}
        self.data_access_log = []
    
    def track_component(self, component_id: str, component_name: str, data_sources: List[DataSourceType]):
        """Track a component's compliance status."""
        self.tracked_components[component_id] = {
            'name': component_name,
            'data_sources': data_sources,
            'timestamp': datetime.now(),
            'status': 'active'
        }
        logger.debug(f"Tracking component: {component_id} - {component_name}")
    
    def log_data_access(self, component_id: str, data_type: DataSourceType, details: str = ""):
        """Log data access for compliance tracking."""
        self.data_access_log.append({
            'component_id': component_id,
            'data_type': data_type,
            'details': details,
            'timestamp': datetime.now()
        })
        logger.debug(f"Data access logged: {component_id} -> {data_type.value}")

# Global tracker instance
_compliance_tracker = ComponentComplianceTracker()

def get_compliance_tracker() -> ComponentComplianceTracker:
    """Get the global compliance tracker instance."""
    return _compliance_tracker

def track_component_creation(component_id: str, component_name: str, data_sources: List[DataSourceType]):
    """Track component creation."""
    _compliance_tracker.track_component(component_id, component_name, data_sources)

def track_data_access(component_id: str, data_type: DataSourceType, details: str = ""):
    """Track data access."""
    _compliance_tracker.log_data_access(component_id, data_type, details)