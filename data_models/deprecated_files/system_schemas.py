from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from data_models.ui_component_schemas import SystemComponentStatuses

class SystemStateV2_5(BaseModel):
    """Represents the overall operational state of the EOTS v2.5 system."""
    is_running: bool = Field(..., description="True if the system is actively running, False otherwise.")
    current_mode: str = Field(..., description="Current operational mode (e.g., 'operational', 'maintenance', 'diagnostic').")
    active_processes: List[str] = Field(default_factory=list, description="List of currently active system processes or modules.")
    status_message: str = Field(..., description="A human-readable message describing the current system status.")
    last_heartbeat: Optional[str] = Field(None, description="Timestamp of the last successful system heartbeat.")
    errors: List[str] = Field(default_factory=list, description="List of recent critical errors or warnings.")

    class Config:
        extra = 'forbid'

class AISystemHealthV2_5(BaseModel):
    """Comprehensive Pydantic model for AI system health monitoring."""
    # Database connectivity
    database_connected: bool = Field(default=False, description="Database connection status")
    ai_tables_available: bool = Field(default=False, description="AI tables availability status")

    # Component health
    predictions_manager_healthy: bool = Field(default=False, description="AI Predictions Manager health")
    learning_system_healthy: bool = Field(default=False, description="AI Learning System health")
    adaptation_engine_healthy: bool = Field(default=False, description="AI Adaptation Engine health")

    # Performance metrics
    overall_health_score: float = Field(default=0.0, description="Overall system health score", ge=0.0, le=1.0)
    response_time_ms: float = Field(default=0.0, description="System response time in milliseconds", ge=0.0)
    error_rate: float = Field(default=0.0, description="System error rate", ge=0.0, le=1.0)

    # Status details
    status_message: str = Field(default="System initializing...", description="Current status message")
    component_status: SystemComponentStatuses = Field(default_factory=SystemComponentStatuses, description="Detailed status of system components")
    last_checked: datetime = Field(default_factory=datetime.now, description="Last health check timestamp")

    class Config:
        extra = 'forbid'
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }