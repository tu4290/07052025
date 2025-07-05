"""
HuiHui Monitoring - Performance Tracking and Analytics
======================================================

Comprehensive monitoring system for HuiHui AI experts including:
- Usage pattern analysis and optimization
- Performance metrics per expert
- Supabase-only data storage (no SQLite)
- Safety and security management
- Real-time system health monitoring

Features:
- Individual expert performance tracking
- Cross-expert analytics and insights
- Optimization recommendations
- Safety timeout and retry management
- Security validation and threat detection

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Monitoring configuration
MONITORING_CONFIG = {
    "name": "HuiHui Monitoring System",
    "version": "2.5.0",
    "storage_backend": "supabase_only",  # No SQLite support
    "experts_monitored": ["market_regime", "options_flow", "sentiment", "orchestrator"],
    "metrics_tracked": ["usage", "performance", "safety", "security"],
    "real_time_monitoring": True
}

# Monitoring status
_monitoring_initialized = False
_supabase_connected = False
_safety_enabled = False
_security_enabled = False

def get_monitoring_status() -> Dict[str, Any]:
    """Get HuiHui monitoring system status."""
    return {
        "config": MONITORING_CONFIG,
        "status": {
            "monitoring_initialized": _monitoring_initialized,
            "supabase_connected": _supabase_connected,
            "safety_enabled": _safety_enabled,
            "security_enabled": _security_enabled,
            "all_systems_operational": all([
                _monitoring_initialized,
                _supabase_connected,
                _safety_enabled,
                _security_enabled
            ])
        }
    }

# Import guards
try:
    from .usage_monitor import HuiHuiUsageMonitor, get_usage_monitor
    USAGE_MONITOR_AVAILABLE = True
except ImportError:
    USAGE_MONITOR_AVAILABLE = False
    logger.debug("Usage Monitor not available")

try:
    from .supabase_manager import HuiHuiSupabaseManager, get_supabase_manager
    SUPABASE_MANAGER_AVAILABLE = True
except ImportError:
    SUPABASE_MANAGER_AVAILABLE = False
    logger.debug("Supabase Manager not available")

try:
    from .safety_manager import HuiHuiSafetyManager, get_safety_manager
    SAFETY_MANAGER_AVAILABLE = True
except ImportError:
    SAFETY_MANAGER_AVAILABLE = False
    logger.debug("Safety Manager not available")

try:
    from .security_manager import HuiHuiSecurityManager, get_security_manager
    SECURITY_MANAGER_AVAILABLE = True
except ImportError:
    SECURITY_MANAGER_AVAILABLE = False
    logger.debug("Security Manager not available")

# Convenience functions
def get_usage_monitor_instance():
    """Get HuiHui Usage Monitor instance."""
    if not USAGE_MONITOR_AVAILABLE:
        raise ImportError("Usage Monitor not available")
    return get_usage_monitor()

def get_supabase_manager_instance():
    """Get HuiHui Supabase Manager instance."""
    if not SUPABASE_MANAGER_AVAILABLE:
        raise ImportError("Supabase Manager not available")
    return get_supabase_manager()

def get_safety_manager_instance():
    """Get HuiHui Safety Manager instance."""
    if not SAFETY_MANAGER_AVAILABLE:
        raise ImportError("Safety Manager not available")
    return get_safety_manager()

def get_security_manager_instance():
    """Get HuiHui Security Manager instance."""
    if not SECURITY_MANAGER_AVAILABLE:
        raise ImportError("Security Manager not available")
    return get_security_manager()

async def initialize_monitoring() -> bool:
    """Initialize the complete HuiHui monitoring system."""
    global _monitoring_initialized, _supabase_connected, _safety_enabled, _security_enabled
    
    try:
        logger.info("📊 Initializing HuiHui Monitoring System...")
        
        # Initialize Supabase manager
        if SUPABASE_MANAGER_AVAILABLE:
            supabase_manager = get_supabase_manager_instance()
            _supabase_connected = await supabase_manager.initialize()
            logger.info(f"✅ Supabase connection: {_supabase_connected}")
        
        # Initialize usage monitor
        if USAGE_MONITOR_AVAILABLE:
            usage_monitor = get_usage_monitor_instance()
            logger.info("✅ Usage monitor initialized")
        
        # Initialize safety manager
        if SAFETY_MANAGER_AVAILABLE:
            safety_manager = get_safety_manager_instance()
            _safety_enabled = True
            logger.info("✅ Safety manager initialized")
        
        # Initialize security manager
        if SECURITY_MANAGER_AVAILABLE:
            security_manager = get_security_manager_instance()
            _security_enabled = True
            logger.info("✅ Security manager initialized")
        
        _monitoring_initialized = True
        logger.info("🚀 HuiHui Monitoring System initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize monitoring system: {e}")
        return False

# Quick monitoring functions
async def record_expert_usage(expert: str, request_type: str, input_tokens: int, 
                             output_tokens: int, processing_time: float, success: bool = True):
    """Quick function to record expert usage."""
    try:
        if USAGE_MONITOR_AVAILABLE:
            monitor = get_usage_monitor_instance()
            monitor.record_usage(expert, request_type, input_tokens, output_tokens, processing_time, success)
    except Exception as e:
        logger.debug(f"Failed to record usage: {e}")

async def get_expert_performance(expert: str, hours: int = 24) -> Dict[str, Any]:
    """Quick function to get expert performance metrics."""
    try:
        if USAGE_MONITOR_AVAILABLE:
            monitor = get_usage_monitor_instance()
            return await monitor.analyze_usage_patterns(expert, hours)
        return {}
    except Exception as e:
        logger.debug(f"Failed to get performance: {e}")
        return {}

__all__ = [
    "MONITORING_CONFIG",
    "get_monitoring_status",
    "get_usage_monitor_instance",
    "get_supabase_manager_instance",
    "get_safety_manager_instance",
    "get_security_manager_instance",
    "initialize_monitoring",
    "record_expert_usage",
    "get_expert_performance",
    "USAGE_MONITOR_AVAILABLE",
    "SUPABASE_MANAGER_AVAILABLE",
    "SAFETY_MANAGER_AVAILABLE",
    "SECURITY_MANAGER_AVAILABLE"
]
