"""
HuiHui Configuration - Expert and System Settings
=================================================

Configuration management for HuiHui AI expert system including:
- Individual expert configurations
- System-wide settings
- Performance optimization parameters
- Learning algorithm settings
- Database connection configurations

Features:
- Expert-specific temperature and token settings
- Dynamic configuration updates
- Performance-based auto-tuning
- Environment-specific configurations
- Validation and error handling

Author: EOTS v2.5 AI Architecture Division
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Configuration system info
CONFIG_INFO = {
    "name": "HuiHui Configuration System",
    "version": "2.5.0",
    "experts_configured": ["market_regime", "options_flow", "sentiment"],
    "config_sources": ["expert_configs.py", "system_settings.py", "environment_variables"],
    "auto_tuning_enabled": True
}

# Configuration status
_config_initialized = False
_expert_configs_loaded = False
_system_settings_loaded = False

def get_config_status() -> Dict[str, Any]:
    """Get HuiHui configuration system status."""
    return {
        "info": CONFIG_INFO,
        "status": {
            "config_initialized": _config_initialized,
            "expert_configs_loaded": _expert_configs_loaded,
            "system_settings_loaded": _system_settings_loaded,
            "all_configs_ready": all([
                _config_initialized,
                _expert_configs_loaded,
                _system_settings_loaded
            ])
        }
    }

# Import guards
try:
    from .expert_configs import HuiHuiConfigManager, get_config_manager
    EXPERT_CONFIGS_AVAILABLE = True
except ImportError:
    EXPERT_CONFIGS_AVAILABLE = False
    logger.debug("Expert Configs not available")

try:
    from .system_settings import HuiHuiSystemSettings
    SYSTEM_SETTINGS_AVAILABLE = True
except ImportError:
    SYSTEM_SETTINGS_AVAILABLE = False
    logger.debug("System Settings not available")

def get_expert_config_manager():
    """Get Expert Configuration Manager instance."""
    if not EXPERT_CONFIGS_AVAILABLE:
        raise ImportError("Expert Configuration Manager not available")
    return get_config_manager()

def get_system_settings():
    """Get System Settings instance."""
    if not SYSTEM_SETTINGS_AVAILABLE:
        raise ImportError("System Settings not available")
    return HuiHuiSystemSettings()

async def initialize_config() -> bool:
    """Initialize the complete HuiHui configuration system."""
    global _config_initialized, _expert_configs_loaded, _system_settings_loaded
    
    try:
        logger.info("⚙️ Initializing HuiHui Configuration System...")
        
        # Load expert configurations
        if EXPERT_CONFIGS_AVAILABLE:
            config_manager = get_expert_config_manager()
            _expert_configs_loaded = True
            logger.info("✅ Expert configurations loaded")
        
        # Load system settings
        if SYSTEM_SETTINGS_AVAILABLE:
            system_settings = get_system_settings()
            _system_settings_loaded = True
            logger.info("✅ System settings loaded")
        
        _config_initialized = True
        logger.info("🚀 HuiHui Configuration System initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize configuration system: {e}")
        return False

# Quick configuration access functions
def get_expert_temperature(expert_name: str) -> float:
    """Get temperature setting for specific expert."""
    try:
        if EXPERT_CONFIGS_AVAILABLE:
            config_manager = get_expert_config_manager()
            return config_manager.get_expert_temperature(expert_name)
        return 0.1  # Default temperature
    except Exception:
        return 0.1

def get_expert_max_tokens(expert_name: str) -> int:
    """Get max tokens setting for specific expert."""
    try:
        if EXPERT_CONFIGS_AVAILABLE:
            config_manager = get_expert_config_manager()
            config = config_manager.get_expert_config(expert_name)
            return config.max_tokens if config else 2000
        return 2000  # Default max tokens
    except Exception:
        return 2000

def get_expert_keywords(expert_name: str) -> list:
    """Get keywords for specific expert."""
    try:
        if EXPERT_CONFIGS_AVAILABLE:
            config_manager = get_expert_config_manager()
            return config_manager.get_expert_keywords(expert_name)
        return []
    except Exception:
        return []

__all__ = [
    "CONFIG_INFO",
    "get_config_status",
    "get_expert_config_manager",
    "get_system_settings",
    "initialize_config",
    "get_expert_temperature",
    "get_expert_max_tokens",
    "get_expert_keywords",
    "EXPERT_CONFIGS_AVAILABLE",
    "SYSTEM_SETTINGS_AVAILABLE"
]
