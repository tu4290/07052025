# utils/config_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL V2.5.3 IMPLEMENTATION (UNABRIDGED)

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

# Change relative import to absolute import
from data_models import EOTSConfigV2_5 # Updated import

logger = logging.getLogger(__name__)

class ConfigManagerV2_5:
    """
    Singleton configuration manager for EOTS v2.5.
    Handles loading, validation, and access to configuration settings.
    """
    _instance = None
    _config: Optional[EOTSConfigV2_5] = None
    _project_root: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManagerV2_5, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the singleton instance."""
        self._determine_project_root()
        self._load_config()

    def _determine_project_root(self):
        """Determine the project root directory."""
        # Start from the current file's directory
        current_dir = Path(__file__).parent
        # Look for the root package file
        while current_dir != current_dir.parent:
            if (current_dir / "elite_options_system_v2_5.py").exists():
                self._project_root = current_dir
                logger.debug(f"Project root determined as: {self._project_root}")
                return
            current_dir = current_dir.parent
        raise RuntimeError("Could not determine project root directory")

    def _substitute_env_vars(self, data):
        """Recursively substitute environment variables in configuration data."""
        import re
        
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Replace ${VAR} with environment variable value
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found, keeping placeholder")
                    return match.group(0)  # Return original ${VAR} if not found
                # Try to convert to appropriate type
                if env_value.isdigit():
                    return int(env_value)
                try:
                    return float(env_value)
                except ValueError:
                    return env_value
            
            pattern = r'\$\{([^}]+)\}'
            result = re.sub(pattern, replace_env_var, data)
            return result
        else:
            return data

    def _load_config(self):
        """Load and validate all configuration files."""
        if not self._project_root:
            raise RuntimeError("Project root not determined")

        # Load main config
        config_path = self._project_root / "config" / "config_v2_5.json"
        logger.info(f"Loading main configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            config_data = self._substitute_env_vars(config_data)
        except FileNotFoundError:
            raise RuntimeError(f"Main configuration file not found at {config_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in main configuration file: {e}")

        # Load HuiHui config
        huihui_config_path = self._project_root / "config" / "huihui_config.json"
        logger.info(f"Loading HuiHui configuration from: {huihui_config_path}")
        try:
            with open(huihui_config_path, 'r') as f:
                huihui_config_data = json.load(f)
            huihui_config_data = self._substitute_env_vars(huihui_config_data)
            # Merge HuiHui config into the main config data under a specific key
            config_data['huihui_settings'] = huihui_config_data
        except FileNotFoundError:
            logger.warning(f"HuiHui configuration file not found at {huihui_config_path}. HuiHui features may be disabled.")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in HuiHui configuration file: {e}. HuiHui features may be disabled.")

        # Validate and parse the combined configuration
        try:
            logger.debug("Parsing combined configuration into Pydantic model...")
            self._config = self._convert_to_pydantic_models(config_data)
            logger.info("Pydantic model parsing successful. All configurations are now loaded and type-safe.")
        except ValidationError as e:
            raise RuntimeError(f"Combined configuration validation failed: {e}")

    def get_setting(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration setting by path.

        Args:
            path: Dot-separated path to the setting (e.g., 'database.host')
            default: Default value to return if setting not found

        Returns:
            The requested setting value or default if not found
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        try:
            parts = path.split('.')
            value = self._config
            for part in parts:
                value = getattr(value, part)
            return value
        except (AttributeError, KeyError):
            return default



    @property
    def config(self) -> EOTSConfigV2_5:
        """Get the loaded configuration object."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        if not self._project_root:
            raise RuntimeError("Project root not determined")
        return self._project_root

    def get_resolved_path(self, path: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration setting that represents a path and resolve it relative to project root.
        
        Args:
            path: Dot-notation path to the setting (e.g., 'performance_tracker_settings_v2_5.performance_data_directory')
            default: Default value to return if setting is not found
            
        Returns:
            The resolved path string or None if not found
        """
        relative_path = self.get_setting(path, default)
        if not relative_path:
            return None
        
        if not self._project_root:
            raise RuntimeError("Project root not determined")
            
        return str(self._project_root / relative_path)

    def _convert_to_pydantic_models(self, config_data):
        """
        Convert configuration dictionary to Pydantic models using strict Pydantic v2 validation.
        This method handles special cases like modes_detail_config conversion.
        """
        try:
            # Handle special conversion for dashboard modes_detail_config
            if 'visualization_settings' in config_data:
                vis_settings = config_data['visualization_settings']
                if isinstance(vis_settings, dict) and 'dashboard' in vis_settings:
                    dashboard_config_data = vis_settings['dashboard']
                    
                    # Manually validate and convert modes_detail_config if present
                    if 'modes_detail_config' in dashboard_config_data and isinstance(dashboard_config_data['modes_detail_config'], dict):
                        from data_models.dashboard_config_models import DashboardModeCollection
                        dashboard_config_data['modes_detail_config'] = DashboardModeCollection.model_validate(dashboard_config_data['modes_detail_config'])
                    
                    from data_models.dashboard_config_models import DashboardServerConfig
                    vis_settings['dashboard'] = DashboardServerConfig.model_validate(dashboard_config_data)

            # Use Pydantic v2 model validation with extra='allow' for flexible configuration
            config_model = EOTSConfigV2_5.model_validate(config_data)
            logger.info("✅ Configuration successfully parsed as Pydantic v2 models")
            return config_model
        except Exception as e:
            logger.error(f"❌ Failed to parse configuration as Pydantic v2 models: {e}")
            raise ValueError(f"Dashboard configuration is not properly parsed as Pydantic model: {e}")

    def load_config(self):
        """Public method to load the configuration."""
        self._load_config()