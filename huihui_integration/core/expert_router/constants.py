"""
🎯 EXPERT ROUTER - CONSTANTS
==================================================================

This module contains all the constants used across the Expert Router system.
"""

# Default configuration values
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MAX_CONNECTIONS = 20
SLIDING_WINDOW_SIZE = 100
DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 10.0
WEIGHT_ADJUSTMENT_STEP = 0.1
CONFIDENCE_THRESHOLD = 0.7

# Cache configuration
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
DEFAULT_CACHE_SIZE = 1000

# Metrics configuration
DEFAULT_METRICS_PORT = 9090
METRICS_UPDATE_INTERVAL = 60  # seconds

# Adaptive learning configuration
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EXPLORATION_RATE = 0.1
DEFAULT_LEARNING_WINDOW = 1000

# Performance tuning
DEFAULT_TIMEOUT = 30.0  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Expert types configuration
DEFAULT_EXPERTS = {
    "market_regime": {
        "display_name": "🏛️ Market Regime Expert",
        "description": "Specializes in market regime analysis and volatility patterns",
        "default_weight": 1.0
    },
    "options_flow": {
        "display_name": "📊 Options Flow Expert",
        "description": "Analyzes options flow and order book dynamics",
        "default_weight": 1.0
    },
    "sentiment": {
        "display_name": "🧠 Sentiment Expert",
        "description": "Analyzes market sentiment and social signals",
        "default_weight": 1.0
    },
    "orchestrator": {
        "display_name": "🎯 Meta-Orchestrator",
        "description": "Coordinates between different experts for comprehensive analysis",
        "default_weight": 1.0
    }
}

# Error messages
ERROR_MESSAGES = {
    "expert_not_found": "Expert type {expert_type} not found",
    "connection_error": "Failed to connect to expert service: {error}",
    "timeout": "Request timed out after {timeout} seconds",
    "invalid_config": "Invalid configuration: {error}",
    "validation_error": "Validation error: {error}",
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Version information
__version__ = "2.5.0"
__author__ = "EOTS Engineering Team"
__status__ = "Production"
