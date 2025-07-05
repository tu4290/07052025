"""
🎯 EXPERT ROUTER - ADAPTIVE LEARNING INTEGRATION
==================================================================

This module provides integration with the adaptive learning system for
continuous improvement of expert routing decisions.
"""

import logging
import random
from typing import Dict, Any, Optional, Tuple, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum
from data_models.configuration_models import AdaptiveLearningConfigV2_5, AnalyticsEngineConfigV2_5

# Optional import with graceful degradation
try:
    from core_analytics_engine.adaptive_learning_integration_v2_5 import (
        AdaptiveLearningIntegrationV2_5
    )
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ADAPTIVE_LEARNING_AVAILABLE = False
    AdaptiveLearningIntegrationV2_5 = None
    logging.warning(
        "AdaptiveLearningIntegrationV2_5 not available. "
        "Adaptive learning features will be disabled."
    )

logger = logging.getLogger(__name__)

class LearningMode(str, Enum):
    """Modes for adaptive learning."""
    DISABLED = "disabled"
    PASSIVE = "passive"  # Only collect data, don't modify behavior
    ACTIVE = "active"    # Actively modify behavior based on learning

@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning integration."""
    enabled: bool = True
    mode: LearningMode = LearningMode.ACTIVE
    update_interval: int = 300  # seconds between model updates
    exploration_rate: float = 0.1  # Rate of exploration vs exploitation
    min_samples: int = 100  # Minimum samples before making decisions
    max_retries: int = 3
    retry_delay: float = 1.0

class AdaptiveLearningManager:
    """Manages integration with the adaptive learning system."""
    
    def __init__(
        self, 
        config: Optional[AdaptiveLearningConfigV2_5] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ):
        """Initialize the adaptive learning manager.
        
        Args:
            config: Configuration for adaptive learning (Pydantic model)
            metrics_callback: Optional callback for reporting metrics
        """
        # Provide all required fields for AdaptiveLearningConfigV2_5 if not supplied
        if config is None:
            self.config = AdaptiveLearningConfigV2_5(
                auto_adaptation=True,
                confidence_threshold=0.7,
                pattern_discovery_threshold=0.6,
                adaptation_frequency_minutes=5,
                analytics_engine=AnalyticsEngineConfigV2_5(
                    metrics_calculation_enabled=True,
                    market_regime_analysis_enabled=True,
                    signal_generation_enabled=True,
                    key_level_identification_enabled=True
                )
            )
        else:
            self.config = config
        self.metrics_callback = metrics_callback
        self._learning_system = None
        self._initialized = False
        self._last_update = None
        self._update_lock = asyncio.Lock()
        self._update_task = None
        
        if ADAPTIVE_LEARNING_AVAILABLE and self.config and getattr(self.config, 'auto_adaptation', True):
            try:
                if AdaptiveLearningIntegrationV2_5 is not None:  # Dynamic import safety check
                    self._learning_system = AdaptiveLearningIntegrationV2_5(self.config)
                else:
                    logger.error("AdaptiveLearningIntegrationV2_5 is None (dynamic import failed)")
                    self._learning_system = None
            except Exception as e:
                logger.error(f"Failed to initialize AdaptiveLearningIntegrationV2_5: {e}")
                self._learning_system = None
        else:
            self._learning_system = None
            logger.warning(
                "Adaptive learning is disabled or not available. "
                "Falling back to basic routing strategies."
            )
    
    async def initialize(self) -> None:
        """Initialize the adaptive learning system."""
        if not self.config or not ADAPTIVE_LEARNING_AVAILABLE:
            self._initialized = True
            return
        try:
            if self._learning_system is not None and hasattr(self._learning_system, 'initialize'):
                await self._learning_system.initialize()  # type: ignore  # Safe: method is guaranteed by system contract
            self._initialized = True
            logger.info("Adaptive learning system initialized successfully")
            # Schedule periodic updates
            self._update_task = asyncio.create_task(self._periodic_update())
        except Exception as e:
            logger.error(f"Failed to initialize adaptive learning: {e}")
            self._initialized = False
    
    async def update_model(self) -> bool:
        """Update the adaptive learning model with new data."""
        if not self._initialized or not self.config:
            return False
        async with self._update_lock:
            try:
                if self._learning_system is not None and hasattr(self._learning_system, 'update_model'):
                    success = await self._learning_system.update_model()  # type: ignore  # Safe: method is guaranteed by system contract
                    if success:
                        self._last_update = datetime.utcnow()
                        logger.info("Adaptive learning model updated successfully")
                    return success
                else:
                    logger.warning("Learning system is not available or missing update_model method.")
                    return False
            except Exception as e:
                logger.error(f"Error updating adaptive learning model: {e}")
                return False
    
    async def get_expert_recommendation(
        self, 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """Get an expert recommendation from the adaptive learning system."""
        if not self._initialized or not self.config:
            return "", 0.0
        try:
            if self._learning_system is not None and hasattr(self._learning_system, 'get_recommendation'):
                result = await self._learning_system.get_recommendation(  # type: ignore  # Safe: method is guaranteed by system contract
                    query=query,
                    context=context or {},
                    explore=self._should_explore()
                )
                return result.get('expert_type', ""), result.get('confidence', 0.0)
            else:
                logger.warning("Learning system is not available or missing get_recommendation method.")
                return "", 0.0
        except Exception as e:
            logger.warning(f"Error getting expert recommendation: {e}")
            return "", 0.0
    
    async def record_feedback(
        self,
        query: str,
        expert_used: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record feedback about a routing decision."""
        if not self._initialized or not self.config:
            return
        try:
            feedback = {
                'query': query,
                'expert_used': expert_used,
                'success': success,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics or {}
            }
            if self._learning_system is not None and hasattr(self._learning_system, 'record_feedback'):
                await self._learning_system.record_feedback(feedback)  # type: ignore  # Safe: method is guaranteed by system contract
            # Trigger metrics callback if provided
            if self.metrics_callback:
                await self.metrics_callback({
                    'type': 'adaptive_learning_feedback',
                    'success': success,
                    'expert': expert_used,
                    'timestamp': datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    def _should_explore(self) -> bool:
        """Determine whether to explore or exploit based on configuration."""
        if not self.config or getattr(self.config, 'auto_adaptation', False) is False:
            return False
        # Use exploration_rate if present, else default to 0.1
        exploration_rate = getattr(self.config, 'exploration_rate', 0.1)
        return random.random() < exploration_rate
    
    async def _periodic_update(self) -> None:
        """Periodically update the adaptive learning model."""
        while True:
            try:
                await asyncio.sleep(getattr(self.config, 'adaptation_frequency_minutes', 5) * 60)
                await self.update_model()
            except asyncio.CancelledError:
                logger.info("Periodic update task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        if self._learning_system and hasattr(self._learning_system, 'shutdown'):
            await self._learning_system.shutdown()
