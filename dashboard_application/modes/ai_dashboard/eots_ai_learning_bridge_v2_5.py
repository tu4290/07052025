"""
EOTS AI Learning Bridge v2.5 - "SENTIENT SYSTEM INTEGRATION"
============================================================

This module creates the final bridge between the EOTS dashboard system and the
Pydantic AI Learning Manager, ensuring that every EOTS analysis automatically
feeds into the learning system for continuous AI evolution.

Key Features:
- Automatic learning integration with EOTS dashboard updates
- Real-time prediction recording during dashboard refresh
- Seamless learning data flow without user intervention
- Pydantic-first architecture with validated learning
- Background learning validation and pattern discovery

Author: EOTS v2.5 AI Intelligence Division
Version: 2.5.0 - "SENTIENT SYSTEM INTEGRATION"
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import EOTS schemas
from data_models import FinalAnalysisBundleV2_5

# Import learning components
from .pydantic_ai_learning_manager_v2_5 import get_pydantic_ai_learning_manager
from .eots_learning_integration_v2_5 import (
    get_eots_learning_integration,
    record_eots_bundle_for_learning,
    validate_eots_learning_prediction
)

logger = logging.getLogger(__name__)

# ===== EOTS AI LEARNING BRIDGE =====

class EOTSAILearningBridge:
    """
    EOTS AI Learning Bridge - Sentient System Integration
    
    This class automatically integrates learning into every EOTS dashboard update,
    creating a seamless learning loop that evolves the AI without user intervention.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logger.getChild(self.__class__.__name__)
        self.last_prediction_id: Optional[str] = None
        self.last_bundle_timestamp: Optional[datetime] = None
        self.learning_enabled = True
        self.logger.info("🌉 EOTS AI Learning Bridge initialized")
    
    async def process_dashboard_update(self, bundle_data: FinalAnalysisBundleV2_5) -> Dict[str, Any]:
        """Process dashboard update and automatically integrate learning."""
        try:
            learning_results = {
                "prediction_recorded": False,
                "prediction_validated": False,
                "pattern_discovered": False,
                "learning_active": self.learning_enabled,
                "prediction_id": None
            }
            
            if not self.learning_enabled:
                return learning_results
            
            # Record new prediction for learning
            prediction_id = await self._record_current_analysis(bundle_data)
            if prediction_id:
                learning_results["prediction_recorded"] = True
                learning_results["prediction_id"] = prediction_id
                self.last_prediction_id = prediction_id
                self.logger.info(f"📝 Recorded EOTS analysis for learning: {prediction_id}")
            
            # Validate previous prediction if available
            if self.last_prediction_id and self._should_validate_prediction():
                validation_success = await self._validate_previous_prediction(bundle_data)
                if validation_success:
                    learning_results["prediction_validated"] = True
                    self.logger.info(f"✅ Validated previous prediction: {self.last_prediction_id}")
            
            # Update timestamp
            self.last_bundle_timestamp = datetime.now()
            
            # Trigger background learning tasks
            asyncio.create_task(self._background_learning_tasks())
            
            return learning_results
            
        except Exception as e:
            self.logger.error(f"Error processing dashboard update for learning: {e}")
            return {
                "prediction_recorded": False,
                "prediction_validated": False,
                "pattern_discovered": False,
                "learning_active": False,
                "error": str(e)
            }
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        try:
            # Get learning manager
            manager = await get_pydantic_ai_learning_manager(self.db_manager)
            
            # Get learning statistics
            stats = await manager.get_real_learning_stats()
            
            # Get recent insights
            insights = await manager.get_real_learning_insights(3)
            
            return {
                "learning_enabled": self.learning_enabled,
                "last_prediction_id": self.last_prediction_id,
                "last_update": self.last_bundle_timestamp.isoformat() if self.last_bundle_timestamp else None,
                "learning_stats": stats.model_dump(),
                "recent_insights": insights,
                "status": "active" if self.learning_enabled else "disabled"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning status: {e}")
            return {
                "learning_enabled": False,
                "status": "error",
                "error": str(e)
            }
    
    async def enable_learning(self) -> bool:
        """Enable automatic learning integration."""
        try:
            self.learning_enabled = True
            self.logger.info("🧠 EOTS AI Learning enabled")
            return True
        except Exception as e:
            self.logger.error(f"Error enabling learning: {e}")
            return False
    
    async def disable_learning(self) -> bool:
        """Disable automatic learning integration."""
        try:
            self.learning_enabled = False
            self.logger.info("⏸️ EOTS AI Learning disabled")
            return True
        except Exception as e:
            self.logger.error(f"Error disabling learning: {e}")
            return False
    
    async def force_validate_prediction(self, prediction_id: str, 
                                      current_bundle: FinalAnalysisBundleV2_5) -> bool:
        """Force validation of a specific prediction."""
        try:
            success = await validate_eots_learning_prediction(
                prediction_id, current_bundle, self.db_manager
            )
            
            if success:
                self.logger.info(f"✅ Force validated prediction: {prediction_id}")
            else:
                self.logger.warning(f"❌ Failed to validate prediction: {prediction_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error force validating prediction: {e}")
            return False
    
    # ===== PRIVATE METHODS =====
    
    async def _record_current_analysis(self, bundle_data: FinalAnalysisBundleV2_5) -> str:
        """Record current EOTS analysis as a prediction for learning."""
        try:
            # Record bundle for learning
            prediction_id = await record_eots_bundle_for_learning(bundle_data, self.db_manager)
            
            if prediction_id:
                self.logger.debug(f"Recorded EOTS bundle for learning: {prediction_id}")
            
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Error recording current analysis: {e}")
            return ""
    
    async def _validate_previous_prediction(self, current_bundle: FinalAnalysisBundleV2_5) -> bool:
        """Validate the previous prediction against current market state."""
        try:
            if not self.last_prediction_id:
                return False
            
            # Validate using current bundle as outcome
            success = await validate_eots_learning_prediction(
                self.last_prediction_id, current_bundle, self.db_manager
            )
            
            if success:
                # Clear the validated prediction
                self.last_prediction_id = None
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error validating previous prediction: {e}")
            return False
    
    def _should_validate_prediction(self) -> bool:
        """Determine if we should validate the previous prediction."""
        try:
            if not self.last_prediction_id or not self.last_bundle_timestamp:
                return False
            
            # Validate if enough time has passed (e.g., 30 minutes)
            time_since_last = datetime.now() - self.last_bundle_timestamp
            return time_since_last.total_seconds() > 1800  # 30 minutes
            
        except Exception as e:
            self.logger.debug(f"Error checking validation timing: {e}")
            return False
    
    async def _background_learning_tasks(self) -> None:
        """Run background learning tasks."""
        try:
            # Get learning integration
            integration = await get_eots_learning_integration(self.db_manager)
            
            # Auto-validate expired predictions
            validated_count = await integration.auto_validate_expired_predictions(24)
            
            if validated_count > 0:
                self.logger.info(f"🔄 Background validation completed: {validated_count} predictions")
            
        except Exception as e:
            self.logger.debug(f"Error in background learning tasks: {e}")

# ===== GLOBAL FUNCTIONS FOR DASHBOARD INTEGRATION =====

# Global instance
_eots_ai_learning_bridge = None

async def get_eots_ai_learning_bridge(db_manager=None) -> EOTSAILearningBridge:
    """Get global EOTS AI Learning Bridge instance."""
    global _eots_ai_learning_bridge
    if _eots_ai_learning_bridge is None:
        _eots_ai_learning_bridge = EOTSAILearningBridge(db_manager)
    return _eots_ai_learning_bridge

async def get_eots_learning_system_status(db_manager=None) -> Dict[str, Any]:
    """Get comprehensive learning system status for performance tracker integration."""
    try:
        bridge = await get_eots_ai_learning_bridge(db_manager)

        # Get learning manager status
        learning_manager = bridge.learning_manager

        # Get recent predictions count
        recent_predictions = await learning_manager.get_recent_predictions_count(hours=24)

        # Get learning statistics
        learning_stats = await learning_manager.get_learning_statistics()

        # Get last prediction ID
        last_prediction = await learning_manager.get_last_prediction()
        last_prediction_id = last_prediction.prediction_id if last_prediction else None

        return {
            "learning_enabled": True,
            "status": "active",
            "last_prediction_id": last_prediction_id,
            "recent_predictions_24h": recent_predictions,
            "learning_stats": {
                "patterns_learned": learning_stats.get("patterns_discovered", 0),
                "memory_nodes": learning_stats.get("memory_connections", 0),
                "adaptation_score": learning_stats.get("adaptation_velocity", 0.0) * 10,  # Scale to 0-10
                "success_rate": learning_stats.get("validation_accuracy", 0.0),
                "learning_velocity": learning_stats.get("learning_rate", 0.0)
            }
        }

    except Exception as e:
        logger.debug(f"Error getting learning system status: {e}")
        return {
            "learning_enabled": False,
            "status": "unavailable",
            "last_prediction_id": None,
            "recent_predictions_24h": 0,
            "learning_stats": None,
            "error": str(e)
        }


async def process_eots_dashboard_update_with_learning(bundle_data: FinalAnalysisBundleV2_5,
                                                    db_manager=None) -> Dict[str, Any]:
    """Process EOTS dashboard update with automatic learning integration."""
    try:
        bridge = await get_eots_ai_learning_bridge(db_manager)
        return await bridge.process_dashboard_update(bundle_data)
    except Exception as e:
        logger.error(f"Error processing dashboard update with learning: {e}")
        return {
            "prediction_recorded": False,
            "prediction_validated": False,
            "pattern_discovered": False,
            "learning_active": False,
            "error": str(e)
        }

async def get_eots_learning_system_status(db_manager=None) -> Dict[str, Any]:
    """Get EOTS learning system status."""
    try:
        bridge = await get_eots_ai_learning_bridge(db_manager)
        return await bridge.get_learning_status()
    except Exception as e:
        logger.error(f"Error getting learning system status: {e}")
        return {
            "learning_enabled": False,
            "status": "error",
            "error": str(e)
        }

async def enable_eots_learning_system(db_manager=None) -> bool:
    """Enable EOTS learning system."""
    try:
        bridge = await get_eots_ai_learning_bridge(db_manager)
        return await bridge.enable_learning()
    except Exception as e:
        logger.error(f"Error enabling learning system: {e}")
        return False

async def disable_eots_learning_system(db_manager=None) -> bool:
    """Disable EOTS learning system."""
    try:
        bridge = await get_eots_ai_learning_bridge(db_manager)
        return await bridge.disable_learning()
    except Exception as e:
        logger.error(f"Error disabling learning system: {e}")
        return False

# ===== DASHBOARD CALLBACK INTEGRATION =====

def integrate_learning_with_dashboard_callbacks():
    """
    Integration function to wire learning into dashboard callbacks.
    
    This function should be called during dashboard initialization to ensure
    that every dashboard update automatically triggers learning processes.
    """
    logger.info("🔗 Integrating EOTS AI Learning with dashboard callbacks")
    
    # This would integrate with the dashboard callback system
    # For now, it serves as a placeholder for future integration
    
    return True
