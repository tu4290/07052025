"""
EOTS Learning Integration v2.5 - "SENTIENT EOTS LEARNING BRIDGE"
================================================================

This module integrates the EOTS system with the Pydantic AI Learning Manager
to create a real learning loop that records predictions, validates outcomes,
and evolves the AI intelligence based on actual market performance.

Key Features:
- Automatic EOTS prediction recording for learning
- Real-time market outcome validation
- Pydantic AI-powered learning extraction
- Seamless integration with AI Learning Center
- Validated learning data with EOTS schema compliance

Author: EOTS v2.5 AI Intelligence Division
Version: 2.5.0 - "SENTIENT LEARNING BRIDGE"
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Pydantic imports
from pydantic import BaseModel, Field

# Import EOTS schemas
from data_models import FinalAnalysisBundleV2_5

# Import Pydantic AI Learning Manager
from .pydantic_ai_learning_manager_v2_5 import (
    get_pydantic_ai_learning_manager,
    record_eots_prediction_for_learning,
    validate_eots_prediction_for_learning
)

logger = logging.getLogger(__name__)

# ===== PYDANTIC MODELS FOR EOTS LEARNING INTEGRATION =====

class EOTSLearningContext(BaseModel):
    """Pydantic model for EOTS learning context."""
    symbol: str = Field(..., description="Trading symbol")
    current_regime: str = Field(..., description="Current market regime")
    signal_strength: float = Field(..., ge=0.0, le=5.0, description="Overall signal strength")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="System confidence")
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market conditions")
    eots_metrics: Dict[str, Any] = Field(default_factory=dict, description="EOTS metrics snapshot")

class EOTSPredictionOutcome(BaseModel):
    """Pydantic model for EOTS prediction outcome validation."""
    prediction_id: str = Field(..., description="Original prediction ID")
    actual_regime: str = Field(..., description="Actual market regime that occurred")
    regime_accuracy: float = Field(..., ge=0.0, le=1.0, description="Regime prediction accuracy")
    signal_performance: float = Field(..., ge=0.0, le=1.0, description="Signal performance score")
    market_events: List[str] = Field(default_factory=list, description="Actual market events")
    outcome_timestamp: datetime = Field(default_factory=datetime.now, description="When outcome was recorded")
    learning_feedback: Dict[str, Any] = Field(default_factory=dict, description="Learning feedback data")

# ===== EOTS LEARNING INTEGRATION CLASS =====

class EOTSLearningIntegration:
    """
    EOTS Learning Integration - Sentient Learning Bridge
    
    This class creates a seamless bridge between the EOTS system and the
    Pydantic AI Learning Manager, enabling real learning from market outcomes.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logger.getChild(self.__class__.__name__)
        self.active_predictions: Dict[str, EOTSLearningContext] = {}
        self.logger.info("🌉 EOTS Learning Integration initialized")
    
    async def record_eots_analysis_for_learning(self, bundle_data: FinalAnalysisBundleV2_5,
                                              ai_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Record EOTS analysis as a prediction for future learning validation."""
        try:
            # Extract learning context from bundle data
            context = self._extract_learning_context(bundle_data)
            
            # Prepare AI analysis data
            if not ai_analysis:
                ai_analysis = self._extract_ai_analysis_from_bundle(bundle_data)
            
            # Record prediction using Pydantic AI Learning Manager
            prediction_id = await record_eots_prediction_for_learning(
                bundle_data, ai_analysis, self.db_manager
            )
            
            if prediction_id:
                # Store context for future validation
                self.active_predictions[prediction_id] = context
                
                self.logger.info(f"📝 Recorded EOTS analysis for learning: {prediction_id}")
                
                # Trigger pattern discovery
                await self._trigger_pattern_discovery(bundle_data, context)
                
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Error recording EOTS analysis for learning: {e}")
            return ""
    
    async def validate_eots_prediction_outcome(self, prediction_id: str,
                                             current_bundle: FinalAnalysisBundleV2_5,
                                             time_horizon_minutes: int = 60) -> bool:
        """Validate EOTS prediction against current market state for learning."""
        try:
            if prediction_id not in self.active_predictions:
                self.logger.warning(f"Prediction {prediction_id} not found in active predictions")
                return False
            
            original_context = self.active_predictions[prediction_id]
            
            # Create outcome validation
            outcome = self._create_prediction_outcome(
                prediction_id, original_context, current_bundle
            )
            
            # Validate using Pydantic AI Learning Manager
            success = await validate_eots_prediction_for_learning(
                prediction_id, outcome.model_dump(), self.db_manager
            )
            
            if success:
                # Remove from active predictions
                del self.active_predictions[prediction_id]
                
                self.logger.info(f"✅ Validated EOTS prediction: {prediction_id}")
                
                # Trigger learning insight generation
                await self._trigger_learning_insight_generation(outcome)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error validating EOTS prediction outcome: {e}")
            return False
    
    async def auto_validate_expired_predictions(self, max_age_hours: int = 24) -> int:
        """Automatically validate predictions that have exceeded their time horizon."""
        try:
            validated_count = 0
            current_time = datetime.now()
            expired_predictions = []
            
            # Find expired predictions
            for prediction_id, context in self.active_predictions.items():
                # Check if prediction has expired (simplified logic)
                if prediction_id.startswith("eots_pred_"):
                    # Extract timestamp from prediction ID
                    timestamp_str = prediction_id.split("_")[2]  # eots_pred_YYYYMMDD_HHMMSS_...
                    try:
                        pred_time = datetime.strptime(timestamp_str, "%Y%m%d")
                        if (current_time - pred_time).total_seconds() > max_age_hours * 3600:
                            expired_predictions.append(prediction_id)
                    except ValueError:
                        # If we can't parse the timestamp, consider it expired
                        expired_predictions.append(prediction_id)
            
            # Validate expired predictions with neutral outcomes
            for prediction_id in expired_predictions:
                neutral_outcome = EOTSPredictionOutcome(
                    prediction_id=prediction_id,
                    actual_regime="NEUTRAL",
                    regime_accuracy=0.5,  # Neutral accuracy for expired predictions
                    signal_performance=0.5,
                    market_events=["Prediction expired without validation"],
                    learning_feedback={"validation_type": "auto_expired"}
                )
                
                success = await validate_eots_prediction_for_learning(
                    prediction_id, neutral_outcome.model_dump(), self.db_manager
                )
                
                if success:
                    validated_count += 1
                    if prediction_id in self.active_predictions:
                        del self.active_predictions[prediction_id]
            
            if validated_count > 0:
                self.logger.info(f"🔄 Auto-validated {validated_count} expired predictions")
            
            return validated_count
            
        except Exception as e:
            self.logger.error(f"Error auto-validating expired predictions: {e}")
            return 0
    
    def _extract_learning_context(self, bundle_data: FinalAnalysisBundleV2_5) -> EOTSLearningContext:
        """Extract learning context from EOTS bundle data."""
        try:
            # Get current regime (with fallback)
            current_regime = getattr(bundle_data, 'current_regime', 'UNKNOWN')
            if current_regime == 'UNKNOWN':
                # Try to extract from other bundle data
                if hasattr(bundle_data, 'regime_analysis'):
                    current_regime = getattr(bundle_data.regime_analysis, 'current_regime', 'UNKNOWN')
            
            # Calculate signal strength from EOTS metrics
            signal_strength = self._calculate_signal_strength(bundle_data)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(bundle_data)
            
            return EOTSLearningContext(
                symbol=bundle_data.target_symbol,
                current_regime=current_regime,
                signal_strength=signal_strength,
                confidence_level=confidence_level,
                market_conditions=self._extract_market_conditions(bundle_data),
                eots_metrics=bundle_data.model_dump()
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting learning context: {e}")
            # Return fallback context
            return EOTSLearningContext(
                symbol=bundle_data.target_symbol,
                current_regime="UNKNOWN",
                signal_strength=2.5,
                confidence_level=0.5,
                market_conditions={},
                eots_metrics={}
            )
    
    def _extract_ai_analysis_from_bundle(self, bundle_data: FinalAnalysisBundleV2_5) -> Dict[str, Any]:
        """Extract AI analysis data from bundle for prediction recording."""
        try:
            # Extract regime prediction
            predicted_regime = getattr(bundle_data, 'current_regime', 'UNKNOWN')
            
            # Extract confidence from various sources
            confidence = 0.5  # Default
            if hasattr(bundle_data, 'confidence_score'):
                confidence = bundle_data.confidence_score
            elif hasattr(bundle_data, 'overall_confidence'):
                confidence = bundle_data.overall_confidence
            
            # Extract insights from bundle data
            insights = []
            if hasattr(bundle_data, 'key_insights'):
                insights = bundle_data.key_insights
            elif hasattr(bundle_data, 'analysis_insights'):
                insights = bundle_data.analysis_insights
            
            return {
                "predicted_regime": predicted_regime,
                "confidence": confidence,
                "insights": insights,
                "analysis_type": "eots_bundle_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting AI analysis from bundle: {e}")
            return {
                "predicted_regime": "UNKNOWN",
                "confidence": 0.5,
                "insights": [],
                "analysis_type": "fallback_analysis"
            }

    def _calculate_signal_strength(self, bundle_data: FinalAnalysisBundleV2_5) -> float:
        """Calculate overall signal strength from EOTS metrics."""
        try:
            # Extract key metrics for signal strength calculation
            signal_factors = []

            # Check for tier 1 core metrics
            if hasattr(bundle_data, 'tier_1_core_metrics'):
                tier_1 = bundle_data.tier_1_core_metrics
                if tier_1:
                    # Use A-MSPI as primary signal strength indicator
                    a_mspi = getattr(tier_1, 'a_mspi_und', 0.0)
                    signal_factors.append(min(abs(a_mspi) / 2.0, 2.5))  # Normalize to 0-2.5

            # Check for tier 3 core metrics (VAPI-FA, DWFD, TW-LAF)
            if hasattr(bundle_data, 'tier_3_core_metrics'):
                tier_3 = bundle_data.tier_3_core_metrics
                if tier_3:
                    vapi_fa = abs(getattr(tier_3, 'vapi_fa_z_score_und', 0.0))
                    dwfd = abs(getattr(tier_3, 'dwfd_z_score_und', 0.0))
                    tw_laf = abs(getattr(tier_3, 'tw_laf_z_score_und', 0.0))

                    # Average of core signals
                    core_signal = (vapi_fa + dwfd + tw_laf) / 3.0
                    signal_factors.append(min(core_signal, 2.5))

            # Calculate overall signal strength
            if signal_factors:
                return min(sum(signal_factors) / len(signal_factors), 5.0)
            else:
                return 2.5  # Neutral signal strength

        except Exception as e:
            self.logger.debug(f"Error calculating signal strength: {e}")
            return 2.5  # Fallback

    def _calculate_confidence_level(self, bundle_data: FinalAnalysisBundleV2_5) -> float:
        """Calculate confidence level from EOTS analysis."""
        try:
            confidence_factors = []

            # Check for explicit confidence scores
            if hasattr(bundle_data, 'confidence_score'):
                confidence_factors.append(bundle_data.confidence_score)

            # Calculate confidence from signal confluence
            signal_strength = self._calculate_signal_strength(bundle_data)
            signal_confidence = min(signal_strength / 5.0, 1.0)
            confidence_factors.append(signal_confidence)

            # Calculate confidence from data completeness
            data_completeness = self._calculate_data_completeness(bundle_data)
            confidence_factors.append(data_completeness)

            # Return average confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5  # Neutral confidence

        except Exception as e:
            self.logger.debug(f"Error calculating confidence level: {e}")
            return 0.5  # Fallback

    def _calculate_data_completeness(self, bundle_data: FinalAnalysisBundleV2_5) -> float:
        """Calculate data completeness score."""
        try:
            completeness_score = 0.0
            total_checks = 0

            # Check for tier 1 metrics
            if hasattr(bundle_data, 'tier_1_core_metrics') and bundle_data.tier_1_core_metrics:
                completeness_score += 0.3
            total_checks += 1

            # Check for tier 3 metrics
            if hasattr(bundle_data, 'tier_3_core_metrics') and bundle_data.tier_3_core_metrics:
                completeness_score += 0.4
            total_checks += 1

            # Check for OHLCV data
            if hasattr(bundle_data, 'ohlcv_data') and bundle_data.ohlcv_data:
                completeness_score += 0.2
            total_checks += 1

            # Check for news intelligence
            if hasattr(bundle_data, 'news_intelligence_v2_5') and bundle_data.news_intelligence_v2_5:
                completeness_score += 0.1
            total_checks += 1

            return completeness_score

        except Exception as e:
            self.logger.debug(f"Error calculating data completeness: {e}")
            return 0.5  # Fallback

    def _extract_market_conditions(self, bundle_data: FinalAnalysisBundleV2_5) -> Dict[str, Any]:
        """Extract market conditions from bundle data."""
        try:
            conditions = {
                "symbol": bundle_data.target_symbol,
                "timestamp": datetime.now().isoformat()
            }

            # Add OHLCV data if available
            if hasattr(bundle_data, 'ohlcv_data') and bundle_data.ohlcv_data:
                ohlcv = bundle_data.ohlcv_data
                conditions["price"] = getattr(ohlcv, 'close', 0.0)
                conditions["volume"] = getattr(ohlcv, 'volume', 0)

            # Add volatility information
            if hasattr(bundle_data, 'tier_1_core_metrics') and bundle_data.tier_1_core_metrics:
                tier_1 = bundle_data.tier_1_core_metrics
                conditions["atr"] = getattr(tier_1, 'atr_und', 0.0)

            # Add regime information
            conditions["regime"] = getattr(bundle_data, 'current_regime', 'UNKNOWN')

            return conditions

        except Exception as e:
            self.logger.debug(f"Error extracting market conditions: {e}")
            return {"symbol": bundle_data.target_symbol}

    def _create_prediction_outcome(self, prediction_id: str,
                                 original_context: EOTSLearningContext,
                                 current_bundle: FinalAnalysisBundleV2_5) -> EOTSPredictionOutcome:
        """Create prediction outcome for validation."""
        try:
            # Get current regime
            actual_regime = getattr(current_bundle, 'current_regime', 'UNKNOWN')

            # Calculate regime accuracy
            regime_accuracy = 1.0 if original_context.current_regime == actual_regime else 0.3

            # Calculate signal performance
            current_signal_strength = self._calculate_signal_strength(current_bundle)
            signal_performance = min(current_signal_strength / 5.0, 1.0)

            # Extract market events
            market_events = self._extract_market_events(original_context, current_bundle)

            return EOTSPredictionOutcome(
                prediction_id=prediction_id,
                actual_regime=actual_regime,
                regime_accuracy=regime_accuracy,
                signal_performance=signal_performance,
                market_events=market_events,
                learning_feedback={
                    "original_signal_strength": original_context.signal_strength,
                    "current_signal_strength": current_signal_strength,
                    "original_confidence": original_context.confidence_level,
                    "validation_type": "real_time"
                }
            )

        except Exception as e:
            self.logger.error(f"Error creating prediction outcome: {e}")
            # Return neutral outcome
            return EOTSPredictionOutcome(
                prediction_id=prediction_id,
                actual_regime="NEUTRAL",
                regime_accuracy=0.5,
                signal_performance=0.5,
                market_events=["Error creating outcome"],
                learning_feedback={"validation_type": "error_fallback"}
            )

    def _extract_market_events(self, original_context: EOTSLearningContext,
                             current_bundle: FinalAnalysisBundleV2_5) -> List[str]:
        """Extract market events that occurred between prediction and validation."""
        try:
            events = []

            # Compare regimes
            if original_context.current_regime != getattr(current_bundle, 'current_regime', 'UNKNOWN'):
                events.append(f"Regime changed from {original_context.current_regime} to {getattr(current_bundle, 'current_regime', 'UNKNOWN')}")

            # Compare signal strengths
            current_signal = self._calculate_signal_strength(current_bundle)
            if abs(current_signal - original_context.signal_strength) > 1.0:
                events.append(f"Signal strength changed from {original_context.signal_strength:.1f} to {current_signal:.1f}")

            # Add news events if available
            if hasattr(current_bundle, 'news_intelligence_v2_5') and current_bundle.news_intelligence_v2_5:
                news = current_bundle.news_intelligence_v2_5
                if isinstance(news, dict) and news.get('article_count', 0) > 5:
                    events.append(f"High news activity: {news.get('article_count', 0)} articles")

            return events if events else ["No significant market events detected"]

        except Exception as e:
            self.logger.debug(f"Error extracting market events: {e}")
            return ["Error extracting market events"]

    async def _trigger_pattern_discovery(self, bundle_data: FinalAnalysisBundleV2_5,
                                       context: EOTSLearningContext) -> None:
        """Trigger pattern discovery using Pydantic AI."""
        try:
            manager = await get_pydantic_ai_learning_manager(self.db_manager)

            # Prepare market data for pattern discovery
            market_data = {
                "signal_strength": context.signal_strength,
                "confidence_level": context.confidence_level,
                "market_conditions": context.market_conditions,
                "regime": context.current_regime
            }

            # Trigger pattern discovery
            pattern = await manager.discover_pattern(market_data, bundle_data)
            if pattern:
                self.logger.info(f"🔍 Discovered pattern during EOTS analysis: {pattern.pattern_name}")

        except Exception as e:
            self.logger.debug(f"Error triggering pattern discovery: {e}")

    async def _trigger_learning_insight_generation(self, outcome: EOTSPredictionOutcome) -> None:
        """Trigger learning insight generation from prediction outcome."""
        try:
            # This would trigger insight generation in the Pydantic AI Learning Manager
            # For now, just log the successful validation
            self.logger.info(f"💡 Learning insight opportunity from prediction {outcome.prediction_id}")

        except Exception as e:
            self.logger.debug(f"Error triggering learning insight generation: {e}")

# ===== GLOBAL FUNCTIONS FOR INTEGRATION =====

# Global instance
_eots_learning_integration = None

async def get_eots_learning_integration(db_manager=None) -> EOTSLearningIntegration:
    """Get global EOTS Learning Integration instance."""
    global _eots_learning_integration
    if _eots_learning_integration is None:
        _eots_learning_integration = EOTSLearningIntegration(db_manager)
    return _eots_learning_integration

async def record_eots_bundle_for_learning(bundle_data: FinalAnalysisBundleV2_5,
                                        db_manager=None) -> str:
    """Record EOTS bundle analysis for learning (Global Function)."""
    try:
        integration = await get_eots_learning_integration(db_manager)
        return await integration.record_eots_analysis_for_learning(bundle_data)
    except Exception as e:
        logger.error(f"Error recording EOTS bundle for learning: {e}")
        return ""

async def validate_eots_learning_prediction(prediction_id: str,
                                          current_bundle: FinalAnalysisBundleV2_5,
                                          db_manager=None) -> bool:
    """Validate EOTS learning prediction (Global Function)."""
    try:
        integration = await get_eots_learning_integration(db_manager)
        return await integration.validate_eots_prediction_outcome(prediction_id, current_bundle)
    except Exception as e:
        logger.error(f"Error validating EOTS learning prediction: {e}")
        return False
