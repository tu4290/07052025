from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from data_models.ai_ml_models import (
    AIPredictionV2_5,
    AIPredictionRequestV2_5,
    AIPredictionPerformanceV2_5,
    AIPredictionSummaryV2_5
)
from data_models.configuration_models import PredictionConfigV2_5
from data_models import ProcessedUnderlyingAggregatesV2_5

class AIPredictionMetricsV2_5(BaseModel):
    """Pydantic model for tracking AI prediction metrics."""
    total_predictions: int = Field(default=0, ge=0)
    successful_predictions: int = Field(default=0, ge=0)
    failed_predictions: int = Field(default=0, ge=0)
    average_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_cycles_completed: int = Field(default=0, ge=0)
    total_processing_time_ms: float = Field(default=0.0, ge=0.0)
    
    class Config:
        extra = 'forbid'

class AIPredictionsManagerV2_5:
    """
    AI Predictions Manager for Elite Options Trading System v2.5
    
    This class manages the AI prediction process, generating and tracking
    market predictions using advanced machine learning models.
    """
    
    def __init__(self, config: Optional[PredictionConfigV2_5] = None):
        """Initialize the AI predictions manager."""
        self.config = config or PredictionConfigV2_5(
            enabled=True,
            model_name="default_prediction_model",
            prediction_interval_seconds=300,
            max_data_age_seconds=120,
            success_threshold=0.7
        )
        self.metrics = AIPredictionMetricsV2_5(
            total_predictions=0,
            successful_predictions=0,
            failed_predictions=0,
            average_confidence_score=0.0,
            prediction_cycles_completed=0,
            total_processing_time_ms=0.0
        )
        self.predictions: List[AIPredictionV2_5] = []
        self.start_time = datetime.now()
        
    def generate_prediction(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> Optional[AIPredictionV2_5]:
        """Generate a new market prediction."""
        if not self.config.enabled:
            return None

        signal_strength = self.calculate_ai_prediction_signal_strength(market_data)
        if signal_strength is None:
            return None

        prediction = AIPredictionV2_5(
            symbol=market_data.symbol,
            prediction_type="eots_direction",
            prediction_value=signal_strength,
            prediction_direction="UP" if signal_strength > 0 else "DOWN" if signal_strength < 0 else "NEUTRAL",
            confidence_score=min(0.85, max(0.5, abs(signal_strength) / 3.0)),
            time_horizon="1D",
            target_timestamp=datetime.utcnow() + timedelta(days=1),
            model_name=self.config.model_name,
            market_context={"signal_strength": signal_strength},
            actual_value=None,
            actual_direction=None,
            prediction_accurate=None,
            accuracy_score=None
        )

        self.predictions.append(prediction)
        return prediction
        
    def _validate_prediction_criteria(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate if market data meets prediction criteria.
        """
        # Check data completeness
        required_fields = ['price', 'volume', 'indicators']
        if not all(field in market_data for field in required_fields):
            return False
            
        # Check data freshness
        if 'timestamp' in market_data:
            data_age = (datetime.now() - market_data['timestamp']).total_seconds()
            if data_age > self.config.max_data_age_seconds:
                return False
                
        return True
        
    def _create_prediction(self, market_data: Dict[str, Any]) -> AIPredictionV2_5:
        """Create a market prediction from input data."""
        # Use the new signal strength calculation
        signal_strength_metrics = self.calculate_ai_prediction_signal_strength(market_data)

        return AIPredictionV2_5(
            prediction_id=str(uuid.uuid4()),
            symbol=market_data.get('symbol', 'UNKNOWN'),
            prediction_type="market_direction",
            confidence_score=signal_strength_metrics['confidence_score'],
            prediction_horizon="short_term",
            market_context={
                "price": market_data.get('price'),
                "volume": market_data.get('volume'),
                "indicators": market_data.get('indicators', {}),
                "prediction_direction": signal_strength_metrics['prediction_direction'],
                "direction_confidence": signal_strength_metrics['direction_confidence'],
                "signal_strength": signal_strength_metrics['signal_strength']
            },
            prediction_timestamp=datetime.now()
        )
        
    def _calculate_confidence(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a prediction.
        This method now primarily calls calculate_ai_prediction_signal_strength.
        """
        signal_strength_metrics = self.calculate_ai_prediction_signal_strength(market_data)
        return signal_strength_metrics['confidence_score']
        
    def _check_data_quality(self, market_data: Dict[str, Any]) -> bool:
        """Check quality of input market data."""
        # Implement data quality checks
        return all([
            'price' in market_data and isinstance(market_data['price'], (int, float)),
            'volume' in market_data and isinstance(market_data['volume'], (int, float)),
            'indicators' in market_data and isinstance(market_data['indicators'], dict)
        ])
        
    def _check_favorable_conditions(self, market_data: Dict[str, Any]) -> bool:
        """Check if market conditions are favorable for prediction."""
        # Implement market condition checks
        return True  # Placeholder
        
    def update_prediction_performance(self, prediction_id: str, actual_outcome: Dict[str, Any]):
        """
        Update performance metrics for a prediction.
        """
        try:
            # Find prediction
            prediction = next((p for p in self.predictions if p.prediction_id == prediction_id), None)
            if not prediction:
                return
                
            # Create performance record
            # Update metrics directly in self.metrics
            self.metrics.prediction_cycles_completed += 1
            self.metrics.total_processing_time_ms += (datetime.now() - prediction.prediction_timestamp).total_seconds() * 1000

            # No need to create a separate performance object if we are updating metrics directly
            # The performance score calculation can be used to update successful/failed predictions
            performance_score = self._calculate_performance_score(prediction, actual_outcome)

            if performance_score >= self.config.success_threshold:
                self.metrics.successful_predictions += 1
            else:
                self.metrics.failed_predictions += 1

            self.metrics.total_predictions = self.metrics.successful_predictions + self.metrics.failed_predictions

                
            # Update average confidence
            self._update_average_confidence()
            
        except Exception:
            # Log error but continue
            pass
            
    def _calculate_performance_score(self, prediction: AIPredictionV2_5, actual_outcome: Dict[str, Any]) -> float:
        """
        Calculate performance score for a prediction.
        """
        try:
            # Implement performance scoring logic
            predicted_direction = prediction.market_context.get('predicted_direction')
            actual_direction = actual_outcome.get('actual_direction')
            
            if predicted_direction == actual_direction:
                return 1.0
            return 0.0
            
        except Exception:
            return 0.0
            
    def _update_average_confidence(self):
        """
        Update average confidence score metric.
        """
        if not self.predictions:
            return
            
        total_confidence = sum(p.confidence_score for p in self.predictions)
        self.metrics.average_confidence_score = total_confidence / len(self.predictions)
        
    def get_prediction_summary(self) -> AIPredictionSummaryV2_5:
        """
        Generate a summary of prediction performance.
        """
        return AIPredictionSummaryV2_5(
             symbol="SYSTEM",
             analysis_timestamp=datetime.utcnow(),
             latest_prediction=self.predictions[-1] if self.predictions else None,
             active_predictions_count=len([p for p in self.predictions if p.prediction_accurate is None]),
             pending_predictions_count=len([p for p in self.predictions if p.prediction_accurate is None]),
             prediction_quality_score=self._calculate_accuracy(),
             confidence_score=self.metrics.average_confidence_score,
             optimization_recommendations=[],
             metadata={}
)
        
    def _calculate_accuracy(self) -> float:
        """
        Calculate overall prediction accuracy.
        """
        if not self.metrics.total_predictions:
            return 0.0
        return self.metrics.successful_predictions / self.metrics.total_predictions
        
    def _get_recent_predictions(self) -> List[Dict[str, Any]]:
        """
        Get most recent predictions.
        """
        sorted_predictions = sorted(
            self.predictions,
            key=lambda x: x.prediction_timestamp,
            reverse=True
        )
        return [pred.model_dump() for pred in sorted_predictions[:5]]
        
    def _calculate_performance_trend(self) -> str:
        """
        Calculate the trend in prediction performance.
        """
        if len(self.predictions) < 2:
            return "STABLE"
            
        recent_scores = [p.confidence_score for p in self.predictions[-5:]]
        if not recent_scores:
            return "STABLE"
            
        recent_avg = sum(recent_scores) / len(recent_scores)
        overall_avg = self.metrics.average_confidence_score
        
        if recent_avg > overall_avg * 1.1:
            return "IMPROVING"
        elif recent_avg < overall_avg * 0.9:
            return "DECLINING"
        return "STABLE"
        
    def _calculate_next_cycle(self) -> datetime:
        """
        Calculate the next prediction cycle timestamp.
        """
        return datetime.now() + timedelta(minutes=15)  # Default to 15-minute cycles

    def calculate_ai_prediction_signal_strength(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> Optional[float]:
        """Calculate the signal strength for AI prediction."""
        try:
            # Use the elite impact score as the primary signal
            signal = market_data.elite_impact_score_und
            if signal is None:
                # Fallback to a combination of flow metrics
                signal = (
                    (market_data.vapi_fa_z_score_und or 0) +
                    (market_data.dwfd_z_score_und or 0) +
                    (market_data.tw_laf_z_score_und or 0)
                ) / 3.0
            return float(signal)
        except (ValueError, TypeError, AttributeError):
            return None


# API compatibility functions
def get_ai_predictions_manager(config: Optional[PredictionConfigV2_5] = None) -> AIPredictionsManagerV2_5:
    """Get AI predictions manager instance."""
    return AIPredictionsManagerV2_5(config)

def generate_market_prediction(market_data: ProcessedUnderlyingAggregatesV2_5, config: Optional[PredictionConfigV2_5] = None) -> Optional[AIPredictionV2_5]:
    """Generate a market prediction using the AI predictions manager."""
    manager = get_ai_predictions_manager(config)
    return manager.generate_prediction(market_data)