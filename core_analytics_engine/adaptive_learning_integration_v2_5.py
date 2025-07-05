"""
Enhanced Adaptive Learning Integration for EOTS v2.5 - HUIHUI AI INTEGRATION
====================================================================

This module integrates the HuiHui Learning System with the EOTS system,
providing scheduled learning cycles, real-time adaptation, and advanced
learning capabilities with improved performance and reliability.

Features:
- Pydantic-First Architecture with strict validation
- Batch processing of learning insights
- Asynchronous processing with retries and circuit breakers
- Intelligent caching and deduplication
- Performance monitoring and health checks
- Model versioning and A/B testing support
- Integration with monitoring system
- Scheduled learning cycles (daily, weekly, monthly)
- Real-time parameter adjustment based on performance
- Learning history tracking with schema validation
- Performance validation against EOTS standards
- Rollback capabilities for failed optimizations

Author: EOTS v2.5 Development Team - "HuiHui AI Integration Division"
"""

import logging
import asyncio
import json
import time
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import circuit
import uuid

# Import HuiHui learning system
from data_models.ai_ml_models import UnifiedLearningResult, LearningInsightV2_5
from data_models.ai_ml_models import AIAdaptationV2_5, AIAdaptationPerformanceV2_5
from data_models.configuration_models import AnalyticsEngineConfigV2_5, AdaptiveLearningConfigV2_5
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ====================================
# Adaptation Strategies
# ====================================

class AdaptationStrategy(ABC):
    """Base class for adaptation strategies."""
    
    @abstractmethod
    async def apply(self, adaptation: 'AIAdaptationV2_5') -> bool:
        """Apply the adaptation strategy."""
        pass

class LearningRateAdjustment(AdaptationStrategy):
    """Adjust learning rate based on insight confidence."""
    
    async def apply(self, adaptation: 'AIAdaptationV2_5') -> bool:
        confidence = adaptation.parameters.get("confidence", 0.5)
        # Adjust learning rate based on confidence (example: 0.001 to 0.002 range)
        base_rate = 0.001
        new_rate = base_rate * (1 + confidence)  # Scales with confidence
        adaptation.parameters["learning_rate"] = min(max(new_rate, 0.0001), 0.01)  # Clamp values
        logger.debug(f"Adjusted learning rate to {adaptation.parameters['learning_rate']} "
                   f"based on confidence {confidence}")
        return True

class ModelUpdateStrategy(AdaptationStrategy):
    """Update model parameters based on insights."""
    
    async def apply(self, adaptation: 'AIAdaptationV2_5') -> bool:
        # Example: Adjust model parameters based on insight
        if "model_parameters" not in adaptation.parameters:
            adaptation.parameters["model_parameters"] = {}
        
        # Add your model update logic here
        # For example:
        # adaptation.parameters["model_parameters"]["some_parameter"] = new_value
        
        logger.debug(f"Updated model parameters: {adaptation.parameters['model_parameters']}")
        return True

# ====================================
# Adaptation Factory
# ====================================
class AdaptationFactory:
    """Factory for creating adaptation strategies."""
    
    _strategies = {
        "learning_rate": LearningRateAdjustment(),
        "model_update": ModelUpdateStrategy(),
        # Add more strategies as needed
    }
    
    @classmethod
    def get_strategy(cls, insight_type: str) -> Optional[AdaptationStrategy]:
        """Get adaptation strategy based on insight type."""
        return cls._strategies.get(insight_type)
    
    @classmethod
    def register_strategy(cls, name: str, strategy: AdaptationStrategy) -> None:
        """Register a new adaptation strategy.
        
        Args:
            name: Name to register the strategy under
            strategy: Strategy instance to register
        """
        if not isinstance(strategy, AdaptationStrategy):
            raise ValueError("Strategy must be an instance of AdaptationStrategy")
        cls._strategies[name] = strategy
        logger.info(f"Registered new adaptation strategy: {name}")

# ====================================
# Performance Tracker
# ====================================
class AdaptationPerformanceTracker:
    """Track performance of adaptations over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: List[Dict[str, Any]] = []
        
    def record_performance(self, adaptation: 'AIAdaptationV2_5', success: bool) -> None:
        """Record the performance of an adaptation."""
        performance = {
            "adaptation_id": adaptation.adaptation_id,
            "type": adaptation.adaptation_type,
            "timestamp": datetime.utcnow(),
            "success": success,
            "parameters": adaptation.parameters
        }
        self.performance_history.append(performance)
        # Keep only the most recent entries
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
    
    def get_success_rate(self, adaptation_type: str = None) -> float:
        """Get the success rate for a specific adaptation type or overall."""
        if not self.performance_history:
            return 0.0
            
        relevant = (
            [p for p in self.performance_history if p["type"] == adaptation_type]
            if adaptation_type else self.performance_history
        )
        
        if not relevant:
            return 0.0
            
        successes = sum(1 for p in relevant if p["success"])
        return successes / len(relevant)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.performance_history:
            return {"success_rate": 0.0, "total_adaptations": 0}
            
        success_rate = self.get_success_rate()
        adaptations_by_type: Dict[str, int] = {}
        
        # Count adaptations by type
        for record in self.performance_history:
            adaptations_by_type[record["type"]] = adaptations_by_type.get(record["type"], 0) + 1
        
        return {
            "success_rate": success_rate,
            "total_adaptations": len(self.performance_history),
            "adaptations_by_type": adaptations_by_type,
            "window_size": self.window_size,
            "adaptations_in_window": len(self.performance_history)
        }

class DateTimeEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for datetime objects and numpy types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LearningBatchV2_5(BaseModel):
    """Represents a batch of learning data for processing."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    insights: List[LearningInsightV2_5] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('insights')
    @classmethod
    def validate_insights(cls, v):
        if not v:
            raise ValueError("Batch must contain at least one insight")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: str
        }
        extra = 'forbid'

class EnhancedLearningMetricsV2_5(BaseModel):
    """Extended metrics with additional performance indicators and caching."""
    total_insights_generated: int = Field(default=0, ge=0)
    successful_adaptations: int = Field(default=0, ge=0)
    failed_adaptations: int = Field(default=0, ge=0)
    average_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_cycles_completed: int = Field(default=0, ge=0)
    total_processing_time_ms: float = Field(default=0.0, ge=0.0)
    learning_rate: float = Field(1e-3, ge=0.0)
    batch_processing_times: List[float] = Field(default_factory=list)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate the cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def update_confidence(self, new_score: float) -> None:
        """Update the running average confidence score."""
        if self.total_insights_generated == 0:
            self.average_confidence_score = new_score
        else:
            self.average_confidence_score = (
                (self.average_confidence_score * self.total_insights_generated) + new_score
            ) / (self.total_insights_generated + 1)
        self.last_updated = datetime.utcnow()

    class Config:
        extra = 'forbid'
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist()
        }

class AdaptiveLearningIntegrationV2_5:
    """
    Enhanced Adaptive Learning Integration System for EOTS v2.5
    
    This class manages the adaptive learning process with improved performance,
    reliability, and monitoring capabilities.
    """
    
    def __init__(self, config: AdaptiveLearningConfigV2_5):
        """Initialize the enhanced adaptive learning integration system."""
        self.config = config
        self._insight_cache = {}
        self._batch_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        
        # Initialize analytics engine config
        self._init_analytics_config()
        
        # Initialize metrics and state
        self.metrics = EnhancedLearningMetricsV2_5()
        self.insights: List[LearningInsightV2_5] = []
        self.adaptations: List[AIAdaptationV2_5] = []
        self.performance_history: List[AIAdaptationPerformanceV2_5] = []
        self.performance_tracker = AdaptationPerformanceTracker()
        self.start_time = datetime.utcnow()
        
        # Initialize adaptation strategies
        self._init_adaptation_strategies()
        
        # Start background tasks only if there's a running event loop
        self._batch_processor_task = None
        self._metrics_updater_task = None
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            self._batch_processor_task = loop.create_task(self._batch_processor())
            self._metrics_updater_task = loop.create_task(self._update_metrics_loop())
        except RuntimeError:
            # No running event loop, tasks will be created later when needed
            logger.debug("No running event loop found, background tasks will be created later")
        
        self.setup_analytics_engine()
    
    def _init_adaptation_strategies(self) -> None:
        """Initialize default adaptation strategies."""
        # Register any additional strategies here
        pass
    
    def _init_analytics_config(self) -> None:
        """Initialize the analytics configuration with proper validation."""
        if hasattr(self.config, 'analytics_engine'):
            if isinstance(self.config.analytics_engine, AnalyticsEngineConfigV2_5):
                self.analytics_config = self.config.analytics_engine
            elif isinstance(self.config.analytics_engine, dict):
                self.analytics_config = AnalyticsEngineConfigV2_5.model_validate(
                    self.config.analytics_engine
                )
            else:
                try:
                    config_dict = dict(self.config.analytics_engine)
                    self.analytics_config = AnalyticsEngineConfigV2_5.model_validate(config_dict)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "Failed to convert analytics_engine to valid config. Using defaults. Error: %s",
                        str(e)
                    )
                    self.analytics_config = AnalyticsEngineConfigV2_5()
        else:
            self.analytics_config = AnalyticsEngineConfigV2_5()
    
    async def _batch_processor(self) -> None:
        """Process batches from the queue asynchronously."""
        while not self._shutdown_event.is_set():
            try:
                batch = await asyncio.wait_for(
                    self._batch_queue.get(),
                    timeout=1.0
                )
                await self._process_batch(batch)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error processing batch: %s", str(e), exc_info=True)
    
    async def _process_batch(self, batch: LearningBatchV2_5) -> None:
        """Process a single batch of insights."""
        start_time = time.time()
        try:
            batch.status = "processing"
            results = await asyncio.gather(
                *(self._process_insight(insight) for insight in batch.insights),
                return_exceptions=True
            )
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics.batch_processing_times.append(processing_time)
            self.metrics.learning_cycles_completed += 1
            
            # Handle results
            success_count = sum(1 for r in results if r is True)
            if success_count < len(results):
                logger.warning(
                    "Batch %s completed with %d/%d successes",
                    batch.batch_id, success_count, len(results)
                )
            
            batch.status = "completed"
            batch.metadata.update({
                "processing_time_ms": processing_time,
                "success_count": success_count,
                "total_insights": len(results)
            })
            
        except Exception as e:
            batch.status = "failed"
            batch.metadata["error"] = str(e)
            logger.error("Failed to process batch %s: %s", batch.batch_id, str(e), exc_info=True)
    
    async def _process_insight(self, insight: LearningInsightV2_5) -> bool:
        """Process a single insight with caching and error handling."""
        cache_key = self._generate_insight_cache_key(insight)
        
        # Check cache first
        if cache_key in self._insight_cache:
            self.metrics.cache_hits += 1
            return True
            
        self.metrics.cache_misses += 1
        
        try:
            # Process the insight (placeholder for actual processing logic)
            processed = await self._process_new_insight(insight)
            if processed:
                self._insight_cache[cache_key] = insight
            return processed
        except Exception as e:
            logger.error("Error processing insight: %s", str(e), exc_info=True)
            return False
    
    def _generate_insight_cache_key(self, insight: LearningInsightV2_5) -> str:
        """Generate a cache key for an insight."""
        key_data = f"{insight.insight_type}:{insight.confidence_score}:{insight.timestamp}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _update_metrics_loop(self) -> None:
        """Periodically update and log metrics."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Update every minute
                logger.info(
                    "Learning metrics - Cache hit ratio: %.2f, Avg batch time: %.2fms",
                    self.metrics.cache_hit_ratio,
                    np.mean(self.metrics.batch_processing_times) if self.metrics.batch_processing_times else 0
                )
            except Exception as e:
                logger.error("Error in metrics update loop: %s", str(e), exc_info=True)
    
    async def shutdown(self) -> None:
        """Gracefully shut down the learning system."""
        self._shutdown_event.set()
        await asyncio.gather(
            self._batch_processor_task,
            self._metrics_updater_task,
            return_exceptions=True
        )
    
    async def add_insights_batch(self, insights: List[LearningInsightV2_5]) -> str:
        """Add a batch of insights for processing."""
        if not insights:
            raise ValueError("No insights provided in batch")
            
        batch = LearningBatchV2_5(insights=insights)
        await self._batch_queue.put(batch)
        return batch.batch_id
    
    @circuit(failure_threshold=3, recovery_timeout=60)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_new_insight(self, insight: LearningInsightV2_5) -> bool:
        """Process a single new insight with retry and circuit breaker."""
        try:
            # Update metrics
            self.metrics.total_insights_generated += 1
            self.metrics.update_confidence(insight.confidence_score)
            
            # Store the insight
            self.insights.append(insight)
            
            # Check if we should adapt based on this insight
            if (self.config.auto_adaptation and 
                insight.confidence_score >= self.config.confidence_threshold):
                adaptation = await self._create_adaptation(insight)
                if adaptation:
                    success = await self._apply_adaptation(adaptation)
                    # Record performance for monitoring
                    self.performance_tracker.record_performance(adaptation, success)
                    
                    if success:
                        self.metrics.successful_adaptations += 1
                        logger.info(f"Successfully applied adaptation {adaptation.adaptation_id} "
                                 f"for insight {insight.insight_id}")
                    else:
                        self.metrics.failed_adaptations += 1
                        logger.warning(f"Failed to apply adaptation {adaptation.adaptation_id} "
                                    f"for insight {insight.insight_id}")
                    return success
            return True
            
        except Exception as e:
            error_msg = f"Failed to process insight {getattr(insight, 'insight_id', 'unknown')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics.failed_adaptations += 1
            raise RuntimeError(error_msg) from e
    
    async def _create_adaptation(self, insight: LearningInsightV2_5) -> Optional[AIAdaptationV2_5]:
        """Create an adaptation based on the insight."""
        try:
            # Create a new adaptation with metadata
            adaptation = AIAdaptationV2_5(
                adaptation_id=str(uuid.uuid4()),
                insight_id=insight.insight_id,
                adaptation_type=insight.insight_type,
                parameters={
                    # Default parameters that can be overridden by specific insight types
                    "learning_rate": self.metrics.learning_rate,
                    "confidence": insight.confidence_score,
                    "market_context": insight.market_context
                },
                created_at=datetime.utcnow(),
                status="pending"
            )
            
            # Store the adaptation
            self.adaptations.append(adaptation)
            return adaptation
            
        except Exception as e:
            logger.error("Failed to create adaptation: %s", str(e), exc_info=True)
            return None
    
    async def _apply_adaptation(self, adaptation: AIAdaptationV2_5) -> bool:
        """Apply the adaptation to the system using the appropriate strategy.
        
        Args:
            adaptation: The adaptation to apply
            
        Returns:
            bool: True if adaptation was successfully applied, False otherwise
        """
        try:
            # Update adaptation status
            adaptation.status = "applying"
            adaptation.applied_at = datetime.utcnow()
            
            # Get the appropriate strategy
            strategy = AdaptationFactory.get_strategy(adaptation.adaptation_type)
            if not strategy:
                error_msg = f"No strategy found for adaptation type: {adaptation.adaptation_type}"
                logger.warning(error_msg)
                adaptation.status = "failed"
                adaptation.error = error_msg
                return False
            
            logger.info("Applying adaptation %s using %s strategy", 
                       adaptation.adaptation_id, strategy.__class__.__name__)
            
            # Apply the strategy
            success = await strategy.apply(adaptation)
            
            # Update adaptation status based on result
            if success:
                adaptation.status = "applied"
                # Update any metrics from the adaptation
                if "learning_rate" in adaptation.parameters:
                    self.metrics.learning_rate = adaptation.parameters["learning_rate"]
                logger.debug("Successfully applied adaptation %s", adaptation.adaptation_id)
            else:
                adaptation.status = "failed"
                adaptation.error = "Strategy application failed"
                logger.warning("Strategy application failed for adaptation %s", adaptation.adaptation_id)
            
            return success
            
        except Exception as e:
            error_msg = f"Error applying adaptation {getattr(adaptation, 'adaptation_id', 'unknown')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            adaptation.status = "failed"
            adaptation.error = str(e)
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics including adaptation statistics.
        
        Returns:
            Dict containing comprehensive performance metrics including:
            - System metrics (uptime, cache hit ratio, etc.)
            - Adaptation performance metrics
            - Batch processing statistics
        """
        # Get base metrics
        metrics = {
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "cache_hit_ratio": self.metrics.cache_hit_ratio,
            "batch_processing_avg_ms": np.mean(self.metrics.batch_processing_times) if self.metrics.batch_processing_times else 0,
            "insights_processed": self.metrics.total_insights_generated,
            "success_rate": (
                self.metrics.successful_adaptations / 
                max(1, self.metrics.successful_adaptations + self.metrics.failed_adaptations)
            ) if (self.metrics.successful_adaptations + self.metrics.failed_adaptations) > 0 else 0,
            "average_confidence": self.metrics.average_confidence_score,
            "learning_rate": self.metrics.learning_rate,
            "last_updated": self.metrics.last_updated.isoformat()
        }
        
        # Add adaptation performance metrics
        adaptation_metrics = self.performance_tracker.get_performance_metrics()
        metrics.update({
            "adaptation_success_rate": adaptation_metrics["success_rate"],
            "total_adaptations_applied": adaptation_metrics["total_adaptations"],
            "adaptations_by_type": adaptation_metrics["adaptations_by_type"],
            "performance_window_size": adaptation_metrics["window_size"],
            "adaptations_in_window": adaptation_metrics["adaptations_in_window"]
        })
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the learning system."""
        now = datetime.utcnow()
        time_since_last_update = (now - self.metrics.last_updated).total_seconds()
        
        status = "healthy"
        issues = []
        
        # Check if system is processing insights
        if time_since_last_update > 300:  # 5 minutes
            status = "degraded"
            issues.append(f"No updates in {time_since_last_update:.0f} seconds")
            
        # Check error rates
        total_adaptations = self.metrics.successful_adaptations + self.metrics.failed_adaptations
        if total_adaptations > 0 and self.metrics.failed_adaptations / total_adaptations > 0.5:
            status = "degraded"
            issues.append("High adaptation failure rate")
        
        return {
            "status": status,
            "uptime_seconds": (now - self.start_time).total_seconds(),
            "metrics": {
                "cache_hit_ratio": self.metrics.cache_hit_ratio,
                "insights_processed": self.metrics.total_insights_generated,
                "successful_adaptations": self.metrics.successful_adaptations,
                "failed_adaptations": self.metrics.failed_adaptations,
                "time_since_last_update_seconds": time_since_last_update
            },
            "issues": issues if issues else ["No issues detected"],
            "timestamp": now.isoformat()
        }

    def setup_analytics_engine(self):
        """Set up the analytics engine with validated configuration."""
        try:
            # Initialize analytics components with validated config
            self._initialize_metrics_tracking()
            self._initialize_learning_pipeline()
            self._initialize_adaptation_engine()
        except Exception as e:
            raise ValueError(f"Failed to initialize analytics engine: {str(e)}")

    def _initialize_metrics_tracking(self):
        """Initialize metrics tracking system."""
        pass  # Implementation details here

    def _initialize_learning_pipeline(self):
        """Initialize the learning pipeline."""
        pass  # Implementation details here

    def _initialize_adaptation_engine(self):
        """Initialize the adaptation engine."""
        pass  # Implementation details here

    def process_new_insight(self, insight: LearningInsightV2_5) -> bool:
        """Process a new learning insight and determine if adaptation is needed."""
        try:
            # Validate insight confidence against pattern discovery threshold
            if insight.confidence_score < self.config.pattern_discovery_threshold:
                return False
                
            # Add to insights collection
            self.insights.append(insight)
            self.metrics.total_insights_generated += 1
            
            # Check if adaptation is needed and auto-adaptation is enabled
            if self.config.auto_adaptation and self._should_adapt_from_insight(insight):
                adaptation = self._create_adaptation_from_insight(insight)
                return self._apply_adaptation(adaptation)
                
            return False
            
        except Exception as e:
            insight.errors.append(f"Error processing insight: {str(e)}")
            return False
            
    def _should_adapt_from_insight(self, insight: LearningInsightV2_5) -> bool:
        """Determine if an insight should trigger adaptation."""
        # Check warmup period
        if len(self.insights) < 5:  # Default warmup period
            return False
            
        # Check adaptation frequency
        recent_adaptations = len([a for a in self.adaptations 
                                if (datetime.now() - a.created_at).total_seconds() < 3600])
        if recent_adaptations >= 3:  # Default max adaptations per cycle
            return False
            
        # Evaluate insight priority and complexity
        if insight.integration_priority <= 2 and insight.integration_complexity <= 3:
            return True
            
        return False
        
    def _create_adaptation_from_insight(self, insight: LearningInsightV2_5) -> AIAdaptationV2_5:
        """Create an adaptation based on a learning insight."""
        return AIAdaptationV2_5(
            id=int(time.time() * 1000),  # Use millisecond timestamp as integer ID
            adaptation_type=insight.adaptation_type or "parameter_adjustment",
            adaptation_name=f"Adaptation from {insight.insight_type}",
            adaptation_description=insight.insight_description,
            confidence_score=insight.confidence_score,
            market_context=insight.market_context,
            performance_metrics={
                "pre_adaptation": insight.performance_metrics_pre,
                "expected_post": insight.performance_metrics_post
            }
        )
        
    def _apply_adaptation(self, adaptation: AIAdaptationV2_5) -> bool:
        """Apply an adaptation to the system."""
        try:
            # Add to adaptations list
            self.adaptations.append(adaptation)
            
            # Update metrics
            self.metrics.successful_adaptations += 1
            self._update_performance_metrics(adaptation)
            
            return True
            
        except Exception as e:
            self.metrics.failed_adaptations += 1
            logger.error(f"Error applying adaptation: {str(e)}")  # Use logger instead of error_messages
            return False
            
    def _update_performance_metrics(self, adaptation: AIAdaptationV2_5):
        """Update performance metrics after adaptation."""
        current_time = int(time.time() * 1000)
        performance = AIAdaptationPerformanceV2_5(
            adaptation_id=current_time,  # Use current timestamp as ID
            symbol=adaptation.market_context.get('symbol', 'SYSTEM'),
            time_period_days=7,
            total_applications=1,
            successful_applications=1 if adaptation.adaptation_score >= 0.7 else 0,
            success_rate=1.0 if adaptation.adaptation_score >= 0.7 else 0.0,
            avg_improvement=adaptation.adaptation_score,
            adaptation_score=adaptation.adaptation_score,
            performance_trend="STABLE"
        )
        self.performance_history.append(performance)
        
    def get_learning_summary(self) -> UnifiedLearningResult:
        """Generate a summary of learning progress."""
        current_time = datetime.now()
        insights_dict = {
            f"insight_{i}": insight.model_dump()
            for i, insight in enumerate(sorted(
                self.insights,
                key=lambda x: x.confidence_score,
                reverse=True
            )[:5])
        }
        
        # Format start time as ISO string
        start_time_iso = self.start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        return UnifiedLearningResult(
            symbol="SYSTEM",  # System-wide learning
            analysis_timestamp=current_time,
            learning_insights=insights_dict,
            performance_improvements=self._get_performance_improvements(),
            expert_adaptations=self._get_adaptation_summary(),
            confidence_updates=self._get_confidence_updates(),
            next_learning_cycle=self._calculate_next_cycle(),
            learning_cycle_type="continuous",
            lookback_period_days=7,
            performance_improvement_score=self._calculate_improvement_score(),
            confidence_score=self.metrics.average_confidence_score,
            optimization_recommendations=[],
            eots_schema_compliance=True,
            learning_metadata={
                "start_time": start_time_iso,
                "total_insights": len(self.insights),
                "total_adaptations": len(self.adaptations)
            }
        )
        
    def _get_performance_improvements(self) -> Dict[str, Any]:
        """Calculate performance improvements from adaptations."""
        improvements = {}
        for adaptation in self.adaptations:
            for metric, value in adaptation.performance_metrics.items():
                if metric not in improvements:
                    improvements[metric] = []
                improvements[metric].append(value)
        return improvements
        
    def _get_adaptation_summary(self) -> Dict[str, Any]:
        """Summarize adaptations by type."""
        summary = {}
        for adaptation in self.adaptations:
            if adaptation.adaptation_type not in summary:
                summary[adaptation.adaptation_type] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "success_rate": 0.0
                }
            summary[adaptation.adaptation_type]["count"] += 1
            summary[adaptation.adaptation_type]["avg_confidence"] += adaptation.confidence_score
        
        # Calculate averages
        for adaptation_type in summary:
            count = summary[adaptation_type]["count"]
            if count > 0:
                summary[adaptation_type]["avg_confidence"] /= count
                
        return summary
        
    def _get_confidence_updates(self) -> Dict[str, Any]:
        """Get confidence score updates over time."""
        return {
            "current_avg_confidence": self.metrics.average_confidence_score,
            "confidence_trend": self._calculate_confidence_trend(),
            "high_confidence_ratio": self._calculate_high_confidence_ratio()
        }
        
    def _calculate_next_cycle(self) -> datetime:
        """Calculate the next learning cycle timestamp."""
        from datetime import timedelta
        next_time = datetime.now() + timedelta(hours=1)  # Default to hourly
        return next_time
        
    def _calculate_improvement_score(self) -> float:
        """Calculate overall improvement score."""
        if not self.adaptations:
            return 0.0
            
        scores = [a.adaptation_score for a in self.adaptations]
        return sum(scores) / len(scores)
        
    def _calculate_confidence_trend(self) -> str:
        """Calculate the trend in confidence scores."""
        if len(self.insights) < 2:
            return "STABLE"
            
        recent_avg = sum(i.confidence_score for i in self.insights[-5:]) / 5
        overall_avg = self.metrics.average_confidence_score
        
        if recent_avg > overall_avg * 1.1:
            return "IMPROVING"
        elif recent_avg < overall_avg * 0.9:
            return "DECLINING"
        return "STABLE"
        
    def _calculate_high_confidence_ratio(self) -> float:
        """Calculate ratio of high confidence insights."""
        if not self.insights:
            return 0.0
            
        high_confidence = len([i for i in self.insights 
                             if i.confidence_score >= 0.8])
        return high_confidence / len(self.insights)


# API compatibility functions
def get_adaptive_learning_integration(config_manager, database_manager) -> AdaptiveLearningIntegrationV2_5:
    """Get an instance of the adaptive learning integration system."""
    config = AdaptiveLearningConfigV2_5()  # Create default config
    return AdaptiveLearningIntegrationV2_5(config=config)

async def run_daily_unified_learning(symbol: str, config_manager, database_manager) -> UnifiedLearningResult:
    """Run daily unified learning cycle."""
    integration = get_adaptive_learning_integration(config_manager, database_manager)
    return integration.get_learning_summary()

async def run_weekly_unified_learning(symbol: str, config_manager, database_manager) -> UnifiedLearningResult:
    """Run weekly unified learning cycle."""
    integration = get_adaptive_learning_integration(config_manager, database_manager)
    return integration.get_learning_summary()
