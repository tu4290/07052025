# data_models/huihui_learning_schemas.py
"""
Pydantic Schemas for HuiHui AI Learning System v3.0
===================================================

This module defines the Pydantic models that correspond to the database
schema for the HuiHui AI's continuous learning and adaptation capabilities.
These models ensure that all data related to expert knowledge, configuration,
performance, and learning feedback is strictly validated and type-safe.

This schema provides the foundation for a continuously adapting
and improving "Market of Experts" system.

Models:
- ExpertKnowledgeBase: For storing curated facts, rules, and patterns.
- ExpertConfiguration: For dynamic, per-expert parameter tuning.
- ExpertPerformanceHistory: For tracking historical performance of each expert.
- LearningCycle: For logging the outcomes of the central learning system.
- PredictionOutcome: For tracking every prediction and its real-world result.
- FeedbackLoop: For storing explicit feedback to guide the learning process.

Author: EOTS v2.5 AI Architecture Division
Version: 3.0.0 - "ADAPTIVE INTELLIGENCE"
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, field_validator

# Import shared base models and enums for consistency
from .core_models import EOTSBaseModel
from .ai_ml_models import HuiHuiExpertType


class ExpertKnowledgeBase(EOTSBaseModel):
    """
    Stores a single piece of curated knowledge, a rule, or a pattern for an expert.
    This acts as the expert's long-term, queryable memory or "brain".
    Corresponds to the `expert_knowledge_base` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the knowledge entry.")
    expert_type: HuiHuiExpertType = Field(..., description="The expert this knowledge belongs to.")
    knowledge_type: str = Field(..., description="Type of knowledge, e.g., 'rule', 'pattern', 'fact', 'heuristic'.")
    content: Dict[str, Any] = Field(..., description="The structured knowledge content, stored as a JSON-compatible dict.")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the correctness of this knowledge.")
    source: str = Field(default='human_curated', description="Origin of the knowledge, e.g., 'human_curated', 'auto_learned'.")
    version: int = Field(default=1, ge=1, description="Version of this knowledge entry.")
    is_active: bool = Field(default=True, description="Whether this knowledge is currently active and should be used by the expert.")
    description: Optional[str] = Field(None, description="A human-readable description of the knowledge.")


class ExpertConfiguration(EOTSBaseModel):
    """
    Represents a single, dynamic configuration parameter for an expert.
    This allows for real-time tuning of expert behavior without code deployments.
    Corresponds to the `expert_configurations` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the configuration entry.")
    expert_type: HuiHuiExpertType = Field(..., description="The expert this configuration applies to.")
    config_name: str = Field(..., description="The name of the configuration parameter, e.g., 'confidence_threshold'.")
    config_value: str = Field(..., description="The value of the parameter, stored as a string.")
    value_type: Literal['float', 'integer', 'string', 'boolean'] = Field(..., description="The data type of the configuration value.")
    description: Optional[str] = Field(None, description="A human-readable description of the configuration parameter.")
    is_active: bool = Field(default=True, description="Whether this configuration is currently in effect.")

    @property
    def value(self) -> Union[float, int, str, bool]:
        """
        Casts the string-based `config_value` to its proper Python type.
        This provides a safe, type-casted way to access the configuration value.
        """
        try:
            if self.value_type == 'float':
                return float(self.config_value)
            elif self.value_type == 'integer':
                return int(self.config_value)
            elif self.value_type == 'boolean':
                return self.config_value.lower() in ['true', '1', 't', 'y', 'yes']
            else:  # 'string'
                return self.config_value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not cast config '{self.config_name}' with value '{self.config_value}' to type '{self.value_type}': {e}") from e


class ExpertPerformanceHistory(EOTSBaseModel):
    """
    A record of a single analysis performed by an expert, used for historical
    performance tracking and learning.
    Corresponds to the `expert_performance_history` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the performance record.")
    expert_type: HuiHuiExpertType = Field(..., description="The expert that performed the analysis.")
    analysis_id: str = Field(..., description="A unique identifier for the analysis event, linking it to other logs.")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the analysis was performed.")
    success: bool = Field(..., description="Whether the analysis completed without errors.")
    processing_time_ms: int = Field(..., ge=0, description="The time taken for the analysis in milliseconds.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="The expert's confidence in its own analysis.")
    outcome_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="The measured accuracy of the analysis outcome, populated later by a feedback loop.")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Other relevant performance metrics, e.g., tokens_used.")


class LearningCycle(EOTSBaseModel):
    """
    A record of a single execution of the HuiHuiLearningSystem, capturing
    what was learned and what actions were taken.
    Corresponds to the `learning_cycles` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the learning cycle record.")
    cycle_id: str = Field(..., description="A unique identifier for this learning cycle.")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the learning cycle was executed.")
    duration_seconds: int = Field(..., ge=0, description="The duration of the learning cycle in seconds.")
    focus_area: str = Field(..., description="The primary focus of this learning cycle, e.g., 'regime_transition_accuracy'.")
    insights_generated: Optional[Dict[str, Any]] = Field(None, description="Key insights derived during the cycle, stored as JSON.")
    adaptations_applied: Optional[Dict[str, Any]] = Field(None, description="Changes made to configurations or knowledge base as a result of the cycle.")
    performance_change: Optional[Dict[str, Any]] = Field(None, description="Measured performance improvement or degradation after the cycle.")


class PredictionOutcome(EOTSBaseModel):
    """
    Logs a specific, trackable prediction made by the system and its eventual
    real-world outcome. This is the primary data source for learning.
    Corresponds to the `prediction_outcomes` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the prediction record.")
    prediction_id: str = Field(..., description="A unique identifier for the prediction.")
    analysis_id: str = Field(..., description="The analysis ID that generated this prediction.")
    expert_type: HuiHuiExpertType = Field(..., description="The expert that made the prediction.")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the prediction was made.")
    prediction_details: Dict[str, Any] = Field(..., description="The specific prediction, e.g., {'direction': 'UP', 'target_price': 450.50}.")
    outcome_timestamp: Optional[datetime] = Field(None, description="When the outcome was observed and recorded.")
    actual_outcome: Optional[Dict[str, Any]] = Field(None, description="The actual market result corresponding to the prediction.")
    is_correct: Optional[bool] = Field(None, description="A simple boolean indicating if the prediction was correct.")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="A numerical score of correctness (e.g., 0.0 to 1.0).")
    status: Literal['pending', 'validated', 'expired'] = Field(default='pending', description="The current status of the prediction.")


class FeedbackLoop(EOTSBaseModel):
    """
    Stores explicit feedback from an external source (human or automated)
    to guide the learning process.
    Corresponds to the `feedback_loops` table.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Primary key for the feedback entry.")
    feedback_type: str = Field(..., description="The type of feedback, e.g., 'human_correction', 'system_alert'.")
    source: str = Field(..., description="The source of the feedback, e.g., 'analyst_john_doe'.")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the feedback was submitted.")
    target_id: str = Field(..., description="The ID of the item being corrected (e.g., a prediction_id or analysis_id).")
    target_type: str = Field(..., description="The type of item being corrected, e.g., 'prediction', 'knowledge_entry'.")
    feedback_content: Dict[str, Any] = Field(..., description="The structured feedback content, e.g., {'corrected_value': 'BEARISH'}.")
    is_processed: bool = Field(default=False, description="Whether the learning system has processed this feedback.")
    processed_at: Optional[datetime] = Field(None, description="Timestamp of when the feedback was processed.")


# Explicitly define what is exported from this module
__all__ = [
    "ExpertKnowledgeBase",
    "ExpertConfiguration",
    "ExpertPerformanceHistory",
    "LearningCycle",
    "PredictionOutcome",
    "FeedbackLoop",
]
