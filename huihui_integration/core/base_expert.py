# huihui_integration/core/base_expert.py
"""
Abstract Base Class for HuiHui AI Experts
=========================================

This module defines the abstract base class (ABC) for all specialist experts
within the HuiHui "Market of Experts" (MOE) system. It establishes a common
interface and enforces a Pydantic-first architecture, ensuring that all experts
are cohesive, type-safe, and consistently integrated into the EOTS v2.5 ecosystem.

Key Principles Enforced by this Base Class:
-   **Standardized Interface**: All experts must implement the `analyze` method,
    providing a predictable entry point for the orchestrator.
-   **Configuration-Driven**: Each expert must be initialized with a Pydantic
    configuration model, preventing unconfigured instances.
-   **Pydantic-First Contract**: The `analyze` method's signature requires
    Pydantic models for both input (`ProcessedDataBundleV2_5`) and output,
    eliminating raw dictionaries and enforcing fail-fast validation.
-   **Self-Description**: Experts must declare their specialization via keywords,
    aiding in automated routing and system diagnostics.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

# Import the primary data bundle from the consolidated data_models package
# This ensures experts operate on validated, system-wide data structures.
from data_models import ProcessedDataBundleV2_5


class BaseHuiHuiExpert(ABC):
    """
    An abstract base class that defines the common interface for all HuiHui
    specialist experts.

    This class ensures that every expert in the system adheres to a consistent
    structure for initialization, analysis, and self-reporting, which is
    critical for the orchestrator's ability to manage them effectively.
    """

    def __init__(self, expert_config: BaseModel, db_manager=None):
        """
        Initializes the expert with a validated configuration.

        Args:
            expert_config: A Pydantic model containing the specific
                           configuration for the expert subclass. This enforces
                           that all experts are configured at instantiation.
        """
        if not isinstance(expert_config, BaseModel):
            raise TypeError("expert_config must be a Pydantic BaseModel instance.")
        self.config = expert_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Expert '{self.__class__.__name__}' initialized.")

    @abstractmethod
    async def analyze(self, data_bundle: ProcessedDataBundleV2_5) -> BaseModel:
        """
        The primary analysis method for the expert.

        This abstract method must be implemented by all concrete expert classes.
        It takes a validated `ProcessedDataBundleV2_5` and is required to return
        a Pydantic `BaseModel`, ensuring a strict, type-safe data contract.

        Args:
            data_bundle: The fully processed and validated data bundle from the
                         main EOTS analysis cycle.

        Returns:
            A Pydantic `BaseModel` instance containing the structured results
            of the expert's analysis. The specific model will vary by expert.
        """
        pass

    @abstractmethod
    def get_specialization_keywords(self) -> List[str]:
        """
        Returns a list of keywords that define this expert's specialization.

        These keywords are used by the orchestrator and routing mechanisms to
        determine which expert is best suited for a given task. This method
        forces each expert to be self-describing.

        Returns:
            A list of strings representing the expert's areas of focus.
        """
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Returns performance metrics for the expert.

        This is a concrete method that can be overridden by subclasses if they
        have more specific metrics to report. By default, it provides a basic
        set of performance indicators.

        Returns:
            A dictionary containing performance metrics.
        """
        # Default implementation; subclasses can provide more detail.
        return {
            "expert_name": self.__class__.__name__,
            "status": "active",
            "analysis_count": 0,
            "average_processing_time_ms": 0.0,
        }

