"""
Base Routing Strategy

Provides the abstract base class for all routing strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    query: str
    metadata: Dict[str, Any]
    available_experts: List[str]
    performance_history: Optional[Dict[str, float]] = None


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    selected_expert: str
    confidence: float
    reasoning: str
    fallback_experts: List[str]


class BaseRoutingStrategy(ABC):
    """Abstract base class for routing strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the routing strategy.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def route(self, context: RoutingContext) -> RoutingResult:
        """Route a query to the most appropriate expert.
        
        Args:
            context: The routing context containing query and metadata
            
        Returns:
            RoutingResult with selected expert and confidence
        """
        pass
    
    @abstractmethod
    def update_performance(self, expert: str, performance_score: float) -> None:
        """Update performance metrics for an expert.
        
        Args:
            expert: The expert identifier
            performance_score: Performance score (0.0 to 1.0)
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the configuration."""
        self.config.update(config)
