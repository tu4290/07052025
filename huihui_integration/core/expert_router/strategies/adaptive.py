"""
Adaptive Routing Strategy

Combines multiple routing strategies and adapts based on performance.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseRoutingStrategy, RoutingContext, RoutingResult
from .vector_based import VectorBasedRouting
from .performance_based import PerformanceBasedRouting


class AdaptiveRouting(BaseRoutingStrategy):
    """Adaptive routing that combines multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive routing.
        
        Args:
            config: Configuration including strategy weights and adaptation settings
        """
        super().__init__(config)
        
        # Initialize sub-strategies
        self.vector_strategy = VectorBasedRouting(config.get('vector_config', {}))
        self.performance_strategy = PerformanceBasedRouting(config.get('performance_config', {}))
        
        # Strategy weights (will be adapted over time)
        self.strategy_weights = {
            'vector': self.config.get('vector_weight', 0.4),
            'performance': self.config.get('performance_weight', 0.6)
        }
        
        # Adaptation tracking
        self.strategy_performance = {
            'vector': [],
            'performance': []
        }
        
        # Configuration
        self.adaptation_rate = self.config.get('adaptation_rate', 0.05)
        self.min_samples_for_adaptation = self.config.get('min_samples_for_adaptation', 20)
    
    async def route(self, context: RoutingContext) -> RoutingResult:
        """Route using adaptive strategy combination.
        
        Args:
            context: The routing context
            
        Returns:
            RoutingResult with selected expert
        """
        if not context.available_experts:
            raise ValueError("No experts available for routing")
        
        # Get results from both strategies
        vector_result, performance_result = await asyncio.gather(
            self.vector_strategy.route(context),
            self.performance_strategy.route(context),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(vector_result, Exception):
            vector_result = None
        if isinstance(performance_result, Exception):
            performance_result = None
        
        # If both strategies failed, use simple fallback
        if vector_result is None and performance_result is None:
            return RoutingResult(
                selected_expert=context.available_experts[0],
                confidence=0.5,
                reasoning="Fallback routing (both strategies failed)",
                fallback_experts=context.available_experts[1:3]
            )
        
        # If only one strategy succeeded, use it
        if vector_result is None:
            return performance_result
        if performance_result is None:
            return vector_result
        
        # Combine results using weighted approach
        combined_result = self._combine_results(vector_result, performance_result, context)
        
        return combined_result
    
    def update_performance(self, expert: str, performance_score: float) -> None:
        """Update performance for all sub-strategies.
        
        Args:
            expert: The expert identifier
            performance_score: Performance score (0.0 to 1.0)
        """
        # Update sub-strategies
        self.vector_strategy.update_performance(expert, performance_score)
        self.performance_strategy.update_performance(expert, performance_score)
        
        # Adapt strategy weights based on performance
        self._adapt_weights(performance_score)
    
    def update_strategy_performance(self, strategy: str, performance_score: float) -> None:
        """Update performance tracking for a specific strategy.
        
        Args:
            strategy: Strategy name ('vector' or 'performance')
            performance_score: Performance score (0.0 to 1.0)
        """
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy].append(performance_score)
            
            # Keep only recent samples
            if len(self.strategy_performance[strategy]) > 100:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
    
    def _combine_results(self, vector_result: RoutingResult, performance_result: RoutingResult, 
                        context: RoutingContext) -> RoutingResult:
        """Combine results from multiple strategies.
        
        Args:
            vector_result: Result from vector-based strategy
            performance_result: Result from performance-based strategy
            context: The routing context
            
        Returns:
            Combined routing result
        """
        # Calculate weighted confidence scores for each expert
        expert_scores = {}
        
        # Add vector strategy contribution
        vector_weight = self.strategy_weights['vector']
        expert_scores[vector_result.selected_expert] = (
            expert_scores.get(vector_result.selected_expert, 0) + 
            vector_weight * vector_result.confidence
        )
        
        # Add performance strategy contribution
        performance_weight = self.strategy_weights['performance']
        expert_scores[performance_result.selected_expert] = (
            expert_scores.get(performance_result.selected_expert, 0) + 
            performance_weight * performance_result.confidence
        )
        
        # Select expert with highest combined score
        selected_expert = max(expert_scores, key=expert_scores.get)
        combined_confidence = expert_scores[selected_expert]
        
        # Create combined fallback list
        all_fallbacks = set(vector_result.fallback_experts + performance_result.fallback_experts)
        all_fallbacks.discard(selected_expert)  # Remove selected expert
        fallback_experts = list(all_fallbacks)[:3]  # Top 3 fallbacks
        
        reasoning = (
            f"Adaptive routing: vector={vector_weight:.2f}*{vector_result.confidence:.3f}, "
            f"performance={performance_weight:.2f}*{performance_result.confidence:.3f}"
        )
        
        return RoutingResult(
            selected_expert=selected_expert,
            confidence=combined_confidence,
            reasoning=reasoning,
            fallback_experts=fallback_experts
        )
    
    def _adapt_weights(self, performance_score: float) -> None:
        """Adapt strategy weights based on performance feedback.
        
        Args:
            performance_score: Recent performance score
        """
        # Only adapt if we have enough samples
        vector_samples = len(self.strategy_performance['vector'])
        performance_samples = len(self.strategy_performance['performance'])
        
        if (vector_samples < self.min_samples_for_adaptation or 
            performance_samples < self.min_samples_for_adaptation):
            return
        
        # Calculate recent average performance for each strategy
        vector_avg = sum(self.strategy_performance['vector'][-10:]) / min(10, vector_samples)
        performance_avg = sum(self.strategy_performance['performance'][-10:]) / min(10, performance_samples)
        
        # Adjust weights based on relative performance
        if vector_avg > performance_avg:
            # Vector strategy is performing better
            adjustment = self.adaptation_rate * (vector_avg - performance_avg)
            self.strategy_weights['vector'] = min(0.8, self.strategy_weights['vector'] + adjustment)
            self.strategy_weights['performance'] = 1.0 - self.strategy_weights['vector']
        else:
            # Performance strategy is performing better
            adjustment = self.adaptation_rate * (performance_avg - vector_avg)
            self.strategy_weights['performance'] = min(0.8, self.strategy_weights['performance'] + adjustment)
            self.strategy_weights['vector'] = 1.0 - self.strategy_weights['performance']
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self.strategy_weights.copy()
    
    def get_strategy_performance(self) -> Dict[str, List[float]]:
        """Get performance history for each strategy."""
        return {
            strategy: scores.copy() 
            for strategy, scores in self.strategy_performance.items()
        }
