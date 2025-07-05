"""
Performance-Based Routing Strategy

Routes queries based on historical performance metrics of experts.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from .base import BaseRoutingStrategy, RoutingContext, RoutingResult


class PerformanceBasedRouting(BaseRoutingStrategy):
    """Performance-based routing using historical metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance-based routing.
        
        Args:
            config: Configuration including performance tracking settings
        """
        super().__init__(config)
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.success_rates: Dict[str, float] = defaultdict(float)
        self.last_used: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.performance_weight = self.config.get('performance_weight', 0.6)
        self.response_time_weight = self.config.get('response_time_weight', 0.2)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.min_samples = self.config.get('min_samples', 5)
    
    async def route(self, context: RoutingContext) -> RoutingResult:
        """Route based on performance metrics.
        
        Args:
            context: The routing context
            
        Returns:
            RoutingResult with selected expert
        """
        if not context.available_experts:
            raise ValueError("No experts available for routing")
        
        # Calculate scores for each expert
        expert_scores = {}
        for expert in context.available_experts:
            score = self._calculate_expert_score(expert)
            expert_scores[expert] = score
        
        # Select the best expert
        selected_expert = max(expert_scores, key=expert_scores.get)
        confidence = expert_scores[selected_expert]
        
        # Create fallback list (sorted by score, excluding selected)
        fallback_experts = sorted(
            [e for e in context.available_experts if e != selected_expert],
            key=lambda e: expert_scores[e],
            reverse=True
        )[:3]  # Top 3 fallbacks
        
        reasoning = f"Performance-based routing: score={confidence:.3f}"
        
        # Update last used timestamp
        self.last_used[selected_expert] = time.time()
        
        return RoutingResult(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            fallback_experts=fallback_experts
        )
    
    def update_performance(self, expert: str, performance_score: float) -> None:
        """Update performance metrics for an expert.
        
        Args:
            expert: The expert identifier
            performance_score: Performance score (0.0 to 1.0)
        """
        # Add to performance history
        self.performance_history[expert].append(performance_score)
        
        # Update success rate (exponential moving average)
        if expert not in self.success_rates:
            self.success_rates[expert] = performance_score
        else:
            alpha = 0.1
            self.success_rates[expert] = (
                alpha * performance_score + 
                (1 - alpha) * self.success_rates[expert]
            )
    
    def update_response_time(self, expert: str, response_time: float) -> None:
        """Update response time metrics for an expert.
        
        Args:
            expert: The expert identifier
            response_time: Response time in seconds
        """
        self.response_times[expert].append(response_time)
    
    def _calculate_expert_score(self, expert: str) -> float:
        """Calculate overall score for an expert.
        
        Args:
            expert: The expert identifier
            
        Returns:
            Overall score (0.0 to 1.0)
        """
        # Performance component
        performance_score = self._get_performance_score(expert)
        
        # Response time component (inverted - lower is better)
        response_time_score = self._get_response_time_score(expert)
        
        # Recency component (avoid overusing same expert)
        recency_score = self._get_recency_score(expert)
        
        # Weighted combination
        total_score = (
            self.performance_weight * performance_score +
            self.response_time_weight * response_time_score +
            self.recency_weight * recency_score
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _get_performance_score(self, expert: str) -> float:
        """Get performance score for an expert."""
        if expert not in self.performance_history or len(self.performance_history[expert]) < self.min_samples:
            return 0.5  # Default score for new experts
        
        # Use recent performance with some weight on overall success rate
        recent_scores = list(self.performance_history[expert])[-10:]  # Last 10 scores
        recent_avg = sum(recent_scores) / len(recent_scores)
        
        overall_success = self.success_rates.get(expert, 0.5)
        
        # Combine recent and overall performance
        return 0.7 * recent_avg + 0.3 * overall_success
    
    def _get_response_time_score(self, expert: str) -> float:
        """Get response time score for an expert (inverted)."""
        if expert not in self.response_times or not self.response_times[expert]:
            return 0.5  # Default score
        
        # Calculate average response time
        times = list(self.response_times[expert])
        avg_time = sum(times) / len(times)
        
        # Convert to score (lower time = higher score)
        # Assume 1 second is good, 5+ seconds is poor
        if avg_time <= 1.0:
            return 1.0
        elif avg_time >= 5.0:
            return 0.1
        else:
            return 1.0 - (avg_time - 1.0) / 4.0
    
    def _get_recency_score(self, expert: str) -> float:
        """Get recency score to avoid overusing same expert."""
        if expert not in self.last_used:
            return 1.0  # New expert gets full score
        
        time_since_last = time.time() - self.last_used[expert]
        
        # Score increases with time since last use
        # Full score after 60 seconds, minimum score immediately after use
        if time_since_last >= 60:
            return 1.0
        else:
            return 0.3 + 0.7 * (time_since_last / 60.0)
