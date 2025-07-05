"""
Vector-Based Routing Strategy

Routes queries based on semantic similarity using vector embeddings.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseRoutingStrategy, RoutingContext, RoutingResult


class VectorBasedRouting(BaseRoutingStrategy):
    """Vector-based routing using semantic similarity."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vector-based routing.
        
        Args:
            config: Configuration including embedding model settings
        """
        super().__init__(config)
        self.expert_embeddings: Dict[str, np.ndarray] = {}
        self.performance_weights: Dict[str, float] = {}
        
        # Default configuration
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.performance_weight = self.config.get('performance_weight', 0.3)
    
    async def route(self, context: RoutingContext) -> RoutingResult:
        """Route based on vector similarity.
        
        Args:
            context: The routing context
            
        Returns:
            RoutingResult with selected expert
        """
        if not context.available_experts:
            raise ValueError("No experts available for routing")
        
        # For now, use simple fallback routing until embeddings are implemented
        # This is a placeholder implementation
        selected_expert = context.available_experts[0]
        confidence = 0.8
        reasoning = "Vector-based routing (placeholder implementation)"
        fallback_experts = context.available_experts[1:3] if len(context.available_experts) > 1 else []
        
        return RoutingResult(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            fallback_experts=fallback_experts
        )
    
    def update_performance(self, expert: str, performance_score: float) -> None:
        """Update performance weights for an expert.
        
        Args:
            expert: The expert identifier
            performance_score: Performance score (0.0 to 1.0)
        """
        if expert not in self.performance_weights:
            self.performance_weights[expert] = performance_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_weights[expert] = (
                alpha * performance_score + 
                (1 - alpha) * self.performance_weights[expert]
            )
    
    def add_expert_embedding(self, expert: str, embedding: np.ndarray) -> None:
        """Add or update an expert's embedding.
        
        Args:
            expert: The expert identifier
            embedding: The expert's embedding vector
        """
        self.expert_embeddings[expert] = embedding
    
    def _compute_similarity(self, query_embedding: np.ndarray, expert_embedding: np.ndarray) -> float:
        """Compute cosine similarity between query and expert embeddings.
        
        Args:
            query_embedding: Query embedding vector
            expert_embedding: Expert embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        query_norm = np.linalg.norm(query_embedding)
        expert_norm = np.linalg.norm(expert_embedding)
        
        if query_norm == 0 or expert_norm == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(query_embedding, expert_embedding) / (query_norm * expert_norm)
        return float(similarity)
