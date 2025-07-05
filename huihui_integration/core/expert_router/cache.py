"""
🎯 EXPERT ROUTER - EMBEDDING CACHE
==================================================================

This module implements an ultra-fast embedding cache for the ExpertRouter
with optional integration with EnhancedCacheManagerV2_5.
"""

import hashlib
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Optional import for enhanced cache
try:
    from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
    ENHANCED_CACHE_AVAILABLE = True
except ImportError:
    ENHANCED_CACHE_AVAILABLE = False
    EnhancedCacheManagerV2_5 = None
    logging.warning(
        "EnhancedCacheManagerV2_5 not available. "
        "Falling back to in-memory cache."
    )

logger = logging.getLogger(__name__)

class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get a cached embedding if it exists and is not expired."""
        pass
    
    @abstractmethod
    def set(self, text: str, embedding: Union[np.ndarray, List[float]]) -> None:
        """Cache an embedding for the given text."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached embeddings."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
    
    def __contains__(self, text: str) -> bool:
        """Check if text is in the cache and not expired."""
        return self.get(text) is not None
    
    def __len__(self) -> int:
        """Get the number of cached items."""
        stats = self.get_stats()
        return stats.get('total_entries', 0)

class UltraFastEmbeddingCache(BaseCache):
    """
    A high-performance cache for embeddings with automatic eviction.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 86400,  # 24 hours
        cleanup_interval: int = 3600,  # 1 hour
        use_enhanced_cache: bool = ENHANCED_CACHE_AVAILABLE,
        enhanced_cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embedding cache with optional enhanced cache backend.
        
        Args:
            max_size: Maximum number of embeddings to cache (in-memory only)
            ttl_seconds: Time-to-live for cache entries in seconds
            cleanup_interval: How often to clean up expired entries (seconds)
            use_enhanced_cache: Whether to use EnhancedCacheManager if available
            enhanced_cache_config: Configuration for the enhanced cache
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._last_cleanup = datetime.utcnow()
        
        # Initialize enhanced cache if available and requested
        self.enhanced_cache = None
        if use_enhanced_cache and ENHANCED_CACHE_AVAILABLE:
            try:
                config = enhanced_cache_config or {}
                self.enhanced_cache = EnhancedCacheManagerV2_5(**config)
                logger.info("Using EnhancedCacheManagerV2_5 for embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize EnhancedCacheManager: {e}")
                self.enhanced_cache = None
        
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        # Use SHA-256 for better distribution and to avoid collisions
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        now = datetime.utcnow()
        if (now - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return
            
        expired_keys = [
            key for key, timestamp in self._access_times.items()
            if now - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
        
        self._last_cleanup = now
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if not self._access_times:
            return
            
        oldest_key = min(self._access_times, key=self._access_times.get)  # type: ignore
        self._cache.pop(oldest_key, None)
        self._access_times.pop(oldest_key, None)
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get a cached embedding if it exists and is not expired.
        
        Args:
            text: The input text
            
        Returns:
            The cached embedding as a numpy array, or None if not found/expired
        """
        # Try enhanced cache first if available
        if self.enhanced_cache is not None:
            try:
                cached = self.enhanced_cache.get_embedding(text)
                if cached is not None:
                    return np.array(cached)
            except Exception as e:
                logger.warning(f"Error getting from enhanced cache: {e}")
        
        # Fall back to in-memory cache
        self._cleanup_expired()
        key = self._get_cache_key(text)
        
        if key in self._cache:
            entry = self._cache[key]
            self._access_times[key] = datetime.utcnow()
            return np.array(entry['embedding'])
            
        return None
    
    def set(self, text: str, embedding: Union[np.ndarray, List[float]]) -> None:
        """
        Cache an embedding for the given text.
        
        Args:
            text: The input text
            embedding: The embedding to cache (numpy array or list)
        """
        # Convert to list if it's a numpy array
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        # Try to set in enhanced cache if available
        if self.enhanced_cache is not None:
            try:
                self.enhanced_cache.set_embedding(text, embedding_list, ttl_seconds=int(self.ttl.total_seconds()))
            except Exception as e:
                logger.warning(f"Error setting in enhanced cache: {e}")
        
        # Always update in-memory cache
        self._cleanup_expired()
        
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._get_cache_key(text)
        self._cache[key] = {
            'embedding': embedding_list,
            'created_at': datetime.utcnow().isoformat()
        }
        self._access_times[key] = datetime.utcnow()
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        if self.enhanced_cache is not None:
            try:
                self.enhanced_cache.clear_embeddings()
            except Exception as e:
                logger.warning(f"Error clearing enhanced cache: {e}")
                
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        now = datetime.utcnow()
        expired_count = sum(
            1 for timestamp in self._access_times.values()
            if (now - timestamp) > self.ttl
        )
        
        stats = {
            'total_entries': len(self._cache),
            'max_size': self.max_size,
            'expired_entries': expired_count,
            'ttl_seconds': self.ttl.total_seconds(),
            'last_cleanup': self._last_cleanup.isoformat(),
            'next_cleanup': (self._last_cleanup + timedelta(seconds=self.cleanup_interval)).isoformat(),
            'using_enhanced_cache': self.enhanced_cache is not None
        }
        
        # Add enhanced cache stats if available
        if self.enhanced_cache is not None:
            try:
                enhanced_stats = self.enhanced_cache.get_stats()
                stats['enhanced_cache'] = enhanced_stats
            except Exception as e:
                stats['enhanced_cache_error'] = str(e)
        
        return stats
    
# Factory function to create the appropriate cache instance
def create_embedding_cache(
    use_enhanced: bool = True,
    **kwargs
) -> BaseCache:
    """
    Create an appropriate cache instance based on availability.
    
    Args:
        use_enhanced: Whether to use enhanced cache if available
        **kwargs: Additional arguments to pass to the cache constructor
        
    Returns:
        A cache instance implementing the BaseCache interface
    """
    if use_enhanced and ENHANCED_CACHE_AVAILABLE:
        try:
            # Try to create enhanced cache
            enhanced_config = kwargs.pop('enhanced_cache_config', {})
            return UltraFastEmbeddingCache(
                use_enhanced_cache=True,
                enhanced_cache_config=enhanced_config,
                **kwargs
            )
        except Exception as e:
            logging.warning(f"Failed to create enhanced cache, falling back to in-memory: {e}")
    
    # Fall back to in-memory cache
    return UltraFastEmbeddingCache(use_enhanced_cache=False, **kwargs)
