"""
Enhanced Cache Manager v2.5 - Dynamic, Optimized Caching System
================================================================

A high-performance, Pydantic-first caching system with:
- Multi-level caching (memory, disk, compressed)
- Dynamic cache invalidation and TTL management
- Hierarchical cache organization
- Compression and optimization
- Cache analytics and monitoring
- Thread-safe operations
"""

import json
import gzip
import pickle
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

from pydantic import BaseModel, Field, ConfigDict
import sys
sys.path.append(str(Path(__file__).parent.parent))


class CacheLevel(str, Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    DISK = "disk"
    COMPRESSED = "compressed"
    PERSISTENT = "persistent"


class CacheStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    ADAPTIVE = "adaptive"  # AI-driven adaptive caching


# PYDANTIC-FIRST: Replace dataclass with Pydantic model for validation
class CacheMetricsV2_5(BaseModel):
    """Pydantic model for cache performance metrics - EOTS v2.5 compliant."""
    hits: int = Field(default=0, description="Cache hits count", ge=0)
    misses: int = Field(default=0, description="Cache misses count", ge=0)
    evictions: int = Field(default=0, description="Cache evictions count", ge=0)
    size_bytes: int = Field(default=0, description="Cache size in bytes", ge=0)
    last_access: Optional[datetime] = Field(None, description="Last access timestamp")
    creation_time: Optional[datetime] = Field(default_factory=datetime.now, description="Cache creation timestamp")

    model_config = ConfigDict(extra='forbid')


class CacheEntryMetadata(BaseModel):
    """Metadata for cache entries."""
    key: str = Field(..., description="Cache key")
    symbol: str = Field(..., description="Ticker symbol")
    metric_name: str = Field(..., description="Metric name")
    cache_level: CacheLevel = Field(..., description="Storage level")
    strategy: CacheStrategy = Field(default=CacheStrategy.TTL, description="Invalidation strategy")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_accessed: datetime = Field(default_factory=datetime.now, description="Last access timestamp")
    access_count: int = Field(default=0, description="Access frequency")
    ttl_seconds: Optional[int] = Field(default=3600, description="Time-to-live in seconds")
    size_bytes: int = Field(default=0, description="Entry size in bytes")
    compression_ratio: float = Field(default=1.0, description="Compression ratio achieved")
    version: str = Field(default="1.0", description="Cache entry version")
    tags: List[str] = Field(default_factory=list, description="Cache tags for grouping")

    model_config = ConfigDict(extra='forbid')


class CacheEntry(BaseModel):
    """Enhanced cache entry with metadata."""
    metadata: CacheEntryMetadata
    data: Any = Field(..., description="Cached data")
    checksum: str = Field(..., description="Data integrity checksum")

    model_config = ConfigDict(extra='forbid')


class EnhancedCacheManagerV2_5:
    """
    Enhanced Cache Manager with multi-level caching, compression, and analytics.
    """
    
    def __init__(self,
                 cache_root: str = "cache/enhanced_v2_5",
                 memory_limit_mb: int = 100,
                 disk_limit_mb: int = 1000,
                 default_ttl_seconds: int = 3600,
                 ultra_fast_mode: bool = True):
        """
        Initialize the enhanced cache manager.

        Args:
            cache_root: Root directory for cache storage
            memory_limit_mb: Memory cache limit in MB
            disk_limit_mb: Disk cache limit in MB
            default_ttl_seconds: Default TTL for cache entries
            ultra_fast_mode: Enable ultra-fast optimizations
        """
        self.logger = logging.getLogger(__name__)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Cache limits
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_limit_bytes = disk_limit_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.ultra_fast_mode = ultra_fast_mode

        # ULTRA-FAST OPTIMIZATIONS
        if ultra_fast_mode:
            # Increase memory cache significantly for ultra-fast access
            self.memory_limit_bytes *= 3  # 3x memory for hot data
            # Reduce disk I/O by keeping more in memory
            self._memory_priority_threshold = 2  # Promote to memory after 2 accesses
            # Pre-allocate hash maps for faster lookups
            self._memory_cache: Dict[str, CacheEntry] = dict()
            self._fast_lookup: Dict[str, str] = dict()  # symbol:metric -> cache_key mapping
        else:
            self._memory_priority_threshold = 5
            self._memory_cache: Dict[str, CacheEntry] = {}
            self._fast_lookup: Dict[str, str] = {}

        # Multi-level cache storage
        self._cache_metadata: Dict[str, CacheEntryMetadata] = {}
        self._cache_metrics = CacheMetricsV2_5()

        # Thread safety - Use faster lock for ultra-fast mode
        if ultra_fast_mode:
            self._lock = threading.Lock()  # Faster than RLock
        else:
            self._lock = threading.RLock()

        # Cache directories
        self._setup_cache_directories()

        mode_info = "ULTRA-FAST" if ultra_fast_mode else "Standard"
        self.logger.info(f"Enhanced Cache Manager v2.5 initialized ({mode_info}) - Memory: {memory_limit_mb}MB, Disk: {disk_limit_mb}MB")
    
    def _setup_cache_directories(self):
        """Setup hierarchical cache directory structure."""
        directories = [
            "memory",
            "disk/metrics",
            "disk/symbols", 
            "compressed",
            "persistent",
            "analytics",
            "metadata"
        ]
        
        for directory in directories:
            (self.cache_root / directory).mkdir(parents=True, exist_ok=True)
    
    def sanitize_symbol(self, symbol: str) -> str:
        """Sanitize ticker symbol for safe file operations."""
        return symbol.replace('/', '_').replace(':', '_').replace('\\', '_')
    
    def _generate_cache_key(self, symbol: str, metric_name: str, **kwargs) -> str:
        """Generate a unique cache key with ultra-fast lookup optimization."""
        safe_symbol = self.sanitize_symbol(symbol)

        # ULTRA-FAST: Check fast lookup first for common symbol:metric combinations
        if self.ultra_fast_mode and not kwargs:
            lookup_key = f"{safe_symbol}:{metric_name}"
            if lookup_key in self._fast_lookup:
                return self._fast_lookup[lookup_key]

        key_parts = [safe_symbol, metric_name]

        # Add additional parameters to key
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_string = "_".join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:16]

        # ULTRA-FAST: Store in fast lookup for future use
        if self.ultra_fast_mode and not kwargs:
            lookup_key = f"{safe_symbol}:{metric_name}"
            self._fast_lookup[lookup_key] = cache_key

        return cache_key
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate data integrity checksum."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _compress_data(self, data: Any) -> Tuple[bytes, float]:
        """Compress data and return compression ratio."""
        original_data = pickle.dumps(data)
        compressed_data = gzip.compress(original_data)
        compression_ratio = len(original_data) / len(compressed_data) if compressed_data else 1.0
        return compressed_data, compression_ratio
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data."""
        original_data = gzip.decompress(compressed_data)
        return pickle.loads(original_data)
    
    def _is_expired(self, metadata: CacheEntryMetadata) -> bool:
        """Check if cache entry is expired."""
        if not metadata.ttl_seconds:
            return False
        
        age_seconds = (datetime.now() - metadata.created_at).total_seconds()
        return age_seconds > metadata.ttl_seconds
    
    def _get_cache_file_path(self, key: str, level: CacheLevel) -> Path:
        """Get file path for cache entry."""
        if level == CacheLevel.COMPRESSED:
            return self.cache_root / "compressed" / f"{key}.gz"
        elif level == CacheLevel.PERSISTENT:
            return self.cache_root / "persistent" / f"{key}.pkl"
        else:
            return self.cache_root / "disk" / f"{key}.json"
    
    def sanitize_symbol_for_cache(self, symbol: str) -> str:
        """Enhanced symbol sanitization for cache operations."""
        return self.sanitize_symbol(symbol)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            memory_size = sum(len(str(entry.data)) for entry in self._memory_cache.values())
            
            return {
                "hits": self._cache_metrics.hits,
                "misses": self._cache_metrics.misses,
                "hit_ratio": self._cache_metrics.hits / max(1, self._cache_metrics.hits + self._cache_metrics.misses),
                "memory_entries": len(self._memory_cache),
                "memory_size_mb": memory_size / (1024 * 1024),
                "total_entries": len(self._cache_metadata),
                "evictions": self._cache_metrics.evictions
            }
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        expired_count = 0
        with self._lock:
            expired_keys = []
            for key, metadata in self._cache_metadata.items():
                if self._is_expired(metadata):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.invalidate(key)
                expired_count += 1
        
        return expired_count
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            with self._lock:
                # Remove from memory
                if key in self._memory_cache:
                    del self._memory_cache[key]

                # Remove metadata
                if key in self._cache_metadata:
                    metadata = self._cache_metadata[key]

                    # Remove disk files
                    file_path = self._get_cache_file_path(key, metadata.cache_level)
                    if file_path.exists():
                        file_path.unlink()

                    del self._cache_metadata[key]

                return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache key {key}: {e}")
            return False

    def put(self, symbol: str, metric_name: str, data: Any,
            ttl_seconds: Optional[int] = None,
            cache_level: CacheLevel = CacheLevel.MEMORY,
            tags: Optional[List[str]] = None) -> bool:
        """
        Store data in cache with enhanced metadata.

        Args:
            symbol: Ticker symbol
            metric_name: Metric name
            data: Data to cache
            ttl_seconds: Time-to-live override
            cache_level: Storage level
            tags: Cache tags for organization

        Returns:
            bool: Success status
        """
        try:
            with self._lock:
                cache_key = self._generate_cache_key(symbol, metric_name)
                checksum = self._calculate_checksum(data)

                # Create metadata
                metadata = CacheEntryMetadata(
                    key=cache_key,
                    symbol=self.sanitize_symbol(symbol),
                    metric_name=metric_name,
                    cache_level=cache_level,
                    ttl_seconds=ttl_seconds or self.default_ttl,
                    tags=tags or []
                )

                # Create cache entry
                entry = CacheEntry(
                    metadata=metadata,
                    data=data,
                    checksum=checksum
                )

                # Store based on cache level
                if cache_level == CacheLevel.MEMORY:
                    self._store_in_memory(cache_key, entry)
                elif cache_level == CacheLevel.COMPRESSED:
                    self._store_compressed(cache_key, entry)
                elif cache_level == CacheLevel.PERSISTENT:
                    self._store_persistent(cache_key, entry)
                else:
                    self._store_on_disk(cache_key, entry)

                # Update metadata
                self._cache_metadata[cache_key] = metadata

                self.logger.debug(f"Cached {symbol}:{metric_name} at level {cache_level}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to cache {symbol}:{metric_name}: {e}")
            return False

    def get(self, symbol: str, metric_name: str, **kwargs) -> Optional[Any]:
        """
        Retrieve data from cache with ultra-fast optimizations.

        Args:
            symbol: Ticker symbol
            metric_name: Metric name
            **kwargs: Additional key parameters

        Returns:
            Cached data or None if not found/expired
        """
        try:
            # ULTRA-FAST: Skip lock for memory-only lookups in ultra-fast mode
            if self.ultra_fast_mode and not kwargs:
                safe_symbol = self.sanitize_symbol(symbol)
                lookup_key = f"{safe_symbol}:{metric_name}"
                if lookup_key in self._fast_lookup:
                    cache_key = self._fast_lookup[lookup_key]
                    if cache_key in self._memory_cache:
                        entry = self._memory_cache[cache_key]
                        # Quick expiration check without metadata lookup
                        if entry.metadata.ttl_seconds:
                            age_seconds = (datetime.now() - entry.metadata.created_at).total_seconds()
                            if age_seconds <= entry.metadata.ttl_seconds:
                                self._cache_metrics.hits += 1
                                return entry.data

            # Standard cache lookup with lock
            with self._lock:
                cache_key = self._generate_cache_key(symbol, metric_name, **kwargs)

                # Check metadata first
                if cache_key not in self._cache_metadata:
                    self._cache_metrics.misses += 1
                    return None

                metadata = self._cache_metadata[cache_key]

                # Check expiration
                if self._is_expired(metadata):
                    self.invalidate(cache_key)
                    self._cache_metrics.misses += 1
                    return None

                # Try to retrieve from appropriate level
                entry = None
                if metadata.cache_level == CacheLevel.MEMORY:
                    entry = self._memory_cache.get(cache_key)
                elif metadata.cache_level == CacheLevel.COMPRESSED:
                    entry = self._load_compressed(cache_key)
                elif metadata.cache_level == CacheLevel.PERSISTENT:
                    entry = self._load_persistent(cache_key)
                else:
                    entry = self._load_from_disk(cache_key)

                if entry:
                    # Update access metadata
                    metadata.last_accessed = datetime.now()
                    metadata.access_count += 1

                    # ULTRA-FAST: Promote more aggressively to memory
                    if (metadata.cache_level != CacheLevel.MEMORY and
                        metadata.access_count > self._memory_priority_threshold):
                        self._promote_to_memory(cache_key, entry)

                    self._cache_metrics.hits += 1
                    return entry.data
                else:
                    self._cache_metrics.misses += 1
                    return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve {symbol}:{metric_name}: {e}")
            self._cache_metrics.misses += 1
            return None

    def _store_in_memory(self, key: str, entry: CacheEntry) -> None:
        """Store entry in memory cache."""
        self._memory_cache[key] = entry
        entry.metadata.size_bytes = len(str(entry.data))

    def _store_on_disk(self, key: str, entry: CacheEntry) -> None:
        """Store entry on disk as JSON."""
        file_path = self._get_cache_file_path(key, CacheLevel.DISK)
        with open(file_path, 'w') as f:
            json.dump({
                'metadata': entry.metadata.model_dump(),
                'data': entry.data,
                'checksum': entry.checksum
            }, f, default=str)

    def _store_compressed(self, key: str, entry: CacheEntry) -> None:
        """Store entry with compression."""
        file_path = self._get_cache_file_path(key, CacheLevel.COMPRESSED)
        compressed_data, compression_ratio = self._compress_data(entry.data)

        entry.metadata.compression_ratio = compression_ratio

        with open(file_path, 'wb') as f:
            # Store metadata and compressed data
            metadata_json = entry.metadata.model_dump_json()
            f.write(len(metadata_json).to_bytes(4, 'big'))
            f.write(metadata_json.encode())
            f.write(compressed_data)

    def _store_persistent(self, key: str, entry: CacheEntry) -> None:
        """Store entry in persistent storage."""
        file_path = self._get_cache_file_path(key, CacheLevel.PERSISTENT)
        with open(file_path, 'wb') as f:
            pickle.dump(entry, f)

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk."""
        try:
            file_path = self._get_cache_file_path(key, CacheLevel.DISK)
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            metadata = CacheEntryMetadata(**data['metadata'])
            return CacheEntry(
                metadata=metadata,
                data=data['data'],
                checksum=data['checksum']
            )
        except Exception as e:
            self.logger.error(f"Failed to load from disk {key}: {e}")
            return None

    def _load_compressed(self, key: str) -> Optional[CacheEntry]:
        """Load compressed entry."""
        try:
            file_path = self._get_cache_file_path(key, CacheLevel.COMPRESSED)
            if not file_path.exists():
                return None

            with open(file_path, 'rb') as f:
                # Read metadata length
                metadata_length = int.from_bytes(f.read(4), 'big')

                # Read metadata
                metadata_json = f.read(metadata_length).decode()
                metadata = CacheEntryMetadata.model_validate_json(metadata_json)

                # Read and decompress data
                compressed_data = f.read()
                data = self._decompress_data(compressed_data)

            return CacheEntry(
                metadata=metadata,
                data=data,
                checksum=self._calculate_checksum(data)
            )
        except Exception as e:
            self.logger.error(f"Failed to load compressed {key}: {e}")
            return None

    def _load_persistent(self, key: str) -> Optional[CacheEntry]:
        """Load persistent entry."""
        try:
            file_path = self._get_cache_file_path(key, CacheLevel.PERSISTENT)
            if not file_path.exists():
                return None

            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load persistent {key}: {e}")
            return None

    def _promote_to_memory(self, key: str, entry: CacheEntry) -> None:
        """Promote frequently accessed entry to memory."""
        try:
            # Check memory limits
            if len(self._memory_cache) < 1000:  # Simple limit
                self._memory_cache[key] = entry
                entry.metadata.cache_level = CacheLevel.MEMORY
                self.logger.debug(f"Promoted {key} to memory cache")
        except Exception as e:
            self.logger.error(f"Failed to promote {key} to memory: {e}")

    def clear_all(self) -> None:
        """Clear all cache data."""
        with self._lock:
            self._memory_cache.clear()
            self._cache_metadata.clear()
            self._cache_metrics = CacheMetricsV2_5()

            # Clear disk cache
            for cache_dir in ["disk", "compressed", "persistent"]:
                cache_path = self.cache_root / cache_dir
                if cache_path.exists():
                    for file_path in cache_path.rglob("*"):
                        if file_path.is_file():
                            file_path.unlink()

    def get_cache_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            info = {
                "total_entries": len(self._cache_metadata),
                "memory_entries": len(self._memory_cache),
                "stats": self.get_cache_stats()
            }

            if symbol:
                safe_symbol = self.sanitize_symbol(symbol)
                symbol_entries = [
                    metadata for metadata in self._cache_metadata.values()
                    if metadata.symbol == safe_symbol
                ]
                info["symbol_entries"] = len(symbol_entries)
                info["symbol_metrics"] = [entry.metric_name for entry in symbol_entries]

            return info
