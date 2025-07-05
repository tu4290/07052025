"""
🎯 EXPERT ROUTER - UTILITIES
==================================================================

This module provides utility functions and helpers for the ExpertRouter system.
"""

import asyncio
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Awaitable, TypeVar, Type
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import functools
import random
import string

# Type variables
T = TypeVar('T')
P = TypeVar('P', bound=type)
R = TypeVar('R')

# Logger
logger = logging.getLogger(__name__)

def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a unique request ID.
    
    Args:
        prefix: Optional prefix for the request ID
        
    Returns:
        A unique request ID string
    """
    timestamp = int(time.time() * 1000)
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{rand_str}"

def time_execution(coro):
    """
    Decorator to measure and log the execution time of an async function.
    
    Args:
        coro: The coroutine function to decorate
        
    Returns:
        The decorated coroutine function
    """
    @functools.wraps(coro)
    async def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        result = await coro(*args, **kwargs)
        duration = time.monotonic() - start_time
        
        # Get function name
        func_name = coro.__name__
        
        # Log the duration
        logger.debug(f"Function {func_name} executed in {duration:.4f}s")
        
        # Add timing info to the result if it's a dict
        if isinstance(result, dict):
            result['_timing'] = {
                'function': func_name,
                'duration_seconds': duration,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return result
    
    return wrapper

def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to retry a function on exception with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for the delay between retries
        exceptions: Exception type(s) to catch and retry on
        logger: Optional logger for retry attempts
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                        
                    # Log the retry
                    log_message = (
                        f"Attempt {attempt + 1}/{max_retries} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if logger:
                        logger.warning(log_message)
                    else:
                        print(log_message)
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
                    
                    # Increase delay for next retry
                    delay = min(delay * backoff_factor, max_delay)
            
            # If we get here, all retries failed
            raise last_exception if last_exception else Exception("Unknown error in retry decorator")
        
        return async_wrapper
    
    return decorator

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate a configuration dictionary.
    
    Args:
        config: The configuration dictionary to validate
        required_keys: List of required keys
        
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

def to_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON-serializable representation of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif is_dataclass(obj):
        return {k: to_json_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, 'isoformat'):  # Handle datetime, date, time
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return to_json_serializable(obj.__dict__)
    else:
        return str(obj)

def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """
    Generate a hash of a string.
    
    Args:
        text: The text to hash
        algorithm: The hashing algorithm to use
        
    Returns:
        The hexadecimal digest of the hash
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()

class Timer:
    """A simple context manager for timing code blocks."""
    
    def __init__(self, name: str = "block", logger: Optional[logging.Logger] = None):
        """
        Initialize the timer.
        
        Args:
            name: Name for the timer (used in logs)
            logger: Optional logger for timing output
        """
        self.name = name
        self.logger = logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and log the duration."""
        self.end_time = time.monotonic()
        duration = self.duration
        
        if self.logger:
            self.logger.debug(f"{self.name} took {duration:.4f} seconds")
        
    @property
    def duration(self) -> float:
        """Get the elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.monotonic()
        return end_time - self.start_time

async def gather_with_concurrency(
    n: int,
    *coros: Awaitable[T],
    return_exceptions: bool = False
) -> List[T]:
    """
    Run coroutines with limited concurrency.
    
    Args:
        n: Maximum number of concurrent coroutines
        *coros: Coroutines to run
        return_exceptions: Whether to return exceptions as results
        
    Returns:
        List of results in the same order as the input coroutines
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *(sem_coro(coro) for coro in coros),
        return_exceptions=return_exceptions
    )

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop, creating one if necessary.
    
    Returns:
        The current event loop
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        if "There is no current event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise

def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async function in the current event loop, creating one if necessary.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)

class AsyncCache:
    """
    A simple async cache with TTL support.
    """
    
    def __init__(self, ttl: float = 300.0, maxsize: int = 1000):
        """
        Initialize the cache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds
            maxsize: Maximum number of cache entries
        """
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl = ttl
        self._maxsize = maxsize
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            The cached value or default
        """
        self._cleanup()
        
        if key in self._cache:
            expiry, value = self._cache[key]
            if time.monotonic() < expiry:
                return value
            
            # Remove expired entry
            del self._cache[key]
        
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (overrides default)
        """
        self._cleanup()
        
        # Evict if cache is full
        if len(self._cache) >= self._maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        expiry = time.monotonic() + (ttl if ttl is not None else self._ttl)
        self._cache[key] = (expiry, value)
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
    
    def _cleanup(self) -> None:
        """Remove expired entries from the cache."""
        now = time.monotonic()
        expired_keys = [k for k, (expiry, _) in self._cache.items() if expiry <= now]
        
        for key in expired_keys:
            del self._cache[key]
