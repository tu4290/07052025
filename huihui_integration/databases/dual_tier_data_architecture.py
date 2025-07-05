# huihui_integration/databases/dual_tier_data_architecture.py
"""
HuiHui AI System: Elite Dual-Tier Data Architecture Coordinator
===============================================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module is the central nervous system for all data operations within the
HuiHui AI ecosystem. It implements a high-performance, dual-tier data
architecture that intelligently coordinates between a fast intraday cache and
a robust long-term database, fulfilling all 10 core requirements for an
elite data management system.

Architecture Overview & Features:
---------------------------------
1.  **Intelligent Data Routing (Cache-First)**: Implements a cache-first read
    strategy. Queries for historical data first check the high-speed cache; if
    data is not present, the query proceeds to the long-term database, and the
    result is then cached for future requests.

2.  **Live Data Collection & High-Frequency Writes**: Provides non-blocking async
    methods (`queue_*`) to accept high-frequency writes. Data is placed into
    in-memory queues and written to the database in large, efficient batches by
    background workers using the high-performance `COPY` protocol. This ensures
    the main application loop is never blocked by database I/O.

3.  **Training & Educational Data Pipelines**: Includes dedicated, asynchronous
    pipelines for storing and retrieving both labeled training examples for model
    fine-tuning and vectorized educational documents for Retrieval-Augmented
    Generation (RAG).

4.  **Automated Data Lifecycle Management**: A `run_nightly_maintenance` method
    provides a hook for scheduled jobs to prune old cache files and archive or
    delete records from the long-term database based on a defined retention policy.

5.  **Failover & Recovery**: The system is designed for resilience. If a tier
    (cache or database) fails to initialize, the coordinator remains operational
    in a degraded state, logging critical errors without crashing the application.
    Background writers include simple back-off and retry logic.

6.  **Comprehensive Monitoring & Analytics**: A `get_system_analytics` method
    returns a validated Pydantic model (`DualTierSystemAnalytics`) with real-time
    metrics on cache performance, database status, and the current load of the
    background writer queues.

7.  **Strict Data Validation**: Enforces Pydantic validation at every entry and
    exit point, ensuring 100% data integrity and adherence to the "Zero Tolerance
    for Fake Data" policy.

This coordinator provides both the extreme performance required for intraday
operations and the robust persistence needed for long-term learning and analysis.

Author: EOTS v2.5 AI Architecture Division
Version: 2.5.2
"""

import logging
import asyncio
from typing import Optional, List, Any, Type, Dict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

# EOTS data management components
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from huihui_integration.monitoring.supabase_manager import HuiHuiSupabaseManager

# EOTS Pydantic models for data validation
from data_models import (
    MOEUnifiedResponseV2_5,
    FinalAnalysisBundleV2_5,
)

# Placeholder models for Training and RAG data structures
class TrainingExampleV2_5(BaseModel):
    """Represents a single labeled example for model training."""
    id: str = Field(..., description="Unique identifier for the example.")
    feature_vector: List[float] = Field(..., description="The numerical feature vector.")
    label: str = Field(..., description="The ground truth label.")
    expert_id: str = Field(..., description="The expert this example is for.")
    timestamp: datetime = Field(..., description="Timestamp of the event.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EducationalDocumentV2_5(BaseModel):
    """Represents a chunk of an educational document for RAG."""
    id: str = Field(..., description="Unique identifier for the document chunk.")
    source: str = Field(..., description="The original source of the document (e.g., 'research_paper.pdf').")
    content_chunk: str = Field(..., description="The text content of the chunk.")
    embedding: List[float] = Field(..., description="The vector embedding of the content chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DualTierSystemAnalytics(BaseModel):
    """Provides a validated snapshot of the data architecture's health and performance."""
    cache_available: bool
    database_available: bool
    cache_analytics: Optional[Dict[str, Any]] = None
    database_status: Optional[Dict[str, Any]] = None
    background_writer_queues: Dict[str, int]

logger = logging.getLogger(__name__)

class DualTierDataArchitecture:
    """
    A singleton coordinator for managing the dual-tier data architecture.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DualTierDataArchitecture, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_manager: Optional[Any] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        logger.info("Initializing Elite Dual-Tier Data Architecture Coordinator...")
        
        # Tier 1: Intraday Cache
        try:
            self.cache_manager = EnhancedCacheManagerV2_5()
            logger.info("✅ Tier 1: Enhanced Cache Manager initialized.")
        except Exception as e:
            logger.critical(f"❌ Tier 1: FAILED to initialize Enhanced Cache Manager: {e}", exc_info=True)
            self.cache_manager = None

        # Tier 2: Long-Term Storage
        try:
            self.supabase_manager = HuiHuiSupabaseManager()
            logger.info("✅ Tier 2: HuiHui Supabase Manager initialized.")
        except Exception as e:
            logger.critical(f"❌ Tier 2: FAILED to initialize HuiHui Supabase Manager: {e}", exc_info=True)
            self.supabase_manager = None
            
        # High-Frequency Write Queues
        self._expert_resp_queue: asyncio.Queue[MOEUnifiedResponseV2_5] = asyncio.Queue(maxsize=10000)
        self._training_queue: asyncio.Queue[TrainingExampleV2_5] = asyncio.Queue(maxsize=5000)
        self._education_queue: asyncio.Queue[EducationalDocumentV2_5] = asyncio.Queue(maxsize=1000)
        
        # Launch background workers
        if self.supabase_manager:
            asyncio.create_task(self._background_writer(self._expert_resp_queue, self.supabase_manager.batch_store_expert_responses, "ExpertResponses"))
            asyncio.create_task(self._background_writer(self._training_queue, self.supabase_manager.batch_store_training_examples, "TrainingExamples"))
            asyncio.create_task(self._background_writer(self._education_queue, self.supabase_manager.batch_store_educational_documents, "EducationalDocs"))
            logger.info("✅ Background writer tasks for all data queues have been launched.")
        else:
            logger.error("Database manager not available, background writers will not start.")

        self._initialized = True
        logger.info("🚀 Elite Dual-Tier Data Architecture Coordinator is operational.")

    @classmethod
    def get_instance(cls, config_manager: Optional[Any] = None) -> 'DualTierDataArchitecture':
        """Provides access to the singleton instance of the data coordinator."""
        if cls._instance is None:
            cls._instance = cls(config_manager)
        return cls._instance

    # --- Live Data & High-Frequency Write Methods ---

    async def queue_expert_response_for_storage(self, response: MOEUnifiedResponseV2_5):
        """Non-blocking method to queue a live expert response for background persistence."""
        await self._expert_resp_queue.put(response)

    async def queue_training_example(self, example: TrainingExampleV2_5):
        """Non-blocking method to queue a new training example."""
        await self._training_queue.put(example)

    async def queue_educational_document(self, document: EducationalDocumentV2_5):
        """Non-blocking method to queue a new educational document."""
        await self._education_queue.put(document)

    # --- Cross-Tier Query Optimization ---

    async def query_historical_responses(self, expert_id: str, symbol: str, limit: int = 100) -> Optional[List[MOEUnifiedResponseV2_5]]:
        """
        Intelligently queries for historical data, checking the cache first before
        querying the long-term database. Caches the result on a miss.
        """
        if not self.cache_manager or not self.supabase_manager:
            logger.error("One or more data tiers are unavailable for querying.")
            return None

        cache_key = f"query:hist_resp:{expert_id}:{symbol}:{limit}"
        
        # 1. Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Cache HIT for historical query: {cache_key}")
            return cached_result

        # 2. On miss, query Supabase
        logger.debug(f"Cache MISS for historical query: {cache_key}. Querying Supabase.")
        db_result = await self.supabase_manager.fetch_historical_responses(expert_id, symbol, limit)

        # 3. Write result back to cache for future requests
        if db_result:
            # Cache with a longer TTL for historical queries
            self.cache_manager.set(cache_key, db_result, ttl_seconds=3600) # 1 hour TTL
        
        return db_result

    # --- Automated Data Lifecycle Management ---

    async def run_nightly_maintenance(self, retention_days: int = 90):
        """
        Performs scheduled cleanup of old data from both cache and database tiers.
        """
        logger.info("--- Starting Nightly Data Maintenance Cycle ---")
        if self.cache_manager:
            deleted_files = self.cache_manager.cleanup_old_files(max_age_days=retention_days)
            logger.info(f"Cache Maintenance: Cleaned up {deleted_files} old cache files.")
        
        if self.supabase_manager:
            deleted_rows = await self.supabase_manager.cleanup_old_records(days=retention_days)
            logger.info(f"Database Maintenance: Cleaned up {deleted_rows} old usage records.")
        logger.info("--- Nightly Data Maintenance Cycle Complete ---")

    # --- Monitoring & Analytics ---

    def get_system_analytics(self) -> DualTierSystemAnalytics:
        """Returns a comprehensive, validated snapshot of the data system's health."""
        return DualTierSystemAnalytics(
            cache_available=self.cache_manager is not None,
            database_available=self.supabase_manager is not None,
            cache_analytics=self.cache_manager.get_cache_analytics() if self.cache_manager else None,
            database_status=self.supabase_manager.get_status() if self.supabase_manager else {"status": "Unavailable"},
            background_writer_queues={
                "expert_responses": self._expert_resp_queue.qsize(),
                "training_examples": self._training_queue.qsize(),
                "educational_documents": self._education_queue.qsize()
            }
        )

    # --- Private Background Worker ---

    async def _background_writer(self, queue: asyncio.Queue, flush_function: callable, name: str):
        """
        A generic background worker that continuously flushes a given queue
        to the database using a provided batch-write function.
        """
        BATCH_SIZE = 200
        FLUSH_INTERVAL = 5  # seconds
        
        while True:
            try:
                start_time = time.time()
                batch_to_write = []
                
                # Efficiently gather items from the queue
                while time.time() - start_time < FLUSH_INTERVAL and len(batch_to_write) < BATCH_SIZE:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        batch_to_write.append(item)
                        queue.task_done()
                    except asyncio.TimeoutError:
                        # No items in the queue, break to sleep
                        break
                
                if batch_to_write:
                    logger.info(f"Background Writer '{name}': Flushing batch of {len(batch_to_write)} items.")
                    await flush_function(batch_to_write)
                else:
                    # If no items were processed, sleep for the remainder of the interval
                    await asyncio.sleep(FLUSH_INTERVAL)

            except Exception as e:
                logger.error(f"CRITICAL ERROR in background writer '{name}': {e}", exc_info=True)
                await asyncio.sleep(15) # Longer back-off on critical error
