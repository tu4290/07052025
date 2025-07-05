"""
HuiHui Supabase Database Manager
===============================

Manages HuiHui usage monitoring data in Supabase database.
Integrates with existing EOTS database infrastructure for
long-term storage and analysis of usage patterns.

Author: EOTS v2.5 AI Database Division
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Optional asyncpg import
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Import existing database manager
try:
    from data_management.database_manager_v2_5 import DatabaseManagerV2_5
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

# PYDANTIC-FIRST: Replace dataclass with Pydantic model for validation
from pydantic import BaseModel, Field
# NEW: Import adaptive tuning models
try:
    from huihui_integration.learning.adaptive_parameter_tuning import (
        ExpertParameterSet,
        ParameterAdjustmentLog,
    )
    ADAPTIVE_MODELS_AVAILABLE = True
except ImportError:  # Graceful degradation if tuner not imported yet
    ADAPTIVE_MODELS_AVAILABLE = False

class HuiHuiUsageRecordV2_5(BaseModel):
    """Pydantic model for HuiHui usage record for Supabase storage - EOTS v2.5 compliant."""
    expert_name: str = Field(..., description="HuiHui expert name")
    request_type: str = Field(..., description="Type of request")
    input_tokens: int = Field(..., description="Input tokens count", ge=0)
    output_tokens: int = Field(..., description="Output tokens count", ge=0)
    total_tokens: int = Field(..., description="Total tokens used", ge=0)
    processing_time_seconds: float = Field(..., description="Processing time in seconds", ge=0.0)
    success: bool = Field(..., description="Whether request was successful")
    market_condition: str = Field(..., description="Market condition during request")
    vix_level: Optional[float] = Field(None, description="VIX level at request time", ge=0.0)
    symbol: Optional[str] = Field(None, description="Trading symbol if applicable")
    error_type: Optional[str] = Field(None, description="Error type if failed")
    retry_count: int = Field(default=0, description="Number of retries", ge=0)
    timeout_occurred: bool = Field(default=False, description="Whether timeout occurred")
    api_token_hash: Optional[str] = Field(None, description="Hashed API token for tracking")
    user_session_id: Optional[str] = Field(None, description="User session identifier")
    request_metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")
    response_metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Record timestamp")

    class Config:
        extra = 'forbid'

# Legacy alias for backward compatibility
HuiHuiUsageRecord = HuiHuiUsageRecordV2_5

@dataclass
class HuiHuiOptimizationRecommendation:
    """Optimization recommendation for Supabase storage."""
    expert_name: str
    current_rate_limit: int
    current_token_limit: int
    current_timeout_seconds: int
    recommended_rate_limit: int
    recommended_token_limit: int
    recommended_timeout_seconds: int
    confidence_score: float
    urgency_level: str
    market_condition_factor: str
    based_on_requests: int
    analysis_period_hours: int
    peak_usage_factor: float
    reasoning: str
    implementation_priority: int = 5
    estimated_improvement_percent: Optional[float] = None

class HuiHuiSupabaseManager:
    """
    Manages HuiHui usage monitoring data in Supabase.
    
    Features:
    - Store detailed usage records
    - Generate usage patterns
    - Store optimization recommendations
    - Track system health
    - Provide analytics and insights
    """
    
    def __init__(self):
        if not DATABASE_MANAGER_AVAILABLE:
            logger.warning("Database manager not available, Supabase integration disabled")
            self._initialized = False
            return

        self.db_manager = DatabaseManagerV2_5()
        self.connection_pool = None
        self._initialized = False
        # --- Batch-write infrastructure ---
        # Queues for high-throughput writes (filled by public helpers below)
        self._usage_queue: "asyncio.Queue[HuiHuiUsageRecordV2_5]" = asyncio.Queue()
        self._expert_resp_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self._training_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self._education_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        # Metrics for simple monitoring
        self._batch_metrics: Dict[str, int] = {
            "usage_written": 0,
            "expert_resp_written": 0,
            "training_written": 0,
            "education_written": 0,
            "usage_errors": 0,
            "expert_resp_errors": 0,
            "training_errors": 0,
            "education_errors": 0,
        }
    
    async def initialize(self) -> bool:
        """Initialize Supabase connection and ensure tables exist."""
        if not DATABASE_MANAGER_AVAILABLE or not ASYNCPG_AVAILABLE:
            logger.warning("Required dependencies not available for Supabase integration")
            return False

        try:
            # Initialize database manager
            if not await self.db_manager.initialize():
                logger.error("Failed to initialize database manager")
                return False

            # Create connection pool
            self.connection_pool = await self._create_connection_pool()

            # Ensure HuiHui tables exist
            await self._ensure_huihui_tables_exist()

            self._initialized = True
            logger.info("✅ HuiHui Supabase manager initialized successfully")

            # ------------------------------------------------------------------
            # Launch background batch-writer tasks (daemon-style)
            # ------------------------------------------------------------------
            asyncio.create_task(self._background_batch_writer_usage(), name="huihui_usage_batch_writer")
            # Note: We only implement usage batch writer now; other queues can
            # follow the exact same pattern with their own tasks.

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize HuiHui Supabase manager: {e}")
            return False

# ============================================================================+
#                        ─── HIGH-PERFORMANCE BATCH WRITES ───                +
# ============================================================================+

    # ---------------------- Public helper APIs ------------------------------+

    async def batch_store_usage_records(self, records: List[HuiHuiUsageRecordV2_5]) -> None:
        """
        Queue a list of usage records for batched insertion (non-blocking).
        These will be written by the background writer using
        asyncpg.copy_records_to_table for ~10× throughput.
        """
        if not self._initialized:
            logger.warning("Supabase manager not initialized. Dropping usage batch.")
            return
        for rec in records:
            await self._usage_queue.put(rec)

    # Stubs – can be implemented analogously in a follow-up patch
    async def batch_store_expert_responses(self, responses: List[Dict[str, Any]]) -> None:
        for resp in responses:
            await self._expert_resp_queue.put(resp)

    async def batch_store_training_examples(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            await self._training_queue.put(row)

    async def batch_store_educational_documents(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            await self._education_queue.put(doc)

    # ------------------- Background writer coroutine ------------------------+

    async def _background_batch_writer_usage(self):
        """
        Continuously flushes the usage queue to Supabase in batches using the
        COPY binary protocol. Runs until the application exits.
        """
        BATCH_SIZE = 500
        SLEEP_INTERVAL = 1  # seconds
        while True:
            try:
                batch: List[HuiHuiUsageRecordV2_5] = []
                # Collect up to BATCH_SIZE items without blocking indefinitely
                while len(batch) < BATCH_SIZE:
                    try:
                        item = self._usage_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                if batch:
                    await self._flush_usage_batch(batch)
                else:
                    await asyncio.sleep(SLEEP_INTERVAL)
            except Exception as exc:
                self._batch_metrics["usage_errors"] += 1
                logger.error(f"Usage batch writer encountered error: {exc}", exc_info=True)
                await asyncio.sleep(5)  # Simple back-off

    async def _flush_usage_batch(self, batch: List[HuiHuiUsageRecordV2_5]):
        """
        Performs the actual COPY ... FROM insertion using asyncpg's efficient
        copy_records_to_table API. Falls back to individual inserts on failure.
        """
        # Transform records to a sequence of tuples in table column order
        records_iter = (
            (
                rec.expert_name,
                rec.request_type,
                rec.input_tokens,
                rec.output_tokens,
                rec.total_tokens,
                rec.processing_time_seconds,
                rec.success,
                rec.market_condition,
                rec.vix_level,
                rec.symbol,
                rec.error_type,
                rec.retry_count,
                rec.timeout_occurred,
                rec.api_token_hash,
                rec.user_session_id,
                json.dumps(rec.request_metadata or {}),
                json.dumps(rec.response_metadata or {}),
                rec.timestamp,
            )
            for rec in batch
        )

        try:
            async with self.connection_pool.acquire() as conn:
                await conn.copy_records_to_table(
                    "huihui_usage_records",
                    records=records_iter,
                    columns=[
                        "expert_name",
                        "request_type",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "processing_time_seconds",
                        "success",
                        "market_condition",
                        "vix_level",
                        "symbol",
                        "error_type",
                        "retry_count",
                        "timeout_occurred",
                        "api_token_hash",
                        "user_session_id",
                        "request_metadata",
                        "response_metadata",
                        "created_at",
                    ],
                )
            self._batch_metrics["usage_written"] += len(batch)
            logger.debug(f"🚀 Batch-inserted {len(batch)} usage rows via COPY.")
        except Exception as exc:
            # On failure, fall back to slow path
            self._batch_metrics["usage_errors"] += len(batch)
            logger.warning(f"COPY failed ({exc}); falling back to per-row inserts.")
            for rec in batch:
                await self.store_usage_record(rec)

    
    async def _create_connection_pool(self):
        """Create async connection pool to Supabase."""
        try:
            # Get connection details from database manager
            config = self.db_manager.connection_config
            
            pool = await asyncpg.create_pool(
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"],
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            logger.info("✅ Supabase connection pool created")
            return pool
            
        except Exception as e:
            logger.error(f"❌ Failed to create Supabase connection pool: {e}")
            raise
    
    async def _ensure_huihui_tables_exist(self):
        """Ensure HuiHui monitoring tables exist in Supabase."""
        try:
            # Read the SQL schema file
            schema_file = Path("database_schema/huihui_usage_monitoring_tables.sql")
            
            if not schema_file.exists():
                logger.warning("HuiHui schema file not found, skipping table creation")
                return
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            async with self.connection_pool.acquire() as conn:
                await conn.execute(schema_sql)
            
            logger.info("✅ HuiHui monitoring tables ensured in Supabase")
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure HuiHui tables exist: {e}")
            # Don't raise - continue without tables if needed
    
    async def store_usage_record(self, record: HuiHuiUsageRecordV2_5) -> bool:
        """Store a usage record in Supabase."""
        if not self._initialized:
            logger.warning("Supabase manager not initialized, skipping storage")
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO huihui_usage_records (
                        expert_name, request_type, input_tokens, output_tokens, total_tokens,
                        processing_time_seconds, success, market_condition, vix_level, symbol,
                        error_type, retry_count, timeout_occurred, api_token_hash, user_session_id,
                        request_metadata, response_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """, 
                    record.expert_name, record.request_type, record.input_tokens, 
                    record.output_tokens, record.total_tokens, record.processing_time_seconds,
                    record.success, record.market_condition, record.vix_level, record.symbol,
                    record.error_type, record.retry_count, record.timeout_occurred,
                    record.api_token_hash, record.user_session_id,
                    json.dumps(record.request_metadata or {}),
                    json.dumps(record.response_metadata or {})
                )
            
            logger.debug(f"✅ Stored usage record for {record.expert_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store usage record: {e}")
            return False
    
    async def store_optimization_recommendation(self, recommendation: HuiHuiOptimizationRecommendation) -> bool:
        """Store an optimization recommendation in Supabase."""
        if not self._initialized:
            logger.warning("Supabase manager not initialized, skipping storage")
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO huihui_optimization_recommendations (
                        expert_name, current_rate_limit, current_token_limit, current_timeout_seconds,
                        recommended_rate_limit, recommended_token_limit, recommended_timeout_seconds,
                        confidence_score, urgency_level, market_condition_factor, based_on_requests,
                        analysis_period_hours, peak_usage_factor, reasoning, implementation_priority,
                        estimated_improvement_percent
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    recommendation.expert_name, recommendation.current_rate_limit,
                    recommendation.current_token_limit, recommendation.current_timeout_seconds,
                    recommendation.recommended_rate_limit, recommendation.recommended_token_limit,
                    recommendation.recommended_timeout_seconds, recommendation.confidence_score,
                    recommendation.urgency_level, recommendation.market_condition_factor,
                    recommendation.based_on_requests, recommendation.analysis_period_hours,
                    recommendation.peak_usage_factor, recommendation.reasoning,
                    recommendation.implementation_priority, recommendation.estimated_improvement_percent
                )
            
            logger.info(f"✅ Stored optimization recommendation for {recommendation.expert_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store optimization recommendation: {e}")
            return False
    
    async def get_usage_summary(self, expert: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage summary for an expert from Supabase."""
        if not self._initialized:
            return {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_requests,
                        AVG(processing_time_seconds) as avg_processing_time,
                        AVG(total_tokens) as avg_tokens,
                        MAX(total_tokens) as max_tokens,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as success_rate,
                        SUM(CASE WHEN timeout_occurred THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as timeout_rate
                    FROM huihui_usage_records 
                    WHERE expert_name = $1 
                    AND created_at >= NOW() - INTERVAL '%s hours'
                """ % hours, expert)
                
                if result:
                    return {
                        "expert": expert,
                        "hours": hours,
                        "total_requests": result["total_requests"],
                        "avg_processing_time": float(result["avg_processing_time"] or 0),
                        "avg_tokens": float(result["avg_tokens"] or 0),
                        "max_tokens": result["max_tokens"] or 0,
                        "success_rate": float(result["success_rate"] or 0),
                        "timeout_rate": float(result["timeout_rate"] or 0)
                    }
                else:
                    return {"expert": expert, "hours": hours, "total_requests": 0}
                    
        except Exception as e:
            logger.error(f"❌ Failed to get usage summary: {e}")
            return {}
    
    async def get_recent_recommendations(self, expert: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization recommendations from Supabase."""
        if not self._initialized:
            return []
        
        try:
            async with self.connection_pool.acquire() as conn:
                if expert:
                    results = await conn.fetch("""
                        SELECT * FROM huihui_optimization_recommendations 
                        WHERE expert_name = $1 
                        ORDER BY created_at DESC 
                        LIMIT $2
                    """, expert, limit)
                else:
                    results = await conn.fetch("""
                        SELECT * FROM huihui_optimization_recommendations 
                        ORDER BY created_at DESC 
                        LIMIT $1
                    """, limit)
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Failed to get recent recommendations: {e}")
            return []
    
    async def update_system_health(self, health_data: Dict[str, Any]) -> bool:
        """Update system health record in Supabase."""
        if not self._initialized:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO huihui_system_health (
                        cpu_usage_percent, memory_usage_percent, gpu_memory_used_percent,
                        gpu_temperature, gpu_load_percent, ollama_healthy, ollama_response_time_ms,
                        experts_available, total_requests_last_hour, avg_response_time_last_hour,
                        error_rate_last_hour, current_market_condition, current_vix_level
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    health_data.get("cpu_usage"),
                    health_data.get("memory_usage"),
                    health_data.get("gpu_memory_used"),
                    health_data.get("gpu_temperature"),
                    health_data.get("gpu_load"),
                    health_data.get("ollama_healthy", True),
                    health_data.get("ollama_response_time_ms"),
                    json.dumps(health_data.get("experts_available", {})),
                    health_data.get("total_requests_last_hour", 0),
                    health_data.get("avg_response_time_last_hour", 0.0),
                    health_data.get("error_rate_last_hour", 0.0),
                    health_data.get("current_market_condition", "normal"),
                    health_data.get("current_vix_level")
                )
            
            logger.debug("✅ Updated system health in Supabase")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update system health: {e}")
            return False
    
    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old usage records from Supabase."""
        if not self._initialized:
            return 0
        
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM huihui_usage_records 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """ % days)
                
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1]) if result else 0
                
                logger.info(f"✅ Cleaned up {deleted_count} old usage records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old records: {e}")
            return 0
    
    async def close(self):
        """Close the connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("✅ Supabase connection pool closed")

# ============================================================================
#               ─── ADAPTIVE PARAMETER-TUNING SUPPORT METHODS ───
# ============================================================================

    # ---------- CONFIG TABLES ----------

    async def save_parameter_configs(self, expert_config: "ExpertParameterSet") -> bool:
        """
        Store or update parameter configuration rows for an expert.
        Uses ON CONFLICT to keep the operation idempotent.
        """
        if not self._initialized or not ADAPTIVE_MODELS_AVAILABLE:
            return False

        try:
            async with self.connection_pool.acquire() as conn:
                stmt = """
                    INSERT INTO expert_parameter_configs
                        (expert_id, parameter_name, initial_value,
                         min_bound, max_bound, description)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (expert_id, parameter_name)
                    DO UPDATE SET
                        initial_value = EXCLUDED.initial_value,
                        min_bound = EXCLUDED.min_bound,
                        max_bound = EXCLUDED.max_bound,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """
                for p in expert_config.parameters:
                    await conn.execute(
                        stmt,
                        expert_config.expert_id,
                        p.name,
                        p.initial_value,
                        p.min_bound,
                        p.max_bound,
                        p.description,
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to save parameter configs: {e}", exc_info=True)
            return False

    async def fetch_all_parameter_configs(self) -> List[Dict[str, Any]]:
        """Fetch every parameter config row."""
        if not self._initialized:
            return []
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM expert_parameter_configs")
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch parameter configs: {e}", exc_info=True)
            return []

    # ---------- CURRENT PARAM VALUES ----------

    async def save_current_parameter(
        self,
        expert_id: str,
        param_name: str,
        market_regime: str,
        new_value: float,
    ) -> bool:
        """UPSERT current live parameter value."""
        if not self._initialized:
            return False
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO current_parameter_values
                        (config_id, market_regime, current_value, last_updated_by)
                    SELECT id, $4, $5, 'adaptive_tuner'
                    FROM expert_parameter_configs
                    WHERE expert_id = $1 AND parameter_name = $2
                    ON CONFLICT (config_id, market_regime)
                    DO UPDATE SET
                        current_value = EXCLUDED.current_value,
                        last_updated_by = 'adaptive_tuner',
                        updated_at = NOW()
                    """,
                    expert_id,
                    param_name,
                    expert_id,  # placeholder to keep param order (not used)
                    market_regime,
                    new_value,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save current parameter value: {e}", exc_info=True)
            return False

    async def fetch_all_current_parameters(self) -> List[Dict[str, Any]]:
        """Return all current parameter rows."""
        if not self._initialized:
            return []
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT e.expert_id,
                           e.parameter_name,
                           c.market_regime,
                           c.current_value
                    FROM current_parameter_values c
                    JOIN expert_parameter_configs e ON e.id = c.config_id
                    """
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch current parameters: {e}", exc_info=True)
            return []

    # ---------- PERFORMANCE OUTCOMES ----------

    async def save_performance_outcome(
        self,
        expert_id: str,
        prediction_id: str,
        parameters_used: Dict[str, float],
        market_regime: str,
        outcome_metric: float,
    ) -> bool:
        """Insert a single performance outcome row."""
        if not self._initialized:
            return False
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO performance_outcomes
                        (expert_id, prediction_id, outcome_metric,
                         market_regime, parameters_used)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (prediction_id) DO NOTHING
                    """,
                    expert_id,
                    prediction_id,
                    outcome_metric,
                    market_regime,
                    json.dumps(parameters_used),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save performance outcome: {e}", exc_info=True)
            return False

    async def fetch_performance_for_parameter(
        self, expert_id: str, param_name: str, market_regime: str, lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Return aggregated performance by value for a single parameter."""
        if not self._initialized:
            return []
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH perf AS (
                        SELECT (parameters_used->>$3)::FLOAT AS value_tried,
                               outcome_metric
                        FROM performance_outcomes
                        WHERE expert_id = $1
                          AND market_regime = $2
                          AND created_at >= NOW() - INTERVAL '$4 days'
                          AND parameters_used ? $3
                    )
                    SELECT value_tried,
                           AVG(outcome_metric) AS avg_performance,
                           COUNT(*)            AS trial_count
                    FROM perf
                    GROUP BY value_tried
                    """,
                    expert_id,
                    market_regime,
                    param_name,
                    lookback_days,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch performance data: {e}", exc_info=True)
            return []

    # ---------- ADJUSTMENT LOG ----------

    async def save_adjustment_log(self, log_entry: "ParameterAdjustmentLog") -> bool:
        """Persist an immutable adjustment log row."""
        if not self._initialized or not ADAPTIVE_MODELS_AVAILABLE:
            return False
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO parameter_adjustment_logs
                        (parameter_config_id, market_regime,
                         old_value, new_value, reason, performance_delta)
                    SELECT id, $3, $4, $5, $6, $7
                    FROM expert_parameter_configs
                    WHERE expert_id = $1 AND parameter_name = $2
                    """,
                    log_entry.expert_id,
                    log_entry.parameter_name,
                    log_entry.market_regime,
                    log_entry.old_value,
                    log_entry.new_value,
                    log_entry.reason,
                    log_entry.performance_delta,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save adjustment log: {e}", exc_info=True)
            return False

    # ---------- OPTIMIZER STATE ----------

    async def save_optimizer_state(self, optimizer_key: str, state_dict: Dict[str, Any]) -> bool:
        """Upsert serialized optimizer state."""
        if not self._initialized:
            return False
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bayesian_optimizer_state (optimizer_key, state_json)
                    VALUES ($1, $2)
                    ON CONFLICT (optimizer_key)
                    DO UPDATE SET state_json = EXCLUDED.state_json,
                                  last_updated = NOW()
                    """,
                    optimizer_key,
                    json.dumps(state_dict),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}", exc_info=True)
            return False

    async def load_optimizer_state(self, optimizer_key: str) -> Optional[Dict[str, Any]]:
        """Load optimizer state from DB or return None."""
        if not self._initialized:
            return None
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT state_json FROM bayesian_optimizer_state WHERE optimizer_key = $1",
                    optimizer_key,
                )
                return dict(row["state_json"]) if row else None
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}", exc_info=True)
            return None

# Global Supabase manager instance
_supabase_manager = None

async def get_supabase_manager() -> HuiHuiSupabaseManager:
    """Get global Supabase manager instance."""
    global _supabase_manager
    if _supabase_manager is None:
        _supabase_manager = HuiHuiSupabaseManager()
        await _supabase_manager.initialize()
    return _supabase_manager

async def store_usage_in_supabase(expert: str, request_type: str, input_tokens: int, 
                                 output_tokens: int, processing_time: float, success: bool,
                                 market_condition: str = "normal", **kwargs) -> bool:
    """Convenience function to store usage record in Supabase."""
    manager = await get_supabase_manager()
    
    record = HuiHuiUsageRecordV2_5(
        expert_name=expert,
        request_type=request_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        processing_time_seconds=processing_time,
        success=success,
        market_condition=market_condition,
        **kwargs
    )
    
    return await manager.store_usage_record(record)

# ===== TESTING FUNCTION =====

async def test_supabase_manager():
    """Test the Supabase manager functionality."""
    print("🗄️ Testing HuiHui Supabase Manager...")
    
    manager = await get_supabase_manager()
    
    # Test storing usage record
    record = HuiHuiUsageRecordV2_5(
        expert_name="market_regime",
        request_type="analysis",
        input_tokens=1500,
        output_tokens=800,
        total_tokens=2300,
        processing_time_seconds=2.5,
        success=True,
        market_condition="volatile",
        vix_level=25.3,
        symbol="SPY"
    )
    
    success = await manager.store_usage_record(record)
    print(f"✅ Usage record stored: {success}")
    
    # Test getting usage summary
    summary = await manager.get_usage_summary("market_regime", 24)
    print(f"✅ Usage summary: {summary}")
    
    # Test system health update
    health_data = {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "ollama_healthy": True,
        "current_market_condition": "volatile",
        "current_vix_level": 25.3
    }
    
    health_success = await manager.update_system_health(health_data)
    print(f"✅ System health updated: {health_success}")
    
    await manager.close()
    print("✅ Supabase manager test completed")

if __name__ == "__main__":
    asyncio.run(test_supabase_manager())
