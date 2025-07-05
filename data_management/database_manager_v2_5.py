# data_management/database_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, datetime
import pandas as pd
import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from psycopg import sql  # Import sql module explicitly
# Note: Using built-in psycopg connection management instead of psycopg_pool
from contextlib import contextmanager
import threading
from pydantic import BaseModel, Field, ConfigDict

# --- Module-level initialization ---
logger = logging.getLogger(__name__)

# Load environment variables once at module import
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from: {env_path}")
else:
    logger.warning(f"No .env file found at: {env_path}")

# PostgreSQL availability check
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("PostgreSQL (psycopg) not available. Database functionality will be limited.")

try:
    from typing import LiteralString
except ImportError:
    LiteralString = str  # Fallback for Python < 3.11

class DatabaseManagerV2_5:
    """
    Manages database interactions for the EOTS v2.5 system.
    SUPABASE-ONLY: PostgreSQL backend exclusively - no SQLite support.
    """

    def __init__(self, config_manager=None):
        """
        Initialize DatabaseManagerV2_5 with Pydantic-first architecture.

        Args:
            config_manager: ConfigManagerV2_5 instance with validated Pydantic config
        """
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Class-level flag to track if we've logged the connection info
        if not hasattr(DatabaseManagerV2_5, '_connection_logged'):
            DatabaseManagerV2_5._connection_logged = False

        # PYDANTIC-FIRST: Use ConfigManagerV2_5 with validated schemas
        self.config_manager = config_manager
        self._conn = None
        self.db_type = None
        self.connection_status = "DISCONNECTED"
        
        # Simple connection management to prevent connection exhaustion
        self._connection_lock = threading.Lock()
        self._connection_params = None
        
        # Implement connection retry logic to handle connection exhaustion
        self._connect_with_retry()

    def _connect_with_retry(self, max_retries=3, base_delay=1.0):
        """
        Connect to database with exponential backoff retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
        """
        import time
        import random
        
        for attempt in range(max_retries + 1):
            try:
                # Only log on first attempt to avoid spam
                log_connection = (attempt == 0)
                self._connect(log_connection=log_connection)
                return  # Success!
                
            except Exception as e:
                if "remaining connection slots are reserved" in str(e) and attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(f"Connection attempt {attempt + 1} failed due to connection exhaustion. "
                                      f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    # Either not a connection exhaustion error, or we've exhausted retries
                    if attempt == max_retries:
                        self.logger.error(f"Failed to connect after {max_retries + 1} attempts. "
                                        f"Database may be at connection limit.")
                    raise

    @property
    def db_config(self):
        """Provide access to database config for compatibility."""
        if self.config_manager and hasattr(self.config_manager, 'config'):
            return self.config_manager.config.model_dump().get('database_settings')
        return None

    def _connect(self, log_connection: bool = True):
        """
        Connect to database - prioritize Supabase/PostgreSQL over SQLite.
        
        Args:
            log_connection: Whether to log connection messages
        """
        # Check if we have Supabase/PostgreSQL environment variables
        has_supabase_config = any([
            os.getenv('EOTS_DB_URL'),
            os.getenv('EOTS_DB_USER'),
            os.getenv('EOTS_DB_PASSWORD'),
            os.getenv('EOTS_DB_NAME'),
            os.getenv('EOTS_DB_HOST')
        ])

        # FORCE Supabase/PostgreSQL - NO SQLite fallback
        if has_supabase_config and POSTGRES_AVAILABLE:
            if log_connection and not DatabaseManagerV2_5._connection_logged:
                self.logger.info("🔌 Connecting to Supabase/PostgreSQL...")
                DatabaseManagerV2_5._connection_logged = True
            self._connect_postgres(log_connection=log_connection)
        else:
            # NO SQLite fallback - Supabase is required
            error_msg = "❌ EOTS v2.5 requires Supabase database connection - SQLite not supported. " \
                      "Please configure Supabase environment variables: " \
                      "EOTS_DB_USER, EOTS_DB_PASSWORD, EOTS_DB_HOST, EOTS_DB_NAME"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    # REMOVED: SQLite connection method - Supabase only
    # def _connect_sqlite(self, db_path: str):
    #     """REMOVED: SQLite support - EOTS v2.5 uses Supabase exclusively"""

    def _connect_postgres(self, log_connection: bool = True):
        """
        Connect to PostgreSQL database.
        
        Args:
            log_connection: Whether to log connection details
        """
        # Only log connection details if requested and not already shown
        if log_connection and not DatabaseManagerV2_5._connection_logged:
            self.logger.debug("🔍 Supabase database connection parameters:")
            self.logger.debug(f"  Host: {os.getenv('EOTS_DB_HOST')}")
            self.logger.debug(f"  Database: {os.getenv('EOTS_DB_NAME')}")
            DatabaseManagerV2_5._connection_logged = True
            
        if not POSTGRES_AVAILABLE:
            raise ImportError("❌ PostgreSQL dependencies not available")

        # SUPABASE-ONLY: Extract database settings - NO localhost defaults
        db_url = None
        db_user = None
        db_pass = None
        db_name = None
        db_host = None  # NEVER default to localhost for Supabase
        db_port = 5432

        try:
            # Try to get settings from Pydantic config
            if self.config_manager and hasattr(self.config_manager, 'config'):
                config = self.config_manager.config.model_dump()
                if config.get('database_settings'):
                    db_settings = config['database_settings']
                    db_url = db_settings.get('db_url')
                    db_user = db_settings.get('user')
                    db_pass = db_settings.get('password')
                    db_name = db_settings.get('database')
                    db_host = db_settings.get('host')  # NO localhost default
                    db_port = db_settings.get('port', 5432)
        except Exception as e:
            self.logger.debug(f"Could not extract database settings from config: {e}")

        # Get from environment variables - REQUIRED for Supabase
        db_url = db_url or os.getenv('EOTS_DB_URL')
        db_user = db_user or os.getenv('EOTS_DB_USER')
        db_pass = db_pass or os.getenv('EOTS_DB_PASSWORD')
        db_name = db_name or os.getenv('EOTS_DB_NAME')
        db_host = db_host or os.getenv('EOTS_DB_HOST')  # NO localhost fallback
        db_port = db_port or int(os.getenv('EOTS_DB_PORT', 5432))

        # Validate required parameters
        if not all([db_host, db_user, db_pass, db_name]):
            missing = []
            if not db_host: missing.append('EOTS_DB_HOST')
            if not db_user: missing.append('EOTS_DB_USER')
            if not db_pass: missing.append('EOTS_DB_PASSWORD')
            if not db_name: missing.append('EOTS_DB_NAME')
            raise ValueError(f"Missing required database environment variables: {', '.join(missing)}")
        # Only log connection details if this is the first connection
        if log_connection and not hasattr(DatabaseManagerV2_5, '_connection_logged_full'):
            self.logger.info("🔌 Connecting to Supabase database...")
            self.logger.debug(f"📡 Connection details: {db_user}@{db_host}:{db_port}/{db_name}")
            DatabaseManagerV2_5._connection_logged_full = True
            
        try:
            self._conn = psycopg.connect(
                dbname=db_name,
                user=db_user,
                password=db_pass,
                host=db_host,
                port=db_port,
                autocommit=True,
                row_factory=dict_row
            )
            self.db_type = "postgresql"
            self.connection_status = "CONNECTED"
            if log_connection and not hasattr(DatabaseManagerV2_5, '_connection_success_logged'):
                self.logger.info("✅ Successfully connected to Supabase database")
                DatabaseManagerV2_5._connection_success_logged = True
        except Exception as e:
            self.connection_status = "FAILED"
            if log_connection and not hasattr(DatabaseManagerV2_5, '_connection_error_logged'):
                self.logger.critical(f"❌ Failed to connect to PostgreSQL database: {e}")
                DatabaseManagerV2_5._connection_error_logged = True
            raise

    def get_connection(self) -> Any:
        """Get database connection - use with caution, prefer get_connection_context()"""
        return self._conn
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for proper connection handling"""
        with self._connection_lock:
            if not self._conn:
                self._connect(log_connection=False)
            try:
                yield self._conn
            except Exception as e:
                self.logger.error(f"Database operation failed: {e}")
                # Reconnect on failure
                try:
                    self._connect(log_connection=False)
                except Exception:
                    pass
                raise

    def close_connection(self):
        if self._conn:
            self._conn.close()
            self.logger.info("Database connection closed.")
            self.connection_status = "CLOSED"

    def table_exists(self, table_name: str, schema_name: Optional[str] = None) -> bool:
        """Check if a table exists in the Supabase database across all schemas."""
        try:
            # SUPABASE-ONLY: PostgreSQL syntax - check all schemas if none specified
            cursor = self._conn.cursor()

            if schema_name:
                # Check specific schema
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s
                        AND table_name = %s
                    )
                """, (schema_name, table_name))
            else:
                # Check all schemas (public, auth, storage, etc.)
                cursor.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_name = %s
                    ORDER BY table_schema
                """, (table_name,))
                results = cursor.fetchall()

                if results:
                    # Log which schemas contain this table
                    schemas = [row[0] for row in results]
                    self.logger.info(f"Table '{table_name}' found in schemas: {schemas}")
                    return True
                else:
                    # List all available tables for debugging
                    cursor.execute("""
                        SELECT table_schema, table_name
                        FROM information_schema.tables
                        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                        ORDER BY table_schema, table_name
                    """)
                    all_tables = cursor.fetchall()
                    self.logger.warning(f"Table '{table_name}' not found. Available tables:")
                    for schema, tname in all_tables[:20]:  # Show first 20 tables
                        self.logger.warning(f"  {schema}.{tname}")
                    if len(all_tables) > 20:
                        self.logger.warning(f"  ... and {len(all_tables) - 20} more tables")
                    return False

            result = cursor.fetchone()
            return result[0] if result else False
        except Exception as e:
            self.logger.error(f"Failed to check table existence: {e}")
            return False

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a Supabase table."""
        try:
            # SUPABASE-ONLY: PostgreSQL syntax
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                )
            """, (table_name, column_name))
            result = cursor.fetchone()
            return result[0] if result else False
        except Exception as e:
            self.logger.error(f"Failed to check column existence: {e}")
            return False

    def get_datetime_sql(self, days_ago: int = 0) -> str:
        """Get Supabase PostgreSQL datetime SQL."""
        # SUPABASE-ONLY: PostgreSQL syntax
        if days_ago == 0:
            return "NOW()"
        else:
            return f"NOW() - INTERVAL '{days_ago} days'"

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Dict]]:
        """Execute a query and return results."""
        if not self._conn:
            self.logger.error("No database connection")
            return None
        
        try:
            with self._conn.cursor() as cur:
                if params:
                    cur.execute(query, params)  # Psycopg handles parameter binding safely
                else:
                    cur.execute(query)  # Direct execution for non-parameterized queries
                
                if cur.description:  # Check if query returns data
                    results = cur.fetchall()
                    return results if results else []
                return None
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return None

    def initialize_database_schema(self) -> None:
        """Initialize Supabase PostgreSQL database schema."""
        # SUPABASE-ONLY: PostgreSQL schema
        self._initialize_postgres_schema()

    # REMOVED: SQLite schema initialization - Supabase only
        # REMOVED: All SQLite schema code - Supabase PostgreSQL only

    def _initialize_postgres_schema(self) -> None:
        """Initialize PostgreSQL-specific schema with all EOTS v2.5 tables."""
        # Core OHLCV table
        sql_create_daily_ohlcv = """
        CREATE TABLE IF NOT EXISTS daily_ohlcv (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date DATE NOT NULL,
            open NUMERIC(12, 4) NOT NULL, high NUMERIC(12, 4) NOT NULL,
            low NUMERIC(12, 4) NOT NULL, close NUMERIC(12, 4) NOT NULL,
            volume BIGINT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, date)
        );"""

        # Daily EOTS Metrics Table
        sql_create_daily_eots_metrics = """
        CREATE TABLE IF NOT EXISTS daily_eots_metrics (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            gib_oi_based_und NUMERIC(12, 4),
            ivsdh_und_avg NUMERIC(12, 4),
            vapi_fa_z_score_und NUMERIC(12, 4),
            dwfd_z_score_und NUMERIC(12, 4),
            tw_laf_z_score_und NUMERIC(12, 4),
            market_regime_summary TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, date)
        );"""

        # Trade Outcomes Table
        sql_create_trade_outcomes = """
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            entry_timestamp TIMESTAMPTZ NOT NULL,
            exit_timestamp TIMESTAMPTZ,
            strategy_type TEXT NOT NULL,
            outcome TEXT NOT NULL,
            pnl_percentage NUMERIC(8, 4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # ATIF Recommendations Table
        sql_create_atif_recommendations = """
        CREATE TABLE IF NOT EXISTS atif_recommendations (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            recommendation_text TEXT NOT NULL,
            confidence_score NUMERIC(5, 3),
            status TEXT DEFAULT 'ACTIVE',
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # Daily EOTS Metrics Table
        sql_create_daily_eots_metrics = """
        CREATE TABLE IF NOT EXISTS daily_eots_metrics (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            gib_oi_based_und NUMERIC(12, 4),
            ivsdh_und_avg NUMERIC(12, 4),
            vapi_fa_z_score_und NUMERIC(12, 4),
            dwfd_z_score_und NUMERIC(12, 4),
            tw_laf_z_score_und NUMERIC(12, 4),
            market_regime_summary TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, date)
        );"""

        # Trade Outcomes Table
        sql_create_trade_outcomes = """
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            entry_timestamp TIMESTAMPTZ NOT NULL,
            exit_timestamp TIMESTAMPTZ,
            strategy_type TEXT NOT NULL,
            outcome TEXT NOT NULL,
            pnl_percentage NUMERIC(8, 4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # ATIF Recommendations Table
        sql_create_atif_recommendations = """
        CREATE TABLE IF NOT EXISTS atif_recommendations (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            recommendation_text TEXT NOT NULL,
            confidence_score NUMERIC(5, 3),
            status TEXT DEFAULT 'ACTIVE',
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # ATIF Performance Insights Table
        sql_create_atif_insights = """
        CREATE TABLE IF NOT EXISTS atif_insights (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            insight_type TEXT NOT NULL,
            insight_category TEXT NOT NULL,
            insight_title TEXT NOT NULL,
            insight_description TEXT NOT NULL,
            confidence_score NUMERIC(5, 3),
            impact_score NUMERIC(5, 3),
            time_horizon TEXT,
            supporting_metrics JSONB,
            related_recommendations TEXT[],
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # ATIF Performance Tracking Table
        sql_create_atif_performance = """
        CREATE TABLE IF NOT EXISTS atif_performance (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            conviction_score NUMERIC(5, 3),
            outcome TEXT NOT NULL,
            pnl_percentage NUMERIC(8, 4),
            hold_duration_hours INTEGER,
            entry_timestamp TIMESTAMPTZ NOT NULL,
            exit_timestamp TIMESTAMPTZ,
            exit_reason TEXT,
            performance_metrics JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # AI Learning Patterns Table
        sql_create_ai_learning_patterns = """
        CREATE TABLE IF NOT EXISTS ai_learning_patterns (
            id SERIAL PRIMARY KEY,
            pattern_name TEXT NOT NULL,
            pattern_type TEXT NOT NULL DEFAULT 'market_pattern',
            pattern_description TEXT,
            market_conditions JSONB DEFAULT '{}',
            success_rate NUMERIC(5, 4) DEFAULT 0.0000,
            confidence_score NUMERIC(5, 4) DEFAULT 0.0000,
            sample_size INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_updated TIMESTAMPTZ DEFAULT NOW()
        );"""

        # AI Insights History Table
        sql_create_ai_insights_history = """
        CREATE TABLE IF NOT EXISTS ai_insights_history (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            insight_type TEXT NOT NULL,
            insight_content TEXT NOT NULL,
            confidence_score NUMERIC(5, 4) DEFAULT 0.0000,
            impact_score NUMERIC(5, 4) DEFAULT 0.0000,
            market_regime TEXT,
            market_context JSONB DEFAULT '{}',
            outcome_verified BOOLEAN DEFAULT FALSE,
            outcome_accuracy NUMERIC(5, 4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # AI Predictions Table
        sql_create_ai_predictions = """
        CREATE TABLE IF NOT EXISTS ai_predictions (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_type TEXT NOT NULL,
            prediction_value NUMERIC(10, 4),
            prediction_direction TEXT CHECK (prediction_direction IN ('UP', 'DOWN', 'NEUTRAL')),
            confidence_score NUMERIC(5, 4) DEFAULT 0.0000,
            predicted_confidence NUMERIC(5, 4) DEFAULT 0.0000,
            time_horizon TEXT NOT NULL,
            prediction_timestamp TIMESTAMPTZ DEFAULT NOW(),
            target_timestamp TIMESTAMPTZ,
            actual_value NUMERIC(10, 4),
            actual_direction TEXT CHECK (actual_direction IN ('UP', 'DOWN', 'NEUTRAL')),
            prediction_accurate BOOLEAN,
            accuracy_score NUMERIC(5, 4),
            model_version TEXT DEFAULT 'v2.5',
            market_context JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),

            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            CONSTRAINT chk_predicted_confidence CHECK (predicted_confidence >= 0.0 AND predicted_confidence <= 1.0),
            CONSTRAINT chk_accuracy_score CHECK (accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0))
        );"""

        # AI Model Performance Table
        sql_create_ai_model_performance = """
        CREATE TABLE IF NOT EXISTS ai_model_performance (
            id SERIAL PRIMARY KEY,
            model_type TEXT NOT NULL,
            prediction_type TEXT NOT NULL,
            accuracy_score NUMERIC(5, 4) DEFAULT 0.0000,
            precision_score NUMERIC(5, 4) DEFAULT 0.0000,
            recall_score NUMERIC(5, 4) DEFAULT 0.0000,
            f1_score NUMERIC(5, 4) DEFAULT 0.0000,
            sample_period_start TIMESTAMPTZ,
            sample_period_end TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        # Memory Intelligence Tables
        sql_create_memory_entities = """
        CREATE TABLE IF NOT EXISTS memory_entities (
            entity_id VARCHAR(36) PRIMARY KEY,
            entity_type VARCHAR(50) NOT NULL,
            entity_name VARCHAR(255) NOT NULL,
            description TEXT,
            symbol VARCHAR(20) NOT NULL,
            confidence_score DECIMAL(5,4) DEFAULT 0.0000,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_accessed TIMESTAMPTZ DEFAULT NOW(),
            access_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
        );"""

        sql_create_memory_relations = """
        CREATE TABLE IF NOT EXISTS memory_relations (
            relation_id VARCHAR(36) PRIMARY KEY,
            source_entity_id VARCHAR(36) NOT NULL,
            target_entity_id VARCHAR(36) NOT NULL,
            relation_type VARCHAR(50) NOT NULL,
            relation_strength DECIMAL(5,4) DEFAULT 0.5000,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_validated TIMESTAMPTZ DEFAULT NOW(),
            validation_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            CONSTRAINT chk_relation_strength CHECK (relation_strength >= 0.0 AND relation_strength <= 1.0)
        );"""

        sql_create_memory_observations = """
        CREATE TABLE IF NOT EXISTS memory_observations (
            observation_id VARCHAR(36) PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            observation_type VARCHAR(50) NOT NULL,
            description TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            success_rate DECIMAL(5,4) DEFAULT 0.0000,
            metadata JSONB DEFAULT '{}',
            related_entity_id VARCHAR(36),
            validation_status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0)
        );"""

        # HuiHui Expert Schema Tables
        sql_create_huihui_market_regime = """
        CREATE TABLE IF NOT EXISTS huihui_market_regime_analysis (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            analysis_id VARCHAR(100) NOT NULL UNIQUE,
            expert_type VARCHAR(20) NOT NULL DEFAULT 'market_regime',
            ticker VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            current_regime_id INTEGER NOT NULL,
            regime_confidence DECIMAL(5,4) NOT NULL,
            regime_probability DECIMAL(5,4) NOT NULL,
            regime_name VARCHAR(50) NOT NULL,
            regime_description TEXT,
            vri_3_composite DECIMAL(8,4) NOT NULL,
            volatility_regime_score DECIMAL(8,4) NOT NULL,
            flow_intensity_score DECIMAL(8,4) NOT NULL,
            regime_stability_score DECIMAL(8,4) NOT NULL,
            transition_momentum_score DECIMAL(8,4) NOT NULL,
            volatility_level VARCHAR(20) NOT NULL,
            trend_direction VARCHAR(20) NOT NULL,
            flow_pattern VARCHAR(30) NOT NULL,
            risk_appetite VARCHAR(20) NOT NULL,
            predicted_regime_id INTEGER,
            transition_probability DECIMAL(5,4),
            expected_transition_timeframe VARCHAR(30),
            transition_confidence DECIMAL(5,4),
            confidence_score DECIMAL(5,4) NOT NULL,
            processing_time_ms INTEGER NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            data_quality_score DECIMAL(5,4) NOT NULL,
            supporting_indicators JSONB DEFAULT '{}',
            regime_factors JSONB DEFAULT '{}',
            historical_context JSONB DEFAULT '{}',
            quality_metrics JSONB DEFAULT '{}',
            error_flags TEXT[],
            warning_flags TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_regime_confidence CHECK (regime_confidence >= 0.0 AND regime_confidence <= 1.0),
            CONSTRAINT chk_regime_probability CHECK (regime_probability >= 0.0 AND regime_probability <= 1.0),
            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            CONSTRAINT chk_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
            CONSTRAINT chk_volatility_level CHECK (volatility_level IN ('low', 'medium', 'high', 'extreme')),
            CONSTRAINT chk_trend_direction CHECK (trend_direction IN ('bullish', 'bearish', 'neutral', 'mixed')),
            CONSTRAINT chk_risk_appetite CHECK (risk_appetite IN ('risk_on', 'risk_off', 'neutral', 'mixed'))
        );"""

        sql_create_huihui_options_flow = """
        CREATE TABLE IF NOT EXISTS huihui_options_flow_analysis (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            analysis_id VARCHAR(100) NOT NULL UNIQUE,
            expert_type VARCHAR(20) NOT NULL DEFAULT 'options_flow',
            ticker VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            vapi_fa_z_score DECIMAL(8,4) NOT NULL,
            dwfd_z_score DECIMAL(8,4) NOT NULL,
            tw_laf_score DECIMAL(8,4) NOT NULL,
            gib_oi_based DECIMAL(8,4) NOT NULL,
            flow_intensity VARCHAR(20) NOT NULL,
            flow_direction VARCHAR(20) NOT NULL,
            flow_pattern VARCHAR(30) NOT NULL,
            unusual_activity_detected BOOLEAN DEFAULT FALSE,
            confidence_score DECIMAL(5,4) NOT NULL,
            processing_time_ms INTEGER NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            data_quality_score DECIMAL(5,4) NOT NULL,
            flow_metrics JSONB DEFAULT '{}',
            unusual_patterns JSONB DEFAULT '{}',
            quality_metrics JSONB DEFAULT '{}',
            error_flags TEXT[],
            warning_flags TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            CONSTRAINT chk_data_quality_score CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
            CONSTRAINT chk_flow_intensity CHECK (flow_intensity IN ('low', 'medium', 'high', 'extreme')),
            CONSTRAINT chk_flow_direction CHECK (flow_direction IN ('bullish', 'bearish', 'neutral', 'mixed'))
        );"""

        # MOE Expert Schema Tables
        sql_create_moe_expert_registry = """
        CREATE TABLE IF NOT EXISTS moe_expert_registry (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            expert_id VARCHAR(50) NOT NULL UNIQUE,
            expert_name VARCHAR(100) NOT NULL,
            expert_type VARCHAR(30) NOT NULL,
            expert_category VARCHAR(30) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            specialization_domains TEXT[] NOT NULL,
            capability_scores JSONB DEFAULT '{}',
            success_rate DECIMAL(5,4) DEFAULT 0.0,
            avg_response_time_ms DECIMAL(8,2) DEFAULT 0.0,
            total_queries_processed INTEGER DEFAULT 0,
            confidence_reliability DECIMAL(5,4) DEFAULT 0.0,
            computational_cost DECIMAL(8,4) DEFAULT 0.0,
            memory_requirements_mb INTEGER DEFAULT 0,
            max_concurrent_requests INTEGER DEFAULT 1,
            status VARCHAR(20) DEFAULT 'active',
            availability_score DECIMAL(5,4) DEFAULT 1.0,
            last_health_check TIMESTAMPTZ DEFAULT NOW(),
            description TEXT,
            keywords TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_expert_status CHECK (status IN ('active', 'inactive', 'maintenance', 'deprecated')),
            CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
            CONSTRAINT chk_availability_score CHECK (availability_score >= 0.0 AND availability_score <= 1.0)
        );"""

        # Historical Analysis Tables
        sql_create_daily_regime_snapshots = """
        CREATE TABLE IF NOT EXISTS daily_regime_snapshots (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            snapshot_date DATE NOT NULL,
            regime_id INTEGER NOT NULL,
            regime_name VARCHAR(50) NOT NULL,
            regime_confidence DECIMAL(5,4) NOT NULL,
            vri_3_composite DECIMAL(8,4) NOT NULL,
            volatility_regime_score DECIMAL(8,4) NOT NULL,
            flow_intensity_score DECIMAL(8,4) NOT NULL,
            regime_stability_score DECIMAL(8,4) NOT NULL,
            transition_momentum_score DECIMAL(8,4) NOT NULL,
            market_conditions JSONB DEFAULT '{}',
            performance_metrics JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, snapshot_date)
        );"""

        sql_create_key_level_performance = """
        CREATE TABLE IF NOT EXISTS key_level_performance (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            level_type VARCHAR(20) NOT NULL,
            level_value DECIMAL(12,4) NOT NULL,
            test_date DATE NOT NULL,
            test_outcome VARCHAR(20) NOT NULL,
            price_action_quality DECIMAL(5,4),
            volume_confirmation BOOLEAN DEFAULT FALSE,
            follow_through_strength DECIMAL(5,4),
            market_regime VARCHAR(50),
            context_factors JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        sql_create_moe_gating_network = """
        CREATE TABLE IF NOT EXISTS moe_gating_network (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            routing_id VARCHAR(100) NOT NULL UNIQUE,
            query_hash VARCHAR(64) NOT NULL,
            query_type VARCHAR(30) NOT NULL,
            query_complexity_score DECIMAL(5,4) NOT NULL,
            selected_experts TEXT[] NOT NULL,
            expert_weights JSONB NOT NULL,
            routing_confidence DECIMAL(5,4) NOT NULL,
            routing_strategy VARCHAR(30) NOT NULL,
            load_balancing_factor DECIMAL(5,4) DEFAULT 1.0,
            expected_response_time_ms INTEGER,
            actual_response_time_ms INTEGER,
            routing_success BOOLEAN DEFAULT TRUE,
            fallback_triggered BOOLEAN DEFAULT FALSE,
            fallback_reason TEXT,
            performance_score DECIMAL(5,4),
            cost_efficiency DECIMAL(5,4),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_routing_confidence CHECK (routing_confidence >= 0.0 AND routing_confidence <= 1.0),
            CONSTRAINT chk_query_complexity CHECK (query_complexity_score >= 0.0 AND query_complexity_score <= 1.0)
        );"""

        sql_create_moe_expert_responses = """
        CREATE TABLE IF NOT EXISTS moe_expert_responses (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            response_id VARCHAR(100) NOT NULL UNIQUE,
            routing_id VARCHAR(100) NOT NULL,
            expert_id VARCHAR(50) NOT NULL,
            query_hash VARCHAR(64) NOT NULL,
            response_content JSONB NOT NULL,
            confidence_score DECIMAL(5,4) NOT NULL,
            processing_time_ms INTEGER NOT NULL,
            memory_usage_mb DECIMAL(8,2),
            computational_cost DECIMAL(8,4),
            success_status BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            quality_score DECIMAL(5,4),
            relevance_score DECIMAL(5,4),
            novelty_score DECIMAL(5,4),
            consistency_score DECIMAL(5,4),
            final_weight DECIMAL(5,4),
            included_in_ensemble BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            CONSTRAINT chk_final_weight CHECK (final_weight >= 0.0 AND final_weight <= 1.0),
            FOREIGN KEY (routing_id) REFERENCES moe_gating_network(routing_id),
            FOREIGN KEY (expert_id) REFERENCES moe_expert_registry(expert_id)
        );"""

        sql_create_flow_pattern_library = """
        CREATE TABLE IF NOT EXISTS flow_pattern_library (
            id SERIAL PRIMARY KEY,
            pattern_id VARCHAR(50) NOT NULL UNIQUE,
            pattern_name VARCHAR(100) NOT NULL,
            pattern_type VARCHAR(30) NOT NULL,
            market_regime VARCHAR(50),
            success_rate DECIMAL(5,4) NOT NULL,
            avg_duration_hours INTEGER,
            typical_magnitude DECIMAL(8,4),
            confidence_threshold DECIMAL(5,4) NOT NULL,
            pattern_characteristics JSONB DEFAULT '{}',
            historical_occurrences INTEGER DEFAULT 0,
            last_occurrence DATE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );"""

        sql_create_ai_performance_daily_summary = """
        CREATE TABLE IF NOT EXISTS ai_performance_daily_summary (
            id SERIAL PRIMARY KEY,
            summary_date DATE NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            accuracy_rate DECIMAL(5,4) DEFAULT 0.0,
            avg_confidence DECIMAL(5,4) DEFAULT 0.0,
            total_trades INTEGER DEFAULT 0,
            profitable_trades INTEGER DEFAULT 0,
            total_pnl DECIMAL(12,4) DEFAULT 0.0,
            sharpe_ratio DECIMAL(8,4),
            max_drawdown DECIMAL(8,4),
            model_versions JSONB DEFAULT '{}',
            performance_metrics JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(summary_date, symbol)
        );"""

        # HuiHui Usage Monitoring Tables
        sql_create_huihui_usage_records = """
        CREATE TABLE IF NOT EXISTS huihui_usage_records (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            expert_name VARCHAR(50) NOT NULL,
            request_type VARCHAR(50) NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            processing_time_seconds DECIMAL(8,3) NOT NULL,
            success BOOLEAN NOT NULL DEFAULT TRUE,
            error_type VARCHAR(100),
            retry_count INTEGER DEFAULT 0,
            timeout_occurred BOOLEAN DEFAULT FALSE,
            market_condition VARCHAR(20) NOT NULL DEFAULT 'normal',
            vix_level DECIMAL(6,2),
            symbol VARCHAR(20),
            api_token_hash VARCHAR(64),
            user_session_id VARCHAR(100),
            request_metadata JSONB DEFAULT '{}',
            response_metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_processing_time CHECK (processing_time_seconds >= 0),
            CONSTRAINT chk_tokens CHECK (input_tokens >= 0 AND output_tokens >= 0 AND total_tokens >= 0),
            CONSTRAINT chk_market_condition CHECK (market_condition IN ('normal', 'volatile', 'crisis'))
        );"""

        sql_create_huihui_usage_patterns = """
        CREATE TABLE IF NOT EXISTS huihui_usage_patterns (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            expert_name VARCHAR(50) NOT NULL,
            analysis_period VARCHAR(20) NOT NULL,
            total_requests INTEGER NOT NULL,
            avg_requests_per_minute DECIMAL(8,3) NOT NULL,
            peak_requests_per_minute DECIMAL(8,3) NOT NULL,
            avg_input_tokens DECIMAL(8,1) NOT NULL,
            avg_output_tokens DECIMAL(8,1) NOT NULL,
            avg_total_tokens DECIMAL(8,1) NOT NULL,
            max_input_tokens INTEGER NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            max_total_tokens INTEGER NOT NULL,
            avg_processing_time DECIMAL(8,3) NOT NULL,
            max_processing_time DECIMAL(8,3) NOT NULL,
            success_rate DECIMAL(5,4) NOT NULL,
            timeout_rate DECIMAL(5,4) NOT NULL,
            normal_condition_requests INTEGER DEFAULT 0,
            volatile_condition_requests INTEGER DEFAULT 0,
            crisis_condition_requests INTEGER DEFAULT 0,
            analysis_start_time TIMESTAMPTZ NOT NULL,
            analysis_end_time TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
            CONSTRAINT chk_timeout_rate CHECK (timeout_rate >= 0.0 AND timeout_rate <= 1.0)
        );"""

        sql_create_huihui_optimization_recommendations = """
        CREATE TABLE IF NOT EXISTS huihui_optimization_recommendations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            expert_name VARCHAR(50) NOT NULL,
            current_rate_limit INTEGER NOT NULL,
            current_token_limit INTEGER NOT NULL,
            current_timeout_seconds INTEGER NOT NULL,
            recommended_rate_limit INTEGER NOT NULL,
            recommended_token_limit INTEGER NOT NULL,
            recommended_timeout_seconds INTEGER NOT NULL,
            confidence_score DECIMAL(5,4) NOT NULL,
            urgency_level VARCHAR(20) NOT NULL,
            market_condition_factor VARCHAR(30) NOT NULL,
            based_on_requests INTEGER NOT NULL,
            analysis_period_hours INTEGER NOT NULL,
            peak_usage_factor DECIMAL(6,3) NOT NULL,
            reasoning TEXT NOT NULL,
            implementation_priority INTEGER DEFAULT 5,
            estimated_improvement_percent DECIMAL(5,2),
            status VARCHAR(20) DEFAULT 'pending',
            implemented_at TIMESTAMPTZ,
            implemented_by VARCHAR(100),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            CONSTRAINT chk_urgency_level CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
            CONSTRAINT chk_priority CHECK (implementation_priority >= 1 AND implementation_priority <= 10),
            CONSTRAINT chk_status CHECK (status IN ('pending', 'approved', 'implemented', 'rejected'))
        );"""

        sql_create_huihui_system_health = """
        CREATE TABLE IF NOT EXISTS huihui_system_health (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            cpu_usage_percent DECIMAL(5,2),
            memory_usage_percent DECIMAL(5,2),
            gpu_memory_used_percent DECIMAL(5,2),
            gpu_temperature DECIMAL(5,1),
            gpu_load_percent DECIMAL(5,2),
            ollama_healthy BOOLEAN NOT NULL,
            ollama_response_time_ms INTEGER,
            experts_available JSONB DEFAULT '{}',
            total_requests_last_hour INTEGER DEFAULT 0,
            avg_response_time_last_hour DECIMAL(8,3),
            error_rate_last_hour DECIMAL(5,4),
            current_market_condition VARCHAR(20),
            current_vix_level DECIMAL(6,2),
            recorded_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT chk_cpu_usage CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
            CONSTRAINT chk_memory_usage CHECK (memory_usage_percent >= 0 AND memory_usage_percent <= 100),
            CONSTRAINT chk_error_rate CHECK (error_rate_last_hour >= 0.0 AND error_rate_last_hour <= 1.0)
        );"""

        commands = [sql_create_daily_ohlcv, sql_create_daily_eots_metrics, sql_create_trade_outcomes,
                   sql_create_atif_recommendations, sql_create_atif_insights, sql_create_atif_performance,
                   sql_create_ai_learning_patterns, sql_create_ai_insights_history, sql_create_ai_predictions,
                   sql_create_ai_model_performance, sql_create_memory_entities, sql_create_memory_relations,
                   sql_create_memory_observations, sql_create_huihui_market_regime, sql_create_huihui_options_flow,
                   sql_create_moe_expert_registry, sql_create_moe_gating_network, sql_create_moe_expert_responses,
                   sql_create_daily_regime_snapshots, sql_create_key_level_performance, sql_create_flow_pattern_library,
                   sql_create_ai_performance_daily_summary, sql_create_huihui_usage_records, sql_create_huihui_usage_patterns,
                   sql_create_huihui_optimization_recommendations, sql_create_huihui_system_health]
        # Create indexes for memory tables
        memory_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_symbol ON memory_entities(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_type ON memory_entities(entity_type);",
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_active ON memory_entities(is_active);",
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_created ON memory_entities(created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_entity_id);",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_entity_id);",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_active ON memory_relations(is_active);",
            "CREATE INDEX IF NOT EXISTS idx_memory_observations_symbol ON memory_observations(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_memory_observations_type ON memory_observations(observation_type);"
        ]

        # Add missing columns to existing tables if needed
        alter_commands = [
            "ALTER TABLE memory_entities ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;",
            "ALTER TABLE memory_relations ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;",
            "ALTER TABLE ai_insights_history ADD COLUMN IF NOT EXISTS symbol VARCHAR(20);",
            "ALTER TABLE ai_insights_history ADD COLUMN IF NOT EXISTS market_regime TEXT;",
            "ALTER TABLE ai_learning_patterns ADD COLUMN IF NOT EXISTS pattern_type TEXT NOT NULL DEFAULT 'market_pattern';",
            "ALTER TABLE ai_predictions ADD COLUMN IF NOT EXISTS predicted_confidence NUMERIC(5, 4) DEFAULT 0.0000;"
        ]

        try:
            with self._conn.cursor() as cur:  # type: ignore
                # Enable UUID extension first
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')  # type: ignore
                # Create tables
                for cmd in commands:
                    cur.execute(cmd)  # type: ignore
                # Add missing columns to existing tables
                for alter_cmd in alter_commands:
                    try:
                        cur.execute(alter_cmd)  # type: ignore
                    except Exception as alter_e:
                        # Column might already exist or table might not exist yet
                        self.logger.debug(f"Alter command skipped: {alter_e}")
                # Create indexes
                for idx in memory_indexes:
                    try:
                        cur.execute(idx)  # type: ignore
                    except Exception as idx_e:
                        # Index might already exist
                        self.logger.debug(f"Index creation skipped: {idx_e}")
            self.logger.info("Database schema initialized with memory intelligence tables.")
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            raise

    def query_metric(self, table_name: str, metric_name: str, start_date: date, end_date: date) -> Optional[pd.Series]:
        try:
            # Check metrics schema first, then public
            schemas_to_check = ['metrics', 'public']

            for schema in schemas_to_check:
                try:
                    query = sql.SQL("SELECT * FROM {schema}.{table} WHERE date >= %s AND date <= %s AND {metric} IS NOT NULL ORDER BY date;").format(
                        schema=sql.Identifier(schema),
                        table=sql.Identifier(table_name),
                        metric=sql.Identifier(metric_name)
                    )
                    with self._conn.cursor() as cur:  # type: ignore
                        cur.execute(query, (start_date, end_date))
                        rows = cur.fetchall()
                        
                        if rows:
                            # Get column names from cursor description
                            columns = [desc[0] for desc in cur.description]
                            self.logger.debug(f"✅ Found metric data in {schema}.{table_name}")
                            df = pd.DataFrame(rows, columns=columns)
                            return df.set_index('date')[metric_name]

                except Exception as schema_error:
                    self.logger.debug(f"Schema {schema} check failed for metric query: {schema_error}")
                    continue

            # If no data found in any schema
            self.logger.warning(f"No metric data found for {table_name}.{metric_name} in any schema")
            return None

        except Exception as e:
            self.logger.error(f"Failed to query metric: {e}")
            return None

    def query_ohlcv(self, table_name: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Query OHLCV data from Supabase PostgreSQL database."""
        try:
            # SUPABASE-ONLY: PostgreSQL syntax - check metrics schema first, then public
            schemas_to_check = ['metrics', 'public']

            for schema in schemas_to_check:
                try:
                    query = sql.SQL("SELECT * FROM {schema}.{table} WHERE date >= %s AND date <= %s ORDER BY date;").format(
                        schema=sql.Identifier(schema),
                        table=sql.Identifier(table_name)
                    )
                    with self._conn.cursor() as cur:  # type: ignore
                        cur.execute(query, (start_date, end_date))
                        rows = cur.fetchall()
                        
                        if rows:
                            # Get column names from cursor description
                            columns = [desc[0] for desc in cur.description]
                            self.logger.info(f"✅ Found OHLCV data in {schema}.{table_name}")
                            df = pd.DataFrame(rows, columns=columns)
                            return df
                        else:
                            self.logger.debug(f"No data found in {schema}.{table_name}")

                except Exception as schema_error:
                    self.logger.debug(f"Schema {schema} check failed: {schema_error}")
                    continue

            # If no data found in any schema
            self.logger.warning(f"No OHLCV data found for {table_name} in any schema")
            return None

        except Exception as e:
            self.logger.error(f"Failed to query OHLCV: {e}")
            return None

    def insert_record(self, table_name: str, data: Dict[str, Any], schema_name: Optional[str] = None) -> None:
        """Insert record into Supabase PostgreSQL database."""
        try:
            columns = list(data.keys())
            values = list(data.values())

            # Determine schema - try metrics first for EOTS tables, then public
            if not schema_name:
                if table_name in ['daily_ohlcv', 'daily_eots_metrics']:
                    schema_name = 'metrics'
                else:
                    schema_name = 'public'

            # SUPABASE-ONLY: PostgreSQL syntax with schema
            query = sql.SQL("INSERT INTO {schema}.{table} ({fields}) VALUES ({placeholders}) ON CONFLICT DO NOTHING;").format(
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name),
                fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
                placeholders=sql.SQL(', ').join(sql.Placeholder() * len(columns))
            )
            with self._conn.cursor() as cur:  # type: ignore
                cur.execute(query, values)
            self.logger.debug(f"✅ Inserted record into {schema_name}.{table_name} (Supabase).")
        except Exception as e:
            self.logger.error(f"Failed to insert record into {schema_name}.{table_name}: {e}")
            raise

    # ATIF-specific database methods
    def store_atif_recommendation(self, recommendation_data: Dict[str, Any]) -> None:
        """Store ATIF recommendation to database."""
        try:
            self.insert_record("atif_recommendations", recommendation_data)
            self.logger.info(f"Stored ATIF recommendation for {recommendation_data.get('symbol')}")
        except Exception as e:
            self.logger.error(f"Failed to store ATIF recommendation: {e}")
            raise

    def store_atif_insight(self, insight_data: Dict[str, Any]) -> None:
        """Store ATIF insight to database."""
        try:
            self.insert_record("atif_insights", insight_data)
            self.logger.info(f"Stored ATIF insight: {insight_data.get('insight_title')}")
        except Exception as e:
            self.logger.error(f"Failed to store ATIF insight: {e}")
            raise

    def store_atif_performance(self, performance_data: Dict[str, Any]) -> None:
        """Store ATIF performance data to database."""
        try:
            self.insert_record("atif_performance", performance_data)
            self.logger.info(f"Stored ATIF performance for {performance_data.get('symbol')}")
        except Exception as e:
            self.logger.error(f"Failed to store ATIF performance: {e}")
            raise

    def get_recent_atif_recommendations(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ATIF recommendations for a symbol."""
        try:
            query = sql.SQL("""
                SELECT * FROM atif_recommendations
                WHERE symbol = %s AND status = 'ACTIVE'
                ORDER BY timestamp DESC
                LIMIT %s
            """)
            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(query, (symbol, limit))
                rows = cur.fetchall()
            return rows
        except Exception as e:
            self.logger.error(f"Failed to get ATIF recommendations: {e}")
            return []

    def get_recent_atif_insights(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent ATIF insights for a symbol."""
        try:
            query = sql.SQL("""
                SELECT * FROM atif_insights
                WHERE symbol = %s AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY timestamp DESC
                LIMIT %s
            """)
            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(query, (symbol, limit))
                rows = cur.fetchall()
            return rows
        except Exception as e:
            self.logger.error(f"Failed to get ATIF insights: {e}")
            return []

    def get_atif_performance_stats(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get ATIF performance statistics for a symbol."""
        try:
            query = sql.SQL("""
                SELECT
                    COUNT(*) as total_trades,
                    AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(pnl_percentage) as avg_pnl,
                    AVG(conviction_score) as avg_conviction,
                    AVG(hold_duration_hours) as avg_hold_hours
                FROM atif_performance
                WHERE symbol = %s AND entry_timestamp >= NOW() - INTERVAL '%s days'
            """)
            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(query, (symbol, days))
                result = cur.fetchone()
            return result or {}
        except Exception as e:
            self.logger.error(f"Failed to get ATIF performance stats: {e}")
            return {}

    def insert_batch_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        if not data:
            return
        try:
            columns = list(data[0].keys())
            values = [list(d.values()) for d in data]
            query = sql.SQL("INSERT INTO {table} ({fields}) VALUES ({placeholders}) ON CONFLICT DO NOTHING;").format(
                table=sql.Identifier(table_name),
                fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
                placeholders=sql.SQL(', ').join(sql.Placeholder() * len(columns))
            )
            with self._conn.cursor() as cur:  # type: ignore
                cur.executemany(query, values)
            self.logger.info(f"Inserted {len(data)} records into {table_name}.")
        except Exception as e:
            self.logger.error(f"Failed to insert batch data: {e}")
            raise

    # === MEMORY INTELLIGENCE ENGINE DATABASE METHODS ===

    def store_memory_entity(self, entity_data: Dict[str, Any]) -> None:
        """Store memory entity (pattern, outcome, insight) to database."""
        try:
            # Ensure required fields
            entity_data.setdefault('created_at', datetime.now().isoformat())
            entity_data.setdefault('entity_type', 'pattern')
            entity_data.setdefault('confidence_score', 0.0)

            self.insert_record("memory_entities", entity_data)
            self.logger.info(f"Memory entity stored: {entity_data.get('entity_name', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Error storing memory entity: {str(e)}")
            raise

    def store_memory_relation(self, relation_data: Dict[str, Any]) -> None:
        """Store memory relation (connections between entities) to database."""
        try:
            # Ensure required fields
            relation_data.setdefault('created_at', datetime.now().isoformat())
            relation_data.setdefault('relation_strength', 0.5)

            self.insert_record("memory_relations", relation_data)
            self.logger.info(f"Memory relation stored: {relation_data.get('relation_type', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Error storing memory relation: {str(e)}")
            raise

    def store_memory_observation(self, observation_data: Dict[str, Any]) -> None:
        """Store memory observation (market event, outcome) to database."""
        try:
            # Ensure required fields
            observation_data.setdefault('timestamp', datetime.now().isoformat())
            observation_data.setdefault('observation_type', 'market_event')
            observation_data.setdefault('success_rate', 0.0)

            self.insert_record("memory_observations", observation_data)
            self.logger.info(f"Memory observation stored: {observation_data.get('symbol', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Error storing memory observation: {str(e)}")
            raise

    def get_memory_patterns(self, symbol: Optional[str] = None, pattern_type: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve memory patterns from Supabase database."""
        try:
            base_query = "SELECT * FROM memory_entities WHERE 1=1"
            conditions = []
            params: List[Any] = []

            if symbol:
                conditions.append("AND symbol = %s")
                params.append(symbol)

            if pattern_type:
                conditions.append("AND entity_type = %s")
                params.append(pattern_type)

            conditions.append("ORDER BY confidence_score DESC, created_at DESC LIMIT %s")
            params.append(limit)

            final_query = base_query + " " + " ".join(conditions)

            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(final_query, params)
                return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error retrieving memory patterns: {str(e)}")
            return []

    def search_memory_intelligence(self, search_query: str, symbol: Optional[str] = None,
                                 limit: int = 20) -> Dict[str, Any]:
        """Search memory intelligence using text search in Supabase."""
        try:
            # Search entities
            entity_base = "SELECT * FROM memory_entities WHERE (entity_name ILIKE %s OR description ILIKE %s OR metadata::text ILIKE %s)"
            entity_params: List[Any] = [f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"]

            if symbol:
                entity_base += " AND symbol = %s"
                entity_params.append(symbol)

            entity_base += " ORDER BY confidence_score DESC LIMIT %s"
            entity_params.append(limit)

            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(entity_base, entity_params)
                entities = cur.fetchall()

            # Search observations
            obs_base = "SELECT * FROM memory_observations WHERE (description ILIKE %s OR metadata::text ILIKE %s)"
            obs_params: List[Any] = [f"%{search_query}%", f"%{search_query}%"]

            if symbol:
                obs_base += " AND symbol = %s"
                obs_params.append(symbol)

            obs_base += " ORDER BY success_rate DESC LIMIT %s"
            obs_params.append(limit)

            with self._conn.cursor(row_factory=dict_row) as cur:  # type: ignore
                cur.execute(obs_base, obs_params)
                observations = cur.fetchall()

            return {
                "entities": entities,
                "observations": observations,
                "search_query": search_query,
                "total_results": len(entities) + len(observations)
            }
        except Exception as e:
            self.logger.error(f"Error searching memory intelligence: {str(e)}")
            return {"entities": [], "observations": [], "search_query": search_query, "total_results": 0}

    def _execute_transaction(self, queries: List[str], params_list: Optional[List[Tuple]] = None) -> bool:
        """Execute multiple queries in a transaction."""
        if not self._conn:
            self.logger.error("No database connection")
            return False
        
        try:
            with self._conn.cursor() as cur:
                for i, query in enumerate(queries):
                    params = params_list[i] if params_list and i < len(params_list) else None
                    if params:
                        cur.execute(query, params)  # Psycopg handles parameter binding safely
                    else:
                        cur.execute(query)  # Direct execution for non-parameterized queries
            return True
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            return False

    def _get_table_names(self) -> List[str]:
        """Get list of table names in database."""
        if not self._conn:
            return []
        
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
                results = cur.fetchall()
                return [row['table_name'] for row in results] if results else []
        except Exception as e:
            self.logger.error(f"Failed to get table names: {e}")
            return []
