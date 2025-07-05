import time
from datetime import datetime, time as dtime
import logging
import shutil
from pathlib import Path
import json
import os
from typing import Optional, List, Dict, Any, Union
import stat
import errno

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5
from data_management.initial_processor_v2_5 import InitialDataProcessorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from core_analytics_engine.atif_engine_v2_5 import ATIFEngineV2_5
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5
from data_models import EOTSConfigV2_5
from data_models.deprecated_files.core_system_config import IntradayCollectorSettings

from data_models import ProcessedDataBundleV2_5
from data_models.elite_intelligence import EliteConfig
from data_models.elite_config_models import EliteConfig as EliteConfigModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IntradayCollector")

def is_market_open():
    now = datetime.now().time()
    return dtime(9, 30) <= now <= dtime(16, 0)

def ensure_cache_directory(path: str):
    """Ensure the cache directory exists with full write permissions and log any issues."""
    try:
        cache_path = Path(path)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {path}")
        # Set full permissions (rwxrwxrwx)
        cache_path.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        logger.info(f"Set full permissions for cache directory at {path}")
    except PermissionError as e:
        logger.error(f"Permission denied when accessing cache directory {path}: {e}")
        raise
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.error(f"OS error when accessing cache directory {path}: {e}")
            raise

def build_orchestrator(enhanced_cache):
    config_manager = ConfigManagerV2_5()
    # Extract intraday collector settings (type-agnostic)
    intraday_settings = config_manager.config.intraday_collector_settings or None

    # Properly instantiate EliteConfig from config_manager or defaults
    elite_config = EliteConfigModel()

    # Initialize DatabaseManagerV2_5 and pass it to the orchestrator
    db_manager = DatabaseManagerV2_5(config_manager=config_manager)

    orchestrator = ITSOrchestratorV2_5(
        config_manager=config_manager,
        db_manager=db_manager,  # Pass the initialized db_manager
        enhanced_cache=enhanced_cache,  # Pass the enhanced cache manager
        elite_config=elite_config  # Pass the validated Pydantic model instance
    )

    logger.info("✅ Initialized ITSOrchestratorV2_5 with database manager and elite config")
    return orchestrator

def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize a ticker symbol for safe use in file paths and cache keys.
    Replaces '/' and ':' with '_'.
    """
    return symbol.replace('/', '_').replace(':', '_')

def main():
    ensure_cache_directory('cache/intraday_metrics')

    config_manager = ConfigManagerV2_5()
    elite_config = config_manager.config.elite_config or EliteConfig()
    # Extract intraday collector settings with defaults (type-agnostic)
    intraday_settings = config_manager.config.intraday_collector_settings or None

    # Provide fallback defaults for all settings
    watched_tickers = getattr(intraday_settings, 'watched_tickers', ['SPY', 'QQQ'])
    metrics = getattr(intraday_settings, 'metrics', ['vapi_fa', 'dwfd', 'tw_laf'])
    cache_dir = Path(getattr(intraday_settings, 'cache_dir', 'cache/intraday_metrics'))
    collection_interval = getattr(intraday_settings, 'collection_interval_seconds', 5)
    market_open_time = getattr(intraday_settings, 'market_open_time', '09:30:00')
    market_close_time = getattr(intraday_settings, 'market_close_time', '16:00:00')
    reset_at_eod = getattr(intraday_settings, 'reset_at_eod', True)
    calibration_threshold = getattr(intraday_settings, 'calibration_threshold', 25)
    calibrated_pairs = set()

    # Initialize enhanced cache manager
    from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5, CacheLevel
    enhanced_cache = EnhancedCacheManagerV2_5(
        cache_root="cache/enhanced_v2_5",
        memory_limit_mb=50,
        disk_limit_mb=500,
        default_ttl_seconds=86400  # 24 hours for intraday data
    )
    logger.info("✨ Enhanced Cache Manager initialized for intraday collection")

    # Enhanced Cache is already initialized above - no need for Redis
    # This eliminates Redis limits and network overhead for better performance
    logger.info("✅ Enhanced Cache enabled for intraday collector - no Redis limits!")

    logger.info(f"Loaded intraday collector config: {intraday_settings}")

    def clear_intraday_cache():
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Intraday cache cleared for new trading day.")
        calibrated_pairs.clear()

    # --- LOCAL CACHE ONLY MODE: DatabaseManagerV2_5 REMOVED ---
    # from data_management.database_manager_v2_5 import DatabaseManagerV2_5  # (REMOVED)
    orchestrator = build_orchestrator(enhanced_cache)
    last_reset_date = None

    try:
        while True:
            today = datetime.now().date()
            if last_reset_date != today:
                clear_intraday_cache()
                last_reset_date = today
            if is_market_open():
                for symbol in watched_tickers:
                    try:
                        logger.info(f"Processing {symbol}...")
                        # Handle async orchestrator call properly
                        import asyncio
                        bundle = asyncio.run(orchestrator.run_full_analysis_cycle(
                            ticker=symbol,
                            dte_min=0,
                            dte_max=5,
                            price_range_percent=5
                        ))
                        logger.info(f"Completed {symbol}.")
                        # 🚀 PYDANTIC-FIRST: Extract enriched underlying data as Pydantic model
                        und_data = None
                        if bundle and hasattr(bundle, 'processed_data_bundle') and bundle.processed_data_bundle is not None:
                            und_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
                        if not und_data or not hasattr(und_data, 'model_dump'):
                            logger.warning(f"No valid Pydantic underlying data for {symbol}, skipping metric collection.")
                            continue
                        today = datetime.now().strftime('%Y-%m-%d')
                        for metric in metrics:
                            # 🚀 PYDANTIC-FIRST: Extract value directly from Pydantic model
                            value = getattr(und_data, metric, None)
                            if value is None:
                                logger.warning(f"Metric {metric} missing for {symbol}.")
                                continue
                            # Always store as a list for histories/arrays, wrap scalars
                            if isinstance(value, (list, tuple)):
                                values = list(value)
                            elif isinstance(value, dict):
                                # For dicts (e.g., rolling_flows, greek_flows, flow_ratios), store as-is
                                values = value
                            else:
                                values = [value]
                            safe_symbol = sanitize_symbol(symbol)

                            # Use enhanced cache as primary storage
                            try:
                                # Determine cache level based on data size
                                data_size_mb = len(str(values)) / (1024 * 1024)
                                cache_level = CacheLevel.COMPRESSED if data_size_mb > 0.5 else CacheLevel.MEMORY

                                success = enhanced_cache.put(
                                    symbol=symbol,
                                    metric_name=metric,
                                    data=values,
                                    cache_level=cache_level,
                                    tags=[f"intraday_{today}", "collector", safe_symbol]
                                )

                                # Enhanced Cache already stores the data above - no need for Redis
                                # This eliminates Redis limits and provides better performance
                                logger.debug(f"✅ PYDANTIC-FIRST: Stored {symbol} {metric} in Enhanced Cache for real-time access")

                                if success:
                                    sample_count = len(values) if isinstance(values, list) else (len(values) if hasattr(values, '__len__') else 1)
                                    pair_key = (safe_symbol, metric)
                                    if sample_count >= calibration_threshold and pair_key not in calibrated_pairs:
                                        logger.info(f"Metric for {symbol} ({metric}) is now fully calibrated with {sample_count} samples.")
                                        calibrated_pairs.add(pair_key)
                                else:
                                    logger.warning(f"Enhanced cache storage failed for {symbol} {metric}")

                            except Exception as e:
                                logger.warning(f"Enhanced cache error for {symbol} {metric}: {e}")

                                # 🚀 PYDANTIC-FIRST: Fallback to legacy cache with Pydantic model
                                cache_file = cache_dir / f"{safe_symbol}_{metric}_{today}.json"
                                try:
                                    # IntradayMetricDataV2_5 is deprecated, using a placeholder for now
                                    # Further refactoring needed to align with ProcessedDataBundleV2_5 structure
                                    cache_data = {
                                        "values": values,
                                        "last_updated": datetime.now().isoformat(),
                                        "sample_count": len(values) if isinstance(values, list) else 1
                                    }
                                    with open(cache_file, 'w') as f:
                                        json.dump(cache_data, f)  # Using dict directly as IntradayMetricDataV2_5 is deprecated
                                    logger.debug(f"Stored {symbol} {metric} in legacy cache as fallback")
                                except Exception as fallback_e:
                                    logger.error(f"Both enhanced and legacy cache failed for {symbol} {metric}: {fallback_e}")
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                try:
                    time.sleep(collection_interval)
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal during market hours sleep. Shutting down gracefully...")
                    break
            else:
                logger.info("Market closed. Sleeping until next check.")
                try:
                    time.sleep(60 * 10)  # Sleep 10 minutes when market is closed
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal during market closed sleep. Shutting down gracefully...")
                    break
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"💥 Unexpected error in main loop: {e}")
    finally:
        logger.info("🏁 Intraday collector shutdown complete.")

if __name__ == "__main__":
    main()