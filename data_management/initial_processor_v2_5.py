# data_management/initial_processor_v2_5.py
# EOTS v2.5 - SENTRY-REFACRORED CANONICAL SCRIPT
#
# This module is responsible for the initial processing and validation of raw data
# and for orchestrating the main metric calculation process. It adheres to the
# Single Responsibility Principle and uses Pydantic models for data contracts.

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

try:
    import pandas as pd
    import numpy as np
    from pydantic import ValidationError
except ImportError as e_dep:
    print(f"CRITICAL ERROR: initial_processor.py: Essential libraries not found: {e_dep}.", file=sys.stderr)
    sys.exit(1)

# --- EOTS V2.5 Component and Schema Imports ---
# Assumes a project structure where these are importable
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models import ( # Updated import
    UnprocessedDataBundleV2_5,
    ProcessedDataBundleV2_5,
    RawOptionsContractV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    RawUnderlyingDataCombinedV2_5  # STRICT PYDANTIC V2-ONLY
)
if TYPE_CHECKING:
    from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5

# Module-Specific Logger
logger = logging.getLogger(__name__)

class InitialDataProcessorV2_5:
    """
    Processes raw data bundles from data fetchers.
    1. Converts Pydantic models to a DataFrame for processing.
    2. Validates and prepares inputs.
    3. Invokes the MetricsCalculator to compute all system metrics.
    4. Returns a comprehensive, processed data bundle as a Pydantic model.
    """

    def __init__(self, config_manager: ConfigManagerV2_5, metrics_calculator: 'MetricsCalculatorV2_5'):
        """
        Initializes the InitialDataProcessor.

        Args:
            config_manager: The system's configuration manager instance.
            metrics_calculator: An initialized instance of MetricsCalculatorV2_5.
        """
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initializing InitialDataProcessorV2_5...")

        if not isinstance(config_manager, ConfigManagerV2_5):
            raise TypeError("A valid ConfigManagerV2_5 instance is required.")
        # Note: Type checking for metrics_calculator moved to runtime validation
        # to avoid circular import issues

        self.config_manager = config_manager
        self.metrics_calculator = metrics_calculator

        self.logger.info("InitialDataProcessorV2_5 Initialized successfully.")

    def _prepare_dataframe_from_models(
        self,
        options_contracts: List[RawOptionsContractV2_5],
        underlying_data: RawUnderlyingDataCombinedV2_5,  # STRICT PYDANTIC V2-ONLY
        symbol: str,
        current_time: datetime
    ) -> Tuple[pd.DataFrame, RawUnderlyingDataCombinedV2_5]:  # STRICT PYDANTIC V2-ONLY
        """
        STRICT PYDANTIC V2-ONLY: Converts Pydantic models to DataFrame while preserving Pydantic model for underlying data.
        NO DICTIONARY CONVERSIONS - maintains type safety end-to-end.
        """
        prep_logger = self.logger.getChild("PrepareDataFrame")

        if not options_contracts:
            prep_logger.warning(f"Input options model list for {symbol} is empty.")
            return pd.DataFrame(), underlying_data

        # STRICT PYDANTIC V2 VALIDATION
        if not isinstance(underlying_data, RawUnderlyingDataCombinedV2_5):
            raise TypeError(f"CRITICAL: underlying_data must be RawUnderlyingDataCombinedV2_5, got {type(underlying_data)}")

        try:
            # Convert options contracts to DataFrame (necessary for pandas operations)
            list_of_dicts = [model.model_dump() for model in options_contracts]
            df = pd.DataFrame(list_of_dicts)
            if prep_logger.isEnabledFor(logging.DEBUG):
                prep_logger.debug(f"Converted {len(options_contracts)} Pydantic models to DataFrame for {symbol}. Shape: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to convert Pydantic models to DataFrame for {symbol}: {e}")

        # Add essential context columns using STRICT PYDANTIC V2 access
        underlying_price = float(underlying_data.price) if underlying_data.price is not None else np.nan
        df["underlying_price_at_fetch"] = underlying_price
        df["processing_time_dt_obj"] = current_time
        df["symbol"] = symbol.upper()

        prep_logger.info(f"✅ PYDANTIC V2-ONLY: Underlying price extracted: {underlying_price} from {underlying_data.symbol}")

        # Add missing fields required by metrics calculations
        # CRITICAL FIX: Map ConvexValue field names to expected field names

        # 1. Map dte_calc to dte for compatibility with adaptive calculator
        if 'dte_calc' in df.columns:
            df['dte'] = df['dte_calc']
            prep_logger.info(f"✅ Mapped dte_calc to dte field")
        elif 'dte' not in df.columns:
            prep_logger.warning("⚠️ DTE field missing - adding default values")
            df['dte'] = 30  # Default DTE for weekend/off-hours

        # 2. Map iv to implied_volatility for compatibility with visualization metrics
        if 'iv' in df.columns:
            df['implied_volatility'] = df['iv']
            prep_logger.info(f"✅ Mapped iv to implied_volatility field")
        elif 'volatility' in df.columns:
            df['implied_volatility'] = df['volatility']
            prep_logger.info(f"✅ Mapped volatility to implied_volatility field")
        elif 'implied_volatility' not in df.columns:
            prep_logger.warning("⚠️ Implied volatility field missing - adding default values")
            df['implied_volatility'] = 0.20  # Default IV for weekend/off-hours

        # 3. CRITICAL FIX: Map volm to volume for Greek flow fallback calculations
        if 'volm' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['volm']
            prep_logger.info(f"✅ Mapped volm to volume field for Greek flow calculations")

        # Ensure critical numeric types
        for col in ["strike", "dte", "implied_volatility"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # DEBUG: Log DataFrame columns to verify field mapping
        prep_logger.info(f"🔍 DataFrame columns after field mapping: {list(df.columns)}")
        prep_logger.info(f"🔍 DTE field present: {'dte' in df.columns}")
        prep_logger.info(f"🔍 Implied volatility field present: {'implied_volatility' in df.columns}")

        # CRITICAL: Return Pydantic model directly - NO DICTIONARY CONVERSION
        return df, underlying_data

    def process_data_and_calculate_metrics(self, raw_data_bundle: UnprocessedDataBundleV2_5, dte_max: int = 45) -> ProcessedDataBundleV2_5:
        """
        Main processing method. Validates raw data, prepares it, and orchestrates metric calculation.

        Args:
            raw_data_bundle: An UnprocessedDataBundleV2_5 object from the data fetching layer.
            dte_max: Maximum DTE from control panel settings for DTE-aware calculations.

        Returns:
            A ProcessedDataBundleV2_5 object containing all calculated metrics.
        """
        proc_logger = self.logger.getChild("ProcessAndCalculate")
        
        if not isinstance(raw_data_bundle, UnprocessedDataBundleV2_5):
             raise TypeError("Input must be a valid UnprocessedDataBundleV2_5 object.")

        symbol = raw_data_bundle.underlying_data.symbol.upper()
        proc_logger.info(f"--- InitialProcessor: START for '{symbol}' at {raw_data_bundle.fetch_timestamp.isoformat()} ---")

        try:
            # 1. Prepare DataFrame from Pydantic models (PYDANTIC V2-FIRST)
            df_prepared, und_data_prepared = self._prepare_dataframe_from_models(
                options_contracts=raw_data_bundle.options_contracts,
                underlying_data=raw_data_bundle.underlying_data,  # FIXED: Keep as Pydantic model
                symbol=symbol,
                current_time=raw_data_bundle.fetch_timestamp
            )

            # 2. Invoke the Metrics Calculator with the correct method and arguments
            proc_logger.info(f"Invoking MetricsCalculator for {symbol}...")
            (
                df_strike_all_metrics,     # Strike-level data (first return value)
                df_chain_all_metrics,      # Contract-level data (second return value)
                und_data_enriched
            ) = self.metrics_calculator.calculate_all_metrics(
                options_df_raw=df_prepared,
                und_data_api_raw=und_data_prepared,
                dte_max=dte_max
            )
            proc_logger.info(f"MetricsCalculator finished for {symbol}.")

            # 3. Assemble the final ProcessedDataBundle Pydantic model
            # This step inherently validates the output of the metrics calculator.
            
            # Handle case where strike data might be None
            if df_strike_all_metrics is None:
                proc_logger.warning("Strike-level data is None from metrics calculator")
                df_strike_all_metrics = pd.DataFrame()
            
            # LIVE MODE VALIDATION: Debug data structures
            if proc_logger.isEnabledFor(logging.DEBUG):
                proc_logger.debug(f"Strike-level DataFrame shape: {df_strike_all_metrics.shape}")
                proc_logger.debug(f"Contract-level DataFrame shape: {df_chain_all_metrics.shape}")
            
            # STRICT PYDANTIC V2-ONLY: Handle enriched underlying data
            if isinstance(und_data_enriched, ProcessedUnderlyingAggregatesV2_5):
                # Already a Pydantic model - use directly
                underlying_data_enriched = und_data_enriched
            elif hasattr(und_data_enriched, 'model_dump'):
                # Convert from another Pydantic model
                underlying_data_enriched = ProcessedUnderlyingAggregatesV2_5(**und_data_enriched.model_dump())
            elif isinstance(und_data_enriched, dict):
                # Convert from dictionary (legacy path)
                underlying_data_enriched = ProcessedUnderlyingAggregatesV2_5(**und_data_enriched)
            else:
                raise TypeError(f"CRITICAL: Invalid und_data_enriched type: {type(und_data_enriched)}")

            processed_bundle = ProcessedDataBundleV2_5(
                options_data_with_metrics=[ProcessedContractMetricsV2_5(**{str(k): v for k, v in row.items()}) for row in df_chain_all_metrics.to_dict('records')],
                strike_level_data_with_metrics=[ProcessedStrikeLevelMetricsV2_5(**{str(k): v for k, v in row.items()}) for row in df_strike_all_metrics.to_dict('records')],
                underlying_data_enriched=underlying_data_enriched,
                processing_timestamp=datetime.now(),
                errors=raw_data_bundle.errors
            )

            proc_logger.info(f"--- InitialProcessor: END for '{symbol}'. Successfully created ProcessedDataBundle. ---")
            return processed_bundle

        except (ValidationError, ValueError, TypeError, KeyError) as e:
            err_msg = f"Critical data processing or validation error for '{symbol}': {e}"
            proc_logger.fatal(err_msg, exc_info=True)
            # In a real system, you might return a ProcessedDataBundle with an error state
            # For now, we re-raise to halt the cycle, as this indicates a severe issue.
            # If you ever add a fallback bundle here, set confidence=0.0, transition_risk=1.0
            raise RuntimeError(err_msg) from e