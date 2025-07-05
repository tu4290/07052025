# dashboard_application/callback_manager_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE CALLBACK MANAGER

import logging
import importlib
from typing import Any, Optional
import datetime
import asyncio
import time
import json

import dash
from dash import Input, Output, State, ctx, no_update, html
import dash_bootstrap_components as dbc

# EOTS V2.5 Imports
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models import FinalAnalysisBundleV2_5  # Fixed import
from data_models import ProcessedUnderlyingAggregatesV2_5  # Updated imports for Pydantic v2 compliance
from . import ids
from dashboard_application.layout_manager_v2_5 import _create_regime_display
from data_models.elite_intelligence import MarketRegime, FlowType # Import Elite Enums
from data_models import DashboardServerConfig
# DynamicThresholds is now DynamicThresholdsV2_5
from data_models.trading_market_models import DynamicThresholdsV2_5
from data_models import ProcessedDataBundleV2_5
from data_models.core_models import KeyLevelsDataV2_5, KeyLevelV2_5
from data_models.deprecated_files.trading_market_models import SignalPayloadV2_5
from data_models.trading_market_models import ATIFStrategyDirectivePayloadV2_5, ATIFSituationalAssessmentProfileV2_5
from data_models import ActiveRecommendationPayloadV2_5, TradeParametersV2_5
from data_models import ProcessedContractMetricsV2_5, ProcessedStrikeLevelMetricsV2_5

# --- Module-Specific Logger & Global References ---
callback_logger = logging.getLogger(__name__)
ORCHESTRATOR_REF: Optional[ITSOrchestratorV2_5] = None
CONFIG_REF: Optional[ConfigManagerV2_5] = None

# Simple cache to prevent duplicate processing
_last_request_cache = {}
_cache_timeout = 2  # seconds

def register_v2_5_callbacks(app: dash.Dash, orchestrator: ITSOrchestratorV2_5, config: ConfigManagerV2_5):
    """Registers all v2.5 callbacks with the Dash app instance."""
    global ORCHESTRATOR_REF, CONFIG_REF
    ORCHESTRATOR_REF = orchestrator
    CONFIG_REF = config
    callback_logger.info("Registering EOTS v2.5 authoritative callbacks...")

    # --- Primary Data Fetching and Storage Callback ---
    @app.callback(
        Output(ids.ID_MAIN_DATA_STORE, 'data'),
        Output(ids.ID_STATUS_ALERT_CONTAINER, 'children'),
        Input(ids.ID_MANUAL_REFRESH_BUTTON, 'n_clicks'),
        Input(ids.ID_INTERVAL_LIVE_UPDATE, 'n_intervals'),
        State(ids.ID_SYMBOL_INPUT, 'value'),
        State('dte-min-input', 'value'),
        State('dte-max-input', 'value'),
        State('price-range-input', 'value'),
        prevent_initial_call=False
    )
    def update_analysis_bundle_store(n_clicks: int, n_intervals: int, symbol: str, dte_min: int, dte_max: int, price_range_percent: int) -> tuple:
        """
        Fetches live analysis data from the orchestrator and returns it to the dashboard.
        """
        try:
            # Create cache key to prevent duplicate processing
            cache_key = f"{symbol or 'SPY'}_{dte_min or 0}_{dte_max or 5}_{price_range_percent or 5}"
            current_time = time.time()

            # Check if we have a recent request for the same parameters
            if cache_key in _last_request_cache:
                last_time = _last_request_cache[cache_key]
                if current_time - last_time < _cache_timeout:
                    callback_logger.debug(f"Skipping duplicate request for {symbol or 'SPY'} (too recent)")
                    return no_update, no_update

            # Update cache with current request time
            _last_request_cache[cache_key] = current_time

            # Call the orchestrator's async method using asyncio.run with strict Pydantic v2 validation
            callback_logger.debug(f"🔄 Fetching live data for {symbol or 'SPY'}...")
            bundle = asyncio.run(ORCHESTRATOR_REF.run_full_analysis_cycle(
                ticker=symbol or "SPY",
                dte_min=dte_min or 0,
                dte_max=dte_max or 5,
                price_range_percent=price_range_percent or 5
            ))

            # Ensure bundle is a proper Pydantic v2 model
            if not hasattr(bundle, 'model_dump_json'):
                callback_logger.error("❌ Bundle is not a Pydantic v2 model")
                raise ValueError("Analysis bundle is not properly structured as Pydantic v2 model")

            # Validate bundle structure
            if not hasattr(bundle, 'processed_data_bundle') or not bundle.processed_data_bundle:
                callback_logger.error("❌ Bundle missing processed_data_bundle")
                raise ValueError("Analysis bundle missing required processed data")

            # Serialize using Pydantic v2 model_dump_json
            bundle_dict = json.loads(bundle.model_dump_json())

            bundle_json = json.dumps(bundle_dict)
            callback_logger.info(f"✅ Successfully fetched data for {symbol or 'SPY'}")

            # Create simple status message
            status_message = f"🟢 LIVE - {symbol or 'SPY'} at {datetime.datetime.now().strftime('%H:%M:%S EST')}"

            # Set alert color for live data
            alert_color = "success"
            alert = dbc.Alert(status_message, color=alert_color, duration=6000)
            return bundle_json, alert

        except Exception as e:
            error_msg = f"Error fetching live data: {e}"
            callback_logger.error(error_msg, exc_info=True)
            alert = dbc.Alert(error_msg, color="danger", duration=8000)
            return None, alert

    # --- Dynamic Mode and Chart Rendering Callback ---
    @app.callback(
        Output(ids.ID_PAGE_CONTENT, 'children'),
        Input(ids.ID_MAIN_DATA_STORE, 'data'),
        State(ids.ID_URL_LOCATION, 'pathname') # Use URL to determine the mode
    )
    def render_mode_content(bundle_json: Optional[str], pathname: str) -> Any:
        """
        Renders the entire layout for the currently selected mode.
        This is the central UI update callback.
        """
        callback_logger.info(f"RENDER CALLBACK: bundle_json={'has data' if bundle_json else 'None'}, pathname='{pathname}'")
        
        if not bundle_json:
            callback_logger.warning("❌ No bundle data available - showing wait message")
            return dbc.Alert("Waiting for initial data fetch...", color="info")

        # Determine mode from URL path, default to main
        if not pathname or pathname == '/':
            mode_key = 'main'
        else:
            mode_key = pathname.strip('/').split('/')[0]
            
        callback_logger.info(f"Determined mode_key: '{mode_key}' from pathname: '{pathname}'")
        
        # Get the full dashboard config using strict Pydantic v2 model access
        try:
            dashboard_config = CONFIG_REF.config.visualization_settings.dashboard if CONFIG_REF else None
            if dashboard_config is None:
                callback_logger.error("❌ Dashboard config is None")
                return dbc.Alert("Dashboard configuration not found.", color="danger")

            # Ensure dashboard_config is a proper Pydantic model
            if not hasattr(dashboard_config, 'model_dump'):
                callback_logger.error("❌ Dashboard config is not a Pydantic model")
                return dbc.Alert("Dashboard configuration is not properly parsed as Pydantic model.", color="danger")

            modes_config = dashboard_config.modes_detail_config
            if not hasattr(modes_config, 'model_dump'):
                callback_logger.error("❌ Modes config is not a Pydantic model")
                return dbc.Alert("Dashboard modes configuration is not properly parsed as Pydantic model.", color="danger")
        except Exception as e:
            callback_logger.error(f"❌ Error accessing dashboard configuration: {e}")
            return dbc.Alert(f"Error accessing dashboard configuration: {e}", color="danger")

        # STRICT PYDANTIC V2 ONLY - No dictionary acceptance
        if isinstance(modes_config, dict):
            callback_logger.error(f"❌ modes_detail_config is a dictionary instead of DashboardModeCollection Pydantic model")
            return dbc.Alert("Dashboard configuration is not properly parsed as Pydantic model.", color="danger")

        if not hasattr(modes_config, mode_key) and not hasattr(modes_config, 'main'):
            callback_logger.error(f"❌ Mode '{mode_key}' not found in DashboardModeCollection and no 'main' fallback")
            return dbc.Alert(f"Mode '{mode_key}' not configured.", color="danger")

        # Get mode info from Pydantic model attributes ONLY
        mode_info = None
        if hasattr(modes_config, mode_key):
            mode_info = getattr(modes_config, mode_key)
        elif hasattr(modes_config, 'main'):
            mode_info = getattr(modes_config, 'main')

        # STRICT PYDANTIC V2 ONLY - mode_info must be a Pydantic model
        if isinstance(mode_info, dict):
            callback_logger.error(f"❌ mode_info for '{mode_key}' is a dictionary instead of DashboardModeSettings Pydantic model")
            return dbc.Alert(f"Mode '{mode_key}' configuration is not properly parsed as Pydantic model.", color="danger")

        if not mode_info or not hasattr(mode_info, 'module_name'):
            callback_logger.error(f"❌ Mode '{mode_key}' does not have required module_name attribute")
            return dbc.Alert(f"Mode '{mode_key}' configuration is invalid.", color="danger")

        module_name = mode_info.module_name


        # Enforce Pydantic-only for dynamic_thresholds
        dynamic_thresholds = None
        if hasattr(mode_info, 'dynamic_thresholds'):
            dynamic_thresholds = getattr(mode_info, 'dynamic_thresholds')
            if dynamic_thresholds is not None and not isinstance(dynamic_thresholds, DynamicThresholdsV2_5):
                callback_logger.error(f"dynamic_thresholds for mode '{mode_key}' is not a DynamicThresholdsV2_5 model")
                raise TypeError(f"dynamic_thresholds for mode '{mode_key}' must be a DynamicThresholdsV2_5 Pydantic model, got {type(dynamic_thresholds)}")
        # No dict or fallback logic allowed
        # Import and render with proper module path handling
        try:
            # Handle different module path formats
            if module_name.startswith('dashboard_application.modes.'):
                # Full path already provided
                full_module_name = module_name
            elif '.' in module_name:
                # Subdirectory case like "ai_dashboard.ai_dashboard_display_v2_5"
                full_module_name = f"dashboard_application.modes.{module_name}"
            else:
                # Simple module name like "main_dashboard_display_v2_5"
                full_module_name = f"dashboard_application.modes.{module_name}"

            callback_logger.info(f"🔄 Importing module: {full_module_name} for mode: {mode_key}")
            display_module = importlib.import_module(full_module_name)

            # Validate and parse bundle using strict Pydantic v2
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)

            # Create layout with proper Pydantic v2 models
            mode_layout = display_module.create_layout(bundle, dashboard_config)
            callback_logger.info(f"✅ Successfully created layout for mode: {mode_key}")
            return mode_layout

        except ImportError as e:
            callback_logger.error(f"❌ Could not import display module: {full_module_name} - {e}")
            return dbc.Alert(f"Error loading UI module for mode '{mode_key}': Module not found", color="danger")
        except Exception as e:
            callback_logger.error(f"❌ An unexpected error occurred while rendering the {mode_key} view: {e}", exc_info=True)
            return dbc.Alert(f"An unexpected error occurred while rendering the {mode_key} view: {e}", color="danger")
            
    # --- Callback to update Refresh Interval ---
    @app.callback(
        Output(ids.ID_INTERVAL_LIVE_UPDATE, 'interval'),
        Input(ids.ID_REFRESH_INTERVAL_DROPDOWN, 'value')
    )
    def update_refresh_interval(interval_seconds: str) -> int:
        """Updates the dcc.Interval component's refresh rate."""
        return int(interval_seconds) * 1000 if interval_seconds else 60 * 1000

    # --- Status Update Display Callback ---
    @app.callback(
        [
            Output('current-symbol', 'children'),
            Output('current-dte-range', 'children'),
            Output('current-price-range', 'children'),
            Output('contracts-count', 'children'),
            Output('strikes-count', 'children'),
            Output('processing-time', 'children'),
            Output('last-update-time', 'children'),
            Output(ids.ID_ELITE_IMPACT_SCORE_DISPLAY, 'children'),
            Output(ids.ID_INSTITUTIONAL_FLOW_SCORE_DISPLAY, 'children'),
            Output(ids.ID_FLOW_MOMENTUM_INDEX_DISPLAY, 'children'),
            Output(ids.ID_ELITE_MARKET_REGIME_DISPLAY, 'children'),
            Output(ids.ID_ELITE_FLOW_TYPE_DISPLAY, 'children'),
            Output(ids.ID_ELITE_VOLATILITY_REGIME_DISPLAY, 'children')
        ],
        [
            Input(ids.ID_MAIN_DATA_STORE, 'data'),
            Input(ids.ID_INTERVAL_LIVE_UPDATE, 'n_intervals')
        ],
        [
            State(ids.ID_SYMBOL_INPUT, 'value'),
            State('dte-min-input', 'value'),
            State('dte-max-input', 'value'),
            State('price-range-input', 'value')
        ],
        prevent_initial_call=True
    )
    def update_status_display(bundle_json: str, n_intervals: int, symbol: str, dte_min: int, dte_max: int, price_range_percent: int) -> tuple:
        """
        Updates the status display with current analysis information.
        """
        if not bundle_json:
            # No data, show clear placeholders instead of static values
            return (
                "No data available", "-- to --", "±--%", "---", "---", "---", "--:--:--",
                "---", "---", "---", "---", "---", "---"
            )
        try:
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            
            # Extract information from bundle
            symbol_display = symbol or bundle.target_symbol or "Unknown"
            timestamp = bundle.bundle_timestamp
            
            # Format timestamp
            if timestamp:
                last_update = timestamp.strftime("%H:%M:%S")
            else:
                last_update = "--:--:--"
            
            # Use control panel values for DTE range display
            if dte_min is not None and dte_max is not None:
                if dte_min == dte_max:
                    dte_range = f"{dte_min} DTE"
                else:
                    dte_range = f"{dte_min} to {dte_max}"
            else:
                dte_range = "-- to --"
            
            # Use control panel value for price range display
            if price_range_percent is not None:
                price_range = f"±{price_range_percent}%"
            else:
                price_range = "±--%"
            
            # Get contracts and strikes count from actual data
            contracts_count = len(bundle.processed_data_bundle.options_data_with_metrics) if bundle.processed_data_bundle and bundle.processed_data_bundle.options_data_with_metrics else 0
            strikes_count = len(bundle.processed_data_bundle.strike_level_data_with_metrics) if bundle.processed_data_bundle and bundle.processed_data_bundle.strike_level_data_with_metrics else 0
            
            # Calculate processing time from bundle timestamps
            processing_time_display = "---"
            if (bundle.bundle_timestamp and 
                bundle.processed_data_bundle and 
                bundle.processed_data_bundle.processing_timestamp):
                
                start_time = bundle.processed_data_bundle.processing_timestamp
                end_time = bundle.bundle_timestamp
                
                # Handle timezone compatibility
                if start_time.tzinfo != end_time.tzinfo:
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=end_time.tzinfo)
                    elif end_time.tzinfo is None:
                        end_time = end_time.replace(tzinfo=start_time.tzinfo)
                
                processing_duration = (end_time - start_time).total_seconds()
                
                # Format processing time in a more readable way
                if processing_duration < 0.001:  # Less than 1ms
                    processing_time_display = "<1ms"
                elif processing_duration < 1:  # Less than 1 second, show in milliseconds
                    ms = processing_duration * 1000
                    if ms < 10:
                        processing_time_display = f"{ms:.1f}ms"
                    else:
                        processing_time_display = f"{ms:.0f}ms"
                else:  # 1 minute or more
                    minutes = int(processing_duration // 60)
                    seconds = processing_duration % 60
                    processing_time_display = f"{minutes}m {seconds:.1f}s"
            
            # Elite Metrics for display
            und_enriched = bundle.processed_data_bundle.underlying_data_enriched
            elite_impact_score = f"{und_enriched.elite_impact_score_und:.4f}" if und_enriched.elite_impact_score_und is not None else "---"
            institutional_flow_score = f"{und_enriched.institutional_flow_score_und:.4f}" if und_enriched.institutional_flow_score_und is not None else "---"
            flow_momentum_index = f"{und_enriched.flow_momentum_index_und:.4f}" if und_enriched.flow_momentum_index_und is not None else "---"
            elite_market_regime = (
                und_enriched.market_regime_elite.replace("_", " ").title() if und_enriched.market_regime_elite else "---"
            )
            # Handle both Enum and str for elite_flow_type
            if isinstance(und_enriched.flow_type_elite, FlowType):
                elite_flow_type = und_enriched.flow_type_elite.value.replace("_", " ").title()
            elif isinstance(und_enriched.flow_type_elite, str):
                elite_flow_type = und_enriched.flow_type_elite.replace("_", " ").title()
            else:
                elite_flow_type = "---"
            elite_volatility_regime = und_enriched.volatility_regime_elite.replace("_", " ").title() if und_enriched.volatility_regime_elite else "---"

            return (
                symbol_display,
                dte_range,
                price_range,
                str(contracts_count),
                str(strikes_count),
                processing_time_display,
                last_update,
                elite_impact_score,
                institutional_flow_score,
                flow_momentum_index,
                elite_market_regime,
                elite_flow_type,
                elite_volatility_regime
            )
            
        except Exception as e:
            callback_logger.error(f"Error updating status display: {e}", exc_info=True)
            return (
                "ERROR", "-- to --", "±--%", "---", "---", "---", "ERROR",
                "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"
            )

    # --- Collapsible About Section Callbacks ---
    @app.callback(
        Output("regime-about-collapse", "is_open"),
        Input("regime-about-toggle", "n_clicks"),
        State("regime-about-collapse", "is_open"),
        prevent_initial_call=True
    )
    def toggle_regime_about(n_clicks, is_open):
        """Toggle the regime about section."""
        if n_clicks:
            return not is_open
        return is_open

    # Generic callback for all chart about sections using pattern matching
    @app.callback(
        Output({"type": "about-collapse", "index": dash.MATCH}, "is_open"),
        Input({"type": "about-toggle", "index": dash.MATCH}, "n_clicks"),
        State({"type": "about-collapse", "index": dash.MATCH}, "is_open"),
        prevent_initial_call=True
    )
    def toggle_chart_about(n_clicks, is_open):
        """Toggle chart about section independently using pattern matching (MATCH)."""
        if n_clicks:
            return not is_open
        return is_open

    # --- Collapsible About Section Callback (for all cards using the new pattern) ---
    @app.callback(
        Output({"type": "about-collapse", "section": dash.MATCH}, "is_open"),
        Input({"type": "about-toggle-btn", "section": dash.MATCH}, "n_clicks"),
        State({"type": "about-collapse", "section": dash.MATCH}, "is_open"),
        prevent_initial_call=True
    )
    def toggle_about_section(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # --- Regime Display Live Update Callback ---
    @app.callback(
        Output('regime-display-container', 'children'),
        Input(ids.ID_MAIN_DATA_STORE, 'data'),
        prevent_initial_call=False
    )
    def update_regime_display(bundle_json):
        try:
            if not bundle_json:
                # No data yet, return None so the UI shows nothing or a clear error
                return None
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            und_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
            if und_data is None:
                return None
            if CONFIG_REF is None:
                raise RuntimeError("CONFIG_REF is None. Cannot render regime display without a valid config.")
            return _create_regime_display(und_data, CONFIG_REF)
        except Exception as e:
            callback_logger.error(f"Error in update_regime_display: {e}")
            return None

    callback_logger.info("EOTS v2.5 authoritative callbacks registered successfully.")