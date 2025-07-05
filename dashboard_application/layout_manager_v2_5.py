# dashboard_application/layout_manager_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MASTER LAYOUT DEFINITION

from dash import dcc, html
import dash_bootstrap_components as dbc
import datetime
from typing import Dict, Any, Optional
import logging

from dashboard_application import ids
from utils.config_manager_v2_5 import ConfigManagerV2_5

# Explicit import for clarity and robustness
from data_models import ProcessedUnderlyingAggregatesV2_5  # Explicit import
from data_models.dashboard_config_models import DashboardModeCollection # Add this import
from data_models.elite_intelligence import MarketRegime, FlowType, EliteImpactColumns # Import Elite Enums

def create_control_panel(config: ConfigManagerV2_5) -> dbc.Card:
    """Creates the control panel with symbol input, fetch button, and settings."""
    try:
        # Get default values from config
        vis_defaults = config.config.visualization_settings.dashboard.defaults
        default_symbol = vis_defaults.symbol
        default_refresh = vis_defaults.update_interval_seconds
        default_dte_min = vis_defaults.dte_min
        default_dte_max = vis_defaults.dte_max
        default_price_range = vis_defaults.price_range_percent

        print("Creating control panel...")
        print("Control Panel: symbol=" + default_symbol + ", refresh=" + str(default_refresh))

    except Exception as e:
        print("Error reading config in control panel: " + str(e))
        # Fallback values
        default_symbol = 'SPY'
        default_refresh = 30
        default_dte_min = 0
        default_dte_max = 45
        default_price_range = 20
    
    control_panel = dbc.Card([
        dbc.CardHeader(html.H5("🎛️ EOTS Control Panel", className="mb-0")),
        dbc.CardBody([
            # Row 1: Main Controls
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol:", html_for=ids.ID_SYMBOL_INPUT, className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id=ids.ID_SYMBOL_INPUT,
                            type="text",
                            value=default_symbol,
                            placeholder="Enter symbol (e.g., SPY)",
                            className="form-control"
                        )
                    ], className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("DTE Range:", html_for="dte-range-input", className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="dte-min-input",
                            type="number",
                            value=default_dte_min,
                            placeholder="Min",
                            min=0,
                            max=365,
                            className="form-control me-1"
                        ),
                        dbc.InputGroupText("to", className="bg-elite-surface text-elite-secondary"),
                        dbc.Input(
                            id="dte-max-input",
                            type="number",
                            value=default_dte_max,
                            placeholder="Max",
                            min=0,
                            max=365,
                            className="form-control"
                        )
                    ], size="sm", className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Price Range %:", html_for="price-range-input", className="form-label"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="price-range-input",
                            type="number",
                            value=default_price_range,
                            placeholder="±%",
                            min=1,
                            max=100,
                            step=1,
                            className="form-control"
                        ),
                        dbc.InputGroupText("%", className="bg-elite-surface text-elite-secondary")
                    ], size="sm", className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Refresh:", html_for=ids.ID_REFRESH_INTERVAL_DROPDOWN, className="form-label"),
                    html.Div([
                        dcc.Dropdown(
                            id=ids.ID_REFRESH_INTERVAL_DROPDOWN,
                            options=[
                                {"label": "15s", "value": 15},
                                {"label": "30s", "value": 30},
                                {"label": "1m", "value": 60},
                                {"label": "2m", "value": 120},
                                {"label": "5m", "value": 300},
                                {"label": "Off", "value": 999999999}
                            ],
                            value=default_refresh,
                            style={
                                'backgroundColor': 'var(--elite-bg-primary)',
                                'border': '1px solid var(--elite-border-primary)',
                                'borderRadius': 'var(--elite-radius-md)'
                            }
                        )
                    ], className="control-input-group mb-2")
                ], width=2),
                dbc.Col([
                    dbc.Label("Actions:", className="d-block"),
                    dbc.Button(
                        "🚀 Fetch Data",
                        id=ids.ID_MANUAL_REFRESH_BUTTON,
                        size="sm",
                        className="btn-elite-primary mb-2 elite-focus-visible",
                        style={"width": "100%"}
                    )
                ], width=2),
                dbc.Col([
                    dbc.Label("Status:", className="d-block"),
                    html.Div(id="status-indicator", children=[
                        dbc.Badge("Ready", color="secondary", className="me-1"),
                        html.Small("Enter symbol and click Fetch Data", className="text-muted")
                    ])
                ], width=2)
            ], align="center"),
            
            # Row 2: STATUS UPDATE Section
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("📊 STATUS UPDATE", className="mb-2 text-primary"),
                    html.Div(id="status-update-display", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Small("Symbol:", className="text-muted d-block"),
                                html.Span("---", id="current-symbol", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("DTE Range:", className="text-muted d-block"),
                                html.Span("-- to --", id="current-dte-range", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Price Range:", className="text-muted d-block"),
                                html.Span("±--%", id="current-price-range", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Contracts:", className="text-muted d-block"),
                                html.Span("---", id="contracts-count", className="fw-bold")
                            ], width=1),
                            dbc.Col([
                                html.Small("Strikes:", className="text-muted d-block"),
                                html.Span("---", id="strikes-count", className="fw-bold")
                            ], width=1),
                            dbc.Col([
                                html.Small("Processing Time:", className="text-muted d-block"),
                                html.Span("---", id="processing-time", className="fw-bold")
                            ], width=2),
                            dbc.Col([
                                html.Small("Last Update:", className="text-muted d-block"),
                                html.Span("--:--:--", id="last-update-time", className="fw-bold")
                            ], width=2)
                        ], className="g-2")
                    ])
                ], width=12)
            ], className="mt-2")
        ])
    ], className="mb-4 elite-control-panel")
    
    print("Control panel created successfully")
    return control_panel

def create_header(config: ConfigManagerV2_5) -> dbc.Navbar:
    """Creates the persistent header and navigation bar for the application."""
    
    # Dynamically build navigation links from the Pydantic config model
    modes_config = config.config.visualization_settings.dashboard.modes_detail_config
    nav_links = []
    # Ensure 'main' mode is first if it exists
    if hasattr(modes_config, 'main') and modes_config.main:
        nav_links.append(
            dbc.NavLink(
                modes_config.main.label, 
                href="/", 
                active="exact",
                className="nav-link-custom"
            )
        )
    # Add other modes
    for mode_name in ['flow', 'volatility', 'structure', 'timedecay', 'advanced', 'ai']:
        mode_obj = getattr(modes_config, mode_name, None)
        if mode_obj:
            nav_links.append(
                dbc.NavLink(
                    mode_obj.label, 
                    href=f"/{mode_name}", 
                    active="exact",
                    className="nav-link-custom"
                )
            )

    # Get visualization settings for styling
    vis_settings = config.config.visualization_settings.dashboard.defaults
    
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand("EOTS v2.5 - Elite Options Trading System", className="ms-2"),
                ], width=6),
                dbc.Col([
                    dbc.Nav(nav_links, className="ms-auto", navbar=True)
                ], width=6)
            ], align="center", className="w-100")
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-3"
    )

def create_master_layout(config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the master layout for the entire Dash application.

    This layout includes core non-visual components for state management and routing,
    and defines the main structure including the header and content area.
    """
    print("Creating master layout...")

    # Fetch default refresh interval for dcc.Interval
    try:
        vis_defaults = config.config.visualization_settings.dashboard.defaults
    except Exception:
        vis_defaults = None

    if vis_defaults:
        initial_refresh_seconds = int(vis_defaults.update_interval_seconds)
    else:
        initial_refresh_seconds = 30  # Default fallback value

    initial_refresh_ms = initial_refresh_seconds * 1000

    # Determine if interval should be disabled if "Off" (very large number) is the default
    interval_disabled = True if initial_refresh_seconds >= 999999999 else False

    print("Creating control panel...")
    control_panel_component = create_control_panel(config)
    print("Control panel created: " + str(control_panel_component is not None))

    # --- Regime Display: Insert directly below control panel ---
    # Remove fake placeholder data and enforce Pydantic v2 model validation with fail-fast
    try:
        # Create a minimal valid instance with required fields only, no fake zero data
        und_data_placeholder = ProcessedUnderlyingAggregatesV2_5(
            symbol='SPY',
            timestamp=datetime.datetime.now(),
            price=100.0,  # Valid price > 0
            price_change_abs_und=0.0,
            price_change_pct_und=0.0,
            day_open_price_und=0.0,
            day_high_price_und=0.0,
            day_low_price_und=0.0,
            prev_day_close_price_und=0.0,
            u_volatility=0.0,
            day_volume=0.0,
            call_gxoi=0.0,
            put_gxoi=0.0,
            gammas_call_buy=0.0,
            gammas_call_sell=0.0,
            gammas_put_buy=0.0,
            gammas_put_sell=0.0,
            deltas_call_buy=0.0,
            deltas_call_sell=0.0,
            deltas_put_buy=0.0,
            deltas_put_sell=0.0,
            vegas_call_buy=0.0,
            vegas_call_sell=0.0,
            vegas_put_buy=0.0,
            vegas_put_sell=0.0,
            thetas_call_buy=0.0,
            thetas_call_sell=0.0,
            thetas_put_buy=0.0,
            thetas_put_sell=0.0,
            call_vxoi=0.0,
            put_vxoi=0.0,
            value_bs=0.0,
            volm_bs=0.0,
            deltas_buy=0.0,
            deltas_sell=0.0,
            vegas_buy=0.0,
            vegas_sell=0.0,
            thetas_buy=0.0,
            thetas_sell=0.0,
            volm_call_buy=0.0,
            volm_put_buy=0.0,
            volm_call_sell=0.0,
            volm_put_sell=0.0,
            value_call_buy=0.0,
            value_put_buy=0.0,
            value_call_sell=0.0,
            value_put_sell=0.0,
            vflowratio=0.0,
            dxoi=0.0,
            gxoi=0.0,
            vxoi=0.0,
            txoi=0.0,
            call_dxoi=0.0,
            put_dxoi=0.0,
            tradier_iv5_approx_smv_avg=0.0,
            total_call_oi_und=0.0,
            total_put_oi_und=0.0,
            total_call_vol_und=0.0,
            total_put_vol_und=0.0,
            tradier_open=0.0,
            tradier_high=0.0,
            tradier_low=0.0,
            tradier_close=0.0,
            tradier_volume=0.0,
            tradier_vwap=0.0,
            gib_oi_based_und=0.0,
            td_gib_und=0.0,
            hp_eod_und=0.0,
            net_cust_delta_flow_und=0.0,
            net_cust_gamma_flow_und=0.0,
            net_cust_vega_flow_und=0.0,
            net_cust_theta_flow_und=0.0,
            net_value_flow_5m_und=0.0,
            net_vol_flow_5m_und=0.0,
            net_value_flow_15m_und=0.0,
            net_vol_flow_15m_und=0.0,
            net_value_flow_30m_und=0.0,
            net_vol_flow_30m_und=0.0,
            net_value_flow_60m_und=0.0,
            net_vol_flow_60m_und=0.0,
            vri_0dte_und_sum=0.0,
            vfi_0dte_und_sum=0.0,
            vvr_0dte_und_avg=0.0,
            vci_0dte_agg=0.0,
            arfi_overall_und_avg=0.0,
            a_mspi_und_summary_score=0.0,
            a_sai_und_avg=0.0,
            a_ssi_und_avg=0.0,
            vri_2_0_und_aggregate=0.0,
            vapi_fa_z_score_und=0.0,
            dwfd_z_score_und=0.0,
            tw_laf_z_score_und=0.0,
            ivsdh_surface_data=None,
            current_market_regime_v2_5=None,
            ticker_context_dict_v2_5=None,
            atr_und=0.0,
        )
    except Exception as e:
        print("Error creating und_data_placeholder: " + str(e))
        # Fail fast: re-raise exception to prevent silent fallback
        raise

    regime_display_card = _create_regime_display(und_data_placeholder, config)
    # TODO: Wire up regime display to update with live data via callback if needed

    layout = html.Div(
        id='app-container',
        children=[
            dcc.Location(id=ids.ID_URL_LOCATION, refresh=False),
            dcc.Store(id=ids.ID_MAIN_DATA_STORE, storage_type='memory'), # Stores the main analysis bundle
            dcc.Interval(
                id=ids.ID_INTERVAL_LIVE_UPDATE,
                interval=initial_refresh_ms,
                n_intervals=0,
                disabled=interval_disabled # Control if interval timer is active
            ),
            
            create_header(config), # Header is persistent
            
            # Control panel with symbol input and fetch button
            dbc.Container([
                control_panel_component,
                html.Div(regime_display_card, id='regime-display-container'),  # <--- Regime display now in a container
            ], fluid=True),
            
            # Area for status alerts (e.g., data updated, errors)
            html.Div(id=ids.ID_STATUS_ALERT_CONTAINER,
                     style={"position": "fixed", "top": "120px", "right": "10px", "zIndex": "1050", "width": "auto"}),

            # Main content area, dynamically updated by callbacks based on URL
            html.Main(
                id='app-body',
                className='app-body container-fluid p-3', # Use container-fluid for responsive padding
                children=[
                    dbc.Container(id=ids.ID_PAGE_CONTENT, fluid=True, children=[ # Ensure page content also uses fluid container
                        dbc.Spinner(color="primary", children=html.Div("Waiting for initial data fetch..."))
                    ])
                ]
            )
        ]
    )
    
    print("Master layout created successfully")
    return layout

def _create_regime_display(und_data: ProcessedUnderlyingAggregatesV2_5, config: ConfigManagerV2_5) -> dbc.Card:
    """Creates the market regime display card."""
    logger = logging.getLogger(__name__)
    main_dash_settings = getattr(config, 'main_dashboard_settings', lambda: {})() if hasattr(config, 'main_dashboard_settings') else {}
    regime_settings = main_dash_settings.get("regime_display", {})
    regime_title = regime_settings.get("title", "Market Regime")
    regime_blurb = "🧠 Market Regime Engine: Analyzes current market conditions using multiple metrics. Helps determine optimal strategy types and risk parameters. Green = Bullish conditions, Red = Bearish conditions, Yellow = Transitional/Unclear."

    card_body_children = [
        html.H6(f"{regime_title}", className="elite-card-title text-center"),
        dbc.Button(
            "ℹ️ About",
            id="regime-about-toggle",
            color="link",
            size="sm",
            className="p-0 text-elite-secondary",
            style={'font-size': '0.75em'}
        ),
        dbc.Collapse(
            html.Small(regime_blurb, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
            id="regime-about-collapse",
            is_open=False
        ),
        html.Hr(className="my-2"),
        dbc.Row([
            dbc.Col([
                html.Small("Elite Impact Score:", className="text-muted d-block"),
                html.Span(
                    f"{getattr(und_data, 'elite_impact_score_und', 0.0):.4f}" if getattr(und_data, 'elite_impact_score_und', None) is not None else "---",
                    id=ids.ID_ELITE_IMPACT_SCORE_DISPLAY, className="fw-bold"
                )
            ], width=4),
            dbc.Col([
                html.Small("Institutional Flow:", className="text-muted d-block"),
                html.Span(
                    f"{getattr(und_data, 'institutional_flow_score_und', 0.0):.4f}" if getattr(und_data, 'institutional_flow_score_und', None) is not None else "---",
                    id=ids.ID_INSTITUTIONAL_FLOW_SCORE_DISPLAY, className="fw-bold"
                )
            ], width=4),
            dbc.Col([
                html.Small("Flow Momentum:", className="text-muted d-block"),
                html.Span(
                    f"{getattr(und_data, 'flow_momentum_index_und', 0.0):.4f}" if getattr(und_data, 'flow_momentum_index_und', None) is not None else "---",
                    id=ids.ID_FLOW_MOMENTUM_INDEX_DISPLAY, className="fw-bold"
                )
            ], width=4)
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col([
                html.Small("Elite Market Regime:", className="text-muted d-block"),
                html.Span(
                    str(getattr(und_data, 'market_regime_elite', MarketRegime.LOW_VOL_RANGING.value) or MarketRegime.LOW_VOL_RANGING.value).replace('_', ' ').title(),
                    id=ids.ID_ELITE_MARKET_REGIME_DISPLAY, className="fw-bold"
                )
            ], width=6),
            dbc.Col([
                html.Small("Elite Flow Type:", className="text-muted d-block"),
                html.Span(
                    str(getattr(und_data, 'flow_type_elite', FlowType.MARKET_MAKER_FLOW.value) or FlowType.MARKET_MAKER_FLOW.value).replace('_', ' ').title(),
                    id=ids.ID_ELITE_FLOW_TYPE_DISPLAY, className="fw-bold"
                )
            ], width=6)
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col([
                html.Small("Elite Volatility Regime:", className="text-muted d-block"),
                html.Span(
                    str(getattr(und_data, 'volatility_regime_elite', MarketRegime.LOW_VOL_RANGING.value) or MarketRegime.LOW_VOL_RANGING.value).replace('_', ' ').title(),
                    id=ids.ID_ELITE_VOLATILITY_REGIME_DISPLAY, className="fw-bold"
                )
            ], width=12)
        ], className="g-2 mb-2")
    ]

    # Robust regime extraction and display
    try:
        regime = getattr(und_data, 'current_market_regime_v2_5', None)
        logger.info(f"[Regime Display] Extracted regime: {regime} (type: {type(regime)})")
        if regime is None:
            regime = "UNKNOWN"
        elif hasattr(regime, 'value'):
            regime = regime.value
        elif not isinstance(regime, str):
            regime = str(regime)
        # Defensive: ensure string for .upper() and .replace()
        if not isinstance(regime, str):
            regime = "UNKNOWN"
        if "BULL" in regime.upper() or "POSITIVE" in regime.upper():
            alert_color = "success"
        elif "BEAR" in regime.upper() or "NEGATIVE" in regime.upper():
            alert_color = "danger"
        elif "UNCLEAR" in regime.upper() or "TRANSITION" in regime.upper():
            alert_color = "warning"
        else:
            alert_color = "info"
        card_body_children.append(
            dbc.Alert(regime.replace("_", " ").title(), color=alert_color, className="mt-2 text-center fade-in-up")
        )
    except Exception as e:
        logger.error(f"[Regime Display] Error rendering regime: {e}", exc_info=True)
        card_body_children.append(
            dbc.Alert("Regime display unavailable", color="danger", className="mt-2 text-center fade-in-up")
        )

    return dbc.Card(dbc.CardBody(card_body_children, className="elite-card-body"), className="elite-card")