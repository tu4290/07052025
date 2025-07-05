"""
Compass Update Manager for Real-Time Updates
===========================================

Handles real-time data polling, state management, and callbacks
for the Legendary Market Compass component.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time

from dash import callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data_models import ProcessedDataBundleV2_5
from .compass_metrics_engine import CompassMetricsEngine
from .legendary_compass_component import LegendaryMarketCompass

logger = logging.getLogger(__name__)

class CompassUpdateManager:
    """
    Manages real-time updates and state for the Legendary Market Compass.
    Implements polling, caching, and efficient update strategies.
    """
    
    def __init__(self, app, api_client):
        self.app = app
        self.api_client = api_client
        self.metrics_engine = CompassMetricsEngine()
        self.legendary_compass = LegendaryMarketCompass()
        
        # State management
        self.last_update_time = {}
        self.dimension_cache = {}
        self.pattern_cache = {}
        self.update_interval = 5  # seconds
        self.max_cache_age = 300  # 5 minutes
        
        # Threading for background updates
        self.update_thread = None
        self.is_running = False
        
        # Register callbacks
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register all Dash callbacks for the compass."""
        
        # Main update callback
        @self.app.callback(
            Output('legendary-compass-graph', 'figure'),
            Output('compass-pattern-insights', 'children'),
            Output('compass-last-update', 'children'),
            [
                Input('compass-update-interval', 'n_intervals'),
                Input('compass-timeframe-selector', 'value')
            ],
            [
                State('selected-symbol', 'data'),
                State('compass-history-store', 'data')
            ]
        )
        def update_compass_display(n_intervals, timeframe, symbol, history_data):
            """Update the compass visualization based on interval and user selections."""
            
            if not symbol:
                raise PreventUpdate
            
            try:
                # Get latest data
                dimensions, patterns = self._get_latest_compass_data(symbol)
                
                if not dimensions:
                    raise PreventUpdate
                
                # Create figure based on timeframe selection
                figure = self.legendary_compass.create_enhanced_compass_figure(
                    dimensions, patterns, timeframe
                )
                
                # Create pattern insights
                pattern_insights = self._create_pattern_insights(patterns)
                
                # Update timestamp
                update_text = f"🟢 Live Data • Updated {datetime.now().strftime('%H:%M:%S')}"
                
                return figure, pattern_insights, update_text
                
            except Exception as e:
                logger.error(f"Error updating compass: {e}")
                raise PreventUpdate
        
        # Timeframe change callback
        @self.app.callback(
            Output('compass-analysis-mode', 'data'),
            Input('compass-timeframe-selector', 'value')
        )
        def handle_timeframe_change(timeframe):
            """Handle timeframe selection changes."""
            logger.info(f"Compass timeframe changed to: {timeframe}")
            return {'timeframe': timeframe, 'timestamp': datetime.now().isoformat()}
        
        # Pattern alert callback
        @self.app.callback(
            Output('compass-alert-modal', 'is_open'),
            Output('compass-alert-content', 'children'),
            Input('compass-pattern-store', 'data'),
            State('compass-alert-threshold', 'value')
        )
        def check_pattern_alerts(pattern_data, alert_threshold):
            """Check for high-strength patterns and trigger alerts."""
            
            if not pattern_data or not alert_threshold:
                return False, ""
            
            high_strength_patterns = [
                p for p in pattern_data.get('patterns', [])
                if p['strength'] >= alert_threshold / 100
            ]
            
            if high_strength_patterns:
                alert_content = self._create_alert_content(high_strength_patterns)
                return True, alert_content
            
            return False, ""
        
        # Export callback
        @self.app.callback(
            Output('compass-export-download', 'data'),
            Input('compass-export-btn', 'n_clicks'),
            State('legendary-compass-graph', 'figure'),
            State('selected-symbol', 'data')
        )
        def export_compass_data(n_clicks, figure, symbol):
            """Export compass data and visualization."""
            
            if not n_clicks:
                raise PreventUpdate
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'compass_{symbol}_{timestamp}.json'
            
            export_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'figure': figure,
                'dimensions': self.dimension_cache.get(symbol, []),
                'patterns': self.pattern_cache.get(symbol, [])
            }
            
            return dict(content=export_data, filename=filename)
    
    def _get_latest_compass_data(self, symbol: str) -> tuple:
        """
        Get the latest compass data from cache or API.
        
        Returns:
            Tuple of (dimensions, patterns)
        """
        cache_key = symbol
        current_time = datetime.now()
        
        # Check cache validity
        if cache_key in self.last_update_time:
            age = (current_time - self.last_update_time[cache_key]).total_seconds()
            if age < self.update_interval:
                # Return cached data
                return (
                    self.dimension_cache.get(cache_key, []),
                    self.pattern_cache.get(cache_key, [])
                )
        
        # Fetch fresh data
        try:
            # Get processed data bundle from API
            processed_data = self._fetch_processed_data(symbol)
            
            if processed_data:
                # Calculate dimensions and patterns
                dimensions = self.metrics_engine.calculate_all_dimensions(processed_data)
                patterns = self.metrics_engine.detect_confluence_patterns(dimensions)
                
                # Update cache
                self.dimension_cache[cache_key] = dimensions
                self.pattern_cache[cache_key] = patterns
                self.last_update_time[cache_key] = current_time
                
                # Update history in the compass
                self.legendary_compass.update_compass_history(dimensions, patterns)
                
                return dimensions, patterns
            
        except Exception as e:
            logger.error(f"Error fetching compass data for {symbol}: {e}")
        
        # Return empty data on error
        return [], []
    
    def _fetch_processed_data(self, symbol: str) -> Optional[ProcessedDataBundleV2_5]:
        """
        Fetch processed data bundle from the API.
        This should integrate with your existing API infrastructure.
        """
        try:
            # This is a placeholder - integrate with your actual API
            # response = self.api_client.get_processed_data(symbol)
            # return ProcessedDataBundleV2_5.model_validate(response)
            
            # For now, return None to trigger fallback behavior
            logger.debug(f"Fetching processed data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching processed data: {e}")
            return None
    
    def _create_pattern_insights(self, patterns: List[Dict[str, Any]]) -> List:
        """Create pattern insight components for display."""
        
        if not patterns:
            return []
        
        insights = []
        for p in patterns[:5]:  # Top 5 patterns
            insight = html.Div([
                html.Span(f"⚡ {p['name']}", style={
                    'color': '#ff8000',
                    'fontWeight': 'bold',
                    'marginRight': '10px'
                }),
                html.Span(p['action'], style={'color': 'white'}),
                html.Span(f" ({p['strength']:.0%})", style={
                    'color': '#808080',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '5px'})
            insights.append(insight)
        
        return insights
    
    def _create_alert_content(self, patterns: List[Dict[str, Any]]) -> html.Div:
        """Create alert modal content for high-strength patterns."""
        
        return html.Div([
            html.H4("🚨 High-Strength Pattern Alert", style={'color': '#ff0000'}),
            html.Hr(),
            html.Div([
                html.Div([
                    html.H5(f"{p['name']}", style={'color': '#ff8000'}),
                    html.P(p['description']),
                    html.P(f"Action: {p['action']}", style={'fontWeight': 'bold'}),
                    html.P(f"Strength: {p['strength']:.0%}")
                ], style={'marginBottom': '20px'})
                for p in patterns
            ])
        ])
    
    def start_background_updates(self):
        """Start background thread for continuous updates."""
        
        if not self.is_running:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._background_update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Started compass background update thread")
    
    def stop_background_updates(self):
        """Stop background updates."""
        
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
            logger.info("Stopped compass background update thread")
    
    def _background_update_loop(self):
        """Background loop for pre-fetching data."""
        
        while self.is_running:
            try:
                # Update data for all cached symbols
                for symbol in list(self.dimension_cache.keys()):
                    self._get_latest_compass_data(symbol)
                
                # Clean old cache entries
                self._clean_cache()
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
            
            time.sleep(self.update_interval)
    
    def _clean_cache(self):
        """Remove old cache entries."""
        
        current_time = datetime.now()
        symbols_to_remove = []
        
        for symbol, last_update in self.last_update_time.items():
            age = (current_time - last_update).total_seconds()
            if age > self.max_cache_age:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            self.dimension_cache.pop(symbol, None)
            self.pattern_cache.pop(symbol, None)
            self.last_update_time.pop(symbol, None)
            logger.debug(f"Removed stale cache for {symbol}")

# Add this import statement to the html module
from dash import html