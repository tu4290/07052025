#!/usr/bin/env python3
"""
AI Dashboard Display v2.5 - Main Entry Point
===========================================

This module serves as the main entry point for the AI Dashboard mode,
providing a unified interface for creating AI dashboard layouts.

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from typing import Dict, Any, Optional

# Import the main layout function from enhanced_ai_hub_layout
from .enhanced_ai_hub_layout import create_ai_dashboard_layout

# Set up logging
logger = logging.getLogger(__name__)

def create_layout(bundle, config):
    """
    Create the AI dashboard layout.
    
    This function serves as the main entry point for creating AI dashboard layouts,
    following the standard interface used by all dashboard modes.
    
    Args:
        bundle: FinalAnalysisBundleV2_5 instance containing market and options data
        config: ConfigManagerV2_5 instance containing configuration settings
        
    Returns:
        Dash layout components for the AI dashboard
    """
    try:
        logger.info("Creating AI dashboard layout")
        from data_models import FinalAnalysisBundleV2_5
        from pydantic import BaseModel
        if not isinstance(bundle, FinalAnalysisBundleV2_5):
            raise TypeError(f"bundle must be FinalAnalysisBundleV2_5, got {type(bundle)}")
        if not (hasattr(config, 'model_dump') or (hasattr(config, '__class__') and 'BaseModel' in [base.__name__ for base in config.__class__.__mro__])):
            logger.error(f"config must be a Pydantic model, got {type(config)}")
            raise TypeError(f"config must be a Pydantic model, got {type(config)}")
        ai_settings = getattr(config, 'ai_settings', None)
        symbol = getattr(bundle, 'target_symbol', None)
        return create_ai_dashboard_layout(bundle, ai_settings, symbol, db_manager=None)
    except Exception as e:
        logger.error(f"Error creating AI dashboard layout: {e}")
        from dash import html
        return html.Div([
            html.H3("AI Dashboard Error", className="text-danger"),
            html.P(f"Failed to load AI dashboard: {str(e)}", className="text-muted")
        ], className="p-4")

# Export the main function
__all__ = ['create_layout']