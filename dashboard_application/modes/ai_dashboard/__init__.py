"""
AI Dashboard Module for EOTS v2.5
=================================

This module provides a modular, maintainable AI dashboard system with Pydantic-first architecture.

Modules:
- components: Core UI components and styling
- visualizations: Charts and graph creation
- intelligence: AI analysis and insights
- layouts: Panel assembly and layout management
- utils: Utility functions and helpers
- ai_dashboard_display_v2_5: Main entry point

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

from .ai_dashboard_display_v2_5 import create_layout

__all__ = ['create_layout']
