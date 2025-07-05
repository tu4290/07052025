# dashboard_application/modes/ai_dashboard/deprecated_legacy/__init__.py
"""
DEPRECATION GUARD FILE
======================

This file prevents any modules within the 'deprecated_legacy' directory from being
imported into the EOTS v2.5 system.

These modules have been superseded by a Pydantic-first, modular, and fully
HuiHui-integrated component set. Attempting to import from this directory
will raise an immediate ImportError.

See 'LEGACY_DEPRECATION_PLAN.md' for full details.
"""

import logging

# Define the detailed error message to be logged and raised.
DEPRECATION_ERROR_MESSAGE = """
================================================================================
CRITICAL IMPORT ERROR: Attempted to import from a deprecated legacy directory.
================================================================================

The modules in 'dashboard_application/modes/ai_dashboard/deprecated_legacy/'
have been **ARCHIVED AND DEPRECATED**.

Reason for Deprecation:
-----------------------
These files represent a legacy, monolithic architecture. They are static,
use dictionary-based data structures, and are NOT integrated with the live
HuiHui Mixture-of-Experts (MoE) AI system. They violate the system-wide
"Pydantic-First" and "ZERO DICT ACCEPTANCE" policies.

Action Required:
----------------
Refactor your code to use the new, modular, Pydantic-first components that
are fully integrated with the HuiHui AI system.

Key Replacements:
- For the main layout, use: `dashboard_application.modes.ai_dashboard.enhanced_ai_hub_layout`
- For the compass, use: `dashboard_application.modes.ai_dashboard.market_compass_component`
- For recommendations, use: `dashboard_application.modes.ai_dashboard.ai_recommendations_component`
- For flow metrics, use: `dashboard_application.modes.ai_dashboard.flow_intelligence_panel`
- For volatility, use: `dashboard_application.modes.ai_dashboard.volatility_gamma_panel`

For a complete migration guide, please refer to the document:
'dashboard_application/modes/ai_dashboard/LEGACY_DEPRECATION_PLAN.md'

================================================================================
"""

# Log the critical error to the system logs for monitoring and debugging.
logger = logging.getLogger(__name__)
logger.critical(DEPRECATION_ERROR_MESSAGE)

# Raise the ImportError to halt execution and enforce the deprecation.
raise ImportError(DEPRECATION_ERROR_MESSAGE)
