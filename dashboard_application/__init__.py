"""Dashboard Application Package for EOTS v2.5"""

# Import main dashboard modules
from .app_main import main
from .callback_manager_v2_5 import register_v2_5_callbacks
from .layout_manager_v2_5 import create_master_layout
from .utils_dashboard_v2_5 import initialize_dashboard_utils
from . import ids

__all__ = [
    'main',
    'register_v2_5_callbacks',
    'create_master_layout', 
    'initialize_dashboard_utils',
    'ids'
]