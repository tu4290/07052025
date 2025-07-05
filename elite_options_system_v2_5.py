"""
Elite Options System v2.5 - Root Package

This file serves as the root package for the EOTS v2.5 system.
It provides the top-level namespace for all modules and packages.
"""

__version__ = "2.5.0"
__author__ = "Your Name"
__description__ = "Elite Options Trading System v2.5"

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize the root logger
logger = logging.getLogger(__name__)
logger.info("EOTS v2.5 root package initialized")
