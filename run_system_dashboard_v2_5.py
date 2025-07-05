#!/usr/bin/env python
# elite_options_system_v2_5/run_system_dashboard_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL ENTRY POINT

"""
Elite Options Trading System Dashboard Runner (V2.5 - "Apex Predator")

This is the primary and definitive launcher for the EOTS v2.5 Dashboard.
Its sole responsibilities are:
1. Loading environment variables from the .env file.
2. Configuring the system-wide logger for standard output.
3. Ensuring the Python environment is correctly configured for module resolution
   by adding the project root to the system path.
4. Delegating execution to the main application function.

This script is intentionally lean and does not import or configure specific
application components. It only prepares the environment for the application
core to run successfully.
"""

import os
import sys
import logging
import tempfile
from dotenv import load_dotenv
from pydantic import ValidationError
import subprocess

# Process lock to prevent multiple instances
LOCK_FILE = os.path.join(tempfile.gettempdir(), 'eots_v2_5.lock')

def acquire_lock():
    """Acquire a file lock to prevent multiple instances."""
    try:
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(lock_fd, str(os.getpid()).encode())
        return lock_fd
    except OSError:
        # Lock file exists, try to remove it and retry once
        try:
            os.remove(LOCK_FILE)
            lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(lock_fd, str(os.getpid()).encode())
            return lock_fd
        except OSError:
            print(f"EOTS v2.5 may already be running. If not, delete: {LOCK_FILE}")
            return None

def release_lock(lock_fd):
    """Release the file lock."""
    if lock_fd is not None:
        try:
            os.close(lock_fd)
            os.remove(LOCK_FILE)
        except OSError:
            pass

# Configure root logger first so we can see the environment loading
import sys
import codecs

# Configure root logger first so we can see the environment loading
# Use utf-8 encoding for stdout to handle emojis and special characters
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure package root is treated as a package
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

# Log the .env file path we're trying to load
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
logger.info(f"Attempting to load .env file from: {env_path}")

# Load environment variables from .env file
if not load_dotenv(env_path):
    logger.warning("No .env file found. Please create a .env file with the required environment variables.")
    logger.warning("Required environment variables:")
    logger.warning("- CONVEX_EMAIL")
    logger.warning("- CONVEX_PASSWORD")
    logger.warning("- TRADIER_PRODUCTION_TOKEN")
    logger.warning("- DB_HOST")
    logger.warning("- DB_PORT")
    logger.warning("- DB_NAME")
    logger.warning("- DB_USER")
    logger.warning("- DB_PASSWORD")

# Debug: Print environment variables
logger.info("Environment variables after loading .env:")
logger.info(f"CONVEX_EMAIL exists: {'CONVEX_EMAIL' in os.environ}")
logger.info(f"CONVEX_PASSWORD exists: {'CONVEX_PASSWORD' in os.environ}")
logger.info(f"TRADIER_PRODUCTION_TOKEN exists: {'TRADIER_PRODUCTION_TOKEN' in os.environ}")

# Log path integrity check
if package_root in sys.path:
    logger.debug("Path Integrity Check: Package root is already in sys.path.")
else:
    logger.debug("Path Integrity Check: Package root is not in sys.path.")

# Import root package to ensure all modules are properly initialized
try:
    import elite_options_system_v2_5
    logger.info("EOTS v2.5 root package initialized")
except ImportError as e:
    logger.error(f"Failed to import root package: {e}")
    sys.exit(1)

logger.info("Successfully imported root package")

# Import application core
from dashboard_application.app_main import main

# Monitoring utilities disabled (Docker not required)

if __name__ == "__main__":
    # Acquire process lock to prevent multiple instances
    lock_fd = acquire_lock()
    if lock_fd is None:
        print("Lock acquisition failed, but continuing anyway...")
        lock_fd = None  # Continue without lock for now
    
    # --- Start Monitoring Stack ---
    # DISABLED: Docker-based monitoring requires Docker installation
    # The system works perfectly without monitoring
    logger.info("Monitoring stack disabled (Docker not required)")
    
    # --- Start Intraday Collector as Background Process ---
    try:
        collector_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_intraday_collector.py')
        logger.info(f"Launching Intraday Collector: {collector_script}")
        subprocess.Popen([sys.executable, collector_script])
        logger.info("Intraday Collector started in background.")
    except Exception as e:
        logger.error(f"Failed to launch Intraday Collector: {e}")
    
    logger.info("EOTS v2.5 'Apex Predator' Initialization Sequence Activated.")
    logger.info("Handoff to Application Core...")
    try:
        main()
    except ValidationError as e:
        logger.critical(f"Configuration validation failed: {e}")
        logger.critical("Please check your config_v2_5.json file for errors.")
        release_lock(lock_fd)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down gracefully...")
        release_lock(lock_fd)
        sys.exit(0)
    except Exception:
        logger.critical("CATASTROPHIC FAILURE: An unhandled exception occurred during application startup.", exc_info=True)
        release_lock(lock_fd)
        sys.exit(1)
    finally:
        logger.info("Execution complete. EOTS shutting down.")
        release_lock(lock_fd)
