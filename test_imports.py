import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    import dotenv
    print("✓ dotenv imported successfully")
except ImportError as e:
    print(f"✗ Failed to import dotenv: {e}")

try:
    import pydantic
    print("✓ pydantic imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pydantic: {e}")

try:
    import dash
    print("✓ dash imported successfully")
except ImportError as e:
    print(f"✗ Failed to import dash: {e}")

try:
    import pandas
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

try:
    import numpy
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")

# Now try to run the actual script
print("\nAttempting to run the dashboard script...")
try:
    # Import and run the main script
    import run_system_dashboard_v2_5
except Exception as e:
    print(f"Error running dashboard: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()