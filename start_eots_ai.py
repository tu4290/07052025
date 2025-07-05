#!/usr/bin/env python3
"""
EOTS v2.5 AI System Startup Script
Launches the complete AI-powered Elite Options Trading System
"""

import os
import sys
import logging

def check_environment():
    """Check if all required environment variables and dependencies are set."""
    print("🔍 Checking environment...")

    # Check HuiHui Local LLM availability (primary AI system)
    huihui_key = os.getenv('HUIHUI_MOE_API_KEY')

    if not huihui_key:
        print("ℹ️  HUIHUI_MOE_API_KEY not found - using default local key")
        print("   HuiHui will use default local authentication")
    else:
        print(f"✅ HuiHui API key found: {huihui_key[:8]}...")

    # Note: EOTS v2.5 uses HuiHui exclusively - no external AI providers needed
    
    # Check database credentials
    db_host = os.getenv('DB_HOST')
    if not db_host:
        print("⚠️  DB_HOST not found - using default database settings")
    else:
        print(f"✅ Database host found: {db_host}")
    
    # Check if AI dependencies are installed
    try:
        import pydantic_ai
        print("✅ Pydantic AI installed")
    except ImportError:
        print("❌ Pydantic AI not installed")
        print("   Please run: pip install pydantic-ai==0.0.13")
        return False

    # Check HuiHui Local LLM availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ HuiHui Local LLM server available")
        else:
            print("⚠️  HuiHui Local LLM server not responding properly")
    except Exception:
        print("⚠️  HuiHui Local LLM server not available - please start Ollama")
        print("   Run: ollama serve")
        print("   Then: ollama pull huihui_ai/huihui-moe-abliterated:5b-a1.7b")
    
    return True

def setup_logging():
    """Set up logging for the startup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main startup function."""
    print("🚀 EOTS v2.5 AI System Startup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    print("\n✅ Environment check passed!")
    print("\n🧠 Starting AI-powered EOTS system...")
    
    # Setup logging
    setup_logging()
    
    try:
        # Import and start the main application
        print("📦 Importing EOTS modules...")
        from dashboard_application.app_main import main as app_main
        
        print("🎯 Launching dashboard application...")
        print("\n" + "=" * 50)
        print("🌟 EOTS v2.5 AI System is starting!")
        print("🧠 Unified AI Orchestrator: ENABLED")
        print("🗄️  Multi-Database Manager: ENABLED") 
        print("🎯 ATIF AI Insights: ENABLED")
        print("📊 Dashboard: http://localhost:8050")
        print("=" * 50)
        
        # Start the application
        app_main()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        print("👋 EOTS v2.5 AI System stopped")
    except Exception as e:
        print(f"\n❌ Error starting EOTS system: {e}")
        logging.exception("Startup error")
        sys.exit(1)

if __name__ == "__main__":
    main()
