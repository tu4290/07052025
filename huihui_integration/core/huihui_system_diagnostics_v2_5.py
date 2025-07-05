"""
HuiHui System Diagnostics & Recovery v2.5
========================================

Comprehensive diagnostic and repair system for HuiHui AI expert failures.
Addresses the 100% failure rate identified in logs/huihui_usage.jsonl.

Key Issues Identified:
- All 348 log entries show "success": false
- Zero successful expert requests since June 22nd
- Processing times near 0ms suggest immediate failures
- No error details captured despite all failures

Author: EOTS v2.5 Recovery Team
"""

import os
import json
import logging
import time
import requests
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import Pydantic for validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

class DiagnosticLevel(Enum):
    """Diagnostic severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"

@dataclass
class DiagnosticResult:
    """Single diagnostic test result."""
    test_name: str
    level: DiagnosticLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    fix_available: bool = False
    fix_description: str = ""

class HuiHuiSystemDiagnostics:
    """
    Comprehensive HuiHui system diagnostics and recovery.
    
    Performs deep analysis of:
    - Ollama server connectivity
    - Model availability and status
    - Expert routing functionality
    - Log analysis for failure patterns
    - Configuration validation
    - Performance benchmarking
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        self.results: List[DiagnosticResult] = []
        
        # Expected models and configurations
        self.expected_models = {
            "huihui_ai/huihui-moe-abliterated:5b-a1.7b": "HuiHui MoE Expert System",
            "deepseek-coder-v2:16b": "DeepSeek V2 Coding Assistant"
        }
        
        # Expert tokens from logs
        self.expert_tokens = {
            "market_regime": "4f714d2a",
            "options_flow": "f9d747c2", 
            "sentiment": "6d90476e",
            "orchestrator": "5081436b"
        }
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite."""
        self.logger.info("🔍 Starting HuiHui System Diagnostics v2.5...")
        self.results.clear()
        
        # Test suite execution
        tests = [
            self._test_ollama_connectivity,
            self._test_model_availability,
            self._test_model_functionality,
            self._analyze_failure_logs,
            self._test_expert_routing,
            self._validate_configurations,
            self._benchmark_performance,
            self._test_error_handling
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.results.append(DiagnosticResult(
                    test_name=test.__name__,
                    level=DiagnosticLevel.CRITICAL,
                    message=f"Diagnostic test failed: {str(e)}",
                    details={"exception": str(e)},
                    timestamp=datetime.now()
                ))
        
        return self._generate_diagnostic_report()
    
    def _test_ollama_connectivity(self):
        """Test basic Ollama server connectivity."""
        test_name = "Ollama Server Connectivity"
        
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            
            if response.status_code == 200:
                self.results.append(DiagnosticResult(
                    test_name=test_name,
                    level=DiagnosticLevel.SUCCESS,
                    message="✅ Ollama server is running and responsive",
                    details={"status_code": response.status_code, "response_time": response.elapsed.total_seconds()},
                    timestamp=datetime.now()
                ))
            else:
                self.results.append(DiagnosticResult(
                    test_name=test_name,
                    level=DiagnosticLevel.ERROR,
                    message=f"❌ Ollama server returned status {response.status_code}",
                    details={"status_code": response.status_code},
                    timestamp=datetime.now(),
                    fix_available=True,
                    fix_description="Check Ollama server configuration and restart if needed"
                ))
                
        except requests.ConnectionError:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.CRITICAL,
                message="🚨 Cannot connect to Ollama server",
                details={"host": self.ollama_host},
                timestamp=datetime.now(),
                fix_available=True,
                fix_description="Start Ollama server with 'ollama serve' command"
            ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.ERROR,
                message=f"❌ Connection test failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            ))
    
    def _test_model_availability(self):
        """Test if required models are available."""
        test_name = "Model Availability"
        
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code != 200:
                return
            
            available_models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in available_models]
            
            for expected_model, description in self.expected_models.items():
                if expected_model in model_names:
                    self.results.append(DiagnosticResult(
                        test_name=f"{test_name} - {description}",
                        level=DiagnosticLevel.SUCCESS,
                        message=f"✅ {description} is available",
                        details={"model": expected_model},
                        timestamp=datetime.now()
                    ))
                else:
                    self.results.append(DiagnosticResult(
                        test_name=f"{test_name} - {description}",
                        level=DiagnosticLevel.CRITICAL,
                        message=f"🚨 {description} is missing",
                        details={"expected_model": expected_model, "available_models": model_names},
                        timestamp=datetime.now(),
                        fix_available=True,
                        fix_description=f"Install model with: ollama pull {expected_model}"
                    ))
                    
        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.ERROR,
                message=f"❌ Model availability check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            ))
    
    def _test_model_functionality(self):
        """Test basic model functionality."""
        test_name = "Model Functionality"
        
        # Test HuiHui model with simple prompt
        test_prompt = "Hello, this is a connectivity test."
        
        try:
            data = {
                "model": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                    "num_ctx": 512
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                response_time = round(end_time - start_time, 2)
                
                if content.strip():
                    self.results.append(DiagnosticResult(
                        test_name=test_name,
                        level=DiagnosticLevel.SUCCESS,
                        message=f"✅ HuiHui model is functional ({response_time}s)",
                        details={"response_time": response_time, "response_length": len(content)},
                        timestamp=datetime.now()
                    ))
                else:
                    self.results.append(DiagnosticResult(
                        test_name=test_name,
                        level=DiagnosticLevel.WARNING,
                        message="⚠️ HuiHui model responded but with empty content",
                        details={"response_time": response_time},
                        timestamp=datetime.now(),
                        fix_available=True,
                        fix_description="Check model configuration and context settings"
                    ))
            else:
                self.results.append(DiagnosticResult(
                    test_name=test_name,
                    level=DiagnosticLevel.ERROR,
                    message=f"❌ HuiHui model request failed: {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text},
                    timestamp=datetime.now()
                ))
                
        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.ERROR,
                message=f"❌ Model functionality test failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            ))
    
    def _analyze_failure_logs(self):
        """Analyze failure patterns in logs/huihui_usage.jsonl."""
        test_name = "Log Analysis"
        
        log_file = Path("logs/huihui_usage.jsonl")
        
        if not log_file.exists():
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.WARNING,
                message="⚠️ HuiHui usage log file not found",
                details={"expected_path": str(log_file)},
                timestamp=datetime.now()
            ))
            return
        
        try:
            failures = 0
            successes = 0
            errors = {}
            experts = {}
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        if entry.get("success", False):
                            successes += 1
                        else:
                            failures += 1
                            
                        # Track expert usage
                        expert = entry.get("expert", "unknown")
                        experts[expert] = experts.get(expert, 0) + 1
                        
                        # Track error types
                        error = entry.get("error")
                        if error:
                            errors[error] = errors.get(error, 0) + 1
                            
                    except json.JSONDecodeError:
                        continue
            
            total_requests = failures + successes
            failure_rate = (failures / total_requests * 100) if total_requests > 0 else 0
            
            if failure_rate > 90:
                level = DiagnosticLevel.CRITICAL
                message = f"🚨 CRITICAL: {failure_rate:.1f}% failure rate ({failures}/{total_requests})"
            elif failure_rate > 50:
                level = DiagnosticLevel.ERROR
                message = f"❌ HIGH: {failure_rate:.1f}% failure rate ({failures}/{total_requests})"
            elif failure_rate > 10:
                level = DiagnosticLevel.WARNING
                message = f"⚠️ MODERATE: {failure_rate:.1f}% failure rate ({failures}/{total_requests})"
            else:
                level = DiagnosticLevel.SUCCESS
                message = f"✅ LOW: {failure_rate:.1f}% failure rate ({failures}/{total_requests})"
            
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=level,
                message=message,
                details={
                    "total_requests": total_requests,
                    "failures": failures,
                    "successes": successes,
                    "failure_rate": failure_rate,
                    "expert_usage": experts,
                    "error_types": errors
                },
                timestamp=datetime.now(),
                fix_available=failure_rate > 50,
                fix_description="Implement error handling and connection retry logic"
            ))
            
        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.ERROR,
                message=f"❌ Log analysis failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            ))
    
    def _test_expert_routing(self):
        """Test expert routing functionality."""
        test_name = "Expert Routing"
        
        # Test each expert type
        for expert_type, token in self.expert_tokens.items():
            try:
                # Simulate expert request
                test_prompt = f"Test {expert_type} expert functionality"
                
                # This would normally go through the routing system
                # For now, we'll test direct model access
                data = {
                    "model": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                    "messages": [
                        {"role": "system", "content": f"You are the {expert_type} expert."},
                        {"role": "user", "content": test_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 100,
                        "num_ctx": 512
                    }
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json=data,
                    timeout=20
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("message", {}).get("content", "")
                    response_time = round(end_time - start_time, 2)
                    
                    if content.strip():
                        self.results.append(DiagnosticResult(
                            test_name=f"{test_name} - {expert_type}",
                            level=DiagnosticLevel.SUCCESS,
                            message=f"✅ {expert_type} expert functional ({response_time}s)",
                            details={"expert": expert_type, "token": token, "response_time": response_time},
                            timestamp=datetime.now()
                        ))
                    else:
                        self.results.append(DiagnosticResult(
                            test_name=f"{test_name} - {expert_type}",
                            level=DiagnosticLevel.WARNING,
                            message=f"⚠️ {expert_type} expert returned empty response",
                            details={"expert": expert_type, "token": token, "response_time": response_time},
                            timestamp=datetime.now()
                        ))
                else:
                    self.results.append(DiagnosticResult(
                        test_name=f"{test_name} - {expert_type}",
                        level=DiagnosticLevel.ERROR,
                        message=f"❌ {expert_type} expert failed: {response.status_code}",
                        details={"expert": expert_type, "token": token, "status_code": response.status_code},
                        timestamp=datetime.now()
                    ))
                    
            except Exception as e:
                self.results.append(DiagnosticResult(
                    test_name=f"{test_name} - {expert_type}",
                    level=DiagnosticLevel.ERROR,
                    message=f"❌ {expert_type} expert test failed: {str(e)}",
                    details={"expert": expert_type, "token": token, "error": str(e)},
                    timestamp=datetime.now()
                ))
    
    def _validate_configurations(self):
        """Validate system configurations."""
        test_name = "Configuration Validation"
        
        # Check environment variables
        env_vars = ["HUIHUI_API_KEY", "OLLAMA_HOST"]
        missing_vars = []
        
        for var in env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.WARNING,
                message=f"⚠️ Missing environment variables: {', '.join(missing_vars)}",
                details={"missing_vars": missing_vars},
                timestamp=datetime.now(),
                fix_available=True,
                fix_description="Set missing environment variables in .env file"
            ))
        else:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.SUCCESS,
                message="✅ All required environment variables are set",
                details={"checked_vars": env_vars},
                timestamp=datetime.now()
            ))
    
    def _benchmark_performance(self):
        """Benchmark system performance."""
        test_name = "Performance Benchmark"
        
        # Test response times with different prompt sizes
        test_cases = [
            ("short", "Hello"),
            ("medium", "Analyze the current market regime for SPY options trading."),
            ("long", "Provide a comprehensive analysis of the current market regime including volatility patterns, options flow dynamics, sentiment indicators, and strategic recommendations for SPY options trading over the next 5-10 trading days.")
        ]
        
        for case_name, prompt in test_cases:
            try:
                data = {
                    "model": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                        "num_ctx": 1024
                    }
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json=data,
                    timeout=30
                )
                end_time = time.time()
                
                response_time = round(end_time - start_time, 2)
                
                if response.status_code == 200:
                    if response_time < 5:
                        level = DiagnosticLevel.SUCCESS
                        message = f"🚀 {case_name} prompt: EXCELLENT ({response_time}s)"
                    elif response_time < 10:
                        level = DiagnosticLevel.SUCCESS
                        message = f"⚡ {case_name} prompt: GOOD ({response_time}s)"
                    elif response_time < 20:
                        level = DiagnosticLevel.WARNING
                        message = f"⏱️ {case_name} prompt: ACCEPTABLE ({response_time}s)"
                    else:
                        level = DiagnosticLevel.WARNING
                        message = f"🐌 {case_name} prompt: SLOW ({response_time}s)"
                    
                    self.results.append(DiagnosticResult(
                        test_name=f"{test_name} - {case_name}",
                        level=level,
                        message=message,
                        details={"prompt_type": case_name, "response_time": response_time, "prompt_length": len(prompt)},
                        timestamp=datetime.now()
                    ))
                    
            except Exception as e:
                self.results.append(DiagnosticResult(
                    test_name=f"{test_name} - {case_name}",
                    level=DiagnosticLevel.ERROR,
                    message=f"❌ {case_name} benchmark failed: {str(e)}",
                    details={"prompt_type": case_name, "error": str(e)},
                    timestamp=datetime.now()
                ))
    
    def _test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        test_name = "Error Handling"
        
        # Test invalid model
        try:
            data = {
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=data,
                timeout=10
            )
            
            if response.status_code != 200:
                self.results.append(DiagnosticResult(
                    test_name=test_name,
                    level=DiagnosticLevel.SUCCESS,
                    message="✅ Error handling works: Invalid model properly rejected",
                    details={"status_code": response.status_code},
                    timestamp=datetime.now()
                ))
            else:
                self.results.append(DiagnosticResult(
                    test_name=test_name,
                    level=DiagnosticLevel.WARNING,
                    message="⚠️ Invalid model request unexpectedly succeeded",
                    details={"status_code": response.status_code},
                    timestamp=datetime.now()
                ))
                
        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name=test_name,
                level=DiagnosticLevel.INFO,
                message=f"ℹ️ Error handling test generated expected exception: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            ))
    
    def _generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        
        # Categorize results
        critical = [r for r in self.results if r.level == DiagnosticLevel.CRITICAL]
        errors = [r for r in self.results if r.level == DiagnosticLevel.ERROR]
        warnings = [r for r in self.results if r.level == DiagnosticLevel.WARNING]
        successes = [r for r in self.results if r.level == DiagnosticLevel.SUCCESS]
        
        # Overall health score
        total_tests = len(self.results)
        if total_tests == 0:
            health_score = 0
        else:
            score = (len(successes) * 100 + len(warnings) * 50 + len(errors) * 10) / (total_tests * 100)
            health_score = round(score * 100, 1)
        
        # System status
        if critical:
            system_status = "CRITICAL"
        elif len(errors) > len(successes):
            system_status = "DEGRADED"
        elif warnings:
            system_status = "WARNING"
        else:
            system_status = "HEALTHY"
        
        # Available fixes
        fixes = [r for r in self.results if r.fix_available]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status,
            "health_score": health_score,
            "summary": {
                "total_tests": total_tests,
                "critical": len(critical),
                "errors": len(errors),
                "warnings": len(warnings),
                "successes": len(successes)
            },
            "available_fixes": len(fixes),
            "results": [
                {
                    "test_name": r.test_name,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                    "fix_available": r.fix_available,
                    "fix_description": r.fix_description
                }
                for r in self.results
            ]
        }
        
        return report
    
    def apply_automatic_fixes(self) -> Dict[str, Any]:
        """Apply automatic fixes for identified issues."""
        fixes_applied = []
        fixes_failed = []
        
        for result in self.results:
            if not result.fix_available:
                continue
                
            try:
                # Apply fix based on the issue type
                if "Ollama server" in result.message and "Cannot connect" in result.message:
                    # Attempt to start Ollama server
                    fix_result = self._fix_ollama_connection()
                    if fix_result:
                        fixes_applied.append(f"✅ {result.test_name}: {result.fix_description}")
                    else:
                        fixes_failed.append(f"❌ {result.test_name}: Failed to start Ollama server")
                
                elif "missing" in result.message.lower() and "model" in result.message.lower():
                    # Attempt to install missing models
                    fix_result = self._fix_missing_models()
                    if fix_result:
                        fixes_applied.append(f"✅ {result.test_name}: {result.fix_description}")
                    else:
                        fixes_failed.append(f"❌ {result.test_name}: Failed to install models")
                
                # Add more fix implementations as needed
                
            except Exception as e:
                fixes_failed.append(f"❌ {result.test_name}: Fix failed - {str(e)}")
        
        return {
            "fixes_applied": fixes_applied,
            "fixes_failed": fixes_failed,
            "total_fixes_attempted": len(fixes_applied) + len(fixes_failed)
        }
    
    def _fix_ollama_connection(self) -> bool:
        """Attempt to fix Ollama connection issues."""
        # This would implement actual fix logic
        # For now, just return False as we can't automatically start services
        return False
    
    def _fix_missing_models(self) -> bool:
        """Attempt to install missing models."""
        # This would implement model installation logic
        # For now, just return False as this requires user action
        return False

# ===== CONVENIENCE FUNCTIONS =====

def run_huihui_diagnostics() -> Dict[str, Any]:
    """Run complete HuiHui system diagnostics."""
    diagnostics = HuiHuiSystemDiagnostics()
    return diagnostics.run_full_diagnostic()

def quick_health_check() -> str:
    """Quick health check with simple status."""
    diagnostics = HuiHuiSystemDiagnostics()
    report = diagnostics.run_full_diagnostic()
    
    status = report["system_status"]
    score = report["health_score"]
    
    if status == "HEALTHY":
        return f"✅ HuiHui System: HEALTHY ({score}% health score)"
    elif status == "WARNING":
        return f"⚠️ HuiHui System: WARNING ({score}% health score)"
    elif status == "DEGRADED":
        return f"❌ HuiHui System: DEGRADED ({score}% health score)"
    else:
        return f"🚨 HuiHui System: CRITICAL ({score}% health score)"

if __name__ == "__main__":
    # Run diagnostics when executed directly
    print("🔍 Running HuiHui System Diagnostics...")
    report = run_huihui_diagnostics()
    
    print("\n📊 DIAGNOSTIC REPORT")
    print(f"System Status: {report['system_status']}")
    print(f"Health Score: {report['health_score']}%")
    print(f"Tests Run: {report['summary']['total_tests']}")
    print(f"Critical Issues: {report['summary']['critical']}")
    print(f"Errors: {report['summary']['errors']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Successes: {report['summary']['successes']}")
    print(f"Available Fixes: {report['available_fixes']}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for result in report['results']:
        print(f"  {result['level'].upper()}: {result['message']}")
        if result['fix_available']:
            print(f"    🔧 Fix: {result['fix_description']}")
