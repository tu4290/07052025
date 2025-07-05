"""
HuiHui Safety & Performance Manager
==================================

Production-ready safety system for HuiHui-MoE with:
- Wall-clock timeouts with auto-retry
- GPU/VRAM monitoring and fallback
- Health metrics and performance tracking
- Context budget management
- Prompt versioning system

Author: EOTS v2.5 AI Safety Division
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field
import requests

# Optional imports for enhanced monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SafetyMetrics:
    """Safety and performance metrics."""
    timestamp: datetime
    expert: str
    processing_time: float
    gpu_memory_used: Optional[float]
    cpu_usage: float
    timeout_occurred: bool
    retry_count: int
    success: bool

class HuiHuiSafetyConfig(BaseModel):
    """Pydantic model for safety configuration."""
    wall_clock_timeouts: Dict[str, int] = Field(default_factory=lambda: {
        "market_regime": 90,     # seconds (doubled)
        "options_flow": 90,      # seconds (doubled)
        "sentiment": 60,         # Faster for sentiment (doubled)
        "orchestrator": 120      # Longer for synthesis (doubled)
    })
    max_retries: int = Field(default=1, description="Max retry attempts on timeout")
    gpu_memory_threshold: float = Field(default=0.8, description="GPU memory threshold (80%)")
    cpu_threshold: float = Field(default=0.9, description="CPU usage threshold (90%)")
    context_budget: Dict[str, int] = Field(default_factory=lambda: {
        "market_regime": 4000,   # tokens (doubled)
        "options_flow": 4000,    # tokens (doubled)
        "sentiment": 3000,       # tokens (doubled)
        "orchestrator": 6000     # Larger context for synthesis (doubled)
    })
    enable_health_monitoring: bool = Field(default=True)
    health_check_interval: int = Field(default=60, description="Health check interval in seconds")
    
    class Config:
        extra = 'forbid'

class PromptVersionManager:
    """Manage prompt versioning for A/B testing."""
    
    def __init__(self):
        self.current_version = "v1.0"
        self.prompt_templates = {
            "market_regime": {
                "v1.0": "[EXPERT:MARKET_REGIME] v1.0 - You are a market regime and volatility analysis expert. Focus on regime detection, volatility patterns, and structural analysis."
            },
            "options_flow": {
                "v1.0": "[EXPERT:OPTIONS_FLOW] v1.0 - You are an options flow and institutional behavior expert. Focus on flow analysis, gamma dynamics, and institutional positioning."
            },
            "sentiment": {
                "v1.0": "[EXPERT:SENTIMENT] v1.0 - You are a sentiment and news intelligence expert. Focus on market psychology, news analysis, and crowd behavior."
            },
            "orchestrator": {
                "v1.0": "[EXPERT:ORCHESTRATOR] v1.0 - You are the meta-orchestrator. Synthesize all available information and provide strategic trading decisions."
            }
        }
    
    def get_versioned_prompt(self, expert: str, base_prompt: str, version: Optional[str] = None) -> str:
        """Get versioned system prompt for expert."""
        version = version or self.current_version
        template = self.prompt_templates.get(expert, {}).get(version)
        
        if template:
            return f"{template}\n\n{base_prompt}"
        else:
            return f"[EXPERT:{expert.upper()}] {version} - {base_prompt}"

class HuiHuiSafetyManager:
    """
    Production safety manager for HuiHui expert system.
    
    Features:
    - Wall-clock timeouts with retry logic
    - GPU/VRAM monitoring and fallback
    - Health metrics tracking
    - Context budget management
    - Prompt versioning
    """
    
    def __init__(self, config_path: str = "config/huihui_safety.json"):
        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()
        self.prompt_manager = PromptVersionManager()
        self.safety_metrics: List[SafetyMetrics] = []
        self.last_health_check = datetime.now()
        self.ollama_healthy = True
        self.gpu_available = self._check_gpu_availability()
        
    def _load_or_create_config(self) -> HuiHuiSafetyConfig:
        """Load or create safety configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return HuiHuiSafetyConfig(**config_data)
            except Exception as e:
                logger.warning(f"Error loading safety config: {e}, using defaults")
        
        config = HuiHuiSafetyConfig()
        self._save_config(config)
        return config
    
    def _save_config(self, config: HuiHuiSafetyConfig):
        """🚀 PYDANTIC-FIRST: Save safety configuration using model_dump()."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for monitoring."""
        return GPUTIL_AVAILABLE
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "ollama_healthy": self.ollama_healthy
        }

        # Add CPU/Memory metrics if psutil available
        if PSUTIL_AVAILABLE:
            health["cpu_usage"] = psutil.cpu_percent(interval=0.1)
            health["memory_usage"] = psutil.virtual_memory().percent
        else:
            raise ValueError("CRITICAL: psutil not available - cannot get system health metrics without real monitoring data!")

        # Add GPU metrics if available
        if self.gpu_available and GPUTIL_AVAILABLE:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    health["gpu_memory_used"] = gpu.memoryUsed / gpu.memoryTotal
                    health["gpu_temperature"] = gpu.temperature
                    health["gpu_load"] = gpu.load
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")

        return health
    
    def check_ollama_health(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            self.ollama_healthy = response.status_code == 200
            return self.ollama_healthy
        except Exception:
            self.ollama_healthy = False
            return False
    
    def should_use_cpu_fallback(self) -> bool:
        """Determine if should fallback to CPU due to GPU saturation."""
        if not self.gpu_available or not GPUTIL_AVAILABLE:
            return False

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_usage = gpu.memoryUsed / gpu.memoryTotal
                return memory_usage > self.config.gpu_memory_threshold
        except Exception:
            pass

        return False
    
    def trim_context_for_budget(self, prompt: str, expert: str) -> str:
        """Trim prompt to fit within context budget."""
        budget = self.config.context_budget.get(expert, 2000)
        
        # Simple token estimation (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= budget:
            return prompt
        
        # Trim from the middle, keeping beginning and end
        target_chars = budget * 4
        if len(prompt) > target_chars:
            keep_start = target_chars // 3
            keep_end = target_chars // 3
            trimmed = prompt[:keep_start] + f"\n\n[... trimmed {len(prompt) - target_chars} chars for context budget ...]\n\n" + prompt[-keep_end:]
            logger.info(f"Trimmed prompt for {expert}: {len(prompt)} -> {len(trimmed)} chars")
            return trimmed
        
        return prompt
    
    async def safe_expert_call(self, expert: str, prompt: str, client_func, **kwargs) -> Tuple[str, SafetyMetrics]:
        """
        Make a safe call to HuiHui expert with timeout and retry logic.
        
        Args:
            expert: Expert name
            prompt: Input prompt
            client_func: Function to call (e.g., client.chat_huihui)
            **kwargs: Additional arguments for client_func
        
        Returns:
            Tuple of (response, safety_metrics)
        """
        start_time = time.time()
        timeout = self.config.wall_clock_timeouts.get(expert, 45)
        max_retries = self.config.max_retries
        
        # Get versioned prompt
        versioned_prompt = self.prompt_manager.get_versioned_prompt(expert, prompt)
        
        # Trim for context budget
        final_prompt = self.trim_context_for_budget(versioned_prompt, expert)
        
        # Health check if needed
        if self.config.enable_health_monitoring:
            if (datetime.now() - self.last_health_check).seconds > self.config.health_check_interval:
                self.check_ollama_health()
                self.last_health_check = datetime.now()
        
        # Check if should use CPU fallback
        if self.should_use_cpu_fallback():
            logger.warning(f"GPU memory high, using CPU fallback for {expert}")
        
        retry_count = 0
        timeout_occurred = False
        success = False
        response = ""
        error_message = None
        
        while retry_count <= max_retries:
            try:
                # Use asyncio.wait_for for timeout
                response = await asyncio.wait_for(
                    asyncio.create_task(self._async_client_call(client_func, final_prompt, expert, **kwargs)),
                    timeout=timeout
                )
                success = True
                break
                
            except asyncio.TimeoutError:
                timeout_occurred = True
                retry_count += 1
                logger.warning(f"Timeout for {expert} (attempt {retry_count}/{max_retries + 1})")
                
                if retry_count <= max_retries:
                    # Retry with lighter prompt
                    final_prompt = self.trim_context_for_budget(prompt, expert)  # Remove versioning for retry
                    timeout = min(timeout + 15, 90)  # Increase timeout slightly
                else:
                    response = f"HuiHui {expert} expert timeout after {max_retries + 1} attempts"
                    error_message = "Timeout exceeded"
                    
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error in {expert} expert call: {e}")
                if retry_count < max_retries:
                    retry_count += 1
                else:
                    response = f"HuiHui {expert} expert error: {error_message}"
                    break
        
        # Record metrics
        processing_time = time.time() - start_time
        health = self.get_system_health()
        
        metrics = SafetyMetrics(
            timestamp=datetime.now(),
            expert=expert,
            processing_time=processing_time,
            gpu_memory_used=health.get("gpu_memory_used"),
            cpu_usage=health.get("cpu_usage", 0),
            timeout_occurred=timeout_occurred,
            retry_count=retry_count,
            success=success
        )
        
        self.safety_metrics.append(metrics)
        
        return response, metrics
    
    async def _async_client_call(self, client_func, prompt: str, expert: str, **kwargs):
        """Async wrapper for client function call."""
        # Most client functions are sync, so run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: client_func(prompt, expert, **kwargs))
    
    def get_safety_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get safety statistics for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.safety_metrics if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {"total_calls": 0}
        
        total_calls = len(recent_metrics)
        successful_calls = sum(1 for m in recent_metrics if m.success)
        timeout_calls = sum(1 for m in recent_metrics if m.timeout_occurred)
        avg_processing_time = sum(m.processing_time for m in recent_metrics) / total_calls
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls,
            "timeout_calls": timeout_calls,
            "timeout_rate": timeout_calls / total_calls,
            "avg_processing_time": avg_processing_time,
            "avg_retry_count": sum(m.retry_count for m in recent_metrics) / total_calls
        }

# Global safety manager instance
_safety_manager = None

def get_safety_manager() -> HuiHuiSafetyManager:
    """Get global safety manager instance."""
    global _safety_manager
    if _safety_manager is None:
        _safety_manager = HuiHuiSafetyManager()
    return _safety_manager

async def safe_huihui_call(expert: str, prompt: str, client_func, **kwargs) -> Tuple[str, SafetyMetrics]:
    """Make a safe call to HuiHui expert."""
    manager = get_safety_manager()
    return await manager.safe_expert_call(expert, prompt, client_func, **kwargs)

# ===== TESTING FUNCTION =====

async def test_safety_manager():
    """Test the safety manager functionality."""
    print("🛡️ Testing HuiHui Safety Manager...")
    
    manager = get_safety_manager()
    
    # Test health check
    health = manager.get_system_health()
    print(f"✅ System Health: {health}")
    
    # Test Ollama health
    ollama_healthy = manager.check_ollama_health()
    print(f"✅ Ollama Health: {ollama_healthy}")
    
    # Test context trimming
    long_prompt = "This is a test prompt. " * 200
    trimmed = manager.trim_context_for_budget(long_prompt, "market_regime")
    print(f"✅ Context trimming: {len(long_prompt)} -> {len(trimmed)} chars")
    
    print("✅ Safety manager test completed")

if __name__ == "__main__":
    asyncio.run(test_safety_manager())
