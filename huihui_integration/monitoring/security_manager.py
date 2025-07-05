"""
HuiHui Security & API Key Management
===================================

Production-ready security management for HuiHui-MoE expert system with:
- Unique API tokens per expert for audit trails
- Usage logging and monitoring
- Rate limiting and safety controls
- Performance tracking per expert

Author: EOTS v2.5 AI Security Division
"""

import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class ExpertUsageRecord:
    """Record of expert usage for audit and optimization."""
    expert_id: str
    api_token: str
    timestamp: datetime
    prompt_length: int
    response_length: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class HuiHuiSecurityConfig(BaseModel):
    """Pydantic model for HuiHui security configuration."""
    expert_tokens: Dict[str, str] = Field(default_factory=dict)
    rate_limits: Dict[str, int] = Field(default_factory=lambda: {
        "market_regime": 200,    # requests per hour (doubled)
        "options_flow": 200,     # requests per hour (doubled)
        "sentiment": 300,        # Higher for news analysis (doubled)
        "orchestrator": 100      # Lower for complex analysis (doubled)
    })
    safety_timeouts: Dict[str, int] = Field(default_factory=lambda: {
        "market_regime": 45,     # seconds
        "options_flow": 45,
        "sentiment": 30,         # Faster for sentiment
        "orchestrator": 60       # Longer for complex synthesis
    })
    enable_usage_logging: bool = Field(default=True)
    log_file_path: str = Field(default="logs/huihui_usage.jsonl")
    
    class Config:
        extra = 'forbid'

class HuiHuiSecurityManager:
    """
    Production security manager for HuiHui expert system.
    
    Features:
    - Unique API tokens per expert
    - Usage tracking and audit trails
    - Rate limiting and safety timeouts
    - Performance monitoring
    """
    
    def __init__(self, config_path: str = "config/huihui_security.json"):
        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()
        self.usage_records: List[ExpertUsageRecord] = []
        self.rate_limit_tracker: Dict[str, List[datetime]] = {
            expert: [] for expert in ["market_regime", "options_flow", "sentiment", "orchestrator"]
        }
        
        # Ensure log directory exists
        log_path = Path(self.config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_or_create_config(self) -> HuiHuiSecurityConfig:
        """Load existing config or create new one with unique tokens."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return HuiHuiSecurityConfig(**config_data)
            except Exception as e:
                logger.warning(f"Error loading security config: {e}, creating new one")
        
        # Create new config with unique tokens
        config = HuiHuiSecurityConfig()
        config.expert_tokens = {
            "market_regime": f"huihui-regime-{uuid.uuid4().hex[:16]}",
            "options_flow": f"huihui-flow-{uuid.uuid4().hex[:16]}",
            "sentiment": f"huihui-sentiment-{uuid.uuid4().hex[:16]}",
            "orchestrator": f"huihui-orchestrator-{uuid.uuid4().hex[:16]}"
        }
        
        self._save_config(config)
        return config
    
    def _save_config(self, config: HuiHuiSecurityConfig):
        """🚀 PYDANTIC-FIRST: Save security configuration using model_dump()."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)
        logger.info(f"✅ PYDANTIC-FIRST: Security config saved to {self.config_path}")
    
    def get_expert_token(self, expert: str) -> str:
        """Get unique API token for specific expert."""
        if expert not in self.config.expert_tokens:
            raise ValueError(f"Unknown expert: {expert}")
        return self.config.expert_tokens[expert]
    
    def check_rate_limit(self, expert: str) -> bool:
        """Check if expert is within rate limits."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old records
        self.rate_limit_tracker[expert] = [
            ts for ts in self.rate_limit_tracker[expert] if ts > hour_ago
        ]
        
        # Check limit
        current_count = len(self.rate_limit_tracker[expert])
        limit = self.config.rate_limits.get(expert, 100)
        
        if current_count >= limit:
            logger.warning(f"Rate limit exceeded for {expert}: {current_count}/{limit}")
            return False
        
        return True
    
    def record_usage(self, expert: str, prompt_length: int, response_length: int, 
                    processing_time: float, success: bool, error_message: Optional[str] = None):
        """Record expert usage for audit and optimization."""
        # Add to rate limit tracker
        self.rate_limit_tracker[expert].append(datetime.now())
        
        # Create usage record
        record = ExpertUsageRecord(
            expert_id=expert,
            api_token=self.get_expert_token(expert),
            timestamp=datetime.now(),
            prompt_length=prompt_length,
            response_length=response_length,
            processing_time=processing_time,
            success=success,
            error_message=error_message
        )
        
        self.usage_records.append(record)
        
        # Log to file if enabled
        if self.config.enable_usage_logging:
            self._log_usage_record(record)
    
    def _log_usage_record(self, record: ExpertUsageRecord):
        """Log usage record to JSONL file."""
        try:
            log_data = {
                "timestamp": record.timestamp.isoformat(),
                "expert": record.expert_id,
                "token": record.api_token[-8:],  # Only last 8 chars for security
                "prompt_length": record.prompt_length,
                "response_length": record.response_length,
                "processing_time": record.processing_time,
                "success": record.success,
                "error": record.error_message
            }
            
            with open(self.config.log_file_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging usage record: {e}")
    
    def get_expert_statistics(self, expert: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for specific expert."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        relevant_records = [
            r for r in self.usage_records 
            if r.expert_id == expert and r.timestamp > cutoff
        ]
        
        if not relevant_records:
            return {"total_requests": 0, "success_rate": 0.0, "avg_processing_time": 0.0}
        
        total_requests = len(relevant_records)
        successful_requests = sum(1 for r in relevant_records if r.success)
        success_rate = successful_requests / total_requests
        avg_processing_time = sum(r.processing_time for r in relevant_records) / total_requests
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "avg_prompt_length": sum(r.prompt_length for r in relevant_records) / total_requests,
            "avg_response_length": sum(r.response_length for r in relevant_records) / total_requests
        }
    
    def get_all_statistics(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all experts."""
        return {
            expert: self.get_expert_statistics(expert, hours)
            for expert in ["market_regime", "options_flow", "sentiment", "orchestrator"]
        }
    
    def regenerate_expert_token(self, expert: str) -> str:
        """Regenerate API token for specific expert."""
        new_token = f"huihui-{expert}-{uuid.uuid4().hex[:16]}"
        self.config.expert_tokens[expert] = new_token
        self._save_config(self.config)
        logger.info(f"Regenerated token for {expert}")
        return new_token
    
    def get_safety_timeout(self, expert: str) -> int:
        """Get safety timeout for specific expert."""
        return self.config.safety_timeouts.get(expert, 45)

# Global security manager instance
_security_manager = None

def get_security_manager() -> HuiHuiSecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = HuiHuiSecurityManager()
    return _security_manager

def validate_expert_access(expert: str) -> bool:
    """Validate that expert access is allowed."""
    manager = get_security_manager()
    return manager.check_rate_limit(expert)

def get_expert_api_token(expert: str) -> str:
    """Get API token for specific expert."""
    manager = get_security_manager()
    return manager.get_expert_token(expert)

def record_expert_usage(expert: str, prompt_length: int, response_length: int, 
                       processing_time: float, success: bool, error_message: Optional[str] = None):
    """Record expert usage for audit."""
    manager = get_security_manager()
    manager.record_usage(expert, prompt_length, response_length, processing_time, success, error_message)

# ===== TESTING FUNCTION =====

def test_security_manager():
    """Test the security manager functionality."""
    print("🔐 Testing HuiHui Security Manager...")
    
    manager = get_security_manager()
    
    # Test token generation
    for expert in ["market_regime", "options_flow", "sentiment", "orchestrator"]:
        token = manager.get_expert_token(expert)
        print(f"✅ {expert}: {token}")
    
    # Test rate limiting
    print(f"\n🚦 Rate limit check: {manager.check_rate_limit('market_regime')}")
    
    # Test usage recording
    manager.record_usage("market_regime", 100, 500, 2.5, True)
    stats = manager.get_expert_statistics("market_regime")
    print(f"\n📊 Statistics: {stats}")
    
    print("✅ Security manager test completed")

if __name__ == "__main__":
    test_security_manager()
