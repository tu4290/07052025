"""
HuiHui Usage Monitoring & Pattern Analysis
==========================================

Comprehensive monitoring system for HuiHui expert usage patterns including:
- Real-time rate limit tracking and analysis
- Token usage patterns (input/output) per expert
- Performance optimization recommendations
- Dynamic threshold adjustment based on actual usage
- Market condition correlation analysis

Author: EOTS v2.5 AI Optimization Division
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

# SUPABASE-ONLY: Import only Supabase database manager
try:
    from huihui_integration.monitoring.supabase_manager import (
        get_supabase_manager,
        HuiHuiUsageRecordV2_5,
        HuiHuiOptimizationRecommendation,
        store_usage_in_supabase
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# SUPABASE-ONLY: No SQLite support
if not SUPABASE_AVAILABLE:
    raise ImportError("❌ CRITICAL: HuiHui monitoring requires Supabase database! No SQLite fallback available.")

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    """Detailed usage record for pattern analysis."""
    timestamp: datetime
    expert: str
    request_type: str  # "analysis", "prediction", "synthesis"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    processing_time: float
    market_condition: str  # "normal", "volatile", "crisis"
    vix_level: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None

class UsagePattern(BaseModel):
    """Pydantic model for usage pattern analysis."""
    expert: str
    time_period: str
    total_requests: int
    avg_requests_per_minute: float
    peak_requests_per_minute: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    avg_processing_time: float
    success_rate: float
    market_conditions: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        extra = 'forbid'

class OptimizationRecommendation(BaseModel):
    """Pydantic model for optimization recommendations."""
    expert: str
    current_rate_limit: int
    recommended_rate_limit: int
    current_token_limit: int
    recommended_token_limit: int
    reasoning: str
    confidence: float
    market_condition_factor: str
    
    class Config:
        extra = 'forbid'

class HuiHuiUsageMonitor:
    """
    Comprehensive usage monitoring and optimization system.
    
    Features:
    - Real-time usage tracking
    - Pattern analysis and optimization
    - Dynamic threshold recommendations
    - Market condition correlation
    """
    
    def __init__(self):
        """SUPABASE-ONLY: Initialize HuiHui usage monitoring with Supabase."""
        self.current_market_condition = "normal"
        self.current_vix = None
        self.supabase_manager = None
        self._supabase_initialized = False
        self._init_supabase_only()

    def _init_supabase_only(self):
        """Initialize ONLY Supabase for usage tracking - no local databases."""
        if not SUPABASE_AVAILABLE:
            logger.error("❌ Supabase not available - HuiHui monitoring cannot function without Supabase!")
            raise RuntimeError("HuiHui monitoring requires Supabase database connection")

        try:
            # Initialize Supabase manager asynchronously when first needed
            self._supabase_initialized = False  # Will be set to True when first used
            logger.info("✅ HuiHui monitoring initialized for Supabase-only storage")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase for HuiHui monitoring: {e}")
            raise
    
    def record_usage(self, expert: str, request_type: str, input_tokens: int,
                    output_tokens: int, processing_time: float, success: bool = True,
                    error_type: Optional[str] = None):
        """SUPABASE-ONLY: Record detailed usage for pattern analysis."""
        record = UsageRecord(
            timestamp=datetime.now(),
            expert=expert,
            request_type=request_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            processing_time=processing_time,
            market_condition=self.current_market_condition,
            vix_level=self.current_vix,
            success=success,
            error_type=error_type
        )

        # SUPABASE-ONLY: Store only in Supabase - no local database
        if not SUPABASE_AVAILABLE:
            logger.error("❌ Cannot store usage record - Supabase not available!")
            raise RuntimeError("HuiHui monitoring requires Supabase database connection")

        asyncio.create_task(self._store_in_supabase(record))
        logger.debug(f"✅ Recorded usage in Supabase: {expert} - {input_tokens}+{output_tokens}={record.total_tokens} tokens")

    async def _store_in_supabase(self, record: UsageRecord):
        """SUPABASE-ONLY: Store usage record in Supabase asynchronously."""
        try:
            if not self._supabase_initialized:
                self.supabase_manager = await get_supabase_manager()
                self._supabase_initialized = True

            # Convert to Supabase record format
            supabase_record = HuiHuiUsageRecord(
                expert_name=record.expert,
                request_type=record.request_type,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                total_tokens=record.total_tokens,
                processing_time_seconds=record.processing_time,
                success=record.success,
                market_condition=record.market_condition,
                vix_level=record.vix_level,
                error_type=record.error_type
            )

            await self.supabase_manager.store_usage_record(supabase_record)
            logger.debug(f"✅ Stored usage record in Supabase for {record.expert}")

        except Exception as e:
            logger.error(f"❌ Failed to store in Supabase: {e}")
            raise  # Re-raise to ensure failures are noticed

    def update_market_condition(self, vix_level: float):
        """Update current market condition based on VIX level."""
        self.current_vix = vix_level
        
        if vix_level < 20:
            self.current_market_condition = "normal"
        elif vix_level < 30:
            self.current_market_condition = "volatile"
        else:
            self.current_market_condition = "crisis"
        
        logger.info(f"Market condition updated: {self.current_market_condition} (VIX: {vix_level})")
    
    async def analyze_usage_patterns(self, expert: str, hours: int = 24) -> UsagePattern:
        """SUPABASE-ONLY: Analyze usage patterns for specific expert."""
        try:
            if not self._supabase_initialized:
                self.supabase_manager = await get_supabase_manager()
                self._supabase_initialized = True

            # Get usage summary from Supabase
            summary = await self.supabase_manager.get_usage_summary(expert, hours)

            if not summary or summary.get("total_requests", 0) == 0:
                return UsagePattern(
                    expert=expert,
                    time_period=f"{hours}h",
                    total_requests=0,
                    avg_requests_per_minute=0.0,
                    peak_requests_per_minute=0.0,
                    avg_input_tokens=0.0,
                    avg_output_tokens=0.0,
                    avg_total_tokens=0.0,
                    max_input_tokens=0,
                    max_output_tokens=0,
                    max_total_tokens=0,
                    avg_processing_time=0.0,
                    success_rate=0.0
                )
        except Exception as e:
            logger.error(f"❌ Failed to analyze usage patterns: {e}")
            # Return empty pattern on error
            return UsagePattern(
                expert=expert,
                time_period=f"{hours}h",
                total_requests=0,
                avg_requests_per_minute=0.0,
                peak_requests_per_minute=0.0,
                avg_input_tokens=0.0,
                avg_output_tokens=0.0,
                avg_total_tokens=0.0,
                max_input_tokens=0,
                max_output_tokens=0,
                max_total_tokens=0,
                avg_processing_time=0.0,
                success_rate=0.0
            )

        # SUPABASE-ONLY: Build pattern from Supabase summary data
        total_requests = summary.get("total_requests", 0)
        time_span_minutes = hours * 60
        avg_requests_per_minute = total_requests / time_span_minutes if time_span_minutes > 0 else 0.0

        # Estimate peak rate (simplified - could be enhanced with more detailed Supabase queries)
        peak_requests_per_minute = avg_requests_per_minute * 2.0  # Conservative estimate

        return UsagePattern(
            expert=expert,
            time_period=f"{hours}h",
            total_requests=total_requests,
            avg_requests_per_minute=avg_requests_per_minute,
            peak_requests_per_minute=peak_requests_per_minute,
            avg_input_tokens=summary.get("avg_tokens", 0.0) * 0.6,  # Estimate input as 60% of total
            avg_output_tokens=summary.get("avg_tokens", 0.0) * 0.4,  # Estimate output as 40% of total
            avg_total_tokens=summary.get("avg_tokens", 0.0),
            max_input_tokens=int(summary.get("max_tokens", 0) * 0.6),
            max_output_tokens=int(summary.get("max_tokens", 0) * 0.4),
            max_total_tokens=summary.get("max_tokens", 0),
            avg_processing_time=summary.get("avg_processing_time", 0.0),
            success_rate=summary.get("success_rate", 0.0),
            market_conditions={"normal": total_requests}  # Simplified - could be enhanced
        )
    
    async def get_optimization_recommendations(self, expert: str, hours: int = 24) -> OptimizationRecommendation:
        """SUPABASE-ONLY: Generate optimization recommendations based on usage patterns."""
        pattern = await self.analyze_usage_patterns(expert, hours)
        
        # Current limits (from configuration)
        current_limits = {
            "market_regime": {"rate": 200, "tokens": 4000},
            "options_flow": {"rate": 200, "tokens": 4000},
            "sentiment": {"rate": 300, "tokens": 3000},
            "orchestrator": {"rate": 100, "tokens": 6000}
        }
        
        current_rate_limit = current_limits.get(expert, {}).get("rate", 200)
        current_token_limit = current_limits.get(expert, {}).get("tokens", 4000)
        
        # Calculate recommendations
        # Rate limit: Peak usage * 1.5 safety margin, minimum current limit
        recommended_rate_limit = max(
            current_rate_limit,
            int(pattern.peak_requests_per_minute * 60 * 1.5)
        )
        
        # Token limit: Max usage * 1.2 safety margin, minimum current limit
        recommended_token_limit = max(
            current_token_limit,
            int(pattern.max_total_tokens * 1.2)
        )
        
        # Determine confidence based on data quality
        confidence = min(0.9, pattern.total_requests / 100)  # Higher confidence with more data
        
        # Market condition factor
        volatile_requests = pattern.market_conditions.get("volatile", 0)
        crisis_requests = pattern.market_conditions.get("crisis", 0)
        
        if crisis_requests > pattern.total_requests * 0.2:
            market_factor = "crisis_optimized"
            recommended_rate_limit = int(recommended_rate_limit * 2)
        elif volatile_requests > pattern.total_requests * 0.3:
            market_factor = "volatility_optimized"
            recommended_rate_limit = int(recommended_rate_limit * 1.5)
        else:
            market_factor = "normal_conditions"
        
        # Generate reasoning
        reasoning = f"Based on {pattern.total_requests} requests over {hours}h: "
        reasoning += f"Peak rate {pattern.peak_requests_per_minute:.1f}/min, "
        reasoning += f"Max tokens {pattern.max_total_tokens}, "
        reasoning += f"Success rate {pattern.success_rate:.1%}"
        
        return OptimizationRecommendation(
            expert=expert,
            current_rate_limit=current_rate_limit,
            recommended_rate_limit=recommended_rate_limit,
            current_token_limit=current_token_limit,
            recommended_token_limit=recommended_token_limit,
            reasoning=reasoning,
            confidence=confidence,
            market_condition_factor=market_factor
        )
    
    async def get_all_recommendations(self, hours: int = 24) -> Dict[str, OptimizationRecommendation]:
        """SUPABASE-ONLY: Get optimization recommendations for all experts."""
        experts = ["market_regime", "options_flow", "sentiment", "orchestrator"]
        recommendations = {}
        for expert in experts:
            recommendations[expert] = await self.get_optimization_recommendations(expert, hours)
        return recommendations

    async def generate_usage_report(self, hours: int = 24) -> Dict[str, Any]:
        """SUPABASE-ONLY: Generate comprehensive usage report."""
        experts = ["market_regime", "options_flow", "sentiment", "orchestrator"]

        patterns = {}
        recommendations = {}
        for expert in experts:
            patterns[expert] = await self.analyze_usage_patterns(expert, hours)
            recommendations[expert] = await self.get_optimization_recommendations(expert, hours)
        
        # Overall statistics
        total_requests = sum(p.total_requests for p in patterns.values())
        success_rates = [p.success_rate for p in patterns.values() if p.total_requests > 0]
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0.0
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "time_period": f"{hours} hours",
            "current_market_condition": self.current_market_condition,
            "current_vix": self.current_vix,
            "overall_stats": {
                "total_requests": total_requests,
                "avg_success_rate": avg_success_rate
            },
            "usage_patterns": {expert: pattern.model_dump() for expert, pattern in patterns.items()},
            "optimization_recommendations": {expert: rec.model_dump() for expert, rec in recommendations.items()}
        }

# Global usage monitor instance
_usage_monitor = None

def get_usage_monitor() -> HuiHuiUsageMonitor:
    """Get global usage monitor instance."""
    global _usage_monitor
    if _usage_monitor is None:
        _usage_monitor = HuiHuiUsageMonitor()
    return _usage_monitor

def record_expert_usage(expert: str, request_type: str, input_tokens: int, 
                       output_tokens: int, processing_time: float, success: bool = True):
    """Record expert usage for monitoring."""
    monitor = get_usage_monitor()
    monitor.record_usage(expert, request_type, input_tokens, output_tokens, processing_time, success)

def update_market_condition(vix_level: float):
    """Update market condition for usage analysis."""
    monitor = get_usage_monitor()
    monitor.update_market_condition(vix_level)

async def get_usage_report(hours: int = 24) -> Dict[str, Any]:
    """SUPABASE-ONLY: Get comprehensive usage report."""
    monitor = get_usage_monitor()
    return await monitor.generate_usage_report(hours)

# ===== TESTING FUNCTION =====

async def test_usage_monitor():
    """SUPABASE-ONLY: Test the usage monitoring system."""
    print("📊 Testing HuiHui Usage Monitor (Supabase-only)...")

    monitor = get_usage_monitor()

    # Simulate some usage
    monitor.record_usage("market_regime", "analysis", 1500, 800, 2.5)
    monitor.record_usage("options_flow", "prediction", 2000, 1200, 3.2)
    monitor.record_usage("sentiment", "analysis", 1200, 600, 1.8)
    monitor.record_usage("orchestrator", "synthesis", 3000, 2000, 5.1)

    # Update market condition
    monitor.update_market_condition(25.5)  # Volatile market

    # Wait a moment for async operations
    await asyncio.sleep(1)

    # Generate report
    report = await monitor.generate_usage_report(1)
    print(f"✅ Generated usage report with {report['overall_stats']['total_requests']} requests")

    # Get recommendations
    recommendations = await monitor.get_all_recommendations(1)
    for expert, rec in recommendations.items():
        print(f"✅ {expert}: Rate {rec.current_rate_limit} → {rec.recommended_rate_limit}, "
              f"Tokens {rec.current_token_limit} → {rec.recommended_token_limit}")

    print("✅ Usage monitor test completed (Supabase-only)")

if __name__ == "__main__":
    asyncio.run(test_usage_monitor())
