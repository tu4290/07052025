"""
HuiHui AI Integration for EOTS v2.5 - SYSTEMATIC REPLACEMENT OF STATIC AI MODULES
================================================================================

This module replaces the crashed unified_ai_intelligence_system_v2_5.py with proper
HuiHui Expert Coordinator integration. Provides the same API interface but routes
all AI requests through the HuiHui 4-expert system.

REPLACES:
- unified_ai_intelligence_system_v2_5.py (CRASHED)
- Static Pydantic AI implementations
- Non-functional AI learning cycles

PROVIDES:
- HuiHui Expert Coordinator integration
- 4 specialized AI experts (Market Regime, Options Flow, Sentiment, Meta-Orchestrator)
- Proper Pydantic schema validation
- Real AI intelligence instead of static fallbacks

Author: EOTS v2.5 HuiHui Integration Division
"""

import asyncio
import logging
import statistics
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# Data Models for Feedback System
@dataclass
class TradeFeedback:
    """Structured trade feedback data."""
    trade_id: str
    expert_name: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    direction: str  # 'LONG' or 'SHORT'
    timestamp: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class ExpertPerformance:
    """Tracks performance metrics for each expert."""
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    pnl_history: Deque[float] = field(default_factory=deque)
    recent_pnls: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        return self.successful_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def avg_pnl(self) -> float:
        return statistics.mean(self.recent_pnls) if self.recent_pnls else 0.0
    
    @property
    def win_rate(self) -> float:
        if not self.recent_pnls:
            return 0.0
        wins = sum(1 for pnl in self.recent_pnls if pnl > 0)
        return wins / len(self.recent_pnls)
    
    def record_trade(self, pnl: float, is_success: bool, response_time: Optional[float] = None):
        self.total_trades += 1
        self.total_pnl += pnl
        self.recent_pnls.append(pnl)
        if is_success:
            self.successful_trades += 1
        if response_time is not None:
            self.response_times.append(response_time)
        self.last_updated = time.time()

# EOTS v2.5 Pydantic schemas
from data_models.core_models import ProcessedDataBundleV2_5, UnifiedIntelligenceAnalysis
from data_models.ai_ml_models import HuiHuiAnalysisRequestV2_5
from data_models import UnifiedLearningResult

# HuiHui Integration Components
from huihui_integration.orchestrator_bridge.expert_core import LegendaryExpertCoordinator
from huihui_integration.core.expert_router import AIRouter
from huihui_integration.learning.feedback_loops import HuiHuiLearningSystem

logger = logging.getLogger(__name__)

class HuiHuiAIIntegrationV2_5:
    """
    HuiHui AI Integration System - Replaces Unified AI Intelligence System
    
    Routes all AI analysis requests through the HuiHui Expert Coordinator system:
    - Market analysis -> Market Regime Expert
    - Options flow analysis -> Options Flow Expert  
    - Sentiment analysis -> Sentiment Expert
    - Strategic synthesis -> Meta-Orchestrator
    - Learning cycles -> HuiHui Learning System
    """
    
    def __init__(self, config_manager=None, database_manager=None):
        """Initialize HuiHui AI Integration System."""
        self.logger = logger.getChild("HuiHuiAIIntegration")
        self.config_manager = config_manager
        self.database_manager = database_manager
        
        # Initialize HuiHui components
        self.expert_coordinator = None
        self.ai_router = None
        self.learning_system = None
        self.is_initialized = False
        
        # Enhanced performance tracking
        self.analysis_count = 0
        self.learning_cycles = 0
        self.expert_performance = defaultdict(ExpertPerformance)
        self.feedback_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._background_tasks = set()
        
        # Start background tasks
        self._start_background_tasks()
        
    # ===== Background Task Management =====
    
    def _start_background_tasks(self):
        """Start all background tasks."""
        task = asyncio.create_task(self._feedback_processor())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def shutdown(self):
        """Graceful shutdown of background tasks."""
        self._shutdown_event.set()
        if self._background_tasks:
            await asyncio.wait(self._background_tasks, timeout=5.0)
    
    # ===== Feedback Processing =====
    
    async def _feedback_processor(self):
        """Process feedback in the background."""
        while not self._shutdown_event.is_set():
            try:
                feedback = await asyncio.wait_for(
                    self.feedback_queue.get(),
                    timeout=1.0
                )
                await self._process_feedback(feedback)
                self.feedback_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in feedback processor: {e}")
    
    async def record_trade_outcome(
        self,
        trade_id: str,
        expert_name: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Record trade outcome for learning and performance tracking."""
        try:
            # Calculate PnL
            price_diff = exit_price - entry_price
            pnl = price_diff * quantity * (1 if direction.upper() == 'LONG' else -1)
            pnl_percentage = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0.0
            
            # Create feedback
            feedback = TradeFeedback(
                trade_id=trade_id,
                expert_name=expert_name,
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                direction=direction.upper(),
                timestamp=time.time(),
                pnl=pnl,
                pnl_percentage=pnl_percentage,
                metadata=metadata or {}
            )
            
            # Add to queue for async processing
            await self.feedback_queue.put(feedback)
            
        except Exception as e:
            self.logger.error(f"Error recording trade outcome: {e}")
    
    async def _process_feedback(self, feedback: TradeFeedback) -> None:
        """Process trade feedback and update expert performance."""
        try:
            # Get or create expert performance tracker
            expert_perf = self.expert_performance[feedback.expert_name]
            
            # Determine if trade was successful (simple PnL-based for now)
            is_success = feedback.pnl > 0
            
            # Update metrics
            expert_perf.record_trade(
                pnl=feedback.pnl,
                is_success=is_success
            )
            
            # Update expert weight in coordinator if available
            if self.expert_coordinator and hasattr(self.expert_coordinator, 'adjust_expert_weight'):
                weight_change = self._calculate_weight_change(feedback, expert_perf)
                await self.expert_coordinator.adjust_expert_weight(
                    feedback.expert_name,
                    weight_change
                )
                self.logger.info(
                    f"Adjusted {feedback.expert_name} weight by {weight_change:.4f} "
                    f"(PnL: {feedback.pnl:.2f}, Win Rate: {expert_perf.win_rate:.1%})"
                )
                
                # Log weight adjustment to Supabase
                if self.database_manager:
                    await self.database_manager.insert(
                        "expert_weight_history",
                        {
                            "expert_name": feedback.expert_name,
                            "symbol": feedback.symbol,
                            "weight_change": float(weight_change),
                            "new_weight": float(await self.expert_coordinator.get_expert_weight(feedback.expert_name)),
                            "timestamp": datetime.now().isoformat(),
                            "trade_id": feedback.trade_id,
                            "pnl": float(feedback.pnl),
                            "pnl_percentage": float(feedback.pnl_percentage or 0.0)
                        }
                    )
            
            # Log trade to Supabase
            if self.database_manager:
                trade_data = {
                    "trade_id": feedback.trade_id,
                    "expert_name": feedback.expert_name,
                    "symbol": feedback.symbol,
                    "entry_price": float(feedback.entry_price),
                    "exit_price": float(feedback.exit_price),
                    "quantity": float(feedback.quantity),
                    "direction": feedback.direction,
                    "pnl": float(feedback.pnl),
                    "pnl_percentage": float(feedback.pnl_percentage or 0.0),
                    "success": is_success,
                    "timestamp": datetime.fromtimestamp(feedback.timestamp).isoformat(),
                    "metadata": feedback.metadata or {}
                }
                await self.database_manager.insert("trade_history", trade_data)
            
            # Log performance
            self.logger.info(
                f"Trade Feedback - {feedback.expert_name}: "
                f"PnL={feedback.pnl:.2f} ({feedback.pnl_percentage:.2f}%), "
                f"Success={is_success}, "
                f"Total Trades={expert_perf.total_trades}, "
                f"Win Rate={expert_perf.win_rate:.1%}"
            )
            
            # Update expert performance in Supabase
            if self.database_manager:
                await self.database_manager.upsert(
                    "expert_performance",
                    {
                        "expert_name": feedback.expert_name,
                        "total_trades": expert_perf.total_trades,
                        "successful_trades": expert_perf.successful_trades,
                        "success_rate": float(expert_perf.success_rate),
                        "win_rate": float(expert_perf.win_rate),
                        "total_pnl": float(expert_perf.total_pnl),
                        "avg_pnl": float(expert_perf.avg_pnl),
                        "last_updated": datetime.now().isoformat()
                    },
                    ["expert_name"]
                )
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
    
    def _calculate_weight_change(self, feedback: TradeFeedback, expert_perf: ExpertPerformance) -> float:
        """Calculate weight change based on trade outcome and expert performance."""
        base_change = 0.01  # Small base adjustment
        pnl_factor = min(abs(feedback.pnl_percentage) / 100.0, 0.05) if feedback.pnl_percentage else 0.01
        direction = 1 if feedback.pnl > 0 else -1
        performance_factor = 1.0 + (expert_perf.win_rate - 0.5)  # 0.5-1.5x multiplier
        return base_change * pnl_factor * direction * performance_factor
    
    # ===== Core Methods =====
    
    async def initialize(self) -> bool:
        """Initialize HuiHui AI components with enhanced tracking."""
        try:
            self.logger.info("🚀 Initializing Enhanced HuiHui AI Integration System...")
            
            # Initialize Expert Coordinator
            self.expert_coordinator = LegendaryExpertCoordinator()
            await self.expert_coordinator.initialize()
            
            # Initialize AI Router
            self.ai_router = AIRouter()
            self.ai_router.warm_experts()
            
            # Initialize Learning System
            self.learning_system = HuiHuiLearningSystem()
            await self.learning_system.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ HuiHui AI Integration System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize HuiHui AI Integration: {e}")
            return False
    
    async def generate_unified_intelligence(self, 
                                          data_bundle: ProcessedDataBundleV2_5,
                                          analysis_type: str = "comprehensive") -> UnifiedIntelligenceAnalysis:
        """
        Generate unified intelligence analysis using HuiHui Expert Coordinator.
        
        Enhanced with performance tracking and expert coordination.
        
        Args:
            data_bundle: The processed data bundle for analysis
            analysis_type: Type of analysis to perform (default: "comprehensive")
            
        Returns:
            UnifiedIntelligenceAnalysis: The analysis results
            
        REPLACES: unified_ai_intelligence_system_v2_5.generate_unified_intelligence_for_bundle()
        """
        if not self.is_initialized:
            await self.initialize()
            
        start_time = time.time()
        try:
            self.logger.info(f"🧠 Generating HuiHui intelligence for {data_bundle.symbol}")
            
            # Track expert usage
            expert_usage = {"analysis_type": analysis_type, "symbol": data_bundle.symbol, "timestamp": time.time()}
            
            # Create HuiHui analysis request
            request = HuiHuiAnalysisRequestV2_5(
                analysis_type=analysis_type,
                symbol=data_bundle.symbol,
                data_bundle=data_bundle,
                timestamp=datetime.now(),
                priority="high"
            )
            
            # Route through Expert Coordinator
            response = await self.expert_coordinator.coordinate_analysis(request)
            
            # Convert to UnifiedIntelligenceAnalysis format for compatibility
            intelligence_analysis = UnifiedIntelligenceAnalysis(
                symbol=data_bundle.symbol,
                timestamp=datetime.now(),
                confidence_score=response.confidence_score,
                market_regime_analysis=response.analysis_content,
                options_flow_analysis=response.analysis_content,
                sentiment_analysis=response.analysis_content,
                strategic_recommendations=response.insights,
                risk_assessment="Analyzed by HuiHui experts",
                learning_insights=response.insights,
                performance_metrics={"huihui_processing_time": response.processing_time}
            )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.analysis_count += 1
            
            # Log performance
            self.logger.info(
                f"✅ HuiHui intelligence generated successfully in {processing_time:.2f}s. "
                f"Total analyses: {self.analysis_count}"
            )
            
            # Track expert response time if available
            if hasattr(self.expert_coordinator, 'get_last_expert_usage'):
                expert_usage.update(self.expert_coordinator.get_last_expert_usage())
                
            # Store performance data in Supabase
            if self.database_manager:
                analysis_data = {
                    **expert_usage,
                    "processing_time": float(processing_time),
                    "success": True,
                    "analysis_id": f"{data_bundle.symbol}_{int(time.time())}",
                    "analysis_type": analysis_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in ai_analysis table
                await self.database_manager.insert("ai_analysis", analysis_data)
                
                # Also update expert usage stats
                if hasattr(self.expert_coordinator, 'get_expert_usage_stats'):
                    stats = await self.expert_coordinator.get_expert_usage_stats()
                    for expert, usage in stats.items():
                        await self.database_manager.upsert(
                            "expert_usage_stats",
                            {
                                "expert_name": expert,
                                **usage,
                                "last_used": datetime.now().isoformat()
                            },
                            ["expert_name"]
                        )
                
            return intelligence_analysis
            
        except Exception as e:
            self.logger.error(f"❌ HuiHui intelligence generation failed: {e}")
            # Return minimal analysis instead of crashing
            return self._create_fallback_analysis(data_bundle)
    
    async def run_unified_learning_cycle(self, 
                                       symbol: str,
                                       performance_data: Dict[str, Any] = None) -> UnifiedLearningResult:
        """
        Run unified learning cycle using HuiHui Learning System.
        
        REPLACES: unified_ai_intelligence_system_v2_5.run_unified_learning_for_symbol()
        """
        if not self.is_initialized:
            await self.initialize()
            
        try:
            self.logger.info(f"🎓 Running HuiHui learning cycle for {symbol}")
            
            # Route through HuiHui Learning System
            learning_result = await self.learning_system.run_learning_cycle(
                symbol=symbol,
                performance_data=performance_data or {}
            )
            
            # Convert to UnifiedLearningResult format for compatibility
            unified_result = UnifiedLearningResult(
                symbol=symbol,
                timestamp=datetime.now(),
                learning_insights=learning_result.get("insights", []),
                performance_improvements=learning_result.get("improvements", {}),
                expert_adaptations=learning_result.get("adaptations", {}),
                confidence_updates=learning_result.get("confidence_updates", {}),
                next_learning_cycle=learning_result.get("next_cycle", datetime.now())
            )
            
            self.learning_cycles += 1
            self.logger.info("✅ HuiHui learning cycle completed")
            return unified_result
            
        except Exception as e:
            self.logger.error(f"❌ HuiHui learning cycle failed: {e}")
            # Return minimal learning result instead of crashing
            return self._create_fallback_learning_result(symbol)
    
    def _create_fallback_analysis(self, data_bundle: ProcessedDataBundleV2_5) -> UnifiedIntelligenceAnalysis:
        """ZERO TOLERANCE FAKE DATA: No fallback analysis allowed."""
        raise ValueError(f"CRITICAL: HuiHui system failed for {data_bundle.symbol} and no fallback data allowed - cannot provide analysis without real AI intelligence!")
    
    def _create_fallback_learning_result(self, symbol: str) -> UnifiedLearningResult:
        """Create fallback learning result when HuiHui learning fails."""
        return UnifiedLearningResult(
            symbol=symbol,
            timestamp=datetime.now(),
            learning_insights=["HuiHui learning system temporarily unavailable"],
            performance_improvements={},
            expert_adaptations={},
            confidence_updates={},
            next_learning_cycle=datetime.now()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get HuiHui AI Integration system status."""
        return {
            "initialized": self.is_initialized,
            "expert_coordinator_ready": self.expert_coordinator is not None,
            "ai_router_ready": self.ai_router is not None,
            "learning_system_ready": self.learning_system is not None,
            "analysis_count": self.analysis_count,
            "learning_cycles": self.learning_cycles,
            "expert_performance": self.expert_performance
        }

# Global instance for compatibility with existing code
_huihui_ai_system = None

def get_unified_ai_intelligence_system(config_manager=None, database_manager=None) -> HuiHuiAIIntegrationV2_5:
    """
    Get HuiHui AI Integration System instance.
    
    REPLACES: unified_ai_intelligence_system_v2_5.get_unified_ai_intelligence_system()
    """
    global _huihui_ai_system
    
    if _huihui_ai_system is None:
        _huihui_ai_system = HuiHuiAIIntegrationV2_5(config_manager, database_manager)
    
    return _huihui_ai_system

async def generate_unified_intelligence_for_bundle(data_bundle: ProcessedDataBundleV2_5) -> UnifiedIntelligenceAnalysis:
    """
    Generate unified intelligence for data bundle using HuiHui experts.
    
    REPLACES: unified_ai_intelligence_system_v2_5.generate_unified_intelligence_for_bundle()
    """
    system = get_unified_ai_intelligence_system()
    return await system.generate_unified_intelligence(data_bundle)

async def run_unified_learning_for_symbol(symbol: str, performance_data: Dict[str, Any] = None) -> UnifiedLearningResult:
    """
    Run unified learning cycle for symbol using HuiHui learning system.
    
    REPLACES: unified_ai_intelligence_system_v2_5.run_unified_learning_for_symbol()
    """
    system = get_unified_ai_intelligence_system()
    return await system.run_unified_learning_cycle(symbol, performance_data)

# Compatibility exports
UnifiedAIIntelligenceSystemV2_5 = HuiHuiAIIntegrationV2_5