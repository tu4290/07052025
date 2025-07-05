"""
Expert Coordinator Core

Core coordination logic for the Expert Coordinator.
"""
import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional, Set
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from .expert_models import (
    ExpertPerformanceMetrics,
    CoordinationMode
)
from .expert_config import CoordinationStrategy, DEFAULT_STRATEGIES

logger = logging.getLogger(__name__)

class ExpertCoordinatorCore:
    """Core coordination logic for the Expert Coordinator."""
    
    def __init__(self, expert_router=None, db_manager=None):
        """Initialize the core coordinator.
        
        Args:
            expert_router: Expert router instance
            db_manager: Database manager instance
        """
        self.logger = logger.getChild("CoordinatorCore")
        self.expert_router = expert_router
        self.db_manager = db_manager
        
        # Expert tracking
        self.experts: Dict[str, ExpertPerformanceMetrics] = {}
        self.expert_weights: Dict[str, float] = {}
        self.expert_load: Dict[str, int] = defaultdict(int)
        self.expert_last_used: Dict[str, datetime] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Any] = {}
        
        # Request deduplication
        self.request_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Initialize expert metrics
        asyncio.create_task(self._initialize_expert_metrics())
        
    async def _initialize_expert_metrics(self) -> None:
        """Initialize expert metrics from router."""
        if not self.expert_router:
            return
            
        try:
            # Get initial expert list and performance data
            experts = await self.expert_router.list_experts()
            for expert in experts:
                expert_id = expert.get('id')
                if expert_id and expert_id not in self.experts:
                    self.experts[expert_id] = ExpertPerformanceMetrics(
                        expert_id=expert_id,
                        **expert.get('metadata', {})
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize expert metrics: {e}")
    
    async def coordinate_analysis(
        self,
        request: Any,
        strategy: Optional[CoordinationStrategy] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Coordinate analysis across multiple experts.
        
        Args:
            request: Analysis request
            strategy: Coordination strategy
            timeout: Operation timeout in seconds
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        self.logger.info(f"🚀 Starting coordination for request {request_id}")
        
        try:
            # Use default strategy if none provided
            if strategy is None:
                strategy = self._select_adaptive_strategy(request)
            
            # Set timeout if not provided
            if timeout is None:
                market_context = await self._analyze_market_context(request)
                timeout = self._calculate_timeout(strategy.mode, market_context)
            
            # Check for direct expert routing
            if hasattr(request, 'expert_id') and request.expert_id:
                return await self._route_to_expert(
                    expert_id=request.expert_id,
                    request=request,
                    timeout=timeout
                )
            
            # Select and execute with multiple experts
            return await self._execute_coordinated_analysis(
                request=request,
                strategy=strategy,
                timeout=timeout
            )
            
        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            return await self._handle_coordination_failure(request, str(e))
    
    async def _route_to_expert(
        self,
        expert_id: str,
        request: Any,
        timeout: float
    ) -> Dict[str, Any]:
        """Route request to a specific expert."""
        self.logger.debug(f"Routing directly to expert: {expert_id}")
        
        # Temporarily remove metrics recording
        # metrics.record_request(expert_id, 0)  # Duration updated after completion
        
        try:
            # Execute with circuit breaker
            result = await self._execute_with_circuit_breaker(
                expert_id=expert_id,
                request=request,
                timeout=timeout
            )
            
            # Temporarily remove metrics recording
            # duration = time.time() - (getattr(request, 'start_time', time.time()))
            # metrics.record_request(expert_id, duration, success=True)
            
            return result
            
        except Exception:
            # Temporarily remove metrics recording
            # duration = time.time() - (getattr(request, 'start_time', time.time()))
            # metrics.record_request(expert_id, duration, success=False)
            raise
    
    async def _execute_coordinated_analysis(
        self,
        request: Any,
        strategy: CoordinationStrategy,
        timeout: float
    ) -> Dict[str, Any]:
        """Execute analysis with multiple experts and build consensus."""
        # This method would contain the main coordination logic
        # including expert selection, parallel execution, and consensus building
        raise NotImplementedError("Coordinated analysis not implemented")
    
    async def _execute_with_circuit_breaker(
        self,
        expert_id: str,
        request: Any,
        timeout: float
    ) -> Dict[str, Any]:
        """Execute request with circuit breaker pattern."""
        # This method would implement the circuit breaker logic
        # for executing requests with a specific expert
        raise NotImplementedError("Circuit breaker execution not implemented")
    
    def _select_adaptive_strategy(self, request: Any) -> CoordinationStrategy:
        """Select coordination strategy based on request and conditions."""
        # Default to adaptive strategy
        return DEFAULT_STRATEGIES[CoordinationMode.ADAPTIVE]
    
    # Commented out due to import and parameter issues
    # async def _analyze_market_context(self, request: Any) -> MarketConditionContext:
    #     """Analyze current market conditions."""
    #     # TODO: Implement proper market context analysis
    #     return MarketConditionContext(
    #         volatility_regime="normal",
    #         market_trend="neutral",
    #         options_flow_intensity="medium",
    #         sentiment_regime="neutral",
    #         time_of_day=datetime.utcnow().strftime("%H:%M"),
    #         market_stress_level=0.3,
    #         liquidity_condition="good",
    #         news_impact_level=0.2
    #     )

    # Temporary placeholder for market context analysis
    async def _analyze_market_context(self, request: Any) -> Dict[str, Any]:
        """Temporary market context analysis method."""
        return {
            "volatility": "normal",
            "trend": "neutral",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_timeout(self, mode: CoordinationMode, context: Dict[str, Any]) -> float:
        """Calculate appropriate timeout based on mode and market context."""
        # Base timeouts in seconds
        base_timeouts = {
            CoordinationMode.EMERGENCY: 10.0,
            CoordinationMode.COMPETITIVE: 20.0,
            CoordinationMode.WEIGHTED: 30.0,
            CoordinationMode.CONSENSUS: 45.0,
            CoordinationMode.COLLABORATIVE: 60.0,
            CoordinationMode.ADAPTIVE: 30.0
        }
        
        # Adjust based on market volatility
        volatility_factor = 1.0
        if context.get('volatility') == 'high':
            volatility_factor = 1.5
        elif context.get('volatility') == 'low':
            volatility_factor = 0.8
        
        return base_timeouts.get(mode, 30.0) * volatility_factor
    
    def _get_orchestrator_fallback(self):
        """Retrieve the orchestrator fallback mechanism."""
        # Temporarily disabled due to import issues
        self.logger.warning("Orchestrator fallback is currently unavailable")
        return None
    
    async def _handle_coordination_failure(
        self,
        request: Any,
        error: str
    ) -> Dict[str, Any]:
        """Handle coordination failures with appropriate fallback."""
        self.logger.error(f"Coordination failed: {error}")
        
        # Simplified error handling without orchestrator
        return {
            "success": False,
            "error": f"Coordination failed: {error}",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request, 'request_id', str(uuid.uuid4()))
        }
        
    async def legendary_coordinate_analysis(
        self,
        request: Any,
        strategy: Optional[CoordinationStrategy] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Legacy-compatible coordinate analysis method.
        
        This method provides backward compatibility with the legacy ExpertCoordinator API
        while delegating to the new modular implementation.
        
        Args:
            request: The analysis request object
            strategy: Optional coordination strategy
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing the analysis results
        """
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        start_time = time.time()
        
        try:
            # Log the incoming request
            self.logger.info(
                f"🔍 Processing legacy analysis request {request_id}"
            )
            
            # Convert legacy request format if needed
            if hasattr(request, 'to_dict'):
                request_dict = request.to_dict()
            elif isinstance(request, dict):
                request_dict = request
            else:
                request_dict = dict(request) if hasattr(request, '__dict__') else {}
            
            # Set request ID if not present
            if 'request_id' not in request_dict:
                request_dict['request_id'] = request_id
            
            # Set start time if not present
            if 'start_time' not in request_dict:
                request_dict['start_time'] = start_time
            
            # Delegate to the new coordinate_analysis method
            result = await self.coordinate_analysis(
                request=request_dict,
                strategy=strategy,
                timeout=timeout
            )
            
            # Log success
            duration = time.time() - start_time
            self.logger.info(
                f"✅ Completed legacy analysis request {request_id} in {duration:.2f}s"
            )
            
            # Update metrics
            # metrics.record_request(
            #     expert_id="legacy",
            #     duration=duration,
            #     success=True
            # )
            
            return result
            
        except Exception as e:
            # Log and handle errors
            duration = time.time() - start_time
            error_msg = f"Error in legacy coordinate analysis: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Update metrics
            # metrics.record_request(
            #     expert_id="legacy",
            #     duration=duration,
            #     success=False
            # )
            
            # Return error response
            return {
                "success": False,
                "error": error_msg,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }

# Legacy aliases for backward compatibility
LegendaryExpertCoordinator = ExpertCoordinatorCore

__all__ = [
    'ExpertCoordinatorCore',
    'LegendaryExpertCoordinator'
]

# Global coordinator instance for backward compatibility
_legendary_coordinator = None

def get_legendary_coordinator():
    """Get the global legendary coordinator instance."""
    global _legendary_coordinator
    if _legendary_coordinator is None:
        _legendary_coordinator = LegendaryExpertCoordinator()
    return _legendary_coordinator

def initialize_legendary_coordinator(db_manager=None):
    """Initialize the legendary expert coordinator."""
    global _legendary_coordinator
    _legendary_coordinator = LegendaryExpertCoordinator(db_manager=db_manager)
    return _legendary_coordinator

# Fallback implementations for ExpertPerformance and TradeFeedback
@dataclass
class ExpertPerformance:
    name: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.successful_trades / self.total_trades
    
    @property
    def success_rate(self) -> float:
        return self.win_rate
    
    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

# Remove or comment out metrics recording calls
# Temporarily disable metrics recording
# TODO: Implement proper metrics tracking mechanism

# Comment out or remove metrics.record_request calls
# For example, in the _route_to_expert method:
async def _route_to_expert(
    self,
    expert_id: str,
    request: Any,
    timeout: float
) -> Dict[str, Any]:
    """Route request to a specific expert."""
    self.logger.debug(f"Routing directly to expert: {expert_id}")
    
    # Temporarily remove metrics recording
    # metrics.record_request(expert_id, 0)  # Duration updated after completion
    
    try:
        # Execute with circuit breaker
        result = await self._execute_with_circuit_breaker(
            expert_id=expert_id,
            request=request,
            timeout=timeout
        )
        
        # Temporarily remove metrics recording
        # duration = time.time() - (getattr(request, 'start_time', time.time()))
        # metrics.record_request(expert_id, duration, success=True)
        
        return result
        
    except Exception:
        # Temporarily remove metrics recording
        # duration = time.time() - (getattr(request, 'start_time', time.time()))
        # metrics.record_request(expert_id, duration, success=False)
        raise

# Remove or modify any other problematic method calls
