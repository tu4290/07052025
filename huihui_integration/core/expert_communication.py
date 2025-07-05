"""
HuiHui Expert Communication Protocol
====================================

Communication system for HuiHui AI experts including:
- Inter-expert message passing and coordination
- Request routing and load balancing
- Response aggregation and consensus building
- Performance monitoring and optimization

Author: EOTS v2.5 AI Architecture Division
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pydantic import BaseModel, Field

# EOTS v2.5 Pydantic schemas
from data_models import HuiHuiAnalysisRequestV2_5

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Types of inter-expert messages."""
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    COORDINATION_REQUEST = "coordination_request"
    PERFORMANCE_UPDATE = "performance_update"
    HEALTH_CHECK = "health_check"
    SHUTDOWN_SIGNAL = "shutdown_signal"

class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ExpertMessage(BaseModel):
    """Message structure for inter-expert communication."""
    message_id: str = Field(..., description="Unique message identifier")
    message_type: MessageType = Field(..., description="Type of message")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    sender_expert_id: str = Field(..., description="ID of sending expert")
    recipient_expert_id: Optional[str] = Field(None, description="ID of recipient expert (None for broadcast)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    requires_response: bool = Field(default=False, description="Whether message requires response")
    correlation_id: Optional[str] = Field(None, description="ID for correlating request/response pairs")

class ExpertCommunicationProtocol:
    """
    Communication protocol for HuiHui experts.
    
    Handles message routing, delivery, and coordination between experts.
    """
    
    def __init__(self):
        self.logger = logger.getChild("ExpertCommunication")
        self.message_handlers: Dict[str, Callable] = {}
        self.expert_endpoints: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "average_response_time_ms": 0.0
        }
    
    def register_expert(self, expert_id: str, message_handler: Callable) -> None:
        """
        Register an expert with the communication protocol.
        
        Args:
            expert_id: Unique identifier for the expert
            message_handler: Function to handle incoming messages
        """
        self.message_handlers[expert_id] = message_handler
        self.expert_endpoints[expert_id] = {
            "registered_at": datetime.now(),
            "last_seen": datetime.now(),
            "message_count": 0,
            "status": "active"
        }
        self.logger.info(f"📡 Registered expert for communication: {expert_id}")
    
    def unregister_expert(self, expert_id: str) -> None:
        """Unregister an expert from the communication protocol."""
        if expert_id in self.message_handlers:
            del self.message_handlers[expert_id]
        if expert_id in self.expert_endpoints:
            del self.expert_endpoints[expert_id]
        self.logger.info(f"📡 Unregistered expert: {expert_id}")
    
    async def send_message(self, message: ExpertMessage) -> bool:
        """
        Send a message to an expert or broadcast to all experts.
        
        Args:
            message: The message to send
            
        Returns:
            bool: True if message was queued successfully
        """
        try:
            await self.message_queue.put(message)
            self.stats["messages_sent"] += 1
            self.logger.debug(f"📤 Queued message {message.message_id} from {message.sender_expert_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue message: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def process_messages(self) -> None:
        """Process messages from the queue."""
        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._deliver_message(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _deliver_message(self, message: ExpertMessage) -> None:
        """Deliver a message to the appropriate expert(s)."""
        try:
            start_time = datetime.now()
            
            if message.recipient_expert_id:
                # Direct message to specific expert
                await self._deliver_to_expert(message.recipient_expert_id, message)
            else:
                # Broadcast to all experts except sender
                for expert_id in self.message_handlers:
                    if expert_id != message.sender_expert_id:
                        await self._deliver_to_expert(expert_id, message)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_response_time_stats(processing_time)
            self.stats["messages_received"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to deliver message {message.message_id}: {e}")
            self.stats["messages_failed"] += 1
    
    async def _deliver_to_expert(self, expert_id: str, message: ExpertMessage) -> None:
        """Deliver message to a specific expert."""
        if expert_id not in self.message_handlers:
            self.logger.warning(f"Expert {expert_id} not registered for message delivery")
            return
        
        try:
            handler = self.message_handlers[expert_id]
            await handler(message)
            
            # Update expert endpoint stats
            if expert_id in self.expert_endpoints:
                self.expert_endpoints[expert_id]["last_seen"] = datetime.now()
                self.expert_endpoints[expert_id]["message_count"] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to deliver message to expert {expert_id}: {e}")
    
    def _update_response_time_stats(self, processing_time_ms: float) -> None:
        """Update average response time statistics."""
        current_avg = self.stats["average_response_time_ms"]
        message_count = self.stats["messages_received"] + 1
        
        # Calculate running average
        self.stats["average_response_time_ms"] = (
            (current_avg * (message_count - 1) + processing_time_ms) / message_count
        )
    
    async def broadcast_analysis_request(self, sender_id: str, request: HuiHuiAnalysisRequestV2_5) -> List[str]:
        """
        Broadcast analysis request to all experts.
        
        Args:
            sender_id: ID of the requesting expert/orchestrator
            request: Analysis request to broadcast
            
        Returns:
            List of message IDs for tracking responses
        """
        message_ids = []
        
        for expert_id in self.message_handlers:
            if expert_id != sender_id:
                message_id = f"analysis_{expert_id}_{datetime.now().timestamp()}"
                message = ExpertMessage(
                    message_id=message_id,
                    message_type=MessageType.ANALYSIS_REQUEST,
                    priority=MessagePriority.HIGH,
                    sender_expert_id=sender_id,
                    recipient_expert_id=expert_id,
                    payload=request.model_dump(),  # 🚀 PYDANTIC-FIRST: Use model_dump()
                    requires_response=True
                )
                
                if await self.send_message(message):
                    message_ids.append(message_id)
        
        return message_ids
    
    async def send_health_check(self, sender_id: str) -> Dict[str, Any]:
        """Send health check to all experts and collect responses."""
        responses = {}
        
        for expert_id in self.message_handlers:
            if expert_id != sender_id:
                message = ExpertMessage(
                    message_id=f"health_{expert_id}_{datetime.now().timestamp()}",
                    message_type=MessageType.HEALTH_CHECK,
                    priority=MessagePriority.NORMAL,
                    sender_expert_id=sender_id,
                    recipient_expert_id=expert_id,
                    requires_response=True
                )
                
                await self.send_message(message)
        
        return responses
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication protocol statistics."""
        return {
            "stats": self.stats.copy(),
            "registered_experts": len(self.message_handlers),
            "expert_endpoints": self.expert_endpoints.copy(),
            "queue_size": self.message_queue.qsize(),
            "running": self.running
        }
    
    async def start(self) -> None:
        """Start the communication protocol."""
        self.running = True
        self.logger.info("🚀 Expert communication protocol started")
        
        # Start message processing task
        asyncio.create_task(self.process_messages())
    
    async def stop(self) -> None:
        """Stop the communication protocol."""
        self.running = False
        self.logger.info("🛑 Expert communication protocol stopped")

# Global communication protocol instance
_communication_protocol = None

def get_communication_protocol() -> ExpertCommunicationProtocol:
    """Get the global communication protocol instance."""
    global _communication_protocol
    if _communication_protocol is None:
        _communication_protocol = ExpertCommunicationProtocol()
    return _communication_protocol

async def initialize_communication() -> ExpertCommunicationProtocol:
    """Initialize and start the communication protocol."""
    protocol = get_communication_protocol()
    await protocol.start()
    return protocol
