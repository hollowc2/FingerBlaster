"""Mock WebSocket server for testing CLOB and RTDS connections."""

import asyncio
import json
from typing import List, Dict, Any, Optional
from collections import deque
from unittest.mock import AsyncMock


class MockWebSocketServer:
    """
    Mock WebSocket server that simulates CLOB/RTDS WebSocket behavior.

    Usage:
        server = MockWebSocketServer()
        await server.start()
        server.queue_message({"event_type": "book", "price": "0.52"})
        msg = await server.recv()
        await server.stop()
    """

    def __init__(self, auto_pong: bool = True):
        """
        Initialize mock WebSocket server.

        Args:
            auto_pong: Automatically respond to ping messages with pong
        """
        self.connected = False
        self.messages_to_send: deque = deque()
        self.received_messages: List[Dict] = []
        self.auto_pong = auto_pong

        # Connection behavior configuration
        self.should_disconnect = False
        self.disconnect_after_n_messages = None
        self.message_count = 0

        # Latency simulation
        self.recv_delay = 0.0  # Seconds to wait before returning message
        self.send_delay = 0.0

    async def start(self):
        """Start the mock WebSocket server."""
        self.connected = True

    async def stop(self):
        """Stop the mock WebSocket server."""
        self.connected = False
        self.messages_to_send.clear()
        self.received_messages.clear()

    async def connect(self, uri: str, **kwargs):
        """Mock connection method (for context manager usage)."""
        self.connected = True
        return self

    async def close(self):
        """Close the mock WebSocket connection."""
        await self.stop()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def send(self, message: str):
        """
        Mock send method - records sent messages.

        Args:
            message: JSON string to send
        """
        await asyncio.sleep(self.send_delay)

        if not self.connected:
            raise ConnectionError("WebSocket not connected")

        try:
            parsed = json.loads(message)
            self.received_messages.append(parsed)

            # Auto-respond to pings
            if self.auto_pong and parsed.get("type") == "ping":
                self.queue_message({"type": "pong"})

        except json.JSONDecodeError:
            # Store as-is if not JSON
            self.received_messages.append({"raw": message})

    async def recv(self) -> str:
        """
        Mock recv method - returns queued messages.

        Returns:
            JSON string message

        Raises:
            ConnectionError: If not connected
            asyncio.TimeoutError: If no messages available (simulated)
        """
        await asyncio.sleep(self.recv_delay)

        if not self.connected:
            raise ConnectionError("WebSocket not connected")

        # Check disconnect conditions
        if self.should_disconnect:
            self.connected = False
            raise ConnectionError("Mock disconnect triggered")

        if self.disconnect_after_n_messages is not None:
            if self.message_count >= self.disconnect_after_n_messages:
                self.connected = False
                raise ConnectionError(f"Disconnected after {self.message_count} messages")

        # Return queued message or heartbeat
        if self.messages_to_send:
            message = self.messages_to_send.popleft()
            self.message_count += 1
            return json.dumps(message)

        # Default heartbeat if no messages
        await asyncio.sleep(0.1)
        return json.dumps({"type": "heartbeat", "timestamp": self.message_count})

    def queue_message(self, message: Dict[str, Any]):
        """
        Queue a message to be received by the client.

        Args:
            message: Dictionary to send as JSON
        """
        self.messages_to_send.append(message)

    def queue_messages(self, messages: List[Dict[str, Any]]):
        """
        Queue multiple messages.

        Args:
            messages: List of message dictionaries
        """
        for msg in messages:
            self.queue_message(msg)

    def queue_order_book_update(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
    ):
        """
        Queue a CLOB order book update message.

        Args:
            token_id: Token being updated
            price: Price level
            size: Size at that level
            side: BUY or SELL
        """
        self.queue_message({
            "event_type": "book",
            "market": token_id,
            "asset_id": token_id,
            "price": str(price),
            "size": str(size),
            "side": side,
            "timestamp": self.message_count,
        })

    def queue_rtds_price_update(self, price: float, source: str = "chainlink"):
        """
        Queue an RTDS BTC price update.

        Args:
            price: BTC price
            source: Price source (chainlink or binance)
        """
        self.queue_message({
            "event_type": "price_update",
            "asset": "BTC",
            "price": price,
            "source": source,
            "timestamp": self.message_count,
        })

    def configure_disconnect(self, should_disconnect: bool = True, after_n_messages: Optional[int] = None):
        """
        Configure disconnection behavior for testing reconnection logic.

        Args:
            should_disconnect: Whether to disconnect
            after_n_messages: Disconnect after N messages
        """
        self.should_disconnect = should_disconnect
        self.disconnect_after_n_messages = after_n_messages

    def get_received_messages(self) -> List[Dict]:
        """Get all messages received from the client."""
        return self.received_messages.copy()

    def clear_received(self):
        """Clear received messages history."""
        self.received_messages.clear()

    def reset(self):
        """Reset mock state to defaults."""
        self.connected = False
        self.messages_to_send.clear()
        self.received_messages.clear()
        self.message_count = 0
        self.should_disconnect = False
        self.disconnect_after_n_messages = None


class MockWebSocketConnection:
    """
    Mock WebSocket connection object that can be used with async context managers.

    Usage:
        async with MockWebSocketConnection() as ws:
            await ws.send("test")
            msg = await ws.recv()
    """

    def __init__(self, server: Optional[MockWebSocketServer] = None):
        """
        Initialize mock WebSocket connection.

        Args:
            server: Optional shared server instance
        """
        self.server = server or MockWebSocketServer()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.server.start()
        return self.server

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.server.stop()


def create_mock_websocket(auto_pong: bool = True) -> MockWebSocketServer:
    """
    Factory function to create a mock WebSocket server.

    Args:
        auto_pong: Automatically respond to pings

    Returns:
        Mock WebSocket server instance
    """
    return MockWebSocketServer(auto_pong=auto_pong)
