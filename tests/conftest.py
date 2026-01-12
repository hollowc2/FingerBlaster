import pytest
import os
import sys
import warnings
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress websockets.legacy deprecation warning coming from external libraries (like web3)
# Our internal code (engine.py, coinbase.py) has already been migrated to the new asyncio API.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.legacy")

# Import test fixtures and mocks
from tests.mocks.mock_polymarket import MockPolymarketConnector
from tests.mocks.mock_websocket import MockWebSocketServer, create_mock_websocket
from tests.mocks.mock_web3 import MockWeb3Provider, create_mock_web3
from tests.fixtures.market_data import (
    SAMPLE_MARKET,
    SAMPLE_ORDERBOOK,
    THIN_ORDERBOOK,
    EMPTY_ORDERBOOK,
    generate_orderbook,
    generate_market_with_times,
)


# ========== Event Loop Fixture ==========
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ========== Environment Setup ==========
@pytest.fixture
def mock_env_setup(monkeypatch):
    """Sets up mock environment variables for testing."""
    monkeypatch.setenv("POLYMARKET_API_KEY", "test_key")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "test_secret")
    monkeypatch.setenv("POLYMARKET_PASSPHRASE", "test_passphrase")
    monkeypatch.setenv("PRIVATE_KEY", "0x0000000000000000000000000000000000000000000000000000000000000001")


# ========== Mock Connectors ==========
@pytest.fixture
def mock_polymarket_connector():
    """Mocked PolymarketConnector with common methods."""
    connector = MockPolymarketConnector()
    return connector


@pytest.fixture
def mock_polymarket_with_market(mock_polymarket_connector):
    """Mocked PolymarketConnector with SAMPLE_MARKET pre-loaded."""
    mock_polymarket_connector.set_market_data([SAMPLE_MARKET])
    return mock_polymarket_connector


# ========== WebSocket Mocks ==========
@pytest.fixture
async def mock_websocket_server():
    """Mock WebSocket server for testing."""
    server = MockWebSocketServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def mock_websocket_factory():
    """Factory to create mock WebSocket servers on demand."""
    return create_mock_websocket


# ========== Web3 Mocks ==========
@pytest.fixture
def mock_web3():
    """Mock Web3 provider for transaction signing tests."""
    return create_mock_web3()


# ========== Market Data Fixtures ==========
@pytest.fixture
def sample_market():
    """Sample market data structure."""
    return SAMPLE_MARKET.copy()


@pytest.fixture
def sample_order_book():
    """Sample order book with realistic depth."""
    return SAMPLE_ORDERBOOK.copy()


@pytest.fixture
def thin_order_book():
    """Thin order book for slippage testing."""
    return THIN_ORDERBOOK.copy()


@pytest.fixture
def empty_order_book():
    """Empty order book for edge case testing."""
    return EMPTY_ORDERBOOK.copy()


@pytest.fixture
def orderbook_factory():
    """Factory function to generate custom order books."""
    return generate_orderbook


@pytest.fixture
def market_factory():
    """Factory function to generate custom markets."""
    return generate_market_with_times


# ========== Time Control ==========
@pytest.fixture
def frozen_time():
    """
    Freeze time for deterministic tests.

    Usage:
        def test_something(frozen_time):
            frozen_time.freeze("2026-01-11 12:00:00")
            # Time is now frozen
    """
    try:
        from freezegun import freeze_time
        with freeze_time("2026-01-11 12:00:00") as frozen:
            yield frozen
    except ImportError:
        # If freezegun not installed, return a mock
        yield None


# ========== Analytics Fixtures ==========
@pytest.fixture
def analytics_engine():
    """Fresh AnalyticsEngine instance for testing."""
    from src.activetrader.analytics import AnalyticsEngine
    return AnalyticsEngine()


# ========== Configuration ==========
@pytest.fixture
def test_config():
    """AppConfig with test-safe values (no rate limits, fast intervals)."""
    from src.activetrader.config import AppConfig

    config = AppConfig()
    # Override for testing
    config.order_rate_limit_seconds = 0.0  # No rate limiting in tests
    config.ws_reconnect_delay = 0.1  # Fast reconnection
    config.ws_max_reconnect_attempts = 3  # Fewer attempts
    config.analytics_interval = 0.1  # Fast analytics
    config.position_update_interval = 1.0  # Fast position updates
    config.market_discovery_interval = 1.0  # Fast discovery

    return config


# ========== Callback Tracking ==========
@pytest.fixture
def callback_tracker():
    """
    Utility to track callback invocations during tests.

    Usage:
        def test_callbacks(callback_tracker):
            core.register_callback('market_update', callback_tracker.track('market_update'))
            # ... trigger event
            assert callback_tracker.was_called('market_update')
    """
    class CallbackTracker:
        def __init__(self):
            self.calls = {}

        def track(self, event_name: str):
            """Create a tracking callback for an event."""
            def callback(*args, **kwargs):
                if event_name not in self.calls:
                    self.calls[event_name] = []
                self.calls[event_name].append((args, kwargs))
            return callback

        def was_called(self, event_name: str) -> bool:
            """Check if event was called."""
            return event_name in self.calls and len(self.calls[event_name]) > 0

        def call_count(self, event_name: str) -> int:
            """Get number of times event was called."""
            return len(self.calls.get(event_name, []))

        def get_calls(self, event_name: str):
            """Get all calls for an event."""
            return self.calls.get(event_name, [])

        def reset(self):
            """Reset all tracked calls."""
            self.calls = {}

    return CallbackTracker()
