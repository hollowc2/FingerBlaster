"""Comprehensive tests for RTDSManager - BTC price updates, strike resolution, history management."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from websockets.exceptions import ConnectionClosed, InvalidURI, InvalidState
import pandas as pd

from src.activetrader.engine import RTDSManager
from src.activetrader.config import AppConfig


# ========== Test Fixtures ==========
@pytest.fixture
def config():
    """AppConfig with short timeouts for testing."""
    cfg = AppConfig()
    cfg.rtds_reconnect_delay = 0.1  # Fast reconnect for tests
    cfg.rtds_max_reconnect_attempts = 3
    cfg.rtds_ping_interval = 0.5
    cfg.rtds_recv_timeout = 1.0
    cfg.rtds_price_stale_threshold_seconds = 2.0  # Fast stale detection for tests
    return cfg


@pytest.fixture
def on_btc_price_callback():
    """Mock BTC price callback."""
    return AsyncMock()


@pytest.fixture
def rtds_manager(config, on_btc_price_callback):
    """RTDSManager instance."""
    return RTDSManager(config, on_btc_price_callback)


# ========== BTC Price Update Tests (Chainlink Oracle) ==========
class TestChainlinkPriceUpdates:
    """Test BTC price updates from Chainlink oracle."""

    @pytest.mark.asyncio
    async def test_chainlink_price_update_triggers_callback(self, rtds_manager, on_btc_price_callback):
        """Test Chainlink BTC price update triggers callback."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.50,
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        on_btc_price_callback.assert_called_once_with(95000.50)
        assert rtds_manager.current_chainlink_price == 95000.50

    @pytest.mark.asyncio
    async def test_chainlink_price_stored_in_history(self, rtds_manager):
        """Test Chainlink prices are stored in history with timestamps."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        assert 1700000000000 in rtds_manager.chainlink_price_history
        assert rtds_manager.chainlink_price_history[1700000000000] == 95000.00

    @pytest.mark.asyncio
    async def test_chainlink_price_case_insensitive_symbol(self, rtds_manager, on_btc_price_callback):
        """Test symbol matching is case insensitive."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'BTC/USD',  # Uppercase
                'value': 96000.00,
                'timestamp': 1700000001000
            }
        }

        await rtds_manager._process_message(message)

        on_btc_price_callback.assert_called_once_with(96000.00)

    @pytest.mark.asyncio
    async def test_chainlink_ignores_invalid_price(self, rtds_manager, on_btc_price_callback):
        """Test invalid prices are ignored."""
        # Zero price
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 0,
                'timestamp': 1700000000000
            }
        }
        await rtds_manager._process_message(message)
        on_btc_price_callback.assert_not_called()

        # Negative price
        message['payload']['value'] = -100.0
        await rtds_manager._process_message(message)
        on_btc_price_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_chainlink_ignores_non_btc_symbols(self, rtds_manager, on_btc_price_callback):
        """Test non-BTC symbols are ignored."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'eth/usd',
                'value': 3000.00,
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        on_btc_price_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_chainlink_handles_string_price(self, rtds_manager, on_btc_price_callback):
        """Test string price values are not processed (only int/float)."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': '95000.00',  # String, not number
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        # String values should be ignored by isinstance check
        on_btc_price_callback.assert_not_called()


# ========== Binance Fallback Tests ==========
class TestBinanceFallback:
    """Test fallback to Binance when Chainlink unavailable."""

    @pytest.mark.asyncio
    async def test_binance_used_when_chainlink_unavailable(self, rtds_manager, on_btc_price_callback):
        """Test Binance price used when Chainlink not available."""
        message = {
            'topic': 'crypto_prices',
            'payload': {
                'symbol': 'btcusdt',
                'value': 94500.00
            }
        }

        await rtds_manager._process_message(message)

        # Should trigger callback since Chainlink is not available
        on_btc_price_callback.assert_called_once_with(94500.00)
        assert rtds_manager.current_btc_price == 94500.00

    @pytest.mark.asyncio
    async def test_chainlink_preferred_over_binance(self, rtds_manager, on_btc_price_callback):
        """Test Chainlink is preferred when both are available."""
        # First, set up Chainlink price
        chainlink_message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': 1700000000000
            }
        }
        await rtds_manager._process_message(chainlink_message)

        on_btc_price_callback.reset_mock()

        # Now send Binance price
        binance_message = {
            'topic': 'crypto_prices',
            'payload': {
                'symbol': 'btcusdt',
                'value': 94500.00
            }
        }
        await rtds_manager._process_message(binance_message)

        # Binance should NOT trigger callback since Chainlink is available
        on_btc_price_callback.assert_not_called()
        # But Binance price should still be stored
        assert rtds_manager.current_btc_price == 94500.00

    @pytest.mark.asyncio
    async def test_binance_ignores_non_btc(self, rtds_manager, on_btc_price_callback):
        """Test non-BTC Binance symbols are ignored."""
        message = {
            'topic': 'crypto_prices',
            'payload': {
                'symbol': 'ethusdt',
                'value': 3000.00
            }
        }

        await rtds_manager._process_message(message)

        on_btc_price_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_current_price_prefers_chainlink(self, rtds_manager):
        """Test get_current_price returns Chainlink when available."""
        rtds_manager.current_chainlink_price = 95000.00
        rtds_manager.current_btc_price = 94500.00

        price = rtds_manager.get_current_price()

        assert price == 95000.00  # Chainlink preferred

    @pytest.mark.asyncio
    async def test_get_current_price_falls_back_to_binance(self, rtds_manager):
        """Test get_current_price falls back to Binance."""
        rtds_manager.current_chainlink_price = None
        rtds_manager.current_btc_price = 94500.00

        price = rtds_manager.get_current_price()

        assert price == 94500.00  # Falls back to Binance

    @pytest.mark.asyncio
    async def test_get_current_price_returns_none_when_unavailable(self, rtds_manager):
        """Test get_current_price returns None when no prices available."""
        rtds_manager.current_chainlink_price = None
        rtds_manager.current_btc_price = None

        price = rtds_manager.get_current_price()

        assert price is None


# ========== Strike Price Resolution Tests ==========
class TestStrikePriceResolution:
    """Test strike price resolution from historical data."""

    def test_get_chainlink_price_at_exact_match(self, rtds_manager):
        """Test exact timestamp match for strike price."""
        timestamp_ms = 1700000000000
        rtds_manager.chainlink_price_history[timestamp_ms] = 95000.00

        ts = pd.Timestamp.fromtimestamp(timestamp_ms / 1000, tz='UTC')
        price = rtds_manager.get_chainlink_price_at(ts)

        assert price == 95000.00

    def test_get_chainlink_price_at_closest_match(self, rtds_manager):
        """Test closest timestamp match within tolerance."""
        # Store price at specific timestamp
        stored_ts = 1700000000000
        rtds_manager.chainlink_price_history[stored_ts] = 95000.00

        # Request price 30 seconds later (within 60s tolerance)
        request_ts = pd.Timestamp.fromtimestamp((stored_ts + 30000) / 1000, tz='UTC')
        price = rtds_manager.get_chainlink_price_at(request_ts)

        assert price == 95000.00

    def test_get_chainlink_price_at_boundary_60_seconds(self, rtds_manager):
        """Test 60 second boundary for closest match."""
        stored_ts = 1700000000000
        rtds_manager.chainlink_price_history[stored_ts] = 95000.00

        # Request price exactly at 60s boundary (should still match)
        request_ts = pd.Timestamp.fromtimestamp((stored_ts + 60000) / 1000, tz='UTC')
        price = rtds_manager.get_chainlink_price_at(request_ts)

        assert price == 95000.00

    def test_get_chainlink_price_at_beyond_tolerance(self, rtds_manager):
        """Test no match when beyond 60 second tolerance."""
        stored_ts = 1700000000000
        rtds_manager.chainlink_price_history[stored_ts] = 95000.00

        # Request price 61 seconds later (beyond tolerance)
        request_ts = pd.Timestamp.fromtimestamp((stored_ts + 61000) / 1000, tz='UTC')
        price = rtds_manager.get_chainlink_price_at(request_ts)

        assert price is None

    def test_get_chainlink_price_at_empty_history(self, rtds_manager):
        """Test returns None when history is empty."""
        ts = pd.Timestamp.now(tz='UTC')
        price = rtds_manager.get_chainlink_price_at(ts)

        assert price is None

    def test_get_chainlink_price_at_finds_closest_among_multiple(self, rtds_manager):
        """Test finds closest price when multiple entries exist."""
        # Store multiple prices
        rtds_manager.chainlink_price_history[1700000000000] = 94000.00  # Oldest
        rtds_manager.chainlink_price_history[1700000030000] = 95000.00  # Middle
        rtds_manager.chainlink_price_history[1700000060000] = 96000.00  # Newest

        # Request price closest to middle
        request_ts = pd.Timestamp.fromtimestamp(1700000035 / 1, tz='UTC')  # 35 seconds in
        price = rtds_manager.get_chainlink_price_at(request_ts)

        # Should find the closest (30s entry)
        assert price == 95000.00


# ========== History Deque Boundary Tests ==========
class TestHistoryBoundary:
    """Test history deque boundary conditions and cleanup."""

    @pytest.mark.asyncio
    async def test_history_cleanup_removes_old_entries(self, rtds_manager):
        """Test entries older than 1 hour are cleaned up."""
        # Add an old entry (2 hours ago)
        old_ts = int(time.time() * 1000) - 7200000  # 2 hours ago
        rtds_manager.chainlink_price_history[old_ts] = 90000.00

        # Process a new message (triggers cleanup)
        current_ts = int(time.time() * 1000)
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': current_ts
            }
        }
        await rtds_manager._process_message(message)

        # Old entry should be cleaned up
        assert old_ts not in rtds_manager.chainlink_price_history
        # New entry should exist
        assert current_ts in rtds_manager.chainlink_price_history

    @pytest.mark.asyncio
    async def test_history_retains_recent_entries(self, rtds_manager):
        """Test entries within 1 hour are retained."""
        # Add entry from 30 minutes ago
        recent_ts = int(time.time() * 1000) - 1800000  # 30 min ago
        rtds_manager.chainlink_price_history[recent_ts] = 94000.00

        # Process a new message
        current_ts = int(time.time() * 1000)
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': current_ts
            }
        }
        await rtds_manager._process_message(message)

        # Recent entry should be retained
        assert recent_ts in rtds_manager.chainlink_price_history
        assert rtds_manager.chainlink_price_history[recent_ts] == 94000.00

    @pytest.mark.asyncio
    async def test_history_handles_many_entries(self, rtds_manager):
        """Test history can handle many entries efficiently."""
        base_ts = int(time.time() * 1000)

        # Add 1000 entries (every second for ~16 minutes)
        for i in range(1000):
            ts = base_ts - (i * 1000)  # 1 second apart
            rtds_manager.chainlink_price_history[ts] = 95000.0 + i

        # Should have many entries
        assert len(rtds_manager.chainlink_price_history) == 1000

        # Process new message to trigger cleanup
        new_ts = base_ts + 1000
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 96000.00,
                'timestamp': new_ts
            }
        }
        await rtds_manager._process_message(message)

        # All entries within 1 hour should be retained
        # (1000 seconds = ~16 min, well within 1 hour)
        assert len(rtds_manager.chainlink_price_history) >= 1000


# ========== Stale Data Detection Tests ==========
class TestStaleDataDetection:
    """Test stale data detection and health checks."""

    @pytest.mark.asyncio
    async def test_check_price_health_no_warning_when_fresh(self, rtds_manager):
        """Test no stale warning when data is fresh."""
        rtds_manager.current_chainlink_price = 95000.00
        rtds_manager._last_price_update_time = time.time()
        rtds_manager._stale_warning_shown = False

        await rtds_manager._check_price_health()

        assert rtds_manager._stale_warning_shown is False

    @pytest.mark.asyncio
    async def test_check_price_health_warns_when_stale(self, config, on_btc_price_callback):
        """Test stale warning is shown when data is old."""
        config.rtds_price_stale_threshold_seconds = 0.1  # 100ms threshold
        rtds_manager = RTDSManager(config, on_btc_price_callback)

        rtds_manager.current_chainlink_price = 95000.00
        rtds_manager._last_price_update_time = time.time() - 1.0  # 1 second ago
        rtds_manager._stale_warning_shown = False

        await rtds_manager._check_price_health()

        assert rtds_manager._stale_warning_shown is True

    @pytest.mark.asyncio
    async def test_stale_warning_resets_on_new_price(self, rtds_manager, on_btc_price_callback):
        """Test stale warning resets when new price arrives."""
        rtds_manager._stale_warning_shown = True

        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': int(time.time() * 1000)
            }
        }
        await rtds_manager._process_message(message)

        assert rtds_manager._stale_warning_shown is False

    @pytest.mark.asyncio
    async def test_check_price_health_skips_when_never_received(self, rtds_manager):
        """Test health check skips when no price ever received."""
        rtds_manager.current_chainlink_price = None
        rtds_manager.current_btc_price = None

        # Should not raise or set warning
        await rtds_manager._check_price_health()

        assert rtds_manager._stale_warning_shown is False


# ========== Connection Management Tests ==========
class TestConnectionManagement:
    """Test RTDS connection lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_connection_task(self, rtds_manager):
        """Test start() creates connection task."""
        await rtds_manager.start()

        assert rtds_manager.connection_task is not None
        assert not rtds_manager.connection_task.done()

        await rtds_manager.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, rtds_manager):
        """Test calling start() multiple times is safe."""
        await rtds_manager.start()
        task1 = rtds_manager.connection_task

        await rtds_manager.start()
        task2 = rtds_manager.connection_task

        assert task1 is task2

        await rtds_manager.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_flag(self, rtds_manager):
        """Test stop() sets shutdown flag."""
        await rtds_manager.start()
        assert not rtds_manager.shutdown_flag.is_set()

        await rtds_manager.stop()
        assert rtds_manager.shutdown_flag.is_set()

    @pytest.mark.asyncio
    async def test_stop_before_start_safe(self, rtds_manager):
        """Test calling stop() before start() is safe."""
        await rtds_manager.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_connection_status(self, rtds_manager):
        """Test get_connection_status returns correct info."""
        status = rtds_manager.get_connection_status()

        assert 'connected' in status
        assert 'has_chainlink_price' in status
        assert 'has_binance_price' in status
        assert 'history_entries' in status
        assert 'current_chainlink' in status
        assert 'current_binance' in status

    @pytest.mark.asyncio
    async def test_get_connection_status_with_prices(self, rtds_manager):
        """Test connection status reflects prices."""
        rtds_manager.current_chainlink_price = 95000.00
        rtds_manager.current_btc_price = 94500.00
        rtds_manager.chainlink_price_history[1700000000000] = 95000.00

        status = rtds_manager.get_connection_status()

        assert status['has_chainlink_price'] is True
        assert status['has_binance_price'] is True
        assert status['history_entries'] == 1
        assert status['current_chainlink'] == 95000.00
        assert status['current_binance'] == 94500.00


# ========== Reconnection Logic Tests ==========
class TestReconnectionLogic:
    """Test RTDS reconnection behavior."""

    @pytest.mark.asyncio
    async def test_reconnection_with_exponential_backoff(self, config, on_btc_price_callback):
        """Test reconnection uses exponential backoff."""
        rtds_manager = RTDSManager(config, on_btc_price_callback)

        connection_attempts = []

        def mock_connect_failing(*args, **kwargs):
            connection_attempts.append(asyncio.get_event_loop().time())
            raise ConnectionError("Connection failed")

        with patch('src.activetrader.engine.connect', side_effect=mock_connect_failing):
            await rtds_manager.start()

            # Wait for reconnection attempts
            await asyncio.sleep(1.5)

            await rtds_manager.stop()

        # Should have multiple attempts
        assert len(connection_attempts) >= 2

        # Verify exponential backoff
        if len(connection_attempts) >= 3:
            gap1 = connection_attempts[1] - connection_attempts[0]
            gap2 = connection_attempts[2] - connection_attempts[1]
            assert gap2 >= gap1

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts_respected(self, config, on_btc_price_callback):
        """Test reconnection stops after max attempts."""
        config.rtds_max_reconnect_attempts = 3
        rtds_manager = RTDSManager(config, on_btc_price_callback)

        connection_attempts = []

        async def mock_connect_always_fails(*args, **kwargs):
            connection_attempts.append(1)
            raise ConnectionError("Always fails")

        with patch('src.activetrader.engine.connect', side_effect=mock_connect_always_fails):
            await rtds_manager.start()

            await asyncio.sleep(2.0)

            await rtds_manager.stop()

        assert len(connection_attempts) <= config.rtds_max_reconnect_attempts

    @pytest.mark.asyncio
    async def test_invalid_uri_stops_reconnection(self, config, on_btc_price_callback):
        """Test InvalidURI exception stops reconnection quickly (not max attempts)."""
        config.rtds_max_reconnect_attempts = 10  # Set high to show it doesn't retry that many
        rtds_manager = RTDSManager(config, on_btc_price_callback)

        attempts = []

        def mock_invalid_uri(*args, **kwargs):
            attempts.append(1)
            raise InvalidURI("wss://invalid")

        with patch('src.activetrader.engine.connect', side_effect=mock_invalid_uri):
            await rtds_manager.start()

            await asyncio.sleep(0.5)

            await rtds_manager.stop()

        # Should stop early due to InvalidURI (not retry all 10 attempts)
        # May attempt 1-2 times due to task scheduling before break is processed
        assert len(attempts) <= 3


# ========== Error Handling Tests ==========
class TestErrorHandling:
    """Test error handling in RTDS operations."""

    @pytest.mark.asyncio
    async def test_malformed_message_handled_gracefully(self, rtds_manager, on_btc_price_callback):
        """Test malformed messages don't crash the manager."""
        # Missing required fields - these should be handled gracefully
        await rtds_manager._process_message({})
        await rtds_manager._process_message({'topic': None})
        await rtds_manager._process_message({'topic': 'unknown_topic'})
        await rtds_manager._process_message({'topic': 'crypto_prices_chainlink', 'payload': {}})
        await rtds_manager._process_message({'topic': 'crypto_prices', 'payload': {}})
        # Missing symbol/value in payload
        await rtds_manager._process_message({
            'topic': 'crypto_prices_chainlink',
            'payload': {'symbol': 'btc/usd'}  # Missing value
        })
        await rtds_manager._process_message({
            'topic': 'crypto_prices_chainlink',
            'payload': {'value': 95000.0}  # Missing symbol
        })

        # Should not have called callback
        on_btc_price_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, config):
        """Test errors in callback are caught and don't crash the manager."""
        callback_called = [False]

        async def failing_callback(price):
            callback_called[0] = True
            raise ValueError("Callback error")

        rtds_manager = RTDSManager(config, failing_callback)

        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': 1700000000000
            }
        }

        # Should not raise - the error is caught internally
        await rtds_manager._process_message(message)

        # Callback was called but error was caught
        assert callback_called[0] is True
        # Price should still be stored despite callback error
        assert rtds_manager.current_chainlink_price == 95000.00

    @pytest.mark.asyncio
    async def test_sync_callback_supported(self, config):
        """Test synchronous callbacks are supported."""
        prices_received = []

        def sync_callback(price):
            prices_received.append(price)

        rtds_manager = RTDSManager(config, sync_callback)

        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00,
                'timestamp': 1700000000000
            }
        }
        await rtds_manager._process_message(message)

        assert len(prices_received) == 1
        assert prices_received[0] == 95000.00


# ========== Accessor Methods Tests ==========
class TestAccessorMethods:
    """Test getter/accessor methods."""

    def test_get_chainlink_price(self, rtds_manager):
        """Test get_chainlink_price returns correct value."""
        rtds_manager.current_chainlink_price = 95000.00

        assert rtds_manager.get_chainlink_price() == 95000.00

    def test_get_chainlink_price_none(self, rtds_manager):
        """Test get_chainlink_price returns None when unavailable."""
        rtds_manager.current_chainlink_price = None

        assert rtds_manager.get_chainlink_price() is None


# ========== Edge Cases ==========
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_price_without_timestamp(self, rtds_manager, on_btc_price_callback):
        """Test price update without timestamp still updates current price."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95000.00
                # No timestamp
            }
        }

        await rtds_manager._process_message(message)

        # Should update current price
        assert rtds_manager.current_chainlink_price == 95000.00
        # But not add to history
        assert len(rtds_manager.chainlink_price_history) == 0

    @pytest.mark.asyncio
    async def test_very_large_price_handled(self, rtds_manager, on_btc_price_callback):
        """Test very large prices are handled."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 1000000000.0,  # 1 billion
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        on_btc_price_callback.assert_called_once_with(1000000000.0)

    @pytest.mark.asyncio
    async def test_decimal_precision_preserved(self, rtds_manager, on_btc_price_callback):
        """Test decimal precision is preserved in prices."""
        message = {
            'topic': 'crypto_prices_chainlink',
            'payload': {
                'symbol': 'btc/usd',
                'value': 95123.456789,
                'timestamp': 1700000000000
            }
        }

        await rtds_manager._process_message(message)

        assert rtds_manager.current_chainlink_price == 95123.456789

    @pytest.mark.asyncio
    async def test_rapid_price_updates(self, rtds_manager, on_btc_price_callback):
        """Test rapid sequential price updates are all processed."""
        for i in range(100):
            message = {
                'topic': 'crypto_prices_chainlink',
                'payload': {
                    'symbol': 'btc/usd',
                    'value': 95000.0 + i,
                    'timestamp': 1700000000000 + (i * 1000)
                }
            }
            await rtds_manager._process_message(message)

        assert on_btc_price_callback.call_count == 100
        assert rtds_manager.current_chainlink_price == 95099.0
        assert len(rtds_manager.chainlink_price_history) == 100

    @pytest.mark.asyncio
    async def test_concurrent_access_safe(self, rtds_manager, on_btc_price_callback):
        """Test concurrent message processing is safe."""
        async def send_price(price_offset):
            message = {
                'topic': 'crypto_prices_chainlink',
                'payload': {
                    'symbol': 'btc/usd',
                    'value': 95000.0 + price_offset,
                    'timestamp': 1700000000000 + (price_offset * 1000)
                }
            }
            await rtds_manager._process_message(message)

        # Process many messages concurrently
        await asyncio.gather(*[send_price(i) for i in range(50)])

        assert on_btc_price_callback.call_count == 50

    @pytest.mark.asyncio
    async def test_multiple_stops_safe(self, rtds_manager):
        """Test calling stop() multiple times is safe."""
        await rtds_manager.start()
        await rtds_manager.stop()
        await rtds_manager.stop()
        await rtds_manager.stop()

        # Should not raise

    @pytest.mark.asyncio
    async def test_restart_after_stop(self, rtds_manager):
        """Test manager can be restarted after stop."""
        await rtds_manager.start()
        await rtds_manager.stop()

        # Should be able to restart
        rtds_manager.shutdown_flag.clear()
        await rtds_manager.start()

        assert rtds_manager.connection_task is not None

        await rtds_manager.stop()
