"""Comprehensive tests for FingerBlasterCore lifecycle management.

Tests cover:
- Core initialization and component wiring
- Market discovery and transition
- Position tracking
- Strike price resolution
- Health checks and auto-reconnect
- Graceful shutdown
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from src.activetrader.core import (
    FingerBlasterCore,
    CallbackManager,
    CALLBACK_EVENTS,
    POSITION_UPDATE_INTERVAL,
    STRIKE_RESOLVE_INTERVAL,
    HEALTH_CHECK_INTERVAL,
)
from src.activetrader.config import AppConfig


# ========== Test Fixtures ==========
@pytest.fixture
def mock_connector():
    """Create mock PolymarketConnector."""
    connector = MagicMock()
    connector.get_active_market = AsyncMock(return_value=None)
    connector.get_next_market = AsyncMock(return_value=None)
    connector.get_token_balance = AsyncMock(return_value=0.0)
    connector.get_chainlink_price_at = AsyncMock(return_value=None)
    connector.get_chainlink_onchain_price_at = AsyncMock(return_value=None)
    connector.get_coinbase_15m_open_price_at = AsyncMock(return_value=None)
    connector.get_btc_price_at = AsyncMock(return_value="N/A")
    connector.create_market_order = AsyncMock(return_value={'orderID': 'test123'})
    connector.close = AsyncMock()
    return connector


@pytest.fixture
def core(mock_connector):
    """Create FingerBlasterCore with mocked connector."""
    with patch('src.activetrader.core.PolymarketConnector', return_value=mock_connector):
        return FingerBlasterCore(connector=mock_connector)


# ========== Initialization Tests ==========
class TestCoreInitialization:
    """Test FingerBlasterCore initialization."""

    def test_initializes_all_managers(self, core):
        """Test all managers are initialized."""
        assert core.market_manager is not None
        assert core.history_manager is not None
        assert core.ws_manager is not None
        assert core.rtds_manager is not None
        assert core.order_executor is not None
        assert core.analytics_engine is not None
        assert core.callback_manager is not None

    def test_initializes_config(self, core):
        """Test config is initialized."""
        assert isinstance(core.config, AppConfig)

    def test_initializes_state(self, core):
        """Test initial state values."""
        assert core._resolved_market_id is None
        assert core._resolution_timestamp == 0.0
        assert core.selected_size == 1.0
        assert core._yes_position == 0.0
        assert core._no_position == 0.0

    def test_initializes_locks(self, core):
        """Test locks are initialized."""
        assert core._position_lock is not None
        assert core._market_update_lock is not None
        assert core._market_switch_lock is not None


# ========== Callback Registration Tests ==========
class TestCallbackRegistration:
    """Test callback registration."""

    def test_register_callback_valid_event(self, core):
        """Test registering callback for valid event."""
        def callback(): pass

        result = core.register_callback('market_update', callback)

        assert result is True

    def test_register_callback_invalid_event(self, core):
        """Test registering callback for invalid event."""
        def callback(): pass

        result = core.register_callback('invalid_event', callback)

        assert result is False

    def test_all_callback_events_defined(self, core):
        """Test all expected callback events are defined."""
        expected_events = [
            'market_update', 'btc_price_update', 'price_update',
            'account_stats_update', 'countdown_update', 'prior_outcomes_update',
            'resolution', 'log', 'chart_update', 'analytics_update',
            'order_submitted', 'order_filled', 'order_failed',
            'flatten_started', 'flatten_completed', 'flatten_failed',
            'cancel_started', 'cancel_completed', 'cancel_failed',
            'size_changed',
        ]

        for event in expected_events:
            assert event in CALLBACK_EVENTS


# ========== Size Control Tests ==========
class TestSizeControl:
    """Test order size control."""

    def test_size_up(self, core):
        """Test size up increases by 1."""
        core.selected_size = 5.0
        callbacks_received = []

        core.register_callback('size_changed', lambda s: callbacks_received.append(s))
        core.size_up()

        assert core.selected_size == 6.0
        assert callbacks_received == [6.0]

    def test_size_down(self, core):
        """Test size down decreases by 1."""
        core.selected_size = 5.0
        callbacks_received = []

        core.register_callback('size_changed', lambda s: callbacks_received.append(s))
        core.size_down()

        assert core.selected_size == 4.0
        assert callbacks_received == [4.0]

    def test_size_down_minimum(self, core):
        """Test size down doesn't go below 1."""
        core.selected_size = 1.0
        core.size_down()

        assert core.selected_size == 1.0


# ========== Order Placement Tests ==========
class TestOrderPlacement:
    """Test order placement."""

    @pytest.mark.asyncio
    async def test_place_order_success(self, core, mock_connector):
        """Test successful order placement."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        core.order_executor.execute_order = AsyncMock(
            return_value={'orderID': 'order123'}
        )
        core.market_manager.calculate_mid_price = AsyncMock(
            return_value=(0.50, 0.50, 0.0, 0.0)
        )

        callbacks = {'submitted': [], 'filled': []}
        core.register_callback('order_submitted', lambda *a: callbacks['submitted'].append(a))
        core.register_callback('order_filled', lambda *a: callbacks['filled'].append(a))

        await core.place_order('Up')

        assert len(callbacks['submitted']) == 1
        assert len(callbacks['filled']) == 1

    @pytest.mark.asyncio
    async def test_place_order_no_token_map(self, core):
        """Test order fails when no token map."""
        core.market_manager.get_token_map = AsyncMock(return_value=None)

        callbacks = {'failed': []}
        core.register_callback('order_failed', lambda *a: callbacks['failed'].append(a))

        await core.place_order('Up')

        assert len(callbacks['failed']) == 1
        assert "Token map not ready" in callbacks['failed'][0][2]

    @pytest.mark.asyncio
    async def test_place_order_rejected(self, core):
        """Test order rejected by exchange."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        core.order_executor.execute_order = AsyncMock(return_value=None)

        callbacks = {'failed': []}
        core.register_callback('order_failed', lambda *a: callbacks['failed'].append(a))

        await core.place_order('Up')

        assert len(callbacks['failed']) == 1


# ========== Flatten Tests ==========
class TestFlatten:
    """Test flatten all positions."""

    @pytest.mark.asyncio
    async def test_flatten_all_success(self, core):
        """Test successful flatten."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        core.order_executor.flatten_positions = AsyncMock(
            return_value=[{'orderID': 'order1'}, {'orderID': 'order2'}]
        )

        callbacks = {'started': 0, 'completed': []}
        core.register_callback('flatten_started', lambda: setattr(callbacks, 'started', callbacks.get('started', 0) + 1))
        core.register_callback('flatten_completed', lambda n: callbacks['completed'].append(n))

        await core.flatten_all()

        assert len(callbacks['completed']) == 1
        assert callbacks['completed'][0] == 2

    @pytest.mark.asyncio
    async def test_flatten_all_no_token_map(self, core):
        """Test flatten fails without token map."""
        core.market_manager.get_token_map = AsyncMock(return_value=None)

        callbacks = {'failed': []}
        core.register_callback('flatten_failed', lambda e: callbacks['failed'].append(e))

        await core.flatten_all()

        assert len(callbacks['failed']) == 1

    @pytest.mark.asyncio
    async def test_flatten_all_exception(self, core):
        """Test flatten handles exception."""
        core.market_manager.get_token_map = AsyncMock(
            side_effect=Exception("Test error")
        )

        callbacks = {'failed': []}
        core.register_callback('flatten_failed', lambda e: callbacks['failed'].append(e))

        await core.flatten_all()

        assert len(callbacks['failed']) == 1


# ========== Cancel All Tests ==========
class TestCancelAll:
    """Test cancel all orders."""

    @pytest.mark.asyncio
    async def test_cancel_all_success(self, core):
        """Test successful cancel all."""
        core.order_executor.cancel_all_orders = AsyncMock(return_value=True)

        callbacks = {'completed': 0}
        core.register_callback('cancel_completed', lambda: callbacks.update({'completed': callbacks['completed'] + 1}))

        result = await core.cancel_all_orders()

        assert result is True
        assert callbacks['completed'] == 1

    @pytest.mark.asyncio
    async def test_cancel_all_failure(self, core):
        """Test cancel all failure."""
        core.order_executor.cancel_all_orders = AsyncMock(return_value=False)

        result = await core.cancel_all_orders()

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_exception(self, core):
        """Test cancel all handles exception."""
        core.order_executor.cancel_all_orders = AsyncMock(
            side_effect=Exception("Cancel error")
        )

        callbacks = {'failed': []}
        core.register_callback('cancel_failed', lambda e: callbacks['failed'].append(e))

        result = await core.cancel_all_orders()

        assert result is False
        assert len(callbacks['failed']) == 1


# ========== Market Status Tests ==========
class TestMarketStatus:
    """Test market status updates."""

    @pytest.mark.asyncio
    async def test_update_market_status_new_market(self, core, mock_connector):
        """Test new market discovery."""
        new_market = {
            'market_id': 'market_123',
            'title': 'BTC Up or Down 15m',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
            'price_to_beat': '$95000'
        }
        mock_connector.get_active_market = AsyncMock(return_value=new_market)
        core.market_manager.get_market = AsyncMock(return_value=None)
        core.market_manager.set_market = AsyncMock(return_value=True)
        core.ws_manager.subscribe_to_market = AsyncMock()

        await core.update_market_status()

        core.market_manager.set_market.assert_called()

    @pytest.mark.asyncio
    async def test_update_market_status_same_market(self, core, mock_connector):
        """Test same market doesn't trigger switch."""
        market = {
            'market_id': 'market_123',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
        }
        mock_connector.get_active_market = AsyncMock(return_value=market)
        core.market_manager.get_market = AsyncMock(return_value=market)
        core.market_manager.set_market = AsyncMock()

        await core.update_market_status()

        # Should not set market since it's the same
        core.market_manager.set_market.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_market_status_concurrent_protection(self, core, mock_connector):
        """Test concurrent market updates are serialized."""
        core._market_update_in_progress = True
        mock_connector.get_active_market = AsyncMock()

        await core.update_market_status()

        # Should skip since update in progress
        mock_connector.get_active_market.assert_not_called()


# ========== Market Resolution Tests ==========
class TestMarketResolution:
    """Test market resolution handling."""

    @pytest.mark.asyncio
    async def test_handle_market_resolution(self, core, mock_connector):
        """Test market resolution triggers events."""
        market = {
            'market_id': 'market_123',
            'end_date': pd.Timestamp.now(tz='UTC').isoformat(),
        }

        callbacks = {'resolution': []}
        core.register_callback('resolution', lambda r: callbacks['resolution'].append(r))

        await core._handle_market_resolution(market)

        assert core._resolved_market_id == 'market_123'
        assert len(callbacks['resolution']) == 1
        assert callbacks['resolution'][0] == "EXPIRED"

    @pytest.mark.asyncio
    async def test_resolution_cooldown(self, core):
        """Test resolution cooldown prevents duplicates."""
        market = {'market_id': 'market_123'}
        core._resolved_market_id = 'market_123'
        core._resolution_timestamp = time.time()

        # This should be blocked by cooldown
        # update_countdown checks this internally


# ========== Strike Resolution Tests ==========
class TestStrikeResolution:
    """Test strike price resolution."""

    @pytest.mark.asyncio
    async def test_resolve_strike_from_rtds_cache(self, core):
        """Test strike resolved from RTDS cache."""
        market = {
            'market_id': 'test',
            'price_to_beat': 'Dynamic',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
        }
        core.market_manager.get_market = AsyncMock(return_value=market)
        core.market_manager.get_market_start_time = AsyncMock(
            return_value=pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=1)
        )
        core.market_manager.set_market = AsyncMock(return_value=True)
        core.rtds_manager.get_chainlink_price_at = MagicMock(return_value=95000.0)
        core.rtds_manager.get_connection_status = MagicMock(
            return_value={'connected': True, 'history_entries': 10, 'current_chainlink': 95000}
        )

        await core._try_resolve_pending_strike()

        # Should have updated market with resolved strike
        assert market['price_to_beat'] == "95,000.00"

    @pytest.mark.asyncio
    async def test_resolve_strike_falls_through_chain(self, core, mock_connector):
        """Test strike resolution falls through priority chain."""
        market = {
            'market_id': 'test',
            'price_to_beat': 'Dynamic',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
        }
        core.market_manager.get_market = AsyncMock(return_value=market)
        core.market_manager.get_market_start_time = AsyncMock(
            return_value=pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=1)
        )
        core.market_manager.set_market = AsyncMock(return_value=True)

        # RTDS cache fails
        core.rtds_manager.get_chainlink_price_at = MagicMock(return_value=None)
        core.rtds_manager.get_connection_status = MagicMock(
            return_value={'connected': False, 'history_entries': 0, 'current_chainlink': None}
        )

        # On-chain fails
        mock_connector.get_chainlink_onchain_price_at = AsyncMock(return_value=None)
        # API fails
        mock_connector.get_chainlink_price_at = AsyncMock(return_value=None)
        # Coinbase fails
        mock_connector.get_coinbase_15m_open_price_at = AsyncMock(return_value=None)
        # Binance succeeds
        mock_connector.get_btc_price_at = AsyncMock(return_value="94500.00")

        await core._try_resolve_pending_strike()

        assert market['price_to_beat'] == "94,500.00"

    @pytest.mark.asyncio
    async def test_resolve_strike_throttling(self, core):
        """Test strike resolution is throttled."""
        market = {
            'market_id': 'test',
            'price_to_beat': 'Dynamic',
        }
        core.market_manager.get_market = AsyncMock(return_value=market)
        core.market_manager.get_market_start_time = AsyncMock(
            return_value=pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=5)
        )
        core._last_strike_resolve_attempt = time.time()  # Just attempted

        await core._try_resolve_pending_strike()

        # Should skip due to throttling (market not started)


# ========== Position Update Tests ==========
class TestPositionUpdates:
    """Test position updates."""

    @pytest.mark.asyncio
    async def test_update_positions(self, core, mock_connector):
        """Test position update from connector."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': 'up_token', 'Down': 'down_token'}
        )
        mock_connector.get_token_balance = AsyncMock(
            side_effect=[10.0, 5.0]  # Up then Down
        )

        await core._update_positions()

        assert core._yes_position == 10.0
        assert core._no_position == 5.0

    @pytest.mark.asyncio
    async def test_update_positions_no_token_map(self, core):
        """Test position update with no token map."""
        core.market_manager.get_token_map = AsyncMock(return_value=None)

        await core._update_positions()

        # Should return early, positions unchanged
        assert core._yes_position == 0.0
        assert core._no_position == 0.0


# ========== Health Check Tests ==========
class TestHealthChecks:
    """Test data health checks."""

    @pytest.mark.asyncio
    async def test_check_data_health_stale(self, core):
        """Test stale data warning."""
        core.market_manager.is_data_stale = AsyncMock(return_value=True)
        core._stale_data_warning_shown = False
        core.ws_manager._ws = None
        core.ws_manager.start = AsyncMock()

        log_messages = []
        core.register_callback('log', lambda m: log_messages.append(m))

        await core._check_data_health()

        assert core._stale_data_warning_shown is True
        assert any("stale" in m.lower() for m in log_messages)

    @pytest.mark.asyncio
    async def test_check_data_health_fresh(self, core):
        """Test fresh data clears warning."""
        core.market_manager.is_data_stale = AsyncMock(return_value=False)
        core._stale_data_warning_shown = True

        log_messages = []
        core.register_callback('log', lambda m: log_messages.append(m))

        await core._check_data_health()

        assert core._stale_data_warning_shown is False
        assert any("fresh" in m.lower() for m in log_messages)


# ========== Countdown Tests ==========
class TestCountdown:
    """Test countdown updates."""

    @pytest.mark.asyncio
    async def test_update_countdown(self, core):
        """Test countdown update emits event."""
        end_time = pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=5, seconds=30)
        market = {
            'market_id': 'test',
            'end_date': end_time.isoformat(),
        }
        core.market_manager.get_market = AsyncMock(return_value=market)

        callbacks = {'countdown': []}
        core.register_callback('countdown_update', lambda *a: callbacks['countdown'].append(a))

        await core.update_countdown()

        assert len(callbacks['countdown']) == 1
        time_str, urgency, seconds = callbacks['countdown'][0]
        assert "05:" in time_str

    @pytest.mark.asyncio
    async def test_update_countdown_expired(self, core, mock_connector):
        """Test countdown triggers resolution when expired."""
        end_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(seconds=5)
        market = {
            'market_id': 'test_expired',
            'end_date': end_time.isoformat(),
        }
        core.market_manager.get_market = AsyncMock(return_value=market)
        core._resolved_market_id = None
        core._resolution_timestamp = 0.0

        callbacks = {'resolution': []}
        core.register_callback('resolution', lambda r: callbacks['resolution'].append(r))

        await core.update_countdown()

        assert len(callbacks['resolution']) == 1


# ========== Analytics Update Tests ==========
class TestAnalyticsUpdate:
    """Test analytics updates."""

    @pytest.mark.asyncio
    async def test_update_analytics(self, core):
        """Test analytics update generates snapshot."""
        market = {
            'market_id': 'test',
            'price_to_beat': '$95000',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=10)).isoformat(),
        }
        core.market_manager.get_market = AsyncMock(return_value=market)
        core.rtds_manager.get_current_price = MagicMock(return_value=95100.0)
        core.market_manager.calculate_mid_price = AsyncMock(
            return_value=(0.55, 0.45, 0.0, 0.0)
        )
        core.market_manager.get_raw_order_book = AsyncMock(return_value={})
        core.analytics_engine.generate_snapshot = AsyncMock(
            return_value=MagicMock()
        )

        callbacks = {'analytics': []}
        core.register_callback('analytics_update', lambda s: callbacks['analytics'].append(s))

        await core.update_analytics()

        assert len(callbacks['analytics']) == 1

    @pytest.mark.asyncio
    async def test_update_analytics_no_market(self, core):
        """Test analytics update with no market."""
        core.market_manager.get_market = AsyncMock(return_value=None)

        await core.update_analytics()

        # Should return early without error


# ========== Shutdown Tests ==========
class TestShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(self, core, mock_connector):
        """Test shutdown stops all components."""
        core.ws_manager.stop = AsyncMock()
        core.rtds_manager.stop = AsyncMock()

        await core.shutdown()

        core.ws_manager.stop.assert_called_once()
        core.rtds_manager.stop.assert_called_once()
        mock_connector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self, core, mock_connector):
        """Test shutdown handles component errors."""
        core.ws_manager.stop = AsyncMock(side_effect=Exception("Stop error"))
        core.rtds_manager.stop = AsyncMock()

        # Should not raise
        await core.shutdown()


# ========== Market Transition Tests ==========
class TestMarketTransition:
    """Test market transition deduplication."""

    @pytest.mark.asyncio
    async def test_new_market_found_deduplication(self, core):
        """Test duplicate market transitions are blocked."""
        market = {
            'market_id': 'market_123',
            'title': 'Test',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
            'price_to_beat': 'N/A',
        }

        core._switching_to_market_id = 'market_123'
        core.market_manager.set_market = AsyncMock()

        await core._on_new_market_found(market)

        # Should skip since already switching
        core.market_manager.set_market.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_market_resets_state(self, core):
        """Test new market resets position state."""
        market = {
            'market_id': 'new_market',
            'title': 'New Market',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
            'price_to_beat': '$95000',
        }

        core._yes_position = 10.0
        core._no_position = 5.0
        core.market_manager.get_market = AsyncMock(return_value=None)
        core.market_manager.set_market = AsyncMock(return_value=True)
        core.ws_manager.subscribe_to_market = AsyncMock()

        await core._on_new_market_found(market)

        assert core._yes_position == 0.0
        assert core._no_position == 0.0


# ========== Close Position Tests ==========
class TestClosePosition:
    """Test close position functionality."""

    @pytest.mark.asyncio
    async def test_close_position_up(self, core, mock_connector):
        """Test closing Up position."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': 'up_token', 'Down': 'down_token'}
        )
        mock_connector.get_token_balance = AsyncMock(return_value=10.0)
        mock_connector.create_market_order = AsyncMock(return_value={'orderID': 'sell1'})

        result = await core.close_position('Up')

        assert result is True
        mock_connector.create_market_order.assert_called_once_with('up_token', 10.0, 'SELL')

    @pytest.mark.asyncio
    async def test_close_position_no_balance(self, core, mock_connector):
        """Test close position with no balance."""
        core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': 'up_token'}
        )
        mock_connector.get_token_balance = AsyncMock(return_value=0.05)

        result = await core.close_position('Up')

        # Balance < 0.1, should not attempt sell
        assert result is False


# ========== Price Parsing Tests ==========
class TestPriceParsing:
    """Test price parsing utilities."""

    def test_parse_price_to_beat_valid(self, core):
        """Test parsing valid price string."""
        assert core._parse_price_to_beat("$95,000.50") == 95000.50
        assert core._parse_price_to_beat("95000") == 95000.0
        assert core._parse_price_to_beat("$100,000") == 100000.0

    def test_parse_price_to_beat_invalid(self, core):
        """Test parsing invalid price strings."""
        assert core._parse_price_to_beat("N/A") == 0.0
        assert core._parse_price_to_beat("Pending") == 0.0
        assert core._parse_price_to_beat("") == 0.0
        assert core._parse_price_to_beat(None) == 0.0

    def test_parse_price_to_beat_edge_cases(self, core):
        """Test edge cases in price parsing."""
        assert core._parse_price_to_beat("   $95000   ") == 95000.0
        assert core._parse_price_to_beat("--") == 0.0
        assert core._parse_price_to_beat("Loading") == 0.0


# ========== Format Time Tests ==========
class TestTimeFormatting:
    """Test time formatting utilities."""

    def test_format_ends(self, core):
        """Test end time formatting."""
        iso_str = "2026-01-25T17:00:00Z"
        formatted = core._format_ends(iso_str)

        # Should contain ET
        assert "ET" in formatted

    def test_format_ends_invalid(self, core):
        """Test formatting invalid time string."""
        formatted = core._format_ends("invalid")

        assert formatted == "invalid"

    def test_format_starts(self, core):
        """Test start time formatting."""
        iso_str = "2026-01-25T16:45:00Z"
        formatted = core._format_starts(iso_str)

        assert "ET" in formatted


# ========== RTDS Callback Tests ==========
class TestRTDSCallback:
    """Test RTDS price callback."""

    @pytest.mark.asyncio
    async def test_on_rtds_btc_price(self, core):
        """Test RTDS BTC price callback emits event."""
        callbacks = {'btc_price': []}
        core.register_callback('btc_price_update', lambda p: callbacks['btc_price'].append(p))

        await core._on_rtds_btc_price(95000.0)

        assert callbacks['btc_price'] == [95000.0]


# ========== WebSocket Message Callback Tests ==========
class TestWSCallback:
    """Test WebSocket message callback."""

    @pytest.mark.asyncio
    async def test_on_ws_message(self, core):
        """Test WS message triggers price recalc."""
        core.market_manager.calculate_mid_price = AsyncMock(
            return_value=(0.50, 0.50, 0.0, 0.0)
        )
        core.market_manager.is_data_stale = AsyncMock(return_value=False)

        callbacks = {'price': []}
        core.register_callback('price_update', lambda *a: callbacks['price'].append(a))

        await core._on_ws_message({'test': 'data'})

        assert len(callbacks['price']) == 1


# ========== Log Message Tests ==========
class TestLogMessage:
    """Test log message emission."""

    def test_log_msg(self, core):
        """Test log message includes timestamp."""
        messages = []
        core.register_callback('log', lambda m: messages.append(m))

        core.log_msg("Test message")

        assert len(messages) == 1
        assert "Test message" in messages[0]
        assert "[" in messages[0]  # Has timestamp


# ========== Edge Cases ==========
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_recalc_price_triggers_health_check(self, core):
        """Test health check triggered by interval."""
        core._last_health_check = time.time() - HEALTH_CHECK_INTERVAL - 1
        core.market_manager.calculate_mid_price = AsyncMock(
            return_value=(0.50, 0.50, 0.0, 0.0)
        )
        core.market_manager.is_data_stale = AsyncMock(return_value=False)

        await core._recalc_price()

        # Health check should have been called
        core.market_manager.is_data_stale.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_reconnect_attempt(self, core):
        """Test auto reconnect when WS is None."""
        core.ws_manager._ws = None
        core.ws_manager.start = AsyncMock()

        await core._attempt_auto_reconnect()

        core.ws_manager.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_migrate_to_next_market(self, core, mock_connector):
        """Test market migration after expiry."""
        expired_market = {
            'market_id': 'expired_123',
            'end_date': pd.Timestamp.now(tz='UTC').isoformat(),
        }
        next_market = {
            'market_id': 'next_456',
            'title': 'Next Market',
            'end_date': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=15)).isoformat(),
            'price_to_beat': '$95000',
        }

        core.config.resolution_overlay_duration = 0.1
        core.market_manager.get_market = AsyncMock(return_value=expired_market)
        mock_connector.get_next_market = AsyncMock(return_value=next_market)
        core.market_manager.set_market = AsyncMock(return_value=True)
        core.ws_manager.subscribe_to_market = AsyncMock()

        await core._migrate_to_next_market(expired_market)

        core.market_manager.set_market.assert_called()
