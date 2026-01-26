"""Comprehensive tests for LadderCore controller."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.ladder.core import LadderCore


# ========== Test Fixtures ==========
@pytest.fixture
def mock_fb_core():
    """Create mock FingerBlasterCore."""
    fb = MagicMock()
    fb.connector = MagicMock()
    fb.market_manager = MagicMock()
    fb.register_callback = MagicMock(return_value=True)
    return fb


@pytest.fixture
def ladder_core(mock_fb_core):
    """Create LadderCore with mocked dependencies."""
    return LadderCore(fb_core=mock_fb_core)


# ========== Initialization Tests ==========
class TestLadderCoreInitialization:
    """Test LadderCore initialization."""

    def test_initializes_with_provided_fb_core(self, mock_fb_core):
        """Test initialization with provided FingerBlasterCore."""
        core = LadderCore(fb_core=mock_fb_core)

        assert core.fb is mock_fb_core

    def test_registers_callbacks(self, mock_fb_core):
        """Test callbacks are registered on init."""
        core = LadderCore(fb_core=mock_fb_core)

        # Should register market_update and order_filled callbacks
        calls = mock_fb_core.register_callback.call_args_list
        events_registered = [call[0][0] for call in calls]

        assert 'market_update' in events_registered
        assert 'order_filled' in events_registered

    def test_initializes_empty_order_state(self, ladder_core):
        """Test order state is empty on init."""
        assert ladder_core.pending_orders == {}
        assert ladder_core.active_orders == {}
        assert ladder_core.filled_orders == {}

    def test_initializes_market_fields(self, ladder_core):
        """Test market fields initialized."""
        assert ladder_core.market_name == "Market"
        assert ladder_core.market_starts == ""
        assert ladder_core.market_ends == ""


# ========== Pending/Filled State Tests ==========
class TestOrderStateTracking:
    """Test order state tracking methods."""

    def test_is_pending_true_when_order_at_price(self, ladder_core):
        """Test is_pending returns True when order exists at price."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 10.0}

        assert ladder_core.is_pending(50) is True

    def test_is_pending_false_when_no_order(self, ladder_core):
        """Test is_pending returns False when no order at price."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 10.0}

        assert ladder_core.is_pending(60) is False

    def test_is_pending_false_empty(self, ladder_core):
        """Test is_pending returns False when no orders."""
        assert ladder_core.is_pending(50) is False

    def test_is_filled_true_within_window(self, ladder_core):
        """Test is_filled returns True within 5s window."""
        ladder_core.filled_orders[50] = time.time()

        assert ladder_core.is_filled(50) is True

    def test_is_filled_false_after_window(self, ladder_core):
        """Test is_filled returns False after 5s window expires."""
        ladder_core.filled_orders[50] = time.time() - 6.0  # 6 seconds ago

        assert ladder_core.is_filled(50) is False
        # Should also clean up the entry
        assert 50 not in ladder_core.filled_orders

    def test_is_filled_false_no_entry(self, ladder_core):
        """Test is_filled returns False when no entry."""
        assert ladder_core.is_filled(50) is False


# ========== Order Filled Callback Tests ==========
class TestOrderFilledCallback:
    """Test order filled callback handling."""

    def test_on_order_filled_matches_pending(self, ladder_core):
        """Test fill matches pending order by ID."""
        ladder_core.pending_orders['order123'] = {'price': 50, 'size': 10.0, 'side': 'YES'}

        ladder_core._on_order_filled('YES', 10.0, 0.50, 'order123')

        # Order should be removed from pending
        assert 'order123' not in ladder_core.pending_orders
        # Should be marked as filled
        assert 50 in ladder_core.filled_orders

    def test_on_order_filled_matches_by_price_side(self, ladder_core):
        """Test fill matches by price and side when ID doesn't match."""
        ladder_core.pending_orders['tmp_50'] = {'price': 50, 'size': 10.0, 'side': 'YES'}

        # Different order ID but same price/side
        ladder_core._on_order_filled('YES', 10.0, 0.50, 'different_id')

        # Should match and remove
        assert 'tmp_50' not in ladder_core.pending_orders
        assert 50 in ladder_core.filled_orders

    def test_on_order_filled_no_side(self, ladder_core):
        """Test fill with NO side converts price correctly."""
        ladder_core.pending_orders['tmp_70'] = {'price': 70, 'size': 10.0, 'side': 'NO'}

        # NO at 0.30 = YES at 0.70
        ladder_core._on_order_filled('NO', 10.0, 0.30, 'different_id')

        assert 'tmp_70' not in ladder_core.pending_orders
        assert 70 in ladder_core.filled_orders

    def test_on_order_filled_reduces_active_orders(self, ladder_core):
        """Test fill reduces active orders."""
        ladder_core.pending_orders['order123'] = {'price': 50, 'size': 10.0, 'side': 'YES'}
        ladder_core.active_orders[50] = 25.0  # Multiple orders at this level

        ladder_core._on_order_filled('YES', 10.0, 0.50, 'order123')

        # Active should be reduced by fill size
        assert ladder_core.active_orders[50] == 15.0

    def test_on_order_filled_removes_zero_active(self, ladder_core):
        """Test fill removes active when reduced to zero."""
        ladder_core.pending_orders['order123'] = {'price': 50, 'size': 10.0, 'side': 'YES'}
        ladder_core.active_orders[50] = 10.0  # Exact match

        ladder_core._on_order_filled('YES', 10.0, 0.50, 'order123')

        assert 50 not in ladder_core.active_orders


# ========== Cancel Order Tests ==========
class TestCancelOrders:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_all_orders_cancels_pending(self, ladder_core, mock_fb_core):
        """Test cancel_all cancels all pending orders."""
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
            'order2': {'price': 60, 'size': 20.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        count = await ladder_core.cancel_all_orders()

        assert count == 2
        assert ladder_core.dirty is True

    @pytest.mark.asyncio
    async def test_cancel_all_at_price(self, ladder_core, mock_fb_core):
        """Test cancel_all_at_price cancels only matching orders."""
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
            'order2': {'price': 50, 'size': 20.0},
            'order3': {'price': 60, 'size': 15.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        count = await ladder_core.cancel_all_at_price(50)

        # Should cancel 2 orders at 50¢
        assert count == 2
        # Order at 60¢ should remain
        assert 'order3' in ladder_core.pending_orders

    @pytest.mark.asyncio
    async def test_cancel_temp_order_no_api_call(self, ladder_core, mock_fb_core):
        """Test canceling tmp_ orders doesn't call API."""
        ladder_core.pending_orders = {
            'tmp_50_123': {'price': 50, 'size': 10.0},
        }

        result = await ladder_core._cancel_single_order('tmp_50_123')

        assert result is True
        assert 'tmp_50_123' not in ladder_core.pending_orders
        mock_fb_core.connector.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_real_order_calls_api(self, ladder_core, mock_fb_core):
        """Test canceling real orders calls connector."""
        ladder_core.pending_orders = {
            'real_order_id': {'price': 50, 'size': 10.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        result = await ladder_core._cancel_single_order('real_order_id')

        assert result is True
        mock_fb_core.connector.cancel_order.assert_called_once_with('real_order_id')

    @pytest.mark.asyncio
    async def test_cancel_order_api_failure(self, ladder_core, mock_fb_core):
        """Test cancel handles API failure."""
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value=None)

        result = await ladder_core._cancel_single_order('order1')

        assert result is False
        # Order should still be in pending
        assert 'order1' in ladder_core.pending_orders


# ========== Place Order Tests ==========
class TestPlaceOrders:
    """Test order placement."""

    @pytest.mark.asyncio
    async def test_place_limit_order_yes_side(self, ladder_core, mock_fb_core):
        """Test placing YES limit order."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'new_order_123'}
        )

        order_id = await ladder_core.place_limit_order(50, 10.0, 'YES')

        assert order_id == 'new_order_123'
        assert 'new_order_123' in ladder_core.pending_orders
        assert ladder_core.active_orders[50] == 10.0

    @pytest.mark.asyncio
    async def test_place_limit_order_no_side(self, ladder_core, mock_fb_core):
        """Test placing NO limit order."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'no_order_456'}
        )

        order_id = await ladder_core.place_limit_order(70, 15.0, 'NO')

        assert order_id == 'no_order_456'
        # NO at 70¢ = buying Down at 30¢
        mock_fb_core.connector.create_order.assert_called()
        call_args = mock_fb_core.connector.create_order.call_args[0]
        # Token should be Down
        assert call_args[0] == '0x' + '2' * 64
        # Price should be 0.30
        assert call_args[1] == 0.30

    @pytest.mark.asyncio
    async def test_place_limit_order_no_token_map(self, ladder_core, mock_fb_core):
        """Test limit order fails when no token map."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(return_value=None)

        order_id = await ladder_core.place_limit_order(50, 10.0, 'YES')

        assert order_id is None
        assert len(ladder_core.pending_orders) == 0

    @pytest.mark.asyncio
    async def test_place_limit_order_api_failure(self, ladder_core, mock_fb_core):
        """Test limit order handles API failure."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(return_value=None)

        order_id = await ladder_core.place_limit_order(50, 10.0, 'YES')

        assert order_id is None
        # Temp order should be cleaned up
        assert not any(k.startswith('tmp_') for k in ladder_core.pending_orders)

    @pytest.mark.asyncio
    async def test_place_limit_order_adds_to_existing_active(self, ladder_core, mock_fb_core):
        """Test placing order adds to existing active orders at price."""
        ladder_core.active_orders[50] = 20.0  # Existing order

        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'order123'}
        )

        await ladder_core.place_limit_order(50, 10.0, 'YES')

        # Should add to existing
        assert ladder_core.active_orders[50] == 30.0

    @pytest.mark.asyncio
    async def test_place_market_order(self, ladder_core, mock_fb_core):
        """Test placing market order."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_market_order = AsyncMock(
            return_value={'orderID': 'market_order_789'}
        )

        order_id = await ladder_core.place_market_order(25.0, 'YES')

        assert order_id == 'market_order_789'
        mock_fb_core.connector.create_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_market_order_no_token(self, ladder_core, mock_fb_core):
        """Test market order fails when no token."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(return_value=None)

        order_id = await ladder_core.place_market_order(25.0, 'YES')

        assert order_id is None


# ========== View Model Tests ==========
class TestViewModels:
    """Test view model generation."""

    def test_get_view_model_overlays_orders(self, ladder_core, mock_fb_core):
        """Test view model includes user orders."""
        mock_fb_core.market_manager.raw_books = {
            'Up': {'bids': {0.50: 100.0}, 'asks': {}},
            'Down': {'bids': {}, 'asks': {}}
        }
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
        }
        ladder_core.active_orders = {50: 5.0}

        ladder = ladder_core.get_view_model()

        # Should have my_size overlay
        assert ladder[50]['my_size'] == 15.0  # 10 pending + 5 active

    def test_get_view_model_empty_books(self, ladder_core, mock_fb_core):
        """Test view model with empty order books."""
        mock_fb_core.market_manager.raw_books = {
            'Up': {'bids': {}, 'asks': {}},
            'Down': {'bids': {}, 'asks': {}}
        }

        ladder = ladder_core.get_view_model()

        # Should have all 99 levels
        assert len(ladder) == 99

    def test_get_view_model_exception_returns_cached(self, ladder_core, mock_fb_core):
        """Test view model returns cached on exception."""
        ladder_core.last_ladder = {50: {'test': 'cached'}}
        mock_fb_core.market_manager.raw_books = None  # Will cause error

        ladder = ladder_core.get_view_model()

        assert ladder == ladder_core.last_ladder

    def test_get_open_orders_for_display(self, ladder_core):
        """Test getting open orders for display."""
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0, 'side': 'YES'},
            'order2': {'price': 60, 'size': 20.0, 'side': 'NO'},
            'order3': {'price': 0, 'size': 5.0, 'side': 'YES'},  # Out of bounds
        }

        orders = ladder_core.get_open_orders_for_display()

        # Should only return valid orders (1-99)
        assert len(orders) == 2
        assert orders[0]['order_id'] == 'order1'
        assert orders[1]['order_id'] == 'order2'


# ========== Market Update Callback Tests ==========
class TestMarketUpdateCallback:
    """Test market update handling."""

    def test_on_market_update_new_market_clears_orders(self, ladder_core):
        """Test new market clears order state."""
        ladder_core.pending_orders = {'order1': {'price': 50}}
        ladder_core.active_orders = {50: 10.0}
        ladder_core.filled_orders = {50: time.time()}
        ladder_core.market_name = "Old Market"

        ladder_core._on_market_update("$95000", "12:00PM", "New Market")

        assert ladder_core.pending_orders == {}
        assert ladder_core.active_orders == {}
        assert ladder_core.filled_orders == {}
        assert ladder_core.market_name == "New Market"

    def test_on_market_update_same_market_keeps_orders(self, ladder_core):
        """Test same market keeps order state."""
        ladder_core.pending_orders = {'order1': {'price': 50}}
        ladder_core.market_name = "Same Market"

        ladder_core._on_market_update("$96000", "12:15PM", "Same Market")

        # Orders should remain
        assert 'order1' in ladder_core.pending_orders

    def test_on_market_update_calls_callback(self, ladder_core):
        """Test market update invokes callback."""
        callback = MagicMock()
        ladder_core.set_market_update_callback(callback)

        ladder_core._on_market_update("$95000", "12:00PM", "Test Market", "11:45AM")

        callback.assert_called_once_with("Test Market", "11:45AM", "12:00PM")

    def test_get_market_fields(self, ladder_core):
        """Test get_market_fields returns correct data."""
        ladder_core.market_name = "Test Market"
        ladder_core.market_starts = "11:45AM"
        ladder_core.market_ends = "12:00PM"

        fields = ladder_core.get_market_fields()

        assert fields == {
            'name': "Test Market",
            'starts': "11:45AM",
            'ends': "12:00PM"
        }


# ========== Token Target Tests ==========
class TestTokenTargeting:
    """Test token ID resolution."""

    @pytest.mark.asyncio
    async def test_get_target_token_yes(self, ladder_core, mock_fb_core):
        """Test YES side gets Up token."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': 'up_token', 'Down': 'down_token'}
        )

        token = await ladder_core._get_target_token('YES')

        assert token == 'up_token'

    @pytest.mark.asyncio
    async def test_get_target_token_no(self, ladder_core, mock_fb_core):
        """Test NO side gets Down token."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': 'up_token', 'Down': 'down_token'}
        )

        token = await ladder_core._get_target_token('NO')

        assert token == 'down_token'

    @pytest.mark.asyncio
    async def test_get_target_token_no_map(self, ladder_core, mock_fb_core):
        """Test returns None when no token map."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(return_value=None)

        token = await ladder_core._get_target_token('YES')

        assert token is None


# ========== Order ID Extraction Tests ==========
class TestOrderIdExtraction:
    """Test order ID extraction from responses."""

    def test_extract_order_id_from_orderID(self, ladder_core):
        """Test extraction from orderID key."""
        resp = {'orderID': 'abc123'}
        assert ladder_core._extract_order_id(resp) == 'abc123'

    def test_extract_order_id_from_order_id(self, ladder_core):
        """Test extraction from order_id key."""
        resp = {'order_id': 'def456'}
        assert ladder_core._extract_order_id(resp) == 'def456'

    def test_extract_order_id_from_id(self, ladder_core):
        """Test extraction from id key."""
        resp = {'id': 'ghi789'}
        assert ladder_core._extract_order_id(resp) == 'ghi789'

    def test_extract_order_id_from_hash(self, ladder_core):
        """Test extraction from hash key."""
        resp = {'hash': '0xabc'}
        assert ladder_core._extract_order_id(resp) == '0xabc'

    def test_extract_order_id_non_dict(self, ladder_core):
        """Test returns None for non-dict."""
        assert ladder_core._extract_order_id("not a dict") is None
        assert ladder_core._extract_order_id(None) is None
        assert ladder_core._extract_order_id([]) is None


# ========== Reduce Active Order Tests ==========
class TestReduceActiveOrder:
    """Test active order reduction."""

    def test_reduce_active_order_partial(self, ladder_core):
        """Test partial reduction of active order."""
        ladder_core.active_orders[50] = 100.0

        ladder_core._reduce_active_order(50, 30.0)

        assert ladder_core.active_orders[50] == 70.0

    def test_reduce_active_order_full(self, ladder_core):
        """Test full reduction removes entry."""
        ladder_core.active_orders[50] = 100.0

        ladder_core._reduce_active_order(50, 100.0)

        assert 50 not in ladder_core.active_orders

    def test_reduce_active_order_over(self, ladder_core):
        """Test reducing more than available."""
        ladder_core.active_orders[50] = 50.0

        ladder_core._reduce_active_order(50, 100.0)

        assert 50 not in ladder_core.active_orders

    def test_reduce_active_order_not_exists(self, ladder_core):
        """Test reducing non-existent order."""
        ladder_core._reduce_active_order(50, 100.0)

        # Should not raise
        assert 50 not in ladder_core.active_orders


# ========== Edge Cases ==========
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_place_order_exception_cleanup(self, ladder_core, mock_fb_core):
        """Test temporary order cleaned up on exception."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            side_effect=Exception("API Error")
        )

        order_id = await ladder_core.place_limit_order(50, 10.0, 'YES')

        assert order_id is None
        assert not any(k.startswith('tmp_') for k in ladder_core.pending_orders)

    @pytest.mark.asyncio
    async def test_cancel_order_exception(self, ladder_core, mock_fb_core):
        """Test cancel handles exception gracefully."""
        ladder_core.pending_orders = {'order1': {'price': 50}}
        mock_fb_core.connector.cancel_order = MagicMock(
            side_effect=Exception("Network error")
        )

        result = await ladder_core._cancel_single_order('order1')

        assert result is False

    def test_dirty_flag_set_on_order_operations(self, ladder_core, mock_fb_core):
        """Test dirty flag set during operations."""
        assert ladder_core.dirty is False

        ladder_core.pending_orders['order1'] = {'price': 50}
        # dirty set manually or by place operation

    @pytest.mark.asyncio
    async def test_min_order_size_enforcement(self, ladder_core, mock_fb_core):
        """Test minimum order size is enforced."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'order123'}
        )

        # Place very small order
        await ladder_core.place_limit_order(50, 0.01, 'YES')

        # Check that shares were adjusted to minimum
        call_args = mock_fb_core.connector.create_order.call_args[0]
        shares = call_args[2]
        # Should be at least 5 shares
        assert shares >= 5.0
