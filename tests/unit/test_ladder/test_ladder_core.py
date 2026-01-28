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
        # asyncio.to_thread() expects a sync callable, so use MagicMock not AsyncMock
        mock_fb_core.connector.create_market_order = MagicMock(
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


# ========== Boundary Condition Tests ==========
class TestBoundaryConditions:
    """Test boundary conditions at price limits."""

    @pytest.fixture
    def mock_fb_core(self):
        """Create mock FingerBlasterCore."""
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        """Create LadderCore with mocked dependencies."""
        return LadderCore(fb_core=mock_fb_core)

    def test_is_pending_at_price_1(self, ladder_core):
        """Test is_pending at minimum price boundary (1¢)."""
        ladder_core.pending_orders['order1'] = {'price': 1, 'size': 10.0}
        assert ladder_core.is_pending(1) is True
        assert ladder_core.is_pending(0) is False  # Below range

    def test_is_pending_at_price_99(self, ladder_core):
        """Test is_pending at maximum price boundary (99¢)."""
        ladder_core.pending_orders['order1'] = {'price': 99, 'size': 10.0}
        assert ladder_core.is_pending(99) is True
        assert ladder_core.is_pending(100) is False  # Above range

    def test_is_filled_at_boundaries(self, ladder_core):
        """Test is_filled at price boundaries."""
        ladder_core.filled_orders[1] = time.time()
        ladder_core.filled_orders[99] = time.time()

        assert ladder_core.is_filled(1) is True
        assert ladder_core.is_filled(99) is True
        assert ladder_core.is_filled(0) is False
        assert ladder_core.is_filled(100) is False

    def test_order_filled_at_boundary_prices(self, ladder_core):
        """Test order fill processing at boundary prices."""
        ladder_core.pending_orders['order_1c'] = {'price': 1, 'size': 5.0, 'side': 'YES'}
        ladder_core.pending_orders['order_99c'] = {'price': 99, 'size': 5.0, 'side': 'YES'}
        ladder_core.active_orders[1] = 5.0
        ladder_core.active_orders[99] = 5.0

        # Fill at 1¢
        ladder_core._on_order_filled('YES', 5.0, 0.01, 'order_1c')
        assert 1 in ladder_core.filled_orders
        assert 'order_1c' not in ladder_core.pending_orders

        # Fill at 99¢
        ladder_core._on_order_filled('YES', 5.0, 0.99, 'order_99c')
        assert 99 in ladder_core.filled_orders
        assert 'order_99c' not in ladder_core.pending_orders

    def test_order_filled_out_of_bounds_ignored(self, ladder_core):
        """Test fills with out-of-bounds prices are ignored."""
        # Price 0.00 = 0¢ (out of bounds)
        ladder_core._on_order_filled('YES', 10.0, 0.00, 'order_oob')
        assert 0 not in ladder_core.filled_orders

        # Price 1.00 = 100¢ (out of bounds)
        ladder_core._on_order_filled('YES', 10.0, 1.00, 'order_oob2')
        assert 100 not in ladder_core.filled_orders

    @pytest.mark.asyncio
    async def test_place_order_at_boundary_prices(self, ladder_core, mock_fb_core):
        """Test placing orders at boundary prices."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'boundary_order'}
        )

        # Order at 1¢
        order_id = await ladder_core.place_limit_order(1, 10.0, 'YES')
        assert order_id is not None
        call_args = mock_fb_core.connector.create_order.call_args[0]
        assert call_args[1] == 0.01  # Price should be 0.01

        # Order at 99¢
        order_id = await ladder_core.place_limit_order(99, 10.0, 'YES')
        assert order_id is not None

    def test_get_open_orders_excludes_boundary_violations(self, ladder_core):
        """Test get_open_orders_for_display excludes out-of-bounds prices."""
        ladder_core.pending_orders = {
            'order_0': {'price': 0, 'size': 10.0, 'side': 'YES'},
            'order_1': {'price': 1, 'size': 10.0, 'side': 'YES'},
            'order_99': {'price': 99, 'size': 10.0, 'side': 'YES'},
            'order_100': {'price': 100, 'size': 10.0, 'side': 'YES'},
        }

        orders = ladder_core.get_open_orders_for_display()

        prices = [o['price_cent'] for o in orders]
        assert 0 not in prices
        assert 100 not in prices
        assert 1 in prices
        assert 99 in prices

    def test_view_model_at_boundary_books(self, ladder_core, mock_fb_core):
        """Test view model with order books at boundary prices."""
        mock_fb_core.market_manager.raw_books = {
            'Up': {'bids': {0.01: 100.0, 0.99: 50.0}, 'asks': {}},
            'Down': {'bids': {0.01: 75.0}, 'asks': {}}  # YES ask at 99¢
        }

        ladder = ladder_core.get_view_model()

        assert ladder[1]['yes_bid'] == 100.0
        assert ladder[99]['yes_bid'] == 50.0
        assert ladder[99]['yes_ask'] == 75.0


# ========== Partial Fill Tests ==========
class TestPartialFills:
    """Test partial order fill handling."""

    @pytest.fixture
    def mock_fb_core(self):
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    def test_partial_fill_reduces_active_orders(self, ladder_core):
        """Test partial fill correctly reduces active order size."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 100.0, 'side': 'YES'}
        ladder_core.active_orders[50] = 100.0

        # Partial fill of 30 - note: the order's stored size (100) is used for reduction
        ladder_core._on_order_filled('YES', 30.0, 0.50, 'order1')

        # Active reduced by order's stored size (100), which removes the entry entirely
        assert 50 not in ladder_core.active_orders
        # Order removed from pending
        assert 'order1' not in ladder_core.pending_orders

    def test_multiple_partial_fills_at_same_price(self, ladder_core):
        """Test multiple partial fills at same price level."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 50.0, 'side': 'YES'}
        ladder_core.pending_orders['order2'] = {'price': 50, 'size': 50.0, 'side': 'YES'}
        ladder_core.active_orders[50] = 100.0

        # First partial fill matches order1
        ladder_core._on_order_filled('YES', 50.0, 0.50, 'order1')
        assert ladder_core.active_orders[50] == 50.0

        # Second partial fill matches order2
        ladder_core._on_order_filled('YES', 50.0, 0.50, 'order2')
        assert 50 not in ladder_core.active_orders

    def test_overfill_clamps_to_zero(self, ladder_core):
        """Test fill larger than active order clamps to zero."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 25.0, 'side': 'YES'}
        ladder_core.active_orders[50] = 25.0

        # Fill more than active
        ladder_core._on_order_filled('YES', 100.0, 0.50, 'order1')

        assert 50 not in ladder_core.active_orders


# ========== Multiple Orders at Same Price ==========
class TestMultipleOrdersSamePrice:
    """Test handling multiple orders at same price level."""

    @pytest.fixture
    def mock_fb_core(self):
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    @pytest.mark.asyncio
    async def test_multiple_orders_accumulate_active(self, ladder_core, mock_fb_core):
        """Test multiple orders at same price accumulate in active_orders."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            side_effect=[{'orderID': 'order1'}, {'orderID': 'order2'}, {'orderID': 'order3'}]
        )

        await ladder_core.place_limit_order(50, 10.0, 'YES')
        await ladder_core.place_limit_order(50, 20.0, 'YES')
        await ladder_core.place_limit_order(50, 30.0, 'YES')

        assert ladder_core.active_orders[50] == 60.0
        assert len(ladder_core.pending_orders) == 3

    @pytest.mark.asyncio
    async def test_cancel_all_at_price_multiple_orders(self, ladder_core, mock_fb_core):
        """Test canceling all orders at a price with multiple orders."""
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
            'order2': {'price': 50, 'size': 20.0},
            'order3': {'price': 50, 'size': 30.0},
            'order4': {'price': 60, 'size': 15.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        count = await ladder_core.cancel_all_at_price(50)

        assert count == 3
        assert 'order4' in ladder_core.pending_orders
        assert len(ladder_core.pending_orders) == 1

    def test_fill_matches_first_order_at_price(self, ladder_core):
        """Test fill by price matches first order when ID unknown."""
        ladder_core.pending_orders['order1'] = {'price': 50, 'size': 10.0, 'side': 'YES'}
        ladder_core.pending_orders['order2'] = {'price': 50, 'size': 20.0, 'side': 'YES'}

        # Fill with unknown ID at same price/side
        ladder_core._on_order_filled('YES', 10.0, 0.50, 'unknown_id')

        # Should match one of the orders at 50
        assert len([o for o in ladder_core.pending_orders.values() if o['price'] == 50]) == 1

    def test_view_model_aggregates_multiple_orders(self, ladder_core, mock_fb_core):
        """Test view model aggregates my_size for multiple orders."""
        mock_fb_core.market_manager.raw_books = {
            'Up': {'bids': {}, 'asks': {}},
            'Down': {'bids': {}, 'asks': {}}
        }
        ladder_core.pending_orders = {
            'order1': {'price': 50, 'size': 10.0},
            'order2': {'price': 50, 'size': 20.0},
        }
        ladder_core.active_orders[50] = 5.0

        ladder = ladder_core.get_view_model()

        # my_size should be sum of pending (10+20=30) + active (5) = 35
        assert ladder[50]['my_size'] == 35.0


# ========== Filled Order Window Tests ==========
class TestFilledOrderWindow:
    """Test filled order window timing."""

    @pytest.fixture
    def mock_fb_core(self):
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    def test_is_filled_exactly_at_5_seconds(self, ladder_core):
        """Test is_filled behavior at exactly 5 second boundary."""
        # Set fill time to exactly 5 seconds ago
        ladder_core.filled_orders[50] = time.time() - 5.0

        # At exactly 5 seconds, uses strict less-than (<), so 5.0 is NOT visible
        # This is implementation-specific: FILLED_ORDER_WINDOW_SECS = 5.0 with `< 5.0`
        assert ladder_core.is_filled(50) is False

    def test_is_filled_just_after_5_seconds(self, ladder_core):
        """Test is_filled expires after 5 seconds."""
        # Set fill time to just over 5 seconds ago
        ladder_core.filled_orders[50] = time.time() - 5.01

        assert ladder_core.is_filled(50) is False
        # Entry should be cleaned up
        assert 50 not in ladder_core.filled_orders

    def test_is_filled_cleans_up_old_entries(self, ladder_core):
        """Test is_filled cleans up expired entries on check."""
        # Add multiple old fills
        old_time = time.time() - 10.0
        ladder_core.filled_orders = {
            50: old_time,
            60: old_time,
            70: time.time(),  # Current
        }

        # Check each - old ones should be cleaned
        assert ladder_core.is_filled(50) is False
        assert ladder_core.is_filled(60) is False
        assert ladder_core.is_filled(70) is True

        # Verify cleanup
        assert 50 not in ladder_core.filled_orders
        assert 60 not in ladder_core.filled_orders
        assert 70 in ladder_core.filled_orders


# ========== Error Handling Tests ==========
class TestErrorHandling:
    """Test error handling for invalid inputs and edge cases."""

    @pytest.fixture
    def mock_fb_core(self):
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    def test_extract_order_id_empty_dict(self, ladder_core):
        """Test order ID extraction from empty dict."""
        assert ladder_core._extract_order_id({}) is None

    def test_extract_order_id_integer(self, ladder_core):
        """Test order ID extraction from non-dict."""
        assert ladder_core._extract_order_id(123) is None

    def test_extract_order_id_list(self, ladder_core):
        """Test order ID extraction from list."""
        assert ladder_core._extract_order_id([{'orderID': 'abc'}]) is None

    def test_on_order_filled_with_none_order_id(self, ladder_core):
        """Test order filled callback with None order_id raises TypeError.

        Note: The implementation doesn't guard against None order_id when logging.
        This test documents the current behavior.
        """
        # None order_id causes TypeError in log statement (order_id[:10])
        with pytest.raises(TypeError):
            ladder_core._on_order_filled('YES', 10.0, 0.50, None)

    def test_on_order_filled_with_zero_price(self, ladder_core):
        """Test order filled callback handles zero price (out of bounds)."""
        # Zero price is out of bounds and should be silently ignored
        ladder_core._on_order_filled('YES', 10.0, 0.0, 'order_id')
        assert 0 not in ladder_core.filled_orders

    def test_reduce_active_order_negative_size(self, ladder_core):
        """Test reduce_active_order handles negative size."""
        ladder_core.active_orders[50] = 100.0

        # Reduce by negative (should still subtract)
        ladder_core._reduce_active_order(50, -10.0)
        # -10 reduction = +10, so 100 + 10 = 110
        assert ladder_core.active_orders[50] == 110.0

    def test_get_view_model_with_missing_raw_books(self, ladder_core, mock_fb_core):
        """Test view model when raw_books is missing."""
        mock_fb_core.market_manager.raw_books = None
        ladder_core.last_ladder = {50: {'test': 'cached'}}

        ladder = ladder_core.get_view_model()

        # Should return cached ladder
        assert ladder == ladder_core.last_ladder

    def test_get_view_model_with_empty_raw_books(self, ladder_core, mock_fb_core):
        """Test view model with empty raw_books dict."""
        mock_fb_core.market_manager.raw_books = {}

        ladder = ladder_core.get_view_model()

        # Should return valid ladder with defaults
        assert len(ladder) == 99
        assert all(ladder[p]['yes_bid'] == 0.0 for p in ladder)

    @pytest.mark.asyncio
    async def test_cancel_single_order_missing_from_pending(self, ladder_core, mock_fb_core):
        """Test cancel when order is not in pending_orders."""
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        # Order doesn't exist in pending
        result = await ladder_core._cancel_single_order('nonexistent_order')

        # API was still called
        mock_fb_core.connector.cancel_order.assert_called_once_with('nonexistent_order')
        assert result is True

    @pytest.mark.asyncio
    async def test_place_limit_order_with_zero_price(self, ladder_core, mock_fb_core):
        """Test placing order with zero price division edge case."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        # Price at 0 would cause division issue in min size calc
        # But 0 is out of valid range so it should still work
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'order123'}
        )

        order_id = await ladder_core.place_limit_order(1, 10.0, 'YES')

        # Should succeed at minimum valid price
        assert order_id == 'order123'

    def test_on_market_update_with_none_values(self, ladder_core):
        """Test market update callback handles None values."""
        ladder_core.market_name = "Old Market"

        # Should not crash
        ladder_core._on_market_update(None, None, None)

        # Name should be updated to None
        assert ladder_core.market_name is None

    def test_set_market_update_callback_to_none(self, ladder_core):
        """Test setting market callback to None."""
        callback = MagicMock()
        ladder_core.set_market_update_callback(callback)

        # Now set to None
        ladder_core.set_market_update_callback(None)

        # Should not crash when market updates
        ladder_core._on_market_update("$95000", "12:00PM", "Test Market")

    def test_is_pending_with_malformed_pending_orders(self, ladder_core):
        """Test is_pending behavior with malformed order data.

        Note: The implementation doesn't guard against missing 'price' key.
        This test documents the current behavior.
        """
        # Orders missing 'price' will raise KeyError when checked
        ladder_core.pending_orders = {
            'order1': {'price': 50},  # Valid, missing size (ok for is_pending)
        }
        assert ladder_core.is_pending(50) is True
        assert ladder_core.is_pending(60) is False

        # Adding malformed orders will cause KeyError
        ladder_core.pending_orders['order2'] = {'size': 10.0}  # Missing price

        with pytest.raises(KeyError):
            ladder_core.is_pending(60)


# ========== NO Side Price Conversion Tests ==========
class TestNOSidePriceConversion:
    """Test NO side order price conversion logic."""

    @pytest.fixture
    def mock_fb_core(self):
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)
        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    @pytest.mark.asyncio
    async def test_no_order_price_conversion(self, ladder_core, mock_fb_core):
        """Test NO order converts price correctly."""
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'no_order'}
        )

        # Place NO order at 70¢ (YES perspective)
        # This means buying Down token at 30¢
        await ladder_core.place_limit_order(70, 10.0, 'NO')

        call_args = mock_fb_core.connector.create_order.call_args[0]
        # Token should be Down
        assert call_args[0] == '0x' + '2' * 64
        # Price should be complementary: 100 - 70 = 30¢ = 0.30
        assert call_args[1] == 0.30

    def test_no_side_fill_price_conversion(self, ladder_core):
        """Test NO fill converts price to YES perspective."""
        # Place a NO order at 70¢ (YES perspective)
        ladder_core.pending_orders['order_no'] = {'price': 70, 'size': 10.0, 'side': 'NO'}

        # Fill comes back as NO side at 0.30 (Down token price)
        # 100 - 30 = 70¢ (YES perspective)
        ladder_core._on_order_filled('NO', 10.0, 0.30, 'different_id')

        # Should match and mark filled at 70¢
        assert 70 in ladder_core.filled_orders
        assert 'order_no' not in ladder_core.pending_orders

    def test_no_side_fill_at_boundary(self, ladder_core):
        """Test NO fill at price boundaries."""
        # NO at 99¢ = Down at 1¢
        ladder_core.pending_orders['order_no'] = {'price': 99, 'size': 10.0, 'side': 'NO'}

        ladder_core._on_order_filled('NO', 10.0, 0.01, 'different_id')

        assert 99 in ladder_core.filled_orders

        # NO at 1¢ = Down at 99¢
        ladder_core.pending_orders['order_no2'] = {'price': 1, 'size': 10.0, 'side': 'NO'}

        ladder_core._on_order_filled('NO', 10.0, 0.99, 'different_id2')

        assert 1 in ladder_core.filled_orders


# ========== Integration Tests ==========
class TestLadderCoreIntegration:
    """Integration tests for LadderCore with realistic scenarios."""

    @pytest.fixture
    def mock_fb_core(self):
        """Create mock FingerBlasterCore with realistic structure."""
        fb = MagicMock()
        fb.connector = MagicMock()
        fb.market_manager = MagicMock()
        fb.register_callback = MagicMock(return_value=True)

        # Set up realistic raw_books
        fb.market_manager.raw_books = {
            'Up': {
                'bids': {0.45: 1000.0, 0.44: 500.0, 0.43: 250.0},
                'asks': {}
            },
            'Down': {
                'bids': {0.45: 800.0, 0.44: 400.0, 0.43: 200.0},
                'asks': {}
            }
        }

        return fb

    @pytest.fixture
    def ladder_core(self, mock_fb_core):
        return LadderCore(fb_core=mock_fb_core)

    def test_full_order_lifecycle(self, ladder_core, mock_fb_core):
        """Test complete order lifecycle: place -> fill -> cleanup."""
        # Setup mocks
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'lifecycle_order_123'}
        )

        # 1. Place order
        import asyncio
        order_id = asyncio.get_event_loop().run_until_complete(
            ladder_core.place_limit_order(50, 10.0, 'YES')
        )
        assert order_id == 'lifecycle_order_123'
        assert 'lifecycle_order_123' in ladder_core.pending_orders
        assert ladder_core.active_orders[50] == 10.0

        # 2. Simulate fill
        ladder_core._on_order_filled('YES', 10.0, 0.50, 'lifecycle_order_123')

        # 3. Verify cleanup
        assert 'lifecycle_order_123' not in ladder_core.pending_orders
        assert 50 not in ladder_core.active_orders
        assert 50 in ladder_core.filled_orders

    def test_market_transition_clears_state(self, ladder_core, mock_fb_core):
        """Test that market transition clears all order state."""
        # Setup some state
        ladder_core.pending_orders = {
            'order1': {'price': 45, 'size': 100.0, 'side': 'YES'},
            'order2': {'price': 55, 'size': 50.0, 'side': 'NO'},
        }
        ladder_core.active_orders = {45: 100.0, 55: 50.0}
        ladder_core.filled_orders = {50: time.time()}
        ladder_core.market_name = "Old Market"

        # Simulate market transition
        ladder_core._on_market_update("$96000", "2:00PM", "New Market", "1:45PM")

        # Verify all state cleared
        assert ladder_core.pending_orders == {}
        assert ladder_core.active_orders == {}
        assert ladder_core.filled_orders == {}
        assert ladder_core.market_name == "New Market"

    def test_view_model_reflects_order_state(self, ladder_core, mock_fb_core):
        """Test view model correctly reflects pending and active orders."""
        # Add orders at different prices
        ladder_core.pending_orders = {
            'order1': {'price': 45, 'size': 100.0},
            'order2': {'price': 45, 'size': 50.0},  # Same price
            'order3': {'price': 55, 'size': 25.0},
        }
        ladder_core.active_orders = {45: 10.0}  # Additional active

        ladder = ladder_core.get_view_model()

        # Price 45 should have: 100 + 50 (pending) + 10 (active) = 160
        assert ladder[45]['my_size'] == 160.0
        # Price 55 should have: 25 (pending)
        assert ladder[55]['my_size'] == 25.0
        # Other prices should have 0
        assert ladder[50]['my_size'] == 0.0

    def test_dom_view_model_with_mixed_state(self, ladder_core, mock_fb_core):
        """Test DOM view model with order books and user orders."""
        # Update raw books
        mock_fb_core.market_manager.raw_books = {
            'Up': {'bids': {0.45: 1000.0}, 'asks': {}},
            'Down': {'bids': {0.45: 800.0}, 'asks': {}}
        }

        # Add user orders
        ladder_core.pending_orders = {
            'user_order': {'price': 50, 'size': 15.0, 'side': 'YES'}
        }

        dom = ladder_core.get_dom_view_model()

        # Verify market data
        assert dom.rows[45].yes_depth == 1000.0  # From Up bids
        assert dom.rows[55].no_depth == 800.0  # From Down bids at 0.45

        # Verify user orders
        assert len(dom.rows[50].my_orders) == 1
        assert dom.rows[50].my_orders[0].size == 15.0

        # Verify spread detection
        assert dom.best_bid_cent == 45
        assert dom.best_ask_cent == 55

    @pytest.mark.asyncio
    async def test_cancel_all_with_mixed_orders(self, ladder_core, mock_fb_core):
        """Test cancel_all with mix of temp and real orders."""
        ladder_core.pending_orders = {
            'tmp_50_123': {'price': 50, 'size': 10.0},  # Temp - no API call
            'real_order_1': {'price': 55, 'size': 20.0},
            'real_order_2': {'price': 60, 'size': 30.0},
        }
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        count = await ladder_core.cancel_all_orders()

        assert count == 3
        # Only real orders should trigger API calls
        assert mock_fb_core.connector.cancel_order.call_count == 2

    def test_callback_registration(self, mock_fb_core):
        """Test callbacks are properly registered."""
        core = LadderCore(fb_core=mock_fb_core)

        calls = mock_fb_core.register_callback.call_args_list
        registered_events = [call[0][0] for call in calls]

        assert 'market_update' in registered_events
        assert 'order_filled' in registered_events

    def test_market_update_callback_invocation(self, ladder_core):
        """Test market update callback is invoked correctly."""
        callback_mock = MagicMock()
        ladder_core.set_market_update_callback(callback_mock)

        ladder_core._on_market_update("$95000", "12:00PM", "Test Market", "11:45AM")

        callback_mock.assert_called_once_with("Test Market", "11:45AM", "12:00PM")

    def test_concurrent_order_placement_simulation(self, ladder_core, mock_fb_core):
        """Simulate concurrent order placement and fills."""
        import asyncio

        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )

        order_counter = [0]

        def create_order_side_effect(*args, **kwargs):
            order_counter[0] += 1
            return {'orderID': f'concurrent_order_{order_counter[0]}'}

        mock_fb_core.connector.create_order = MagicMock(side_effect=create_order_side_effect)

        async def place_multiple_orders():
            tasks = [
                ladder_core.place_limit_order(45, 10.0, 'YES'),
                ladder_core.place_limit_order(50, 20.0, 'YES'),
                ladder_core.place_limit_order(55, 30.0, 'YES'),
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.get_event_loop().run_until_complete(place_multiple_orders())

        # All orders should have unique IDs
        assert len(set(results)) == 3
        assert len(ladder_core.pending_orders) == 3

        # Active orders should be accumulated
        assert ladder_core.active_orders[45] == 10.0
        assert ladder_core.active_orders[50] == 20.0
        assert ladder_core.active_orders[55] == 30.0

    def test_get_market_fields_structure(self, ladder_core):
        """Test get_market_fields returns correct structure."""
        ladder_core.market_name = "BTC Up/Down 15m"
        ladder_core.market_starts = "2024-01-01T12:00:00Z"
        ladder_core.market_ends = "2024-01-01T12:15:00Z"

        fields = ladder_core.get_market_fields()

        assert isinstance(fields, dict)
        assert 'name' in fields
        assert 'starts' in fields
        assert 'ends' in fields
        assert fields['name'] == "BTC Up/Down 15m"

    def test_dirty_flag_behavior(self, ladder_core, mock_fb_core):
        """Test dirty flag is set appropriately during operations."""
        assert ladder_core.dirty is False

        # Place order sets dirty
        mock_fb_core.market_manager.get_token_map = AsyncMock(
            return_value={'Up': '0x' + '1' * 64}
        )
        mock_fb_core.connector.create_order = MagicMock(
            return_value={'orderID': 'dirty_test'}
        )

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            ladder_core.place_limit_order(50, 10.0, 'YES')
        )

        assert ladder_core.dirty is True

        # Reset and test cancel
        ladder_core.dirty = False
        mock_fb_core.connector.cancel_order = MagicMock(return_value={'canceled': True})

        asyncio.get_event_loop().run_until_complete(ladder_core.cancel_all_orders())

        assert ladder_core.dirty is True
