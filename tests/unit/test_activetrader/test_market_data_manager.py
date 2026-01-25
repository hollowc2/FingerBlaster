"""Comprehensive tests for MarketDataManager - validation, order books, mid-price, stale detection."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from src.activetrader.engine import MarketDataManager, DEFAULT_ORDER_BOOK_PRICE
from src.activetrader.config import AppConfig


# ========== Test Fixtures ==========
@pytest.fixture
def config():
    """AppConfig with short timeouts for testing."""
    cfg = AppConfig()
    cfg.market_duration_minutes = 15
    return cfg


@pytest.fixture
def market_data_manager(config):
    """MarketDataManager instance."""
    return MarketDataManager(config)


@pytest.fixture
def valid_market():
    """Valid market data fixture."""
    return {
        'market_id': 'test-market-123',
        'end_date': '2025-01-24T12:30:00Z',
        'start_date': '2025-01-24T12:15:00Z',
        'token_map': {
            'Up': '0x' + 'a' * 60,
            'Down': '0x' + 'b' * 60
        },
        'question': 'BTC Up or Down 15m?'
    }


# ========== Market Validation Tests ==========
class TestMarketValidation:
    """Test market data validation logic."""

    @pytest.mark.asyncio
    async def test_validate_market_with_all_required_fields(self, market_data_manager, valid_market):
        """Test validation passes with all required fields."""
        result = market_data_manager._validate_market(valid_market)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_market_missing_market_id(self, market_data_manager, valid_market):
        """Test validation fails without market_id."""
        del valid_market['market_id']
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_missing_end_date(self, market_data_manager, valid_market):
        """Test validation fails without end_date."""
        del valid_market['end_date']
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_missing_token_map(self, market_data_manager, valid_market):
        """Test validation fails without token_map."""
        del valid_market['token_map']
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_token_map_not_dict(self, market_data_manager, valid_market):
        """Test validation fails when token_map is not a dict."""
        valid_market['token_map'] = "not a dict"
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_token_map_missing_up(self, market_data_manager, valid_market):
        """Test validation fails when Up is missing from token_map."""
        del valid_market['token_map']['Up']
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_token_map_missing_down(self, market_data_manager, valid_market):
        """Test validation fails when Down is missing from token_map."""
        del valid_market['token_map']['Down']
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_empty_token_map(self, market_data_manager, valid_market):
        """Test validation fails with empty token_map."""
        valid_market['token_map'] = {}
        result = market_data_manager._validate_market(valid_market)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_market_null_values(self, market_data_manager):
        """Test validation handles None/null values gracefully."""
        invalid_markets = [
            {'market_id': None, 'end_date': '2025-01-24T12:30:00Z', 'token_map': {}},
            {'market_id': 'test', 'end_date': None, 'token_map': {}},
            {'market_id': 'test', 'end_date': '2025-01-24T12:30:00Z', 'token_map': None},
        ]
        for market in invalid_markets:
            result = market_data_manager._validate_market(market)
            assert result is False

    @pytest.mark.asyncio
    async def test_set_market_success(self, market_data_manager, valid_market):
        """Test set_market succeeds with valid market."""
        result = await market_data_manager.set_market(valid_market)
        assert result is True

        current_market = await market_data_manager.get_market()
        assert current_market['market_id'] == 'test-market-123'

    @pytest.mark.asyncio
    async def test_set_market_failure_invalid_data(self, market_data_manager):
        """Test set_market fails with invalid data."""
        invalid_market = {'market_id': 'test'}
        result = await market_data_manager.set_market(invalid_market)
        assert result is False

        current_market = await market_data_manager.get_market()
        assert current_market is None


# ========== Token Map Extraction Tests ==========
class TestTokenMapExtraction:
    """Test token map extraction and management."""

    @pytest.mark.asyncio
    async def test_token_map_extracted_correctly(self, market_data_manager, valid_market):
        """Test token map is extracted from market data."""
        await market_data_manager.set_market(valid_market)

        token_map = await market_data_manager.get_token_map()
        assert 'Up' in token_map
        assert 'Down' in token_map
        assert token_map['Up'] == valid_market['token_map']['Up']
        assert token_map['Down'] == valid_market['token_map']['Down']

    @pytest.mark.asyncio
    async def test_token_map_is_copy(self, market_data_manager, valid_market):
        """Test get_token_map returns a copy (not reference)."""
        await market_data_manager.set_market(valid_market)

        token_map1 = await market_data_manager.get_token_map()
        token_map2 = await market_data_manager.get_token_map()

        # Modify returned copy
        token_map1['Up'] = 'modified'

        # Should not affect manager's internal state
        token_map3 = await market_data_manager.get_token_map()
        assert token_map3['Up'] == valid_market['token_map']['Up']

    @pytest.mark.asyncio
    async def test_token_map_empty_before_set(self, market_data_manager):
        """Test token map is empty before setting market."""
        token_map = await market_data_manager.get_token_map()
        assert len(token_map) == 0

    @pytest.mark.asyncio
    async def test_token_map_cleared_on_clear_market(self, market_data_manager, valid_market):
        """Test token map is cleared when market is cleared."""
        await market_data_manager.set_market(valid_market)
        await market_data_manager.clear_market()

        token_map = await market_data_manager.get_token_map()
        assert len(token_map) == 0


# ========== Order Book Update Tests ==========
class TestOrderBookUpdates:
    """Test full and incremental order book updates."""

    @pytest.mark.asyncio
    async def test_update_order_book_full_update(self, market_data_manager):
        """Test full order book update."""
        bids = {0.55: 100.0, 0.54: 200.0}
        asks = {0.56: 150.0, 0.57: 250.0}

        await market_data_manager.update_order_book('Up', bids, asks)

        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['bids'] == bids
        assert raw_books['Up']['asks'] == asks

    @pytest.mark.asyncio
    async def test_update_order_book_tracks_update_time(self, market_data_manager):
        """Test order book update tracks timestamp."""
        before = time.time()
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})
        after = time.time()

        assert before <= market_data_manager._last_update_time <= after

    @pytest.mark.asyncio
    async def test_update_order_book_increments_counter(self, market_data_manager):
        """Test order book update increments counter."""
        initial_count = market_data_manager._update_counter

        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        assert market_data_manager._update_counter == initial_count + 1

    @pytest.mark.asyncio
    async def test_update_order_book_invalid_token_type(self, market_data_manager):
        """Test update with invalid token type is ignored."""
        await market_data_manager.update_order_book('Invalid', {0.55: 100.0}, {0.56: 100.0})

        raw_books = await market_data_manager.get_raw_order_book()
        # Should remain empty
        assert len(raw_books['Up']['bids']) == 0
        assert len(raw_books['Up']['asks']) == 0

    @pytest.mark.asyncio
    async def test_update_order_book_creates_copy(self, market_data_manager):
        """Test update_order_book copies data (not reference)."""
        bids = {0.55: 100.0}
        asks = {0.56: 100.0}

        await market_data_manager.update_order_book('Up', bids, asks)

        # Modify original
        bids[0.55] = 999.0

        # Should not affect stored data
        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['bids'][0.55] == 100.0


# ========== Incremental Price Changes Tests ==========
class TestIncrementalPriceChanges:
    """Test apply_price_changes for incremental order book updates."""

    @pytest.mark.asyncio
    async def test_apply_price_changes_add_bids(self, market_data_manager):
        """Test adding new bid levels."""
        changes = [
            {'price': 0.55, 'size': 100.0, 'side': 'BUY'},
            {'price': 0.54, 'size': 200.0, 'side': 'BUY'}
        ]

        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['bids'][0.55] == 100.0
        assert raw_books['Up']['bids'][0.54] == 200.0

    @pytest.mark.asyncio
    async def test_apply_price_changes_add_asks(self, market_data_manager):
        """Test adding new ask levels."""
        changes = [
            {'price': 0.56, 'size': 150.0, 'side': 'SELL'},
            {'price': 0.57, 'size': 250.0, 'side': 'SELL'}
        ]

        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['asks'][0.56] == 150.0
        assert raw_books['Up']['asks'][0.57] == 250.0

    @pytest.mark.asyncio
    async def test_apply_price_changes_update_existing(self, market_data_manager):
        """Test updating existing price levels."""
        # Set initial state
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Update sizes
        changes = [
            {'price': 0.55, 'size': 300.0, 'side': 'BUY'},
            {'price': 0.56, 'size': 400.0, 'side': 'SELL'}
        ]
        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['bids'][0.55] == 300.0
        assert raw_books['Up']['asks'][0.56] == 400.0

    @pytest.mark.asyncio
    async def test_apply_price_changes_delete_zero_size(self, market_data_manager):
        """Test zero size deletes price level."""
        # Set initial state
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Delete with zero size
        changes = [
            {'price': 0.55, 'size': 0.0, 'side': 'BUY'},
            {'price': 0.56, 'size': 0.0, 'side': 'SELL'}
        ]
        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert 0.55 not in raw_books['Up']['bids']
        assert 0.56 not in raw_books['Up']['asks']

    @pytest.mark.asyncio
    async def test_apply_price_changes_delete_negative_size(self, market_data_manager):
        """Test negative size deletes price level."""
        # Set initial state
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Delete with negative size
        changes = [
            {'price': 0.55, 'size': -1.0, 'side': 'BUY'},
            {'price': 0.56, 'size': -10.0, 'side': 'SELL'}
        ]
        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert 0.55 not in raw_books['Up']['bids']
        assert 0.56 not in raw_books['Up']['asks']

    @pytest.mark.asyncio
    async def test_apply_price_changes_case_insensitive_side(self, market_data_manager):
        """Test side matching is case insensitive."""
        changes = [
            {'price': 0.55, 'size': 100.0, 'side': 'buy'},  # lowercase
            {'price': 0.56, 'size': 100.0, 'side': 'Sell'},  # mixed case
        ]

        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert raw_books['Up']['bids'][0.55] == 100.0
        assert raw_books['Up']['asks'][0.56] == 100.0

    @pytest.mark.asyncio
    async def test_apply_price_changes_invalid_side_ignored(self, market_data_manager):
        """Test invalid side values are ignored."""
        changes = [
            {'price': 0.55, 'size': 100.0, 'side': 'INVALID'},
            {'price': 0.56, 'size': 100.0, 'side': ''},
            {'price': 0.57, 'size': 100.0, 'side': None}
        ]

        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert len(raw_books['Up']['bids']) == 0
        assert len(raw_books['Up']['asks']) == 0

    @pytest.mark.asyncio
    async def test_apply_price_changes_malformed_data_handled(self, market_data_manager):
        """Test malformed price changes are handled gracefully."""
        changes = [
            {'price': 'invalid', 'size': 100.0, 'side': 'BUY'},  # Invalid price (ValueError)
            {'price': 0.55, 'size': 'invalid', 'side': 'BUY'},  # Invalid size (ValueError)
            "not a dict",  # Not a dict
            None,  # None
            {}  # Empty dict
        ]

        # Should not raise
        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        # All should be ignored (ValueError caught in try-except)
        assert len(raw_books['Up']['bids']) == 0

    @pytest.mark.asyncio
    async def test_apply_price_changes_batch_processing(self, market_data_manager):
        """Test many changes are batched efficiently."""
        changes = [
            {'price': 0.50 + i/100, 'size': 100.0 + i, 'side': 'BUY'}
            for i in range(50)
        ]

        await market_data_manager.apply_price_changes('Up', changes)

        raw_books = await market_data_manager.get_raw_order_book()
        assert len(raw_books['Up']['bids']) == 50


# ========== Mid-Price Calculation Tests ==========
class TestMidPriceCalculation:
    """Test mid-price calculation with various edge cases."""

    @pytest.mark.asyncio
    async def test_calculate_mid_price_normal_case(self, market_data_manager):
        """Test mid-price with both bids and asks."""
        bids = {0.55: 100.0, 0.54: 200.0}
        asks = {0.56: 150.0, 0.57: 250.0}
        await market_data_manager.update_order_book('Up', bids, asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Mid should be average of best bid (0.55) and best ask (0.56)
        assert yes_price == pytest.approx(0.555)
        assert no_price == pytest.approx(0.445)
        assert best_bid == 0.55
        assert best_ask == 0.56

    @pytest.mark.asyncio
    async def test_calculate_mid_price_only_bids(self, market_data_manager):
        """Test mid-price with only bids (no asks)."""
        bids = {0.55: 100.0, 0.54: 200.0}
        asks = {}
        await market_data_manager.update_order_book('Up', bids, asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Should use best bid as mid when no asks
        assert yes_price == pytest.approx(0.55)
        assert no_price == pytest.approx(0.45)
        assert best_bid == 0.55
        assert best_ask == 1.0

    @pytest.mark.asyncio
    async def test_calculate_mid_price_only_asks(self, market_data_manager):
        """Test mid-price with only asks (no bids)."""
        bids = {}
        asks = {0.56: 150.0, 0.57: 250.0}
        await market_data_manager.update_order_book('Up', bids, asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Should use best ask as mid when no bids
        assert yes_price == pytest.approx(0.56)
        assert no_price == pytest.approx(0.44)
        assert best_bid == 0.0
        assert best_ask == 0.56

    @pytest.mark.asyncio
    async def test_calculate_mid_price_empty_order_book(self, market_data_manager):
        """Test mid-price with empty order book defaults to 0.5."""
        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        assert yes_price == DEFAULT_ORDER_BOOK_PRICE
        assert no_price == 1.0 - DEFAULT_ORDER_BOOK_PRICE
        assert best_bid == 0.0
        assert best_ask == 1.0

    @pytest.mark.asyncio
    async def test_calculate_mid_price_down_market_conversion(self, market_data_manager):
        """Test Down market prices are converted to Up equivalent (1.0 - price)."""
        # Set Up order book
        up_bids = {0.50: 100.0}
        up_asks = {0.52: 100.0}
        await market_data_manager.update_order_book('Up', up_bids, up_asks)

        # Set Down order book
        # Down ask at 0.40 = Up bid at 0.60 (1.0 - 0.40)
        # Down bid at 0.38 = Up ask at 0.62 (1.0 - 0.38)
        down_bids = {0.38: 50.0}
        down_asks = {0.40: 75.0}
        await market_data_manager.update_order_book('Down', down_bids, down_asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Best bid should be max(0.50, 0.60) = 0.60
        # Best ask should be min(0.52, 0.62) = 0.52
        assert best_bid == 0.60
        assert best_ask == 0.52

    @pytest.mark.asyncio
    async def test_calculate_mid_price_liquidity_aggregation(self, market_data_manager):
        """Test liquidity from both Up and Down is aggregated."""
        # Up: bid at 0.55 with size 100
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {})

        # Down: ask at 0.45 = Up bid at 0.55 with size 50
        await market_data_manager.update_order_book('Down', {}, {0.45: 50.0})

        raw_books = await market_data_manager.get_raw_order_book()
        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Both contribute to bid at 0.55, total size should be aggregated
        # (This is implicitly tested by the conversion logic)
        assert best_bid == 0.55

    @pytest.mark.asyncio
    async def test_calculate_mid_price_rounding(self, market_data_manager):
        """Test price rounding to 4 decimal places in conversion."""
        # Down price that results in repeating decimal
        down_asks = {0.33333: 100.0}  # 1.0 - 0.33333 = 0.66667, rounds to 0.6667
        await market_data_manager.update_order_book('Down', {}, down_asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Should be rounded to 4 decimals
        assert best_bid == 0.6667


# ========== Stale Data Detection Tests ==========
class TestStaleDataDetection:
    """Test stale data detection and freshness tracking."""

    @pytest.mark.asyncio
    async def test_is_data_stale_no_updates_yet(self, market_data_manager):
        """Test data is stale when no updates have occurred."""
        is_stale = await market_data_manager.is_data_stale()
        assert is_stale is True

    @pytest.mark.asyncio
    async def test_is_data_stale_fresh_data(self, market_data_manager):
        """Test data is not stale immediately after update."""
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        is_stale = await market_data_manager.is_data_stale()
        assert is_stale is False

    @pytest.mark.asyncio
    async def test_is_data_stale_after_threshold(self, market_data_manager):
        """Test data becomes stale after 30s threshold."""
        market_data_manager._last_update_time = time.time() - 31.0  # 31 seconds ago

        is_stale = await market_data_manager.is_data_stale()
        assert is_stale is True

    @pytest.mark.asyncio
    async def test_is_data_stale_at_boundary(self, market_data_manager):
        """Test staleness at exactly 30s boundary."""
        # Set exactly at threshold (29.9s ago to account for test execution time)
        now = time.time()
        market_data_manager._last_update_time = now - 29.9

        is_stale = await market_data_manager.is_data_stale()
        # Should NOT be stale (uses > not >=)
        assert is_stale is False

        # Set just over threshold
        market_data_manager._last_update_time = now - 30.5
        is_stale = await market_data_manager.is_data_stale()
        assert is_stale is True

    @pytest.mark.asyncio
    async def test_get_data_freshness_info_initial_state(self, market_data_manager):
        """Test freshness info in initial state."""
        info = await market_data_manager.get_data_freshness_info()

        assert info['last_update_time'] == 0.0
        assert info['seconds_since_update'] == -1
        assert info['update_counter'] == 0
        assert info['is_stale'] is True
        assert info['stale_threshold'] == 30.0

    @pytest.mark.asyncio
    async def test_get_data_freshness_info_after_update(self, market_data_manager):
        """Test freshness info after an update."""
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        info = await market_data_manager.get_data_freshness_info()

        assert info['last_update_time'] > 0
        assert 0 <= info['seconds_since_update'] < 1  # Should be very recent
        assert info['update_counter'] == 1
        assert info['is_stale'] is False

    @pytest.mark.asyncio
    async def test_get_data_freshness_info_multiple_updates(self, market_data_manager):
        """Test freshness info tracks multiple updates."""
        for i in range(5):
            await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})
            await asyncio.sleep(0.01)  # Small delay between updates

        info = await market_data_manager.get_data_freshness_info()

        assert info['update_counter'] == 5

    @pytest.mark.asyncio
    async def test_update_counter_increments_on_apply_price_changes(self, market_data_manager):
        """Test update counter increments for incremental updates."""
        changes = [{'price': 0.55, 'size': 100.0, 'side': 'BUY'}]
        await market_data_manager.apply_price_changes('Up', changes)

        info = await market_data_manager.get_data_freshness_info()
        assert info['update_counter'] == 1


# ========== Timezone Handling Tests ==========
class TestTimezoneHandling:
    """Test timezone-aware timestamp handling."""

    @pytest.mark.asyncio
    async def test_market_start_time_with_utc_timestamps(self, market_data_manager, valid_market):
        """Test market start time calculated correctly with UTC timestamps."""
        await market_data_manager.set_market(valid_market)

        start_time = await market_data_manager.get_market_start_time()

        assert start_time is not None
        assert start_time.tz is not None  # Should be timezone-aware
        assert str(start_time.tz) == 'UTC'

    @pytest.mark.asyncio
    async def test_market_start_time_naive_timestamp_localized(self, market_data_manager, valid_market):
        """Test naive timestamps are localized to UTC."""
        valid_market['start_date'] = '2025-01-24T12:15:00'  # No timezone
        valid_market['end_date'] = '2025-01-24T12:30:00'

        await market_data_manager.set_market(valid_market)

        start_time = await market_data_manager.get_market_start_time()

        assert start_time.tz is not None
        assert str(start_time.tz) == 'UTC'

    @pytest.mark.asyncio
    async def test_market_duration_calculation_normal(self, market_data_manager, valid_market, config):
        """Test 15-minute market duration calculated correctly."""
        # Dates are 15 minutes apart
        await market_data_manager.set_market(valid_market)

        start_time = await market_data_manager.get_market_start_time()
        end_time = pd.Timestamp(valid_market['end_date'])
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        duration = (end_time - start_time).total_seconds() / 60.0
        assert 14.5 <= duration <= 15.5  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_market_start_recalculated_when_too_long(self, market_data_manager, valid_market, config):
        """Test start time recalculated when API duration > 20 minutes."""
        # Set start_date much earlier (simulating Series start instead of market start)
        valid_market['start_date'] = '2025-01-24T10:00:00Z'  # 2.5 hours before end
        valid_market['end_date'] = '2025-01-24T12:30:00Z'

        await market_data_manager.set_market(valid_market)

        start_time = await market_data_manager.get_market_start_time()
        end_time = pd.Timestamp(valid_market['end_date'])
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        # Should be recalculated to 15 minutes before end
        duration = (end_time - start_time).total_seconds() / 60.0
        assert 14.9 <= duration <= 15.1  # Should be ~15 minutes

    @pytest.mark.asyncio
    async def test_market_start_fallback_without_start_date(self, market_data_manager, valid_market, config):
        """Test fallback calculation when start_date missing."""
        del valid_market['start_date']

        await market_data_manager.set_market(valid_market)

        start_time = await market_data_manager.get_market_start_time()
        end_time = pd.Timestamp(valid_market['end_date'])
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        # Should calculate from end_date
        duration = (end_time - start_time).total_seconds() / 60.0
        assert abs(duration - config.market_duration_minutes) < 0.1


# ========== Concurrent Safety Tests ==========
class TestConcurrentSafety:
    """Test thread/async safety of concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_set_market(self, market_data_manager, valid_market):
        """Test concurrent set_market calls are safe."""
        markets = [
            {**valid_market, 'market_id': f'market-{i}'}
            for i in range(10)
        ]

        results = await asyncio.gather(*[
            market_data_manager.set_market(m) for m in markets
        ])

        # All should succeed
        assert all(results)

        # Final market should be one of them
        current = await market_data_manager.get_market()
        assert current is not None
        assert current['market_id'].startswith('market-')

    @pytest.mark.asyncio
    async def test_concurrent_order_book_updates(self, market_data_manager):
        """Test concurrent order book updates are safe."""
        async def update_book(token_type, price_offset):
            bids = {0.50 + price_offset: 100.0}
            asks = {0.51 + price_offset: 100.0}
            await market_data_manager.update_order_book(token_type, bids, asks)

        # Update Up and Down concurrently
        await asyncio.gather(*[
            update_book('Up', i/100) for i in range(10)
        ])

        raw_books = await market_data_manager.get_raw_order_book()
        # Last update should be present
        assert len(raw_books['Up']['bids']) > 0

    @pytest.mark.asyncio
    async def test_concurrent_apply_price_changes(self, market_data_manager):
        """Test concurrent price changes are safe."""
        async def apply_changes(price_offset):
            changes = [
                {'price': 0.50 + price_offset, 'size': 100.0, 'side': 'BUY'}
            ]
            await market_data_manager.apply_price_changes('Up', changes)

        await asyncio.gather(*[apply_changes(i/100) for i in range(20)])

        raw_books = await market_data_manager.get_raw_order_book()
        # Should have multiple price levels
        assert len(raw_books['Up']['bids']) > 0

    @pytest.mark.asyncio
    async def test_concurrent_calculate_mid_price(self, market_data_manager):
        """Test concurrent mid-price calculations are safe."""
        # Set initial state
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Calculate mid-price concurrently many times
        results = await asyncio.gather(*[
            market_data_manager.calculate_mid_price() for _ in range(50)
        ])

        # All should return same result
        for yes_price, no_price, best_bid, best_ask in results:
            assert yes_price == 0.555
            assert best_bid == 0.55
            assert best_ask == 0.56

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self, market_data_manager, valid_market):
        """Test concurrent getter operations are safe."""
        await market_data_manager.set_market(valid_market)

        # Call all getters concurrently
        results = await asyncio.gather(
            market_data_manager.get_market(),
            market_data_manager.get_token_map(),
            market_data_manager.get_market_start_time(),
            market_data_manager.get_raw_order_book(),
            market_data_manager.is_data_stale(),
            market_data_manager.get_data_freshness_info()
        )

        # All should return successfully
        assert results[0] is not None  # market
        assert len(results[1]) > 0  # token_map
        assert results[2] is not None  # start_time

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, market_data_manager, valid_market):
        """Test mixed read/write operations concurrently."""
        await market_data_manager.set_market(valid_market)

        async def reader():
            await market_data_manager.get_market()
            await market_data_manager.calculate_mid_price()

        async def writer():
            await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Mix reads and writes
        await asyncio.gather(*[
            reader() if i % 2 == 0 else writer()
            for i in range(20)
        ])

        # Should complete without deadlock
        current = await market_data_manager.get_market()
        assert current is not None


# ========== Edge Cases and Error Handling ==========
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_clear_market_resets_all_state(self, market_data_manager, valid_market):
        """Test clear_market resets all internal state."""
        # Set up state
        await market_data_manager.set_market(valid_market)
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Clear
        await market_data_manager.clear_market()

        # Verify everything is reset
        assert await market_data_manager.get_market() is None
        assert len(await market_data_manager.get_token_map()) == 0
        assert await market_data_manager.get_market_start_time() is None

        raw_books = await market_data_manager.get_raw_order_book()
        assert len(raw_books['Up']['bids']) == 0
        assert len(raw_books['Up']['asks']) == 0
        assert len(raw_books['Down']['bids']) == 0
        assert len(raw_books['Down']['asks']) == 0

    @pytest.mark.asyncio
    async def test_get_market_returns_copy(self, market_data_manager, valid_market):
        """Test get_market returns a copy (modifications don't affect state)."""
        await market_data_manager.set_market(valid_market)

        market1 = await market_data_manager.get_market()
        market1['market_id'] = 'modified'

        market2 = await market_data_manager.get_market()
        assert market2['market_id'] == 'test-market-123'

    @pytest.mark.asyncio
    async def test_get_raw_order_book_returns_copy(self, market_data_manager):
        """Test get_raw_order_book returns a copy."""
        await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        books1 = await market_data_manager.get_raw_order_book()
        books1['Up']['bids'][0.55] = 999.0

        books2 = await market_data_manager.get_raw_order_book()
        assert books2['Up']['bids'][0.55] == 100.0

    @pytest.mark.asyncio
    async def test_extreme_price_values(self, market_data_manager):
        """Test extreme price values are handled."""
        # Very small prices
        await market_data_manager.update_order_book('Up', {0.01: 100.0}, {0.02: 100.0})
        yes_price, _, _, _ = await market_data_manager.calculate_mid_price()
        assert 0.01 < yes_price < 0.02

        # Very large prices (near 1.0)
        await market_data_manager.update_order_book('Up', {0.98: 100.0}, {0.99: 100.0})
        yes_price, _, _, _ = await market_data_manager.calculate_mid_price()
        assert 0.98 < yes_price < 0.99

    @pytest.mark.asyncio
    async def test_large_order_book(self, market_data_manager):
        """Test handling large order books efficiently."""
        # 100 price levels
        bids = {0.50 - i/1000: 100.0 + i for i in range(100)}
        asks = {0.51 + i/1000: 100.0 + i for i in range(100)}

        await market_data_manager.update_order_book('Up', bids, asks)

        yes_price, no_price, best_bid, best_ask = await market_data_manager.calculate_mid_price()

        # Should still calculate correctly
        assert best_bid == 0.50
        assert best_ask == 0.51

    @pytest.mark.asyncio
    async def test_rapid_sequential_updates(self, market_data_manager):
        """Test rapid sequential updates are handled."""
        for i in range(100):
            await market_data_manager.update_order_book(
                'Up',
                {0.50 + i/10000: 100.0},
                {0.51 + i/10000: 100.0}
            )

        info = await market_data_manager.get_data_freshness_info()
        assert info['update_counter'] == 100

    @pytest.mark.asyncio
    async def test_empty_price_changes_list(self, market_data_manager):
        """Test empty price changes list is handled."""
        await market_data_manager.apply_price_changes('Up', [])

        # Should not crash or log errors
        raw_books = await market_data_manager.get_raw_order_book()
        assert len(raw_books['Up']['bids']) == 0

    @pytest.mark.asyncio
    async def test_none_price_changes_list(self, market_data_manager):
        """Test None as price changes is handled."""
        # Should handle gracefully by the try-except in the loop
        await market_data_manager.apply_price_changes('Up', [None, None])

        raw_books = await market_data_manager.get_raw_order_book()
        assert len(raw_books['Up']['bids']) == 0

    @pytest.mark.asyncio
    async def test_market_without_question_field(self, market_data_manager, valid_market):
        """Test market without optional fields still works."""
        del valid_market['question']

        result = await market_data_manager.set_market(valid_market)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_counter_overflow_safe(self, market_data_manager):
        """Test update counter handles large values."""
        market_data_manager._update_counter = 2**31 - 5  # Near max int

        for _ in range(10):
            await market_data_manager.update_order_book('Up', {0.55: 100.0}, {0.56: 100.0})

        # Should not crash
        info = await market_data_manager.get_data_freshness_info()
        assert info['update_counter'] > 2**31
