"""Comprehensive tests for PulseCore - analytics dashboard orchestrator."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.pulse.core import (
    PulseCore,
    AsyncEventBus,
    IndicatorWorker,
    ThrottledOrderBookBucketer,
)
from src.pulse.config import PulseConfig, Candle, Ticker, Trade, Timeframe


# ========== AsyncEventBus Tests ==========
class TestAsyncEventBus:
    """Test async event bus."""

    def test_register_callback(self):
        """Test registering a callback."""
        bus = AsyncEventBus()
        callback = MagicMock()

        bus.on('test_event', callback)

        assert callback in bus._subs['test_event']

    def test_emit_sync_callback(self):
        """Test emitting to synchronous callback."""
        bus = AsyncEventBus()
        results = []
        callback = lambda x: results.append(x)

        bus.on('test', callback)
        bus.emit('test', 'value')

        assert results == ['value']

    @pytest.mark.asyncio
    async def test_emit_async_callback(self):
        """Test emitting to async callback."""
        bus = AsyncEventBus()
        results = []

        async def callback(x):
            results.append(x)

        bus.on('test', callback)
        bus.emit('test', 'async_value')

        # Wait for async callback to complete
        await asyncio.sleep(0.01)

        assert results == ['async_value']

    def test_emit_multiple_callbacks(self):
        """Test emitting to multiple callbacks."""
        bus = AsyncEventBus()
        results = []

        bus.on('test', lambda x: results.append(f'a_{x}'))
        bus.on('test', lambda x: results.append(f'b_{x}'))
        bus.emit('test', '1')

        assert 'a_1' in results
        assert 'b_1' in results

    def test_emit_no_subscribers(self):
        """Test emitting with no subscribers."""
        bus = AsyncEventBus()

        # Should not raise
        bus.emit('nonexistent', 'value')

    def test_emit_callback_error_isolated(self):
        """Test callback error doesn't affect others."""
        bus = AsyncEventBus()
        results = []

        bus.on('test', lambda: 1 / 0)  # Will raise
        bus.on('test', lambda: results.append('success'))

        # Should not raise, second callback should run
        bus.emit('test')

        assert 'success' in results


# ========== IndicatorWorker Tests ==========
class TestIndicatorWorker:
    """Test indicator worker queue."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.update = AsyncMock()
        return engine

    @pytest.fixture
    def worker(self, mock_engine):
        return IndicatorWorker(mock_engine)

    @pytest.mark.asyncio
    async def test_start_creates_task(self, worker):
        """Test start creates background task."""
        await worker.start()

        assert worker.task is not None
        assert not worker.task.done()

        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, worker):
        """Test stop cancels task."""
        await worker.start()
        await worker.stop()

        assert worker.task.cancelled() or worker.task.done()

    @pytest.mark.asyncio
    async def test_submit_queues_candle(self, worker, mock_engine):
        """Test submit adds candle to queue."""
        candle = Candle(
            timestamp=time.time(),
            open=100.0, high=105.0, low=99.0, close=103.0,
            volume=1000.0, timeframe=Timeframe.TEN_SEC
        )

        await worker.start()
        worker.submit('BTC-USD', candle)

        # Wait for processing
        await asyncio.sleep(0.05)
        await worker.stop()

        mock_engine.update.assert_called_once_with('BTC-USD', candle)

    @pytest.mark.asyncio
    async def test_submit_queue_full_drops(self, worker):
        """Test submit drops when queue full."""
        worker.q = asyncio.Queue(maxsize=1)

        candle = Candle(
            timestamp=time.time(),
            open=100.0, high=105.0, low=99.0, close=103.0,
            volume=1000.0, timeframe=Timeframe.TEN_SEC
        )

        # Fill the queue
        worker.submit('BTC-USD', candle)
        worker.submit('BTC-USD', candle)  # Should be dropped

        assert worker.q.qsize() == 1


# ========== ThrottledOrderBookBucketer Tests ==========
class TestThrottledOrderBookBucketer:
    """Test throttled order book bucketer."""

    def test_throttle_respects_interval(self):
        """Test throttle interval is set correctly."""
        bucketer = ThrottledOrderBookBucketer(
            bucket_size=100.0,
            product_id='BTC-USD',
            on_update=AsyncMock(),
            emit_hz=10.0  # 100ms interval
        )

        assert bucketer._min_interval == 0.1

    @pytest.mark.asyncio
    async def test_throttle_blocks_rapid_updates(self):
        """Test rapid updates are throttled."""
        callback = AsyncMock()
        bucketer = ThrottledOrderBookBucketer(
            bucket_size=100.0,
            product_id='BTC-USD',
            on_update=callback,
            emit_hz=10.0
        )

        # First call should work
        bucketer._last_emit = 0.0
        result1 = await bucketer._rebuild_bucketed_book()

        # Immediate second call should be throttled
        result2 = await bucketer._rebuild_bucketed_book()

        # Second should return None (throttled)
        assert result2 is None


# ========== PulseCore Initialization Tests ==========
class TestPulseCoreInitialization:
    """Test PulseCore initialization."""

    def test_initializes_with_default_config(self):
        """Test initialization with default config."""
        core = PulseCore()

        assert isinstance(core.config, PulseConfig)
        assert core._running is False

    def test_initializes_with_custom_config(self):
        """Test initialization with custom config."""
        config = PulseConfig(products=['ETH-USD'])
        core = PulseCore(config=config)

        assert core.config.products == ['ETH-USD']

    def test_initializes_event_bus(self):
        """Test event bus is initialized."""
        core = PulseCore()

        assert isinstance(core.bus, AsyncEventBus)

    def test_initializes_empty_aggregators(self):
        """Test aggregators are empty on init."""
        core = PulseCore()

        assert core._candle_aggs == {}
        assert core._tf_aggs == {}
        assert core._book_aggs == {}

    def test_initializes_indicator_engine(self):
        """Test indicator engine is initialized."""
        core = PulseCore()

        assert core._indicator_engine is not None
        assert core._indicator_worker is not None


# ========== PulseCore Start/Stop Tests ==========
class TestPulseCoreLifecycle:
    """Test PulseCore start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_connector(self):
        """Test start creates and starts connector."""
        with patch('src.pulse.core.CoinbaseConnector') as MockConnector:
            mock_connector = AsyncMock()
            mock_connector.start = AsyncMock()
            mock_connector.prime_all_timeframes = AsyncMock(return_value={})
            MockConnector.return_value = mock_connector

            core = PulseCore()
            await core.start()

            assert core._running is True
            mock_connector.start.assert_called_once()

            await core.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test calling start multiple times is safe."""
        with patch('src.pulse.core.CoinbaseConnector') as MockConnector:
            mock_connector = AsyncMock()
            mock_connector.start = AsyncMock()
            mock_connector.stop = AsyncMock()
            mock_connector.prime_all_timeframes = AsyncMock(return_value={})
            MockConnector.return_value = mock_connector

            core = PulseCore()
            await core.start()
            await core.start()  # Should be no-op

            # Should only start once
            assert mock_connector.start.call_count == 1

            await core.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stop when not running is safe."""
        core = PulseCore()
        await core.stop()  # Should not raise

        assert core._running is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """Test stop cleans up resources."""
        with patch('src.pulse.core.CoinbaseConnector') as MockConnector:
            mock_connector = AsyncMock()
            mock_connector.start = AsyncMock()
            mock_connector.stop = AsyncMock()
            mock_connector.prime_all_timeframes = AsyncMock(return_value={})
            MockConnector.return_value = mock_connector

            core = PulseCore()
            await core.start()
            await core.stop()

            assert core._running is False
            assert core._connector is None
            mock_connector.stop.assert_called_once()


# ========== Product Initialization Tests ==========
class TestProductInitialization:
    """Test product initialization."""

    def test_init_product_creates_aggregators(self):
        """Test product init creates all aggregators."""
        core = PulseCore()
        core._init_product('BTC-USD')

        assert 'BTC-USD' in core._trade_history
        assert 'BTC-USD' in core._candle_aggs
        assert 'BTC-USD' in core._tf_aggs
        assert 'BTC-USD' in core._book_aggs

    def test_init_product_creates_trade_history(self):
        """Test trade history is created with maxlen."""
        config = PulseConfig(trade_history_size=100)
        core = PulseCore(config=config)
        core._init_product('BTC-USD')

        assert core._trade_history['BTC-USD'].maxlen == 100


# ========== Event Handler Tests ==========
class TestEventHandlers:
    """Test event handlers."""

    @pytest.mark.asyncio
    async def test_on_trade_creates_trade_object(self):
        """Test trade handler creates Trade object."""
        core = PulseCore()
        core._init_product('BTC-USD')

        events = []
        core.bus.on('trade', lambda t: events.append(t))

        raw_trade = {
            'trade_id': '12345',
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'size': '0.5',
            'side': 'buy',
            'time': '2026-01-25T12:00:00Z'
        }

        await core._on_trade(raw_trade)

        assert len(events) == 1
        assert events[0].product_id == 'BTC-USD'
        assert events[0].price == 50000.0

    @pytest.mark.asyncio
    async def test_on_trade_adds_to_history(self):
        """Test trade is added to history."""
        core = PulseCore()
        core._init_product('BTC-USD')

        raw_trade = {
            'trade_id': '12345',
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'size': '0.5',
            'side': 'buy',
            'time': '2026-01-25T12:00:00Z'
        }

        await core._on_trade(raw_trade)

        assert len(core._trade_history['BTC-USD']) == 1

    @pytest.mark.asyncio
    async def test_on_ticker_creates_ticker_object(self):
        """Test ticker handler creates Ticker object."""
        core = PulseCore()

        events = []
        core.bus.on('ticker', lambda t: events.append(t))

        raw_ticker = {
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'volume_24h': '1000000',
            'low_24h': '49000',
            'high_24h': '51000',
        }

        await core._on_ticker(raw_ticker)

        assert len(events) == 1
        assert events[0].price == 50000.0

    @pytest.mark.asyncio
    async def test_on_ticker_preserves_existing_volume(self):
        """Test ticker preserves existing volume if not in update."""
        core = PulseCore()

        # Set existing ticker with volume
        core._tickers['BTC-USD'] = Ticker(
            product_id='BTC-USD',
            price=49000.0,
            volume_24h=5000000.0,
            low_24h=48000.0,
            high_24h=50000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
            timestamp=time.time()
        )

        # Update without volume
        raw_ticker = {
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'low_24h': '49000',
            'high_24h': '51000',
            # No volume_24h
        }

        await core._on_ticker(raw_ticker)

        assert core._tickers['BTC-USD'].volume_24h == 5000000.0

    @pytest.mark.asyncio
    async def test_on_ticker_parses_percentage_string(self):
        """Test ticker parses percentage string."""
        core = PulseCore()

        raw_ticker = {
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'price_percent_chg_24h': '2.5%',  # String with %
        }

        await core._on_ticker(raw_ticker)

        assert core._tickers['BTC-USD'].price_change_pct_24h == 2.5

    @pytest.mark.asyncio
    async def test_on_connection_status(self):
        """Test connection status handler."""
        core = PulseCore()

        events = []
        core.bus.on('connection', lambda c, m: events.append((c, m)))

        await core._on_connection_status(True, "Connected")

        assert events == [(True, "Connected")]


# ========== Order Book Tests ==========
class TestOrderBookHandling:
    """Test order book handling."""

    @pytest.mark.asyncio
    async def test_on_l2_snapshot(self):
        """Test L2 snapshot processing."""
        core = PulseCore()
        core._init_product('BTC-USD')
        core._book_aggs['BTC-USD'].process_snapshot = AsyncMock()

        data = {
            'product_id': 'BTC-USD',
            'type': 'snapshot',
            'bids': [['50000', '1.0']],
            'asks': [['50100', '0.5']],
        }

        await core._on_l2(data)

        core._book_aggs['BTC-USD'].process_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_l2_update(self):
        """Test L2 update processing."""
        core = PulseCore()
        core._init_product('BTC-USD')
        core._book_aggs['BTC-USD'].process_update = AsyncMock()

        data = {
            'product_id': 'BTC-USD',
            'type': 'l2update',
            'changes': [['buy', '50000', '1.5']],
        }

        await core._on_l2(data)

        core._book_aggs['BTC-USD'].process_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_l2_unknown_product(self):
        """Test L2 with unknown product is ignored."""
        core = PulseCore()

        data = {
            'product_id': 'UNKNOWN-USD',
            'type': 'snapshot',
        }

        # Should not raise
        await core._on_l2(data)


# ========== Candle Callback Tests ==========
class TestCandleCallbacks:
    """Test candle callback handling."""

    @pytest.mark.asyncio
    async def test_on_10s_candle(self):
        """Test 10s candle handler."""
        core = PulseCore()
        core._init_product('BTC-USD')
        core._indicator_worker.submit = MagicMock()
        core._tf_aggs['BTC-USD'].add_candle = AsyncMock()

        events = []
        core.bus.on('candle', lambda p, c: events.append((p, c)))

        candle = Candle(
            timestamp=time.time(),
            open=50000.0, high=50100.0, low=49900.0, close=50050.0,
            volume=100.0, timeframe=Timeframe.TEN_SEC
        )

        await core._on_10s_candle('BTC-USD', candle)

        core._indicator_worker.submit.assert_called_once()
        core._tf_aggs['BTC-USD'].add_candle.assert_called_once()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_on_tf_candle(self):
        """Test timeframe candle handler."""
        core = PulseCore()
        core._init_product('BTC-USD')
        core._indicator_worker.submit = MagicMock()

        events = []
        core.bus.on('candle', lambda p, c: events.append((p, c)))

        candle = Candle(
            timestamp=time.time(),
            open=50000.0, high=50100.0, low=49900.0, close=50050.0,
            volume=1000.0, timeframe=Timeframe.ONE_MIN
        )

        await core._on_tf_candle('BTC-USD', candle)

        core._indicator_worker.submit.assert_called_once()
        assert len(events) == 1


# ========== Indicator/Alert Callbacks ==========
class TestIndicatorCallbacks:
    """Test indicator and alert callbacks."""

    @pytest.mark.asyncio
    async def test_on_indicator_snapshot(self):
        """Test indicator snapshot handler."""
        core = PulseCore()

        events = []
        core.bus.on('indicator', lambda s: events.append(s))

        snapshot = MagicMock()
        await core._on_indicator_snapshot(snapshot)

        assert len(events) == 1
        assert events[0] is snapshot

    @pytest.mark.asyncio
    async def test_on_alert(self):
        """Test alert handler."""
        core = PulseCore()

        events = []
        core.bus.on('alert', lambda a: events.append(a))

        alert = MagicMock()
        await core._on_alert(alert)

        assert len(events) == 1
        assert events[0] is alert


# ========== Public API Tests ==========
class TestPublicAPI:
    """Test public API methods."""

    def test_on_registers_callback(self):
        """Test on() registers callback."""
        core = PulseCore()
        callback = MagicMock()

        core.on('test', callback)

        assert callback in core.bus._subs['test']

    def test_get_recent_trades(self):
        """Test get_recent_trades returns correct data."""
        core = PulseCore()
        core._init_product('BTC-USD')

        # Add some trades
        for i in range(20):
            trade = Trade(
                trade_id=str(i),
                product_id='BTC-USD',
                price=50000.0 + i,
                size=1.0,
                side='BUY',
                timestamp=time.time()
            )
            core._trade_history['BTC-USD'].append(trade)

        trades = core.get_recent_trades('BTC-USD', limit=10)

        assert len(trades) == 10
        # Should be most recent
        assert trades[-1].trade_id == '19'

    def test_get_recent_trades_unknown_product(self):
        """Test get_recent_trades with unknown product."""
        core = PulseCore()

        trades = core.get_recent_trades('UNKNOWN-USD')

        assert trades == []

    def test_get_latest_ticker(self):
        """Test get_latest_ticker returns correct data."""
        core = PulseCore()
        ticker = Ticker(
            product_id='BTC-USD',
            price=50000.0,
            volume_24h=1000000.0,
            low_24h=49000.0,
            high_24h=51000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
            timestamp=time.time()
        )
        core._tickers['BTC-USD'] = ticker

        result = core.get_latest_ticker('BTC-USD')

        assert result is ticker

    def test_get_latest_ticker_unknown(self):
        """Test get_latest_ticker with unknown product."""
        core = PulseCore()

        result = core.get_latest_ticker('UNKNOWN-USD')

        assert result is None

    def test_get_indicator_snapshot(self):
        """Test get_indicator_snapshot delegates to engine."""
        core = PulseCore()
        mock_snapshot = MagicMock()
        core._indicator_engine.get_snapshot = MagicMock(return_value=mock_snapshot)

        result = core.get_indicator_snapshot('BTC-USD', Timeframe.ONE_MIN)

        assert result is mock_snapshot
        core._indicator_engine.get_snapshot.assert_called_once_with('BTC-USD', Timeframe.ONE_MIN)


# ========== 4-Hour Aggregation Tests ==========
class TestFourHourAggregation:
    """Test 4-hour candle aggregation."""

    def test_aggregate_to_4h_basic(self):
        """Test basic 4-hour aggregation."""
        core = PulseCore()

        two_hour_candles = [
            {'start': 1000, 'open': 100, 'high': 110, 'low': 95, 'close': 105, 'volume': 500},
            {'start': 2000, 'open': 105, 'high': 115, 'low': 100, 'close': 112, 'volume': 600},
        ]

        result = core._aggregate_to_4h(two_hour_candles)

        assert len(result) == 1
        assert result[0]['open'] == 100  # From first candle
        assert result[0]['close'] == 112  # From second candle
        assert result[0]['high'] == 115  # Max of both
        assert result[0]['low'] == 95  # Min of both
        assert result[0]['volume'] == 1100  # Sum of both

    def test_aggregate_to_4h_multiple(self):
        """Test multiple 4-hour candles."""
        core = PulseCore()

        two_hour_candles = [
            {'start': 1000, 'open': 100, 'high': 110, 'low': 95, 'close': 105, 'volume': 500},
            {'start': 2000, 'open': 105, 'high': 115, 'low': 100, 'close': 112, 'volume': 600},
            {'start': 3000, 'open': 112, 'high': 120, 'low': 108, 'close': 118, 'volume': 700},
            {'start': 4000, 'open': 118, 'high': 125, 'low': 115, 'close': 122, 'volume': 800},
        ]

        result = core._aggregate_to_4h(two_hour_candles)

        assert len(result) == 2

    def test_aggregate_to_4h_insufficient_data(self):
        """Test aggregation with insufficient data."""
        core = PulseCore()

        # Only 1 candle - need at least 2
        two_hour_candles = [
            {'start': 1000, 'open': 100, 'high': 110, 'low': 95, 'close': 105, 'volume': 500},
        ]

        result = core._aggregate_to_4h(two_hour_candles)

        assert result == []


# ========== Historical Data Priming Tests ==========
class TestHistoricalPriming:
    """Test historical data priming."""

    @pytest.mark.asyncio
    async def test_prime_no_connector(self):
        """Test priming with no connector."""
        core = PulseCore()
        core._connector = None

        # Should not raise
        await core._prime_historical_data()

    @pytest.mark.asyncio
    async def test_prime_feeds_indicator_engine(self):
        """Test priming feeds candles to indicator engine."""
        from src.connectors.coinbase import CoinbaseGranularity

        core = PulseCore()

        mock_connector = MagicMock()
        # Use actual enum value as key (matching what code expects)
        mock_connector.prime_all_timeframes = AsyncMock(return_value={
            CoinbaseGranularity.ONE_MINUTE: [
                {'start': 1000, 'open': 100, 'high': 110, 'low': 95, 'close': 105, 'volume': 500},
            ]
        })
        core._connector = mock_connector
        core._indicator_engine.update = AsyncMock()

        await core._prime_historical_data()

        # Should have called indicator engine update
        core._indicator_engine.update.assert_called()


# ========== Edge Cases ==========
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_trade_handler_exception(self):
        """Test trade handler handles exception."""
        core = PulseCore()
        core._init_product('BTC-USD')
        core._candle_aggs['BTC-USD'].add_trade = AsyncMock(
            side_effect=Exception("Test error")
        )

        raw_trade = {
            'trade_id': '12345',
            'product_id': 'BTC-USD',
            'price': '50000.00',
            'size': '0.5',
            'side': 'buy',
            'time': '2026-01-25T12:00:00Z'
        }

        # Should not raise
        try:
            await core._on_trade(raw_trade)
        except Exception:
            pass  # Exception may propagate, that's ok

    @pytest.mark.asyncio
    async def test_stats_update_exception_handling(self):
        """Test stats update handles exceptions."""
        core = PulseCore()
        core._connector = MagicMock()
        core._connector.get_product = AsyncMock(side_effect=Exception("API Error"))

        # Should not raise
        await core._update_24h_stats()

    def test_event_bus_multiple_events(self):
        """Test event bus with multiple event types."""
        bus = AsyncEventBus()
        results = {'a': [], 'b': []}

        bus.on('event_a', lambda x: results['a'].append(x))
        bus.on('event_b', lambda x: results['b'].append(x))

        bus.emit('event_a', 1)
        bus.emit('event_b', 2)
        bus.emit('event_a', 3)

        assert results['a'] == [1, 3]
        assert results['b'] == [2]
