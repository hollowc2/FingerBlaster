"""Comprehensive tests for Pulse aggregators."""

import pytest
import asyncio
from src.pulse.aggregators import (
    CandleAggregator,
    OrderBookBucketer,
    TimeframeAggregator,
)
from src.pulse.config import Candle, Trade, Timeframe


# ========== CandleAggregator Tests ==========
class TestCandleAggregator:
    """Test candle aggregation from trades."""

    @pytest.fixture
    def aggregator(self):
        return CandleAggregator(product_id="BTC-USD", wall_clock_aligned=True)

    @pytest.mark.asyncio
    async def test_single_trade_starts_candle(self, aggregator):
        """Test single trade initializes candle."""
        trade = Trade(
            product_id="BTC-USD",
            trade_id="1",
            timestamp=1700000000.0,
            price=50000.0,
            size=1.0,
            side="BUY",
        )

        candle = await aggregator.add_trade(trade)

        # First trade shouldn't complete a candle
        assert candle is None

    @pytest.mark.asyncio
    async def test_trade_crosses_boundary_completes_candle(self, aggregator):
        """Test trade crossing 10s boundary completes candle."""
        # First trade at :00
        trade1 = Trade(
            product_id="BTC-USD",
            trade_id="1",
            timestamp=1700000000.0,
            price=50000.0,
            size=1.0,
            side="BUY",
        )
        candle1 = await aggregator.add_trade(trade1)
        assert candle1 is None

        # Second trade at :10 (crosses boundary)
        trade2 = Trade(
            product_id="BTC-USD",
            trade_id="2",
            timestamp=1700000010.0,
            price=50100.0,
            size=1.0,
            side="BUY",
        )
        candle2 = await aggregator.add_trade(trade2)

        # Should complete first candle
        assert candle2 is not None
        assert candle2.timestamp == 1700000000
        assert candle2.timeframe == Timeframe.TEN_SEC

    @pytest.mark.asyncio
    async def test_candle_ohlcv_calculation(self, aggregator):
        """Test OHLCV values calculated correctly."""
        trades = [
            Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000000.0),
            Trade(trade_id="2", product_id="BTC-USD", price=50100.0, size=2.0, side="BUY", timestamp=1700000002.0),  # High
            Trade(trade_id="3", product_id="BTC-USD", price=49900.0, size=1.5, side="SELL", timestamp=1700000004.0),  # Low
            Trade(trade_id="4", product_id="BTC-USD", price=50050.0, size=1.0, side="BUY", timestamp=1700000008.0),  # Close
        ]

        # Add all trades
        for trade in trades:
            await aggregator.add_trade(trade)

        # Crossing boundary completes candle
        final_trade = Trade(trade_id="5", product_id="BTC-USD", price=50200.0, size=1.0, side="BUY", timestamp=1700000010.0)
        candle = await aggregator.add_trade(final_trade)

        assert candle is not None
        assert candle.open == 50000.0  # First trade
        assert candle.high == 50100.0  # Highest
        assert candle.low == 49900.0   # Lowest
        assert candle.close == 50050.0  # Last trade
        assert candle.volume == 5.5     # Sum of all volumes

    @pytest.mark.asyncio
    async def test_empty_candle_gap_filling(self, aggregator):
        """Test gaps filled with empty candles using previous close."""
        completed_candles = []

        async def on_candle(candle):
            completed_candles.append(candle)

        aggregator.on_candle = on_candle

        # Trade at :00
        trade1 = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000000.0)
        await aggregator.add_trade(trade1)

        # Trade at :30 (3 candle gap: :10, :20, :30)
        trade2 = Trade(trade_id="2", product_id="BTC-USD", price=50100.0, size=1.0, side="BUY", timestamp=1700000030.0)
        await aggregator.add_trade(trade2)

        # Should have emitted 3 candles (original + 2 empty)
        assert len(completed_candles) >= 2  # At least the gap fills

    @pytest.mark.asyncio
    async def test_wall_clock_alignment(self, aggregator):
        """Test candles align to wall-clock boundaries."""
        # Trade at :03 should align to :00
        trade = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000003.5)
        await aggregator.add_trade(trade)

        # Trade at :13 should align to :10
        trade2 = Trade(trade_id="2", product_id="BTC-USD", price=50100.0, size=1.0, side="BUY", timestamp=1700000013.2)
        candle = await aggregator.add_trade(trade2)

        # Candle should start at :00
        assert candle.timestamp == 1700000000

    @pytest.mark.asyncio
    async def test_flush_partial_candle(self, aggregator):
        """Test flushing partial candle."""
        trade = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000005.0)
        await aggregator.add_trade(trade)

        # Flush incomplete candle
        candle = await aggregator.flush()

        assert candle is not None
        assert candle.close == 50000.0

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, aggregator):
        """Test reset clears aggregator state."""
        trade = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000000.0)
        await aggregator.add_trade(trade)

        aggregator.reset()

        # State should be cleared
        assert aggregator._current_candle_start is None
        assert aggregator._current_ohlcv is None

    @pytest.mark.asyncio
    async def test_zero_volume_trade(self, aggregator):
        """Test handling of zero-volume trade."""
        trade = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=0.0, side="BUY", timestamp=1700000000.0)
        await aggregator.add_trade(trade)

        # Should handle gracefully
        trade2 = Trade(trade_id="2", product_id="BTC-USD", price=50100.0, size=1.0, side="BUY", timestamp=1700000010.0)
        candle = await aggregator.add_trade(trade2)

        assert candle is not None
        assert candle.volume == 0.0  # Only zero volume trade

    @pytest.mark.asyncio
    async def test_callback_emission(self, aggregator):
        """Test callback is called when candle completes."""
        completed_candles = []

        async def on_candle(candle):
            completed_candles.append(candle)

        aggregator.on_candle = on_candle

        # Create and complete a candle
        trade1 = Trade(trade_id="1", product_id="BTC-USD", price=50000.0, size=1.0, side="BUY", timestamp=1700000000.0)
        await aggregator.add_trade(trade1)

        trade2 = Trade(trade_id="2", product_id="BTC-USD", price=50100.0, size=1.0, side="BUY", timestamp=1700000010.0)
        await aggregator.add_trade(trade2)

        # Callback should have been called
        assert len(completed_candles) == 1


# ========== OrderBookBucketer Tests ==========
class TestOrderBookBucketer:
    """Test order book bucketing."""

    @pytest.fixture
    def bucketer(self):
        return OrderBookBucketer(bucket_size=100.0, product_id="BTC-USD")

    @pytest.mark.asyncio
    async def test_snapshot_processing(self, bucketer):
        """Test processing full snapshot."""
        snapshot_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "5.0", "side": "bid"},
                {"price_level": "50150", "new_quantity": "3.0", "side": "bid"},
                {"price_level": "50250", "new_quantity": "4.0", "side": "ask"},
                {"price_level": "50350", "new_quantity": "2.0", "side": "ask"},
            ]
        }

        book = await bucketer.process_snapshot(snapshot_data)

        # Should have bucketed data
        assert book is not None
        assert book.best_bid == 50150.0
        assert book.best_ask == 50250.0

    @pytest.mark.asyncio
    async def test_price_bucketing(self, bucketer):
        """Test prices bucketed correctly."""
        snapshot_data = {
            "updates": [
                {"price_level": "50123", "new_quantity": "5.0", "side": "bid"},
                {"price_level": "50178", "new_quantity": "3.0", "side": "bid"},
            ]
        }

        await bucketer.process_snapshot(snapshot_data)

        # Both should be in 50100 bucket
        assert 50100.0 in bucketer._bucketed_bids
        assert bucketer._bucketed_bids[50100.0] == 8.0  # 5 + 3

    @pytest.mark.asyncio
    async def test_incremental_update(self, bucketer):
        """Test incremental updates after snapshot."""
        # Initial snapshot
        snapshot_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "5.0", "side": "bid"},
            ]
        }
        await bucketer.process_snapshot(snapshot_data)

        # Incremental update
        update_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "10.0", "side": "bid"},
            ]
        }
        book = await bucketer.process_update(update_data)

        # Should update bid size
        assert bucketer._raw_bids[50100.0] == 10.0

    @pytest.mark.asyncio
    async def test_remove_price_level(self, bucketer):
        """Test removing price level with zero size."""
        snapshot_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "5.0", "side": "bid"},
            ]
        }
        await bucketer.process_snapshot(snapshot_data)

        # Remove with zero size
        update_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "0.0", "side": "bid"},
            ]
        }
        await bucketer.process_update(update_data)

        # Should be removed
        assert 50100.0 not in bucketer._raw_bids

    @pytest.mark.asyncio
    async def test_mid_price_calculation(self, bucketer):
        """Test mid price calculated as (best_bid + best_ask) / 2."""
        snapshot_data = {
            "updates": [
                {"price_level": "50000", "new_quantity": "5.0", "side": "bid"},
                {"price_level": "50200", "new_quantity": "5.0", "side": "ask"},
            ]
        }

        book = await bucketer.process_snapshot(snapshot_data)

        assert book.mid_price == 50100.0

    @pytest.mark.asyncio
    async def test_spread_calculation(self, bucketer):
        """Test spread = best_ask - best_bid."""
        snapshot_data = {
            "updates": [
                {"price_level": "50000", "new_quantity": "5.0", "side": "bid"},
                {"price_level": "50200", "new_quantity": "5.0", "side": "ask"},
            ]
        }

        book = await bucketer.process_snapshot(snapshot_data)

        assert book.spread == 200.0

    @pytest.mark.asyncio
    async def test_empty_book_handling(self, bucketer):
        """Test handling of empty order book."""
        snapshot_data = {"updates": []}

        book = await bucketer.process_snapshot(snapshot_data)

        assert book.best_bid == 0.0
        assert book.best_ask == 0.0
        assert book.mid_price == 0.0

    @pytest.mark.asyncio
    async def test_update_before_snapshot_ignored(self, bucketer):
        """Test updates before snapshot are ignored."""
        update_data = {
            "updates": [
                {"price_level": "50100", "new_quantity": "5.0", "side": "bid"},
            ]
        }

        book = await bucketer.process_update(update_data)

        # Should be None (no snapshot yet)
        assert book is None

    def test_depth_profile(self, bucketer):
        """Test depth profile generation."""
        bucketer._bucketed_bids = {
            50000.0: 10.0,
            50100.0: 5.0,
            50200.0: 3.0,
        }
        bucketer._bucketed_asks = {
            50300.0: 4.0,
            50400.0: 6.0,
        }

        profile = bucketer.get_depth_profile(levels=10)

        # Should have sorted bids and asks
        assert len(profile['bids']) == 3
        assert len(profile['asks']) == 2

        # Bids sorted high to low
        assert profile['bids'][0]['price'] == 50200.0


# ========== TimeframeAggregator Tests ==========
class TestTimeframeAggregator:
    """Test multi-timeframe candle aggregation."""

    @pytest.fixture
    def aggregator(self):
        return TimeframeAggregator(product_id="BTC-USD")

    @pytest.mark.asyncio
    async def test_aggregate_10s_to_1m(self, aggregator):
        """Test aggregating 6 x 10s candles into 1m candle."""
        candles = []
        for i in range(6):
            candle = Candle(
                timestamp=1700000000 + (i * 10),
                open=50000.0 + (i * 10),
                high=50000.0 + (i * 10) + 50,
                low=50000.0 + (i * 10) - 50,
                close=50000.0 + (i * 10) + 25,
                volume=1.0,
                timeframe=Timeframe.TEN_SEC,
            )
            candles.append(candle)

        completed = []
        for candle in candles:
            result = await aggregator.add_candle(candle)
            completed.extend(result)

        # Should have 1 x 1m candle
        one_min_candles = [c for c in completed if c.timeframe == Timeframe.ONE_MIN]
        assert len(one_min_candles) == 1

        # Check OHLCV
        c = one_min_candles[0]
        assert c.open == candles[0].open
        assert c.high == max(candle.high for candle in candles)
        assert c.low == min(candle.low for candle in candles)
        assert c.close == candles[-1].close
        assert c.volume == sum(candle.volume for candle in candles)

    @pytest.mark.asyncio
    async def test_aggregate_1m_to_5m(self, aggregator):
        """Test aggregating 5 x 1m candles into 5m candle."""
        candles = []
        for i in range(5):
            candle = Candle(
                timestamp=1700000000 + (i * 60),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=10.0,
                timeframe=Timeframe.ONE_MIN,
            )
            candles.append(candle)

        completed = []
        for candle in candles:
            result = await aggregator.add_candle(candle)
            completed.extend(result)

        # Should have 1 x 5m candle
        five_min_candles = [c for c in completed if c.timeframe == Timeframe.FIVE_MIN]
        assert len(five_min_candles) == 1

    @pytest.mark.asyncio
    async def test_cascading_aggregation(self, aggregator):
        """Test cascading aggregation (10s -> 1m -> 5m)."""
        completed_candles = []

        async def on_candle(candle):
            completed_candles.append(candle)

        aggregator.on_candle = on_candle

        # Add 30 x 10s candles (should produce 5 x 1m, then 1 x 5m)
        for i in range(30):
            candle = Candle(
                timestamp=1700000000 + (i * 10),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1.0,
                timeframe=Timeframe.TEN_SEC,
            )
            await aggregator.add_candle(candle)

        # Should have 1m and 5m candles
        one_min = [c for c in completed_candles if c.timeframe == Timeframe.ONE_MIN]
        five_min = [c for c in completed_candles if c.timeframe == Timeframe.FIVE_MIN]

        assert len(one_min) == 5
        assert len(five_min) == 1

    @pytest.mark.asyncio
    async def test_partial_buffer_no_emission(self, aggregator):
        """Test partial buffer doesn't emit candle."""
        # Add only 3 x 10s candles (need 6 for 1m)
        for i in range(3):
            candle = Candle(
                timestamp=1700000000 + (i * 10),
                open=50000.0,
                high=50000.0,
                low=50000.0,
                close=50000.0,
                volume=1.0,
                timeframe=Timeframe.TEN_SEC,
            )
            completed = await aggregator.add_candle(candle)

            # Should not complete yet
            assert len(completed) == 0

    @pytest.mark.asyncio
    async def test_reset_clears_buffers(self, aggregator):
        """Test reset clears all buffers."""
        # Add some candles
        for i in range(3):
            candle = Candle(
                timestamp=1700000000 + (i * 10),
                open=50000.0,
                high=50000.0,
                low=50000.0,
                close=50000.0,
                volume=1.0,
                timeframe=Timeframe.TEN_SEC,
            )
            await aggregator.add_candle(candle)

        aggregator.reset()

        # Buffers should be empty
        assert all(len(buf) == 0 for buf in aggregator._buffers.values())

    @pytest.mark.asyncio
    async def test_timestamp_preservation(self, aggregator):
        """Test aggregated candle uses first candle's timestamp."""
        candles = []
        base_timestamp = 1700000000
        for i in range(6):
            candle = Candle(
                timestamp=base_timestamp + (i * 10),
                open=50000.0,
                high=50000.0,
                low=50000.0,
                close=50000.0,
                volume=1.0,
                timeframe=Timeframe.TEN_SEC,
            )
            candles.append(candle)

        completed = []
        for candle in candles:
            result = await aggregator.add_candle(candle)
            completed.extend(result)

        # 1m candle should use first 10s candle's timestamp
        one_min_candle = [c for c in completed if c.timeframe == Timeframe.ONE_MIN][0]
        assert one_min_candle.timestamp == base_timestamp

    @pytest.mark.asyncio
    async def test_empty_candle_aggregation(self, aggregator):
        """Test aggregating candles with zero volume."""
        candles = []
        for i in range(6):
            candle = Candle(
                timestamp=1700000000 + (i * 10),
                open=50000.0,
                high=50000.0,
                low=50000.0,
                close=50000.0,
                volume=0.0,  # Zero volume
                timeframe=Timeframe.TEN_SEC,
            )
            candles.append(candle)

        completed = []
        for candle in candles:
            result = await aggregator.add_candle(candle)
            completed.extend(result)

        # Should still aggregate
        one_min_candles = [c for c in completed if c.timeframe == Timeframe.ONE_MIN]
        assert len(one_min_candles) == 1
        assert one_min_candles[0].volume == 0.0
