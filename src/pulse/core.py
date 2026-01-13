# Refactored Pulse Core – Textual-safe, non-blocking
# Key guarantees:
# - UI callbacks are fire-and-forget
# - Indicator math runs off the UI event loop
# - Order book updates are throttled
# - Shutdown and resize are always responsive

import asyncio
import contextlib
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, Deque, List, Optional

from src.connectors.coinbase import CoinbaseConnector, CoinbaseConfig, CoinbaseGranularity
from src.pulse.aggregators import CandleAggregator, OrderBookBucketer, TimeframeAggregator
from src.pulse.config import (
    Alert,
    BucketedOrderBook,
    Candle,
    IndicatorSnapshot,
    PulseConfig,
    Ticker,
    Timeframe,
    Trade,
)
from src.pulse.indicators import IndicatorEngine

logger = logging.getLogger("Pulse.Core")

# -----------------------------------------------------------------------------
# Async Event Bus (never blocks UI)
# -----------------------------------------------------------------------------

class AsyncEventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable]] = {}

    def on(self, event: str, cb: Callable):
        self._subs.setdefault(event, []).append(cb)

    def emit(self, event: str, *args):
        for cb in self._subs.get(event, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(*args))
                else:
                    cb(*args)
            except Exception:
                logger.exception("event handler error")

# -----------------------------------------------------------------------------
# Indicator Worker (CPU isolation)
# -----------------------------------------------------------------------------

class IndicatorWorker:
    def __init__(self, engine: IndicatorEngine):
        self.engine = engine
        self.q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    def submit(self, product_id: str, candle: Candle):
        try:
            self.q.put_nowait((product_id, candle))
        except asyncio.QueueFull:
            logger.warning("Indicator queue full – dropping candle")

    async def _run(self):
        while True:
            product_id, candle = await self.q.get()
            try:
                await self.engine.update(product_id, candle)
            except Exception:
                logger.exception("Indicator update failed")

# -----------------------------------------------------------------------------
# Throttled Order Book Wrapper
# -----------------------------------------------------------------------------

class ThrottledOrderBookBucketer(OrderBookBucketer):
    def __init__(self, *args, emit_hz: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval = 1.0 / emit_hz
        self._last_emit = 0.0

    async def _rebuild_bucketed_book(self) -> Optional[BucketedOrderBook]:
        now = time.time()
        if now - self._last_emit < self._min_interval:
            return None
        self._last_emit = now
        return await super()._rebuild_bucketed_book()

# -----------------------------------------------------------------------------
# Pulse Core
# -----------------------------------------------------------------------------

class PulseCore:
    def __init__(self, config: Optional[PulseConfig] = None):
        self.config = config or PulseConfig()
        self.bus = AsyncEventBus()

        self._connector: Optional[CoinbaseConnector] = None
        self._running = False

        self._candle_aggs: Dict[str, CandleAggregator] = {}
        self._tf_aggs: Dict[str, TimeframeAggregator] = {}
        self._book_aggs: Dict[str, ThrottledOrderBookBucketer] = {}

        self._trade_history: Dict[str, Deque[Trade]] = {}
        self._tickers: Dict[str, Ticker] = {}

        self._indicator_engine = IndicatorEngine(
            config=self.config,
            on_indicator_update=self._on_indicator_snapshot,
            on_alert=self._on_alert,
        )
        self._indicator_worker = IndicatorWorker(self._indicator_engine)

        # Task for periodic 24h stats updates
        self._stats_update_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if self._running:
            return
        self._running = True

        await self._indicator_worker.start()

        self._connector = CoinbaseConnector(
            CoinbaseConfig(
                ws_reconnect_delay=self.config.ws_reconnect_delay,
                ws_max_reconnect_attempts=self.config.ws_max_reconnect_attempts,
                rest_rate_limit_per_sec=self.config.rest_rate_limit_per_sec,
            )
        )

        self._connector.on_trade = self._on_trade
        self._connector.on_l2_update = self._on_l2
        self._connector.on_ticker = self._on_ticker
        self._connector.on_connection_status = self._on_connection_status

        for product in self.config.products:
            self._init_product(product)

        # Prime historical data before starting WebSocket
        await self._prime_historical_data()

        await self._connector.start(
            product_ids=self.config.products,
            channels=["market_trades", "level2", "ticker"],
        )

        # Start periodic 24h stats update
        self._stats_update_task = asyncio.create_task(self._periodic_stats_update())

    async def stop(self):
        if not self._running:
            return
        self._running = False

        # Stop stats update task
        if self._stats_update_task:
            self._stats_update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stats_update_task
            self._stats_update_task = None

        if self._connector:
            await self._connector.stop()
            self._connector = None

        await self._indicator_worker.stop()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_product(self, product_id: str):
        self._trade_history[product_id] = deque(maxlen=self.config.trade_history_size)

        self._candle_aggs[product_id] = CandleAggregator(
            product_id=product_id,
            on_candle=lambda c, pid=product_id: asyncio.create_task(self._on_10s_candle(pid, c)),
        )

        self._tf_aggs[product_id] = TimeframeAggregator(
            product_id=product_id,
            on_candle=lambda c, pid=product_id: asyncio.create_task(self._on_tf_candle(pid, c)),
        )

        self._book_aggs[product_id] = ThrottledOrderBookBucketer(
            bucket_size=self.config.bucket_size_usd,
            product_id=product_id,
            on_update=lambda b, pid=product_id: asyncio.create_task(self._on_orderbook_update(pid, b)),
        )

    def _aggregate_to_4h(self, two_hour_candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate TWO_HOUR candles into FOUR_HOUR candles.

        Args:
            two_hour_candles: List of 2-hour candles (sorted by timestamp)

        Returns:
            List of 4-hour candles
        """
        if len(two_hour_candles) < 2:
            return []

        four_hour_candles = []

        # Process pairs of 2-hour candles
        for i in range(0, len(two_hour_candles) - 1, 2):
            c1 = two_hour_candles[i]
            c2 = two_hour_candles[i + 1]

            # Aggregate into 4-hour candle
            four_hour_candles.append({
                "start": c1["start"],  # Use timestamp of first candle
                "open": c1["open"],
                "high": max(c1["high"], c2["high"]),
                "low": min(c1["low"], c2["low"]),
                "close": c2["close"],
                "volume": c1["volume"] + c2["volume"],
            })

        return four_hour_candles

    async def _prime_historical_data(self):
        """
        Prime indicators with historical data before starting live WebSocket.

        Fetches historical candles for each enabled timeframe and feeds them
        through the indicator engine to warm up calculations.
        """
        if not self._connector:
            logger.warning("Cannot prime historical data: connector not initialized")
            return

        logger.info("Starting historical data priming...")

        # Map Pulse timeframes to Coinbase granularities (skip TEN_SEC as it's local-only)
        timeframe_to_granularity = {
            Timeframe.ONE_MIN: CoinbaseGranularity.ONE_MINUTE,
            Timeframe.FIVE_MIN: CoinbaseGranularity.FIVE_MINUTE,
            Timeframe.FIFTEEN_MIN: CoinbaseGranularity.FIFTEEN_MINUTE,
            Timeframe.ONE_HOUR: CoinbaseGranularity.ONE_HOUR,
            Timeframe.FOUR_HOUR: CoinbaseGranularity.TWO_HOUR,  # Will aggregate 2x2h candles
            Timeframe.ONE_DAY: CoinbaseGranularity.ONE_DAY,
        }

        # Get enabled timeframes that can be primed (excluding TEN_SEC)
        timeframes_to_prime = [
            tf for tf in self.config.enabled_timeframes
            if tf != Timeframe.TEN_SEC
        ]

        if not timeframes_to_prime:
            logger.info("No timeframes to prime (only TEN_SEC enabled)")
            return

        # Get corresponding granularities
        granularities = [timeframe_to_granularity[tf] for tf in timeframes_to_prime]

        # Prime each product
        for product_id in self.config.products:
            logger.info(f"Priming {product_id} with {len(granularities)} timeframes...")

            try:
                # Fetch historical candles for all timeframes
                results = await self._connector.prime_all_timeframes(
                    product_id=product_id,
                    granularities=granularities,
                    bars_per_tf=self.config.prime_bars_per_timeframe,
                    parallel=self.config.prime_parallel,
                    smallest_first=self.config.prime_smallest_first,
                )

                # Feed candles through indicator engine
                for tf in timeframes_to_prime:
                    granularity = timeframe_to_granularity[tf]
                    candles_data = results.get(granularity, [])

                    if not candles_data:
                        logger.warning(f"No historical data for {product_id} {tf.value}")
                        continue

                    # Handle FOUR_HOUR special case (aggregate 2x TWO_HOUR candles)
                    if tf == Timeframe.FOUR_HOUR:
                        candles_data = self._aggregate_to_4h(candles_data)
                        if not candles_data:
                            logger.warning(f"Failed to aggregate 4h candles for {product_id}")
                            continue

                    # Convert Coinbase candle format to Pulse Candle objects
                    for raw_candle in candles_data:
                        candle = Candle(
                            timestamp=raw_candle["start"],
                            open=raw_candle["open"],
                            high=raw_candle["high"],
                            low=raw_candle["low"],
                            close=raw_candle["close"],
                            volume=raw_candle["volume"],
                            timeframe=tf,
                        )

                        # Submit to indicator engine (bypasses candle callback emission)
                        await self._indicator_engine.update(product_id, candle)

                    logger.info(
                        f"Primed {len(candles_data)} candles for {product_id} {tf.value}"
                    )

            except Exception as e:
                logger.error(f"Failed to prime {product_id}: {e}", exc_info=True)

        logger.info("Historical data priming complete")

    # ------------------------------------------------------------------
    # WebSocket Handlers
    # ------------------------------------------------------------------

    async def _on_trade(self, raw: Dict[str, Any]):
        ts = datetime.fromisoformat(raw["time"].replace("Z", "+00:00")).timestamp()
        trade = Trade(
            trade_id=str(raw.get("trade_id")),
            product_id=raw.get("product_id"),
            price=float(raw.get("price")),
            size=float(raw.get("size")),
            side=raw.get("side", "").upper(),
            timestamp=ts,
        )

        self._trade_history[trade.product_id].append(trade)
        await self._candle_aggs[trade.product_id].add_trade(trade)
        self.bus.emit("trade", trade)

    async def _on_l2(self, data: Dict[str, Any]):
        product = data.get("product_id")
        agg = self._book_aggs.get(product)
        if not agg:
            return
        if data.get("type") == "snapshot":
            await agg.process_snapshot(data)
        else:
            await agg.process_update(data)

    async def _on_ticker(self, raw: Dict[str, Any]):
        product_id = raw.get("product_id")

        # Preserve existing data from REST API if WebSocket doesn't provide it
        # WebSocket ticker often doesn't include volume_24h or price_change_pct_24h
        existing_volume = 0.0
        existing_price_change_pct = 0.0
        if product_id in self._tickers:
            existing_volume = self._tickers[product_id].volume_24h
            existing_price_change_pct = self._tickers[product_id].price_change_pct_24h

        # Use volume from WebSocket if available, otherwise preserve existing
        ws_volume = raw.get("volume_24h")
        volume_24h = float(ws_volume) if ws_volume else existing_volume

        # Extract price change percentage from WebSocket
        # Coinbase ticker channel may provide this as a STRING with % sign or as a number
        # Try multiple possible field names
        price_pct_raw = (
            raw.get("price_percent_chg_24h") or
            raw.get("price_percentage_change_24h") or
            raw.get("price_change_pct_24h") or
            raw.get("price_pct_change_24h") or
            raw.get("change_pct")
        )

        # Parse the percentage if WebSocket provides it, otherwise preserve existing
        if price_pct_raw is not None:
            # Parse the percentage (handle both string "9.43%" and numeric 9.43)
            try:
                if isinstance(price_pct_raw, str):
                    price_change_pct_24h = float(price_pct_raw.rstrip('%'))
                else:
                    price_change_pct_24h = float(price_pct_raw)
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse WebSocket price percentage: {price_pct_raw}")
                price_change_pct_24h = existing_price_change_pct
        else:
            # WebSocket didn't provide it, preserve existing value from REST API
            price_change_pct_24h = existing_price_change_pct

        # Calculate absolute price change from percentage and current price
        current_price = float(raw.get("price", 0))
        if price_change_pct_24h != 0 and current_price > 0:
            # price_change_24h = current_price * (price_change_pct_24h / 100)
            # But we can also calculate it from 24h high/low if percentage not available
            price_change_24h = current_price * (price_change_pct_24h / 100.0)
        else:
            price_change_24h = 0.0

        ticker = Ticker(
            product_id=product_id,
            price=current_price,
            volume_24h=volume_24h,
            low_24h=float(raw.get("low_24h", 0) or 0),
            high_24h=float(raw.get("high_24h", 0) or 0),
            price_change_24h=price_change_24h,
            price_change_pct_24h=price_change_pct_24h,
            timestamp=time.time(),
        )
        self._tickers[ticker.product_id] = ticker
        self.bus.emit("ticker", ticker)

    async def _on_connection_status(self, connected: bool, message: str):
        """Handle WebSocket connection status changes."""
        self.bus.emit("connection", connected, message)

    async def _on_orderbook_update(self, product_id: str, book: BucketedOrderBook):
        """Handle order book updates."""
        self.bus.emit("orderbook", product_id, book)

    async def _update_24h_stats(self):
        """Fetch 24h stats from REST API and update tickers with volume data."""
        if not self._connector:
            return

        # Fetch product details for each product
        for product_id in self.config.products:
            try:
                # Get product details which includes volume_24h
                product_data = await self._connector.get_product(product_id)

                if not product_data:
                    logger.debug(f"No product data received for {product_id}")
                    continue

                # Extract volume_24h from the response
                # Prefer approximate_quote_24h_volume (in USD) over volume_24h (in base currency)
                volume_24h = 0.0

                # Try approximate_quote_24h_volume first (USD value)
                if "approximate_quote_24h_volume" in product_data:
                    volume_24h_raw = product_data.get("approximate_quote_24h_volume", 0)
                    logger.debug(f"Found approximate_quote_24h_volume: {volume_24h_raw}")
                    volume_24h = float(volume_24h_raw or 0)
                # Fall back to volume_24h (in base currency, need to multiply by price)
                elif "volume_24h" in product_data:
                    volume_24h_raw = product_data.get("volume_24h", 0)
                    price = product_data.get("price", 0)
                    logger.debug(f"Found volume_24h in root: {volume_24h_raw}, price: {price}")
                    # Convert base currency volume to USD
                    volume_24h = float(volume_24h_raw or 0) * float(price or 0)
                elif "product" in product_data and "volume_24h" in product_data["product"]:
                    volume_24h_raw = product_data["product"].get("volume_24h", 0)
                    price = product_data.get("price", 0)
                    logger.debug(f"Found volume_24h in product: {volume_24h_raw}, price: {price}")
                    # Convert base currency volume to USD
                    volume_24h = float(volume_24h_raw or 0) * float(price or 0)
                else:
                    # Log available fields for debugging
                    logger.warning(f"volume_24h not found for {product_id}. Available fields: {list(product_data.keys())}")
                    if "product" in product_data:
                        logger.warning(f"Product sub-fields: {list(product_data['product'].keys())}")

                # Extract price change percentage from REST API
                # Coinbase can return either a STRING "9.43%" or NUMBER 9.43
                price_pct_raw = product_data.get("price_percentage_change_24h", 0)

                # Parse the percentage (handle both string and numeric formats)
                try:
                    if isinstance(price_pct_raw, str):
                        # Strip % sign if present: "9.43%" -> 9.43
                        price_change_pct_24h = float(price_pct_raw.rstrip('%'))
                    else:
                        # Already a number
                        price_change_pct_24h = float(price_pct_raw)
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(f"Failed to parse price_percentage_change_24h: {price_pct_raw} - {e}")
                    price_change_pct_24h = 0.0

                # Calculate absolute price change from current price if we have the percentage
                current_price = float(product_data.get("price", 0))
                if price_change_pct_24h != 0 and current_price > 0:
                    price_change_24h = current_price * (price_change_pct_24h / 100.0)
                else:
                    price_change_24h = 0.0

                logger.debug(f"REST API price_change_pct_24h for {product_id}: {price_change_pct_24h}%")

                # Update existing ticker with 24h volume and price change, or create if doesn't exist
                if product_id in self._tickers:
                    ticker = self._tickers[product_id]
                    # Create updated ticker with new volume and price change
                    updated_ticker = Ticker(
                        product_id=ticker.product_id,
                        price=ticker.price,
                        volume_24h=volume_24h,
                        low_24h=ticker.low_24h,
                        high_24h=ticker.high_24h,
                        price_change_24h=price_change_24h,
                        price_change_pct_24h=price_change_pct_24h,
                        timestamp=ticker.timestamp,
                    )
                    self._tickers[product_id] = updated_ticker
                    self.bus.emit("ticker", updated_ticker)
                else:
                    # Ticker doesn't exist yet (REST API ran before WebSocket), create it
                    new_ticker = Ticker(
                        product_id=product_id,
                        price=current_price,
                        volume_24h=volume_24h,
                        low_24h=0.0,
                        high_24h=0.0,
                        price_change_24h=price_change_24h,
                        price_change_pct_24h=price_change_pct_24h,
                        timestamp=time.time(),
                    )
                    self._tickers[product_id] = new_ticker
                    self.bus.emit("ticker", new_ticker)

                if volume_24h > 0:
                    logger.info(f"Updated 24h volume for {product_id}: ${volume_24h:,.0f}")
                else:
                    logger.warning(f"24h volume for {product_id} is 0 or missing")
            except Exception as e:
                logger.warning(f"Failed to fetch stats for {product_id}: {e}", exc_info=True)

    async def _periodic_stats_update(self):
        """Periodically fetch 24h stats from REST API to update volume data."""
        # Wait for initial ticker data to be available
        await asyncio.sleep(5)

        # Initial update
        await self._update_24h_stats()

        # Continue periodic updates
        while self._running:
            try:
                # Wait 30 seconds between updates
                await asyncio.sleep(30)

                if not self._running:
                    break

                await self._update_24h_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic stats update: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    # ------------------------------------------------------------------
    # Candle Handlers
    # ------------------------------------------------------------------

    async def _on_10s_candle(self, product_id: str, candle: Candle):
        self._indicator_worker.submit(product_id, candle)
        await self._tf_aggs[product_id].add_candle(candle)
        self.bus.emit("candle", product_id, candle)

    async def _on_tf_candle(self, product_id: str, candle: Candle):
        self._indicator_worker.submit(product_id, candle)
        self.bus.emit("candle", product_id, candle)

    # ------------------------------------------------------------------
    # Indicator + Alerts
    # ------------------------------------------------------------------

    async def _on_indicator_snapshot(self, snapshot: IndicatorSnapshot):
        self.bus.emit("indicator", snapshot)

    async def _on_alert(self, alert: Alert):
        self.bus.emit("alert", alert)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on(self, event: str, cb: Callable):
        """Register event callback."""
        self.bus.on(event, cb)

    def get_recent_trades(self, product_id: str, limit: int = 15) -> list:
        """Get recent trades for a product."""
        history = self._trade_history.get(product_id, [])
        return list(history)[-limit:] if history else []

    def get_latest_ticker(self, product_id: str) -> Optional[Ticker]:
        """Get latest ticker data for a product."""
        return self._tickers.get(product_id)

    def get_indicator_snapshot(self, product_id: str, timeframe: Timeframe) -> Optional[IndicatorSnapshot]:
        """Get latest indicator snapshot for a product and timeframe."""
        return self._indicator_engine.get_snapshot(product_id, timeframe)
