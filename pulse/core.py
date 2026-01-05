"""
Pulse Core - Main orchestrator for the Pulse module.

Coordinates:
- CoinbaseConnector for data fetching
- CandleManager for timeframe storage
- Aggregators for 10s candles and order book
- IndicatorEngine for technical analysis
- Callback system for event-driven updates
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set

from connectors.coinbase import CoinbaseConnector, CoinbaseConfig, CoinbaseGranularity
from pulse.aggregators import CandleAggregator, OrderBookBucketer, TimeframeAggregator
from pulse.config import (
    Alert,
    BucketedOrderBook,
    Candle,
    IndicatorSnapshot,
    PULSE_EVENTS,
    PulseConfig,
    Ticker,
    Timeframe,
    Trade,
)
from pulse.indicators import IndicatorEngine

logger = logging.getLogger("Pulse.Core")


class CandleManager:
    """
    Manages candle storage with lazy loading per timeframe.

    Features:
    - Enable/disable timeframes at runtime
    - Fixed-size deque storage per timeframe
    - Thread-safe access
    """

    def __init__(self, config: PulseConfig):
        """
        Initialize candle manager.

        Args:
            config: Pulse configuration
        """
        self.config = config

        # Storage: product_id -> timeframe -> deque of candles
        self._candles: Dict[str, Dict[Timeframe, Deque[Candle]]] = {}

        # Lock for thread-safe access
        self._lock = asyncio.Lock()

    def _ensure_storage(self, product_id: str, timeframe: Timeframe):
        """Ensure storage exists for product/timeframe."""
        if product_id not in self._candles:
            self._candles[product_id] = {}

        if timeframe not in self._candles[product_id]:
            self._candles[product_id][timeframe] = deque(
                maxlen=self.config.candle_history_size
            )

    async def add_candle(self, product_id: str, candle: Candle):
        """
        Add a candle to storage.

        Args:
            product_id: Product ID
            candle: Candle to add
        """
        async with self._lock:
            self._ensure_storage(product_id, candle.timeframe)
            self._candles[product_id][candle.timeframe].append(candle)

    async def add_candles(self, product_id: str, candles: List[Candle]):
        """
        Add multiple candles to storage.

        Args:
            product_id: Product ID
            candles: List of candles to add
        """
        async with self._lock:
            for candle in candles:
                self._ensure_storage(product_id, candle.timeframe)
                self._candles[product_id][candle.timeframe].append(candle)

    def get_candles(
        self,
        product_id: str,
        timeframe: Timeframe,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Get candles for product/timeframe.

        Args:
            product_id: Product ID
            timeframe: Timeframe
            limit: Optional limit on number of candles

        Returns:
            List of candles (oldest first)
        """
        if product_id not in self._candles:
            return []

        if timeframe not in self._candles[product_id]:
            return []

        candles = list(self._candles[product_id][timeframe])

        if limit:
            return candles[-limit:]

        return candles

    def get_latest_candle(
        self,
        product_id: str,
        timeframe: Timeframe
    ) -> Optional[Candle]:
        """
        Get most recent candle.

        Args:
            product_id: Product ID
            timeframe: Timeframe

        Returns:
            Latest candle or None
        """
        candles = self.get_candles(product_id, timeframe, limit=1)
        return candles[-1] if candles else None

    def enable_timeframe(self, product_id: str, timeframe: Timeframe):
        """Enable a timeframe for tracking."""
        self._ensure_storage(product_id, timeframe)
        self.config.enabled_timeframes.add(timeframe)

    def disable_timeframe(self, product_id: str, timeframe: Timeframe):
        """Disable a timeframe."""
        self.config.enabled_timeframes.discard(timeframe)
        # Don't clear data - just stop updating

    def clear(self, product_id: Optional[str] = None):
        """
        Clear candle storage.

        Args:
            product_id: Specific product to clear, or None for all
        """
        if product_id:
            self._candles.pop(product_id, None)
        else:
            self._candles.clear()


class CallbackManager:
    """Manages event callbacks following FingerBlaster pattern."""

    def __init__(self):
        """Initialize callback manager."""
        self._callbacks: Dict[str, List[Callable]] = {
            event: [] for event in PULSE_EVENTS
        }

    def register(self, event: str, callback: Callable) -> bool:
        """
        Register a callback for an event.

        Args:
            event: Event name
            callback: Callback function

        Returns:
            True if registered, False if event unknown
        """
        if event not in self._callbacks:
            logger.warning(f"Unknown event: {event}")
            return False

        self._callbacks[event].append(callback)
        return True

    def unregister(self, event: str, callback: Callable) -> bool:
        """
        Unregister a callback.

        Args:
            event: Event name
            callback: Callback function

        Returns:
            True if unregistered, False otherwise
        """
        if event not in self._callbacks:
            return False

        try:
            self._callbacks[event].remove(callback)
            return True
        except ValueError:
            return False

    async def emit(self, event: str, *args, **kwargs):
        """
        Emit an event to all registered callbacks.

        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if event not in self._callbacks:
            return

        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")


class PulseCore:
    """
    Main Pulse orchestrator.

    Coordinates all components:
    - CoinbaseConnector for data
    - CandleManager for storage
    - Aggregators for processing
    - IndicatorEngine for analysis
    - CallbackManager for events
    """

    def __init__(self, config: Optional[PulseConfig] = None):
        """
        Initialize PulseCore.

        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or PulseConfig()

        # Core components
        self._connector: Optional[CoinbaseConnector] = None
        self._candle_manager = CandleManager(self.config)
        self._callback_manager = CallbackManager()

        # Per-product aggregators
        self._candle_aggregators: Dict[str, CandleAggregator] = {}
        self._order_book_bucketers: Dict[str, OrderBookBucketer] = {}
        self._timeframe_aggregators: Dict[str, TimeframeAggregator] = {}

        # Indicator engine
        self._indicator_engine = IndicatorEngine(
            config=self.config,
            on_indicator_update=self._on_indicator_update,
            on_alert=self._on_alert,
        )

        # State
        self._running = False
        self._priming_complete: Dict[str, bool] = {}
        self._trade_history: Dict[str, Deque[Trade]] = {}
        self._current_ticker: Dict[str, Ticker] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Start PulseCore - prime data and connect WebSocket."""
        if self._running:
            logger.warning("PulseCore already running")
            return

        self._running = True
        logger.info("Starting PulseCore...")

        # Create connector
        coinbase_config = CoinbaseConfig(
            ws_reconnect_delay=self.config.ws_reconnect_delay,
            ws_max_reconnect_attempts=self.config.ws_max_reconnect_attempts,
            rest_rate_limit_per_sec=self.config.rest_rate_limit_per_sec,
        )
        self._connector = CoinbaseConnector(config=coinbase_config)

        # Set up connector callbacks
        self._connector.on_trade = self._on_ws_trade
        self._connector.on_l2_update = self._on_ws_l2_update
        self._connector.on_ticker = self._on_ws_ticker
        self._connector.on_connection_status = self._on_connection_status

        # Initialize aggregators for each product
        for product_id in self.config.products:
            self._init_product(product_id)

        # Prime historical data
        await self._prime_all_products()

        # Start WebSocket
        await self._connector.start(
            product_ids=self.config.products,
            channels=["level2", "market_trades", "ticker"],
        )

        logger.info("PulseCore started")

    async def stop(self):
        """Stop PulseCore and cleanup."""
        self._running = False
        logger.info("Stopping PulseCore...")

        if self._connector:
            await self._connector.stop()
            self._connector = None

        logger.info("PulseCore stopped")

    def _init_product(self, product_id: str):
        """Initialize aggregators for a product."""
        # 10-second candle aggregator
        self._candle_aggregators[product_id] = CandleAggregator(
            product_id=product_id,
            on_candle=lambda c, pid=product_id: self._on_10s_candle(pid, c),
            wall_clock_aligned=self.config.ten_sec_wall_clock_aligned,
            use_prev_close_for_empty=self.config.ten_sec_empty_use_prev_close,
        )

        # Timeframe aggregator (rolls up 10s → 1m → 5m → 15m → 1h → 4h → 1d)
        self._timeframe_aggregators[product_id] = TimeframeAggregator(
            product_id=product_id,
            on_candle=lambda c, pid=product_id: self._on_aggregated_candle(pid, c),
        )

        # Order book bucketer
        self._order_book_bucketers[product_id] = OrderBookBucketer(
            bucket_size=self.config.bucket_size_usd,
            product_id=product_id,
            on_update=lambda b, pid=product_id: self._on_orderbook_update(pid, b),
        )

        # Trade history
        self._trade_history[product_id] = deque(
            maxlen=self.config.trade_history_size
        )

    # -------------------------------------------------------------------------
    # Priming
    # -------------------------------------------------------------------------

    async def _prime_all_products(self):
        """Prime historical data for all products."""
        for product_id in self.config.products:
            await self._prime_product(product_id)

    async def _prime_product(self, product_id: str):
        """
        Prime historical data for a single product.

        Fetches candles for all enabled timeframes.
        """
        logger.info(f"Priming data for {product_id}...")
        logger.info(f"Enabled timeframes: {[tf.value for tf in self.config.enabled_timeframes]}")

        enabled_tfs = self.config.get_enabled_timeframes_list()
        logger.info(f"Timeframes to prime: {[tf.value for tf in enabled_tfs]}")

        # Convert to Coinbase granularities (skip 10s - it's local only)
        granularities = []
        for tf in enabled_tfs:
            if tf == Timeframe.TEN_SEC:
                continue
            try:
                gran = CoinbaseGranularity[tf.coinbase_granularity]
                granularities.append((tf, gran))
            except KeyError:
                logger.warning(f"Unknown granularity for {tf}")

        if not granularities:
            logger.info(f"No timeframes to prime for {product_id}")
            self._priming_complete[product_id] = True
            await self._callback_manager.emit('priming_complete', product_id)
            return

        # Fetch candles for each timeframe (parallel or sequential based on config)
        total = len(granularities)

        if self.config.prime_parallel:
            # Parallel priming - much faster!
            async def prime_single_tf(idx: int, tf: Timeframe, gran: CoinbaseGranularity):
                try:
                    # Emit progress
                    progress = (idx + 1) / total
                    await self._callback_manager.emit(
                        'priming_progress', product_id, tf, progress
                    )

                    # Fetch candles
                    logger.info(f"Fetching {self.config.prime_bars_per_timeframe} candles for {product_id} {tf.value}...")
                    raw_candles = await self._connector.prime_timeframe(
                        product_id,
                        gran,
                        bars=self.config.prime_bars_per_timeframe,
                    )
                    logger.info(f"Received {len(raw_candles)} raw candles for {tf.value}")

                    # Convert to Candle objects and store
                    candles = [
                        Candle(
                            timestamp=c['start'],
                            open=c['open'],
                            high=c['high'],
                            low=c['low'],
                            close=c['close'],
                            volume=c['volume'],
                            timeframe=tf,
                        )
                        for c in raw_candles
                    ]

                    await self._candle_manager.add_candles(product_id, candles)
                    logger.info(f"Primed {len(candles)} {tf.value} candles for {product_id}")

                    # Update indicators with ALL candles to ensure full warmup
                    # Then emit the final snapshot so GUI shows data immediately
                    for candle in candles:
                        await self._indicator_engine.update(product_id, candle)

                    # The last candle update above will have triggered the indicator_update callback
                    # So the GUI will immediately show the latest indicator values
                    logger.info(f"Indicators primed for {tf.value}")

                except Exception as e:
                    logger.error(f"Error priming {tf.value} for {product_id}: {e}")

            # Execute all timeframe priming in parallel
            tasks = [
                prime_single_tf(idx, tf, gran)
                for idx, (tf, gran) in enumerate(granularities)
            ]
            await asyncio.gather(*tasks)

        else:
            # Sequential priming (slower but safer for rate limits)
            for idx, (tf, gran) in enumerate(granularities):
                try:
                    # Emit progress
                    progress = (idx + 1) / total
                    await self._callback_manager.emit(
                        'priming_progress', product_id, tf, progress
                    )

                    # Fetch candles
                    logger.info(f"Fetching {self.config.prime_bars_per_timeframe} candles for {product_id} {tf.value}...")
                    raw_candles = await self._connector.prime_timeframe(
                        product_id,
                        gran,
                        bars=self.config.prime_bars_per_timeframe,
                    )
                    logger.info(f"Received {len(raw_candles)} raw candles for {tf.value}")

                    # Convert to Candle objects and store
                    candles = [
                        Candle(
                            timestamp=c['start'],
                            open=c['open'],
                            high=c['high'],
                            low=c['low'],
                            close=c['close'],
                            volume=c['volume'],
                            timeframe=tf,
                        )
                        for c in raw_candles
                    ]

                    await self._candle_manager.add_candles(product_id, candles)
                    logger.info(f"Primed {len(candles)} {tf.value} candles for {product_id}")

                    # Update indicators with ALL candles to ensure full warmup
                    # Then emit the final snapshot so GUI shows data immediately
                    for candle in candles:
                        await self._indicator_engine.update(product_id, candle)

                    # The last candle update above will have triggered the indicator_update callback
                    # So the GUI will immediately show the latest indicator values
                    logger.info(f"Indicators primed for {tf.value}")

                except Exception as e:
                    logger.error(f"Error priming {tf.value} for {product_id}: {e}")

        self._priming_complete[product_id] = True
        await self._callback_manager.emit('priming_complete', product_id)
        logger.info(f"Priming complete for {product_id}")

    # -------------------------------------------------------------------------
    # WebSocket Handlers
    # -------------------------------------------------------------------------

    async def _on_ws_trade(self, raw_trade: Dict[str, Any]):
        """Handle trade from WebSocket."""
        try:
            # Parse timestamp: Coinbase returns ISO 8601 format
            time_str = raw_trade.get('time')
            if isinstance(time_str, str):
                # Convert ISO 8601 to UNIX timestamp
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
            else:
                # Fallback to current time if parsing fails
                timestamp = time.time()

            trade = Trade(
                trade_id=str(raw_trade.get('trade_id', '')),
                product_id=raw_trade.get('product_id', ''),
                price=float(raw_trade.get('price', 0)),
                size=float(raw_trade.get('size', 0)),
                side=raw_trade.get('side', 'UNKNOWN').upper(),
                timestamp=timestamp,
            )

            product_id = trade.product_id

            # Store in history
            if product_id in self._trade_history:
                self._trade_history[product_id].append(trade)

            # Aggregate into 10s candle
            if product_id in self._candle_aggregators:
                logger.debug(f"Adding trade to 10s aggregator: {product_id} @ {trade.price}")
                await self._candle_aggregators[product_id].add_trade(trade)

            # Emit trade event
            await self._callback_manager.emit('trade_update', trade)

        except Exception as e:
            logger.error(f"Error processing trade: {e}")

    async def _on_ws_l2_update(self, data: Dict[str, Any]):
        """Handle L2 order book update from WebSocket."""
        try:
            product_id = data.get('product_id', self.config.products[0])
            event_type = data.get('type', '')

            if product_id not in self._order_book_bucketers:
                return

            bucketer = self._order_book_bucketers[product_id]

            if event_type == 'snapshot':
                await bucketer.process_snapshot(data)
            else:
                await bucketer.process_update(data)

        except Exception as e:
            logger.error(f"Error processing L2 update: {e}")

    async def _on_ws_ticker(self, raw_ticker: Dict[str, Any]):
        """Handle ticker from WebSocket."""
        try:
            # Try multiple possible field names for volume
            volume_24h = (
                raw_ticker.get('volume_24_h') or 
                raw_ticker.get('volume_24h') or 
                raw_ticker.get('volume24h') or
                raw_ticker.get('volume') or
                0
            )
            
            # Log raw ticker data for debugging if volume is 0
            if float(volume_24h or 0) == 0:
                logger.debug(f"Ticker volume is 0. Available fields: {list(raw_ticker.keys())}")
            
            ticker = Ticker(
                product_id=raw_ticker.get('product_id', ''),
                price=float(raw_ticker.get('price', 0)),
                volume_24h=float(volume_24h or 0),
                low_24h=float(raw_ticker.get('low_24_h', 0) or raw_ticker.get('low24h', 0) or 0),
                high_24h=float(raw_ticker.get('high_24_h', 0) or raw_ticker.get('high24h', 0) or 0),
                price_change_24h=float(raw_ticker.get('price_percent_chg_24_h', 0) or raw_ticker.get('price_change_24h', 0) or 0),
                price_change_pct_24h=float(raw_ticker.get('price_percent_chg_24_h', 0) or raw_ticker.get('price_change_pct_24h', 0) or 0),
                timestamp=time.time(),
            )

            self._current_ticker[ticker.product_id] = ticker

            # Update live price indicators for all enabled timeframes
            # This makes the UI more responsive between candle closes
            for timeframe in self.config.enabled_timeframes:
                await self._indicator_engine.update_live_price(
                    ticker.product_id,
                    timeframe,
                    ticker.price
                )

            # Emit ticker event
            await self._callback_manager.emit('ticker_update', ticker)

        except Exception as e:
            logger.error(f"Error processing ticker: {e}")

    async def _on_connection_status(self, connected: bool, message: str):
        """Handle connection status change."""
        await self._callback_manager.emit('connection_status', connected, message)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------

    async def _on_10s_candle(self, product_id: str, candle: Candle):
        """Handle completed 10-second candle."""
        logger.info(f"10s candle completed: {product_id} @ {candle.close} (vol={candle.volume})")
        
        # Store candle
        await self._candle_manager.add_candle(product_id, candle)

        # Update indicators with this 10s candle (10s is not aggregated, so update directly)
        logger.info(f"Updating indicators for 10s candle...")
        snapshot = await self._indicator_engine.update(product_id, candle)
        logger.info(f"Indicator update completed for 10s: RSI={snapshot.rsi}, ADX={snapshot.adx}")

        # Pass to timeframe aggregator (will emit candles for all timeframes)
        await self._timeframe_aggregators[product_id].add_candle(candle)

        # Emit candle event
        await self._callback_manager.emit('candle_update', candle)

    async def _on_aggregated_candle(self, product_id: str, candle: Candle):
        """Handle completed candle from any timeframe (via TimeframeAggregator)."""
        # Store candle for all timeframes
        await self._candle_manager.add_candle(product_id, candle)

        # Update indicators with this candle (includes timeframe context)
        await self._indicator_engine.update(product_id, candle)

    async def _on_orderbook_update(self, product_id: str, book: BucketedOrderBook):
        """Handle order book update."""
        await self._callback_manager.emit('orderbook_update', book)

    async def _on_indicator_update(self, snapshot: IndicatorSnapshot):
        """Handle indicator update from engine (per-timeframe)."""
        logger.info(f"Indicator update received: {snapshot.product_id} {snapshot.timeframe.value} - emitting to callbacks")
        # Emit indicator update with product_id, timeframe, and snapshot
        await self._callback_manager.emit('indicator_update', snapshot.product_id, snapshot.timeframe, snapshot)

        # Also emit specific events
        if snapshot.vwap is not None:
            await self._callback_manager.emit(
                'vwap_update',
                snapshot.product_id,
                snapshot.timeframe,
                snapshot.vwap,
            )

        # Emit regime update
        await self._callback_manager.emit('regime_update', snapshot.product_id, snapshot.timeframe, snapshot)

    async def _on_alert(self, alert: Alert):
        """Handle alert from indicator engine."""
        await self._callback_manager.emit('alert', alert)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def register_callback(self, event: str, callback: Callable) -> bool:
        """
        Register a callback for an event.

        Args:
            event: Event name (see PULSE_EVENTS)
            callback: Callback function

        Returns:
            True if registered
        """
        return self._callback_manager.register(event, callback)

    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """
        Unregister a callback.

        Args:
            event: Event name
            callback: Callback function

        Returns:
            True if unregistered
        """
        return self._callback_manager.unregister(event, callback)

    def get_candles(
        self,
        product_id: str,
        timeframe: Timeframe,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Get candle history.

        Args:
            product_id: Product ID
            timeframe: Timeframe
            limit: Optional limit

        Returns:
            List of candles (oldest first)
        """
        return self._candle_manager.get_candles(product_id, timeframe, limit)

    def get_orderbook(self, product_id: str) -> Optional[BucketedOrderBook]:
        """
        Get current bucketed order book.

        Args:
            product_id: Product ID

        Returns:
            Current order book or None
        """
        bucketer = self._order_book_bucketers.get(product_id)
        return bucketer.get_current_book() if bucketer else None

    def get_recent_trades(self, product_id: str, limit: int = 100) -> List[Trade]:
        """
        Get recent trades.

        Args:
            product_id: Product ID
            limit: Number of trades to return

        Returns:
            List of recent trades (newest first)
        """
        history = self._trade_history.get(product_id, deque())
        trades = list(history)
        return trades[-limit:][::-1]

    def get_ticker(self, product_id: str) -> Optional[Ticker]:
        """
        Get current ticker.

        Args:
            product_id: Product ID

        Returns:
            Current ticker or None
        """
        return self._current_ticker.get(product_id)

    def get_indicators(self, product_id: str, timeframe: Optional[Timeframe] = None) -> Optional[IndicatorSnapshot]:
        """
        Get current indicator snapshot.

        Args:
            product_id: Product ID
            timeframe: Timeframe (if None, returns latest from any timeframe)

        Returns:
            Current indicator snapshot or None
        """
        return self._indicator_engine.get_snapshot(product_id, timeframe)

    def get_vwap(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current VWAP for product and timeframe."""
        return self._indicator_engine.get_vwap(product_id, timeframe)

    def enable_timeframe(self, product_id: str, timeframe: Timeframe):
        """Enable a timeframe for tracking."""
        self._candle_manager.enable_timeframe(product_id, timeframe)

    def disable_timeframe(self, product_id: str, timeframe: Timeframe):
        """Disable a timeframe."""
        self._candle_manager.disable_timeframe(product_id, timeframe)

    def is_priming_complete(self, product_id: str) -> bool:
        """Check if priming is complete for a product."""
        return self._priming_complete.get(product_id, False)

    @property
    def is_running(self) -> bool:
        """Check if PulseCore is running."""
        return self._running
