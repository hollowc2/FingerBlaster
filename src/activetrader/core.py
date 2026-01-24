import asyncio
import logging
import time
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd

from src.connectors.polymarket import PolymarketConnector
from src.activetrader.config import AppConfig
from src.activetrader.engine import (
    MarketDataManager, HistoryManager, WebSocketManager, 
    OrderExecutor, RTDSManager
)
from src.activetrader.analytics import AnalyticsEngine, AnalyticsSnapshot, TimerUrgency, EdgeDirection

logger = logging.getLogger("FingerBlaster")

# System Constants
POSITION_UPDATE_INTERVAL = 5.0  # Seconds between position API calls
STRIKE_RESOLVE_INTERVAL = 2.0   # Seconds between strike resolution attempts
PRICE_CACHE_TTL = 0.1           # Seconds to cache price calculations
HEALTH_CHECK_INTERVAL = 10.0    # Seconds between data health checks

CALLBACK_EVENTS: Tuple[str, ...] = (
    'market_update',
    'btc_price_update',
    'price_update',
    'account_stats_update',
    'countdown_update',
    'prior_outcomes_update',
    'resolution',
    'log',
    'chart_update',
    'analytics_update',
    'order_submitted',
    'order_filled',
    'order_failed',
    'flatten_started',
    'flatten_completed',
    'flatten_failed',
    'cancel_started',
    'cancel_completed',
    'cancel_failed',
    'size_changed',
)

class CallbackManager:
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {
            event: [] for event in CALLBACK_EVENTS
        }
        self._lock = threading.Lock()

    def register(self, event: str, callback: Callable) -> bool:
        if event not in self._callbacks:
            return False
        with self._lock:
            if callback not in self._callbacks[event]:
                self._callbacks[event].append(callback)
        return True

    def unregister(self, event: str, callback: Callable) -> bool:
        if event not in self._callbacks:
            return False
        with self._lock:
            if callback in self._callbacks[event]:
                self._callbacks[event].remove(callback)
        return True

    def clear(self, event: Optional[str] = None) -> None:
        with self._lock:
            if event and event in self._callbacks:
                self._callbacks[event].clear()
            elif not event:
                for callbacks in self._callbacks.values():
                    callbacks.clear()

    def get_callbacks(self, event: str) -> List[Callable]:
        """Get list of callbacks for an event (returns copy for thread safety)."""
        with self._lock:
            return list(self._callbacks.get(event, []))

    def emit(self, event: str, *args, **kwargs) -> None:
        """Fire-and-forget callback emission (never blocks)."""
        if event not in self._callbacks:
            return
        with self._lock:
            callbacks = list(self._callbacks[event])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")

class FingerBlasterCore:

    def __init__(self, connector: Optional[PolymarketConnector] = None):
        self.config = AppConfig()
        self.connector = connector or PolymarketConnector()
        
        # Initialize engine managers
        self.market_manager = MarketDataManager(self.config)
        self.history_manager = HistoryManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.connector)
        
        # WebSocket listens and updates price via callback
        self.ws_manager = WebSocketManager(
            self.config,
            self.market_manager,
            self._on_ws_message
        )
        
        self.rtds_manager = RTDSManager(self.config, self._on_rtds_btc_price)
        self.analytics_engine = AnalyticsEngine()
        self.callback_manager = CallbackManager()
        
        # State
        self.resolution_shown = False
        self.last_resolution: Optional[str] = None
        self.last_chart_update: float = 0.0
        self.selected_size: float = 1.0
        self._yes_position: float = 0.0
        self._no_position: float = 0.0
        self._last_position_update: float = 0.0
        self._last_strike_resolve_attempt: float = 0.0

        # Caching & Health
        self._cached_prices: Optional[Tuple[float, float, float, float]] = None
        self._cache_timestamp: float = 0.0
        self._last_health_check: float = 0.0
        self._stale_data_warning_shown: bool = False
        self._position_lock = asyncio.Lock()
        self._market_update_lock = asyncio.Lock()  # Prevent concurrent market updates
        self._market_update_in_progress: bool = False  # Track update state

        # Market transition deduplication - prevents flood of WebSocket subscriptions
        self._switching_to_market_id: Optional[str] = None  # Market ID we're currently switching to
        self._market_switch_lock = asyncio.Lock()  # Serialize market transitions

    async def _on_ws_message(self, item: Dict[str, Any]) -> None:
        await self._recalc_price()

    async def _recalc_price(self) -> None:
        now = time.time()

        # Check for stale data
        if now - self._last_health_check >= HEALTH_CHECK_INTERVAL:
            await self._check_data_health()
            self._last_health_check = now

        # Calculate new prices from market manager books
        prices = await self.market_manager.calculate_mid_price()
        self._cached_prices = prices
        self._cache_timestamp = now

        # Notify UI
        self.callback_manager.emit('price_update', *prices)

    async def _check_data_health(self) -> None:
        is_stale = await self.market_manager.is_data_stale()
        if is_stale and not self._stale_data_warning_shown:
            self.log_msg("⚠️ WARNING: Price data stale. WebSocket may be down.")
            self._stale_data_warning_shown = True
            await self._attempt_auto_reconnect()
        elif not is_stale and self._stale_data_warning_shown:
            self.log_msg("✓ Price data is fresh again")
            self._stale_data_warning_shown = False

    def register_callback(self, event: str, callback: Callable) -> bool:
        return self.callback_manager.register(event, callback)

    def log_msg(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.callback_manager.emit('log', f"[{timestamp}] {message}")

    async def start_rtds(self) -> None:
        await self.rtds_manager.start()
        await self.ws_manager.start()

    async def _on_rtds_btc_price(self, btc_price: float) -> None:
        self.callback_manager.emit('btc_price_update', btc_price)

    def size_up(self) -> None:
        self.selected_size += 1.0
        self.callback_manager.emit('size_changed', self.selected_size)

    def size_down(self) -> None:
        self.selected_size = max(1.0, self.selected_size - 1.0)
        self.callback_manager.emit('size_changed', self.selected_size)

    async def place_order(self, side: str) -> None:
        self.log_msg(f"Action: PLACE ORDER {side} (${self.selected_size})")
        
        # Initial submission event
        # Use 0.0 as a placeholder for price as it's a market order
        self.callback_manager.emit('order_submitted', side, self.selected_size, 0.0)
        
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map:
                error = "Token map not ready"
                self.log_msg(f"Order error: {error}")
                self.callback_manager.emit('order_failed', side, self.selected_size, error)
                return

            # Execute order
            resp = await self.order_executor.execute_order(
                side=side,
                size=self.selected_size,
                token_map=token_map,
                price=None # Market order
            )
            
            if resp and isinstance(resp, dict) and resp.get('orderID'):
                # Success
                order_id = resp.get('orderID')
                # We don't have the exact filled price immediately from market order response
                # until it fills. For now, emit filled with 0.0 or best available price.
                prices = await self.market_manager.calculate_mid_price()
                # If side is Up, use yes_price. If Down, use no_price.
                fill_price = prices[0] if side == 'Up' else prices[1]
                
                self.callback_manager.emit('order_filled', side, self.selected_size, fill_price, order_id)
                self.log_msg(f"✓ {side} order filled: {order_id}")
                
                # Force position update soon
                self._last_position_update = 0
            else:
                error = "Order rejected by exchange"
                self.log_msg(f"Order error: {error}")
                self.callback_manager.emit('order_failed', side, self.selected_size, error)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error placing order: {e}", exc_info=True)
            self.callback_manager.emit('order_failed', side, self.selected_size, error_msg)

    async def flatten_all(self) -> None:
        self.log_msg("Action: FLATTEN ALL")
        self.callback_manager.emit('flatten_started')
        await asyncio.sleep(0)  # Yield control to let UI update
        
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map:
                error_msg = "Error: Token map not ready."
                self.log_msg(error_msg)
                self.callback_manager.emit('flatten_failed', error_msg)
                return
            
            results = await self.order_executor.flatten_positions(token_map)
            orders_processed = len(results) if results else 0
            
            if results:
                self.log_msg(f"Flatten completed. {orders_processed} orders processed.")
            else:
                self.log_msg("Flatten completed. No positions to flatten.")
            
            self.callback_manager.emit('flatten_completed', orders_processed)
        except Exception as e:
            error_msg = f"Flatten error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Flatten error: {e}", exc_info=True)
            self.callback_manager.emit('flatten_failed', error_msg)

    async def cancel_all_orders(self) -> bool:
        self.log_msg("Action: CANCEL ALL ORDERS")
        self.callback_manager.emit('cancel_started')
        await asyncio.sleep(0)  # Yield control to let UI update
        
        try:
            result = await self.order_executor.cancel_all_orders()
            if result:
                self.log_msg("All orders cancelled successfully.")
                self.callback_manager.emit('cancel_completed')
            else:
                self.log_msg("No orders to cancel or cancellation failed.")
                self.callback_manager.emit('cancel_completed')
            return result
        except Exception as e:
            error_msg = f"Cancel all error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Cancel all error: {e}", exc_info=True)
            self.callback_manager.emit('cancel_failed', error_msg)
            return False

    async def _attempt_auto_reconnect(self) -> None:
        """Attempt to reconnect WebSocket if stale."""
        try:
            # Check if WebSocket is actually connected by checking if _ws exists
            if self.ws_manager._ws is None:
                await self.ws_manager.start()
        except Exception as e:
            logger.debug(f"Auto-reconnect attempt failed: {e}")

    async def update_countdown(self) -> None:
        """Update market countdown and emit event."""
        market = await self.market_manager.get_market()
        if not market:
            return

        try:
            end_str = market.get('end_date')
            if not end_str:
                return

            dt_end = pd.Timestamp(end_str)
            if dt_end.tz is None:
                dt_end = dt_end.tz_localize('UTC')

            now = pd.Timestamp.now(tz='UTC')
            time_remaining = max(0, int((dt_end - now).total_seconds()))
            
            # Format time string
            minutes = time_remaining // 60
            seconds = time_remaining % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            # Use analytics engine for urgency logic
            urgency = self.analytics_engine.get_timer_urgency(time_remaining)
            
            self.callback_manager.emit('countdown_update', time_str, urgency, time_remaining)
            
            # Check for resolution
            if time_remaining <= 0 and not self.resolution_shown:
                await self._handle_market_resolution(market)
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}")

    async def update_analytics(self) -> None:
        """Gather all data and generate analytics snapshot."""
        market = await self.market_manager.get_market()
        if not market:
            return

        try:
            btc_price = self.rtds_manager.get_current_price()
            if not btc_price or btc_price <= 0:
                # Fallback to history
                prices = await self.history_manager.get_btc_history()
                btc_price = prices[-1] if prices else 0.0
            
            if btc_price <= 0:
                return

            # Get strike price
            strike_str = str(market.get('price_to_beat', '')).replace('$', '').replace(',', '').strip()
            strike_price = self._parse_price_to_beat(strike_str)
            
            # Get other data
            dt_end = pd.Timestamp(market.get('end_date'))
            if dt_end.tz is None: dt_end = dt_end.tz_localize('UTC')
            time_remaining = max(0, int((dt_end - pd.Timestamp.now(tz='UTC')).total_seconds()))
            
            prices = await self.market_manager.calculate_mid_price()
            yes_price, no_price, _, _ = prices
            order_book = await self.market_manager.get_raw_order_book()
            
            # Get positions (cached to avoid excessive API calls)
            now_ts = time.time()
            if now_ts - self._last_position_update >= POSITION_UPDATE_INTERVAL:
                # Trigger background update of positions
                asyncio.create_task(self._update_positions())
                self._last_position_update = now_ts

            # Thread-safe read of positions
            async with self._position_lock:
                yes_position = self._yes_position
                no_position = self._no_position
            
            # Generate snapshot
            snapshot = await self.analytics_engine.generate_snapshot(
                btc_price=btc_price,
                price_to_beat=strike_price,
                time_remaining_seconds=time_remaining,
                yes_market_price=yes_price,
                no_market_price=no_price,
                order_book=order_book,
                yes_position=yes_position,
                no_position=no_position,
                avg_entry_yes=None, # PositionTracker removed as per user's "carry over" request
                avg_entry_no=None,
                prior_outcomes=[], # Simplified for now
                order_size_usd=self.selected_size
            )
            
            self.callback_manager.emit('analytics_update', snapshot)
        except Exception as e:
            logger.error(f"Error updating analytics: {e}", exc_info=True)

    def _parse_price_to_beat(self, strike_str: str) -> float:
        if not strike_str:
            return 0.0
        try:
            clean = strike_str.replace('$', '').replace(',', '').strip()
            if not clean or clean in ('N/A', 'Pending', 'Loading', '--', 'None', ''):
                return 0.0
            return float(clean)
        except (ValueError, AttributeError):
            return 0.0

    async def update_market_status(self) -> None:
        """Poll for active market and handle discovery."""
        # Prevent concurrent market updates (race condition during market transitions)
        # Pre-check to avoid queueing on the lock
        if self._market_update_in_progress:
            logger.debug("Market update already in progress, skipping")
            return

        async with self._market_update_lock:
            # Double-check inside lock (in case flag changed while waiting)
            if self._market_update_in_progress:
                logger.debug("Market update already in progress (double-check), skipping")
                return

            self._market_update_in_progress = True
            try:
                # Fetch market data INSIDE the lock to avoid stale data
                market = await self.connector.get_active_market()
                if market:
                    # Check if it's a new market
                    current = await self.market_manager.get_market()
                    if not current or current.get('market_id') != market.get('market_id'):
                        # Prevent switching back to an expired market
                        # Only switch if the new market ends AFTER the current market
                        should_switch = False

                        if not current:
                            # No current market - always switch
                            should_switch = True
                        else:
                            # Compare end times - only switch to markets that end later
                            try:
                                current_end = pd.Timestamp(current.get('end_date'))
                                if current_end.tz is None:
                                    current_end = current_end.tz_localize('UTC')

                                new_end = pd.Timestamp(market.get('end_date'))
                                if new_end.tz is None:
                                    new_end = new_end.tz_localize('UTC')

                                # Only switch if new market ends after current market
                                # This prevents flip-flopping back to expired markets
                                if new_end > current_end:
                                    should_switch = True
                                    logger.info(
                                        f"Switching to newer market: {market.get('market_id')} "
                                        f"(ends {new_end} vs current {current_end})"
                                    )
                                else:
                                    logger.debug(
                                        f"Ignoring older/same market: {market.get('market_id')} "
                                        f"(ends {new_end} vs current {current_end})"
                                    )
                            except Exception as e:
                                logger.warning(f"Error comparing market times: {e}")
                                # If we can't compare times, don't switch
                                should_switch = False

                        if should_switch:
                            await self._on_new_market_found(market)

                # Always try to resolve strike if pending/dynamic
                await self._try_resolve_pending_strike()
            except Exception as e:
                logger.error(f"Error updating market status: {e}")
            finally:
                self._market_update_in_progress = False

    async def _on_new_market_found(self, market: Dict[str, Any]) -> None:
        """Handle transition to new market with deduplication.

        Uses a lock and tracking flag to prevent multiple concurrent transitions
        to the same market, which would cause a flood of WebSocket subscriptions.
        """
        market_id = market.get('market_id')

        # Fast path: check if we're already switching to this market (no lock needed)
        if self._switching_to_market_id == market_id:
            logger.debug(f"Already switching to market {market_id}, skipping duplicate")
            return

        # Serialize market transitions to prevent race conditions
        async with self._market_switch_lock:
            # Double-check inside lock (another task may have completed the switch)
            if self._switching_to_market_id == market_id:
                logger.debug(f"Already switching to market {market_id} (inside lock), skipping")
                return

            # Check if we've already switched to this market
            current = await self.market_manager.get_market()
            if current and current.get('market_id') == market_id:
                logger.debug(f"Already on market {market_id}, skipping switch")
                return

            # Mark that we're switching to this market
            self._switching_to_market_id = market_id
            logger.info(f"Switching to market {market_id}")

            try:
                self.log_msg(f"New Market Found: {market.get('title', 'Unknown')}")
                self.resolution_shown = False

                # Set market in manager
                await self.market_manager.set_market(market)

                # Subscribe WebSocket
                await self.ws_manager.subscribe_to_market(market)

                # Reset positions on new market
                self._yes_position = 0.0
                self._no_position = 0.0
                self._last_position_update = 0.0  # Force update for new market

                # Emit update
                ends = self._format_ends(market.get('end_date', ''))
                market_name = market.get('title', 'Market')
                self.callback_manager.emit('market_update', market.get('price_to_beat', 'N/A'), ends, market_name)

                # Immediately try to resolve strike if dynamic
                if market.get('price_to_beat') in ('Dynamic', 'Pending', 'N/A', '', None):
                    asyncio.create_task(self._try_resolve_pending_strike())

            finally:
                # Clear the switching flag after transition completes (success or failure)
                # This allows retrying if something went wrong
                self._switching_to_market_id = None

    def _format_ends(self, end_str: str) -> str:
        try:
            dt = pd.Timestamp(end_str)
            if dt.tz is None: dt = dt.tz_localize('UTC')
            # Convert to ET
            dt_et = dt.tz_convert('America/New_York')
            return dt_et.strftime('%B %d, %I:%M%p ET')
        except: return str(end_str)

    def _format_starts(self, start_str: str) -> str:
        """Format start time to human-readable ET time."""
        try:
            dt = pd.Timestamp(start_str)
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            dt_et = dt.tz_convert('America/New_York')
            return dt_et.strftime('%B %d, %I:%M%p ET')
        except:
            return str(start_str)

    async def _handle_market_resolution(self, market: Dict[str, Any]) -> None:
        """Handle market expiry, emit event, and migrate to next market."""
        self.resolution_shown = True
        self.log_msg("⚠️ MARKET EXPIRED - Searching for next market...")
        self.callback_manager.emit('resolution', "EXPIRED")

        # Schedule market migration after overlay
        asyncio.create_task(self._migrate_to_next_market(market))

    async def _migrate_to_next_market(self, expired_market: Dict[str, Any]) -> None:
        """Automatically migrate to next market after current expires.

        Waits for overlay, then searches with retries and exponential backoff.
        Skips migration if polling has already found a newer market.
        """
        expired_market_id = expired_market.get('market_id')

        # Wait for resolution overlay to finish
        await asyncio.sleep(self.config.resolution_overlay_duration)

        # Check if we've already migrated to a new market (e.g., via polling)
        current = await self.market_manager.get_market()
        if current and current.get('market_id') != expired_market_id:
            logger.info(f"Migration skipped: already on market {current.get('market_id')}")
            return

        max_attempts = 5
        retry_delay = 2.0

        for attempt in range(1, max_attempts + 1):
            # Re-check before each attempt in case polling found the market
            current = await self.market_manager.get_market()
            if current and current.get('market_id') != expired_market_id:
                logger.info(f"Migration cancelled: polling already found market {current.get('market_id')}")
                return

            try:
                self.log_msg(f"Migration attempt {attempt}/{max_attempts}...")

                # Get next market using connector
                next_market = await self.connector.get_next_market(
                    current_end_date_str=expired_market.get('end_date'),
                    series_id="10192"  # BTC Up or Down 15m series
                )

                if next_market:
                    # Found next market - transition
                    await self._on_new_market_found(next_market)
                    self.log_msg(f"✓ Migrated to: {next_market.get('title', 'Unknown')}")
                    return
                else:
                    self.log_msg(f"No next market found (attempt {attempt}/{max_attempts})")

                    if attempt < max_attempts:
                        wait_time = retry_delay * (2 ** (attempt - 1))
                        self.log_msg(f"Retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Migration error (attempt {attempt}): {e}", exc_info=True)
                self.log_msg(f"Migration error: {str(e)}")

                if attempt < max_attempts:
                    wait_time = retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(wait_time)

        # All attempts failed - normal polling will continue
        self.log_msg("⚠️ Could not find next market. Continuing to poll...")

    async def _try_resolve_pending_strike(self) -> None:
        """
        Resolve dynamic strike prices using a 6-tier fallback chain.

        Priority chain (Chainlink sources → CEX fallbacks):
        1. RTDS Chainlink history (local cache, fastest)
        2. Chainlink on-chain oracle (Polygon) - GUARANTEED, matches Polymarket resolution
        3. Chainlink API (experimental)
        4. Coinbase 15m candle open price (matches market timeframe)
        5. Binance 1m candle open price (last resort)
        6. Current RTDS/Binance spot price (emergency fallback)
        """
        market = await self.market_manager.get_market()
        if not market:
            return

        strike = market.get('price_to_beat')
        if strike not in ("Dynamic", "Pending", "N/A", "None", "", None):
            return

        # Don't throttle if market has started - retry aggressively
        start_dt = await self.market_manager.get_market_start_time()
        if not start_dt:
            logger.debug("No market start time available for strike resolution")
            return

        now_utc = pd.Timestamp.now(tz='UTC')
        market_started = start_dt <= now_utc

        # Throttle only if market hasn't started yet
        if not market_started:
            now = time.time()
            if now - self._last_strike_resolve_attempt < STRIKE_RESOLVE_INTERVAL:
                return
            self._last_strike_resolve_attempt = now

        try:
            logger.info(f"Attempting to resolve dynamic strike for {start_dt} (market_started={market_started})...")

            # Check RTDS connection status before attempting resolution
            rtds_status = self.rtds_manager.get_connection_status()
            logger.info(
                f"  RTDS status: connected={rtds_status['connected']}, "
                f"history_entries={rtds_status['history_entries']}, "
                f"chainlink_price={rtds_status['current_chainlink']}"
            )

            strike_val = None
            source = "Unknown"

            # METHOD 1: RTDS Chainlink history (local cache, fastest)
            logger.info("  Attempting METHOD 1: RTDS/Chainlink cache...")
            if strike_val is None:
                strike_val = self.rtds_manager.get_chainlink_price_at(start_dt)
                if strike_val:
                    source = "RTDS/Chainlink Cache"
                    logger.info(f"  ✓ METHOD 1 SUCCESS: Strike resolved from RTDS/Chainlink cache: ${strike_val:,.2f}")
                else:
                    logger.warning(f"  ✗ METHOD 1 FAILED: No price in RTDS cache for {start_dt}")

            # METHOD 2: Chainlink on-chain oracle (GUARANTEED - same source Polymarket uses)
            if strike_val is None:
                logger.info("  Attempting METHOD 2: Chainlink on-chain oracle...")
                strike_val = await self.connector.get_chainlink_onchain_price_at(start_dt)
                if strike_val:
                    source = "Chainlink On-Chain Oracle"
                    logger.info(f"  ✓ METHOD 2 SUCCESS: Strike resolved from Chainlink on-chain: ${strike_val:,.2f}")
                else:
                    logger.warning("  ✗ METHOD 2 FAILED: Chainlink on-chain oracle unavailable")

            # METHOD 3: Chainlink API (experimental, but worth trying)
            if strike_val is None:
                logger.info("  Attempting METHOD 3: Chainlink API...")
                strike_val = await self.connector.get_chainlink_price_at(start_dt)
                if strike_val:
                    source = "Chainlink API"
                    logger.info(f"  ✓ METHOD 3 SUCCESS: Strike resolved from Chainlink API: ${strike_val:,.2f}")
                else:
                    logger.warning("  ✗ METHOD 3 FAILED: Chainlink API unavailable")

            # METHOD 4: Coinbase 15m candle (best CEX match for 15m markets)
            if strike_val is None:
                logger.info("  Attempting METHOD 4: Coinbase 15m candle...")
                strike_val = await self.connector.get_coinbase_15m_open_price_at(start_dt)
                if strike_val:
                    source = "Coinbase 15m"
                    logger.info(f"  ✓ METHOD 4 SUCCESS: Strike resolved from Coinbase 15m: ${strike_val:,.2f}")
                else:
                    logger.warning("  ✗ METHOD 4 FAILED: Coinbase 15m candle unavailable")

            # METHOD 5: Binance 1m candle (reliable CEX fallback)
            if strike_val is None:
                logger.info("  Attempting METHOD 5: Binance 1m candle...")
                btc_price_str = await self.connector.get_btc_price_at(start_dt)
                if btc_price_str and btc_price_str != "N/A":
                    try:
                        strike_val = float(btc_price_str)
                        source = "Binance 1m"
                        logger.info(f"  ✓ METHOD 5 SUCCESS: Strike resolved from Binance 1m: ${strike_val:,.2f}")
                    except (ValueError, TypeError):
                        logger.warning("  ✗ METHOD 5 FAILED: Binance returned invalid price")
                else:
                    logger.warning("  ✗ METHOD 5 FAILED: Binance 1m candle unavailable")

            # METHOD 6: Emergency fallback - current price (only if market has started)
            if strike_val is None and market_started:
                logger.info("  Attempting METHOD 6: Current price (emergency fallback)...")
                current_btc = self.rtds_manager.get_current_price()
                if current_btc and current_btc > 0:
                    strike_val = current_btc
                    source = "Current Price (Emergency)"
                    logger.warning(f"  ⚠️ METHOD 6 USED: Using current BTC price as emergency fallback: ${strike_val:,.2f}")
                else:
                    logger.warning("  ✗ METHOD 6 FAILED: Current price unavailable")

            # ALWAYS have a price when market has started
            if strike_val:
                logger.info(f"✅ STRIKE RESOLVED: ${strike_val:,.2f} from {source}")
                # Update market in manager
                market['price_to_beat'] = f"{strike_val:,.2f}"
                await self.market_manager.set_market(market)
                # Emit update
                ends = self._format_ends(market.get('end_date', ''))
                market_name = market.get('title', 'Market')
                self.callback_manager.emit('market_update', market['price_to_beat'], ends, market_name)
            else:
                # Market hasn't started yet - keep as Pending (will retry)
                if not market_started:
                    logger.info(f"Market hasn't started yet ({start_dt}). Strike will be available at start time.")
                    market['price_to_beat'] = "Pending"
                    await self.market_manager.set_market(market)
                    ends = self._format_ends(market.get('end_date', ''))
                    market_name = market.get('title', 'Market')
                    self.callback_manager.emit('market_update', market['price_to_beat'], ends, market_name)
                else:
                    # This should NEVER happen with 6-tier fallback chain
                    logger.error(
                        "❌ CRITICAL: ALL 6 PRICE SOURCES FAILED after market start! "
                        f"Market started at {start_dt}, current time {now_utc}. "
                        "Check RTDS connection, Chainlink oracle, and CEX APIs. "
                        f"RTDS status: {rtds_status}"
                    )
                    self.log_msg(f"⚠️ Strike price resolution failed - all 6 sources unavailable")

        except Exception as e:
            logger.error(f"Error resolving strike: {e}", exc_info=True)

    async def _update_positions(self) -> None:
        """Update cached positions from connector (thread-safe)."""
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map: return

            # Fetch positions without lock (I/O operation)
            positions = {}
            up_id = token_map.get('Up')
            if up_id:
                positions['yes'] = await self.connector.get_token_balance(up_id)

            down_id = token_map.get('Down')
            if down_id:
                positions['no'] = await self.connector.get_token_balance(down_id)

            # Atomic update with lock (fast critical section)
            async with self._position_lock:
                if 'yes' in positions:
                    self._yes_position = positions['yes']
                if 'no' in positions:
                    self._no_position = positions['no']
        except Exception as e:
            logger.debug(f"Error updating positions in background: {e}")

    async def close_position(self, side: str) -> bool:
        """Close position for a specific side."""
        self.log_msg(f"Action: CLOSE {side.upper()} POSITION")
        try:
            token_map = await self.market_manager.get_token_map()
            token_id = token_map.get(side)
            if not token_id:
                return False
            
            balance = await self.connector.get_token_balance(token_id)
            if balance > 0.1:
                await self.connector.create_market_order(token_id, balance, 'SELL')
                return True
            return False
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        try:
            if self.ws_manager:
                await self.ws_manager.stop()
            if self.rtds_manager:
                await self.rtds_manager.stop()
            if self.connector:
                await self.connector.close()
            self.callback_manager.clear()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")