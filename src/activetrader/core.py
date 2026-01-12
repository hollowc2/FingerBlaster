"""Refactored core business logic controller with improved architecture."""

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

# Event types for callback registration
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
    """Manages event callbacks with proper cleanup."""
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {
            event: [] for event in CALLBACK_EVENTS
        }
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
    
    def register(self, event: str, callback: Callable) -> bool:
        if event not in self._callbacks:
            return False
        with self._lock:
            if callback not in self._callbacks[event]:
                self._callbacks[event].append(callback)
        return True

    def unregister(self, event: str, callback: Callable) -> bool:
        if event not in self._callbacks: return False
        with self._lock:
            try:
                if callback in self._callbacks[event]:
                    self._callbacks[event].remove(callback)
                return True
            except: return False

    def clear(self, event: Optional[str] = None) -> None:
        with self._lock:
            if event:
                if event in self._callbacks: self._callbacks[event].clear()
            else:
                for callbacks in self._callbacks.values(): callbacks.clear()

    def get_callbacks(self, event: str) -> List[Callable]:
        with self._lock: return list(self._callbacks.get(event, []))

    async def emit(self, event: str, *args, **kwargs) -> None:
        if event not in self._callbacks: return
        async with self._async_lock:
            callbacks = list(self._callbacks[event])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    loop = asyncio.get_running_loop()
                    loop.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")

class FingerBlasterCore:
    """Shared business logic controller with improved architecture."""
    
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
        self._cex_btc_price: Optional[float] = None
        self._cex_btc_timestamp: float = 0.0
        self._yes_position: float = 0.0
        self._no_position: float = 0.0
        self._last_position_update: float = 0.0
        self._position_update_interval: float = 5.0  # Only update positions every 5 seconds
        self._last_strike_resolve_attempt: float = 0.0
        self._strike_resolve_interval: float = 2.0  # Faster resolution (2s instead of 10s)
        
        # State & Caching
        self._cached_prices: Optional[Tuple[float, float, float, float]] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 0.1
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 10.0
        self._stale_data_warning_shown: bool = False

    async def _on_ws_message(self, item: Dict[str, Any]) -> None:
        """Called by WebSocketManager when new book data arrives."""
        await self._recalc_price()

    async def _recalc_price(self) -> None:
        """Recalculate mid price and update UI."""
        now = time.time()

        # Check for stale data
        if now - self._last_health_check >= self._health_check_interval:
            await self._check_data_health()
            self._last_health_check = now

        # Calculate new prices from market manager books
        prices = await self.market_manager.calculate_mid_price()
        self._cached_prices = prices
        self._cache_timestamp = now

        # Notify UI
        self._emit('price_update', *prices)

    async def _check_data_health(self) -> None:
        """Monitors WebSocket health and reconnection."""
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

    def _emit(self, event: str, *args, **kwargs) -> None:
        callbacks = self.callback_manager.get_callbacks(event)
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {event}: {e}")

    def log_msg(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._emit('log', f"[{timestamp}] {message}")

    async def start_rtds(self) -> None:
        await self.rtds_manager.start()
        await self.ws_manager.start()

    async def _on_rtds_btc_price(self, btc_price: float) -> None:
        self._emit('btc_price_update', btc_price)

    def size_up(self) -> None:
        """Increase the selected order size."""
        self.selected_size += 1.0
        self._emit('size_changed', self.selected_size)

    def size_down(self) -> None:
        """Decrease the selected order size, minimum $1."""
        self.selected_size = max(1.0, self.selected_size - 1.0)
        self._emit('size_changed', self.selected_size)

    async def place_order(self, side: str) -> None:
        """Place a market order for the specified side."""
        self.log_msg(f"Action: PLACE ORDER {side} (${self.selected_size})")
        
        # Initial submission event
        # Use 0.0 as a placeholder for price as it's a market order
        self._emit('order_submitted', side, self.selected_size, 0.0)
        
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map:
                error = "Token map not ready"
                self.log_msg(f"Order error: {error}")
                self._emit('order_failed', side, self.selected_size, error)
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
                
                self._emit('order_filled', side, self.selected_size, fill_price, order_id)
                self.log_msg(f"✓ {side} order filled: {order_id}")
                
                # Force position update soon
                self._last_position_update = 0
            else:
                error = "Order rejected by exchange"
                self.log_msg(f"Order error: {error}")
                self._emit('order_failed', side, self.selected_size, error)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error placing order: {e}", exc_info=True)
            self._emit('order_failed', side, self.selected_size, error_msg)

    async def flatten(self) -> None:
        """Alias for flatten_all to match UI calls."""
        await self.flatten_all()

    async def flatten_all(self) -> None:
        """Flatten all positions by selling all tokens at market price."""
        self.log_msg("Action: FLATTEN ALL")
        self._emit('flatten_started')
        await asyncio.sleep(0)  # Yield control to let UI update
        
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map:
                error_msg = "Error: Token map not ready."
                self.log_msg(error_msg)
                self._emit('flatten_failed', error_msg)
                return
            
            results = await self.order_executor.flatten_positions(token_map)
            orders_processed = len(results) if results else 0
            
            if results:
                self.log_msg(f"Flatten completed. {orders_processed} orders processed.")
            else:
                self.log_msg("Flatten completed. No positions to flatten.")
            
            self._emit('flatten_completed', orders_processed)
        except Exception as e:
            error_msg = f"Flatten error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Flatten error: {e}", exc_info=True)
            self._emit('flatten_failed', error_msg)

    async def cancel_all(self) -> bool:
        """Alias for cancel_all_orders to match UI calls."""
        return await self.cancel_all_orders()

    async def cancel_all_orders(self) -> bool:
        """Cancel all pending orders."""
        self.log_msg("Action: CANCEL ALL ORDERS")
        self._emit('cancel_started')
        await asyncio.sleep(0)  # Yield control to let UI update
        
        try:
            result = await self.order_executor.cancel_all_orders()
            if result:
                self.log_msg("All orders cancelled successfully.")
                self._emit('cancel_completed')
            else:
                self.log_msg("No orders to cancel or cancellation failed.")
                self._emit('cancel_completed')
            return result
        except Exception as e:
            error_msg = f"Cancel all error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Cancel all error: {e}", exc_info=True)
            self._emit('cancel_failed', error_msg)
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
            
            self._emit('countdown_update', time_str, urgency, time_remaining)
            
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
            if now_ts - self._last_position_update >= self._position_update_interval:
                # Trigger background update of positions
                asyncio.create_task(self._update_positions())
                self._last_position_update = now_ts

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
            
            self._emit('analytics_update', snapshot)
        except Exception as e:
            logger.error(f"Error updating analytics: {e}", exc_info=True)

    def _parse_price_to_beat(self, strike_str: str) -> float:
        if not strike_str: return 0.0
        try:
            clean = strike_str.replace('$', '').replace(',', '').strip()
            if not clean or clean in ('N/A', 'Pending', 'Loading', '--', 'None', ''):
                return 0.0
            return float(clean)
        except: return 0.0

    async def update_market_status(self) -> None:
        """Poll for active market and handle discovery."""
        try:
            market = await self.connector.get_active_market()
            if market:
                # Check if it's a new market
                current = await self.market_manager.get_market()
                if not current or current.get('market_id') != market.get('market_id'):
                    await self._on_new_market_found(market)
            
            # Always try to resolve strike if pending/dynamic
            await self._try_resolve_pending_strike()
        except Exception as e:
            logger.error(f"Error updating market status: {e}")

    async def _on_new_market_found(self, market: Dict[str, Any]) -> None:
        """Handle transition to new market."""
        self.log_msg(f"New Market Found: {market.get('title', 'Unknown')}")
        self.resolution_shown = False
        
        # Set market in manager
        await self.market_manager.set_market(market)
        
        # Subscribe WebSocket
        await self.ws_manager.subscribe_to_market(market)
        
        # Reset positions on new market
        self._yes_position = 0.0
        self._no_position = 0.0
        self._last_position_update = 0.0 # Force update for new market
        
        # Emit update
        ends = self._format_ends(market.get('end_date', ''))
        starts = self._format_starts(market.get('start_date', ''))
        market_name = market.get('title', 'Market')
        self._emit('market_update', market.get('price_to_beat', 'N/A'), ends, market_name, starts)

        # Immediately try to resolve strike if dynamic
        if market.get('price_to_beat') in ('Dynamic', 'Pending', 'N/A', '', None):
            asyncio.create_task(self._try_resolve_pending_strike())

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
        """Handle market expiry and emit resolution event."""
        self.resolution_shown = True
        self.log_msg("⚠️ MARKET EXPIRED - Waiting for resolution...")
        self._emit('resolution', "EXPIRED")
        
    async def _try_resolve_pending_strike(self) -> None:
        """Resolve dynamic strike prices using Chainlink or Binance data."""
        market = await self.market_manager.get_market()
        if not market: return
        
        strike = market.get('price_to_beat')
        if strike not in ("Dynamic", "Pending", "N/A", "None", "", None):
            return
            
        # Throttle resolution attempts
        now = time.time()
        if now - self._last_strike_resolve_attempt < self._strike_resolve_interval:
            return
        self._last_strike_resolve_attempt = now

        try:
            # For dynamic strikes, the strike is the price at the market start time
            start_dt = await self.market_manager.get_market_start_time()
            if not start_dt:
                logger.debug("No market start time available for strike resolution")
                return
                
            # Try to get from RTDS history first (fastest)
            logger.info(f"Attempting to resolve dynamic strike for {start_dt}...")
            
            strike_val = None
            
            # METHOD 1: Fetch exact value from Polymarket Frontend (100% Accuracy)
            # This is the user's preferred method to ensure displayed values match
            if 'event_slug' in market:
                 end_dt = pd.Timestamp(market.get('end_date'))
                 strike_val = await self.connector.get_strike_from_polymarket_page(
                     market.get('event_slug'), start_dt, end_dt
                 )
            
            # METHOD 2: RTDS History (Fast local cache)
            if strike_val is None:
                strike_val = self.rtds_manager.get_chainlink_price_at(start_dt)
                logger.info(f"RTDS lookup result: {strike_val}")
            
            # METHOD 3: Chainlink API (Fallback)
            if strike_val is None:
                logger.debug(f"Strike not in RTDS history for {start_dt}, trying Chainlink API...")
                strike_val = await self.connector.get_chainlink_price_at(start_dt)
            
            # If still None, try Binance API as fallback
            if strike_val is None:
                logger.info(f"Chainlink APIs failed for {start_dt}, trying Binance fallback...")
                btc_price_str = await self.connector.get_btc_price_at(start_dt)
                if btc_price_str and btc_price_str != "N/A":
                    try:
                        strike_val = float(btc_price_str)
                        logger.info(f"✓ Resolved dynamic strike using Binance fallback: ${strike_val:,.2f}")
                    except (ValueError, TypeError):
                        pass
                
            if strike_val:
                logger.info(f"✓ Final resolved strike value: ${strike_val:,.2f} at {start_dt}")
                # Update market in manager
                market['price_to_beat'] = f"{strike_val:,.2f}"
                await self.market_manager.set_market(market)
                # Emit update
                ends = self._format_ends(market.get('end_date', ''))
                starts = self._format_starts(market.get('start_date', ''))
                market_name = market.get('title', 'Market')
                self._emit('market_update', market['price_to_beat'], ends, market_name, starts)
            else:
                logger.warning(f"Failed to resolve dynamic strike for {start_dt} after all attempts")
                
        except Exception as e:
            logger.error(f"Error resolving strike: {e}", exc_info=True)

    async def _update_positions(self) -> None:
        """Update cached positions from connector."""
        try:
            token_map = await self.market_manager.get_token_map()
            if not token_map: return
            
            up_id = token_map.get('Up')
            if up_id:
                self._yes_position = await self.connector.get_token_balance(up_id)
                
            down_id = token_map.get('Down')
            if down_id:
                self._no_position = await self.connector.get_token_balance(down_id)
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