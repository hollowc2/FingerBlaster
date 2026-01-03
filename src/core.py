"""Refactored core business logic controller with improved architecture.

Key improvements:
- Better separation of concerns (SRP)
- Proper callback management with cleanup
- Improved error handling
- Type safety with comprehensive type hints
- Performance optimizations
"""

import asyncio
import logging
import time
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd

from connectors.polymarket import PolymarketConnector
from src.config import AppConfig
from src.engine import (
    MarketDataManager, HistoryManager, WebSocketManager, 
    OrderExecutor, RTDSManager
)
from src.analytics import AnalyticsEngine, AnalyticsSnapshot, TimerUrgency, EdgeDirection
from src.strategy_data_sync import StrategyDataProvider, set_provider

logger = logging.getLogger("FingerBlaster")


# Event types for callback registration
CALLBACK_EVENTS: Tuple[str, ...] = (
    'market_update',      # (strike: str, ends: str, market_name: str)
    'btc_price_update',   # (price: float)
    'price_update',       # (yes_price: float, no_price: float, best_bid: float, best_ask: float)
    'account_stats_update',  # (balance, yes_bal, no_bal, size, avg_yes, avg_no)
    'countdown_update',   # (time_str: str, urgency: TimerUrgency, seconds_remaining: int)
    'prior_outcomes_update',  # (outcomes: List[str])
    'resolution',         # (resolution: Optional[str])
    'log',               # (message: str)
    'chart_update',      # (data, ...) - variable args based on chart type
    'analytics_update',  # (snapshot: AnalyticsSnapshot)
    'order_submitted',   # (side: str, size: float, price: float)
    'order_filled',      # (side: str, size: float, price: float, order_id: str)
    'order_failed',      # (side: str, size: float, error: str)
    'flatten_started',   # () - flatten operation started
    'flatten_completed', # (orders_processed: int) - flatten operation completed
    'flatten_failed',    # (error: str) - flatten operation failed
)


class CallbackManager:
    """Manages event callbacks with proper cleanup.
    
    This class addresses the memory leak issue by:
    1. Supporting callback unregistration
    2. Providing cleanup methods
    3. Using regular lists (not WeakSet) to prevent premature garbage collection
    
    Supported events (defined in CALLBACK_EVENTS):
    - market_update: Market found/changed
    - btc_price_update: BTC price changed
    - price_update: YES/NO prices changed
    - account_stats_update: Balance/position changed
    - countdown_update: Timer tick
    - prior_outcomes_update: Prior outcomes display
    - resolution: Market resolved
    - log: Log message
    - chart_update: Chart data changed
    """
    
    def __init__(self):
        """Initialize callback manager with all supported event types."""
        # Use regular lists (not WeakSet) to prevent callbacks from being GC'd
        # The unregister_callback method provides proper cleanup
        self._callbacks: Dict[str, List[Callable]] = {
            event: [] for event in CALLBACK_EVENTS
        }
        # Use threading.Lock for synchronous methods, asyncio.Lock for async methods
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
    
    def register(self, event: str, callback: Callable) -> bool:
        """Register a callback for a specific event.
        
        Args:
            event: Event name
            callback: Callback function
            
        Returns:
            True if registered successfully, False otherwise
        """
        if event not in self._callbacks:
            logger.warning(f"Unknown event type: {event}")
            return False
        
        with self._lock:
            if callback not in self._callbacks[event]:
                self._callbacks[event].append(callback)
        return True
    
    def unregister(self, event: str, callback: Callable) -> bool:
        """Unregister a callback for a specific event.
        
        Args:
            event: Event name
            callback: Callback function to remove
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if event not in self._callbacks:
            return False
        
        with self._lock:
            try:
                if callback in self._callbacks[event]:
                    self._callbacks[event].remove(callback)
                return True
            except (ValueError, TypeError):
                return False
    
    def clear(self, event: Optional[str] = None) -> None:
        """Clear callbacks for an event or all events.
        
        Args:
            event: Event name to clear, or None to clear all
        """
        with self._lock:
            if event:
                if event in self._callbacks:
                    self._callbacks[event].clear()
            else:
                for callbacks in self._callbacks.values():
                    callbacks.clear()
    
    def get_callbacks(self, event: str) -> List[Callable]:
        """Get a snapshot of callbacks for an event (thread-safe).
        
        Args:
            event: Event name
            
        Returns:
            List of callbacks
        """
        with self._lock:
            return list(self._callbacks.get(event, []))
    
    async def emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered callbacks (async version).
        
        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if event not in self._callbacks:
            logger.warning(f"Unknown event type: {event}")
            return
        
        # Get a snapshot of callbacks to avoid issues during iteration
        async with self._async_lock:
            callbacks = list(self._callbacks[event])
        
        # Execute callbacks outside the lock to avoid deadlocks
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Try to get the running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(callback(*args, **kwargs))
                    except RuntimeError:
                        # No running loop - schedule it
                        asyncio.ensure_future(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}", exc_info=True)


class PositionTracker:
    """Tracks average entry prices for YES/NO positions.
    
    Provides clean separation of position tracking logic with proper
    validation and weighted average calculations.
    """
    
    def __init__(self):
        """Initialize position tracker."""
        self._avg_prices: Dict[str, Optional[float]] = {
            'YES': None,
            'NO': None,
        }
    
    def update_position(
        self, 
        side: str, 
        old_balance: float,
        new_shares: float, 
        entry_price: float
    ) -> bool:
        """Update position with new trade.
        
        Args:
            side: Which side was traded ('YES' or 'NO')
            old_balance: Balance before trade
            new_shares: Shares received from trade
            entry_price: Price paid per share
            
        Returns:
            True if update was successful, False otherwise
        """
        side = side.upper()
        if side not in ('YES', 'NO'):
            logger.warning(f"Invalid side for position update: {side}")
            return False
        
        if entry_price <= 0 or new_shares <= 0:
            logger.warning(
                f"Invalid position update: price={entry_price}, shares={new_shares}"
            )
            return False
        
        total_shares = old_balance + new_shares
        if total_shares <= 0:
            self._avg_prices[side] = None
            return True
        
        current_avg = self._avg_prices[side]
        if current_avg is not None and old_balance > 0:
            # Weighted average: (old_cost + new_cost) / total_shares
            old_cost = current_avg * old_balance
            new_cost = entry_price * new_shares
            self._avg_prices[side] = (old_cost + new_cost) / total_shares
        else:
            # First position
            self._avg_prices[side] = entry_price
        
        return True
    
    def reset(self, side: Optional[str] = None) -> None:
        """Reset position tracking for one or all sides.
        
        Args:
            side: Side to reset ('YES', 'NO'), or None for both
        """
        if side is None:
            self._avg_prices['YES'] = None
            self._avg_prices['NO'] = None
        elif side.upper() in ('YES', 'NO'):
            self._avg_prices[side.upper()] = None
    
    def get_average_price(self, side: str) -> Optional[float]:
        """Get current average entry price for a side.
        
        Args:
            side: Side to get price for ('YES' or 'NO')
            
        Returns:
            Average entry price or None if no position
        """
        return self._avg_prices.get(side.upper())
    
    @property
    def avg_entry_price_yes(self) -> Optional[float]:
        """Get YES average entry price."""
        return self._avg_prices['YES']
    
    @avg_entry_price_yes.setter
    def avg_entry_price_yes(self, value: Optional[float]) -> None:
        """Set YES average entry price."""
        self._avg_prices['YES'] = value
    
    @property
    def avg_entry_price_no(self) -> Optional[float]:
        """Get NO average entry price."""
        return self._avg_prices['NO']
    
    @avg_entry_price_no.setter
    def avg_entry_price_no(self, value: Optional[float]) -> None:
        """Set NO average entry price."""
        self._avg_prices['NO'] = value


class FingerBlasterCore:
    """Shared business logic controller with improved architecture.
    
    Refactored improvements:
    1. Separated callback management into CallbackManager
    2. Added proper resource cleanup
    3. Improved error handling with recovery strategies
    4. Better type safety
    5. Performance optimizations
    """
    
    def __init__(self, connector: Optional[PolymarketConnector] = None):
        """Initialize the core controller.
        
        Args:
            connector: Optional connector instance (for dependency injection)
        """
        self.config = AppConfig()
        self.connector = connector or PolymarketConnector()
        
        # Initialize managers
        self.market_manager = MarketDataManager(self.config)
        self.history_manager = HistoryManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.connector)
        self.ws_manager = WebSocketManager(
            self.config,
            self.market_manager,
            self._on_ws_message
        )
        # RTDS manager for real-time BTC prices matching Polymarket
        self.rtds_manager = RTDSManager(
            self.config,
            self._on_rtds_btc_price
        )
        
        # Analytics engine for quantitative analysis
        self.analytics_engine = AnalyticsEngine()
        
        # Strategy data provider for low-latency synchronous access
        self.strategy_data_provider = StrategyDataProvider(self)
        set_provider(self.strategy_data_provider)
        
        # CEX price tracking for oracle lag
        self._cex_btc_price: Optional[float] = None
        self._cex_btc_timestamp: Optional[float] = None
        
        # State
        self.resolution_shown = False
        self.last_resolution: Optional[str] = None
        self.displayed_prior_outcomes: List[str] = []
        self.last_chart_update: float = 0.0
        self.selected_size: float = 1.0
        
        # Position tracking with average entry prices
        self.position_tracker = PositionTracker()
        
        # Backward-compatible property accessors are defined below
        
        # Callback management
        self.callback_manager = CallbackManager()
        
        # Cached values for performance
        self._cached_prices: Optional[Tuple[float, float, float, float]] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 0.1  # 100ms cache
        self._cached_market_id: Optional[str] = None  # Track market for cache invalidation
        
        # Cache for prior outcomes from API (refresh periodically)
        self._prior_outcomes_cache: List[Dict[str, Any]] = []
        self._prior_outcomes_cache_timestamp: float = 0.0
        self._prior_outcomes_cache_ttl: float = 60.0  # Cache for 60 seconds
    
    # Backward-compatible property accessors for position tracking
    @property
    def avg_entry_price_yes(self) -> Optional[float]:
        """Get YES average entry price (backward compatible)."""
        return self.position_tracker.avg_entry_price_yes
    
    @avg_entry_price_yes.setter
    def avg_entry_price_yes(self, value: Optional[float]) -> None:
        """Set YES average entry price (backward compatible)."""
        self.position_tracker.avg_entry_price_yes = value
    
    @property
    def avg_entry_price_no(self) -> Optional[float]:
        """Get NO average entry price (backward compatible)."""
        return self.position_tracker.avg_entry_price_no
    
    @avg_entry_price_no.setter
    def avg_entry_price_no(self, value: Optional[float]) -> None:
        """Set NO average entry price (backward compatible)."""
        self.position_tracker.avg_entry_price_no = value
    
    @staticmethod
    def calculate_spreads(
        best_bid: float, 
        best_ask: float
    ) -> Tuple[str, str]:
        """Calculate formatted spread strings for YES and NO.
        
        Centralizes spread calculation logic (DRY principle).
        
        Args:
            best_bid: Best bid price in YES terms
            best_ask: Best ask price in YES terms
            
        Returns:
            Tuple of (yes_spread_str, no_spread_str)
        """
        # YES spread is in YES terms: best_bid / best_ask
        yes_spread = f"{best_bid:.2f} / {best_ask:.2f}"
        
        # NO spread is in NO terms: (1 - best_ask) / (1 - best_bid)
        no_best_bid = 1.0 - best_ask if best_ask < 1.0 else 0.0
        no_best_ask = 1.0 - best_bid if best_bid > 0.0 else 1.0
        no_spread = f"{no_best_bid:.2f} / {no_best_ask:.2f}"
        
        return yes_spread, no_spread
    
    async def start_rtds(self) -> None:
        """Start RTDS for real-time BTC prices.
        
        This should be called early to ensure accurate prices matching Polymarket.
        Can be called independently of market status.
        """
        try:
            await self.rtds_manager.start()
            logger.info("RTDS started for real-time BTC prices")
        except Exception as e:
            logger.error(f"Error starting RTDS: {e}", exc_info=True)
    
    def register_callback(self, event: str, callback: Callable) -> bool:
        """Register a callback for a specific event.
        
        Args:
            event: Event name ('market_update', 'btc_price_update', etc.)
            callback: Callback function to call when event occurs
            
        Returns:
            True if registered successfully, False otherwise
        """
        return self.callback_manager.register(event, callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """Unregister a callback for a specific event.
        
        Args:
            event: Event name
            callback: Callback function to remove
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        return self.callback_manager.unregister(event, callback)
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered callbacks.
        
        This method works like the original - it directly handles callbacks
        to support both sync and async contexts (especially Qt integration).
        
        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        # Get a snapshot of callbacks (thread-safe)
        callbacks = self.callback_manager.get_callbacks(event)
        
        # Execute callbacks outside the lock to avoid deadlocks
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Try to get the running event loop, create task if possible
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(callback(*args, **kwargs))
                    except RuntimeError:
                        # No running loop - this can happen in Qt
                        # Try to get any event loop
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(callback(*args, **kwargs))
                            else:
                                # Loop exists but not running - schedule it
                                asyncio.ensure_future(callback(*args, **kwargs), loop=loop)
                        except (RuntimeError, AttributeError):
                            # No loop available - this will be handled by Qt integration
                            # Just log a warning, the Qt layer will handle it
                            logger.debug(f"No event loop available for async callback {event}")
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}", exc_info=True)
    
    def log_msg(self, message: str) -> None:
        """Log message and emit to UI.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(message)
        self._emit('log', f"[{timestamp}] {message}")
    
    async def _on_ws_message(self, item: Dict[str, Any]) -> None:
        """Handle WebSocket message by recalculating price.
        
        Args:
            item: WebSocket message item
        """
        await self._recalc_price()
    
    async def _on_rtds_btc_price(self, btc_price: float) -> None:
        """Handle RTDS BTC price update.
        
        This uses the same price source as Polymarket for accurate resolution.
        
        Args:
            btc_price: BTC price from RTDS
        """
        try:
            if btc_price and isinstance(btc_price, (int, float)) and btc_price > 0:
                btc_price = float(btc_price)
                await self.history_manager.add_btc_price(btc_price)
                
                # Emit BTC price update
                self._emit('btc_price_update', btc_price)
                
                # Emit BTC chart update
                prices = await self.history_manager.get_btc_history()
                if len(prices) >= 2:
                    market = await self.market_manager.get_market()
                    strike_val = None
                    if market:
                        strike_str = str(market.get('strike_price', ''))
                        strike_val = self._parse_strike(strike_str)
                    self._emit('chart_update', prices, strike_val, 'btc')
        except Exception as e:
            logger.error(f"Error handling RTDS BTC price: {e}", exc_info=True)
    
    def _invalidate_price_cache(self) -> None:
        """Invalidate the price cache.
        
        Call this when market changes to ensure fresh price calculation.
        """
        self._cached_prices = None
        self._cache_timestamp = 0.0
        self._cached_market_id = None
    
    async def _recalc_price(self) -> None:
        """Recalculate mid price and update UI with caching.
        
        Uses caching to avoid redundant calculations.
        Automatically invalidates cache on market change.
        """
        now = time.time()
        
        # Check if market has changed - invalidate cache if so
        current_market = await self.market_manager.get_market()
        current_market_id = current_market.get('market_id') if current_market else None
        
        if current_market_id != self._cached_market_id:
            self._invalidate_price_cache()
            self._cached_market_id = current_market_id
        
        # Use cached value if recent and valid
        if (self._cached_prices is not None and 
            now - self._cache_timestamp < self._cache_ttl):
            yes_price, no_price, best_bid, best_ask = self._cached_prices
        else:
            # Calculate new prices
            prices = await self.market_manager.calculate_mid_price()
            yes_price, no_price, best_bid, best_ask = prices
            
            # Update cache
            self._cached_prices = prices
            self._cache_timestamp = now
        
        # Emit price update
        self._emit('price_update', yes_price, no_price, best_bid, best_ask)
        
        # Update history
        market_start_time = await self.market_manager.get_market_start_time()
        if market_start_time:
            now_ts = pd.Timestamp.now(tz='UTC')
            elapsed = (now_ts - market_start_time).total_seconds()
            await self.history_manager.add_price_point(
                elapsed, yes_price, market_start_time
            )
        
        # Emit chart update (throttled)
        if now - self.last_chart_update >= self.config.chart_update_throttle_seconds:
            self.last_chart_update = now
            history = await self.history_manager.get_yes_history()
            self._emit('chart_update', history)
    
    async def _resolve_dynamic_strike(
        self, 
        new_market: Dict[str, Any], 
        market_start_time: pd.Timestamp
    ) -> str:
        """Resolve dynamic strike price using RTDS or API fallback.
        
        This method attempts to get the Chainlink BTC/USD price at market start
        using multiple fallback strategies:
        1. RTDS historical data (if app was running when market started)
        2. Chainlink API historical lookup
        3. Binance API fallback
        
        Args:
            new_market: Market data dictionary (modified in place)
            market_start_time: When the market started
            
        Returns:
            Resolved strike price as formatted string, or "Dynamic" if unresolved
        """
        now = pd.Timestamp.now(tz='UTC')
        time_since_start = (now - market_start_time).total_seconds()
        
        # Strategy 1: Try RTDS if market started recently (within 2 minutes)
        if time_since_start <= 120:
            chainlink_price = self.rtds_manager.get_chainlink_price_at(market_start_time)
            if chainlink_price and chainlink_price > 0:
                strike = f"{chainlink_price:,.2f}"
                self.log_msg(
                    f"Dynamic strike: Using RTDS Chainlink price at market start: ${chainlink_price:,.2f}"
                )
                new_market['strike_price'] = strike
                await self.market_manager.set_market(new_market)
                return strike
        else:
            self.log_msg(
                f"Market started {time_since_start:.0f}s ago - using API for historical price lookup"
            )
        
        # Strategy 2: Try Chainlink API (with short timeout to prevent blocking)
        print("[CORE] Trying Chainlink API with 2s timeout...", flush=True)
        chainlink_price = None
        try:
            chainlink_price = await asyncio.wait_for(
                self.connector.get_chainlink_price_at(market_start_time),
                timeout=2.0
            )
            print(f"[CORE] Chainlink API returned: {chainlink_price}", flush=True)
        except asyncio.TimeoutError:
            print("[CORE] Chainlink API timeout, skipping", flush=True)
        except Exception as e:
            print(f"[CORE] Chainlink API error: {e}", flush=True)
        
        if chainlink_price and chainlink_price > 0:
            strike = f"{chainlink_price:,.2f}"
            self.log_msg(
                f"Dynamic strike: Using Chainlink API price at market start: ${chainlink_price:,.2f}"
            )
            new_market['strike_price'] = strike
            await self.market_manager.set_market(new_market)
            return strike
        
        # Strategy 3: Fallback to Binance (with short timeout)
        print("[CORE] Trying Binance fallback with 2s timeout...", flush=True)
        price_str = None
        try:
            price_str = await asyncio.wait_for(
                self.connector.get_btc_price_at(market_start_time),
                timeout=2.0
            )
            print(f"[CORE] Binance fallback returned: {price_str}", flush=True)
        except asyncio.TimeoutError:
            print("[CORE] Binance fallback timeout", flush=True)
        except Exception as e:
            print(f"[CORE] Binance fallback error: {e}", flush=True)
        
        if price_str and price_str != "N/A":
            self.log_msg(
                f"WARNING: Dynamic strike using Binance fallback (Chainlink not available): ${price_str}"
            )
            new_market['strike_price'] = price_str
            await self.market_manager.set_market(new_market)
            return price_str
        
        # Could not resolve - use current market price as fallback
        print("[CORE] Could not resolve dynamic strike, using current market price", flush=True)
        try:
            # Try to get current BTC price as fallback
            current_price = await asyncio.wait_for(
                self.connector.get_btc_price(),
                timeout=2.0
            )
            if current_price and current_price > 0:
                strike = f"{current_price:,.2f}"
                self.log_msg(
                    f"WARNING: Could not determine dynamic strike at market start ({market_start_time}). "
                    f"Market started {time_since_start:.0f}s ago. Using current price fallback: ${strike}"
                )
                new_market['strike_price'] = strike
                await self.market_manager.set_market(new_market)
                return strike
        except Exception as e:
            print(f"[CORE] Could not get current price fallback: {e}", flush=True)

        # Last resort - return placeholder
        self.log_msg(
            f"WARNING: Could not determine dynamic strike price at market start ({market_start_time}). "
            f"Market started {time_since_start:.0f}s ago. This may cause incorrect strike calculation."
        )
        return "Pending"
    
    async def _log_current_btc_prices(self) -> None:
        """Log current BTC prices from various sources for comparison."""
        chainlink_price = self.rtds_manager.get_chainlink_price()
        rtds_price = self.rtds_manager.get_current_price()
        
        if chainlink_price:
            self.log_msg(f"RTDS Chainlink BTC/USD: ${chainlink_price:,.2f}")
        if rtds_price and rtds_price != chainlink_price:
            self.log_msg(f"RTDS Binance BTC/USDT: ${rtds_price:,.2f}")
        if not rtds_price:
            try:
                binance_price = await asyncio.wait_for(
                    self.connector.get_btc_price(),
                    timeout=5.0
                )
                if binance_price:
                    self.log_msg(f"Binance API BTC Price: ${binance_price:,.2f}")
            except asyncio.TimeoutError:
                self.log_msg("Binance API timeout")
    
    async def update_market_status(self) -> None:
        """Update market status and search for new markets.
        
        Improved error handling with recovery strategies.
        """
        print("[CORE] update_market_status called", flush=True)
        market = await self.market_manager.get_market()
        print(f"[CORE] Current market: {'exists' if market else 'None'}", flush=True)
        if not market:
            # Search for new market
            try:
                print("[CORE] Searching for new market...", flush=True)
                try:
                    new_market = await asyncio.wait_for(
                        self.connector.get_active_market(),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    print("[CORE] get_active_market timeout", flush=True)
                    new_market = None
                print(f"[CORE] get_active_market returned: {'found' if new_market else 'None'}", flush=True)
                if new_market:
                    print("[CORE] Setting market...", flush=True)
                    success = await self.market_manager.set_market(new_market)
                    print(f"[CORE] set_market: {success}", flush=True)
                    if success:
                        print("[CORE] Clearing YES history...", flush=True)
                        await self.history_manager.clear_yes_history()
                        strike = str(new_market.get('strike_price', 'N/A'))

                        # Resolve dynamic strikes
                        print("[CORE] Getting market start time...", flush=True)
                        market_start_time = await self.market_manager.get_market_start_time()
                        if strike == "Dynamic" and market_start_time:
                            print("[CORE] Resolving dynamic strike...", flush=True)
                            # Set strike to "Loading" while resolving
                            new_market['strike_price'] = "Loading"
                            await self.market_manager.set_market(new_market)
                            # Emit market update to show "Loading"
                            self._emit('market_update', "Loading", new_market.get('end_date', 'N/A'), new_market.get('question', 'Market'))

                            strike = await self._resolve_dynamic_strike(new_market, market_start_time)

                        # Ensure resolved strike is stored in market data for resolution calculation
                        # This is critical - resolution reads from market.get('strike_price')
                        if strike and strike not in ('N/A', 'None', ''):
                            new_market['strike_price'] = strike
                            # Update the stored market data
                            await self.market_manager.set_market(new_market)
                            logger.info(f"Strike stored in market data: {strike}")
                        else:
                            # Keep whatever strike we have (could be Loading, Pending, N/A, etc.)
                            logger.warning(f"Strike resolution incomplete or failed: {strike}")

                        self.log_msg(
                            f"Market Found: Strike={strike}, End={new_market.get('end_date', 'N/A')}"
                        )
                        print(f"[CORE] Market found: Strike={strike}", flush=True)
                        
                        # Log current BTC prices for comparison
                        print("[CORE] Logging BTC prices...", flush=True)
                        await self._log_current_btc_prices()
                        
                        print("[CORE] Starting WS manager...", flush=True)
                        await self.ws_manager.start()
                        print("[CORE] Starting RTDS manager...", flush=True)
                        # Start RTDS for real-time BTC prices
                        await self.rtds_manager.start()
                        print("[CORE] RTDS started", flush=True)
                        
                        # Emit market update
                        ends = self._format_ends(new_market.get('end_date', 'N/A'))
                        # Extract market name (prefer question, then title, then slug)
                        market_name = (new_market.get('question') or 
                                      new_market.get('title') or 
                                      new_market.get('event_slug') or 
                                      'Market')
                        self._emit('market_update', strike, ends, market_name)
                        
                        # Reset countdown display
                        print("[CORE] Updating countdown...", flush=True)
                        await self.update_countdown()
                        
                        # Re-check prior outcomes after a short delay
                        # (API may not have updated immediately after market close)
                        # Run in background task to avoid blocking startup
                        print("[CORE] Scheduling prior outcomes check in background...", flush=True)
                        asyncio.create_task(self._check_and_add_prior_outcomes_with_timeout(retry_count=3))
            except Exception as e:
                logger.error(f"Error searching for market: {e}", exc_info=True)
                # Recovery: retry after delay
                await asyncio.sleep(1.0)
        
        # Check market expiry with timeout to prevent blocking
        try:
            await asyncio.wait_for(self.check_if_market_expired(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("check_if_market_expired timed out")
        except Exception as e:
            logger.error(f"Error in check_if_market_expired: {e}", exc_info=True)
    
    async def update_btc_price(self) -> None:
        """Update BTC price and refresh chart.
        
        Now uses RTDS for real-time prices matching Polymarket.
        Falls back to Binance API if RTDS is not available.
        """
        try:
            # Try RTDS first (matches Polymarket's price source)
            rtds_price = self.rtds_manager.get_current_price()
            if rtds_price and rtds_price > 0:
                # RTDS is handling updates via callback, just log
                logger.debug(f"Using RTDS BTC price: ${rtds_price:,.2f}")
                return
            
            # Fallback to Binance API if RTDS not available
            logger.debug("RTDS price not available, falling back to Binance API")
            try:
                price = await asyncio.wait_for(
                    self.connector.get_btc_price(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.debug("Binance API timeout in update_btc_price")
                return
            if price and isinstance(price, (int, float)) and price > 0:
                btc_price = float(price)
                await self.history_manager.add_btc_price(btc_price)
                
                # Emit BTC price update
                self._emit('btc_price_update', btc_price)
                
                # Emit BTC chart update
                prices = await self.history_manager.get_btc_history()
                if len(prices) >= 2:
                    market = await self.market_manager.get_market()
                    strike_val = None
                    if market:
                        strike_str = str(market.get('strike_price', ''))
                        strike_val = self._parse_strike(strike_str)
                    self._emit('chart_update', prices, strike_val, 'btc')
        except Exception as e:
            logger.error(f"Error updating BTC price: {e}", exc_info=True)
    
    async def update_account_stats(self) -> None:
        """Update account statistics with improved error handling."""
        try:
            token_map = await self.market_manager.get_token_map()
            
            def get_stats() -> Tuple[float, float, float]:
                """Get account statistics synchronously."""
                bal = self.connector.get_usdc_balance()
                yes_bal = 0.0
                no_bal = 0.0
                if token_map:
                    y_id = token_map.get('YES')
                    n_id = token_map.get('NO')
                    if y_id:
                        yes_bal = self.connector.get_token_balance(y_id)
                    if n_id:
                        no_bal = self.connector.get_token_balance(n_id)
                return float(bal or 0.0), float(yes_bal or 0.0), float(no_bal or 0.0)
            
            try:
                bal, y, n = await asyncio.wait_for(
                    asyncio.to_thread(get_stats),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.debug("Account stats fetch timeout")
                return
            
            # Reset average entry prices if positions are zero (using threshold to handle floating point precision)
            MIN_BALANCE_THRESHOLD = 0.1
            if y <= MIN_BALANCE_THRESHOLD:
                self.avg_entry_price_yes = None
                self.position_tracker.reset('YES')
            if n <= MIN_BALANCE_THRESHOLD:
                self.avg_entry_price_no = None
                self.position_tracker.reset('NO')
            
            # Pass average entry prices to UI
            self._emit('account_stats_update', bal, y, n, self.selected_size, 
                      self.avg_entry_price_yes, self.avg_entry_price_no)
        except Exception as e:
            logger.error(f"Error updating account stats: {e}", exc_info=True)
    
    async def update_countdown(self) -> None:
        """Update the countdown timer with improved timezone handling and urgency."""
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            # Ensure timezone-aware timestamp
            dt_end = pd.Timestamp(end_str)
            if dt_end.tz is None:
                dt_end = dt_end.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            diff = dt_end - now
            total_seconds = diff.total_seconds()
            
            # Clamp to zero to prevent negative display before showing EXPIRED
            total_seconds = max(0.0, total_seconds)
            seconds_remaining = int(total_seconds)
            
            if total_seconds <= 0:
                time_str = "EXPIRED"
            else:
                secs = int(total_seconds)
                mins = secs // 60
                remaining_secs = secs % 60
                time_str = f"{mins:02d}:{remaining_secs:02d}"
            
            # Calculate timer urgency
            urgency = self.analytics_engine.get_timer_urgency(seconds_remaining)
            
            self._emit('countdown_update', time_str, urgency, seconds_remaining)
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}", exc_info=True)
    
    async def update_analytics(self) -> None:
        """Update full analytics snapshot and emit to UI."""
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            # Get BTC price
            btc_price = self.rtds_manager.get_current_price()
            if not btc_price or btc_price <= 0:
                btc_history = await self.history_manager.get_btc_history()
                btc_price = btc_history[-1] if btc_history else 0.0
            
            if btc_price <= 0:
                return
            
            # Get strike price
            strike_str = str(market.get('strike_price', '')).replace(',', '').replace('$', '').strip()
            strike_price = self._parse_strike(strike_str)
            if not strike_price or strike_price <= 0:
                return
            
            # Get time remaining
            end_str = market.get('end_date')
            if not end_str:
                return
            
            dt_end = pd.Timestamp(end_str)
            if dt_end.tz is None:
                dt_end = dt_end.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            time_remaining = max(0, int((dt_end - now).total_seconds()))
            
            # Get current prices
            prices = await self.market_manager.calculate_mid_price()
            yes_price, no_price, best_bid, best_ask = prices
            
            # Get order book for liquidity/slippage
            order_book = await self.market_manager.get_raw_order_book()
            
            # Get positions
            token_map = await self.market_manager.get_token_map()
            yes_position = 0.0
            no_position = 0.0
            if token_map:
                y_id = token_map.get('YES')
                n_id = token_map.get('NO')
                if y_id:
                    yes_position = self.connector.get_token_balance(y_id)
                if n_id:
                    no_position = self.connector.get_token_balance(n_id)
            
            # Get prior outcomes for regime detection
            prior_outcomes_data = await self._get_prior_outcomes()
            outcomes_list = [o.get('outcome', '') for o in prior_outcomes_data if isinstance(o, dict)]
            
            # Update oracle lag with CEX price
            chainlink_price = self.rtds_manager.get_chainlink_price()
            if self._cex_btc_price:
                self.analytics_engine.update_oracle_prices(
                    chainlink_price=chainlink_price,
                    cex_price=self._cex_btc_price,
                    chainlink_timestamp=self._cex_btc_timestamp,
                    cex_timestamp=time.time()
                )
            
            # Generate full analytics snapshot
            snapshot = await self.analytics_engine.generate_snapshot(
                btc_price=btc_price,
                strike_price=strike_price,
                time_remaining_seconds=time_remaining,
                yes_market_price=yes_price,
                no_market_price=no_price,
                order_book=order_book,
                yes_position=yes_position,
                no_position=no_position,
                avg_entry_yes=self.avg_entry_price_yes,
                avg_entry_no=self.avg_entry_price_no,
                prior_outcomes=outcomes_list,
                order_size_usd=self.selected_size
            )
            
            # Update strategy data provider cache (synchronous, low-latency)
            self.strategy_data_provider.update_cache(
                snapshot=snapshot,
                market=market,
                btc_price=btc_price,
                strike=strike_price,
                time_remaining=time_remaining,
                prior_outcomes=outcomes_list
            )
            
            self._emit('analytics_update', snapshot)
            
        except Exception as e:
            logger.debug(f"Error updating analytics: {e}", exc_info=True)
    
    async def fetch_cex_price(self) -> None:
        """Fetch CEX price for oracle lag comparison."""
        try:
            price = await self.connector.get_btc_price()
            if price and isinstance(price, (int, float)) and price > 0:
                self._cex_btc_price = float(price)
                self._cex_btc_timestamp = time.time()
        except Exception as e:
            logger.debug(f"Error fetching CEX price: {e}")
    
    async def check_if_market_expired(self) -> None:
        """Check if market has expired and show resolution.
        
        Improved error handling and timezone consistency.
        """
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            # Ensure timezone-aware timestamp
            end_dt = pd.Timestamp(end_str)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            if now > end_dt and not self.resolution_shown:
                self.resolution_shown = True
                await self._show_resolution()
                await asyncio.sleep(self.config.resolution_overlay_duration)
                await self._reset_market_after_resolution()
        except Exception as e:
            logger.error(f"Error checking market expiry: {e}", exc_info=True)
    
    async def _show_resolution(self) -> None:
        """Calculate and emit resolution with improved error handling.
        
        Uses Chainlink BTC price at market end time for accurate resolution.
        
        IMPORTANT: Polymarket resolves based on Chainlink price at the exact
        market end timestamp, not the current price. We try to match this.
        """
        try:
            market = await self.market_manager.get_market()
            if not market:
                return
            
            # Get market end time for accurate price lookup
            end_str = market.get('end_date')
            market_end_time = None
            if end_str:
                market_end_time = pd.Timestamp(end_str)
                if market_end_time.tz is None:
                    market_end_time = market_end_time.tz_localize('UTC')
            
            # Get BTC price - try multiple sources in order of accuracy
            btc_price = None
            price_source = "unknown"
            
            # 1. Try to get Chainlink price at market end time (most accurate)
            if market_end_time:
                # Log the exact timestamp we're looking for
                logger.info(f"Looking up Chainlink price at market end: {market_end_time} (epoch: {int(market_end_time.timestamp() * 1000)}ms)")
                
                chainlink_at_end = self.rtds_manager.get_chainlink_price_at(market_end_time)
                if chainlink_at_end and chainlink_at_end > 0:
                    btc_price = chainlink_at_end
                    price_source = f"Chainlink@{market_end_time.strftime('%H:%M:%S')} UTC"
                    logger.info(f"Using Chainlink price at market end: ${btc_price:,.2f}")
            
            # 2. Fallback to current Chainlink price
            if not btc_price:
                chainlink_current = self.rtds_manager.get_chainlink_price()
                if chainlink_current and chainlink_current > 0:
                    btc_price = chainlink_current
                    price_source = "Chainlink (current)"
                    logger.info(f"Using current Chainlink price: ${btc_price:,.2f}")
            
            # 3. Fallback to any RTDS price
            if not btc_price:
                rtds_price = self.rtds_manager.get_current_price()
                if rtds_price and rtds_price > 0:
                    btc_price = rtds_price
                    price_source = "RTDS (Binance)"
                    logger.warning(f"Chainlink unavailable, using RTDS: ${btc_price:,.2f}")
            
            # 4. Last resort: history
            if not btc_price:
                btc_history = await self.history_manager.get_btc_history()
                btc_price = btc_history[-1] if btc_history else 0.0
                price_source = "History (fallback)"
                logger.warning(f"Using history fallback price: ${btc_price:,.2f}")
            
            if btc_price <= 0:
                logger.error("Invalid BTC price for resolution")
                return
            
            # Get and validate strike
            strike_str = str(market.get('strike_price', '')).replace(',', '').replace('$', '').strip()
            strike_val = self._parse_strike(strike_str)
            strike_source = market.get('_strike_source', 'API')
            
            # Log strike details for debugging
            logger.info(f"=== STRIKE VALIDATION FOR RESOLUTION ===")
            logger.info(f"  Raw strike from market: {market.get('strike_price')}")
            logger.info(f"  Cleaned strike string: {strike_str}")
            logger.info(f"  Parsed strike value: {strike_val}")
            logger.info(f"  Strike source: {strike_source}")

            if strike_val is None:
                logger.error(f"!!! STRIKE UNAVAILABLE FOR RESOLUTION !!!")
                logger.error(f"  Raw strike: {market.get('strike_price')}")
                logger.error(f"  Cleaned: {strike_str}")
                logger.error(f"  This will cause incorrect resolution calculation!")
                logger.warning("Strike unavailable or unparseable (%s); resolution deferred", strike_str)
                self._emit('resolution', None)
                return

            # Calculate resolution
            # Polymarket: YES wins if BTC >= Strike at market end
            resolution = "YES" if btc_price >= strike_val else "NO"
            
            # Log detailed resolution info for debugging
            diff = btc_price - strike_val
            diff_bps = (diff / strike_val) * 10000 if strike_val > 0 else 0
            
            logger.info(f"=== RESOLUTION DETAILS ===")
            logger.info(f"  BTC Price: ${btc_price:,.2f} (source: {price_source})")
            logger.info(f"  Strike:    ${strike_val:,.2f} (source: {strike_source})")
            logger.info(f"  Diff:      ${diff:+,.2f} ({diff_bps:+.0f} bps)")
            logger.info(f"  Result:    {resolution} ({'BTC >= Strike' if resolution == 'YES' else 'BTC < Strike'})")
            logger.info(f"===========================")
            
            # Query Polymarket's actual resolution for comparison
            market_id = market.get('market_id')
            polymarket_resolution = None
            if market_id:
                try:
                    pm_result = await self.connector.get_market_resolution(market_id)
                    if pm_result and pm_result.get('resolved'):
                        polymarket_resolution = pm_result.get('resolution') or pm_result.get('winner')
                        pm_strike = pm_result.get('strike')
                        pm_price = pm_result.get('resolution_price')
                        
                        logger.info(f"Polymarket API resolution: {polymarket_resolution}")
                        if pm_strike:
                            logger.info(f"Polymarket API strike: {pm_strike}")
                        if pm_price:
                            logger.info(f"Polymarket resolution price: {pm_price}")
                        
                        # Check for discrepancy
                        if polymarket_resolution and polymarket_resolution.upper() != resolution:
                            logger.error(f"!!! RESOLUTION MISMATCH !!!")
                            logger.error(f"  Our calculation: {resolution}")
                            logger.error(f"  Polymarket says: {polymarket_resolution}")
                            logger.error(f"  Our BTC: ${btc_price:,.2f}, Our Strike: ${strike_val:,.2f}")
                            if pm_strike:
                                logger.error(f"  PM Strike: {pm_strike}")
                            self.log_msg(f"⚠️ RESOLUTION MISMATCH: We said {resolution}, PM says {polymarket_resolution}")
                            # Use Polymarket's resolution since they're authoritative
                            resolution = polymarket_resolution.upper()
                except Exception as e:
                    logger.debug(f"Could not query Polymarket resolution: {e}")
            
            self.last_resolution = resolution
            direction = "UP" if resolution == "YES" else "DOWN"
            diff = btc_price - strike_val
            self.log_msg(
                f"Market Resolved: {resolution} "
                f"(BTC: ${btc_price:,.2f} vs Strike: ${strike_val:,.2f}, diff: ${diff:+,.2f})"
            )
            self.log_msg(f"Market resolved {direction} | Price source: {price_source}")
            self._emit('resolution', resolution)
        except Exception as e:
            logger.error(f"Error showing resolution: {e}", exc_info=True)
    
    async def _reset_market_after_resolution(self) -> None:
        """Reset market state after resolution with proper cleanup."""
        try:
            # Prepend the just-resolved outcome to displayed prior outcomes
            # This is more reliable than waiting for the API to update
            if self.last_resolution:
                # Insert at beginning (most recent first for display)
                self.displayed_prior_outcomes.insert(0, self.last_resolution)
                
                # Trim to max length
                max_outcomes = self.config.max_prior_outcomes
                if len(self.displayed_prior_outcomes) > max_outcomes:
                    self.displayed_prior_outcomes = self.displayed_prior_outcomes[:max_outcomes]
                
                # Update display immediately
                self._update_prior_outcomes_display()
                
                # Log the update (oldest to newest, left to right)
                arrows = ''.join(['▲' if o == 'YES' else '▼' for o in reversed(self.displayed_prior_outcomes)])
                logger.info(f"Added {self.last_resolution} to prior outcomes: {arrows}")
            
            self.last_resolution = None
            
            self.log_msg("Clearing expired market...")
            await self.market_manager.clear_market()
            await self.history_manager.clear_yes_history()
            await self.ws_manager.stop()
            # Invalidate price cache on market reset
            self._invalidate_price_cache()
            # Reset analytics for new market
            self.analytics_engine.reset()
            # Don't invalidate prior outcomes cache - we've already added the latest outcome
            # The cache will naturally expire or be refreshed on next API call
            # Note: Keep RTDS running for next market
            self.resolution_shown = False
            self._emit('resolution', None)  # Hide overlay
        except Exception as e:
            logger.error(f"Error resetting market: {e}", exc_info=True)
    
    async def _check_and_add_prior_outcomes_with_timeout(self, retry_count: int = 0) -> None:
        """Wrapper for _check_and_add_prior_outcomes with timeout to prevent hanging.
        
        Args:
            retry_count: Number of retries if no matching outcomes found
        """
        try:
            print("[CORE] Checking prior outcomes (with 15s timeout)...", flush=True)
            await asyncio.wait_for(
                self._check_and_add_prior_outcomes(retry_count=retry_count),
                timeout=15.0
            )
            print("[CORE] Prior outcomes checked", flush=True)
        except asyncio.TimeoutError:
            logger.warning("Prior outcomes check timed out after 15s, continuing without blocking")
            print("[CORE] Prior outcomes check timed out, continuing...", flush=True)
            # Don't clear existing outcomes on timeout
        except Exception as e:
            logger.error(f"Error in prior outcomes check: {e}", exc_info=True)
            print(f"[CORE] Prior outcomes check error: {e}", flush=True)
    
    async def _check_and_add_prior_outcomes(self, retry_count: int = 0) -> None:
        """Check and fetch prior outcomes from Polymarket API.
        
        Fetches closed markets from Polymarket and validates that they align
        with the current market's time window. Only displays outcomes that
        are consecutive and properly timed relative to the current market.
        
        Args:
            retry_count: Number of retries if no matching outcomes found (for API propagation delay)
        """
        try:
            market = await self.market_manager.get_market()
            if not market:
                logger.debug("No market found, clearing prior outcomes")
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            market_start_time = await self.market_manager.get_market_start_time()
            if not market_start_time:
                logger.debug("No market start time, clearing prior outcomes")
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            logger.info(f"Checking prior outcomes for market starting at {market_start_time}")
            
            # Fetch prior outcomes from API (cached)
            prior_outcomes_data = await self._get_prior_outcomes()
            
            if not prior_outcomes_data:
                logger.warning("No prior outcomes data returned from API")
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            logger.info(f"Fetched {len(prior_outcomes_data)} prior outcomes from API")
            
            # The previous market should have ended when the current market started
            # (markets are consecutive: previous ends at T, current starts at T)
            expected_previous_market_end = market_start_time
            
            # Tolerance for time matching (configurable)
            tolerance_seconds = self.config.prior_outcome_tolerance_seconds
            
            logger.info(
                f"Expected previous market end: {expected_previous_market_end}, "
                f"tolerance: ±{tolerance_seconds}s"
            )
            
            # Filter outcomes to only include consecutive ones that match time windows
            consecutive_outcomes: List[str] = []
            expected_end_time = expected_previous_market_end
            
            # Iterate through prior outcomes from API
            for idx, outcome_entry in enumerate(prior_outcomes_data):
                if not isinstance(outcome_entry, dict):
                    continue
                
                # Get outcome value
                outcome_value = outcome_entry.get('outcome', '')
                if not isinstance(outcome_value, str) or not outcome_value:
                    continue
                
                # Get and validate timestamp
                outcome_timestamp = outcome_entry.get('end_timestamp')
                if not outcome_timestamp:
                    # Try parsing from end_date string
                    end_date_str = outcome_entry.get('end_date')
                    if not end_date_str:
                        break
                    
                    try:
                        outcome_timestamp = pd.Timestamp(end_date_str)
                        if outcome_timestamp.tz is None:
                            outcome_timestamp = outcome_timestamp.tz_localize('UTC')
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing timestamp: {e}")
                        break
                
                # Check if this outcome's time matches expected time window
                time_diff = abs((outcome_timestamp - expected_end_time).total_seconds())
                
                logger.debug(
                    f"Outcome {idx+1}: {outcome_value} at {outcome_timestamp}, "
                    f"expected {expected_end_time}, diff: {time_diff:.1f}s"
                )
                
                if time_diff <= tolerance_seconds:
                    # This outcome is part of the consecutive chain
                    consecutive_outcomes.append(outcome_value)
                    logger.debug(f"  ✓ Added to chain (diff: {time_diff:.1f}s)")
                    
                    # Update expected time for next iteration (going backwards)
                    expected_end_time = outcome_timestamp - pd.Timedelta(
                        minutes=self.config.market_duration_minutes
                    )
                else:
                    # Time window doesn't match - stop here
                    logger.info(
                        f"  ✗ Stopping chain at outcome {idx+1} "
                        f"(time diff {time_diff:.1f}s exceeds tolerance {tolerance_seconds}s)"
                    )
                    break
            
            if consecutive_outcomes:
                # Update displayed outcomes with API data
                self.displayed_prior_outcomes = consecutive_outcomes
                self._update_prior_outcomes_display()
                
                # Build arrow string for log message (oldest to newest, left to right)
                arrows = ''.join(['▲' if o == 'YES' else '▼' for o in reversed(consecutive_outcomes)])
                self.log_msg(
                    f"Prior outcomes loaded: {arrows} ({len(consecutive_outcomes)} markets)"
                )
            else:
                # No matches found - might be API propagation delay after market close
                # Keep existing displayed_prior_outcomes (don't clear them!)
                
                if retry_count > 0 and prior_outcomes_data:
                    # Log what we're seeing for debugging
                    first_end = prior_outcomes_data[0].get('end_timestamp')
                    logger.info(
                        f"API's most recent closed market ended at {first_end}, "
                        f"but we expect {expected_previous_market_end}. "
                        f"Retrying in 3s... ({retry_count} retries left)"
                    )
                    
                    # Wait and retry - API might not have updated yet
                    await asyncio.sleep(3.0)
                    # Invalidate cache to force fresh fetch
                    self._prior_outcomes_cache = []
                    self._prior_outcomes_cache_timestamp = 0.0
                    # Use timeout wrapper for recursive call to prevent hanging
                    try:
                        await asyncio.wait_for(
                            self._check_and_add_prior_outcomes(retry_count=retry_count - 1),
                            timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Prior outcomes retry timed out, stopping retries")
                    return
                
                # If we still have displayed outcomes (from resolution), keep them
                if self.displayed_prior_outcomes:
                    logger.info(
                        f"API not yet updated, keeping existing {len(self.displayed_prior_outcomes)} prior outcomes"
                    )
                elif prior_outcomes_data:
                    first_end = prior_outcomes_data[0].get('end_timestamp')
                    self.log_msg(
                        f"No consecutive prior outcomes matched time window "
                        f"(API latest: {first_end}, expected: {expected_previous_market_end})"
                    )
                else:
                    self.log_msg("No prior outcomes available from API")
        
        except Exception as e:
            logger.error(f"Error in _check_and_add_prior_outcomes: {e}", exc_info=True)
            self.displayed_prior_outcomes = []
            self._update_prior_outcomes_display()
            # Log the error to user
            self.log_msg(f"Error fetching prior outcomes: {e}")
    
    async def _get_prior_outcomes(self) -> List[Dict[str, Any]]:
        """Get prior outcomes from Polymarket API with caching.
        
        Fetches closed markets from Polymarket's Gamma API and caches results
        for a short period to avoid excessive API calls.
        
        Returns:
            List of closed market outcome dictionaries
        """
        now = time.time()
        
        # Return cached data if recent
        if (self._prior_outcomes_cache and 
            now - self._prior_outcomes_cache_timestamp < self._prior_outcomes_cache_ttl):
            return self._prior_outcomes_cache
        
        # Fetch fresh data from API
        try:
            closed_markets = await asyncio.wait_for(
                self.connector.get_closed_markets(
                    "10192",  # series_id: BTC 15m series
                    20,  # limit: Get last 20 markets
                    False  # ascending: Most recent first
                ),
                timeout=8.0  # Reduced timeout to prevent hanging
            )
            
            if closed_markets:
                # Update cache
                self._prior_outcomes_cache = closed_markets
                self._prior_outcomes_cache_timestamp = now
                logger.info(f"Fetched {len(closed_markets)} closed markets from API")
                # Log sample for debugging
                if closed_markets:
                    sample = closed_markets[0]
                    logger.debug(
                        f"Sample market: outcome={sample.get('outcome')}, "
                        f"end_date={sample.get('end_date')}"
                    )
            else:
                logger.warning("No closed markets returned from API")
            
            return closed_markets
        
        except Exception as e:
            logger.error(f"Error fetching prior outcomes from API: {e}", exc_info=True)
            # Return cached data if available, even if stale
            return self._prior_outcomes_cache if self._prior_outcomes_cache else []
    
    def _update_prior_outcomes_display(self) -> None:
        """Update prior outcomes display by emitting to UI."""
        # Simply emit the displayed outcomes list
        self._emit('prior_outcomes_update', self.displayed_prior_outcomes)
    
    def _format_ends(self, end_str: str) -> str:
        """Format end date to PST with improved error handling.
        
        Args:
            end_str: End date string
            
        Returns:
            Formatted date string
        """
        try:
            dt = pd.Timestamp(end_str)
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            dt_local = dt.tz_convert('US/Pacific')
            return dt_local.strftime("%H:%M:%S PST")
        except Exception as e:
            logger.debug(f"Error formatting end date: {e}")
            return str(end_str)
    
    def _parse_strike(self, strike_str: str) -> Optional[float]:
        """Parse strike price string to float.
        
        Args:
            strike_str: Strike price string
            
        Returns:
            Parsed float or None if invalid
        """
        try:
            clean_strike = strike_str.replace(',', '').replace('$', '').strip()
            if clean_strike and clean_strike != "N/A":
                return float(clean_strike)
        except (ValueError, AttributeError):
            pass
        return None
    
    def size_up(self) -> None:
        """Increase order size with validation."""
        new_size = self.selected_size + self.config.size_increment
        # Add maximum size validation
        max_size = 1000.0  # Configurable maximum
        if new_size <= max_size:
            self.selected_size = new_size
            self.log_msg(f"Size increased to ${self.selected_size:.2f}")
        else:
            self.log_msg(f"Size limit reached: ${max_size:.2f}")
    
    def size_down(self) -> None:
        """Decrease order size with validation."""
        new_size = self.selected_size - self.config.size_increment
        if new_size >= self.config.min_order_size:
            self.selected_size = new_size
            self.log_msg(f"Size decreased to ${self.selected_size:.2f}")
        else:
            self.log_msg(f"Minimum size: ${self.config.min_order_size:.2f}")
    
    async def place_order(self, side: str) -> None:
        """Place an order with improved validation and error handling.
        
        Args:
            side: Order side ('YES' or 'NO')
        """
        # Validate side
        if side.upper() not in ('YES', 'NO'):
            self.log_msg(f"Invalid order side: {side}")
            return
        
        # Validate size
        if self.selected_size < self.config.min_order_size:
            self.log_msg(
                f"Order size too small: ${self.selected_size:.2f} "
                f"(minimum: ${self.config.min_order_size:.2f})"
            )
            return
        
        try:
            size = self.selected_size
            self.log_msg(f"Order: BUY {side} (${size:.2f})")
            
            # Emit order submitted IMMEDIATELY with estimated price (for fast UI feedback)
            # Use cached prices if available, otherwise use fallback
            estimated_price = 0.5  # Default fallback
            if self._cached_prices is not None:
                # Use cached prices for immediate notification
                yes_price, no_price, best_bid, best_ask = self._cached_prices
                if side.upper() == 'YES':
                    estimated_price = best_ask if best_ask > 0 else yes_price
                else:  # NO
                    no_best_ask = 1.0 - best_bid if best_bid > 0 else no_price
                    estimated_price = no_best_ask
            
            # Emit order submitted callback IMMEDIATELY (before any async operations)
            self._emit('order_submitted', side, size, estimated_price)
            
            token_map = await self.market_manager.get_token_map()
            if not token_map:
                error_msg = "Error: Token map not available"
                self.log_msg(error_msg)
                self._emit('order_failed', side, size, error_msg)
                return
            
            # Get current market price before placing order (for average entry price calculation)
            # Use best ask for BUY orders (the price we'll pay)
            prices = await self.market_manager.calculate_mid_price()
            yes_price, no_price, best_bid, best_ask = prices
            # For BUY orders, use best ask (the price we pay)
            if side.upper() == 'YES':
                entry_price = best_ask if best_ask > 0 else yes_price
            else:  # NO
                # For NO, best ask in YES terms = 1 - best_bid in NO terms
                # Best ask for NO = 1 - best_bid (when buying NO, we pay 1 - best_bid in YES terms)
                no_best_ask = 1.0 - best_bid if best_bid > 0 else no_price
                entry_price = no_best_ask
            
            # Get current position balances before order
            y_id = token_map.get('YES')
            n_id = token_map.get('NO')
            old_yes_bal = 0.0
            old_no_bal = 0.0
            if y_id:
                old_yes_bal = self.connector.get_token_balance(y_id)
            if n_id:
                old_no_bal = self.connector.get_token_balance(n_id)
            
            resp = await self.order_executor.execute_order(side, size, token_map)
            
            if resp and resp.get('orderID'):
                order_id = resp.get('orderID', '')
                self.log_msg(f"Order FILLED: {order_id[:10]}...")
                
                # Emit order filled callback
                self._emit('order_filled', side, size, entry_price, order_id)
                
                # Update average entry price with proper validation
                self._update_average_entry_price(
                    side=side.upper(),
                    size=size,
                    entry_price=entry_price,
                    old_yes_bal=old_yes_bal,
                    old_no_bal=old_no_bal
                )
                
                await self.update_account_stats()
            else:
                error_msg = "Order execution failed - no order ID returned"
                self.log_msg("Order FAILED")
                self._emit('order_failed', side, size, error_msg)
        except Exception as e:
            error_msg = f"Execution Error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Order placement error: {e}", exc_info=True)
            self._emit('order_failed', side, self.selected_size, error_msg)
    
    def _update_average_entry_price(
        self,
        side: str,
        size: float,
        entry_price: float,
        old_yes_bal: float,
        old_no_bal: float
    ) -> None:
        """Update average entry price with proper validation.
        
        Args:
            side: Order side ('YES' or 'NO')
            size: Order size in dollars
            entry_price: Price paid per share
            old_yes_bal: YES balance before order
            old_no_bal: NO balance before order
        """
        # Validate entry price to prevent division by zero
        if entry_price <= 0:
            logger.warning(
                f"Invalid entry price {entry_price} for {side} order, "
                "skipping average calculation"
            )
            return
        
        # Calculate estimated shares received
        new_shares = size / entry_price
        if new_shares <= 0:
            logger.warning(f"Invalid shares calculation: {new_shares}")
            return
        
        if side == 'YES':
            total_shares = old_yes_bal + new_shares
            
            if total_shares <= 0:
                self.avg_entry_price_yes = None
                return
            
            if self.avg_entry_price_yes is not None and old_yes_bal > 0:
                # Weighted average: (old_cost + new_cost) / total_shares
                old_cost = self.avg_entry_price_yes * old_yes_bal
                new_cost = entry_price * new_shares
                self.avg_entry_price_yes = (old_cost + new_cost) / total_shares
            else:
                # First position or no previous position
                self.avg_entry_price_yes = entry_price
                
        elif side == 'NO':
            total_shares = old_no_bal + new_shares
            
            if total_shares <= 0:
                self.avg_entry_price_no = None
                return
            
            if self.avg_entry_price_no is not None and old_no_bal > 0:
                old_cost = self.avg_entry_price_no * old_no_bal
                new_cost = entry_price * new_shares
                self.avg_entry_price_no = (old_cost + new_cost) / total_shares
            else:
                self.avg_entry_price_no = entry_price
    
    async def flatten(self) -> None:
        """Flatten all positions with improved error handling."""
        self.log_msg("Action: FLATTEN")
        
        # Emit flatten started callback immediately
        self._emit('flatten_started')
        
        token_map = await self.market_manager.get_token_map()
        if not token_map:
            error_msg = "Error: Token map not ready."
            self.log_msg(error_msg)
            self._emit('flatten_failed', error_msg)
            return
        
        try:
            results = await self.order_executor.flatten_positions(token_map)
            orders_processed = len(results) if results else 0
            
            if results:
                self.log_msg(f"Flatten completed. {orders_processed} orders processed.")
            else:
                self.log_msg("Flatten completed. No orders to process.")
            
            # Reset all position tracking after flattening
            self.position_tracker.reset()
            
            # Emit flatten completed callback
            self._emit('flatten_completed', orders_processed)
        except Exception as e:
            error_msg = f"Flatten error: {e}"
            self.log_msg(error_msg)
            logger.error(f"Flatten error: {e}", exc_info=True)
            self._emit('flatten_failed', error_msg)
        
        await self.update_account_stats()
    
    async def close_position(self, side: str) -> None:
        """Close a single position for one side (YES or NO).
        
        Args:
            side: Side to close ('YES' or 'NO')
        """
        side = side.upper()
        if side not in ('YES', 'NO'):
            self.log_msg(f"Invalid side for close position: {side}")
            return
        
        self.log_msg(f"Closing {side} position...")
        
        token_map = await self.market_manager.get_token_map()
        if not token_map:
            error_msg = "Error: Token map not ready."
            self.log_msg(error_msg)
            return
        
        token_id = token_map.get(side)
        if not token_id:
            error_msg = f"Error: Token ID not found for {side}"
            self.log_msg(error_msg)
            return
        
        try:
            # Get current balance
            balance = await self.connector.get_token_balance(token_id)

            if balance <= 0.1:  # MIN_BALANCE_THRESHOLD
                self.log_msg(f"No {side} position to close (balance: {balance:.2f})")
                return

            # Create SELL order to close position
            resp = await self.connector.create_market_order(token_id, balance, 'SELL')
            
            if resp and resp.get('orderID'):
                order_id = resp.get('orderID', '')
                self.log_msg(f"{side} position closed: {order_id[:10]}...")
                # Reset position tracking for this side
                self.position_tracker.reset(side)
                await self.update_account_stats()
            else:
                error_msg = f"Failed to close {side} position"
                self.log_msg(error_msg)
        except Exception as e:
            error_msg = f"Error closing {side} position: {e}"
            self.log_msg(error_msg)
            logger.error(error_msg, exc_info=True)
    
    async def cancel_all(self) -> None:
        """Cancel all pending orders with improved error handling."""
        self.log_msg("Action: CANCEL ALL")
        try:
            success = await self.order_executor.cancel_all_orders()
            if success:
                self.log_msg("All orders cancelled.")
            else:
                self.log_msg("Cancel operation failed.")
        except Exception as e:
            self.log_msg(f"Cancel error: {e}")
            logger.error(f"Cancel all error: {e}", exc_info=True)
    
    async def shutdown(self) -> None:
        """Handle graceful shutdown with proper cleanup.
        
        Cleans up all resources including callbacks and connections.
        """
        self.log_msg("Initiating graceful shutdown...")
        
        try:
            # Stop WebSocket
            await self.ws_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {e}")
        
        try:
            # Stop RTDS
            await self.rtds_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping RTDS: {e}")
        
        try:
            # Clear all callbacks
            self.callback_manager.clear()
        except Exception as e:
            logger.error(f"Error clearing callbacks: {e}")
        
        # Clear cache
        self._cached_prices = None
        
        self.log_msg("Shutdown complete.")

