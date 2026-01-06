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
from src.activetrader.strategy_data_sync import StrategyDataProvider, set_provider

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

    async def _on_rtds_btc_price(self, btc_price: float) -> None:
        self._emit('btc_price_update', btc_price)

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

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        try:
            if self.ws_manager:
                await self.ws_manager.stop()
            if self.rtds_manager:
                await self.rtds_manager.stop()
            self.callback_manager.clear()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")