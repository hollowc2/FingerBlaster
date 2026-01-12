"""Core engine components for activetrader.

This module provides the core infrastructure and abstractions for:
- Market data management and order book state
- WebSocket connectivity and reconnection logic
- Historical price caching and RTDS history
- Order execution
- Integration with configuration and async workflows

Classes in this file are designed to be imported and orchestrated by the main controller in src/activetrader/core.py.
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional, Dict, List, Tuple, Any, Callable, Awaitable, Union

import pandas as pd
from websockets.asyncio.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidURI, InvalidState

from src.activetrader.config import AppConfig

logger = logging.getLogger("FingerBlaster")

# Constants
MAX_WEBSOCKET_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB limit for security
DEFAULT_ORDER_BOOK_PRICE = 0.5  # Default mid price when order book is empty


class MarketDataManager:
    """Manages market data and order book state with optimizations.

    Improvements:
    - Better validation
    - Optimized price calculations
    - Thread-safe operations
    - Stale data detection
    """

    def __init__(self, config: AppConfig):
        """Initialize market data manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.lock = asyncio.Lock()
        self.current_market: Optional[Dict[str, Any]] = None
        self.token_map: Dict[str, str] = {}
        # Use dict for O(1) lookups instead of nested dicts
        self.raw_books: Dict[str, Dict[str, Dict[float, float]]] = {
            'Up': {'bids': {}, 'asks': {}},
            'Down': {'bids': {}, 'asks': {}}
        }
        self.market_start_time: Optional[pd.Timestamp] = None
        # Cache for validation results
        self._validation_cache: Optional[bool] = None
        self._cached_market_id: Optional[str] = None

        # Stale data detection
        self._last_update_time: float = 0.0
        self._update_counter: int = 0
        self._stale_threshold_seconds: float = 30.0  # Alert if no updates for 30s
    
    async def set_market(self, market: Dict[str, Any]) -> bool:
        """Set the current market with validation and caching.
        
        Args:
            market: Market data dictionary
            
        Returns:
            True if market was set successfully, False otherwise
        """
        # Check cache first
        market_id = market.get('market_id')
        if (market_id == self._cached_market_id and 
            self._validation_cache is not None):
            if not self._validation_cache:
                return False
        else:
            # Validate and cache result
            if not self._validate_market(market):
                self._validation_cache = False
                self._cached_market_id = market_id
                return False
            self._validation_cache = True
            self._cached_market_id = market_id
        
        async with self.lock:
            self.current_market = market
            self.token_map = market.get('token_map', {}).copy()

            # Get market start time from API (more accurate than calculating from end_date)
            start_date_str = market.get('start_date')
            if start_date_str:
                # Use actual start time from Polymarket API
                start_dt = pd.Timestamp(start_date_str)
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize('UTC')
                
                # Check if this start date makes sense for a short-term market
                # For 15m markets, start should be ~15m before end.
                # If it's much longer (e.g. > 20 mins), it's likely the Series start time, NOT the market start.
                end_dt = pd.Timestamp(market.get('end_date'))
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')
                
                duration_mins = (end_dt - start_dt).total_seconds() / 60.0
                
                if duration_mins > 20:
                    logger.info(f"API start_date yields {duration_mins:.1f}m duration (expected ~15m). Recalculating start time from end date.")
                    self.market_start_time = end_dt - pd.Timedelta(minutes=self.config.market_duration_minutes)
                else:
                    self.market_start_time = start_dt
                
                logger.debug(f"Final market start time: {self.market_start_time}")
            else:
                # Fallback: Calculate from end_date (old behavior)
                end_dt = pd.Timestamp(market.get('end_date'))
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')
                self.market_start_time = end_dt - pd.Timedelta(
                    minutes=self.config.market_duration_minutes
                )
                logger.debug(f"Calculated market start from end_date: {self.market_start_time}")
        
        return True
    
    async def clear_market(self) -> None:
        """Clear current market state."""
        async with self.lock:
            self.current_market = None
            self.token_map = {}
            self.market_start_time = None
            self.raw_books = {
                'Up': {'bids': {}, 'asks': {}},
                'Down': {'bids': {}, 'asks': {}}
            }
            # Clear validation cache
            self._validation_cache = None
            self._cached_market_id = None
    
    async def update_order_book(
        self,
        token_type: str,
        bids: Dict[float, float],
        asks: Dict[float, float]
    ) -> None:
        """Update order book for a token type.

        Args:
            token_type: Token type ('Up' or 'Down')
            bids: Dictionary of bid prices to sizes
            asks: Dictionary of ask prices to sizes
        """
        if token_type not in self.raw_books:
            logger.warning(f"Unknown token type: {token_type}")
            return

        async with self.lock:
            self.raw_books[token_type]['bids'] = bids.copy()
            self.raw_books[token_type]['asks'] = asks.copy()
            # Track update timing
            self._last_update_time = time.time()
            self._update_counter += 1
    
    async def apply_price_changes(
        self,
        token_type: str,
        changes: List[Dict[str, Any]]
    ) -> None:
        """Apply incremental price changes to order book.

        Optimized to batch updates.

        Args:
            token_type: Token type ('Up' or 'Down')
            changes: List of price change dictionaries
        """
        if token_type not in self.raw_books:
            return

        async with self.lock:
            target_book = self.raw_books[token_type]

            # Batch updates for better performance
            bid_updates = {}
            ask_updates = {}

            for change in changes:
                if not isinstance(change, dict):
                    continue

                try:
                    price = float(change.get('price', 0))
                    size = float(change.get('size', 0))
                    side = str(change.get('side', '')).upper()

                    if side not in ('BUY', 'SELL'):
                        continue

                    if side == 'BUY':
                        if size <= 0:
                            bid_updates[price] = None  # Mark for deletion
                        else:
                            bid_updates[price] = size
                    else:  # SELL
                        if size <= 0:
                            ask_updates[price] = None  # Mark for deletion
                        else:
                            ask_updates[price] = size
                except (ValueError, KeyError, TypeError):
                    continue

            # Apply batch updates
            for price, size in bid_updates.items():
                if size is None:
                    target_book['bids'].pop(price, None)
                else:
                    target_book['bids'][price] = size

            for price, size in ask_updates.items():
                if size is None:
                    target_book['asks'].pop(price, None)
                else:
                    target_book['asks'][price] = size

            # Track update timing
            self._last_update_time = time.time()
            self._update_counter += 1
    
    async def calculate_mid_price(self) -> Tuple[float, float, float, float]:
        """Calculate mid price from order books with improved edge case handling.
        
        Returns:
            Tuple of (yes_price, no_price, best_bid, best_ask)
        """
        async with self.lock:
            raw = self.raw_books
            up_bids = raw['Up']['bids']
            up_asks = raw['Up']['asks']
            down_bids = raw['Down']['bids']
            down_asks = raw['Down']['asks']
        
        # Convert Down prices to Up prices (optimized)
        combined_bids = dict(up_bids)
        combined_asks = dict(up_asks)
        
        # Batch conversion for better performance
        for p_down, s_down in down_asks.items():
            if s_down > 0:
                p_up = round(1.0 - p_down, 4)
                combined_bids[p_up] = combined_bids.get(p_up, 0.0) + s_down
        
        for p_down, s_down in down_bids.items():
            if s_down > 0:
                p_up = round(1.0 - p_down, 4)
                combined_asks[p_up] = combined_asks.get(p_up, 0.0) + s_down
        
        # Get best bid and ask
        bids_sorted = sorted(combined_bids.keys(), reverse=True) if combined_bids else []
        asks_sorted = sorted(combined_asks.keys()) if combined_asks else []
        
        best_bid = bids_sorted[0] if bids_sorted else 0.0
        best_ask = asks_sorted[0] if asks_sorted else 1.0
        
        # Calculate mid price with improved edge case handling
        if best_bid > 0 and best_ask < 1.0:
            mid = (best_bid + best_ask) / 2.0
        elif best_ask < 1.0:
            mid = best_ask
        elif best_bid > 0:
            mid = best_bid
        else:
            # Fallback when order book is empty
            mid = DEFAULT_ORDER_BOOK_PRICE
            logger.warning("Empty order book, using default mid price")
        
        return mid, 1.0 - mid, best_bid, best_ask
    
    def _validate_market(self, market: Dict[str, Any]) -> bool:
        """Validate market data structure.
        
        Args:
            market: Market data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['market_id', 'end_date', 'token_map']
        if not all(key in market for key in required_keys):
            logger.warning("Invalid market data: missing required keys")
            return False
        
        token_map = market.get('token_map', {})
        if not isinstance(token_map, dict):
            logger.warning("Invalid token_map: not a dictionary")
            return False
        
        if 'Up' not in token_map or 'Down' not in token_map:
            logger.warning("Invalid token_map: missing Up or Down")
            return False
        
        return True
    
    async def get_market(self) -> Optional[Dict[str, Any]]:
        """Get current market (thread-safe).
        
        Returns:
            Market dictionary or None
        """
        async with self.lock:
            return self.current_market.copy() if self.current_market else None
    
    async def get_token_map(self) -> Dict[str, str]:
        """Get token map (thread-safe copy).
        
        Returns:
            Copy of token map dictionary
        """
        async with self.lock:
            return self.token_map.copy()
    
    async def get_market_start_time(self) -> Optional[pd.Timestamp]:
        """Get market start time (thread-safe).
        
        Returns:
            Market start timestamp or None
        """
        async with self.lock:
            return self.market_start_time
    
    async def get_raw_order_book(self) -> Dict[str, Dict[str, Dict[float, float]]]:
        """Get raw order book data for analytics (thread-safe copy).

        Returns:
            Copy of raw order book: {Up/Down: {bids/asks: {price: size}}}
        """
        async with self.lock:
            return {
                'Up': {
                    'bids': dict(self.raw_books['Up']['bids']),
                    'asks': dict(self.raw_books['Up']['asks'])
                },
                'Down': {
                    'bids': dict(self.raw_books['Down']['bids']),
                    'asks': dict(self.raw_books['Down']['asks'])
                }
            }

    async def is_data_stale(self) -> bool:
        """Check if order book data is stale (no updates recently).

        Returns:
            True if data hasn't been updated within threshold
        """
        async with self.lock:
            if self._last_update_time == 0.0:
                # No updates yet
                return True
            now = time.time()
            time_since_update = now - self._last_update_time
            return time_since_update > self._stale_threshold_seconds

    async def get_data_freshness_info(self) -> Dict[str, Any]:
        """Get diagnostic information about data freshness.

        Returns:
            Dictionary with freshness metrics
        """
        async with self.lock:
            now = time.time()
            time_since_update = now - self._last_update_time if self._last_update_time > 0 else -1
            return {
                'last_update_time': self._last_update_time,
                'seconds_since_update': time_since_update,
                'update_counter': self._update_counter,
                'is_stale': time_since_update > self._stale_threshold_seconds if time_since_update >= 0 else True,
                'stale_threshold': self._stale_threshold_seconds
            }


class HistoryManager:
    """Manages price history with efficient data structures.
    
    Improvements:
    - Use deque for O(1) append operations
    - Thread-safe operations
    - Memory-efficient
    """
    
    def __init__(self, config: AppConfig):
        """Initialize history manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.lock = asyncio.Lock()
        # Use deque with maxlen for O(1) append and automatic size limiting
        self.yes_history: deque = deque(maxlen=config.max_history_size)
        self.btc_history: deque = deque(maxlen=config.max_btc_history_size)
    
    async def add_price_point(
        self, 
        elapsed_seconds: float, 
        price: float, 
        market_start_time: Optional[pd.Timestamp]
    ) -> None:
        """Add a price point to history if within market duration.
        
        Args:
            elapsed_seconds: Elapsed seconds since market start
            price: Price value
            market_start_time: Market start timestamp
        """
        if market_start_time is None:
            return
        
        # Validate elapsed time
        if 0 <= elapsed_seconds <= self.config.market_duration_seconds:
            async with self.lock:
                self.yes_history.append((elapsed_seconds, price))
    
    async def add_btc_price(self, price: float) -> None:
        """Add BTC price to history.
        
        Args:
            price: BTC price value
        """
        if price > 0:  # Validate price
            async with self.lock:
                self.btc_history.append(price)
    
    async def get_yes_history(self) -> List[Tuple[float, float]]:
        """Get Up price history (thread-safe copy).
        
        Returns:
            List of (elapsed_seconds, price) tuples
        """
        async with self.lock:
            return list(self.yes_history)
    
    async def get_btc_history(self) -> List[float]:
        """Get BTC price history (thread-safe copy).
        
        Returns:
            List of BTC prices
        """
        async with self.lock:
            return list(self.btc_history)
    
    async def clear_yes_history(self) -> None:
        """Clear Up price history."""
        async with self.lock:
            self.yes_history.clear()


class WebSocketManager:
    """Manages WebSocket connection with improved reconnection logic.
    
    Improvements:
    - Better error handling
    - Message size validation
    - Improved reconnection strategy
    - Market change detection
    """
    
    def __init__(
        self, 
        config: AppConfig, 
        market_manager: MarketDataManager,
        on_message: Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
    ):
        """Initialize WebSocket manager.
        
        Args:
            config: Application configuration
            market_manager: Market data manager instance
            on_message: Callback function for messages
        """
        self.config = config
        self.market_manager = market_manager
        self.on_message = on_message
        self.shutdown_flag = asyncio.Event()
        self.connection_task: Optional[asyncio.Task] = None
        self._ws: Optional[ClientConnection] = None
    
    async def start(self) -> None:
        """Start WebSocket connection."""
        if self.connection_task and not self.connection_task.done():
            logger.debug("WebSocket already running")
            return
        
        self.shutdown_flag.clear()
        self.connection_task = asyncio.create_task(self._connect_loop())
    
    async def stop(self) -> None:
        """Stop WebSocket connection with proper cleanup."""
        self.shutdown_flag.set()
        
        # Close WebSocket if open
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None
        
        # Wait for connection task to finish
        if self.connection_task:
            try:
                await asyncio.wait_for(
                    self.connection_task, 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("WebSocket stop timeout, cancelling task")
                self.connection_task.cancel()
                try:
                    await self.connection_task
                except asyncio.CancelledError:
                    pass
            finally:
                self.connection_task = None

    async def subscribe_to_market(self, market: Dict[str, Any]) -> None:
        """Subscribe to a specific market.
        
        This ensures the WebSocket is started and will connect to the 
        current market in the manager.
        
        Args:
            market: Market data dictionary (unused here, as we poll manager)
        """
        logger.info(f"Subscribing WebSocket to market: {market.get('market_id')}")
        await self.start()
    
    async def _connect_loop(self) -> None:
        """Main connection loop with automatic reconnection and market change detection."""
        reconnect_attempts = 0
        last_successful_connection = 0.0
        connection_storm_threshold = 5  # Consider it a storm if 5+ failures in 30 seconds

        while (not self.shutdown_flag.is_set() and
               reconnect_attempts < self.config.ws_max_reconnect_attempts):

            # Get current market
            market = await self.market_manager.get_market()
            if not market:
                await asyncio.sleep(1)
                continue

            token_map = await self.market_manager.get_token_map()
            subscribe_ids = list(token_map.values())
            if not subscribe_ids:
                await asyncio.sleep(1)
                continue

            subscribed_market_id = market.get('market_id')

            try:
                # Detect connection storm (rapid reconnections)
                now = time.time()
                if last_successful_connection > 0 and (now - last_successful_connection) < 30:
                    if reconnect_attempts >= connection_storm_threshold:
                        logger.error(
                            f"WebSocket connection storm detected: {reconnect_attempts} failures in "
                            f"{now - last_successful_connection:.1f}s. Backing off 30s..."
                        )
                        await asyncio.sleep(30)
                        reconnect_attempts = 0

                async with connect(
                    self.config.ws_uri,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout,
                    max_size=MAX_WEBSOCKET_MESSAGE_SIZE  # Security: limit message size
                ) as ws:
                    self._ws = ws

                    # Subscribe to market
                    msg = {"assets_ids": subscribe_ids, "type": "market"}
                    await ws.send(json.dumps(msg))
                    reconnect_attempts = 0
                    last_successful_connection = time.time()

                    logger.info(f"WebSocket connected to market {subscribed_market_id}")
                    
                    while (not self.shutdown_flag.is_set()):
                        # Re-check market in case it changed
                        current_market = await self.market_manager.get_market()
                        if (not current_market or 
                            current_market.get('market_id') != subscribed_market_id):
                            logger.info("Market changed, reconnecting...")
                            break
                        
                        try:
                            message = await asyncio.wait_for(
                                ws.recv(), 
                                timeout=self.config.ws_recv_timeout
                            )
                            
                            # Validate message size
                            if len(message) > MAX_WEBSOCKET_MESSAGE_SIZE:
                                logger.warning(
                                    f"Message too large: {len(message)} bytes, "
                                    f"max: {MAX_WEBSOCKET_MESSAGE_SIZE}"
                                )
                                continue
                            
                            data = json.loads(message)
                            
                            # Check for error messages from server
                            if isinstance(data, dict):
                                error_msg = data.get('error') or data.get('message', '')
                                if 'authentication' in str(error_msg).lower() or 'auth' in str(error_msg).lower():
                                    logger.error(
                                        f"Polymarket WebSocket authentication failure: {error_msg} - "
                                        f"This may indicate invalid API credentials or connection issues. "
                                        f"Check your PRIVATE_KEY and Polymarket API credentials in .env"
                                    )
                                    # Don't break - let reconnection handle it
                            
                            if isinstance(data, list):
                                for item in data:
                                    await self._process_message(item)
                            else:
                                await self._process_message(data)
                                
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            continue
                        except ConnectionClosed as e:
                            logger.warning(f"WebSocket connection closed: {e}")
                            break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON received: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error in message loop: {e}", exc_info=True)
                            continue
                        
            except (InvalidURI, InvalidState) as e:
                logger.error(f"WebSocket connection error: {e}")
                break
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts < self.config.ws_max_reconnect_attempts:
                    wait_time = self.config.ws_reconnect_delay * (
                        2 ** min(reconnect_attempts - 1, 3)
                    )
                    logger.info(
                        f"WebSocket error: {e}. "
                        f"Reconnecting in {wait_time}s "
                        f"(attempt {reconnect_attempts}/{self.config.ws_max_reconnect_attempts})..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"WebSocket failed after "
                        f"{self.config.ws_max_reconnect_attempts} attempts"
                    )
                    break
            finally:
                self._ws = None
    
    async def _process_message(self, item: Dict[str, Any]) -> None:
        """Process a WebSocket message with improved error handling.

        Args:
            item: Message item dictionary
        """
        if not isinstance(item, dict):
            logger.debug("WebSocket message is not a dict, skipping")
            return

        asset_id = item.get('asset_id')
        if not asset_id:
            logger.debug("WebSocket message missing asset_id, skipping")
            return

        token_map = await self.market_manager.get_token_map()
        if not token_map:
            logger.warning(f"WebSocket message received but token_map is empty. asset_id: {asset_id}")
            return

        token_type = None
        for outcome, tid in token_map.items():
            if tid == asset_id:
                token_type = outcome
                break

        if not token_type or token_type not in ('Up', 'Down'):
            # This is expected occasionally (other markets on same feed)
            logger.debug(
                f"WebSocket message asset_id {asset_id} not found in token_map. "
                f"Expected tokens: {list(token_map.values())}"
            )
            return

        logger.debug(f"Processing WebSocket message for {token_type} (asset_id: {asset_id})")

        try:
            if 'bids' in item and 'asks' in item:
                # Full order book update
                bids = {
                    float(x['price']): float(x['size'])
                    for x in item['bids']
                    if isinstance(x, dict) and 'price' in x and 'size' in x
                }
                asks = {
                    float(x['price']): float(x['size'])
                    for x in item['asks']
                    if isinstance(x, dict) and 'price' in x and 'size' in x
                }
                await self.market_manager.update_order_book(token_type, bids, asks)
                logger.debug(f"Updated order book for {token_type}: {len(bids)} bids, {len(asks)} asks")

                # Trigger price recalculation
                if self.on_message:
                    result = self.on_message(item)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    logger.warning("on_message callback is None, price won't update!")

            elif 'price_changes' in item:
                # Incremental price changes
                price_changes = item['price_changes']
                if not price_changes:
                    logger.debug(f"Empty price_changes for {token_type}")
                    return

                await self.market_manager.apply_price_changes(
                    token_type,
                    price_changes
                )
                logger.debug(f"Applied {len(price_changes)} price changes for {token_type}")

                # Trigger price recalculation
                if self.on_message:
                    result = self.on_message(item)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    logger.warning("on_message callback is None, price won't update!")
            else:
                logger.debug(f"WebSocket message for {token_type} has neither bids/asks nor price_changes")

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Error processing WebSocket message for {token_type}: {e}", exc_info=True)


class OrderExecutor:
    """Handles order execution with rate limiting and validation.
    
    Improvements:
    - Better rate limiting
    - Input validation
    - Improved error handling
    """
    
    def __init__(self, config: AppConfig, connector):
        """Initialize order executor.
        
        Args:
            config: Application configuration
            connector: Connector instance
        """
        self.config = config
        self.connector = connector
        self.last_order_time: float = 0.0
        self.lock = asyncio.Lock()
    
    async def execute_order(
        self, 
        side: str, 
        size: float, 
        token_map: Dict[str, str],
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute an order with rate limiting and validation.
        
        Args:
            side: Order side ('Up' or 'Down')
            size: Order size (in USDC for market orders, in shares for limit orders)
            token_map: Token map dictionary
            price: Optional limit price. If provided, creates a limit order.
                   If None, creates a market order (aggressive limit).
            
        Returns:
            Order response dictionary or None
        """
        # Input validation
        if side not in token_map:
            logger.error(f"Invalid side: {side}")
            return None
        
        if size < self.config.min_order_size:
            logger.error(f"Order size too small: {size}")
            return None
        
        # Rate limiting
        now = time.time()
        async with self.lock:
            time_since_last = now - self.last_order_time
            if time_since_last < self.config.order_rate_limit_seconds:
                wait_time = self.config.order_rate_limit_seconds - time_since_last
                logger.warning(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                now = time.time()
            self.last_order_time = now
        
        target_token_id = token_map[side]
        try:
            if price is not None:
                # Limit order: size is in shares
                # Convert USDC to shares if needed (assume size is in USDC)
                shares = size / price if price > 0 else size
                
                # Use asyncio.to_thread to wrap synchronous create_order
                resp = await asyncio.to_thread(
                    self.connector.create_order,
                    target_token_id,
                    price,
                    shares,
                    'BUY'
                )
            else:
                # Market order: size is in USDC
                resp = await self.connector.create_market_order(target_token_id, size, 'BUY')
            
            if resp and isinstance(resp, dict) and resp.get('orderID'):
                return resp
            else:
                logger.error(f"Order failed: {resp}")
                return None
        except Exception as e:
            logger.error(f"Order execution error: {e}", exc_info=True)
            return None
    
    async def flatten_positions(self, token_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Flatten all market positions.
        
        Args:
            token_map: Token map dictionary
            
        Returns:
            List of order responses
        """
        try:
            results = await self.connector.flatten_market(token_map)
            return results if results else []
        except Exception as e:
            logger.error(f"Flatten error: {e}", exc_info=True)
            return []

    async def cancel_all_orders(self) -> bool:
        """Cancel all pending orders.

        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.connector.cancel_all_orders()
            return bool(result)
        except Exception as e:
            logger.error(f"Cancel all error: {e}", exc_info=True)
            return False


class RTDSManager:
    """Manages RTDS (Real Time Data Stream) WebSocket connection for crypto prices.
    
    This provides real-time BTC prices from Polymarket's RTDS service, which
    matches the prices used by Polymarket for market resolution.
    
    Improvements:
    - Real-time price updates matching Polymarket's internal prices
    - Automatic reconnection
    - Better error handling
    - Stores historical Chainlink prices with timestamps for strike price lookup
    """
    
    def __init__(
        self,
        config: AppConfig,
        on_btc_price: Callable[[float], Union[None, Awaitable[None]]]
    ):
        """Initialize RTDS manager.

        Args:
            config: Application configuration
            on_btc_price: Callback function for BTC price updates
        """
        self.config = config
        self.on_btc_price = on_btc_price
        self.shutdown_flag = asyncio.Event()
        self.connection_task: Optional[asyncio.Task] = None
        self._ws: Optional[ClientConnection] = None
        self.current_btc_price: Optional[float] = None
        self.current_chainlink_price: Optional[float] = None  # Track Chainlink separately
        # Store historical Chainlink prices with timestamps for strike price lookup
        # Format: {timestamp_ms: price}
        self.chainlink_price_history: Dict[int, float] = {}
        self._history_lock = asyncio.Lock()
        # Message counter for rate-limited logging
        self._message_count = 0
    
    async def start(self) -> None:
        """Start RTDS WebSocket connection."""
        if self.connection_task and not self.connection_task.done():
            logger.debug("RTDS already running")
            return
        
        self.shutdown_flag.clear()
        self.connection_task = asyncio.create_task(self._connect_loop())
    
    async def stop(self) -> None:
        """Stop RTDS WebSocket connection with proper cleanup."""
        self.shutdown_flag.set()
        
        # Close WebSocket if open
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing RTDS WebSocket: {e}")
            finally:
                self._ws = None
        
        # Wait for connection task to finish
        if self.connection_task:
            try:
                await asyncio.wait_for(
                    self.connection_task,
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("RTDS stop timeout, cancelling task")
                self.connection_task.cancel()
                try:
                    await self.connection_task
                except asyncio.CancelledError:
                    pass
    
    async def _connect_loop(self) -> None:
        """Main RTDS connection loop with automatic reconnection."""
        reconnect_attempts = 0
        
        while (not self.shutdown_flag.is_set() and
               reconnect_attempts < self.config.rtds_max_reconnect_attempts):
            
            try:
                async with connect(
                    self.config.rtds_uri,
                    ping_interval=self.config.rtds_ping_interval,
                    ping_timeout=self.config.rtds_ping_timeout,
                    max_size=MAX_WEBSOCKET_MESSAGE_SIZE
                ) as ws:
                    self._ws = ws
                    
                    # Subscribe to BTC price from both Binance and Chainlink sources
                    # Chainlink is what Polymarket uses for resolution
                    # Try subscribing separately to see which format works
                    
                    # First, subscribe to Binance
                    binance_sub = {
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": "crypto_prices",
                                "type": "update",
                                "filters": "btcusdt"
                            }
                        ]
                    }
                    logger.debug(f"RTDS subscribing to Binance: {json.dumps(binance_sub)}")
                    await ws.send(json.dumps(binance_sub))
                    await asyncio.sleep(0.5)  # Small delay between subscriptions
                    
                    # Then subscribe to Chainlink (try different formats)
                    # Format 1: JSON string in filters
                    chainlink_sub1 = {
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": "crypto_prices_chainlink",
                                "type": "*",
                                "filters": '{"symbol":"btc/usd"}'
                            }
                        ]
                    }
                    logger.debug(f"RTDS subscribing to Chainlink (format 1): {json.dumps(chainlink_sub1)}")
                    await ws.send(json.dumps(chainlink_sub1))
                    await asyncio.sleep(0.5)
                    
                    # Format 2: Try without filters first
                    chainlink_sub2 = {
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": "crypto_prices_chainlink",
                                "type": "*"
                            }
                        ]
                    }
                    logger.debug(f"RTDS subscribing to Chainlink (format 2, no filters): {json.dumps(chainlink_sub2)}")
                    await ws.send(json.dumps(chainlink_sub2))
                    reconnect_attempts = 0
                    
                    logger.info("RTDS connected and subscribed to BTC price (Binance + Chainlink)")
                    
                    while not self.shutdown_flag.is_set():
                        try:
                            message = await asyncio.wait_for(
                                ws.recv(),
                                timeout=self.config.rtds_recv_timeout
                            )
                            
                            # Validate message size
                            if len(message) > MAX_WEBSOCKET_MESSAGE_SIZE:
                                logger.warning(
                                    f"RTDS message too large: {len(message)} bytes"
                                )
                                continue
                            
                            data = json.loads(message)
                            await self._process_message(data)
                            
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            continue
                        except ConnectionClosed as e:
                            logger.warning(f"RTDS connection closed: {e}")
                            break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from RTDS: {e}")
                            continue
                        except Exception as e:
                            logger.error(
                                f"Unexpected error in RTDS message loop: {e}",
                                exc_info=True
                            )
                            continue
                            
            except (InvalidURI, InvalidState) as e:
                logger.error(f"RTDS connection error: {e}")
                break
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts < self.config.rtds_max_reconnect_attempts:
                    wait_time = self.config.rtds_reconnect_delay * (
                        2 ** min(reconnect_attempts - 1, 3)
                    )
                    logger.info(
                        f"RTDS error: {e}. "
                        f"Reconnecting in {wait_time}s "
                        f"(attempt {reconnect_attempts}/{self.config.rtds_max_reconnect_attempts})..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"RTDS failed after "
                        f"{self.config.rtds_max_reconnect_attempts} attempts"
                    )
                    break
            finally:
                self._ws = None
    
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """Process RTDS message and extract BTC price.
        
        Processes both Binance (crypto_prices) and Chainlink (crypto_prices_chainlink) sources.
        Prefers Chainlink as it matches Polymarket's resolution source.
        
        Args:
            data: RTDS message dictionary
        """
        try:
            topic = data.get('topic')
            payload = data.get('payload', {})

            # Rate-limited logging: only log every 100th message to reduce log bloat
            self._message_count += 1
            if self._message_count % 100 == 0:
                if topic or payload:
                    logger.debug(f"RTDS message #{self._message_count}: topic={topic}, symbol={payload.get('symbol', 'N/A')}")
                else:
                    logger.debug(f"RTDS message #{self._message_count}: empty (subscription confirmation)")
            
            btc_price = None
            
            # Process Chainlink prices (preferred - matches Polymarket resolution)
            if topic == 'crypto_prices_chainlink':
                symbol = payload.get('symbol', '').lower()
                if symbol == 'btc/usd':
                    price = payload.get('value')
                    timestamp_ms = payload.get('timestamp')
                    if price and isinstance(price, (int, float)) and price > 0:
                        chainlink_price = float(price)
                        self.current_chainlink_price = chainlink_price
                        
                        # Store historical price with timestamp for strike price lookup
                        if timestamp_ms:
                            async with self._history_lock:
                                # Store price, keeping only recent history (last 1 hour = 3.6M ms)
                                self.chainlink_price_history[int(timestamp_ms)] = chainlink_price
                                # Clean up old entries (older than 1 hour)
                                cutoff = int(timestamp_ms) - 3600000
                                self.chainlink_price_history = {
                                    ts: p for ts, p in self.chainlink_price_history.items()
                                    if ts >= cutoff
                                }
                        
                        # Use Chainlink for BTC price (matches Polymarket)
                        btc_price = chainlink_price
                        # Log price updates at DEBUG to reduce log size (only log significant changes)
                        logger.debug(f"RTDS Chainlink BTC/USD: ${btc_price:,.2f}")
            
            # Process Binance prices (fallback only)
            elif topic == 'crypto_prices':
                symbol = payload.get('symbol', '').lower()
                if symbol == 'btcusdt':
                    price = payload.get('value')
                    if price and isinstance(price, (int, float)) and price > 0:
                        binance_price = float(price)
                        self.current_btc_price = binance_price
                        # Only use Binance if Chainlink not available
                        if not self.current_chainlink_price:
                            btc_price = binance_price
                            logger.debug(f"RTDS Binance BTC/USDT: ${btc_price:,.2f} (Chainlink not available)")
            
            # Update price and trigger callback (prefer Chainlink)
            if btc_price and btc_price > 0:
                # Call callback with Chainlink price if available, otherwise Binance
                if self.on_btc_price:
                    result = self.on_btc_price(btc_price)
                    if asyncio.iscoroutine(result):
                        await result
                    
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing RTDS message: {e}", exc_info=True)
    
    def get_current_price(self) -> Optional[float]:
        """Get the current BTC price from RTDS.

        Prefers Chainlink (matches Polymarket resolution) over Binance.

        Returns:
            Current BTC price or None if not available
        """
        price = self.current_chainlink_price or self.current_btc_price
        if price:
            logger.debug(f"RTDS current price: ${price:,.2f} (Chainlink: {self.current_chainlink_price}, Binance: {self.current_btc_price})")
        return price
    
    def get_chainlink_price(self) -> Optional[float]:
        """Get the current Chainlink BTC price.
        
        Returns:
            Current Chainlink BTC/USD price or None if not available
        """
        return self.current_chainlink_price
    
    def get_chainlink_price_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get Chainlink BTC price at a specific timestamp.
        
        Looks up the price from stored RTDS history, or returns None if not available.
        This is used for dynamic strike prices that need the Chainlink price at market start.
        
        Args:
            timestamp: Timestamp to get price for
            
        Returns:
            Chainlink BTC/USD price at that timestamp, or None if not available
        """
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        # Check synchronously (history dict is thread-safe for reads)
        # We don't need the lock for read-only access
        if not self.chainlink_price_history:
            return None
        
        # Try exact match first
        if timestamp_ms in self.chainlink_price_history:
            return self.chainlink_price_history[timestamp_ms]
        
        # Find closest match within 60 seconds (60000 ms)
        closest_ts = None
        closest_diff = float('inf')
        for ts, price in self.chainlink_price_history.items():
            diff = abs(ts - timestamp_ms)
            if diff < closest_diff and diff <= 60000:  # Within 1 minute
                closest_diff = diff
                closest_ts = ts
        
        if closest_ts is not None:
            found_price = self.chainlink_price_history[closest_ts]
            logger.info(
                f"Chainlink price lookup: requested {timestamp_ms}ms, "
                f"found {closest_ts}ms (diff: {closest_diff}ms), price: ${found_price:,.2f}"
            )
            return found_price
        
        # Log available timestamps for debugging
        if self.chainlink_price_history:
            available_ts = sorted(self.chainlink_price_history.keys())
            logger.warning(
                f"No Chainlink price found at {timestamp_ms}ms. "
                f"Available range: {available_ts[0]}ms to {available_ts[-1]}ms "
                f"({len(available_ts)} entries)"
            )
        else:
            logger.warning(f"Chainlink price history is empty, cannot look up {timestamp_ms}ms")
        
        return None

