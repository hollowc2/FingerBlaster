"""Core engine components: DataManagers, WebSocket, and OrderExecutor."""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional, Dict, List, Tuple, Any, Callable, Awaitable, Union

import pandas as pd
import websockets

from src.config import AppConfig

logger = logging.getLogger("FingerBlaster")


class MarketDataManager:
    """Manages market data and order book state."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lock = asyncio.Lock()
        self.current_market: Optional[Dict[str, Any]] = None
        self.token_map: Dict[str, str] = {}
        self.raw_books: Dict[str, Dict[str, Dict[float, float]]] = {
            'YES': {'bids': {}, 'asks': {}},
            'NO': {'bids': {}, 'asks': {}}
        }
        self.market_start_time: Optional[pd.Timestamp] = None
    
    async def set_market(self, market: Dict[str, Any]) -> bool:
        """Set the current market with validation."""
        if not self._validate_market(market):
            return False
        
        async with self.lock:
            self.current_market = market
            self.token_map = market.get('token_map', {})
            
            # Calculate market start time
            end_dt = pd.Timestamp(market.get('end_date'))
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            self.market_start_time = end_dt - pd.Timedelta(
                minutes=self.config.market_duration_minutes
            )
        
        return True
    
    async def clear_market(self) -> None:
        """Clear current market state."""
        async with self.lock:
            self.current_market = None
            self.token_map = {}
            self.market_start_time = None
            self.raw_books = {
                'YES': {'bids': {}, 'asks': {}},
                'NO': {'bids': {}, 'asks': {}}
            }
    
    async def update_order_book(
        self, 
        token_type: str, 
        bids: Dict[float, float], 
        asks: Dict[float, float]
    ) -> None:
        """Update order book for a token type."""
        if token_type not in self.raw_books:
            return
        
        async with self.lock:
            self.raw_books[token_type]['bids'] = bids
            self.raw_books[token_type]['asks'] = asks
    
    async def apply_price_changes(
        self, 
        token_type: str, 
        changes: List[Dict[str, Any]]
    ) -> None:
        """Apply incremental price changes to order book."""
        if token_type not in self.raw_books:
            return
        
        async with self.lock:
            target_book = self.raw_books[token_type]
            for change in changes:
                if not isinstance(change, dict):
                    continue
                
                try:
                    price = float(change.get('price', 0))
                    size = float(change.get('size', 0))
                    side = str(change.get('side', '')).upper()
                    
                    if side not in ('BUY', 'SELL'):
                        continue
                    
                    target_dict = target_book['bids'] if side == 'BUY' else target_book['asks']
                    
                    if size <= 0:
                        target_dict.pop(price, None)
                    else:
                        target_dict[price] = size
                except (ValueError, KeyError, TypeError):
                    continue
    
    async def calculate_mid_price(self) -> Tuple[float, float, float, float]:
        """
        Calculate mid price from order books.
        
        Returns:
            Tuple of (yes_price, no_price, best_bid, best_ask)
        """
        async with self.lock:
            raw = self.raw_books
            yes_bids = raw['YES']['bids']
            yes_asks = raw['YES']['asks']
            no_bids = raw['NO']['bids']
            no_asks = raw['NO']['asks']
        
        # Convert NO prices to YES prices
        combined_bids = dict(yes_bids)
        combined_asks = dict(yes_asks)
        
        for p_no, s_no in no_asks.items():
            if s_no > 0:
                p_yes = round(1.0 - p_no, 4)
                combined_bids[p_yes] = combined_bids.get(p_yes, 0.0) + s_no
        
        for p_no, s_no in no_bids.items():
            if s_no > 0:
                p_yes = round(1.0 - p_no, 4)
                combined_asks[p_yes] = combined_asks.get(p_yes, 0.0) + s_no
        
        bids_sorted = sorted(combined_bids.keys(), reverse=True)
        asks_sorted = sorted(combined_asks.keys())
        
        best_bid = bids_sorted[0] if bids_sorted else 0.0
        best_ask = asks_sorted[0] if asks_sorted else 1.0
        
        # Calculate mid price
        if best_bid > 0 and best_ask < 1.0:
            mid = (best_bid + best_ask) / 2
        elif best_ask < 1.0:
            mid = best_ask
        elif best_bid > 0:
            mid = best_bid
        else:
            mid = 0.5
        
        return mid, 1.0 - mid, best_bid, best_ask
    
    def _validate_market(self, market: Dict[str, Any]) -> bool:
        """Validate market data structure."""
        required_keys = ['market_id', 'end_date', 'token_map']
        if not all(key in market for key in required_keys):
            logger.warning("Invalid market data: missing required keys")
            return False
        
        token_map = market.get('token_map', {})
        if not isinstance(token_map, dict) or 'YES' not in token_map or 'NO' not in token_map:
            logger.warning("Invalid token_map in market data")
            return False
        
        return True
    
    async def get_market(self) -> Optional[Dict[str, Any]]:
        """Get current market (thread-safe)."""
        async with self.lock:
            return self.current_market
    
    async def get_token_map(self) -> Dict[str, str]:
        """Get token map (thread-safe)."""
        async with self.lock:
            return self.token_map.copy()
    
    async def get_market_start_time(self) -> Optional[pd.Timestamp]:
        """Get market start time (thread-safe)."""
        async with self.lock:
            return self.market_start_time


class HistoryManager:
    """Manages price history with efficient data structures."""
    
    def __init__(self, config: AppConfig):
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
        """Add a price point to history if within market duration."""
        if market_start_time is None:
            return
        
        if 0 <= elapsed_seconds <= self.config.market_duration_seconds:
            async with self.lock:
                self.yes_history.append((elapsed_seconds, price))
    
    async def add_btc_price(self, price: float) -> None:
        """Add BTC price to history."""
        async with self.lock:
            self.btc_history.append(price)
    
    async def get_yes_history(self) -> List[Tuple[float, float]]:
        """Get YES price history (thread-safe copy)."""
        async with self.lock:
            return list(self.yes_history)
    
    async def get_btc_history(self) -> List[float]:
        """Get BTC price history (thread-safe copy)."""
        async with self.lock:
            return list(self.btc_history)
    
    async def clear_yes_history(self) -> None:
        """Clear YES price history."""
        async with self.lock:
            self.yes_history.clear()


class WebSocketManager:
    """Manages WebSocket connection and message processing."""
    
    def __init__(
        self, 
        config: AppConfig, 
        market_manager: MarketDataManager,
        on_message: Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
    ):
        self.config = config
        self.market_manager = market_manager
        self.on_message = on_message
        self.shutdown_flag = asyncio.Event()
        self.connection_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start WebSocket connection."""
        if self.connection_task and not self.connection_task.done():
            return
        
        self.shutdown_flag.clear()
        self.connection_task = asyncio.create_task(self._connect_loop())
    
    async def stop(self) -> None:
        """Stop WebSocket connection."""
        self.shutdown_flag.set()
        if self.connection_task:
            await asyncio.wait_for(self.connection_task, timeout=5.0)
    
    async def _connect_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        reconnect_attempts = 0
        
        while not self.shutdown_flag.is_set() and reconnect_attempts < self.config.ws_max_reconnect_attempts:
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
                async with websockets.connect(
                    self.config.ws_uri,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout
                ) as ws:
                    msg = {"assets_ids": subscribe_ids, "type": "market"}
                    await ws.send(json.dumps(msg))
                    reconnect_attempts = 0
                    
                    while (not self.shutdown_flag.is_set() and 
                           market and 
                           market.get('market_id') == subscribed_market_id):
                        try:
                            message = await asyncio.wait_for(
                                ws.recv(), 
                                timeout=self.config.ws_recv_timeout
                            )
                            data = json.loads(message)
                            
                            if isinstance(data, list):
                                for item in data:
                                    await self._process_message(item)
                            else:
                                await self._process_message(data)
                                
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed as e:
                            logger.warning(f"WebSocket connection closed: {e}")
                            break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON received: {e}")
                            continue
                        
                        # Re-check market in case it changed
                        market = await self.market_manager.get_market()
                        
            except (websockets.InvalidURI, websockets.InvalidState) as e:
                logger.error(f"WebSocket connection error: {e}")
                break
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts < self.config.ws_max_reconnect_attempts:
                    wait_time = self.config.ws_reconnect_delay * (2 ** min(reconnect_attempts - 1, 3))
                    logger.info(f"WebSocket error: {e}. Reconnecting in {wait_time}s (attempt {reconnect_attempts}/{self.config.ws_max_reconnect_attempts})...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"WebSocket failed after {self.config.ws_max_reconnect_attempts} attempts")
                    break
    
    async def _process_message(self, item: Dict[str, Any]) -> None:
        """Process a WebSocket message."""
        if not isinstance(item, dict):
            return
        
        asset_id = item.get('asset_id')
        if not asset_id:
            return
        
        token_map = await self.market_manager.get_token_map()
        token_type = None
        for outcome, tid in token_map.items():
            if tid == asset_id:
                token_type = outcome
                break
        
        if not token_type or token_type not in ('YES', 'NO'):
            return
        
        try:
            if 'bids' in item and 'asks' in item:
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
                # Trigger price recalculation after order book update
                if self.on_message:
                    result = self.on_message(item)
                    # If the callback is async, await it
                    if asyncio.iscoroutine(result):
                        await result
            elif 'price_changes' in item:
                await self.market_manager.apply_price_changes(
                    token_type, 
                    item['price_changes']
                )
                # Trigger price recalculation after price changes
                if self.on_message:
                    result = self.on_message(item)
                    # If the callback is async, await it
                    if asyncio.iscoroutine(result):
                        await result
                
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Error processing WebSocket message: {e}")


class OrderExecutor:
    """Handles order execution with rate limiting."""
    
    def __init__(self, config: AppConfig, connector):
        self.config = config
        self.connector = connector
        self.last_order_time: float = 0.0
        self.lock = asyncio.Lock()
    
    async def execute_order(
        self, 
        side: str, 
        size: float, 
        token_map: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Execute a market order with rate limiting."""
        # Rate limiting
        now = time.time()
        async with self.lock:
            if now - self.last_order_time < self.config.order_rate_limit_seconds:
                return None
            self.last_order_time = now
        
        if side not in token_map:
            logger.error(f"Invalid side: {side}")
            return None
        
        target_token_id = token_map[side]
        try:
            resp = self.connector.create_market_order(target_token_id, size, 'BUY')
            if resp and isinstance(resp, dict) and resp.get('orderID'):
                return resp
            else:
                logger.error(f"Order failed: {resp}")
                return None
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None
    
    async def flatten_positions(self, token_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Flatten all market positions."""
        try:
            results = self.connector.flatten_market(token_map)
            return results if results else []
        except Exception as e:
            logger.error(f"Flatten error: {e}")
            return []
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all pending orders."""
        try:
            self.connector.cancel_all_orders()
            return True
        except Exception as e:
            logger.error(f"Cancel all error: {e}")
            return False

