"""LadderCore: Controller for Polymarket Ladder UI."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable

from src.activetrader.core import FingerBlasterCore
from src.ladder.ladder_data import LadderDataManager, DOMViewModel

logger = logging.getLogger("LadderCore")

# Constants
FILLED_ORDER_WINDOW_SECS = 5.0
PRICE_CENT_RANGE = (1, 99)
MIN_SHARES = 5.0
MIN_ORDER_VALUE = 1.0
POLYMARKET_PRICE_SCALE = 100  # Convert price (0-1) to cents (0-100)


class LadderCore:
    def __init__(self, fb_core: Optional[FingerBlasterCore] = None):
        self.fb = fb_core or FingerBlasterCore()
        self.data_manager = LadderDataManager()

        self.last_ladder: Dict[int, Dict] = {}
        self.pending_orders: Dict[str, dict] = {}
        self.active_orders: Dict[int, float] = {}
        self.filled_orders: Dict[int, float] = {}

        self.market_name: str = "Market"
        self.market_starts: str = ""
        self.market_ends: str = ""
        self.market_update_callback: Optional[Callable[[str, str, str], None]] = None

        self.dirty = False
        self._lock = asyncio.Lock()
        self._cached_open_orders: List[Dict] = []
        self._order_cache_timestamp: float = 0.0
        self._order_cache_ttl: float = 2.0

        self.fb.register_callback('market_update', self._on_market_update)
        self.fb.register_callback('order_filled', self._on_order_filled)

    def is_pending(self, price_cent: int) -> bool:
        """Check if there's a pending order at this price."""
        return any(ord.get('price') == price_cent for ord in self.pending_orders.values())

    def is_filled(self, price_cent: int) -> bool:
        """Check if an order at this price was recently filled."""
        if price_cent not in self.filled_orders:
            return False
        if time.time() - self.filled_orders[price_cent] < FILLED_ORDER_WINDOW_SECS:
            return True
        self.filled_orders.pop(price_cent, None)
        return False
    
    def _on_order_filled(self, side: str, size: float, price: float, order_id: str) -> None:
        """Match filled order to pending/active orders and update state."""
        if not order_id:
            logger.debug("Received order fill with empty order_id")
            return

        # Direct match by order ID
        if order_id in self.pending_orders:
            order_data = self.pending_orders.pop(order_id)
            price_cent = order_data.get('price')
            if price_cent:
                self.filled_orders[price_cent] = time.time()
                self._reduce_active_order(price_cent, order_data.get('size', 0))
                logger.info(f"Order {order_id[:10]}... filled at price {price_cent}")
            return

        # Fallback: match by price and side
        fill_price_cent = int(round(price * POLYMARKET_PRICE_SCALE))
        if side.upper() == "NO":
            fill_price_cent = POLYMARKET_PRICE_SCALE - fill_price_cent

        if not PRICE_CENT_RANGE[0] <= fill_price_cent <= PRICE_CENT_RANGE[1]:
            logger.debug(f"Filled price {fill_price_cent}c out of range")
            return

        # Try to match by side and price
        matching = [(oid, od) for oid, od in self.pending_orders.items()
                    if od.get('price') == fill_price_cent and od.get('side', '').upper() == side.upper()]

        if matching:
            order_id_to_remove, order_data = matching[0]
            self.pending_orders.pop(order_id_to_remove, None)
            self.filled_orders[fill_price_cent] = time.time()
            self._reduce_active_order(fill_price_cent, order_data.get('size', 0))
            logger.info(f"Matched fill to pending order at {fill_price_cent}c")
        elif fill_price_cent in self.active_orders:
            self.filled_orders[fill_price_cent] = time.time()
            logger.info(f"Marked active order at {fill_price_cent}c as filled")
        else:
            logger.debug(f"Could not match fill for order {order_id[:10]}...")

    def _reduce_active_order(self, price_cent: int, size: float) -> None:
        """Reduce or remove active order at price level."""
        if price_cent not in self.active_orders:
            return
        self.active_orders[price_cent] = max(0, self.active_orders[price_cent] - size)
        if self.active_orders[price_cent] == 0:
            self.active_orders.pop(price_cent, None)

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of canceled orders."""
        canceled_count = 0
        for order_id in list(self.pending_orders.keys()):
            if await self._cancel_single_order(order_id):
                canceled_count += 1
        self.dirty = True
        return canceled_count

    async def cancel_all_at_price(self, price_cent: int) -> int:
        """Cancel all orders at a specific price. Returns count of canceled orders."""
        canceled_count = 0
        orders_to_cancel = [
            oid for oid, od in self.pending_orders.items()
            if od.get('price') == price_cent
        ]
        for order_id in orders_to_cancel:
            if await self._cancel_single_order(order_id):
                canceled_count += 1
        self.dirty = True
        return canceled_count

    async def _cancel_single_order(self, order_id: str) -> bool:
        """Cancel a single order. Returns True if successful."""
        try:
            if order_id.startswith('tmp_'):
                self.pending_orders.pop(order_id, None)
                return True

            result = await asyncio.to_thread(self.fb.connector.cancel_order, order_id)
            if result:
                order_data = self.pending_orders.pop(order_id, None)
                if order_data:
                    self._reduce_active_order(order_data.get('price'), order_data.get('size', 0))
                return True
            logger.warning(f"Failed to cancel order {order_id}")
            return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def _extract_order_id(self, order_resp: Any) -> Optional[str]:
        """Extract order ID from response dict."""
        if not isinstance(order_resp, dict):
            return None
        # Try common keys in order of likelihood
        return (order_resp.get('orderID') or order_resp.get('order_id') or
                order_resp.get('id') or order_resp.get('hash'))

    async def _get_target_token(self, side: str) -> Optional[str]:
        """Get token ID for given side (YES=Up, NO=Down)."""
        token_map = await self.fb.market_manager.get_token_map()
        if not token_map:
            return None
        token_key = 'Up' if side == "YES" else 'Down'
        return token_map.get(token_key)

    async def place_limit_order(self, price_cent: int, size: float, side: str) -> Optional[str]:
        """Place a limit order at the given price in USDC."""
        if not PRICE_CENT_RANGE[0] <= price_cent <= PRICE_CENT_RANGE[1]:
            logger.warning(f"Price {price_cent}c out of valid range {PRICE_CENT_RANGE}")
            return None

        temp_id = f"tmp_{price_cent}_{asyncio.get_event_loop().time()}"
        self.pending_orders[temp_id] = {"price": price_cent, "size": size, "side": side}
        self.dirty = True

        try:
            target_token = await self._get_target_token(side)
            if not target_token:
                self.pending_orders.pop(temp_id, None)
                return None

            # For NO: Ladder Price 70 means buying Down Token at 0.30
            target_price = price_cent / POLYMARKET_PRICE_SCALE if side == "YES" else (POLYMARKET_PRICE_SCALE - price_cent) / POLYMARKET_PRICE_SCALE

            # Calculate minimum shares to meet Polymarket minimums: $1.00 order value, 5 shares
            shares = max(MIN_SHARES, MIN_ORDER_VALUE / target_price) if target_price > 0 else MIN_SHARES
            shares = round(shares, 6)

            logger.info(f"Placing {side} limit @ {target_price:.4f}, {shares:.2f} shares")

            order_resp = await asyncio.to_thread(
                self.fb.connector.create_order,
                target_token, target_price, shares, 'BUY'
            )

            order_id = self._extract_order_id(order_resp)
            if order_id:
                self.pending_orders[order_id] = self.pending_orders.pop(temp_id)
                self.active_orders[price_cent] = self.active_orders.get(price_cent, 0) + size
                return order_id

            logger.error(f"Order response missing orderID: {order_resp}")
            self.pending_orders.pop(temp_id, None)
            return None

        except Exception as e:
            logger.error(f"Limit order placement failed: {e}", exc_info=True)
            self.pending_orders.pop(temp_id, None)
            return None

    async def place_market_order(self, size: float, side: str) -> Optional[str]:
        """Place a market order for the given side and size in USDC."""
        try:
            target_token = await self._get_target_token(side)
            if not target_token:
                return None

            logger.info(f"Placing market {side} order, size=${size:.2f}")
            order_resp = await asyncio.to_thread(
                self.fb.connector.create_market_order, target_token, size, 'BUY'
            )

            order_id = self._extract_order_id(order_resp)
            if order_id:
                return order_id

            logger.error(f"Market order response missing orderID: {order_resp}")
            return None
        except Exception as e:
            logger.error(f"Market order placement failed: {e}", exc_info=True)
            return None

    def get_view_model(self) -> Dict[int, Dict]:
        """Build ladder view with order books and user orders overlaid."""
        try:
            raw_books = self.fb.market_manager.raw_books
            up_book = raw_books.get('Up', {'bids': {}, 'asks': {}})
            down_book = raw_books.get('Down', {'bids': {}, 'asks': {}})
        except Exception as e:
            logger.debug(f"Error fetching raw_books for view model: {e}")
            return self.last_ladder

        ladder = self.data_manager.build_ladder_data(up_book, down_book)

        for order in self.pending_orders.values():
            p = order['price']
            if p in ladder:
                ladder[p]['my_size'] += order['size']

        for p_cent, total_size in self.active_orders.items():
            if p_cent in ladder:
                ladder[p_cent]['my_size'] += total_size

        self.last_ladder = ladder
        return ladder

    def get_open_orders_for_display(self) -> List[Dict]:
        """Get pending orders as list of {order_id, price_cent, size, side}."""
        result = []
        for oid, od in self.pending_orders.items():
            price = od.get('price')
            if price and PRICE_CENT_RANGE[0] <= price <= PRICE_CENT_RANGE[1]:
                result.append({
                    'order_id': oid,
                    'price_cent': price,
                    'size': od.get('size', 0.0),
                    'side': od.get('side', 'YES')
                })
        return result

    def get_dom_view_model(self) -> DOMViewModel:
        """Build 5-column DOM view model with spread detection."""
        try:
            raw_books = self.fb.market_manager.raw_books
        except Exception as e:
            logger.debug(f"Error fetching market books: {e}")
            raw_books = {}

        up_book = raw_books.get('Up', {'bids': {}, 'asks': {}})
        down_book = raw_books.get('Down', {'bids': {}, 'asks': {}})
        return self.data_manager.build_dom_data(up_book, down_book, self.get_open_orders_for_display())

    def _on_market_update(self, strike: str, ends: str, market_name: str, starts: str = None) -> None:
        """Handle market update, clearing order state on market change."""
        is_new_market = (self.market_name != market_name)
        self.market_name = market_name
        self.market_starts = starts or ""
        self.market_ends = ends or ""

        if is_new_market:
            logger.info(f"New market: {market_name}")
            self.pending_orders.clear()
            self.active_orders.clear()
            self.filled_orders.clear()

        if self.market_update_callback:
            self.market_update_callback(market_name, starts, ends)

    def set_market_update_callback(self, callback: Optional[Callable[[str, str, str], None]]) -> None:
        self.market_update_callback = callback

    def get_market_fields(self) -> Dict[str, str]:
        return {'name': self.market_name, 'starts': self.market_starts, 'ends': self.market_ends}