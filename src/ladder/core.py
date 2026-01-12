"""LadderCore: Controller for Polymarket Ladder UI."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable
import pandas as pd

from src.activetrader.core import FingerBlasterCore
from src.ladder.ladder_data import LadderDataManager, DOMViewModel

logger = logging.getLogger("LadderCore")

class LadderCore:
    def __init__(self, fb_core: Optional[FingerBlasterCore] = None):
        self.fb = fb_core or FingerBlasterCore()
        self.data_manager = LadderDataManager()
        
        # State tracking
        self.last_ladder: Dict[int, Dict] = {}
        self.pending_orders: Dict[str, dict] = {}  # id -> {price, size, side}
        self.active_orders: Dict[int, float] = {}  # price_cent -> total_size
        self.filled_orders: Dict[int, float] = {}  # price_cent -> timestamp (for recently filled orders)
        
        # Market info
        self.market_name: str = "Market"
        self.market_starts: str = ""
        self.market_ends: str = ""
        self.market_update_callback: Optional[Callable[[str, str, str], None]] = None
        
        # Throttling
        self.dirty = False
        self._lock = asyncio.Lock()

        # Open orders cache for My Orders column
        self._cached_open_orders: List[Dict] = []
        self._order_cache_timestamp: float = 0.0
        self._order_cache_ttl: float = 2.0  # Refresh every 2 seconds

        # Registration: Ensure this method exists before registering
        self.fb.register_callback('market_update', self._on_market_update)
        self.fb.register_callback('order_filled', self._on_order_filled)

    def is_pending(self, price_cent: int) -> bool:
        """Explicitly defined to fix the AttributeError."""
        return any(ord['price'] == price_cent for ord in self.pending_orders.values())
    
    def is_filled(self, price_cent: int) -> bool:
        """Check if an order at this price was recently filled."""
        # Keep filled indicator for 5 seconds after fill
        current_time = time.time()
        if price_cent in self.filled_orders:
            if current_time - self.filled_orders[price_cent] < 5.0:
                return True
            else:
                # Clean up old filled orders
                self.filled_orders.pop(price_cent, None)
        return False
    
    def _on_order_filled(self, side: str, size: float, price: float, order_id: str) -> None:
        """Callback when an order is filled. Try to match it to a pending order."""
        matched = False
        
        # Try to find the order in pending_orders by order_id
        if order_id in self.pending_orders:
            order_data = self.pending_orders.pop(order_id, None)
            if order_data:
                price_cent = order_data.get('price')
                if price_cent:
                    # Mark as filled
                    self.filled_orders[price_cent] = time.time()
                    # Remove from active_orders
                    if price_cent in self.active_orders:
                        self.active_orders[price_cent] = max(0, self.active_orders[price_cent] - order_data.get('size', 0))
                        if self.active_orders[price_cent] == 0:
                            self.active_orders.pop(price_cent, None)
                    logger.info(f"Order {order_id[:10]}... filled at price {price_cent}")
                    matched = True
        
        # If not matched by order_id, try to match by fill price and side
        if not matched:
            try:
                # Convert fill price to ladder price based on side
                # For YES: fill price is the YES price
                # For NO: fill price needs to be converted (100 - price_cent)
                fill_price_cent = int(round(price * 100))
                if side.upper() == "NO":
                    fill_price_cent = 100 - fill_price_cent
                
                # Check if any pending order at this price matches
                matching_orders = [
                    (oid, od) for oid, od in self.pending_orders.items()
                    if od.get('price') == fill_price_cent and od.get('side', '').upper() == side.upper()
                ]
                
                if matching_orders:
                    # Mark the first matching order as filled
                    order_id_to_remove, order_data = matching_orders[0]
                    self.pending_orders.pop(order_id_to_remove, None)
                    self.filled_orders[fill_price_cent] = time.time()
                    # Remove from active_orders
                    if fill_price_cent in self.active_orders:
                        self.active_orders[fill_price_cent] = max(0, self.active_orders[fill_price_cent] - order_data.get('size', 0))
                        if self.active_orders[fill_price_cent] == 0:
                            self.active_orders.pop(fill_price_cent, None)
                    logger.info(f"Matched fill to pending order at price {fill_price_cent}")
                    matched = True
                elif fill_price_cent in self.active_orders:
                    # If we have an active order at this price but no pending order,
                    # it might have filled very quickly - mark it as filled anyway
                    self.filled_orders[fill_price_cent] = time.time()
                    logger.info(f"Marked active order at price {fill_price_cent} as filled (no pending order found)")
                    matched = True
            except Exception as e:
                logger.debug(f"Could not match fill to pending order: {e}")
        
        if not matched:
            logger.debug(f"Could not match fill for order {order_id[:10]}... (side={side}, price={price})")

    async def cancel_all_orders(self) -> int:
        """Cancel ALL open orders across all price levels.

        Returns:
            Number of orders canceled
        """
        canceled_count = 0

        # Get all order IDs (make a copy to avoid modification during iteration)
        orders_to_cancel = list(self.pending_orders.keys())

        # Cancel each order
        for order_id in orders_to_cancel:
            try:
                # Skip temporary IDs (they haven't been placed yet)
                if order_id.startswith('tmp_'):
                    self.pending_orders.pop(order_id, None)
                    canceled_count += 1
                    continue

                # Cancel the actual order
                result = await asyncio.to_thread(
                    self.fb.connector.cancel_order,
                    order_id
                )

                if result:
                    canceled_count += 1
                    # Remove from pending orders
                    order_data = self.pending_orders.pop(order_id, None)
                    if order_data:
                        # Update active_orders
                        p = order_data.get('price')
                        if p in self.active_orders:
                            self.active_orders[p] = max(0, self.active_orders[p] - order_data.get('size', 0))
                            if self.active_orders[p] == 0:
                                self.active_orders.pop(p, None)
                else:
                    logger.warning(f"Failed to cancel order {order_id}")
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")

        self.dirty = True
        return canceled_count

    async def cancel_all_at_price(self, price_cent: int) -> int:
        """Cancel all orders at a specific price (in cents).

        Args:
            price_cent: Price in cents (e.g., 86 for $0.86)

        Returns:
            Number of orders canceled
        """
        canceled_count = 0

        # Find all order IDs at this price
        orders_to_cancel = [
            order_id for order_id, order_data in self.pending_orders.items()
            if order_data.get('price') == price_cent
        ]

        # Cancel each order
        for order_id in orders_to_cancel:
            try:
                # Skip temporary IDs (they haven't been placed yet)
                if order_id.startswith('tmp_'):
                    self.pending_orders.pop(order_id, None)
                    canceled_count += 1
                    continue

                # Cancel the actual order
                result = await asyncio.to_thread(
                    self.fb.connector.cancel_order,
                    order_id
                )

                if result:
                    canceled_count += 1
                    # Remove from pending orders
                    order_data = self.pending_orders.pop(order_id, None)
                    if order_data:
                        # Update active_orders
                        p = order_data.get('price')
                        if p in self.active_orders:
                            self.active_orders[p] = max(0, self.active_orders[p] - order_data.get('size', 0))
                            if self.active_orders[p] == 0:
                                self.active_orders.pop(p, None)
                else:
                    logger.warning(f"Failed to cancel order {order_id}")
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")

        self.dirty = True
        return canceled_count

    async def place_limit_order(self, price_cent: int, size: float, side: str) -> Optional[str]:
        """Translates a Ladder click into a signed Polymarket limit order."""
        temp_id = f"tmp_{price_cent}_{asyncio.get_event_loop().time()}"
        self.pending_orders[temp_id] = {"price": price_cent, "size": size, "side": side}
        self.dirty = True

        try:
            token_map = await self.fb.market_manager.get_token_map()
            if not token_map:
                logger.error("No token map available - cannot place order")
                self.pending_orders.pop(temp_id, None)
                return None
            
            # Logic: Buy YES = Buy Up Token. Buy NO = Buy Down Token.
            target_token = token_map.get('Up' if side == "YES" else 'Down')
            
            if not target_token:
                logger.error(f"No target token found for side={side}, token_map={token_map}")
                self.pending_orders.pop(temp_id, None)
                return None
            
            # For NO: Ladder Price 0.70 means buying Down Token at 0.30
            target_price = (price_cent / 100.0) if side == "YES" else ((100 - price_cent) / 100.0)
            
            # Shares = Amount in USDC / Price per share
            shares = size / target_price if target_price > 0 else size
            
            # Polymarket requirements:
            # 1. Minimum order value: $1.00
            # 2. Minimum shares: 5 shares
            MIN_ORDER_VALUE = 1.0
            MIN_SHARES = 5.0
            
            # Calculate actual order value
            actual_order_value = shares * target_price
            
            # First, ensure we meet the $1.00 minimum order value
            if actual_order_value < MIN_ORDER_VALUE:
                # Round up shares to meet $1.00 minimum
                shares = MIN_ORDER_VALUE / target_price
                actual_order_value = shares * target_price
                logger.info(f"Adjusted shares to meet $1.00 minimum: {shares:.6f} shares = ${actual_order_value:.4f}")
            
            # Then, ensure we meet the 5 shares minimum
            if shares < MIN_SHARES:
                shares = MIN_SHARES
                actual_order_value = shares * target_price
                logger.info(f"Adjusted shares to meet 5 shares minimum: {shares:.6f} shares = ${actual_order_value:.4f}")
            
            # Round to 6 decimal places to avoid precision issues
            shares = round(shares, 6)
            actual_order_value = round(actual_order_value, 4)

            logger.info(f"Placing {side} order: token={target_token[:20]}..., price={target_price:.4f}, shares={shares:.6f}, order_value=${actual_order_value:.4f}")

            # Threaded execution to keep the UI loop unblocked
            order_resp = await asyncio.to_thread(
                self.fb.connector.create_order,
                target_token, target_price, shares, 'BUY'
            )
            
            logger.info(f"Order response: {order_resp}")
            
            if not order_resp:
                logger.error("Order response is None - order may have failed")
                self.pending_orders.pop(temp_id, None)
                return None
            
            # Check for orderID in various possible formats
            order_id = None
            if isinstance(order_resp, dict):
                order_id = (
                    order_resp.get('orderID') or 
                    order_resp.get('order_id') or 
                    order_resp.get('id') or
                    order_resp.get('hash')  # Some APIs return hash
                )
            
            if order_id:
                logger.info(f"Order placed successfully: {order_id}")
                self.pending_orders[order_id] = self.pending_orders.pop(temp_id)
                self.active_orders[price_cent] = self.active_orders.get(price_cent, 0) + size
                return order_id
            else:
                logger.error(f"Order response missing orderID. Full response: {order_resp}")
                self.pending_orders.pop(temp_id, None)
                return None
                
        except Exception as e:
            logger.error(f"Order Placement Error: {e}", exc_info=True)
            self.pending_orders.pop(temp_id, None)
            return None

    async def place_market_order(self, size: float, side: str) -> Optional[str]:
        """Place a market order (BUY YES or BUY NO).
        
        Args:
            size: Order size in USDC
            side: 'YES' or 'NO'
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            token_map = await self.fb.market_manager.get_token_map()
            if not token_map:
                logger.error("No token map available - cannot place order")
                return None
            
            # Logic: Buy YES = Buy Up Token. Buy NO = Buy Down Token.
            target_token = token_map.get('Up' if side == "YES" else 'Down')
            
            if not target_token:
                logger.error(f"No target token found for side={side}, token_map={token_map}")
                return None
            
            logger.info(f"Placing market {side} order: token={target_token[:20]}..., size=${size:.2f}")

            # create_market_order is already async, so await it directly
            order_resp = await self.fb.connector.create_market_order(
                target_token, size, 'BUY'
            )
            
            logger.info(f"Market order response: {order_resp}")
            
            if not order_resp:
                logger.error("Market order response is None - order may have failed")
                return None
            
            # Check for orderID in various possible formats
            order_id = None
            if isinstance(order_resp, dict):
                order_id = (
                    order_resp.get('orderID') or 
                    order_resp.get('order_id') or 
                    order_resp.get('id') or
                    order_resp.get('hash')
                )
            
            if order_id:
                logger.info(f"Market order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"Market order response missing orderID. Full response: {order_resp}")
                return None
                
        except Exception as e:
            logger.error(f"Market Order Placement Error: {e}", exc_info=True)
            return None

    def get_view_model(self) -> Dict[int, Dict]:
        """Aggregates books and overlays user state. (Legacy method for compatibility)"""
        try:
            raw_books = self.fb.market_manager.raw_books
            up_book = raw_books.get('Up', {'bids': {}, 'asks': {}})
            down_book = raw_books.get('Down', {'bids': {}, 'asks': {}})
        except Exception:
            return self.last_ladder

        # Build the 1-99 ladder from both books
        ladder = self.data_manager.build_ladder_data(up_book, down_book)

        # Inject user orders
        for order in self.pending_orders.values():
            p = order['price']
            if p in ladder: ladder[p]['my_size'] += order['size']

        for p_cent, total_size in self.active_orders.items():
            if p_cent in ladder: ladder[p_cent]['my_size'] += total_size

        self.last_ladder = ladder
        return ladder

    def get_open_orders_for_display(self) -> List[Dict]:
        """
        Get user's open orders formatted for the My Orders column.

        Returns list of dicts: [{order_id, price_cent, size, side}, ...]
        Uses local tracking (pending_orders + active_orders) for now.
        """
        orders = []

        # Add pending orders
        for order_id, order_data in self.pending_orders.items():
            price_cent = order_data.get('price')
            if price_cent and 1 <= price_cent <= 99:
                orders.append({
                    'order_id': order_id,
                    'price_cent': price_cent,
                    'size': order_data.get('size', 0.0),
                    'side': order_data.get('side', 'YES')
                })

        return orders

    def get_dom_view_model(self) -> DOMViewModel:
        """
        Build the 5-column DOM view model with spread detection.

        Returns:
            DOMViewModel with all 99 price levels, depths, spread info, and user orders
        """
        try:
            raw_books = self.fb.market_manager.raw_books
            up_book = raw_books.get('Up', {'bids': {}, 'asks': {}})
            down_book = raw_books.get('Down', {'bids': {}, 'asks': {}})
        except Exception:
            # Return empty view model on error
            up_book = {'bids': {}, 'asks': {}}
            down_book = {'bids': {}, 'asks': {}}

        # Get user orders for display
        user_orders = self.get_open_orders_for_display()

        # Build DOM view model
        return self.data_manager.build_dom_data(up_book, down_book, user_orders)

    def _on_market_update(self, strike: str, ends: str, market_name: str, starts: str = None) -> None:
        """Handle market update callback - now receives start time too."""
        # Check if this is a new market (different from current)
        is_new_market = (self.market_name != market_name)

        self.market_name = market_name
        self.market_starts = starts or ""
        self.market_ends = ends or ""

        # Clear order state when transitioning to new market
        if is_new_market:
            logger.info(f"New market detected: {market_name}. Clearing order state.")
            self.pending_orders.clear()
            self.active_orders.clear()
            self.filled_orders.clear()

        if self.market_update_callback:
            self.market_update_callback(market_name, starts, ends)

    def _format_market_display(self, name: str, ends: str) -> str:
        """Legacy method - kept for backward compatibility."""
        try:
            if not ends or ends == 'N/A': return name
            dt = pd.Timestamp(ends).tz_localize('UTC').tz_convert('US/Eastern')
            time_str = dt.strftime("%B %d, %I:%M%p ET")
            return f"{name} - {time_str}"
        except: return name

    def set_market_update_callback(self, callback: Optional[Callable[[str, str, str], None]]) -> None:
        """Set callback for market updates - receives (name, starts, ends)."""
        self.market_update_callback = callback

    def get_market_fields(self) -> Dict[str, str]:
        """Get separate market data fields for UI display."""
        return {
            'name': self.market_name,
            'starts': self.market_starts,
            'ends': self.market_ends
        }

    def get_market_display(self):
        """Legacy method - returns formatted display text."""
        return self._format_market_display(self.market_name, self.market_ends)