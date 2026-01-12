"""Data normalization for Polymarket ladder UI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UserOrder:
    """Represents a user's working order at a price level."""
    order_id: str
    size: float
    side: str  # "YES" or "NO"


@dataclass
class DOMRow:
    """A single row in the DOM ladder."""
    price_cent: int       # 1-99 (YES price in cents)
    no_price: int         # 100 - price_cent (complementary NO price)
    no_depth: float       # Liquidity to buy NO at this level
    yes_depth: float      # Liquidity to buy YES at this level
    my_orders: List[UserOrder] = field(default_factory=list)
    is_inside_spread: bool = False
    is_best_bid: bool = False
    is_best_ask: bool = False


@dataclass
class DOMViewModel:
    """Complete view model for the DOM ladder."""
    rows: Dict[int, DOMRow]       # price_cent (1-99) -> DOMRow
    max_depth: float              # Maximum depth across all levels (for bar scaling)
    best_bid_cent: int            # Best YES bid price in cents
    best_ask_cent: int            # Best YES ask price in cents
    mid_price_cent: int           # Mid price for centering


class LadderDataManager:
    """Handles merging Up and Down token books into a single YES ladder."""
    
    @staticmethod
    def to_cent(price: float) -> int:
        return int(round(price * 100))

    def build_ladder_data(self, up_book: Dict, down_book: Dict) -> Dict[int, Dict]:
        """
        Aggregates liquidity:
        - YES Bid (Green) = Up Token Bids
        - YES Ask (Red)   = Down Token Bids (at 1-Price)
        """
        # Initialize the full ladder (1 to 99)
        ladder = {
            i: {"price": i / 100, "yes_bid": 0.0, "yes_ask": 0.0, "my_size": 0.0} 
            for i in range(1, 100)
        }

        # 1. Map UP Bids directly to YES Bids (Liquidity to buy YES)
        for price, size in up_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                if 1 <= cent <= 99:
                    ladder[cent]["yes_bid"] += float(size)
            except (ValueError, TypeError): continue

        # 2. Map DOWN Bids to YES Asks (Liquidity to buy NO = Ask on YES)
        for price, size in down_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                yes_equivalent_cent = 100 - cent
                if 1 <= yes_equivalent_cent <= 99:
                    ladder[yes_equivalent_cent]["yes_ask"] += float(size)
            except (ValueError, TypeError): continue

        return ladder

    def build_dom_data(
        self,
        up_book: Dict,
        down_book: Dict,
        user_orders: Optional[List[Dict]] = None
    ) -> DOMViewModel:
        """
        Build 5-column DOM view model from raw order books.

        Mapping:
        - YES Depth (green) = Up Token Bids at price
        - NO Depth (red) = Down Token Bids at (100 - price)

        Args:
            up_book: Raw order book for Up token {'bids': {price: size}, 'asks': {...}}
            down_book: Raw order book for Down token
            user_orders: List of user's open orders [{order_id, price_cent, size, side}, ...]

        Returns:
            DOMViewModel with all 99 price levels and spread detection
        """
        user_orders = user_orders or []
        max_depth = 0.0

        # Initialize all 99 price levels
        rows: Dict[int, DOMRow] = {}
        for i in range(1, 100):
            rows[i] = DOMRow(
                price_cent=i,
                no_price=100 - i,
                no_depth=0.0,
                yes_depth=0.0,
                my_orders=[],
                is_inside_spread=False,
                is_best_bid=False,
                is_best_ask=False
            )

        # 1. Map UP Bids to YES depth (liquidity to buy YES at this price)
        for price, size in up_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                if 1 <= cent <= 99:
                    rows[cent].yes_depth += float(size)
                    max_depth = max(max_depth, rows[cent].yes_depth)
            except (ValueError, TypeError):
                continue

        # 2. Map DOWN Bids to NO depth (liquidity to buy NO)
        # Down bid at price P means NO liquidity at YES price (100 - P)
        for price, size in down_book.get('bids', {}).items():
            try:
                down_cent = self.to_cent(float(price))
                yes_cent = 100 - down_cent  # Complementary price
                if 1 <= yes_cent <= 99:
                    rows[yes_cent].no_depth += float(size)
                    max_depth = max(max_depth, rows[yes_cent].no_depth)
            except (ValueError, TypeError):
                continue

        # 3. Find best bid/ask for spread detection
        # Best YES bid = highest price where someone wants to buy YES
        yes_bids = [p for p, r in rows.items() if r.yes_depth > 0]
        best_bid_cent = max(yes_bids) if yes_bids else 0

        # Best YES ask = lowest price where NO liquidity exists
        # (someone buying NO at complement = selling YES at this price)
        yes_asks = [p for p, r in rows.items() if r.no_depth > 0]
        best_ask_cent = min(yes_asks) if yes_asks else 100

        # 4. Mark spread and best levels
        for cent, row in rows.items():
            row.is_best_bid = (cent == best_bid_cent and best_bid_cent > 0)
            row.is_best_ask = (cent == best_ask_cent and best_ask_cent < 100)
            row.is_inside_spread = (best_bid_cent < cent < best_ask_cent)

        # 5. Map user orders to rows
        for order in user_orders:
            price_cent = order.get('price_cent')
            if price_cent and 1 <= price_cent <= 99:
                rows[price_cent].my_orders.append(
                    UserOrder(
                        order_id=order.get('order_id', ''),
                        size=order.get('size', 0.0),
                        side=order.get('side', 'YES')
                    )
                )

        # Calculate mid price for centering
        if best_bid_cent > 0 and best_ask_cent < 100:
            mid_price_cent = (best_bid_cent + best_ask_cent) // 2
        elif best_bid_cent > 0:
            mid_price_cent = best_bid_cent
        elif best_ask_cent < 100:
            mid_price_cent = best_ask_cent
        else:
            mid_price_cent = 50

        return DOMViewModel(
            rows=rows,
            max_depth=max_depth,
            best_bid_cent=best_bid_cent,
            best_ask_cent=best_ask_cent,
            mid_price_cent=mid_price_cent
        )