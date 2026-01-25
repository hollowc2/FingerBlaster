"""Data normalization for Polymarket ladder UI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UserOrder:
    order_id: str
    size: float
    side: str  # "YES" or "NO"


@dataclass
class DOMRow:
    price_cent: int
    no_price: int
    no_depth: float
    yes_depth: float
    my_orders: List[UserOrder] = field(default_factory=list)
    is_inside_spread: bool = False
    is_best_bid: bool = False
    is_best_ask: bool = False


@dataclass
class DOMViewModel:
    rows: Dict[int, DOMRow]
    max_depth: float
    best_bid_cent: int
    best_ask_cent: int
    mid_price_cent: int


class LadderDataManager:
    """Merges Up/Down token books into a unified YES ladder."""

    @staticmethod
    def to_cent(price: float) -> int:
        return int(round(price * 100))

    def build_ladder_data(self, up_book: Dict, down_book: Dict) -> Dict[int, Dict]:
        """Build ladder: YES Bid = Up Bids, YES Ask = Down Bids at (1-price)."""
        ladder = {i: {"price": i / 100, "yes_bid": 0.0, "yes_ask": 0.0, "my_size": 0.0} for i in range(1, 100)}

        # Up Bids → YES Bids
        for price, size in up_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                if 1 <= cent <= 99:
                    ladder[cent]["yes_bid"] += float(size)
            except (ValueError, TypeError):
                continue

        # Down Bids → YES Asks (at complementary price)
        for price, size in down_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                yes_cent = 100 - cent
                if 1 <= yes_cent <= 99:
                    ladder[yes_cent]["yes_ask"] += float(size)
            except (ValueError, TypeError):
                continue

        return ladder

    def build_dom_data(
        self,
        up_book: Dict,
        down_book: Dict,
        user_orders: Optional[List[Dict]] = None
    ) -> DOMViewModel:
        """Build 5-column DOM view from Up/Down order books."""
        user_orders = user_orders or []
        max_depth = 0.0
        rows: Dict[int, DOMRow] = {
            i: DOMRow(price_cent=i, no_price=100 - i, no_depth=0.0, yes_depth=0.0)
            for i in range(1, 100)
        }

        # Up Bids → YES depth
        for price, size in up_book.get('bids', {}).items():
            try:
                cent = self.to_cent(float(price))
                if 1 <= cent <= 99:
                    rows[cent].yes_depth += float(size)
                    max_depth = max(max_depth, rows[cent].yes_depth)
            except (ValueError, TypeError):
                continue

        # Down Bids → NO depth (at complementary price)
        for price, size in down_book.get('bids', {}).items():
            try:
                yes_cent = 100 - self.to_cent(float(price))
                if 1 <= yes_cent <= 99:
                    rows[yes_cent].no_depth += float(size)
                    max_depth = max(max_depth, rows[yes_cent].no_depth)
            except (ValueError, TypeError):
                continue

        # Find best bid/ask
        yes_bids = [p for p, r in rows.items() if r.yes_depth > 0]
        yes_asks = [p for p, r in rows.items() if r.no_depth > 0]
        best_bid_cent = max(yes_bids) if yes_bids else 0
        best_ask_cent = min(yes_asks) if yes_asks else 100

        # Mark spread and best levels
        for cent, row in rows.items():
            row.is_best_bid = (cent == best_bid_cent and best_bid_cent > 0)
            row.is_best_ask = (cent == best_ask_cent and best_ask_cent < 100)
            row.is_inside_spread = (best_bid_cent < cent < best_ask_cent)

        # Map user orders
        for order in user_orders:
            price_cent = order.get('price_cent')
            if price_cent and 1 <= price_cent <= 99:
                rows[price_cent].my_orders.append(UserOrder(
                    order_id=order.get('order_id', ''),
                    size=order.get('size', 0.0),
                    side=order.get('side', 'YES')
                ))

        # Calculate mid price
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