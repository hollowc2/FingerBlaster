"""Data normalization for Polymarket ladder UI."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("LadderData")

# Constants
PRICE_SCALE = 100  # Convert price (0-1) to cents (0-100)
PRICE_RANGE = (1, 100)  # range(1, 100) produces 1..99 for all valid cent prices


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


def price_to_cent(price: float) -> int:
    """Convert price (0-1.0) to cents (0-100)."""
    return int(round(price * PRICE_SCALE))


class LadderDataManager:
    """Merges Up/Down token books into a unified YES ladder."""

    def build_ladder_data(self, up_book: Dict, down_book: Dict) -> Dict[int, Dict]:
        """Build ladder: YES Bid = Up Bids, YES Ask = Down Bids at complementary price."""
        ladder = {i: {"price": i / PRICE_SCALE, "yes_bid": 0.0, "yes_ask": 0.0, "my_size": 0.0}
                  for i in range(*PRICE_RANGE)}

        # Up Bids → YES Bids
        self._accumulate_bids(up_book.get('bids', {}), ladder, is_complement=False)

        # Down Bids → YES Asks (at complementary price: 100 - price)
        self._accumulate_bids(down_book.get('bids', {}), ladder, is_complement=True)

        return ladder

    def _accumulate_bids(self, bids: Dict, ladder: Dict[int, Dict], is_complement: bool) -> None:
        """Add bid volumes to ladder, with optional price complement for Down tokens."""
        for price, size in bids.items():
            try:
                cent = price_to_cent(float(price))
                if is_complement:
                    cent = PRICE_SCALE - cent
                if PRICE_RANGE[0] <= cent < PRICE_RANGE[1]:  # Use < for exclusive upper bound
                    field = "yes_ask" if is_complement else "yes_bid"
                    ladder[cent][field] += float(size)
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid bid price={price} size={size}: {e}")
                continue

    def build_dom_data(
        self,
        up_book: Dict,
        down_book: Dict,
        user_orders: Optional[List[Dict]] = None
    ) -> DOMViewModel:
        """Build 5-column DOM view from Up/Down order books."""
        user_orders = user_orders or []
        rows: Dict[int, DOMRow] = {
            i: DOMRow(price_cent=i, no_price=PRICE_SCALE - i, no_depth=0.0, yes_depth=0.0)
            for i in range(*PRICE_RANGE)
        }

        # Up Bids → YES depth, Down Bids → NO depth (complementary)
        max_depth = self._populate_order_depths(up_book.get('bids', {}), down_book.get('bids', {}), rows)

        # Find best bid/ask
        best_bid_cent, best_ask_cent = self._find_best_levels(rows)

        # Mark spread and best levels
        for cent, row in rows.items():
            row.is_best_bid = (cent == best_bid_cent and best_bid_cent > 0)
            row.is_best_ask = (cent == best_ask_cent and best_ask_cent < PRICE_SCALE)
            row.is_inside_spread = (best_bid_cent < cent < best_ask_cent)

        # Map user orders
        self._populate_user_orders(user_orders, rows)

        # Calculate mid price
        mid_price_cent = self._calculate_mid_price(best_bid_cent, best_ask_cent)

        return DOMViewModel(
            rows=rows,
            max_depth=max_depth,
            best_bid_cent=best_bid_cent,
            best_ask_cent=best_ask_cent,
            mid_price_cent=mid_price_cent
        )

    def _populate_order_depths(self, up_bids: Dict, down_bids: Dict, rows: Dict[int, DOMRow]) -> float:
        """Populate order depths from Up and Down books, return max depth."""
        max_depth = 0.0

        # Up Bids → YES depth
        for price, size in up_bids.items():
            try:
                cent = price_to_cent(float(price))
                if PRICE_RANGE[0] <= cent < PRICE_RANGE[1]:  # Use < for exclusive upper bound
                    rows[cent].yes_depth += float(size)
                    max_depth = max(max_depth, rows[cent].yes_depth)
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid up_bid price={price} size={size}: {e}")
                continue

        # Down Bids → NO depth at complementary price
        for price, size in down_bids.items():
            try:
                yes_cent = PRICE_SCALE - price_to_cent(float(price))
                if PRICE_RANGE[0] <= yes_cent < PRICE_RANGE[1]:  # Use < for exclusive upper bound
                    rows[yes_cent].no_depth += float(size)
                    max_depth = max(max_depth, rows[yes_cent].no_depth)
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid down_bid price={price} size={size}: {e}")
                continue

        return max_depth

    def _find_best_levels(self, rows: Dict[int, DOMRow]) -> tuple:
        """Find best bid and ask prices from rows."""
        yes_bids = [p for p, r in rows.items() if r.yes_depth > 0]
        yes_asks = [p for p, r in rows.items() if r.no_depth > 0]
        best_bid = max(yes_bids) if yes_bids else 0
        best_ask = min(yes_asks) if yes_asks else PRICE_SCALE
        return best_bid, best_ask

    def _populate_user_orders(self, user_orders: List[Dict], rows: Dict[int, DOMRow]) -> None:
        """Add user orders to corresponding rows."""
        for order in user_orders:
            price_cent = order.get('price_cent')
            if price_cent and PRICE_RANGE[0] <= price_cent < PRICE_RANGE[1]:  # Use < for exclusive upper bound
                rows[price_cent].my_orders.append(UserOrder(
                    order_id=order.get('order_id', ''),
                    size=order.get('size', 0.0),
                    side=order.get('side', 'YES')
                ))

    def _calculate_mid_price(self, best_bid: int, best_ask: int) -> int:
        """Calculate mid price from best bid/ask."""
        if best_bid > 0 and best_ask < PRICE_SCALE:
            return (best_bid + best_ask) // 2
        return best_bid if best_bid > 0 else (best_ask if best_ask < PRICE_SCALE else 50)