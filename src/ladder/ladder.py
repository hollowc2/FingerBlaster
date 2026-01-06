import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Label

from src.ladder.core import LadderCore

logger = logging.getLogger("LadderUI")

# --- 1. Constants & Models ---

TICK_SIZE = 0.01
MAX_EXPOSURE = 100.0

class Side(Enum):
    YES = "YES"
    NO = "NO"

@dataclass
class Order:
    price: float
    side: Side
    size: int = 1

# --- 2. UI Components ---

class LadderCell(Static):
    """A single interactive cell (Buy YES or Buy NO) in the ladder."""
    
    class Clicked(Message):
        """Custom message emitted when a cell is clicked."""
        def __init__(self, cell: "LadderCell") -> None:
            self.cell = cell
            super().__init__()

    def __init__(self, side: Side, price: float, **kwargs):
        super().__init__(**kwargs)
        self.side = side
        self.price = price
        # Make cells non-focusable to prevent accidental keyboard triggers
        self.can_focus = False

    def on_click(self) -> None:
        """Handle click and post message to parent."""
        # Only trigger on actual mouse clicks, not programmatic updates
        self.post_message(self.Clicked(self))

class LadderRow(Horizontal):
    """A single row representing one price tick."""
    DEFAULT_CSS = """
    LadderRow {
        height: 1;
        align: center middle;
    }
    .price-col {
        width: 12;
        text-align: center;
        background: $surface;
        color: $text;
        border-left: solid $primary;
        border-right: solid $primary;
    }
    .yes-col { width: 15; color: #00ff00; text-align: left; padding-left: 1; }
    .no-col { width: 15; color: #ff0000; text-align: right; padding-right: 1; }
    .mid-price-row { background: $accent 20%; }
    .user-order { background: $warning 40%; text-style: bold; color: $text; }
    .selected-row { border: double $accent; }
    .pending-order { background: $warning 20%; text-style: italic; }
    """

    def __init__(self, price: float):
        super().__init__()
        self.price = price
        self.safe_id = str(price).replace('.', '_')

    def compose(self) -> ComposeResult:
        yield LadderCell(Side.NO, self.price, classes="no-col", id=f"no-{self.safe_id}")
        yield Label(f"{self.price:.2f}", classes="price-col", id=f"price-{self.safe_id}")
        yield LadderCell(Side.YES, self.price, classes="yes-col", id=f"yes-{self.safe_id}")

    def update_data(
        self,
        mid_price: float,
        yes_qty: int,
        no_qty: int,
        user_order: Optional[Order],
        is_selected: bool,
        is_pending: bool = False
    ):
        self.set_class(self.price == mid_price, "mid-price-row")
        self.set_class(is_selected, "selected-row")
        
        no_cell = self.query_one(f"#no-{self.safe_id}", LadderCell)
        yes_cell = self.query_one(f"#yes-{self.safe_id}", LadderCell)

        # Indicators for pending or active orders
        indicator = "⏳" if is_pending else ("●" if user_order else "")

        # Update NO column (Asks in YES terms)
        no_display = f"{no_qty} {indicator}" if no_qty > 0 or indicator else ""
        no_cell.update(no_display)
        no_cell.set_class(is_pending, "pending-order")
        no_cell.set_class(user_order is not None and user_order.side == Side.NO, "user-order")
        # Ensure cell remains non-focusable after updates to prevent accidental keyboard triggers
        if hasattr(no_cell, 'can_focus'):
            no_cell.can_focus = False

        # Update YES column (Bids in YES terms)
        yes_display = f"{indicator} {yes_qty}" if yes_qty > 0 or indicator else ""
        yes_cell.update(yes_display)
        yes_cell.set_class(is_pending, "pending-order")
        yes_cell.set_class(user_order is not None and user_order.side == Side.YES, "user-order")
        # Ensure cell remains non-focusable after updates to prevent accidental keyboard triggers
        if hasattr(yes_cell, 'can_focus'):
            yes_cell.can_focus = False

# --- 3. Main Application ---

class PolyTerm(App):
    TITLE = "FINGERBLASTER YES/NO LADDER"
    CSS = """
    #market-name {
        height: 1;
        width: 100%;
        text-align: center;
        text-style: bold;
        background: $primary-darken-3;
        color: $text;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    #ladder-container {
        align: center middle;
        height: 1fr;
        margin: 1 0;
        width: 100%;
    }
    #ladder-wrapper {
        width: 42;
        align: center middle;
    }
    #header-labels {
        height: 1;
        width: 42;
        background: $primary-darken-2;
    }
    #ladder-scroll {
        width: 42;
    }
    .header-lbl { 
        width: 15; 
        text-align: center; 
        text-style: bold;
        align: center middle;
    }
    .price-lbl { 
        width: 12; 
        text-align: center;
        align: center middle;
    }
    
    #stats-bar {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 2;
        border-top: tall $primary;
        align: center middle;
    }
    .stat-val { margin-right: 4; }
    """

    BINDINGS = [
        Binding("up", "move_cursor(-1)", "Up", show=True),
        Binding("down", "move_cursor(1)", "Down", show=True),
        Binding("y", "place_market_order('YES')", "Market YES"),
        Binding("n", "place_market_order('NO')", "Market NO"),
        Binding("t", "place_limit_order('YES')", "Limit YES"),
        Binding("b", "place_limit_order('NO')", "Limit NO"),
        Binding("plus", "adj_size(1)", "Size +1"),
        Binding("equals", "adj_size(1)", "Size +1"),
        Binding("minus", "adj_size(-1)", "Size -1"),
        Binding("c", "cancel_at_cursor", "Cancel Price"),
        Binding("f", "flatten", "Flatten All"),
        Binding("q", "quit", "Quit"),
    ]

    selected_price = reactive(0.50)
    order_size = reactive(1)
    balance = reactive(0.0)

    def __init__(self, fb_core=None):
        super().__init__()
        self.ladder_core = LadderCore(fb_core)
        self.rows: Dict[float, LadderRow] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Loading market...", id="market-name")
        with Container(id="ladder-container"):
            with Container(id="ladder-wrapper"):
                with Horizontal(id="header-labels"):
                    yield Label("BUY NO", classes="header-lbl")
                    yield Label("PRICE", classes="price-lbl")
                    yield Label("BUY YES", classes="header-lbl")
                
                self.scroll_view = ScrollableContainer(id="ladder-scroll")
                with self.scroll_view:
                    for i in range(99, 0, -1):
                        p = round(i * 0.01, 2)
                        row = LadderRow(p)
                        self.rows[p] = row
                        yield row
        
        with Horizontal(id="stats-bar"):
            yield Label(f"POSITION: 0", id="pos-display", classes="stat-val")
            yield Label(f"BALANCE: $0.00", id="bal-display", classes="stat-val")
            yield Label(f"SIZE: 1", id="size-display", classes="stat-val")
        yield Footer()

    async def on_mount(self):
        self.ladder_core.set_market_update_callback(self._on_market_update)
        
        # Initialize market and start WebSocket connection
        try:
            # Start RTDS for BTC price (if needed)
            await self.ladder_core.fb.start_rtds()
            await asyncio.sleep(1.0)  # Let RTDS initialize
            
            # Get active market and set it
            market = await self.ladder_core.fb.connector.get_active_market()
            if market:
                success = await self.ladder_core.fb.market_manager.set_market(market)
                if success:
                    # Trigger market update callback
                    market_name = market.get('question') or market.get('title') or 'Market'
                    ends = market.get('end_date', 'N/A')
                    strike = market.get('strike', 'Loading')
                    self.ladder_core._on_market_update(strike, ends, market_name)
                    
                    # Start WebSocket connection
                    await self.ladder_core.fb.ws_manager.start()
                    await asyncio.sleep(2.0)  # Let WebSocket connect and receive initial data
                else:
                    self.notify("Warning: Failed to set market", severity="warning")
            else:
                self.notify("Warning: No active market found", severity="warning")
        except Exception as e:
            self.notify(f"Warning: Could not initialize market: {e}", severity="warning")
        
        # 10 FPS UI update loop
        self.set_interval(0.1, self.update_ladder)
        # Update balance every 2 seconds
        self.set_interval(2.0, self.update_balance)
        self.call_after_refresh(self.center_on_price, 0.50)
        
        # Initial balance fetch
        asyncio.create_task(self.update_balance())
    
    def _on_market_update(self, market_display_text: str) -> None:
        try:
            self.query_one("#market-name", Label).update(market_display_text)
        except Exception:
            pass

    async def update_balance(self):
        """Update wallet balance display."""
        try:
            balance = await self.ladder_core.fb.connector.get_usdc_balance()
            self.balance = balance
            self.query_one("#bal-display").update(f"BALANCE: ${balance:.2f}")
        except Exception as e:
            logger.debug(f"Error updating balance: {e}")
    
    def update_ladder(self):
        """Main refresh loop for the UI."""
        data = self.ladder_core.get_view_model()
        if not data:
            # If no data, initialize all rows with zeros
            for p_float, row in self.rows.items():
                row.update_data(
                    mid_price=0.50,
                    yes_qty=0,
                    no_qty=0,
                    user_order=None,
                    is_selected=(p_float == self.selected_price),
                    is_pending=False
                )
            return

        # Calculate Mid Price for visual centering
        yes_bids = [p for p, v in data.items() if v.get('yes_bid', 0) > 0]
        best_bid_cent = max(yes_bids) if yes_bids else 50
        mid_price = best_bid_cent / 100.0

        # Update all rows - iterate through all prices to ensure nothing is missed
        for p_float in self.rows.keys():
            # Find corresponding data entry (data uses cents as keys)
            p_cent = int(round(p_float * 100))
            item = data.get(p_cent, {'price': p_float, 'yes_bid': 0.0, 'yes_ask': 0.0, 'my_size': 0.0})
            
            # Check for active and pending orders at this tick
            my_size = item.get('my_size', 0.0)
            is_pending = self.ladder_core.is_pending(p_cent)
            user_ord = Order(p_float, Side.YES, int(my_size)) if my_size > 0 else None

            self.rows[p_float].update_data(
                mid_price=mid_price,
                yes_qty=int(item.get('yes_bid', 0)),
                no_qty=int(item.get('yes_ask', 0)),
                user_order=user_ord,
                is_selected=(p_float == self.selected_price),
                is_pending=is_pending
            )

    def center_on_price(self, price: float):
        if price in self.rows:
            self.scroll_view.scroll_to_widget(self.rows[price], animate=False)

    def action_move_cursor(self, delta: int):
        self.selected_price = round(max(0.01, min(0.99, self.selected_price + (delta * TICK_SIZE))), 2)
        self.center_on_price(self.selected_price)

    def action_place_market_order(self, side_str: str):
        """Place a market order (Y for YES, N for NO)."""
        async def place_and_notify():
            order_id = await self.ladder_core.place_market_order(float(self.order_size), side_str)
            if order_id:
                self.notify(f"✓ Market {side_str} order placed (ID: {order_id[:10]}...)")
            else:
                self.notify(f"✗ Market {side_str} order failed", severity="error")
        
        asyncio.create_task(place_and_notify())
        self.notify(f"Placing market {side_str} order...")
    
    def action_place_limit_order(self, side_str: str):
        """Place a limit order at current price level (T for YES, B for NO)."""
        price_cent = int(self.selected_price * 100)
        
        async def place_and_notify():
            order_id = await self.ladder_core.place_limit_order(price_cent, float(self.order_size), side_str)
            if order_id:
                self.notify(f"✓ Limit {side_str} @ {self.selected_price:.2f} (ID: {order_id[:10]}...)")
            else:
                self.notify(f"✗ Limit {side_str} @ {self.selected_price:.2f} failed", severity="error")
        
        asyncio.create_task(place_and_notify())
        self.notify(f"Placing limit {side_str} @ {self.selected_price:.2f}...")

    def action_adj_size(self, delta: int):
        self.order_size = max(1, self.order_size + delta)
        self.query_one("#size-display").update(f"SIZE: {self.order_size}")

    async def action_cancel_at_cursor(self):
        price_cent = int(self.selected_price * 100)
        canceled_count = await self.ladder_core.cancel_all_at_price(price_cent)
        if canceled_count > 0:
            self.notify(f"Cancelled {canceled_count} order(s) at {self.selected_price:.2f}")
        else:
            self.notify("No orders found, nothing to cancel", severity="information")

    async def action_flatten(self):
        """Flatten all positions by selling all tokens at market price."""
        async def flatten_and_notify():
            try:
                await self.ladder_core.fb.flatten_all()
                self.notify("✓ Flattening all positions...", severity="warning")
            except Exception as e:
                logger.error(f"Flatten failed: {e}")
                self.notify(f"✗ Flatten failed: {e}", severity="error")
        
        asyncio.create_task(flatten_and_notify())
        self.notify("Flattening all positions...")

    def on_ladder_cell_clicked(self, event: LadderCell.Clicked):
        """Handle cell click - places limit order at clicked price."""
        # Only place order on explicit user click, not programmatic triggers
        cell = event.cell
        self.selected_price = cell.price
        # Log for debugging
        logger.debug(f"Cell clicked: {cell.side.value} @ {cell.price:.2f}")
        self.action_place_limit_order(cell.side.value)

if __name__ == "__main__":
    PolyTerm().run()