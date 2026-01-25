"""DOM-style ladder trading interface for Polymarket binary markets."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Center, Middle, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Static, Label

from src.ladder.core import LadderCore
from src.ladder.ladder_data import DOMRow as DOMRowData, DOMViewModel

logger = logging.getLogger("LadderUI")

COL_NO_SIZE = 12
COL_NO_PRICE = 5
COL_YES_PRICE = 5
COL_YES_SIZE = 12
COL_MY_ORDERS = 10
TOTAL_WIDTH = COL_NO_SIZE + COL_NO_PRICE + COL_YES_PRICE + COL_YES_SIZE + COL_MY_ORDERS + 4

class Side(Enum):
    YES = "YES"
    NO = "NO"



class VolumeBarRenderer:
    """Renders volume bars using Unicode block characters."""

    BLOCKS_LEFT = " ▏▎▍▌▋▊▉█"
    BLOCKS_RIGHT = " ▕▕▐▐▐▐██"

    def __init__(self, max_width: int = 10):
        self.max_width = max_width

    def render_bar(self, depth: float, max_depth: float, align_right: bool = False) -> str:
        """Render a volume bar. align_right=True for NO side (right-to-left)."""
        if max_depth <= 0 or depth <= 0:
            return " " * self.max_width

        fraction = min(1.0, depth / max_depth)
        total_eighths = int(fraction * self.max_width * 8)

        full_blocks = total_eighths // 8
        remainder = total_eighths % 8

        if align_right:
            bar = ""
            if remainder > 0 and full_blocks < self.max_width:
                bar += self.BLOCKS_RIGHT[remainder]
            bar += "█" * full_blocks
            return bar.rjust(self.max_width)
        else:
            bar = "█" * full_blocks
            if remainder > 0 and full_blocks < self.max_width:
                bar += self.BLOCKS_LEFT[remainder]
            return bar.ljust(self.max_width)



class HelpOverlay(ModalScreen):
    """Modal overlay showing all keyboard shortcuts."""

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
    }

    #help-container {
        width: 70;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #help-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        height: 1;
    }

    #help-content {
        width: 100%;
        height: auto;
        padding: 1;
    }

    .help-section {
        width: 100%;
        margin-bottom: 1;
    }

    .help-section-title {
        width: 100%;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .help-row {
        width: 100%;
        height: auto;
        padding: 0 1;
    }

    .help-key {
        width: 20;
        text-style: bold;
        color: $accent;
    }

    .help-desc {
        width: 1fr;
        color: $text;
    }

    #help-footer {
        width: 100%;
        text-align: center;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("question_mark", "close_help", "Close", show=False),
        Binding("h", "close_help", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with ScrollableContainer(id="help-container"):
                    yield Label("KEYBOARD SHORTCUTS", id="help-title")

                    with Vertical(id="help-content"):
                        # Navigation Section
                        with Vertical(classes="help-section"):
                            yield Label("NAVIGATION", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("↑ / ↓ / k / j", classes="help-key")
                                yield Label("Move cursor up/down", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("m", classes="help-key")
                                yield Label("Center view on mid-price", classes="help-desc")

                        # Trading Section
                        with Vertical(classes="help-section"):
                            yield Label("TRADING - MARKET ORDERS", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("y", classes="help-key")
                                yield Label("Place market BUY YES order", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("n", classes="help-key")
                                yield Label("Place market BUY NO order", classes="help-desc")

                        with Vertical(classes="help-section"):
                            yield Label("TRADING - LIMIT ORDERS", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("t", classes="help-key")
                                yield Label("Place limit BUY YES at cursor", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("b", classes="help-key")
                                yield Label("Place limit BUY NO at cursor", classes="help-desc")

                        # Order Management Section
                        with Vertical(classes="help-section"):
                            yield Label("ORDER MANAGEMENT", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("c", classes="help-key")
                                yield Label("Cancel ALL open orders", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("x", classes="help-key")
                                yield Label("Cancel orders at cursor price", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("f", classes="help-key")
                                yield Label("Flatten all positions (market sell)", classes="help-desc")

                        # Size Adjustment Section
                        with Vertical(classes="help-section"):
                            yield Label("SIZE ADJUSTMENT", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("+ / =", classes="help-key")
                                yield Label("Increase order size", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("-", classes="help-key")
                                yield Label("Decrease order size", classes="help-desc")

                        # System Section
                        with Vertical(classes="help-section"):
                            yield Label("SYSTEM", classes="help-section-title")
                            with Horizontal(classes="help-row"):
                                yield Label("? / h", classes="help-key")
                                yield Label("Show this help overlay", classes="help-desc")
                            with Horizontal(classes="help-row"):
                                yield Label("q", classes="help-key")
                                yield Label("Quit application", classes="help-desc")

                    yield Label("Press ESC, ? or h to close", id="help-footer")

    def action_close_help(self) -> None:
        self.dismiss()


class OrderConfirmationDialog(ModalScreen):
    """Modal dialog for confirming order placement."""

    DEFAULT_CSS = """
    OrderConfirmationDialog {
        align: center middle;
    }

    #confirmation-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #dialog-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #order-details {
        width: 100%;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: solid $primary;
    }

    .detail-row {
        width: 100%;
        height: auto;
        padding: 0 1;
    }

    .detail-label {
        width: 15;
        text-style: bold;
        color: $text-muted;
    }

    .detail-value {
        width: 1fr;
        text-style: bold;
    }

    .detail-value-yes {
        color: #00ff00;
    }

    .detail-value-no {
        color: #ff0000;
    }

    #confirmation-prompt {
        width: 100%;
        text-align: center;
        color: $text;
        margin: 1 0;
    }

    #key-hints {
        width: 100%;
        text-align: center;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, order_type: str, side: str, price: Optional[int], order_size: float):
        super().__init__()
        self.order_type = order_type  # "Market" or "Limit"
        self.side = side  # "YES" or "NO"
        self.price = price  # Price in cents (None for market orders)
        self.order_size = order_size

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Container(id="confirmation-dialog"):
                    yield Label("ORDER CONFIRMATION", id="dialog-title")

                    with Vertical(id="order-details"):
                        with Horizontal(classes="detail-row"):
                            yield Label("Type:", classes="detail-label")
                            yield Label(self.order_type, classes="detail-value")

                        with Horizontal(classes="detail-row"):
                            yield Label("Side:", classes="detail-label")
                            side_class = "detail-value-yes" if self.side == "YES" else "detail-value-no"
                            yield Label(self.side, classes=f"detail-value {side_class}")

                        if self.price is not None:
                            with Horizontal(classes="detail-row"):
                                yield Label("Price:", classes="detail-label")
                                yield Label(f"{self.price}c (${self.price/100:.2f})", classes="detail-value")

                        with Horizontal(classes="detail-row"):
                            yield Label("Size:", classes="detail-label")
                            yield Label(f"${self.order_size:.2f}", classes="detail-value")

                    yield Label("Are you sure you want to place this order?", id="confirmation-prompt")
                    yield Label("Press ENTER to confirm or ESC to cancel", id="key-hints")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)



class DOMRowWidget(Horizontal):
    """A single row in the 5-column DOM display."""

    DEFAULT_CSS = """
    DOMRowWidget {
        height: 1;
        width: 100%;
    }

    DOMRowWidget .no-size-col {
        width: 12;
        min-width: 12;
        max-width: 12;
        color: #ff6666;
        padding: 0;
        margin: 0;
        content-align: left middle;
    }

    DOMRowWidget .no-price-col {
        width: 5;
        min-width: 5;
        max-width: 5;
        text-align: center;
        color: #ff4444;
        padding: 0;
        margin: 0;
    }

    DOMRowWidget .yes-price-col {
        width: 5;
        min-width: 5;
        max-width: 5;
        text-align: center;
        color: #44ff44;
        padding: 0;
        margin: 0;
    }

    DOMRowWidget .yes-size-col {
        width: 12;
        min-width: 12;
        max-width: 12;
        color: #66ff66;
        padding: 0;
        margin: 0;
        content-align: left middle;
    }

    DOMRowWidget .my-orders-col {
        width: 10;
        text-align: left;
        color: $warning;
        text-style: bold;
    }

    /* Spread highlighting */
    DOMRowWidget.spread-row {
        background: #1a1a2e;
    }

    /* Best bid/ask highlighting */
    DOMRowWidget.best-bid-row .yes-price-col {
        text-style: bold;
        color: #00ff00;
    }

    DOMRowWidget.best-ask-row .no-price-col {
        text-style: bold;
        color: #ff0000;
    }

    /* Cursor highlighting */
    DOMRowWidget.cursor-row {
        background: $accent 30%;
    }

    DOMRowWidget.cursor-row .no-price-col,
    DOMRowWidget.cursor-row .yes-price-col {
        text-style: bold;
    }
    """

    def __init__(self, price_cent: int):
        super().__init__()
        self.price_cent = price_cent
        self.row_id = f"row_{price_cent}"

    def compose(self) -> ComposeResult:
        yield Static("", classes="no-size-col", id=f"no-size-{self.row_id}")
        yield Static(f"{100 - self.price_cent:2d}", classes="no-price-col", id=f"no-px-{self.row_id}")
        yield Static(f"{self.price_cent:2d}", classes="yes-price-col", id=f"yes-px-{self.row_id}")
        yield Static("", classes="yes-size-col", id=f"yes-size-{self.row_id}")
        yield Static("", classes="my-orders-col", id=f"orders-{self.row_id}")

    def update_data(
        self,
        no_bar: str,
        yes_bar: str,
        my_orders_display: str,
        is_cursor: bool,
        is_spread: bool,
        is_best_bid: bool,
        is_best_ask: bool
    ):
        try:
            self.query_one(f"#no-size-{self.row_id}", Static).update(no_bar)
            self.query_one(f"#yes-size-{self.row_id}", Static).update(yes_bar)
            self.query_one(f"#orders-{self.row_id}", Static).update(my_orders_display)
        except Exception:
            pass

        self.set_class(is_cursor, "cursor-row")
        self.set_class(is_spread, "spread-row")
        self.set_class(is_best_bid, "best-bid-row")
        self.set_class(is_best_ask, "best-ask-row")



class PolyTerm(App):
    """DOM-style ladder trading terminal for Polymarket."""

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

    #dom-container {
        align: center middle;
        height: 1fr;
        margin: 1 0;
        width: 100%;
    }

    #dom-wrapper {
        width: 48;
        align: center middle;
    }

    #header-labels {
        height: 1;
        width: 48;
        background: $primary-darken-2;
    }

    #header-labels .header-lbl {
        text-style: bold;
        text-align: center;
    }

    #h-no-size { width: 12; color: #ff4444; }
    #h-no-px { width: 5; }
    #h-yes-px { width: 5; }
    #h-yes-size { width: 12; color: #44ff44; }
    #h-orders { width: 10; color: $warning; }

    #dom-scroll {
        width: 48;
    }

    #stats-bar {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 2;
        border-top: tall $primary;
    }

    #stats-content {
        width: 1fr;
        align: center middle;
    }

    .stat-val {
        margin-right: 4;
    }

    #help-button {
        width: auto;
        min-width: 8;
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $text;
        text-style: bold;
        text-align: center;
    }

    #help-button:hover {
        background: $accent;
        color: $text;
        text-style: bold reverse;
    }

    #help-button:focus {
        background: $accent;
        color: $text;
    }
    """

    BINDINGS = [
        # Navigation
        Binding("up", "move_cursor(-1)", "Up", show=False),
        Binding("down", "move_cursor(1)", "Down", show=False),
        Binding("k", "move_cursor(-1)", "Up", show=False),
        Binding("j", "move_cursor(1)", "Down", show=False),
        Binding("m", "center_view", "Mid", show=False),

        # Trading - Market Orders
        Binding("y", "place_market_order('YES')", "Mkt YES", show=False),
        Binding("n", "place_market_order('NO')", "Mkt NO", show=False),

        # Trading - Limit Orders (Flick workflow)
        Binding("t", "place_limit_order('YES')", "Lmt YES", show=False),
        Binding("b", "place_limit_order('NO')", "Lmt NO", show=False),

        # Order Management
        Binding("c", "cancel_all", "Cancel All", show=False),
        Binding("x", "cancel_at_cursor", "Cancel@", show=False),
        Binding("f", "flatten", "Flatten", show=False),

        # Size Adjustment
        Binding("+", "adj_size(1)", "+Size", show=False),
        Binding("=", "adj_size(1)", "+Size", show=False),
        Binding("equals", "adj_size(1)", "+Size", show=False),
        Binding("-", "adj_size(-1)", "-Size", show=False),
        Binding("minus", "adj_size(-1)", "-Size", show=False),

        # Help and System
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("h", "show_help", "Help", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    # Reactive properties
    selected_price_cent = reactive(50)
    order_size = reactive(1)
    balance = reactive(0.0)

    def __init__(self, fb_core=None):
        super().__init__()
        self.ladder_core = LadderCore(fb_core)
        self.rows: Dict[int, DOMRowWidget] = {}
        self.bar_renderer = VolumeBarRenderer(max_width=12)  # Match column width
        self.title = "POLYMARKET DOM LADDER"
        self.sub_title = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Loading market...", id="market-name")

        with Container(id="dom-container"):
            with Container(id="dom-wrapper"):
                # 5-column header
                with Horizontal(id="header-labels"):
                    yield Label("NO Size", classes="header-lbl", id="h-no-size")
                    yield Label("NO", classes="header-lbl", id="h-no-px")
                    yield Label("YES", classes="header-lbl", id="h-yes-px")
                    yield Label("YES Size", classes="header-lbl", id="h-yes-size")
                    yield Label("Orders", classes="header-lbl", id="h-orders")

                # Scrollable ladder (99 down to 1, high YES prices at top)
                self.scroll_view = ScrollableContainer(id="dom-scroll")
                with self.scroll_view:
                    for i in range(99, 0, -1):
                        row = DOMRowWidget(i)
                        self.rows[i] = row
                        yield row

        with Horizontal(id="stats-bar"):
            with Horizontal(id="stats-content"):
                yield Label("POSITION: 0", id="pos-display", classes="stat-val")
                yield Label("BALANCE: $0.00", id="bal-display", classes="stat-val")
                yield Label("SIZE: 1", id="size-display", classes="stat-val")
            yield Static("? Help", id="help-button")

    async def on_mount(self):
        self.ladder_core.set_market_update_callback(self._on_market_update)

        # Initialize market and start WebSocket connection
        try:
            await self.ladder_core.fb.start_rtds()
            await asyncio.sleep(1.0)

            market = await self.ladder_core.fb.connector.get_active_market()
            if market:
                success = await self.ladder_core.fb.market_manager.set_market(market)
                if success:
                    market_name = market.get('question') or market.get('title') or 'Market'
                    starts = market.get('start_date', '')
                    ends = market.get('end_date', 'N/A')
                    strike = market.get('strike', 'Loading')
                    self.ladder_core._on_market_update(strike, ends, market_name, starts)

                    await self.ladder_core.fb.ws_manager.start()
                    await asyncio.sleep(2.0)
                else:
                    self.notify("Warning: Failed to set market", severity="warning")
            else:
                self.notify("Warning: No active market found", severity="warning")
        except Exception as e:
            self.notify(f"Warning: Could not initialize market: {e}", severity="warning")

        # Update intervals
        self.set_interval(0.1, self.update_ladder)  # 10 FPS
        self.set_interval(2.0, self.update_balance)
        self.set_interval(5.0, self._update_market_status)
        self.set_interval(0.2, self._update_countdown)

        # Center on mid price after initial load
        self.call_after_refresh(self._center_initial)
        asyncio.create_task(self.update_balance())

    def _center_initial(self):
        """Center on initial price after mount."""
        self.selected_price_cent = 50
        self._scroll_to_cursor()

    def _on_market_update(self, market_name: str, starts: str, ends: str) -> None:
        """Handle market update - update header title and time range."""
        try:
            self.title = market_name if market_name else "POLYMARKET DOM LADDER"
            time_display = self._format_time_range(starts, ends)
            self.query_one("#market-name", Label).update(time_display)
        except Exception as e:
            logger.debug(f"Error updating market display: {e}")

    def _format_time_range(self, starts: str, ends: str) -> str:
        """Format start and end times into display string."""
        try:
            import pandas as pd

            start_dt = pd.Timestamp(starts)
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize('UTC')
            start_dt = start_dt.tz_convert('US/Eastern')

            end_dt = pd.Timestamp(ends)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            end_dt = end_dt.tz_convert('US/Eastern')

            if start_dt.date() == end_dt.date():
                date_str = start_dt.strftime('%B %d')
                start_time = start_dt.strftime('%I:%M%p').lstrip('0')
                end_time = end_dt.strftime('%I:%M%p').lstrip('0')
                return f"{date_str}, {start_time}-{end_time} ET"
            else:
                start_str = start_dt.strftime('%B %d %I:%M%p').lstrip('0')
                end_str = end_dt.strftime('%B %d %I:%M%p').lstrip('0')
                return f"{start_str} - {end_str} ET"
        except Exception:
            return ends if ends else "Loading..."

    async def update_balance(self):
        """Update wallet balance display."""
        try:
            balance = await self.ladder_core.fb.connector.get_usdc_balance()
            self.balance = balance
            self.query_one("#bal-display").update(f"BALANCE: ${balance:.2f}")
        except Exception as e:
            logger.debug(f"Error updating balance: {e}")

    async def _update_market_status(self):
        """Poll for new markets and handle transitions."""
        try:
            if self.ladder_core and self.ladder_core.fb:
                await self.ladder_core.fb.update_market_status()
        except Exception as e:
            logger.debug(f"Error updating market status: {e}")

    async def _update_countdown(self):
        """Update countdown timer and check for market expiry."""
        try:
            if self.ladder_core and self.ladder_core.fb:
                await self.ladder_core.fb.update_countdown()
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}")

    def update_ladder(self):
        """Refresh DOM UI."""
        view_model: DOMViewModel = self.ladder_core.get_dom_view_model()

        for price_cent, row_widget in self.rows.items():
            dom_row = view_model.rows.get(price_cent)
            if not dom_row:
                continue

            no_bar = self.bar_renderer.render_bar(dom_row.no_depth, view_model.max_depth, align_right=True)
            yes_bar = self.bar_renderer.render_bar(dom_row.yes_depth, view_model.max_depth, align_right=False)

            orders_display = self._format_orders(dom_row.my_orders)
            if self.ladder_core.is_filled(price_cent) and not orders_display:
                orders_display = "[FILL]"

            row_widget.update_data(
                no_bar=no_bar,
                yes_bar=yes_bar,
                my_orders_display=orders_display,
                is_cursor=(price_cent == self.selected_price_cent),
                is_spread=dom_row.is_inside_spread,
                is_best_bid=dom_row.is_best_bid,
                is_best_ask=dom_row.is_best_ask
            )

    def _format_orders(self, orders: List) -> str:
        if not orders:
            return ""
        parts = [f"[{int(o.size)}{'Y' if o.side == 'YES' else 'N'}]" for o in orders]
        return "".join(parts)[:10]

    def _scroll_to_cursor(self):
        if self.selected_price_cent in self.rows:
            self.scroll_view.scroll_to_widget(self.rows[self.selected_price_cent], animate=False)

    # Actions

    def action_move_cursor(self, delta: int):
        self.selected_price_cent = max(1, min(99, self.selected_price_cent - delta))
        self._scroll_to_cursor()

    def action_center_view(self):
        view_model = self.ladder_core.get_dom_view_model()
        self.selected_price_cent = view_model.mid_price_cent
        self._scroll_to_cursor()
        self.notify(f"Centered on {view_model.mid_price_cent}c")

    @work
    async def action_place_market_order(self, side_str: str):
        confirmed = await self.push_screen_wait(
            OrderConfirmationDialog("Market", side_str, None, float(self.order_size))
        )
        if not confirmed:
            self.notify("Order cancelled", severity="information")
            return

        order_id = await self.ladder_core.place_market_order(float(self.order_size), side_str)
        if order_id:
            self.notify(f"Market {side_str} placed (ID: {order_id[:10]}...)")
        else:
            self.notify(f"Market {side_str} failed", severity="error")

    @work
    async def action_place_limit_order(self, side_str: str):
        price_cent = self.selected_price_cent
        confirmed = await self.push_screen_wait(
            OrderConfirmationDialog("Limit", side_str, price_cent, float(self.order_size))
        )
        if not confirmed:
            self.notify("Order cancelled", severity="information")
            return

        order_id = await self.ladder_core.place_limit_order(price_cent, float(self.order_size), side_str)
        if order_id:
            self.notify(f"Limit {side_str} @ {price_cent}c placed")
        else:
            self.notify(f"Limit {side_str} @ {price_cent}c failed", severity="error")

    def action_adj_size(self, delta: int):
        self.order_size = max(1, self.order_size + delta)
        self.query_one("#size-display").update(f"SIZE: {self.order_size}")

    def action_show_help(self):
        self.push_screen(HelpOverlay())

    def on_click(self, event) -> None:
        if hasattr(event, 'widget') and event.widget.id == "help-button":
            self.action_show_help()

    async def action_cancel_all(self):
        canceled_count = await self.ladder_core.cancel_all_orders()
        if canceled_count > 0:
            self.notify(f"Cancelled {canceled_count} order(s)", severity="warning")
        else:
            self.notify("No open orders to cancel", severity="information")

    async def action_cancel_at_cursor(self):
        price_cent = self.selected_price_cent
        canceled_count = await self.ladder_core.cancel_all_at_price(price_cent)
        if canceled_count > 0:
            self.notify(f"Cancelled {canceled_count} order(s) at {price_cent}c")
        else:
            self.notify("No orders at cursor", severity="information")

    async def action_flatten(self):
        async def flatten_and_notify():
            try:
                await self.ladder_core.fb.flatten_all()
                self.notify("Flattening all positions...", severity="warning")
            except Exception as e:
                logger.error(f"Flatten failed: {e}")
                self.notify(f"Flatten failed: {e}", severity="error")
        asyncio.create_task(flatten_and_notify())


if __name__ == "__main__":
    PolyTerm().run()
