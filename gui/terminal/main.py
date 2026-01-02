import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
except NameError:
    project_root = Path.cwd()
    for parent in project_root.parents:
        if (parent / 'src').exists() and (parent / 'src' / 'core.py').exists():
            project_root = parent
            break

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Footer, Digits, Header
from textual.reactive import reactive

# Import backend core
from src.core import FingerBlasterCore
from src.analytics import AnalyticsSnapshot, EdgeDirection

# Configure logging
logging.basicConfig(
    filename='data/finger_blaster.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FingerBlaster.Textual")


def format_edge_bps(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}Mbps"
    elif abs_value >= 1_000:
        return f"{value / 1_000:.1f}Kbps"
    else:
        return f"{value:.1f}bps"


def format_time_remaining(seconds: int) -> str:
    if seconds <= 0:
        return "EXPIRED"
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def format_depth(value: float) -> str:
    if not value or value < 1:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.0f}"


class MetricBox(Vertical):
    """Small inner boxes for Time Left, Delta, and Sigma."""
    def __init__(self, label: str, value: str, id: str):
        super().__init__(id=id)
        self.label = label
        self.value = value

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="metric-label")
        yield Static(self.value, classes="metric-value")


class DataCard(Container):
    """Refactored DataCard to match the bordered trading blocks."""
    def __init__(self, title, color_class="", **kwargs):
        super().__init__(**kwargs)
        self.color_class = color_class
        
        self.depth_widget: Optional[Static] = None
        self.spread_widget: Optional[Static] = None
        self.percentage_widget: Optional[Digits] = None
        self.fv_widget: Optional[Static] = None
        self.edge_widget: Optional[Static] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="card-inner"):
            with Horizontal(classes="card-row"):
                self.depth_widget = Static("DEPTH\n0", classes="card-stat-left")
                self.spread_widget = Static("SPREAD\n0.00/0.00", classes="card-stat-right")
                yield self.depth_widget
                yield self.spread_widget
            
            self.percentage_widget = Digits("50", classes="big-percentage")
            yield self.percentage_widget
            
            with Horizontal(classes="card-row"):
                self.fv_widget = Static("FV\n0.00", classes="card-stat-left")
                self.edge_widget = Static("EDGE\n0.0bps", classes="card-stat-right")
                yield self.fv_widget
                yield self.edge_widget
    
    def update(self, price: float, best_bid: float, best_ask: float, 
                depth: float, fv: Optional[float], edge_bps: Optional[float]):
        percentage = price * 100
        if self.percentage_widget:
            self.percentage_widget.update(f"{percentage:.1f}")
        
        depth_str = format_depth(depth) if depth else "0"
        if self.depth_widget:
            self.depth_widget.update(f"DEPTH\n[#EAB308]{depth_str}[/]")
        
        spread_str = f"{best_bid:.2f}/{best_ask:.2f}"
        if self.spread_widget:
            self.spread_widget.update(f"SPREAD\n[#EAB308]{spread_str}[/]")
        
        fv_str = f"{fv:.2f}" if fv is not None else "N/A"
        if self.fv_widget:
            self.fv_widget.update(f"FV\n[#EAB308]{fv_str}[/]")
        
        if edge_bps is not None:
            edge_str = format_edge_bps(edge_bps)
            # Use yellow if edge is 0, otherwise green/red
            if edge_bps == 0:
                if self.edge_widget:
                    self.edge_widget.update(f"EDGE\n[#EAB308]{edge_str}[/]")
                    self.edge_widget.remove_class("green-text", "red-text")
            else:
                edge_color = "$success" if edge_bps >= 0 else "$error"
                if self.edge_widget:
                    self.edge_widget.update(f"EDGE\n[{edge_color}]{edge_str}[/]")
                    self.edge_widget.remove_class("green-text", "red-text")
        else:
            if self.edge_widget:
                self.edge_widget.update("EDGE\n[#EAB308]N/A[/]")
                self.edge_widget.remove_class("green-text", "red-text")


class TradingTUI(App):
    time_remaining = reactive(0)
    btc_price = reactive(0.0)
    strike_price = reactive("0.0")
    delta_val = reactive(0.0)
    sigma_label = reactive("0.0σ")
    market_name = reactive("FINGER BLASTER")
    title = reactive("FINGER BLASTER")
    
    yes_price = 0.5
    no_price = 0.5
    best_bid = 0.5
    best_ask = 0.5
    analytics: Optional[AnalyticsSnapshot] = None
    core: Optional[FingerBlasterCore] = None
    _flash_timer: Optional[asyncio.Task] = None
    _flash_state: bool = False

    BINDINGS = [
        ("y", "place_order('YES')", "Buy YES"),
        ("n", "place_order('NO')", "Buy NO"),
        ("f", "flatten", "Flatten All"),
        ("c", "cancel_orders", "Cancel All"),
        ("q", "quit", "Quit")
    ]

    CSS = """
    Screen { 
        align: center middle; 
        background: #0D0D0D;
    }

    #app-container {
        width: 60;
        height: auto;
    }

    #strike-card {
        border: round #262626;
        background: #161616;
        height: auto;
        min-height: 14;
        padding: 1 2;
        margin-bottom: 1;
        align: center middle;
    }
    #strike-price-row { align: center middle; width: 100%; height: 4; }
    .price-column { width: 1fr; height: 100%; }
    .price-label { color: $text-muted; text-align: center; width: 100%; }
    .price-value { color: #EAB308; text-style: bold; text-align: center; width: 100%; height: 2; }
    #btc-price-value { color: #EAB308; text-style: bold; }

    #metrics-row {
        height: 5;
        margin-top: 1;
    }
    MetricBox {
        width: 1fr;
        height: 100%;
        border: round #262626;
        align: center middle;
        margin: 0 1;
    }
    .metric-label { color: $text-muted; text-align: center; width: 100%; }
    .metric-value { color: #EAB308; text-style: bold; text-align: center; width: 100%; }
    
    /* Border color classes for MetricBox */
    .border-yellow { border: round #EAB308; }
    .border-red { border: round $error; }
    .border-green { border: round $success; }
    .border-flashing-red { border: round $error; }

    #cards-row {
        height: 14;
    }
    DataCard {
        width: 1fr;
        height: 100%;
        margin: 0 1;
        border: solid #262626;
    }
    #card-yes { border: solid $success; }
    #card-no { border: solid $error; }
    
    /* Background highlighting for significant edge (>750bps) */
    .bg-highlight-green {
        background: rgba(16, 185, 129, 0.15);
    }
    .bg-highlight-red {
        background: rgba(239, 68, 68, 0.15);
    }

    .card-inner { padding: 1 1; }
    .card-row { height: 2; color: $text-muted; }
    .card-stat-left { width: 40%; text-align: left; }
    .card-stat-right { width: 60%; text-align: right; }
    
    .big-percentage {
        text-align: center;
        width: 100%;
        height: 6;
        text-style: bold;
        color: white;
        content-align: center middle;
    }

    .red-text { color: $error; }
    .green-text { color: $success; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        
        with Vertical(id="app-container"):
            with Vertical(id="strike-card"):
                with Horizontal(id="strike-price-row"):
                    with Vertical(classes="price-column"):
                        yield Static("STRIKE PRICE", classes="price-label")
                        yield Static("$0.00", id="strike-price-value", classes="price-value")
                    with Vertical(classes="price-column"):
                        yield Static("Bitcoin", classes="price-label")
                        yield Static(f"${self.btc_price:,.2f}", id="btc-price-value", classes="price-value")
                
                with Horizontal(id="metrics-row"):
                    yield MetricBox("TIME LEFT", "00:00", id="metric-time")
                    yield MetricBox("DELTA Δ", "$0.00", id="metric-delta")
                    yield MetricBox("SIGMA Σ", "0.00", id="metric-sigma")

            with Horizontal(id="cards-row"):
                yield DataCard("YES", id="card-yes")
                yield DataCard("NO", id="card-no")
        
        yield Footer()

    def _async_callback_wrapper(self, callback):
        async def wrapper(*args, **kwargs):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}", exc_info=True)
        return wrapper
    
    async def on_mount(self) -> None:
        try:
            self.core = FingerBlasterCore()
            self.core.register_callback('btc_price_update', self._async_callback_wrapper(self._on_btc_price_update))
            self.core.register_callback('price_update', self._async_callback_wrapper(self._on_price_update))
            self.core.register_callback('countdown_update', self._async_callback_wrapper(self._on_countdown_update))
            self.core.register_callback('analytics_update', self._async_callback_wrapper(self._on_analytics_update))
            self.core.register_callback('market_update', self._async_callback_wrapper(self._on_market_update))
            self.core.register_callback('resolution', self._async_callback_wrapper(self._on_resolution))
            self.core.register_callback('order_submitted', self._async_callback_wrapper(self._on_order_submitted))
            self.core.register_callback('order_filled', self._async_callback_wrapper(self._on_order_filled))
            self.core.register_callback('order_failed', self._async_callback_wrapper(self._on_order_failed))
            self.core.register_callback('flatten_started', self._async_callback_wrapper(self._on_flatten_started))
            self.core.register_callback('flatten_completed', self._async_callback_wrapper(self._on_flatten_completed))
            self.core.register_callback('flatten_failed', self._async_callback_wrapper(self._on_flatten_failed))
            
            await self.core.start_rtds()
            await self.core.update_market_status()
            
            # Refresh UI with initial values after mount
            self._refresh_ui_values()
            
            self.set_interval(0.1, self._update_loop)
            self.set_interval(self.core.config.market_status_interval, self._update_market_status)
            
            logger.info("Textual UI initialized")
        except Exception as e:
            logger.error(f"Error initializing backend: {e}", exc_info=True)
            self.notify(f"Error: {e}", severity="error")

    async def _update_loop(self) -> None:
        if self.core:
            await self.core.update_countdown()
            await self.core.update_analytics()
            await self._update_market_name_from_data()
            if not self.strike_price or self.strike_price in ('0.0', 'N/A'):
                await self._update_strike_from_market_data()

    async def _update_market_status(self) -> None:
        if self.core:
            await self.core.update_market_status()
            await asyncio.sleep(0.1)
            await self._update_market_name_from_data()
            await self._update_strike_from_market_data()

    async def _update_strike_from_market_data(self) -> None:
        if not self.core: return
        market = await self.core.market_manager.get_market()
        if market:
            strike = market.get('strike_price')
            if strike and strike not in ('N/A', 'None'):
                self.strike_price = str(strike)

    async def _update_market_name_from_data(self) -> None:
        if not self.core: return
        market = await self.core.market_manager.get_market()
        if market:
            name = market.get('question') or market.get('title') or "Market"
            if name.upper() != self.market_name:
                self.market_name = name.upper()
                self.title = self.market_name

    def _refresh_ui_values(self) -> None:
        """Refresh all UI values after mount to ensure they're displayed."""
        try:
            # Refresh BTC price
            if self.btc_price > 0:
                try:
                    self.query_one("#btc-price-value", Static).update(f"${self.btc_price:,.2f}")
                except: pass
            
            # Refresh strike price
            if self.strike_price and self.strike_price not in ('0.0', 'N/A', 'None'):
                try:
                    strike_float = float(str(self.strike_price).replace('$', '').replace(',', ''))
                    strike_formatted = f"${strike_float:,.2f}"
                    self.query_one("#strike-price-value", Static).update(strike_formatted)
                except: pass
            
            # Refresh delta if both values are available
            if self.btc_price > 0 and self.strike_price and self.strike_price not in ('0.0', 'N/A'):
                try:
                    strike = float(str(self.strike_price).replace('$', '').replace(',', ''))
                    self.delta_val = self.btc_price - strike
                    sign = "+" if self.delta_val >= 0 else ""
                    self.query_one("#metric-delta", MetricBox).query_one(".metric-value", Static).update(f"{sign}${self.delta_val:,.2f}")
                    self._update_delta_border(self.delta_val)
                except: pass
        except Exception as e:
            logger.debug(f"Error refreshing UI values: {e}")

    def watch_btc_price(self, price: float) -> None:
        """Automatically update BTC price widget when reactive property changes."""
        try:
            btc_widget = self.query_one("#btc-price-value", Static)
            btc_widget.update(f"${price:,.2f}")
        except Exception as e:
            logger.debug(f"Error updating BTC price widget: {e}")
        
        # Update delta if strike price is available
        if self.strike_price and self.strike_price not in ('0.0', 'N/A'):
            try:
                strike = float(str(self.strike_price).replace('$', '').replace(',', ''))
                self.delta_val = price - strike
                sign = "+" if self.delta_val >= 0 else ""
                delta_widget = self.query_one("#metric-delta", MetricBox)
                delta_widget.query_one(".metric-value", Static).update(f"{sign}${self.delta_val:,.2f}")
                self._update_delta_border(self.delta_val)
            except Exception as e:
                logger.debug(f"Error updating delta in watch_btc_price: {e}")

    def watch_strike_price(self, strike: str) -> None:
        """Automatically update strike price widget when reactive property changes."""
        try:
            strike_widget = self.query_one("#strike-price-value", Static)
            if strike and strike not in ('0.0', 'N/A', 'None'):
                # Try to format as currency
                try:
                    strike_float = float(str(strike).replace('$', '').replace(',', ''))
                    strike_formatted = f"${strike_float:,.2f}"
                except:
                    strike_formatted = str(strike)
                strike_widget.update(strike_formatted)
            else:
                strike_widget.update("N/A")
        except Exception as e:
            logger.debug(f"Error updating strike price widget: {e}")
        
        # Recalculate delta if BTC price is available
        if self.btc_price > 0:
            try:
                strike_float = float(str(strike).replace('$', '').replace(',', ''))
                self.delta_val = self.btc_price - strike_float
                sign = "+" if self.delta_val >= 0 else ""
                delta_widget = self.query_one("#metric-delta", MetricBox)
                delta_widget.query_one(".metric-value", Static).update(f"{sign}${self.delta_val:,.2f}")
                self._update_delta_border(self.delta_val)
            except Exception as e:
                logger.debug(f"Error updating delta in watch_strike_price: {e}")

    def watch_market_name(self, market_name: str) -> None:
        """Automatically update title when market_name changes."""
        self.title = market_name

    def _on_btc_price_update(self, price: float) -> None:
        """Callback handler for BTC price updates from core."""
        self.btc_price = price

    def _update_time_border(self, seconds_remaining: int) -> None:
        """Update border color and thickness for time left based on remaining seconds (thicker as time decreases)."""
        try:
            time_widget = self.query_one("#metric-time", MetricBox)
            # Remove all border classes
            time_widget.remove_class("border-yellow", "border-red", "border-flashing-red", "border-thin", "border-medium", "border-thick", "border-very-thick")
            
            if seconds_remaining <= 0:
                # Expired - very thick red border
                time_widget.add_class("border-red")
                time_widget.add_class("border-very-thick")
                return
            elif seconds_remaining <= 30:
                # 30 seconds or less - flashing red, very thick
                time_widget.add_class("border-flashing-red")
                time_widget.add_class("border-very-thick")
                if self._flash_timer is None or self._flash_timer.done():
                    self._start_flash_timer()
            elif seconds_remaining <= 120:  # 2 minutes
                # 2 minutes or less - solid red, thick
                time_widget.add_class("border-red")
                time_widget.add_class("border-thick")
                self._stop_flash_timer()
            elif seconds_remaining <= 300:  # 5 minutes
                # 5 minutes or less - yellow, medium
                time_widget.add_class("border-yellow")
                time_widget.add_class("border-medium")
                self._stop_flash_timer()
            else:
                # More than 5 minutes - default border, thin
                time_widget.add_class("border-thin")
                self._stop_flash_timer()
        except Exception as e:
            logger.debug(f"Error updating time border: {e}")

    def _start_flash_timer(self) -> None:
        """Start the flashing timer for red border."""
        async def flash_loop():
            while True:
                try:
                    time_widget = self.query_one("#metric-time", MetricBox)
                    if self._flash_state:
                        # Show red border (thickness class should already be applied)
                        time_widget.add_class("border-flashing-red")
                    else:
                        # Hide red border (flash off) but keep thickness
                        time_widget.remove_class("border-flashing-red")
                    self._flash_state = not self._flash_state
                except:
                    break  # Exit if widget is no longer available
                await asyncio.sleep(0.5)  # Flash every 0.5 seconds
        
        if self._flash_timer is None or self._flash_timer.done():
            self._flash_timer = asyncio.create_task(flash_loop())

    def _stop_flash_timer(self) -> None:
        """Stop the flashing timer."""
        if self._flash_timer and not self._flash_timer.done():
            self._flash_timer.cancel()
            self._flash_timer = None
            self._flash_state = False

    def _on_countdown_update(self, time_str: str, urgency, seconds_remaining: int) -> None:
        try:
            time_widget = self.query_one("#metric-time", MetricBox)
            time_widget.query_one(".metric-value", Static).update(time_str)
            self._update_time_border(seconds_remaining)
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}")

    def _update_delta_border(self, delta_value: float) -> None:
        """Update border color and thickness for delta based on value (positive=green, negative=red, thicker as |value| increases)."""
        try:
            delta_widget = self.query_one("#metric-delta", MetricBox)
            # Remove all border classes
            delta_widget.remove_class("border-red", "border-green", "border-thin", "border-medium", "border-thick", "border-very-thick")
            
            # Set color based on sign
            if delta_value > 0:
                delta_widget.add_class("border-green")
            elif delta_value < 0:
                delta_widget.add_class("border-red")
            
            # Set thickness based on absolute value
            abs_delta = abs(delta_value)
            if abs_delta >= 1000:
                delta_widget.add_class("border-very-thick")
            elif abs_delta >= 500:
                delta_widget.add_class("border-thick")
            elif abs_delta >= 100:
                delta_widget.add_class("border-medium")
            else:
                delta_widget.add_class("border-thin")
        except Exception as e:
            logger.debug(f"Error updating delta border: {e}")

    def _update_sigma_border(self, sigma_value: float) -> None:
        """Update border color and thickness for sigma based on value (positive=green, negative=red, thicker as |value| increases)."""
        try:
            sigma_widget = self.query_one("#metric-sigma", MetricBox)
            # Remove all border classes
            sigma_widget.remove_class("border-red", "border-green", "border-thin", "border-medium", "border-thick", "border-very-thick")
            
            # Set color based on sign
            if sigma_value > 0:
                sigma_widget.add_class("border-green")
            elif sigma_value < 0:
                sigma_widget.add_class("border-red")
            
            # Set thickness based on absolute value (sigma is typically in range -3 to +3)
            abs_sigma = abs(sigma_value)
            if abs_sigma >= 2.5:
                sigma_widget.add_class("border-very-thick")
            elif abs_sigma >= 1.5:
                sigma_widget.add_class("border-thick")
            elif abs_sigma >= 0.5:
                sigma_widget.add_class("border-medium")
            else:
                sigma_widget.add_class("border-thin")
        except Exception as e:
            logger.debug(f"Error updating sigma border: {e}")

    def _on_analytics_update(self, snapshot: AnalyticsSnapshot) -> None:
        self.analytics = snapshot
        try:
            if snapshot.sigma_label:
                val = snapshot.sigma_label.replace("σ", "").strip()
                sigma_widget = self.query_one("#metric-sigma", MetricBox)
                sigma_widget.query_one(".metric-value", Static).update(val)
                # Try to parse sigma value for border styling
                try:
                    sigma_float = float(val)
                    self._update_sigma_border(sigma_float)
                except ValueError:
                    pass
        except Exception as e:
            logger.debug(f"Error updating sigma: {e}")
        self._update_cards()

    def _on_market_update(self, strike: str, ends: str, market_name: str = "Market") -> None:
        self.strike_price = strike
        self.market_name = market_name.upper()
        self.title = self.market_name

    def _update_cards(self) -> None:
        if not self.analytics: return
        try:
            yes_card = self.query_one("#card-yes", DataCard)
            yes_card.update(
                self.yes_price, self.best_bid, self.best_ask,
                self.analytics.yes_ask_depth or 0.0, 
                self.analytics.fair_value_yes, self.analytics.edge_bps_yes
            )
            # Add green background if significant positive edge (>750bps)
            if self.analytics.edge_bps_yes is not None and self.analytics.edge_bps_yes > 750:
                yes_card.add_class("bg-highlight-green")
            else:
                yes_card.remove_class("bg-highlight-green")
            
            no_bid = 1.0 - self.best_ask if self.best_ask < 1.0 else 0.0
            no_ask = 1.0 - self.best_bid if self.best_bid > 0.0 else 1.0
            no_card = self.query_one("#card-no", DataCard)
            no_card.update(
                self.no_price, no_bid, no_ask,
                self.analytics.no_ask_depth or 0.0,
                self.analytics.fair_value_no, self.analytics.edge_bps_no
            )
            # Add red background if significant positive edge (>750bps) for NO
            # Note: For NO, a positive edge_bps_no means it's favorable to buy NO
            if self.analytics.edge_bps_no is not None and self.analytics.edge_bps_no > 750:
                no_card.add_class("bg-highlight-red")
            else:
                no_card.remove_class("bg-highlight-red")
        except: pass

    async def action_place_order(self, side: str) -> None:
        if self.core:
            self.notify(f"Submitting {side} order...")
            await self.core.place_order(side)

    async def action_flatten(self) -> None:
        if self.core:
            self.notify("FLATTENING...", severity="warning")
            await self.core.flatten()

    async def action_cancel_orders(self) -> None:
        if self.core:
            await self.core.cancel_all()
            self.notify("Orders Cancelled")

    def _on_order_submitted(self, side: str, size: float, price: float) -> None: pass
    def _on_order_filled(self, side: str, size: float, price: float, order_id: str) -> None:
        self.notify(f"FILLED: {side} @ ${price:.2f}", severity="success")
    def _on_order_failed(self, side: str, size: float, error: str) -> None:
        self.notify(f"FAILED: {error}", severity="error")
    def _on_flatten_started(self) -> None: pass
    def _on_flatten_completed(self, orders: int) -> None:
        self.notify(f"Flattened {orders} orders", severity="success")
    def _on_flatten_failed(self, error: str) -> None:
        self.notify(f"Flatten Failed: {error}", severity="error")
    def _on_resolution(self, resolution: Optional[str]) -> None:
        if resolution: self.notify(f"RESOLVED: {resolution}", timeout=10)

    def _on_price_update(self, yes_price, no_price, bid, ask):
        self.yes_price = yes_price
        self.no_price = no_price
        self.best_bid = bid
        self.best_ask = ask

    async def on_unmount(self) -> None:
        self._stop_flash_timer()
        if self.core: await self.core.shutdown()


def run_textual_app():
    """Entry point function for running the Textual terminal UI."""
    app = TradingTUI()
    app.run()


if __name__ == "__main__":
    run_textual_app()