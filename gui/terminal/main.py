import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Any

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
from src.analytics import AnalyticsSnapshot, EdgeDirection, TimerUrgency
from gui.terminal.position_manager import PositionManagerApp

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
                depth: Optional[float], fv: Optional[float], edge_bps: Optional[float]):
        percentage = price * 100
        if self.percentage_widget:
            self.percentage_widget.update(f"{percentage:.1f}")

        if depth is None:
            depth_str = "N/A"
        elif depth:
            depth_str = format_depth(depth)
        else:
            depth_str = "0"
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
    price_to_beat = reactive("0.0")
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
    _position_manager: Optional[PositionManagerApp] = None
    _position_manager_open: bool = False

    BINDINGS = [
        ("y", "place_order('YES')", "Buy YES"),
        ("n", "place_order('NO')", "Buy NO"),
        ("f", "flatten", "Flatten All"),
        ("c", "cancel_orders", "Cancel All"),
        ("p", "toggle_positions", "Positions"),
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

    #price-to-beat-card {
        border: round #262626;
        background: #161616;
        height: auto;
        min-height: 14;
        padding: 1 2;
        margin-bottom: 1;
        align: center middle;
    }
    #price-to-beat-row { align: center middle; width: 100%; height: 4; }
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
            with Vertical(id="price-to-beat-card"):
                with Horizontal(id="price-to-beat-row"):
                    with Vertical(classes="price-column"):
                        yield Static("PRICE TO BEAT", classes="price-label")
                        yield Static("$0.00", id="price-to-beat-value", classes="price-value")
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

    def _async_callback_wrapper(self, callback: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> None:
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
            self.core.register_callback('cancel_started', self._async_callback_wrapper(self._on_cancel_started))
            self.core.register_callback('cancel_completed', self._async_callback_wrapper(self._on_cancel_completed))
            self.core.register_callback('cancel_failed', self._async_callback_wrapper(self._on_cancel_failed))

            await self.core.start_rtds()
            # Wait for RTDS to collect some price history before looking for markets
            # This allows Chainlink price data to be available for dynamic strike resolution
            await asyncio.sleep(2.0)
            await self.core.update_market_status()

            # Wait for WebSocket to connect and receive initial order book snapshot
            # This ensures order books are populated before analytics calculations begin
            await asyncio.sleep(2.0)

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
            if not self.price_to_beat or self.price_to_beat in ('0.0', 'N/A'):
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
            strike = market.get('price_to_beat')
            if strike and strike not in ('N/A', 'None'):
                self.price_to_beat = str(strike)

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
            
            # Refresh price to beat
            if self.price_to_beat and self.price_to_beat not in ('0.0', 'N/A', 'None'):
                try:
                    strike_float = float(str(self.price_to_beat).replace('$', '').replace(',', ''))
                    strike_formatted = f"${strike_float:,.2f}"
                    self.query_one("#price-to-beat-value", Static).update(strike_formatted)
                except: pass
            
            # Refresh delta if both values are available
            if self.btc_price > 0 and self.price_to_beat and self.price_to_beat not in ('0.0', 'N/A'):
                try:
                    strike = float(str(self.price_to_beat).replace('$', '').replace(',', ''))
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
        
        # Update delta if price to beat is available
        if self.price_to_beat and self.price_to_beat not in ('0.0', 'N/A'):
            try:
                strike = float(str(self.price_to_beat).replace('$', '').replace(',', ''))
                self.delta_val = price - strike
                sign = "+" if self.delta_val >= 0 else ""
                delta_widget = self.query_one("#metric-delta", MetricBox)
                delta_widget.query_one(".metric-value", Static).update(f"{sign}${self.delta_val:,.2f}")
                self._update_delta_border(self.delta_val)
            except Exception as e:
                logger.debug(f"Error updating delta in watch_btc_price: {e}")

    def watch_price_to_beat(self, strike: str) -> None:
        """Automatically update price to beat widget when reactive property changes."""
        try:
            strike_widget = self.query_one("#price-to-beat-value", Static)
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
            logger.debug(f"Error updating price to beat widget: {e}")
        
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
                logger.debug(f"Error updating delta in watch_price_to_beat: {e}")

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

    def _on_countdown_update(self, time_str: str, urgency: Optional[TimerUrgency], seconds_remaining: int) -> None:
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
        self.price_to_beat = strike
        self.market_name = market_name.upper()
        self.title = self.market_name

    def _update_cards(self) -> None:
        try:
            yes_card = self.query_one("#card-yes", DataCard)
            # Use analytics data if available, otherwise use defaults
            # Note: depth can be None if order book not yet populated
            yes_depth = self.analytics.yes_ask_depth if self.analytics else None
            yes_fv = self.analytics.fair_value_yes if self.analytics else None
            yes_edge = self.analytics.edge_bps_yes if self.analytics else None
            yes_card.update(
                self.yes_price, self.best_bid, self.best_ask,
                yes_depth,
                yes_fv, yes_edge
            )
            # Add green background if significant positive edge (>750bps)
            if yes_edge is not None and yes_edge > 750:
                yes_card.add_class("bg-highlight-green")
            else:
                yes_card.remove_class("bg-highlight-green")

            no_bid = 1.0 - self.best_ask if self.best_ask < 1.0 else 0.0
            no_ask = 1.0 - self.best_bid if self.best_bid > 0.0 else 1.0
            no_card = self.query_one("#card-no", DataCard)
            # Use analytics data if available, otherwise use defaults
            # Note: depth can be None if order book not yet populated
            no_depth = self.analytics.no_ask_depth if self.analytics else None
            no_fv = self.analytics.fair_value_no if self.analytics else None
            no_edge = self.analytics.edge_bps_no if self.analytics else None
            no_card.update(
                self.no_price, no_bid, no_ask,
                no_depth,
                no_fv, no_edge
            )
            # Add red background if significant positive edge (>750bps) for NO
            # Note: For NO, a positive edge_bps_no means it's favorable to buy NO
            if no_edge is not None and no_edge > 750:
                no_card.add_class("bg-highlight-red")
            else:
                no_card.remove_class("bg-highlight-red")
        except Exception as e:
            logger.debug(f"Error updating cards: {e}")

    def action_place_order(self, side: str) -> None:
        if self.core:
            # Run order in background so UI can update immediately
            self.run_worker(self.core.place_order(side), exclusive=False)

    def action_flatten(self) -> None:
        if self.core:
            # Run flatten in background so UI can update immediately
            self.run_worker(self.core.flatten(), exclusive=False)

    def action_cancel_orders(self) -> None:
        if self.core:
            # Run cancel in background so UI can update immediately
            self.run_worker(self.core.cancel_all(), exclusive=False)
    
    async def action_toggle_positions(self) -> None:
        """Toggle position manager window."""
        if not self.core:
            return
        
        # Check if position manager screen is currently active
        if self.screen_stack and len(self.screen_stack) > 1:
            # Position manager is open, close it
            await self.pop_screen()
            self._position_manager_open = False
        else:
            # Open position manager
            if not self._position_manager:
                self._position_manager = PositionManagerApp(self.core)
            await self.push_screen(self._position_manager)
            self._position_manager_open = True

    def _on_order_submitted(self, side: str, size: float, price: float) -> None:
        self.notify(f"SUBMITTING: {side} ${size:.2f}...", severity="warning")

    def _on_order_filled(self, side: str, size: float, price: float, order_id: str) -> None:
        self.notify(f"FILLED: {side} ${size:.2f} @ {price:.2f}", severity="success")
    def _on_order_failed(self, side: str, size: float, error: str) -> None:
        self.notify(f"FAILED: {error}", severity="error")
    def _on_flatten_started(self) -> None:
        self.notify("FLATTENING positions...", severity="warning")

    def _on_flatten_completed(self, orders: int) -> None:
        self.notify(f"FLATTENED: {orders} orders closed", severity="success")

    def _on_flatten_failed(self, error: str) -> None:
        self.notify(f"FLATTEN FAILED: {error}", severity="error")

    def _on_cancel_started(self) -> None:
        self.notify("CANCELLING orders...", severity="warning")

    def _on_cancel_completed(self) -> None:
        self.notify("CANCELLED: All orders cancelled", severity="success")

    def _on_cancel_failed(self, error: str) -> None:
        self.notify(f"CANCEL FAILED: {error}", severity="error")

    def _on_resolution(self, resolution: Optional[str]) -> None:
        if resolution: self.notify(f"RESOLVED: {resolution}", timeout=10)

    def _on_price_update(self, yes_price: float, no_price: float, bid: float, ask: float) -> None:
        self.yes_price = yes_price
        self.no_price = no_price
        self.best_bid = bid
        self.best_ask = ask
        # Update cards with new prices
        self._update_cards()

    async def on_unmount(self) -> None:
        self._stop_flash_timer()
        if self._position_manager:
            self._position_manager.dismiss()
            self._position_manager = None
        if self.core: await self.core.shutdown()


def run_textual_app():
    """Entry point function for running the Textual terminal UI."""
    app = TradingTUI()
    app.run()


if __name__ == "__main__":
    run_textual_app()