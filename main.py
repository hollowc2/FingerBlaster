"""Main entry point for FingerBlaster application."""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Union

import pandas as pd
from dotenv import load_dotenv

from textual.app import App
from textual.widgets import Header, Footer, RichLog
from textual.containers import Horizontal, Vertical
from textual.binding import Binding

from connectors.polymarket import PolymarketConnector
from src.config import AppConfig, CSS, CSS_PURPLE
from src.engine import MarketDataManager, HistoryManager, WebSocketManager, OrderExecutor
from src.ui import MarketPanel, PricePanel, StatsPanel, ChartsPanel, ResolutionOverlay, ProbabilityChart
from src.ui_purple import (
    MarketPanel as MarketPanelPurple,
    PricePanel as PricePanelPurple,
    StatsPanel as StatsPanelPurple,
    ChartsPanel as ChartsPanelPurple,
    ResolutionOverlay as ResolutionOverlayPurple
)

# Type aliases for theme-agnostic panels
MarketPanelType = Union[MarketPanel, MarketPanelPurple]
PricePanelType = Union[PricePanel, PricePanelPurple]
StatsPanelType = Union[StatsPanel, StatsPanelPurple]
from src.utils import handle_ui_errors

load_dotenv()

# Configure logging
logging.basicConfig(
    filename='data/finger_blaster.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FingerBlaster")


class FingerBlasterApp(App):
    """Main application class with refactored architecture."""
    
    CSS = CSS
    
    BINDINGS = [
        Binding("y", "buy_yes", "Buy YES", show=True),
        Binding("n", "buy_no", "Buy NO", show=True),
        Binding("f", "flatten", "Flatten", show=True),
        Binding("c", "cancel_all", "Cancel All", show=True),
        Binding("plus", "size_up", "Size +$1", show=True),
        Binding("=", "size_up", "Size +$1", show=False),
        Binding("minus", "size_down", "Size -$1", show=True),
        Binding("_", "size_down", "Size -$1", show=False),
        Binding("h", "toggle_graphs", "Toggle Graphs", show=True),
        Binding("l", "toggle_log", "Toggle Log", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]
    
    def __init__(self, use_purple_theme: bool = False) -> None:
        """Initialize the application.
        
        Args:
            use_purple_theme: If True, use the purple theme UI. Defaults to False.
        """
        super().__init__()
        self.config = AppConfig()
        self.connector = PolymarketConnector()
        self.use_purple_theme = use_purple_theme
        
        # Set CSS based on theme
        if use_purple_theme:
            self.CSS = CSS_PURPLE
        else:
            self.CSS = CSS
        
        # Initialize managers
        self.market_manager = MarketDataManager(self.config)
        self.history_manager = HistoryManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.connector)
        self.ws_manager = WebSocketManager(
            self.config,
            self.market_manager,
            self._on_ws_message
        )
        
        # UI state
        self.resolution_shown = False
        self.last_resolution: Optional[str] = None
        self.prior_outcomes: List[Dict[str, Any]] = []
        self.displayed_prior_outcomes: List[str] = []  # Filtered consecutive outcomes to display
        self.last_chart_update: float = 0.0
        self.graphs_visible: bool = True
        self.log_visible: bool = True
        
        # Load prior outcomes
        self._load_prior_outcomes()
    
    async def _check_and_add_prior_outcomes(self) -> None:
        """Check if prior outcomes should be displayed based on consecutive market timing."""
        try:
            market = await self.market_manager.get_market()
            if not market or not self.prior_outcomes:
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            # Get current market start time
            market_start_time = await self.market_manager.get_market_start_time()
            if not market_start_time:
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            # Calculate what the immediately preceding market's end time should be
            # (current market start - market duration)
            expected_previous_market_end = market_start_time - pd.Timedelta(
                minutes=self.config.market_duration_minutes
            )
            
            # Tolerance for time matching (1 minute to account for slight variations)
            tolerance_seconds = 60
            
            # Filter outcomes to only include consecutive ones
            consecutive_outcomes = []
            
            # Work backwards through prior_outcomes to find consecutive markets
            # Start by checking if the last outcome matches the immediately preceding market
            expected_end_time = expected_previous_market_end
            
            # Iterate backwards through prior_outcomes
            for outcome_entry in reversed(self.prior_outcomes):
                if isinstance(outcome_entry, str):
                    # Legacy format - can't check time, stop here
                    break
                
                timestamp_str = outcome_entry.get('timestamp')
                if not timestamp_str:
                    # No timestamp available, stop here
                    break
                
                try:
                    outcome_timestamp = pd.Timestamp(timestamp_str)
                    if outcome_timestamp.tz is None:
                        outcome_timestamp = outcome_timestamp.tz_localize('UTC')
                    
                    # Check if this outcome's timestamp matches the expected end time
                    time_diff = abs((outcome_timestamp - expected_end_time).total_seconds())
                    
                    if time_diff <= tolerance_seconds:
                        # This outcome matches! Add it to the consecutive list
                        consecutive_outcomes.insert(0, outcome_entry.get('outcome', ''))
                        
                        # Calculate the expected end time for the next market going backwards
                        expected_end_time = outcome_timestamp - pd.Timedelta(
                            minutes=self.config.market_duration_minutes
                        )
                    else:
                        # Time doesn't match - we've found a gap, stop here
                        break
                        
                except Exception as e:
                    logger.debug(f"Error processing outcome timestamp: {e}")
                    break
            
            # Update displayed outcomes
            self.displayed_prior_outcomes = consecutive_outcomes
            self._update_prior_outcomes_display()
            
            if consecutive_outcomes:
                self.log_msg(f"Displaying {len(consecutive_outcomes)} consecutive prior outcome(s)")
                    
        except Exception as e:
            logger.debug(f"Error in _check_and_add_prior_outcomes: {e}")
            self.displayed_prior_outcomes = []
            self._update_prior_outcomes_display()
    
    def compose(self):
        """Compose the application UI."""
        yield Header(show_clock=True)
        with Horizontal(id="main_grid"):
            with Vertical(id="left_cockpit"):
                if self.use_purple_theme:
                    yield MarketPanelPurple(id="market_panel", classes="cockpit_widget")
                    yield PricePanelPurple(id="price_panel", classes="cockpit_widget")
                    yield StatsPanelPurple(id="stats_panel", classes="cockpit_widget")
                else:
                    yield MarketPanel(id="market_panel", classes="cockpit_widget")
                    yield PricePanel(id="price_panel", classes="cockpit_widget")
                    yield StatsPanel(id="stats_panel", classes="cockpit_widget")
            
            if self.use_purple_theme:
                yield ChartsPanelPurple(id="charts_panel")
            else:
                yield ChartsPanel(id="charts_panel")
        
        yield RichLog(id="log_panel", wrap=True, highlight=True, markup=True)
        yield Footer()
        if self.use_purple_theme:
            yield ResolutionOverlayPurple(id="resolution_overlay")
        else:
            yield ResolutionOverlay(id="resolution_overlay")
    
    async def on_mount(self) -> None:
        """Initialize application on mount."""
        theme_name = "Purple Theme" if self.use_purple_theme else "Cockpit Mode"
        self.title = f"FINGER BLASTER v2.0 ({theme_name})"
        self.log_msg("Ready to Blast. Initializing...")
        
        # Initialize displayed outcomes as empty until we check
        self.displayed_prior_outcomes = []
        self._update_prior_outcomes_display()
        
        # Check and add prior outcomes if app was off (after a delay to let market initialize)
        self.set_timer(3.0, lambda: asyncio.create_task(self._check_and_add_prior_outcomes()))
        
        # Start update intervals
        self.set_interval(self.config.market_status_interval, self.update_market_status)
        self.set_interval(self.config.btc_price_interval, self.update_btc_price)
        self.set_interval(self.config.account_stats_interval, self.update_account_stats)
        self.set_interval(self.config.countdown_interval, self.update_countdown)
    
    async def on_unmount(self) -> None:
        """Handle graceful shutdown."""
        self.log_msg("Initiating graceful shutdown...")
        await self.ws_manager.stop()
    
    @handle_ui_errors
    async def update_market_status(self) -> None:
        """Update market status and search for new markets."""
        market = await self.market_manager.get_market()
        if not market:
            # Search for new market
            try:
                new_market = await asyncio.to_thread(self.connector.get_active_market)
                if new_market:
                    success = await self.market_manager.set_market(new_market)
                    if success:
                        await self.history_manager.clear_yes_history()
                        await self._initialize_market_ui(new_market)
                        self.log_msg(f"Market Found: {new_market.get('strike_price', 'N/A')}")
                        await self.ws_manager.start()
                        # Re-check prior outcomes for the new market
                        await self._check_and_add_prior_outcomes()
            except Exception as e:
                logger.error(f"Error searching for market: {e}")
        
        await self.check_if_market_expired()
    
    async def _initialize_market_ui(self, market: Dict[str, Any]) -> None:
        """Initialize UI with market data."""
        try:
            mp = self.query_one("#market_panel", MarketPanelType)
            mp.strike = str(market.get('strike_price', 'N/A'))
            mp.ends = self._format_ends(market.get('end_date', 'N/A'))
        except Exception as e:
            logger.error(f"Error initializing market UI: {e}")
    
    @handle_ui_errors
    async def update_btc_price(self) -> None:
        """Update BTC price and refresh chart."""
        try:
            price = await asyncio.to_thread(self.connector.get_btc_price)
            if price and isinstance(price, (int, float)) and price > 0:
                mp = self.query_one("#market_panel", MarketPanelType)
                mp.btc_price = float(price)
                
                await self.history_manager.add_btc_price(float(price))
                await self._update_btc_chart()
        except Exception as e:
            logger.error(f"Error updating BTC price: {e}")
    
    async def _update_btc_chart(self) -> None:
        """Update BTC price chart."""
        if not self.graphs_visible:
            return
        
        try:
            prices = await self.history_manager.get_btc_history()
            if len(prices) < 2:
                return
            
            from textual_plotext import PlotextPlot
            plot = self.query_one("#btc_plot", PlotextPlot)
            plt = plot.plt
            plt.clf()
            plt.theme("dark")
            
            y_min, y_max = min(prices), max(prices)
            
            # Get strike for context
            mp = self.query_one("#market_panel", MarketPanelType)
            strike_val = self._parse_strike(mp.strike)
            
            if strike_val is not None:
                y_min = min(y_min, strike_val)
                y_max = max(y_max, strike_val)
            
            spread = y_max - y_min
            padding = spread * self.config.chart_padding_percentage if spread > 0 else 50.0
            
            # Use theme-appropriate colors
            if self.use_purple_theme:
                plt.plot(prices, color="#9d4edd", label="BTC")
                if strike_val is not None:
                    plt.plot([0, len(prices)-1], [strike_val, strike_val], 
                            color="#e0aaff", label="STRIKE")
            else:
                plt.plot(prices, color="cyan", label="BTC")
                if strike_val is not None:
                    plt.plot([0, len(prices)-1], [strike_val, strike_val], 
                            color="yellow", label="STRIKE")
            
            plt.ylim(y_min - padding, y_max + padding)
            plt.grid(False, "x")
            plt.xticks([])
            plot.refresh()
        except Exception as e:
            logger.debug(f"Error updating BTC chart: {e}")
    
    @handle_ui_errors
    async def update_account_stats(self) -> None:
        """Update account statistics."""
        try:
            token_map = await self.market_manager.get_token_map()
            
            def get_stats():
                bal = self.connector.get_usdc_balance()
                yes_bal = 0.0
                no_bal = 0.0
                if token_map:
                    y_id = token_map.get('YES')
                    n_id = token_map.get('NO')
                    if y_id:
                        yes_bal = self.connector.get_token_balance(y_id)
                    if n_id:
                        no_bal = self.connector.get_token_balance(n_id)
                return float(bal or 0.0), float(yes_bal or 0.0), float(no_bal or 0.0)
            
            bal, y, n = await asyncio.to_thread(get_stats)
            sp = self.query_one("#stats_panel", StatsPanelType)
            sp.balance = bal
            sp.yes_balance = y
            sp.no_balance = n
        except Exception as e:
            logger.error(f"Error updating account stats: {e}")
    
    @handle_ui_errors
    async def update_countdown(self) -> None:
        """Update the countdown timer."""
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            dt_end = pd.Timestamp(end_str)
            if dt_end.tz is None:
                dt_end = dt_end.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            diff = dt_end - now
            
            if diff.total_seconds() < 0:
                time_str = "EXPIRED"
            else:
                secs = int(diff.total_seconds())
                mins = secs // 60
                re_secs = secs % 60
                time_str = f"{mins:02d}:{re_secs:02d}"
            
            mp = self.query_one("#market_panel", MarketPanelType)
            mp.time_left = time_str
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}")
    
    async def _on_ws_message(self, item: Dict[str, Any]) -> None:
        """Handle WebSocket message by recalculating price."""
        await self._recalc_price()
    
    async def _recalc_price(self) -> None:
        """Recalculate mid price and update UI."""
        yes_price, no_price, best_bid, best_ask = await self.market_manager.calculate_mid_price()
        
        # Update UI
        try:
            pp = self.query_one("#price_panel", PricePanelType)
            pp.yes_price = yes_price
            pp.no_price = no_price
            pp.spread = f"{best_bid:.2f} / {best_ask:.2f}"
        except Exception as e:
            logger.debug(f"Error updating price panel: {e}")
        
        # Update history
        market_start_time = await self.market_manager.get_market_start_time()
        if market_start_time:
            now = pd.Timestamp.now(tz='UTC')
            elapsed = (now - market_start_time).total_seconds()
            await self.history_manager.add_price_point(elapsed, yes_price, market_start_time)
        
        # Update charts (throttled)
        if self.graphs_visible:
            now = time.time()
            if now - self.last_chart_update >= self.config.chart_update_throttle_seconds:
                self.last_chart_update = now
                await self._update_price_chart()
    
    async def _update_price_chart(self) -> None:
        """Update price history chart using custom widget with fixed x-axis."""
        try:
            history = await self.history_manager.get_yes_history()
            if len(history) < 2:
                return
            
            # Use custom ProbabilityChart widget with fixed x-axis
            chart = self.query_one("#price_plot", ProbabilityChart)
            
            # Update with actual data only - no boundary points needed!
            # The widget handles the fixed x-axis internally
            chart.update_data(history)
            
        except Exception as e:
            logger.debug(f"Error updating price chart: {e}")
    
    async def check_if_market_expired(self) -> None:
        """Check if market has expired and show resolution."""
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            end_dt = pd.Timestamp(end_str)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            if now > end_dt and not self.resolution_shown:
                self.resolution_shown = True
                await self._show_resolution_overlay()
                await asyncio.sleep(self.config.resolution_overlay_duration)
                await self._reset_market_after_resolution()
        except Exception as e:
            logger.error(f"Error checking market expiry: {e}")
    
    async def _show_resolution_overlay(self) -> None:
        """Show resolution overlay."""
        try:
            mp = self.query_one("#market_panel", MarketPanelType)
            btc_price = mp.btc_price
            strike_str = str(mp.strike).replace(',', '').replace('$', '').strip()
            
            if strike_str and strike_str != "N/A":
                try:
                    strike_val = float(strike_str)
                    resolution = "YES" if btc_price >= strike_val else "NO"
                except (ValueError, TypeError):
                    resolution = "YES"
            else:
                resolution = "YES"
            
            self.last_resolution = resolution
            overlay = self.query_one("#resolution_overlay", ResolutionOverlay)
            overlay.show(resolution)
            self.log_msg(f"Market Resolved: {resolution} (BTC: ${btc_price:,.2f} vs Strike: {strike_str})")
        except Exception as e:
            logger.error(f"Error showing resolution overlay: {e}")
    
    async def _reset_market_after_resolution(self) -> None:
        """Reset market state after resolution."""
        try:
            overlay = self.query_one("#resolution_overlay", ResolutionOverlay)
            overlay.hide()
            
            if self.last_resolution:
                # Get market before clearing it
                market = await self.market_manager.get_market()
                await self._add_prior_outcome(self.last_resolution, market)
                self.last_resolution = None
            
            self.log_msg("Clearing expired market...")
            await self.market_manager.clear_market()
            await self.history_manager.clear_yes_history()
            await self.ws_manager.stop()
            self.resolution_shown = False
        except Exception as e:
            logger.error(f"Error resetting market: {e}")
    
    async def _add_prior_outcome(self, outcome: str, market: Optional[Dict[str, Any]] = None) -> None:
        """Add outcome to prior outcomes list with timestamp."""
        outcome_upper = outcome.upper()
        if outcome_upper in ("YES", "NO"):
            # Get the current market's end date as timestamp
            if market is None:
                market = await self.market_manager.get_market()
            
            timestamp = None
            if market and market.get('end_date'):
                try:
                    end_dt = pd.Timestamp(market.get('end_date'))
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize('UTC')
                    timestamp = end_dt.isoformat()
                except Exception as e:
                    logger.debug(f"Error getting timestamp for outcome: {e}")
            
            # If we couldn't get timestamp, use current time
            if not timestamp:
                timestamp = pd.Timestamp.now(tz='UTC').isoformat()
            
            outcome_entry = {
                'outcome': outcome_upper,
                'timestamp': timestamp
            }
            self.prior_outcomes.append(outcome_entry)
            if len(self.prior_outcomes) > self.config.max_prior_outcomes:
                self.prior_outcomes.pop(0)
            self._save_prior_outcomes()
            self._update_prior_outcomes_display()
    
    def _update_prior_outcomes_display(self) -> None:
        """Update prior outcomes display."""
        try:
            mp = self.query_one("#market_panel", MarketPanel)
            prior_str = ""
            
            # Use displayed_prior_outcomes (filtered consecutive outcomes) if available,
            # otherwise fall back to all prior_outcomes for backwards compatibility
            outcomes_to_display = self.displayed_prior_outcomes if self.displayed_prior_outcomes else [
                (outcome_entry if isinstance(outcome_entry, str) else outcome_entry.get('outcome', ''))
                for outcome_entry in self.prior_outcomes
            ]
            
            for outcome in outcomes_to_display:
                if outcome == "YES":
                    # Use theme-appropriate colors
                    if self.use_purple_theme:
                        prior_str += "[bold #9d4edd]▲[/]"
                    else:
                        prior_str += "[green]▲[/]"
                elif outcome == "NO":
                    if self.use_purple_theme:
                        prior_str += "[bold #d63384]▼[/]"
                    else:
                        prior_str += "[red]▼[/]"
            
            if not prior_str:
                prior_str = "---"
            
            mp.prior_outcomes = prior_str
        except Exception as e:
            logger.debug(f"Error updating prior outcomes display: {e}")
    
    def _load_prior_outcomes(self) -> None:
        """Load prior outcomes from file."""
        try:
            if os.path.exists(self.config.prior_outcomes_file):
                with open(self.config.prior_outcomes_file, 'r') as f:
                    data = json.load(f)
                    outcomes = data.get('outcomes', [])
                    
                    # Handle legacy format (list of strings) or new format (list of dicts)
                    normalized_outcomes = []
                    for outcome in outcomes:
                        if isinstance(outcome, str):
                            # Legacy format - convert to new format
                            normalized_outcomes.append({
                                'outcome': outcome,
                                'timestamp': None  # Unknown timestamp for legacy data
                            })
                        elif isinstance(outcome, dict):
                            normalized_outcomes.append(outcome)
                    
                    self.prior_outcomes = normalized_outcomes[:self.config.max_prior_outcomes]
        except Exception as e:
            logger.debug(f"Error loading prior outcomes: {e}")
            self.prior_outcomes = []
    
    def _save_prior_outcomes(self) -> None:
        """Save prior outcomes to file."""
        try:
            os.makedirs(self.config.data_dir, exist_ok=True)
            with open(self.config.prior_outcomes_file, 'w') as f:
                json.dump({'outcomes': self.prior_outcomes}, f)
        except Exception as e:
            logger.debug(f"Error saving prior outcomes: {e}")
    
    def _format_ends(self, end_str: str) -> str:
        """Format end date to PST."""
        try:
            dt = pd.Timestamp(end_str)
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            dt_local = dt.tz_convert('US/Pacific')
            return dt_local.strftime("%H:%M:%S PST")
        except Exception as e:
            logger.debug(f"Error formatting end date: {e}")
            return str(end_str)
    
    def _parse_strike(self, strike_str: str) -> Optional[float]:
        """Parse strike price string to float."""
        try:
            clean_strike = strike_str.replace(',', '').replace('$', '').strip()
            if clean_strike and clean_strike != "N/A":
                return float(clean_strike)
        except (ValueError, AttributeError):
            pass
        return None
    
    def log_msg(self, message: str) -> None:
        """Log message to UI and file."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_panel = self.query_one("#log_panel", RichLog)
            log_panel.write(f"[{timestamp}] {message}")
            logger.info(message)
        except Exception as e:
            logger.debug(f"Error writing to log panel: {e}")
    
    def action_size_up(self) -> None:
        """Increase order size."""
        try:
            sp = self.query_one("#stats_panel", StatsPanelType)
            sp.selected_size += self.config.size_increment
            self.log_msg(f"Size increased to ${sp.selected_size:.2f}")
        except Exception as e:
            logger.debug(f"Error increasing size: {e}")
    
    def action_size_down(self) -> None:
        """Decrease order size."""
        try:
            sp = self.query_one("#stats_panel", StatsPanelType)
            if sp.selected_size > self.config.min_order_size:
                sp.selected_size -= self.config.size_increment
                self.log_msg(f"Size decreased to ${sp.selected_size:.2f}")
        except Exception as e:
            logger.debug(f"Error decreasing size: {e}")
    
    def action_buy_yes(self) -> None:
        """Place BUY YES order."""
        asyncio.create_task(self._place_order('YES'))
    
    def action_buy_no(self) -> None:
        """Place BUY NO order."""
        asyncio.create_task(self._place_order('NO'))
    
    async def _place_order(self, side: str) -> None:
        """Place an order."""
        try:
            sp = self.query_one("#stats_panel", StatsPanelType)
            size = sp.selected_size
            self.log_msg(f"Order: [bold]BUY {side}[/] (${size:.2f})")
            
            token_map = await self.market_manager.get_token_map()
            resp = await self.order_executor.execute_order(side, size, token_map)
            
            if resp and resp.get('orderID'):
                self.log_msg(f"[green]Order FILLED: {resp['orderID'][:10]}...[/]")
                # Refresh balance
                await self.update_account_stats()
            else:
                self.log_msg(f"[red]Order FAILED[/]")
        except Exception as e:
            self.log_msg(f"[red]Execution Error: {e}[/]")
            logger.error(f"Order placement error: {e}")
    
    def action_flatten(self) -> None:
        """Flatten all positions."""
        self.log_msg("Action: FLATTEN")
        asyncio.create_task(self._execute_flatten())
    
    async def _execute_flatten(self) -> None:
        """Execute flatten operation."""
        token_map = await self.market_manager.get_token_map()
        if not token_map:
            self.log_msg("Error: Token map not ready.")
            return
        
        results = await self.order_executor.flatten_positions(token_map)
        if results:
            self.log_msg(f"Flatten completed. {len(results)} orders processed.")
        else:
            self.log_msg("Flatten completed. No orders to process.")
        
        await self.update_account_stats()
    
    def action_cancel_all(self) -> None:
        """Cancel all pending orders."""
        self.log_msg("Action: CANCEL ALL")
        asyncio.create_task(self._execute_cancel_all())
    
    async def _execute_cancel_all(self) -> None:
        """Execute cancel all operation."""
        success = await self.order_executor.cancel_all_orders()
        if success:
            self.log_msg("All orders cancelled.")
        else:
            self.log_msg("Cancel operation failed.")
    
    def action_toggle_graphs(self) -> None:
        """Toggle graphs visibility."""
        try:
            charts_panel = self.query_one("#charts_panel", ChartsPanel)
            left_cockpit = self.query_one("#left_cockpit")
            
            self.graphs_visible = not self.graphs_visible
            
            if self.graphs_visible:
                charts_panel.remove_class("hidden")
                left_cockpit.remove_class("no_graphs")
                self.log_msg("Graphs shown")
            else:
                charts_panel.add_class("hidden")
                left_cockpit.add_class("no_graphs")
                self.log_msg("Graphs hidden")
        except Exception as e:
            logger.debug(f"Error toggling graphs: {e}")
    
    def action_toggle_log(self) -> None:
        """Toggle log panel visibility."""
        try:
            log_panel = self.query_one("#log_panel", RichLog)
            
            self.log_visible = not self.log_visible
            
            if self.log_visible:
                log_panel.remove_class("hidden")
                self.log_msg("Log shown")
            else:
                log_panel.add_class("hidden")
                # Log a message before hiding (it will be visible when shown again)
                logger.info("Log panel hidden")
        except Exception as e:
            logger.debug(f"Error toggling log: {e}")


if __name__ == "__main__":
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FingerBlaster Trading Application")
    parser.add_argument(
        "--theme",
        choices=["default", "purple"],
        default="default",
        help="UI theme to use (default: default)"
    )
    args = parser.parse_args()
    
    use_purple = args.theme == "purple"
    
    try:
        app = FingerBlasterApp(use_purple_theme=use_purple)
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        sys.exit(0)

