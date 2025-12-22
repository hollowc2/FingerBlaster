"""Main entry point for FingerBlaster application (Textual UI)."""

import asyncio
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    filename='data/finger_blaster.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FingerBlaster")


def run_textual_app():
    """Run the Textual terminal UI."""
    # Import Textual only when needed
    from textual.app import App
    from textual.widgets import Header, Footer, RichLog
    from textual.containers import Horizontal, Vertical
    from textual.binding import Binding
    
    from src.config import AppConfig, CSS as CSS_STYLES
    from src.core import FingerBlasterCore
    from src.ui import MarketPanel, PricePanel, StatsPanel, ChartsPanel, ResolutionOverlay, ProbabilityChart
    from src.utils import handle_ui_errors
    
    class FingerBlasterApp(App):
        """Main application class using Textual UI with shared core."""
        
        CSS = CSS_STYLES
        
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
        
        def __init__(self) -> None:
            """Initialize the application."""
            super().__init__()
            self.core = FingerBlasterCore()
            self.config = self.core.config
            
            # UI state
            self.graphs_visible: bool = True
            self.log_visible: bool = True
            
            # Register callbacks
            self.core.register_callback('market_update', self._on_market_update)
            self.core.register_callback('btc_price_update', self._on_btc_price_update)
            self.core.register_callback('price_update', self._on_price_update)
            self.core.register_callback('account_stats_update', self._on_account_stats_update)
            self.core.register_callback('countdown_update', self._on_countdown_update)
            self.core.register_callback('prior_outcomes_update', self._on_prior_outcomes_update)
            self.core.register_callback('resolution', self._on_resolution)
            self.core.register_callback('log', self._on_log)
            self.core.register_callback('chart_update', self._on_chart_update)
        
        # Callback handlers for core events
        async def _on_market_update(self, strike: str, ends: str) -> None:
            """Handle market update from core."""
            try:
                mp = self.query_one("#market_panel", MarketPanel)
                mp.strike = strike
                mp.ends = ends
            except Exception as e:
                logger.debug(f"Error updating market panel: {e}")
        
        async def _on_btc_price_update(self, price: float) -> None:
            """Handle BTC price update from core."""
            try:
                mp = self.query_one("#market_panel", MarketPanel)
                mp.btc_price = price
            except Exception as e:
                logger.debug(f"Error updating BTC price: {e}")
        
        async def _on_price_update(self, yes_price: float, no_price: float, best_bid: float, best_ask: float) -> None:
            """Handle price update from core."""
            try:
                pp = self.query_one("#price_panel", PricePanel)
                pp.yes_price = yes_price
                pp.no_price = no_price
                pp.spread = f"{best_bid:.2f} / {best_ask:.2f}"
            except Exception as e:
                logger.debug(f"Error updating price panel: {e}")
        
        async def _on_account_stats_update(self, balance: float, yes_balance: float, no_balance: float, size: float, 
                                          avg_entry_price_yes: Optional[float], avg_entry_price_no: Optional[float]) -> None:
            """Handle account stats update from core."""
            try:
                sp = self.query_one("#stats_panel", StatsPanel)
                sp.balance = balance
                sp.yes_balance = yes_balance
                sp.no_balance = no_balance
                sp.selected_size = size
                sp.avg_entry_price_yes = avg_entry_price_yes
                sp.avg_entry_price_no = avg_entry_price_no
            except Exception as e:
                logger.debug(f"Error updating stats panel: {e}")
        
        async def _on_countdown_update(self, time_str: str) -> None:
            """Handle countdown update from core."""
            try:
                mp = self.query_one("#market_panel", MarketPanel)
                mp.time_left = time_str
            except Exception as e:
                logger.debug(f"Error updating countdown: {e}")
        
        async def _on_prior_outcomes_update(self, outcomes: list) -> None:
            """Handle prior outcomes update from core."""
            try:
                mp = self.query_one("#market_panel", MarketPanel)
                prior_str = ""
                for outcome in outcomes:
                    if outcome == "YES":
                        prior_str += "[green]▲[/]"
                    elif outcome == "NO":
                        prior_str += "[red]▼[/]"
                if not prior_str:
                    prior_str = "---"
                mp.prior_outcomes = prior_str
            except Exception as e:
                logger.debug(f"Error updating prior outcomes: {e}")
        
        async def _on_resolution(self, resolution: Optional[str]) -> None:
            """Handle resolution from core."""
            try:
                overlay = self.query_one("#resolution_overlay", ResolutionOverlay)
                if resolution:
                    overlay.show(resolution)
                else:
                    overlay.hide()
            except Exception as e:
                logger.debug(f"Error updating resolution overlay: {e}")
        
        def _on_log(self, message: str) -> None:
            """Handle log message from core."""
            try:
                log_panel = self.query_one("#log_panel", RichLog)
                log_panel.write(message)
            except Exception as e:
                logger.debug(f"Error writing to log panel: {e}")
        
        async def _on_chart_update(self, *args) -> None:
            """Handle chart update from core."""
            if not self.graphs_visible:
                return
            
            try:
                # Check if it's BTC chart or price chart
                if len(args) == 3 and args[2] == 'btc':
                    # BTC chart update
                    prices, strike_val, _ = args
                    if len(prices) < 2:
                        return
                    
                    from textual_plotext import PlotextPlot
                    plot = self.query_one("#btc_plot", PlotextPlot)
                    plt = plot.plt
                    plt.clf()
                    plt.theme("dark")
                    
                    y_min, y_max = min(prices), max(prices)
                    if strike_val is not None:
                        y_min = min(y_min, strike_val)
                        y_max = max(y_max, strike_val)
                    
                    spread = y_max - y_min
                    padding = spread * self.config.chart_padding_percentage if spread > 0 else 50.0
                    
                    plt.plot(prices, color="cyan", label="BTC")
                    if strike_val is not None:
                        plt.plot([0, len(prices)-1], [strike_val, strike_val], 
                                color="yellow", label="STRIKE")
                    
                    plt.ylim(y_min - padding, y_max + padding)
                    plt.grid(False, "x")
                    plt.xticks([])
                    plot.refresh()
                else:
                    # Price chart update
                    history = args[0]
                    if len(history) < 2:
                        return
                    
                    chart = self.query_one("#price_plot", ProbabilityChart)
                    chart.update_data(history)
            except Exception as e:
                logger.debug(f"Error updating chart: {e}")
        
        def compose(self):
            """Compose the application UI."""
            yield Header(show_clock=True)
            with Horizontal(id="main_grid"):
                with Vertical(id="left_cockpit"):
                    yield MarketPanel(id="market_panel", classes="cockpit_widget")
                    yield PricePanel(id="price_panel", classes="cockpit_widget")
                    yield StatsPanel(id="stats_panel", classes="cockpit_widget")
                
                yield ChartsPanel(id="charts_panel")
            
            yield RichLog(id="log_panel", wrap=True, highlight=True, markup=True)
            yield Footer()
            yield ResolutionOverlay(id="resolution_overlay")
        
        async def on_mount(self) -> None:
            """Initialize application on mount."""
            self.title = "FINGER BLASTER v2.0 (Cockpit Mode)"
            self.core.log_msg("Ready to Blast. Initializing...")
            
            # Check and add prior outcomes if app was off (after a delay to let market initialize)
            self.set_timer(3.0, lambda: asyncio.create_task(self.core._check_and_add_prior_outcomes()))
            
            # Start update intervals
            self.set_interval(self.config.market_status_interval, self._update_market_status)
            self.set_interval(self.config.btc_price_interval, self._update_btc_price)
            self.set_interval(self.config.account_stats_interval, self._update_account_stats)
            self.set_interval(self.config.countdown_interval, self._update_countdown)
        
        async def on_unmount(self) -> None:
            """Handle graceful shutdown."""
            await self.core.shutdown()
        
        @handle_ui_errors
        async def _update_market_status(self) -> None:
            """Update market status - delegate to core."""
            await self.core.update_market_status()
        
        @handle_ui_errors
        async def _update_btc_price(self) -> None:
            """Update BTC price - delegate to core."""
            await self.core.update_btc_price()
        
        @handle_ui_errors
        async def _update_account_stats(self) -> None:
            """Update account stats - delegate to core."""
            await self.core.update_account_stats()
        
        @handle_ui_errors
        async def _update_countdown(self) -> None:
            """Update countdown - delegate to core."""
            await self.core.update_countdown()
        
        def action_size_up(self) -> None:
            """Increase order size."""
            self.core.size_up()
            # Immediately update UI to reflect size change
            asyncio.create_task(self.core.update_account_stats())
        
        def action_size_down(self) -> None:
            """Decrease order size."""
            self.core.size_down()
            # Immediately update UI to reflect size change
            asyncio.create_task(self.core.update_account_stats())
        
        def action_buy_yes(self) -> None:
            """Place BUY YES order."""
            asyncio.create_task(self.core.place_order('YES'))
        
        def action_buy_no(self) -> None:
            """Place BUY NO order."""
            asyncio.create_task(self.core.place_order('NO'))
        
        def action_flatten(self) -> None:
            """Flatten all positions."""
            asyncio.create_task(self.core.flatten())
        
        def action_cancel_all(self) -> None:
            """Cancel all pending orders."""
            asyncio.create_task(self.core.cancel_all())
        
        def action_toggle_graphs(self) -> None:
            """Toggle graphs visibility."""
            try:
                charts_panel = self.query_one("#charts_panel", ChartsPanel)
                left_cockpit = self.query_one("#left_cockpit")
                
                self.graphs_visible = not self.graphs_visible
                
                if self.graphs_visible:
                    charts_panel.remove_class("hidden")
                    left_cockpit.remove_class("no_graphs")
                    self.core.log_msg("Graphs shown")
                else:
                    charts_panel.add_class("hidden")
                    left_cockpit.add_class("no_graphs")
                    self.core.log_msg("Graphs hidden")
            except Exception as e:
                logger.debug(f"Error toggling graphs: {e}")
        
        def action_toggle_log(self) -> None:
            """Toggle log panel visibility."""
            try:
                log_panel = self.query_one("#log_panel", RichLog)
                
                self.log_visible = not self.log_visible
                
                if self.log_visible:
                    log_panel.remove_class("hidden")
                    self.core.log_msg("Log shown")
                else:
                    log_panel.add_class("hidden")
                    # Log a message before hiding (it will be visible when shown again)
                    logger.info("Log panel hidden")
            except Exception as e:
                logger.debug(f"Error toggling log: {e}")
    
    try:
        app = FingerBlasterApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    """Main entry point - supports --desktop flag for PyQt6 UI."""
    if "--desktop" in sys.argv or "--pyqt" in sys.argv:
        # Import and run PyQt6 UI
        try:
            from main_pyqt import run_pyqt_app
            run_pyqt_app()
        except ImportError:
            logger.error("PyQt6 UI not available. Install PyQt6 to use desktop UI.")
            sys.exit(1)
    else:
        # Default to Textual terminal UI
        run_textual_app()

