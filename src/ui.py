"""UI components for the FingerBlaster application."""

import logging
from typing import List, Tuple, Optional

from textual.widgets import Static, Label, Digits
from textual.containers import Vertical, Center
from textual.reactive import reactive
from textual_plotext import PlotextPlot

logger = logging.getLogger("FingerBlaster")


class MarketPanel(Static):
    """Panel displaying market context information."""
    
    strike = reactive("N/A")
    ends = reactive("N/A")
    btc_price = reactive(0.0)
    time_left = reactive("N/A")
    prior_outcomes = reactive("")
    
    def render(self) -> str:
        """Render the market panel."""
        delta_str = self._calculate_delta()
        time_left_str = self._format_time_left()
        
        return f"[b #00ffff]MARKET CONTEXT[/]\n" \
               f"STRIKE: [yellow]{self.strike}[/]\n" \
               f"ENDS  : [yellow]{self.ends}[/]\n" \
               f"BTC   : [green]${self.btc_price:,.2f}[/]\n" \
               f"PRIOR : {self.prior_outcomes}\n" \
               f"{delta_str}\n" \
               f"REMAIN: {time_left_str}"
    
    def _calculate_delta(self) -> str:
        """Calculate and format delta."""
        try:
            strike_val = float(self.strike.replace(',', '').replace('$', '').strip())
            diff = self.btc_price - strike_val
            symbol = "▲" if diff >= 0 else "▼"
            color = "green" if diff >= 0 else "red"
            return f"DELTA : [{color}]{symbol} ${abs(diff):,.2f}[/]"
        except (ValueError, AttributeError, TypeError):
            return "DELTA : N/A"
    
    def _format_time_left(self) -> str:
        """Format time left with color coding."""
        time_left_str = self.time_left
        if time_left_str not in ("N/A", "EXPIRED"):
            try:
                parts = time_left_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    if minutes < 2:
                        return f"[red]{self.time_left}[/]"
            except (ValueError, AttributeError):
                pass
        return f"[bold #ffff00]{self.time_left}[/]"


class PricePanel(Vertical):
    """Panel displaying live YES/NO prices and spread."""
    
    yes_price = reactive(0.0)
    no_price = reactive(0.0)
    spread = reactive("0.000 / 0.000")
    
    def compose(self):
        """Compose the price panel UI."""
        yield Label("LIVE PRICES", classes="title")
        yield Label("YES", classes="price_label")
        yield Center(Digits("0.00", id="yes_digits", classes="price_yes"))
        yield Label("NO", classes="price_label")
        yield Center(Digits("0.00", id="no_digits", classes="price_no"))
        yield Label("SPREAD: 0.00 / 0.00", id="spread_label", classes="spread_label")
    
    def watch_yes_price(self, value: float) -> None:
        """Update YES price display."""
        try:
            digits = self.query_one("#yes_digits", Digits)
            digits.update(f"{value:.2f}")
        except Exception as e:
            logger.debug(f"Error updating YES price: {e}")
    
    def watch_no_price(self, value: float) -> None:
        """Update NO price display."""
        try:
            digits = self.query_one("#no_digits", Digits)
            digits.update(f"{value:.2f}")
        except Exception as e:
            logger.debug(f"Error updating NO price: {e}")
    
    def watch_spread(self, value: str) -> None:
        """Update spread display."""
        try:
            label = self.query_one("#spread_label", Label)
            label.update(f"SPREAD: {value}")
        except Exception as e:
            logger.debug(f"Error updating spread: {e}")


class StatsPanel(Static):
    """Panel displaying account statistics."""
    
    balance = reactive(0.0)
    selected_size = reactive(1.0)
    yes_balance = reactive(0.0)
    no_balance = reactive(0.0)
    
    def render(self) -> str:
        """Render the stats panel."""
        return f"[b #00ffff]ACCOUNT STATS[/]\n" \
               f"CASH : [bold #00ff00]${self.balance:.2f}[/]\n" \
               f"SIZE : [bold #ffff00]${self.selected_size:.2f}[/]\n" \
               f"POS  : [green]Y:{self.yes_balance:.1f}[/] | [red]N:{self.no_balance:.1f}[/]"


class ProbabilityChart(Static):
    """Custom widget for probability history chart with fixed x-axis."""
    
    def __init__(self, *args, x_max: float = 900.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_max = x_max  # Fixed x-axis maximum (900 seconds)
        self.data: List[Tuple[float, float]] = []  # List of (x, y) tuples
    
    def update_data(self, data: List[Tuple[float, float]]) -> None:
        """Update the chart data and refresh."""
        self.data = data
        self.refresh()
    
    def render(self) -> str:
        """Render the chart as a string."""
        if len(self.data) < 2:
            # Return empty chart area
            return "\n" * (self.size.height - 1) if self.size.height > 1 else ""
        
        # Get widget dimensions
        width = self.size.width
        height = self.size.height
        
        if width < 3 or height < 3:
            return ""
        
        # Fixed axis ranges
        x_min, x_max = 0.0, self.x_max
        y_min, y_max = 0.0, 1.0
        
        # Calculate scaling factors
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Account for borders/padding (leave 1 char on each side)
        plot_width = width - 2
        plot_height = height - 2
        
        if plot_width <= 0 or plot_height <= 0:
            return ""
        
        # Create a 2D grid for the plot (using spaces as background)
        grid = [[' ' for _ in range(plot_width)] for _ in range(plot_height)]
        
        # Plot the line
        for i in range(len(self.data) - 1):
            x1, y1 = self.data[i]
            x2, y2 = self.data[i + 1]
            
            # Convert to pixel coordinates
            px1 = int(((x1 - x_min) / x_range) * plot_width)
            py1 = int(((y_max - y1) / y_range) * plot_height)  # Flip y-axis
            px2 = int(((x2 - x_min) / x_range) * plot_width)
            py2 = int(((y_max - y2) / y_range) * plot_height)  # Flip y-axis
            
            # Clamp to grid bounds
            px1 = max(0, min(plot_width - 1, px1))
            py1 = max(0, min(plot_height - 1, py1))
            px2 = max(0, min(plot_width - 1, px2))
            py2 = max(0, min(plot_height - 1, py2))
            
            # Draw line using Bresenham's algorithm
            self._draw_line(grid, px1, py1, px2, py2, plot_width, plot_height)
        
        # Convert grid to string
        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)
    
    def _draw_line(self, grid: List[List[str]], x1: int, y1: int, x2: int, y2: int, 
                   width: int, height: int) -> None:
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        # Character for drawing the line
        # Using middle dot for a clean, visible line
        line_char = '·'  # Middle dot for line points
        
        while True:
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = line_char
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy


class ChartsPanel(Vertical):
    """Panel containing price history and BTC price charts."""
    
    def compose(self):
        """Compose the charts panel UI."""
        from src.config import AppConfig
        config = AppConfig()
        with Vertical(classes="box", id="price_history_container"):
            yield Label("PROBABILITY HISTORY", classes="chart_label")
            yield ProbabilityChart(id="price_plot", x_max=float(config.market_duration_seconds))
        with Vertical(classes="box", id="btc_history_container"):
            yield Label("BTC PRICE HISTORY", classes="chart_label")
            yield PlotextPlot(id="btc_plot")


class ResolutionOverlay(Static):
    """Full-screen overlay showing market resolution."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resolution = "YES"
        self.add_class("hidden")
    
    def render(self) -> str:
        """Render the resolution overlay."""
        outcome = "YES" if self.resolution == "YES" else "NO"
        return f"\n\n\n\n\n\n\n\n[bold]{outcome}[/bold]\n\nMARKET RESOLVED"
    
    def show(self, resolution: str) -> None:
        """Show the overlay with the given resolution."""
        self.resolution = resolution.upper()
        self.remove_class("hidden")
        self.remove_class("yes")
        self.remove_class("no")
        if self.resolution == "YES":
            self.add_class("yes")
        else:
            self.add_class("no")
        self.display = True
    
    def hide(self) -> None:
        """Hide the overlay."""
        self.add_class("hidden")
        self.remove_class("yes")
        self.remove_class("no")
        self.display = False

