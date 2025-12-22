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
        # Filter to only include data within the fixed x-axis range (0 to x_max)
        # Sort by x (time) to ensure chronological order
        # This prevents the tail from disappearing and ensures proper rendering
        filtered = [(x, y) for x, y in data if 0 <= x <= self.x_max]
        self.data = sorted(filtered, key=lambda p: p[0])
        self.refresh()
    
    def render(self) -> str:
        """Render the chart as a string with axes."""
        if len(self.data) < 2:
            # Return empty chart area
            return "\n" * (self.size.height - 1) if self.size.height > 1 else ""
        
        # Get widget dimensions
        width = self.size.width
        height = self.size.height
        
        # Need enough space for y-axis labels, x-axis labels, and plot area
        if width < 10 or height < 6:
            return ""
        
        # Fixed axis ranges - ALWAYS 0 to x_max (900 seconds)
        x_min, x_max = 0.0, self.x_max
        y_min, y_max = 0.0, 1.0
        
        # Reserve space for y-axis labels (5 chars: "1.0 " + 1 space)
        y_label_width = 5
        # Reserve space for x-axis (1 row for line + 1 row for labels)
        x_axis_height = 2
        
        # Calculate plot area dimensions
        plot_width = width - y_label_width - 1  # -1 for y-axis line
        plot_height = height - x_axis_height  # Exclude x-axis space
        
        if plot_width <= 0 or plot_height <= 0:
            return ""
        
        # Create a 2D grid for the entire widget (including labels and axes)
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw y-axis labels (0.0 to 1.0 in 0.1 increments)
        y_ticks = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
        for y_val in y_ticks:
            # Calculate y position in plot coordinates (flipped: 1.0 at top, 0.0 at bottom)
            # Use (plot_height - 1) to map to the actual grid positions
            plot_y = int(((y_max - y_val) / (y_max - y_min)) * (plot_height - 1))
            # Ensure within bounds
            plot_y = max(0, min(plot_height - 1, plot_y))
            grid_y = plot_y
            
            # Format label (e.g., "1.0", "0.5", "0.0")
            label = f"{y_val:.1f}"
            # Right-align label in the y-axis space (leave 1 char for spacing)
            label_start = y_label_width - len(label) - 1
            for i, char in enumerate(label):
                if label_start + i >= 0 and label_start + i < y_label_width:
                    grid[grid_y][label_start + i] = char
        
        # Draw y-axis line (vertical line at x = y_label_width)
        y_axis_x = y_label_width
        for y in range(plot_height):
            grid[y][y_axis_x] = '│'  # Vertical line character
        
        # Draw x-axis line (horizontal line at bottom of plot area)
        x_axis_y = plot_height
        for x in range(y_label_width + 1, width):
            grid[x_axis_y][x] = '─'  # Horizontal line character
        
        # Draw corner where axes meet
        grid[x_axis_y][y_axis_x] = '└'  # Bottom-left corner
        
        # Draw x-axis labels (1-15 minutes)
        # Convert minutes to seconds for positioning
        x_ticks_minutes = list(range(1, 16))  # 1, 2, 3, ..., 15
        label_y = x_axis_y + 1  # Row below the axis line
        
        if label_y < height:
            for minute in x_ticks_minutes:
                # Convert minute to seconds
                seconds = minute * 60
                # Calculate x position in plot coordinates
                plot_x = int(((seconds - x_min) / (x_max - x_min)) * (plot_width - 1))
                # Convert to grid column (account for y-axis space)
                grid_x = y_label_width + 1 + plot_x
                
                # Ensure within bounds
                if grid_x < y_label_width + 1 or grid_x >= width:
                    continue
                
                # Format label (e.g., "1", "2", "15")
                label = str(minute)
                # Center the label under the tick position
                label_start = grid_x - len(label) // 2
                
                # Draw label, ensuring it stays within plot area bounds
                for i, char in enumerate(label):
                    label_x = label_start + i
                    # Only draw if within the plot area (after y-axis)
                    if label_x >= y_label_width + 1 and label_x < width:
                        grid[label_y][label_x] = char
        
        # Create plot grid for the data (excluding axis labels)
        plot_grid = [[' ' for _ in range(plot_width)] for _ in range(plot_height)]
        
        # Plot the line - iterate through ALL data points in chronological order
        for i in range(len(self.data) - 1):
            x1, y1 = self.data[i]
            x2, y2 = self.data[i + 1]
            
            # Convert to pixel coordinates within plot area
            px1 = int(((x1 - x_min) / (x_max - x_min)) * plot_width)
            py1 = int(((y_max - y1) / (y_max - y_min)) * plot_height)  # Flip y-axis
            px2 = int(((x2 - x_min) / (x_max - x_min)) * plot_width)
            py2 = int(((y_max - y2) / (y_max - y_min)) * plot_height)  # Flip y-axis
            
            # Clamp to plot grid bounds
            px1 = max(0, min(plot_width - 1, px1))
            py1 = max(0, min(plot_height - 1, py1))
            px2 = max(0, min(plot_width - 1, px2))
            py2 = max(0, min(plot_height - 1, py2))
            
            # Draw line using Bresenham's algorithm
            self._draw_line(plot_grid, px1, py1, px2, py2, plot_width, plot_height)
        
        # Copy plot grid to main grid (offset by y-axis width + 1 for the axis line)
        plot_start_x = y_label_width + 1
        for y in range(plot_height):
            for x in range(plot_width):
                if plot_grid[y][x] != ' ':
                    grid[y][plot_start_x + x] = plot_grid[y][x]
        
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

