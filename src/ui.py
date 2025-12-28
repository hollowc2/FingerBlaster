"""UI components for the FingerBlaster application."""

import logging
from typing import Dict, List, Tuple, Optional

from textual.widgets import Static, Label, Digits
from textual.containers import Vertical, Center, Horizontal
from textual.reactive import reactive
from textual_plotext import PlotextPlot

from src.analytics import AnalyticsSnapshot, TimerUrgency, EdgeDirection

logger = logging.getLogger("FingerBlaster")


class MarketPanel(Static):
    """Panel displaying market context information with advanced analytics."""
    
    strike = reactive("N/A")
    ends = reactive("N/A")
    btc_price = reactive(0.0)
    time_left = reactive("N/A")
    prior_outcomes = reactive("")
    timer_urgency = reactive(TimerUrgency.NORMAL)
    
    # Analytics data
    basis_points = reactive(None)
    z_score = reactive(None)
    sigma_label = reactive("")
    regime_direction = reactive("")
    regime_strength = reactive(0.0)
    oracle_lag_ms = reactive(None)
    
    def render(self) -> str:
        """Render the market panel with analytics."""
        delta_str = self._calculate_delta()
        time_left_str = self._format_time_left()
        bps_str = self._format_bps()
        sigma_str = self._format_sigma()
        regime_str = self._format_regime()
        lag_str = self._format_oracle_lag()
        
        return f"[b #00ffff]══ MARKET CONTEXT ══[/]\n" \
               f"STRIKE: [yellow]{self.strike}[/]\n" \
               f"BTC   : [green]${self.btc_price:,.2f}[/]\n" \
               f"{delta_str} {bps_str}\n" \
               f"SIGMA : {sigma_str}\n" \
               f"PRIOR : {self.prior_outcomes}\n" \
               f"REGIME: {regime_str}\n" \
               f"ORACLE: {lag_str}\n" \
               f"REMAIN: {time_left_str}"
    
    def _calculate_delta(self) -> str:
        """Calculate and format delta."""
        try:
            strike_val = float(self.strike.replace(',', '').replace('$', '').strip())
            diff = self.btc_price - strike_val
            symbol = "▲" if diff >= 0 else "▼"
            color = "green" if diff >= 0 else "red"
            return f"DELTA: [{color}]{symbol}${abs(diff):,.0f}[/]"
        except (ValueError, AttributeError, TypeError):
            return "DELTA: N/A"
    
    def _format_bps(self) -> str:
        """Format basis points."""
        if self.basis_points is None:
            return ""
        sign = "+" if self.basis_points >= 0 else ""
        color = "green" if self.basis_points >= 0 else "red"
        return f"[{color}]({sign}{self.basis_points:.0f}bps)[/]"
    
    def _format_sigma(self) -> str:
        """Format z-score/sigma."""
        if self.z_score is None or not self.sigma_label:
            return "[dim]---[/]"
        color = "green" if self.z_score >= 0 else "red"
        return f"[{color}]{self.sigma_label}[/]"
    
    def _format_regime(self) -> str:
        """Format regime detection."""
        if not self.regime_direction or self.regime_strength == 0:
            return "[dim]---[/]"
        color = "green" if self.regime_direction == "BULLISH" else "red"
        if self.regime_direction == "NEUTRAL":
            color = "yellow"
        return f"[{color}]{self.regime_strength:.0f}% {self.regime_direction}[/]"
    
    def _format_oracle_lag(self) -> str:
        """Format oracle lag."""
        if self.oracle_lag_ms is None:
            return "[dim]SYNC[/]"
        if self.oracle_lag_ms < 500:
            return f"[green]{self.oracle_lag_ms}ms[/]"
        elif self.oracle_lag_ms < 2000:
            return f"[yellow]{self.oracle_lag_ms}ms[/]"
        else:
            return f"[red blink]{self.oracle_lag_ms}ms LAG![/]"
    
    def _format_time_left(self) -> str:
        """Format time left with urgency-based color coding."""
        time_left_str = self.time_left
        
        if time_left_str == "EXPIRED":
            return "[red bold]EXPIRED[/]"
        
        if self.timer_urgency == TimerUrgency.CRITICAL:
            return f"[red bold blink]⚠ {self.time_left} ⚠[/]"
        elif self.timer_urgency == TimerUrgency.WATCHFUL:
            return f"[#ff8800 bold]{self.time_left}[/]"
        else:
            return f"[green bold]{self.time_left}[/]"
    
    def update_analytics(self, snapshot: AnalyticsSnapshot) -> None:
        """Update panel with analytics snapshot."""
        self.basis_points = snapshot.basis_points
        self.z_score = snapshot.z_score
        self.sigma_label = snapshot.sigma_label
        self.regime_direction = snapshot.regime_direction
        self.regime_strength = snapshot.regime_strength
        self.oracle_lag_ms = snapshot.oracle_lag_ms
        self.timer_urgency = snapshot.timer_urgency


class PricePanel(Vertical):
    """Panel displaying live YES/NO prices, spread, and edge detection."""
    
    yes_price = reactive(0.0)
    no_price = reactive(0.0)
    yes_spread = reactive("0.000 / 0.000")
    no_spread = reactive("0.000 / 0.000")
    
    # Analytics
    fair_value_yes = reactive(None)
    fair_value_no = reactive(None)
    edge_yes = reactive(None)
    edge_no = reactive(None)
    edge_bps_yes = reactive(0.0)
    edge_bps_no = reactive(0.0)
    yes_depth = reactive(0.0)
    no_depth = reactive(0.0)
    slippage_yes = reactive(0.0)
    slippage_no = reactive(0.0)
    
    def compose(self):
        """Compose the price panel UI."""
        yield Label("═══ LIVE PRICES ═══", classes="title")
        yield Label("SPREAD: 0.00 / 0.00", id="yes_spread_label", classes="spread_label")
        yield Label("FV: --- | EDGE: ---", id="yes_fv_label", classes="spread_label")
        yield Label("YES", classes="price_label")
        yield Center(Digits("0.00", id="yes_digits", classes="price_yes"))
        yield Label("DEPTH: $0 | SLIP: 0bps", id="yes_liq_label", classes="spread_label")
        yield Label("")
        yield Label("NO", classes="price_label")
        yield Center(Digits("0.00", id="no_digits", classes="price_no"))
        yield Label("DEPTH: $0 | SLIP: 0bps", id="no_liq_label", classes="spread_label")
        yield Label("FV: --- | EDGE: ---", id="no_fv_label", classes="spread_label")
        yield Label("SPREAD: 0.00 / 0.00", id="no_spread_label", classes="spread_label")
    
    def watch_yes_price(self, value: float) -> None:
        """Update YES price display with edge coloring."""
        try:
            digits = self.query_one("#yes_digits", Digits)
            digits.update(f"{value:.2f}")
            # Edge coloring is handled in CSS based on class
        except Exception as e:
            logger.debug(f"Error updating YES price: {e}")
    
    def watch_no_price(self, value: float) -> None:
        """Update NO price display."""
        try:
            digits = self.query_one("#no_digits", Digits)
            digits.update(f"{value:.2f}")
        except Exception as e:
            logger.debug(f"Error updating NO price: {e}")
    
    def watch_yes_spread(self, value: str) -> None:
        """Update YES spread display."""
        try:
            label = self.query_one("#yes_spread_label", Label)
            label.update(f"SPREAD: {value}")
        except Exception as e:
            logger.debug(f"Error updating YES spread: {e}")
    
    def watch_no_spread(self, value: str) -> None:
        """Update NO spread display."""
        try:
            label = self.query_one("#no_spread_label", Label)
            label.update(f"SPREAD: {value}")
        except Exception as e:
            logger.debug(f"Error updating NO spread: {e}")
    
    def _format_edge(self, edge: Optional[EdgeDirection], edge_bps: float) -> str:
        """Format edge display with color."""
        if edge is None:
            return "[dim]---[/]"
        
        if edge == EdgeDirection.UNDERVALUED:
            return f"[green bold]+{abs(edge_bps):.0f}bps BUY[/]"
        elif edge == EdgeDirection.OVERVALUED:
            return f"[red bold]-{abs(edge_bps):.0f}bps SELL[/]"
        else:
            return f"[yellow]{abs(edge_bps):.0f}bps FAIR[/]"
    
    def update_analytics(self, snapshot: AnalyticsSnapshot) -> None:
        """Update panel with analytics snapshot."""
        self.fair_value_yes = snapshot.fair_value_yes
        self.fair_value_no = snapshot.fair_value_no
        self.edge_yes = snapshot.edge_yes
        self.edge_no = snapshot.edge_no
        self.edge_bps_yes = snapshot.edge_bps_yes or 0.0
        self.edge_bps_no = snapshot.edge_bps_no or 0.0
        self.yes_depth = snapshot.yes_ask_depth or 0.0
        self.no_depth = snapshot.no_ask_depth or 0.0
        self.slippage_yes = snapshot.estimated_slippage_yes or 0.0
        self.slippage_no = snapshot.estimated_slippage_no or 0.0
        
        # Update FV/Edge labels
        try:
            yes_fv_label = self.query_one("#yes_fv_label", Label)
            fv_str = f"{self.fair_value_yes:.2f}" if self.fair_value_yes else "---"
            edge_str = self._format_edge(self.edge_yes, self.edge_bps_yes)
            yes_fv_label.update(f"FV: {fv_str} | EDGE: {edge_str}")
        except Exception:
            pass
        
        try:
            no_fv_label = self.query_one("#no_fv_label", Label)
            fv_str = f"{self.fair_value_no:.2f}" if self.fair_value_no else "---"
            edge_str = self._format_edge(self.edge_no, self.edge_bps_no)
            no_fv_label.update(f"FV: {fv_str} | EDGE: {edge_str}")
        except Exception:
            pass
        
        # Update liquidity labels
        try:
            yes_liq_label = self.query_one("#yes_liq_label", Label)
            yes_liq_label.update(f"DEPTH: ${self.yes_depth:.0f} | SLIP: {self.slippage_yes:.0f}bps")
        except Exception:
            pass
        
        try:
            no_liq_label = self.query_one("#no_liq_label", Label)
            no_liq_label.update(f"DEPTH: ${self.no_depth:.0f} | SLIP: {self.slippage_no:.0f}bps")
        except Exception:
            pass


class StatsPanel(Static):
    """Panel displaying account statistics with real-time PnL."""
    
    balance = reactive(0.0)
    selected_size = reactive(1.0)
    yes_balance = reactive(0.0)
    no_balance = reactive(0.0)
    avg_entry_price_yes = reactive(None)
    avg_entry_price_no = reactive(None)
    
    # Real-time PnL
    unrealized_pnl = reactive(0.0)
    pnl_percentage = reactive(None)
    
    def render(self) -> str:
        """Render the stats panel with PnL."""
        # Format YES position with average entry price if there's a position
        yes_pos_str = f"Y:{self.yes_balance:.1f}"
        if self.yes_balance > 0 and self.avg_entry_price_yes is not None:
            yes_price_cents = int(self.avg_entry_price_yes * 100)
            yes_pos_str += f"@{yes_price_cents}c"
        
        # Format NO position with average entry price if there's a position
        no_pos_str = f"N:{self.no_balance:.1f}"
        if self.no_balance > 0 and self.avg_entry_price_no is not None:
            no_price_cents = int(self.avg_entry_price_no * 100)
            no_pos_str += f"@{no_price_cents}c"
        
        # Format PnL
        pnl_str = self._format_pnl()
        
        return f"[b #00ffff]══ ACCOUNT ══[/]\n" \
               f"CASH: [bold #00ff00]${self.balance:.2f}[/]\n" \
               f"SIZE: [bold #ffff00]${self.selected_size:.2f}[/]\n" \
               f"POS : [green]{yes_pos_str}[/]|[red]{no_pos_str}[/]\n" \
               f"PnL : {pnl_str}"
    
    def _format_pnl(self) -> str:
        """Format unrealized PnL with color."""
        if self.yes_balance == 0 and self.no_balance == 0:
            return "[dim]---[/]"
        
        sign = "+" if self.unrealized_pnl >= 0 else ""
        color = "green" if self.unrealized_pnl >= 0 else "red"
        
        pnl_str = f"[{color}]{sign}${self.unrealized_pnl:.2f}[/]"
        
        if self.pnl_percentage is not None:
            pct_sign = "+" if self.pnl_percentage >= 0 else ""
            pnl_str += f" [{color}]({pct_sign}{self.pnl_percentage:.1f}%)[/]"
        
        return pnl_str
    
    def update_analytics(self, snapshot: AnalyticsSnapshot) -> None:
        """Update panel with analytics snapshot."""
        self.unrealized_pnl = snapshot.total_unrealized_pnl or 0.0
        self.pnl_percentage = snapshot.pnl_percentage


class ProbabilityChart(Static):
    """Custom widget for probability history chart with fixed x-axis."""
    
    def __init__(self, *args, x_max: float = 900.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_max = x_max  # Fixed x-axis maximum (900 seconds)
        self.data: List[Tuple[float, float]] = []  # List of (x, y) tuples
    
    def update_data(self, data: List[Tuple[float, float]]) -> None:
        """Update the chart data and refresh.
        
        Filters, deduplicates by timestamp, and sorts the data.
        """
        # Filter to only include data within the fixed x-axis range (0 to x_max)
        filtered = [(x, y) for x, y in data if 0 <= x <= self.x_max]
        
        # Deduplicate by timestamp (keep last value for each timestamp)
        # This prevents rendering issues from duplicate points
        seen_timestamps: Dict[float, float] = {}
        for x, y in filtered:
            # Round to 1 decimal place to group nearby timestamps
            rounded_x = round(x, 1)
            seen_timestamps[rounded_x] = y
        
        # Convert back to sorted list
        self.data = sorted(seen_timestamps.items(), key=lambda p: p[0])
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
        num_ticks = len(y_ticks)
        
        for idx, y_val in enumerate(y_ticks):
            # Calculate y position evenly distributed across plot height
            # Map index to position: 0 -> top (plot_height-1), last -> bottom (0)
            # Use exact division to ensure even spacing
            if num_ticks > 1:
                plot_y = int((num_ticks - 1 - idx) / (num_ticks - 1) * (plot_height - 1))
            else:
                plot_y = plot_height // 2
            
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

