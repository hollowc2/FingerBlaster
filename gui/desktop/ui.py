"""PyQt6 UI components for the FingerBlaster application."""

import logging
from typing import List, Tuple, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from src.analytics import AnalyticsSnapshot, TimerUrgency, EdgeDirection

logger = logging.getLogger("FingerBlaster")


class MarketPanel(QFrame):
    """Panel displaying market context information with advanced analytics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: #00ffff;")
        
        layout = QVBoxLayout()
        layout.setSpacing(3)
        
        # Title
        title = QLabel("══ MARKET CONTEXT ══")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 13pt;")
        layout.addWidget(title)
        
        # Strike
        self.strike_label = QLabel("STRIKE: N/A")
        self.strike_label.setStyleSheet("color: #ffff00; font-size: 11pt;")
        layout.addWidget(self.strike_label)
        
        # BTC Price
        self.btc_label = QLabel("BTC: $0.00")
        self.btc_label.setStyleSheet("color: #00ff00; font-size: 11pt;")
        layout.addWidget(self.btc_label)
        
        # Oracle Lag
        self.oracle_label = QLabel("ORACLE: SYNC")
        self.oracle_label.setStyleSheet("color: #00ff00; font-size: 10pt;")
        layout.addWidget(self.oracle_label)
        
        # Prior Outcomes
        self.prior_label = QLabel("PRIOR: ---")
        self.prior_label.setStyleSheet("color: white; font-size: 10pt;")
        layout.addWidget(self.prior_label)
        
        # Regime
        self.regime_label = QLabel("REGIME: ---")
        self.regime_label.setStyleSheet("color: #888888; font-size: 10pt;")
        layout.addWidget(self.regime_label)
        
        # Sigma / Z-Score
        self.sigma_label = QLabel("SIGMA: ---")
        self.sigma_label.setStyleSheet("color: #00ffff; font-size: 11pt;")
        layout.addWidget(self.sigma_label)
        
        # Delta + BPS row
        self.delta_label = QLabel("DELTA: N/A (0bps)")
        self.delta_label.setStyleSheet("color: white; font-size: 11pt;")
        layout.addWidget(self.delta_label)
        
        # Time Left - larger for urgency
        self.time_label = QLabel("REMAIN: N/A")
        self.time_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14pt;")
        layout.addWidget(self.time_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # State
        self.strike = "N/A"
        self.ends = "N/A"
        self.btc_price = 0.0
        self.time_left = "N/A"
        self.prior_outcomes = ""
        self.timer_urgency = TimerUrgency.NORMAL
        self.basis_points = None
        
        # Blink timer for critical urgency
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_visible = True
    
    def _toggle_blink(self):
        """Toggle blink state for critical timer."""
        self.blink_visible = not self.blink_visible
        if self.timer_urgency == TimerUrgency.CRITICAL:
            if self.blink_visible:
                self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 16pt; background-color: #330000;")
            else:
                self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 16pt; background-color: #000000;")
    
    def update_strike(self, strike: str):
        """Update strike price."""
        self.strike = strike
        self.strike_label.setText(f"STRIKE: {strike}")
    
    def update_ends(self, ends: str):
        """Update end time."""
        self.ends = ends
    
    def update_btc_price(self, price: float):
        """Update BTC price."""
        self.btc_price = price
        self.btc_label.setText(f"BTC: ${price:,.2f}")
        self._update_delta()
    
    def update_time_left(self, time_str: str, urgency: TimerUrgency = None, seconds_remaining: int = 0):
        """Update time left with urgency-based styling."""
        if self.time_left == time_str and urgency == self.timer_urgency:
            return
        
        self.time_left = time_str
        if urgency:
            self.timer_urgency = urgency
        
        # Stop blink timer if running
        if self.blink_timer.isActive():
            self.blink_timer.stop()
        
        if time_str == "EXPIRED":
            self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 16pt;")
            self.time_label.setText("⚠ EXPIRED ⚠")
        elif self.timer_urgency == TimerUrgency.CRITICAL:
            self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 16pt; background-color: #330000;")
            self.time_label.setText(f"⚠ {time_str} ⚠")
            self.blink_timer.start(300)  # Blink every 300ms
        elif self.timer_urgency == TimerUrgency.WATCHFUL:
            self.time_label.setStyleSheet("color: #ff8800; font-weight: bold; font-size: 14pt;")
            self.time_label.setText(f"REMAIN: {time_str}")
        else:
            self.time_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14pt;")
            self.time_label.setText(f"REMAIN: {time_str}")
    
    def update_prior_outcomes(self, outcomes: str):
        """Update prior outcomes."""
        self.prior_outcomes = outcomes
        self.prior_label.setText(f"PRIOR: {outcomes}")
    
    def _update_delta(self):
        """Calculate and update delta with basis points."""
        try:
            strike_val = float(str(self.strike).replace(',', '').replace('$', '').strip())
            diff = self.btc_price - strike_val
            symbol = "▲" if diff >= 0 else "▼"
            color = "#00ff00" if diff >= 0 else "#ff0000"
            
            bps_str = ""
            if self.basis_points is not None:
                sign = "+" if self.basis_points >= 0 else ""
                bps_str = f" ({sign}{self.basis_points:.0f}bps)"
            
            self.delta_label.setText(f"DELTA: {symbol}${abs(diff):,.0f}{bps_str}")
            self.delta_label.setStyleSheet(f"color: {color}; font-size: 11pt;")
        except (ValueError, AttributeError, TypeError):
            self.delta_label.setText("DELTA: N/A")
            self.delta_label.setStyleSheet("color: white; font-size: 11pt;")
    
    def update_analytics(self, snapshot: AnalyticsSnapshot):
        """Update panel with analytics snapshot."""
        self.basis_points = snapshot.basis_points
        self._update_delta()
        
        # Sigma/Z-Score
        if snapshot.z_score is not None and snapshot.sigma_label:
            color = "#00ff00" if snapshot.z_score >= 0 else "#ff0000"
            self.sigma_label.setText(f"SIGMA: {snapshot.sigma_label}")
            self.sigma_label.setStyleSheet(f"color: {color}; font-size: 11pt; font-weight: bold;")
        else:
            self.sigma_label.setText("SIGMA: ---")
            self.sigma_label.setStyleSheet("color: #888888; font-size: 11pt;")
        
        # Regime
        if snapshot.regime_direction and snapshot.regime_strength > 0:
            if snapshot.regime_direction == "BULLISH":
                color = "#00ff00"
            elif snapshot.regime_direction == "BEARISH":
                color = "#ff0000"
            else:
                color = "#ffff00"
            self.regime_label.setText(f"REGIME: {snapshot.regime_strength:.0f}% {snapshot.regime_direction}")
            self.regime_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
        else:
            self.regime_label.setText("REGIME: ---")
            self.regime_label.setStyleSheet("color: #888888; font-size: 10pt;")
        
        # Oracle Lag
        if snapshot.oracle_lag_ms is not None:
            if snapshot.oracle_lag_ms < 500:
                self.oracle_label.setText(f"ORACLE: {snapshot.oracle_lag_ms}ms")
                self.oracle_label.setStyleSheet("color: #00ff00; font-size: 10pt;")
            elif snapshot.oracle_lag_ms < 2000:
                self.oracle_label.setText(f"ORACLE: {snapshot.oracle_lag_ms}ms")
                self.oracle_label.setStyleSheet("color: #ff8800; font-size: 10pt;")
            else:
                self.oracle_label.setText(f"⚠ ORACLE LAG: {snapshot.oracle_lag_ms}ms")
                self.oracle_label.setStyleSheet("color: #ff0000; font-size: 10pt; font-weight: bold;")
        else:
            self.oracle_label.setText("ORACLE: SYNC")
            self.oracle_label.setStyleSheet("color: #00ff00; font-size: 10pt;")


class PricePanel(QFrame):
    """Panel displaying live YES/NO prices, spread, and edge detection."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Title
        title = QLabel("═══ LIVE PRICES ═══")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 13pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)
        
        # YES Section
        yes_label = QLabel("YES")
        yes_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00ff00;")
        yes_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(yes_label)
        
        self.yes_price_label = QLabel("0.00")
        self.yes_price_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 28pt;")
        self.yes_price_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.yes_price_label)
        
        # YES Fair Value / Edge
        self.yes_fv_label = QLabel("FV: --- | EDGE: ---")
        self.yes_fv_label.setStyleSheet("color: #888888; font-size: 9pt;")
        self.yes_fv_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.yes_fv_label)
        
        # YES Liquidity
        self.yes_liq_label = QLabel("DEPTH: $0 | SLIP: 0bps")
        self.yes_liq_label.setStyleSheet("color: #666666; font-size: 9pt;")
        self.yes_liq_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.yes_liq_label)
        
        # YES Spread
        self.yes_spread_label = QLabel("SPREAD: 0.00 / 0.00")
        self.yes_spread_label.setStyleSheet("color: #888888; font-size: 9pt;")
        self.yes_spread_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.yes_spread_label)
        
        # Separator
        sep = QLabel("─" * 30)
        sep.setStyleSheet("color: #333333; font-size: 8pt;")
        sep.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(sep)
        
        # NO Section
        no_label = QLabel("NO")
        no_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #ff0000;")
        no_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(no_label)
        
        self.no_price_label = QLabel("0.00")
        self.no_price_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 28pt;")
        self.no_price_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.no_price_label)
        
        # NO Fair Value / Edge
        self.no_fv_label = QLabel("FV: --- | EDGE: ---")
        self.no_fv_label.setStyleSheet("color: #888888; font-size: 9pt;")
        self.no_fv_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.no_fv_label)
        
        # NO Liquidity
        self.no_liq_label = QLabel("DEPTH: $0 | SLIP: 0bps")
        self.no_liq_label.setStyleSheet("color: #666666; font-size: 9pt;")
        self.no_liq_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.no_liq_label)
        
        # NO Spread
        self.no_spread_label = QLabel("SPREAD: 0.00 / 0.00")
        self.no_spread_label.setStyleSheet("color: #888888; font-size: 9pt;")
        self.no_spread_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.no_spread_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # State for edge coloring
        self.edge_yes = None
        self.edge_no = None
    
    def update_prices(self, yes_price: float, no_price: float, yes_spread: str, no_spread: str):
        """Update price displays with edge-based coloring."""
        # Update prices with edge coloring
        yes_color = self._get_edge_color(self.edge_yes, "#00ff00")
        no_color = self._get_edge_color(self.edge_no, "#ff0000")
        
        self.yes_price_label.setText(f"{yes_price:.2f}")
        self.yes_price_label.setStyleSheet(f"color: {yes_color}; font-weight: bold; font-size: 28pt;")
        
        self.no_price_label.setText(f"{no_price:.2f}")
        self.no_price_label.setStyleSheet(f"color: {no_color}; font-weight: bold; font-size: 28pt;")
        
        self.yes_spread_label.setText(f"SPREAD: {yes_spread}")
        self.no_spread_label.setText(f"SPREAD: {no_spread}")
    
    def _get_edge_color(self, edge: Optional[EdgeDirection], default: str) -> str:
        """Get color based on edge direction."""
        if edge == EdgeDirection.UNDERVALUED:
            return "#00ff88"  # Bright green for buy signal
        elif edge == EdgeDirection.OVERVALUED:
            return "#ff4444"  # Dimmer red for sell signal
        return default
    
    def _format_edge_html(self, edge: Optional[EdgeDirection], edge_bps: float) -> str:
        """Format edge as HTML with color."""
        if edge is None:
            return '<span style="color: #888888;">---</span>'
        
        if edge == EdgeDirection.UNDERVALUED:
            return f'<span style="color: #00ff00; font-weight: bold;">+{abs(edge_bps):.0f}bps BUY</span>'
        elif edge == EdgeDirection.OVERVALUED:
            return f'<span style="color: #ff0000; font-weight: bold;">-{abs(edge_bps):.0f}bps SELL</span>'
        else:
            return f'<span style="color: #ffff00;">{abs(edge_bps):.0f}bps FAIR</span>'
    
    def update_analytics(self, snapshot: AnalyticsSnapshot):
        """Update panel with analytics snapshot."""
        self.edge_yes = snapshot.edge_yes
        self.edge_no = snapshot.edge_no
        
        # Update YES FV/Edge
        fv_yes = f"{snapshot.fair_value_yes:.2f}" if snapshot.fair_value_yes else "---"
        edge_yes_html = self._format_edge_html(snapshot.edge_yes, snapshot.edge_bps_yes or 0)
        self.yes_fv_label.setText(f"FV: {fv_yes} | EDGE: ")
        # Use rich text for edge coloring
        self.yes_fv_label.setTextFormat(Qt.TextFormat.RichText)
        self.yes_fv_label.setText(f'FV: {fv_yes} | EDGE: {edge_yes_html}')
        
        # Update NO FV/Edge
        fv_no = f"{snapshot.fair_value_no:.2f}" if snapshot.fair_value_no else "---"
        edge_no_html = self._format_edge_html(snapshot.edge_no, snapshot.edge_bps_no or 0)
        self.no_fv_label.setTextFormat(Qt.TextFormat.RichText)
        self.no_fv_label.setText(f'FV: {fv_no} | EDGE: {edge_no_html}')
        
        # Update liquidity
        yes_depth = snapshot.yes_ask_depth or 0
        no_depth = snapshot.no_ask_depth or 0
        slip_yes = snapshot.estimated_slippage_yes or 0
        slip_no = snapshot.estimated_slippage_no or 0
        
        self.yes_liq_label.setText(f"DEPTH: ${yes_depth:.0f} | SLIP: {slip_yes:.0f}bps")
        self.no_liq_label.setText(f"DEPTH: ${no_depth:.0f} | SLIP: {slip_no:.0f}bps")


class StatsPanel(QFrame):
    """Panel displaying account statistics with real-time PnL."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Title
        title = QLabel("══ ACCOUNT ══")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 13pt;")
        layout.addWidget(title)
        
        # Cash Balance
        self.balance_label = QLabel("CASH: $0.00")
        self.balance_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.balance_label)
        
        # Size
        self.size_label = QLabel("SIZE: $1.00")
        self.size_label.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.size_label)
        
        # Positions
        self.pos_label = QLabel("POS: Y:0.0 | N:0.0")
        self.pos_label.setStyleSheet("color: white; font-size: 11pt;")
        self.pos_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.pos_label)
        
        # Real-time PnL
        self.pnl_label = QLabel("PnL: ---")
        self.pnl_label.setStyleSheet("color: #888888; font-size: 12pt; font-weight: bold;")
        layout.addWidget(self.pnl_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # State for analytics
        self.unrealized_pnl = 0.0
        self.pnl_percentage = None
    
    def update_stats(self, balance: float, yes_balance: float, no_balance: float, size: float,
                     avg_entry_price_yes: float = 0.0, avg_entry_price_no: float = 0.0):
        """Update account statistics."""
        self.balance_label.setText(f"CASH: ${balance:.2f}")
        self.size_label.setText(f"SIZE: ${size:.2f}")
        
        # Format positions with average entry prices and colors
        yes_pos_str = f"Y:{yes_balance:.1f}"
        if yes_balance > 0 and avg_entry_price_yes > 0:
            yes_price_cents = int(avg_entry_price_yes * 100)
            yes_pos_str += f"@{yes_price_cents}c"
        
        no_pos_str = f"N:{no_balance:.1f}"
        if no_balance > 0 and avg_entry_price_no > 0:
            no_price_cents = int(avg_entry_price_no * 100)
            no_pos_str += f"@{no_price_cents}c"
        
        self.pos_label.setText(
            f'POS: <span style="color: #00ff00;">{yes_pos_str}</span> | '
            f'<span style="color: #ff0000;">{no_pos_str}</span>'
        )
        
        # Update PnL display
        self._update_pnl_display()
    
    def update_size_only(self, size: float):
        """Update only the size display immediately."""
        self.size_label.setText(f"SIZE: ${size:.2f}")
    
    def _update_pnl_display(self):
        """Update PnL display with color coding."""
        if self.unrealized_pnl == 0 and self.pnl_percentage is None:
            self.pnl_label.setText("PnL: ---")
            self.pnl_label.setStyleSheet("color: #888888; font-size: 12pt; font-weight: bold;")
            return
        
        sign = "+" if self.unrealized_pnl >= 0 else ""
        color = "#00ff00" if self.unrealized_pnl >= 0 else "#ff0000"
        
        pnl_text = f"PnL: {sign}${self.unrealized_pnl:.2f}"
        if self.pnl_percentage is not None:
            pct_sign = "+" if self.pnl_percentage >= 0 else ""
            pnl_text += f" ({pct_sign}{self.pnl_percentage:.1f}%)"
        
        self.pnl_label.setText(pnl_text)
        self.pnl_label.setStyleSheet(f"color: {color}; font-size: 12pt; font-weight: bold;")
    
    def update_analytics(self, snapshot: AnalyticsSnapshot):
        """Update panel with analytics snapshot."""
        self.unrealized_pnl = snapshot.total_unrealized_pnl or 0.0
        self.pnl_percentage = snapshot.pnl_percentage
        self._update_pnl_display()


class ProbabilityChart(QWidget):
    """Custom widget for probability history chart with fixed x-axis.
    
    Optimized to reuse line objects instead of full redraws.
    """
    
    def __init__(self, parent=None, x_max: float = 900.0):
        super().__init__(parent)
        self.x_max = x_max
        self.data: List[Tuple[float, float]] = []
        
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 4), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='black')
        self._setup_axes()
        
        # Cached line object for efficient updates
        self._prob_line = None
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def _setup_axes(self):
        """Setup axis styling (called once on init)."""
        self.ax.set_xlim(1, 15)  # Fixed: 1-15 minutes
        self.ax.set_ylim(0, 1.0)  # Fixed: 0-100% probability
        self.ax.set_xlabel('', color='white')
        self.ax.set_ylabel('Probability', color='white')
        self.ax.tick_params(colors='white')
        self.ax.set_xticks([])
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for spine in ['bottom', 'top', 'left', 'right']:
            self.ax.spines[spine].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('PROBABILITY HISTORY', color='#00ffff', fontweight='bold')
    
    def update_data(self, data: List[Tuple[float, float]]):
        """Update the chart data efficiently.
        
        Uses set_data() on existing line object instead of full redraws.
        """
        if len(data) < 2:
            return
        
        self.data = sorted([(x, y) for x, y in data if 0 <= x <= self.x_max], key=lambda p: p[0])
        
        if len(self.data) < 2:
            return
        
        x_vals = [p[0] / 60.0 for p in self.data]  # Convert to minutes
        y_vals = [p[1] for p in self.data]
        
        # Update or create probability line
        if self._prob_line is None:
            self._prob_line, = self.ax.plot(x_vals, y_vals, color='cyan', linewidth=2)
        else:
            self._prob_line.set_data(x_vals, y_vals)
        
        # Use draw_idle for better performance
        self.canvas.draw_idle()


class BTCChart(QWidget):
    """Widget for BTC price history chart.
    
    Optimized to reuse line objects instead of full redraws.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 4), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='black')
        self._setup_axes()
        
        # Cached line objects for efficient updates
        self._btc_line = None
        self._strike_line = None
        self._last_ylim = (0, 1)  # Track y-axis limits to avoid unnecessary updates
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def _setup_axes(self):
        """Setup axis styling (called once on init)."""
        self.ax.set_xlabel('', color='white')
        self.ax.set_ylabel('Price ($)', color='white')
        self.ax.tick_params(colors='white')
        self.ax.set_xticks([])
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for spine in ['bottom', 'top', 'left', 'right']:
            self.ax.spines[spine].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('BTC PRICE HISTORY', color='#00ffff', fontweight='bold')
    
    def update_data(self, prices: List[float], strike_val: Optional[float] = None):
        """Update BTC chart data efficiently.
        
        Uses set_data() on existing line objects instead of full redraws.
        """
        if len(prices) < 2:
            return
        
        x_data = list(range(len(prices)))
        
        # Calculate y-axis limits
        y_min, y_max = min(prices), max(prices)
        if strike_val is not None:
            y_min = min(y_min, strike_val)
            y_max = max(y_max, strike_val)
        
        spread = y_max - y_min
        padding = spread * 0.25 if spread > 0 else 50.0
        new_ylim = (y_min - padding, y_max + padding)
        
        # Update or create BTC line
        if self._btc_line is None:
            self._btc_line, = self.ax.plot(x_data, prices, color='cyan', linewidth=2, label='BTC')
        else:
            self._btc_line.set_data(x_data, prices)
        
        # Update or create strike line
        if strike_val is not None:
            if self._strike_line is None:
                self._strike_line = self.ax.axhline(
                    y=strike_val, color='yellow', linestyle='--', linewidth=2, label='STRIKE'
                )
            else:
                self._strike_line.set_ydata([strike_val, strike_val])
        elif self._strike_line is not None:
            self._strike_line.remove()
            self._strike_line = None
        
        # Only update y-limits if they changed significantly (>1% change)
        if abs(new_ylim[0] - self._last_ylim[0]) > abs(self._last_ylim[0]) * 0.01 or \
           abs(new_ylim[1] - self._last_ylim[1]) > abs(self._last_ylim[1]) * 0.01:
            self.ax.set_ylim(new_ylim)
            self._last_ylim = new_ylim
        
        # Update x-limits to match data
        self.ax.set_xlim(0, len(prices) - 1)
        
        # Use draw_idle for better performance (schedules redraw)
        self.canvas.draw_idle()


class ResolutionOverlay(QWidget):
    """Full-screen overlay showing market resolution with green/red flash."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make it a child widget that overlays the parent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.raise_()  # Bring to front
        self.hide()
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.resolution_label = QLabel("")
        self.resolution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resolution_label.setStyleSheet("font-size: 72pt; font-weight: bold;")
        layout.addWidget(self.resolution_label)
        
        market_label = QLabel("MARKET RESOLVED")
        market_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        market_label.setStyleSheet("font-size: 24pt; font-weight: bold;")
        layout.addWidget(market_label)
        
        self.setLayout(layout)
    
    def show_resolution(self, resolution: str):
        """Show the overlay with the given resolution (green for YES, red for NO)."""
        self.resolution_label.setText(resolution)
        if resolution == "YES":
            # Green flash like terminal app
            self.setStyleSheet("background-color: #00ff00; color: black;")
            self.resolution_label.setStyleSheet("font-size: 72pt; font-weight: bold; color: black;")
        else:
            # Red flash like terminal app
            self.setStyleSheet("background-color: #ff0000; color: white;")
            self.resolution_label.setStyleSheet("font-size: 72pt; font-weight: bold; color: white;")
        self.raise_()  # Ensure it's on top
        self.show()
    
    def hide_resolution(self):
        """Hide the overlay."""
        self.hide()

