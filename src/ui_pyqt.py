"""PyQt6 UI components for the FingerBlaster application."""

import logging
from typing import List, Tuple, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger("FingerBlaster")


class MarketPanel(QFrame):
    """Panel displaying market context information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: #00ffff;")
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Title
        title = QLabel("MARKET CONTEXT")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 14pt;")
        layout.addWidget(title)
        
        # Strike
        self.strike_label = QLabel("STRIKE: N/A")
        self.strike_label.setStyleSheet("color: #ffff00; font-size: 12pt;")
        layout.addWidget(self.strike_label)
        
        # Ends
        self.ends_label = QLabel("ENDS: N/A")
        self.ends_label.setStyleSheet("color: #ffff00; font-size: 12pt;")
        layout.addWidget(self.ends_label)
        
        # BTC Price
        self.btc_label = QLabel("BTC: $0.00")
        self.btc_label.setStyleSheet("color: #00ff00; font-size: 12pt;")
        layout.addWidget(self.btc_label)
        
        # Prior Outcomes
        self.prior_label = QLabel("PRIOR: ---")
        self.prior_label.setStyleSheet("color: white; font-size: 11pt;")
        layout.addWidget(self.prior_label)
        
        # Delta
        self.delta_label = QLabel("DELTA: N/A")
        self.delta_label.setStyleSheet("color: white; font-size: 11pt;")
        layout.addWidget(self.delta_label)
        
        # Time Left
        self.time_label = QLabel("REMAIN: N/A")
        self.time_label.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.time_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # State
        self.strike = "N/A"
        self.ends = "N/A"
        self.btc_price = 0.0
        self.time_left = "N/A"
        self.prior_outcomes = ""
    
    def update_strike(self, strike: str):
        """Update strike price."""
        self.strike = strike
        self.strike_label.setText(f"STRIKE: {strike}")
    
    def update_ends(self, ends: str):
        """Update end time."""
        self.ends = ends
        self.ends_label.setText(f"ENDS: {ends}")
    
    def update_btc_price(self, price: float):
        """Update BTC price."""
        self.btc_price = price
        self.btc_label.setText(f"BTC: ${price:,.2f}")
        self._update_delta()
    
    def update_time_left(self, time_str: str):
        """Update time left."""
        self.time_left = time_str
        if time_str == "EXPIRED":
            self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 12pt;")
        elif time_str != "N/A":
            try:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    if minutes < 2:
                        self.time_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 12pt;")
                    else:
                        self.time_label.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 12pt;")
            except (ValueError, AttributeError):
                pass
        self.time_label.setText(f"REMAIN: {time_str}")
    
    def update_prior_outcomes(self, outcomes: str):
        """Update prior outcomes."""
        self.prior_outcomes = outcomes
        self.prior_label.setText(f"PRIOR: {outcomes}")
    
    def _update_delta(self):
        """Calculate and update delta."""
        try:
            strike_val = float(str(self.strike).replace(',', '').replace('$', '').strip())
            diff = self.btc_price - strike_val
            symbol = "▲" if diff >= 0 else "▼"
            color = "#00ff00" if diff >= 0 else "#ff0000"
            self.delta_label.setText(f"DELTA: {symbol} ${abs(diff):,.2f}")
            self.delta_label.setStyleSheet(f"color: {color}; font-size: 11pt;")
        except (ValueError, AttributeError, TypeError):
            self.delta_label.setText("DELTA: N/A")
            self.delta_label.setStyleSheet("color: white; font-size: 11pt;")


class PricePanel(QFrame):
    """Panel displaying live YES/NO prices and spread."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel("LIVE PRICES")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 14pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # YES Price
        yes_label = QLabel("YES")
        yes_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        yes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(yes_label)
        
        self.yes_price_label = QLabel("0.00")
        self.yes_price_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 24pt;")
        self.yes_price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.yes_price_label)
        
        # NO Price
        no_label = QLabel("NO")
        no_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        no_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(no_label)
        
        self.no_price_label = QLabel("0.00")
        self.no_price_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 24pt;")
        self.no_price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.no_price_label)
        
        # Spread
        self.spread_label = QLabel("SPREAD: 0.00 / 0.00")
        self.spread_label.setStyleSheet("color: #888888; font-size: 10pt;")
        self.spread_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.spread_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_prices(self, yes_price: float, no_price: float, spread: str):
        """Update price displays."""
        self.yes_price_label.setText(f"{yes_price:.2f}")
        self.no_price_label.setText(f"{no_price:.2f}")
        self.spread_label.setText(f"SPREAD: {spread}")


class StatsPanel(QFrame):
    """Panel displaying account statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Title
        title = QLabel("ACCOUNT STATS")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 14pt;")
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
        layout.addWidget(self.pos_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_stats(self, balance: float, yes_balance: float, no_balance: float, size: float):
        """Update account statistics."""
        self.balance_label.setText(f"CASH: ${balance:.2f}")
        self.size_label.setText(f"SIZE: ${size:.2f}")
        self.pos_label.setText(f"POS: Y:{yes_balance:.1f} | N:{no_balance:.1f}")
    
    def update_size_only(self, size: float):
        """Update only the size display immediately."""
        self.size_label.setText(f"SIZE: ${size:.2f}")


class ProbabilityChart(QWidget):
    """Custom widget for probability history chart with fixed x-axis."""
    
    def __init__(self, parent=None, x_max: float = 900.0):
        super().__init__(parent)
        self.x_max = x_max
        self.data: List[Tuple[float, float]] = []
        
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 4), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='black')
        self.ax.set_xlim(1, 15)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_xticks(range(1, 16))  # 1-15 minutes, 1 minute increments
        self.ax.set_xlabel('Time (minutes)', color='white')
        self.ax.set_ylabel('Probability', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('PROBABILITY HISTORY', color='#00ffff', fontweight='bold')
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def update_data(self, data: List[Tuple[float, float]]):
        """Update the chart data."""
        if len(data) < 2:
            return
        
        self.data = sorted([(x, y) for x, y in data if 0 <= x <= self.x_max], key=lambda p: p[0])
        
        if len(self.data) < 2:
            return
        
        self.ax.clear()
        self.ax.set_xlim(1, 15)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_xticks(range(1, 16))  # 1-15 minutes, 1 minute increments
        self.ax.set_xlabel('Time (minutes)', color='white')
        self.ax.set_ylabel('Probability', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('PROBABILITY HISTORY', color='#00ffff', fontweight='bold')
        
        x_vals = [p[0] / 60.0 for p in self.data]  # Convert to minutes
        y_vals = [p[1] for p in self.data]
        
        self.ax.plot(x_vals, y_vals, color='#00ff00', linewidth=2)
        self.canvas.draw()


class BTCChart(QWidget):
    """Widget for BTC price history chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 4), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='black')
        self.ax.set_xlabel('', color='white')
        self.ax.set_ylabel('Price ($)', color='white')
        self.ax.tick_params(colors='white')
        # Remove x-axis ticks and labels
        self.ax.set_xticks([])
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('BTC PRICE HISTORY', color='#00ffff', fontweight='bold')
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def update_data(self, prices: List[float], strike_val: Optional[float] = None):
        """Update BTC chart data."""
        if len(prices) < 2:
            return
        
        self.ax.clear()
        self.ax.set_xlabel('', color='white')
        self.ax.set_ylabel('Price ($)', color='white')
        self.ax.tick_params(colors='white')
        # Remove x-axis ticks and labels
        self.ax.set_xticks([])
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_title('BTC PRICE HISTORY', color='#00ffff', fontweight='bold')
        
        y_min, y_max = min(prices), max(prices)
        if strike_val is not None:
            y_min = min(y_min, strike_val)
            y_max = max(y_max, strike_val)
        
        spread = y_max - y_min
        padding = spread * 0.25 if spread > 0 else 50.0
        
        self.ax.set_ylim(y_min - padding, y_max + padding)
        self.ax.plot(prices, color='cyan', linewidth=2, label='BTC')
        
        if strike_val is not None:
            self.ax.axhline(y=strike_val, color='yellow', linestyle='--', linewidth=2, label='STRIKE')
        
        self.canvas.draw()


class ResolutionOverlay(QWidget):
    """Full-screen overlay showing market resolution."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
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
        """Show the overlay with the given resolution."""
        self.resolution_label.setText(resolution)
        if resolution == "YES":
            self.setStyleSheet("background-color: rgba(0, 255, 0, 200); color: black;")
            self.resolution_label.setStyleSheet("font-size: 72pt; font-weight: bold; color: black;")
        else:
            self.setStyleSheet("background-color: rgba(255, 0, 0, 200); color: white;")
            self.resolution_label.setStyleSheet("font-size: 72pt; font-weight: bold; color: white;")
        self.show()
    
    def hide_resolution(self):
        """Hide the overlay."""
        self.hide()

