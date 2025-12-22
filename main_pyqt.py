"""PyQt6 desktop UI entry point for FingerBlaster application."""

import asyncio
import logging
import os
import sys
import threading
from typing import Optional

# Set Qt platform plugin path before importing Qt
# This helps with some Linux systems that have plugin loading issues
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    # Try to find Qt plugins in common locations
    possible_paths = [
        '/usr/lib/x86_64-linux-gnu/qt6/plugins',
        '/usr/lib/qt6/plugins',
        '/usr/local/lib/qt6/plugins',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = path
            break

# Try to use xcb platform, fallback to wayland or offscreen if needed
if 'QT_QPA_PLATFORM' not in os.environ:
    # xcb is the default for X11, but we'll let Qt auto-detect
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QLabel, QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut, QKeyEvent
from PyQt6.QtCore import QObject, QEvent

from src.core import FingerBlasterCore
from src.config import AppConfig
from src.ui_pyqt import (
    MarketPanel, PricePanel, StatsPanel, ProbabilityChart, BTCChart, ResolutionOverlay
)

logger = logging.getLogger("FingerBlaster")


# Global asyncio event loop for Qt integration
_qt_event_loop = None
_qt_event_loop_thread = None


def setup_qt_asyncio():
    """Setup asyncio event loop in a separate thread for Qt integration."""
    global _qt_event_loop, _qt_event_loop_thread
    
    def run_event_loop():
        global _qt_event_loop
        _qt_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_qt_event_loop)
        _qt_event_loop.run_forever()
    
    if _qt_event_loop_thread is None or not _qt_event_loop_thread.is_alive():
        _qt_event_loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        _qt_event_loop_thread.start()
        # Wait a bit for the loop to start
        import time
        time.sleep(0.1)


def run_async_task(coro):
    """Run an async coroutine from a Qt callback."""
    global _qt_event_loop
    if _qt_event_loop is None:
        setup_qt_asyncio()
        # Wait a bit more for the loop to be ready
        import time
        time.sleep(0.2)
    
    if _qt_event_loop and _qt_event_loop.is_running():
        # Use run_coroutine_threadsafe to schedule in the background thread's loop
        asyncio.run_coroutine_threadsafe(coro, _qt_event_loop)
    else:
        # Fallback: try to get or create a loop in current thread
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop - this shouldn't happen, but log it
            logger.warning("No asyncio event loop available for async task")
            # Try to create task in the Qt loop anyway
            if _qt_event_loop:
                asyncio.run_coroutine_threadsafe(coro, _qt_event_loop)


class FingerBlasterPyQtApp(QMainWindow):
    """Main PyQt6 application window."""
    
    def __init__(self):
        super().__init__()
        self.core = FingerBlasterCore()
        self.config = self.core.config
        
        # UI state
        self.graphs_visible = True
        self.log_visible = True
        
        self.init_ui()
        self.setup_callbacks()
        self.setup_timers()
        self.setup_shortcuts()
        
        # Check prior outcomes after a delay
        QTimer.singleShot(3000, lambda: run_async_task(self.core._check_and_add_prior_outcomes()))
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("FINGER BLASTER v2.0 (Desktop Mode)")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        # Set focus policy so the window can receive key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top section: Market info and prices
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left cockpit
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        
        self.market_panel = MarketPanel()
        left_layout.addWidget(self.market_panel)
        
        self.price_panel = PricePanel()
        left_layout.addWidget(self.price_panel)
        
        self.stats_panel = StatsPanel()
        left_layout.addWidget(self.stats_panel)
        
        # Action buttons panel
        buttons_label = QLabel("ACTIONS")
        buttons_label.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 12pt; padding: 5px;")
        left_layout.addWidget(buttons_label)
        
        # Create button grid
        buttons_widget = QWidget()
        buttons_grid = QVBoxLayout()
        buttons_widget.setLayout(buttons_grid)
        
        # Row 1: Buy buttons
        row1 = QHBoxLayout()
        self.buy_yes_btn = QPushButton("Y: BUY YES")
        self.buy_yes_btn.clicked.connect(self.buy_yes)
        row1.addWidget(self.buy_yes_btn)
        
        self.buy_no_btn = QPushButton("N: BUY NO")
        self.buy_no_btn.clicked.connect(self.buy_no)
        row1.addWidget(self.buy_no_btn)
        buttons_grid.addLayout(row1)
        
        # Row 2: Flatten and Cancel
        row2 = QHBoxLayout()
        self.flatten_btn = QPushButton("F: FLATTEN")
        self.flatten_btn.clicked.connect(self.flatten)
        row2.addWidget(self.flatten_btn)
        
        self.cancel_btn = QPushButton("C: CANCEL")
        self.cancel_btn.clicked.connect(self.cancel_all)
        row2.addWidget(self.cancel_btn)
        buttons_grid.addLayout(row2)
        
        # Row 3: Size buttons
        row3 = QHBoxLayout()
        self.size_up_btn = QPushButton("+: SIZE UP")
        self.size_up_btn.clicked.connect(self.size_up)
        row3.addWidget(self.size_up_btn)
        
        self.size_down_btn = QPushButton("-: SIZE DOWN")
        self.size_down_btn.clicked.connect(self.size_down)
        row3.addWidget(self.size_down_btn)
        buttons_grid.addLayout(row3)
        
        # Row 4: Toggle buttons
        row4 = QHBoxLayout()
        self.toggle_graphs_btn = QPushButton("H: TOGGLE GRAPHS")
        self.toggle_graphs_btn.clicked.connect(self.toggle_graphs)
        row4.addWidget(self.toggle_graphs_btn)
        
        self.toggle_log_btn = QPushButton("L: TOGGLE LOG")
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        row4.addWidget(self.toggle_log_btn)
        buttons_grid.addLayout(row4)
        
        # Row 5: Help and Quit
        row5 = QHBoxLayout()
        self.help_button = QPushButton("? HELP")
        self.help_button.clicked.connect(self.show_help)
        row5.addWidget(self.help_button)
        
        self.quit_btn = QPushButton("Q: QUIT")
        self.quit_btn.clicked.connect(self.quit_app)
        row5.addWidget(self.quit_btn)
        buttons_grid.addLayout(row5)
        
        # Style all buttons
        button_style = """
            QPushButton {
                background-color: #1a1a1a;
                color: #00ffff;
                border: 1px solid #00ffff;
                font-weight: bold;
                font-size: 10pt;
                padding: 6px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #00ffff;
                color: #000000;
            }
            QPushButton:pressed {
                background-color: #00cccc;
            }
        """
        for row in [row1, row2, row3, row4, row5]:
            for i in range(row.count()):
                item = row.itemAt(i)
                if item and item.widget():
                    item.widget().setStyleSheet(button_style)
        
        left_layout.addWidget(buttons_widget)
        
        left_widget.setLayout(left_layout)
        top_splitter.addWidget(left_widget)
        
        # Charts section
        charts_widget = QWidget()
        charts_layout = QVBoxLayout()
        charts_layout.setSpacing(5)
        
        self.probability_chart = ProbabilityChart(x_max=float(self.config.market_duration_seconds))
        charts_layout.addWidget(self.probability_chart)
        
        self.btc_chart = BTCChart()
        charts_layout.addWidget(self.btc_chart)
        
        charts_widget.setLayout(charts_layout)
        self.charts_widget = charts_widget
        top_splitter.addWidget(charts_widget)
        
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(top_splitter, 1)
        
        # Log panel
        self.log_panel = QTextEdit()
        self.log_panel_height = 150
        self.log_panel.setMaximumHeight(self.log_panel_height)
        self.log_panel.setMinimumHeight(self.log_panel_height)
        self.log_panel.setStyleSheet("background-color: #000000; color: #00ffff; border: 2px solid white;")
        self.log_panel.setReadOnly(True)
        # Prevent log panel from accepting keyboard focus so shortcuts work
        self.log_panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Install event filter only on log panel to prevent it from processing shortcut keys
        self.log_panel.installEventFilter(self)
        main_layout.addWidget(self.log_panel)
        
        # Resolution overlay (will be resized on show)
        self.resolution_overlay = ResolutionOverlay(self)
    
    def setup_callbacks(self):
        """Setup callbacks from core."""
        self.core.register_callback('market_update', self._on_market_update)
        self.core.register_callback('btc_price_update', self._on_btc_price_update)
        self.core.register_callback('price_update', self._on_price_update)
        self.core.register_callback('account_stats_update', self._on_account_stats_update)
        self.core.register_callback('countdown_update', self._on_countdown_update)
        self.core.register_callback('prior_outcomes_update', self._on_prior_outcomes_update)
        self.core.register_callback('resolution', self._on_resolution)
        self.core.register_callback('log', self._on_log)
        self.core.register_callback('chart_update', self._on_chart_update)
    
    def setup_timers(self):
        """Setup update timers."""
        # Market status timer
        self.market_timer = QTimer()
        self.market_timer.timeout.connect(
            lambda: run_async_task(self.core.update_market_status())
        )
        self.market_timer.start(int(self.config.market_status_interval * 1000))
        
        # BTC price timer
        self.btc_timer = QTimer()
        self.btc_timer.timeout.connect(
            lambda: run_async_task(self.core.update_btc_price())
        )
        self.btc_timer.start(int(self.config.btc_price_interval * 1000))
        
        # Account stats timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(
            lambda: run_async_task(self.core.update_account_stats())
        )
        self.stats_timer.start(int(self.config.account_stats_interval * 1000))
        
        # Countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(
            lambda: run_async_task(self.core.update_countdown())
        )
        self.countdown_timer.start(int(self.config.countdown_interval * 1000))
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts - stored as instance variables to prevent garbage collection."""
        # Store shortcuts as instance variables to prevent garbage collection
        self.shortcuts = []
        
        # Use ApplicationShortcut context so shortcuts work regardless of which widget has focus
        shortcut_context = Qt.ShortcutContext.ApplicationShortcut
        
        # Helper function to create and configure a shortcut
        def create_shortcut(key_sequence, callback):
            shortcut = QShortcut(QKeySequence(key_sequence), self, callback)
            shortcut.setContext(shortcut_context)
            shortcut.setEnabled(True)
            self.shortcuts.append(shortcut)
            return shortcut
        
        # Buy YES
        create_shortcut("Y", self.buy_yes)
        create_shortcut("y", self.buy_yes)
        
        # Buy NO
        create_shortcut("N", self.buy_no)
        create_shortcut("n", self.buy_no)
        
        # Flatten
        create_shortcut("F", self.flatten)
        create_shortcut("f", self.flatten)
        
        # Cancel All
        create_shortcut("C", self.cancel_all)
        create_shortcut("c", self.cancel_all)
        
        # Size up
        create_shortcut("+", self.size_up)
        create_shortcut("=", self.size_up)
        
        # Size down
        create_shortcut("-", self.size_down)
        create_shortcut("_", self.size_down)
        
        # Toggle graphs
        create_shortcut("H", self.toggle_graphs)
        create_shortcut("h", self.toggle_graphs)
        
        # Toggle log
        create_shortcut("L", self.toggle_log)
        create_shortcut("l", self.toggle_log)
        
        # Quit
        create_shortcut("Q", self.quit_app)
        create_shortcut("q", self.quit_app)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events for keyboard shortcuts."""
        key = event.key()
        modifiers = event.modifiers()
        
        # Handle key presses (case-insensitive)
        key_char = event.text().upper() if event.text() else ""
        
        if key == Qt.Key.Key_Y or key_char == "Y":
            self.buy_yes()
            event.accept()
        elif key == Qt.Key.Key_N or key_char == "N":
            self.buy_no()
            event.accept()
        elif key == Qt.Key.Key_F or key_char == "F":
            self.flatten()
            event.accept()
        elif key == Qt.Key.Key_C or key_char == "C":
            self.cancel_all()
            event.accept()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal or key_char == "+" or key_char == "=":
            self.size_up()
            event.accept()
        elif key == Qt.Key.Key_Minus or key == Qt.Key.Key_Underscore or key_char == "-" or key_char == "_":
            self.size_down()
            event.accept()
        elif key == Qt.Key.Key_H or key_char == "H":
            self.toggle_graphs()
            event.accept()
        elif key == Qt.Key.Key_L or key_char == "L":
            self.toggle_log()
            event.accept()
        elif key == Qt.Key.Key_Q or key_char == "Q":
            self.quit_app()
            event.accept()
        else:
            # Let other keys be handled normally
            super().keyPressEvent(event)
    
    # Callback handlers (wrapped to handle async callbacks)
    async def _on_market_update(self, strike: str, ends: str):
        """Handle market update from core."""
        self.market_panel.update_strike(strike)
        self.market_panel.update_ends(ends)
    
    async def _on_btc_price_update(self, price: float):
        """Handle BTC price update from core."""
        self.market_panel.update_btc_price(price)
    
    async def _on_price_update(self, yes_price: float, no_price: float, best_bid: float, best_ask: float):
        """Handle price update from core."""
        spread = f"{best_bid:.2f} / {best_ask:.2f}"
        self.price_panel.update_prices(yes_price, no_price, spread)
    
    async def _on_account_stats_update(self, balance: float, yes_balance: float, no_balance: float, size: float):
        """Handle account stats update from core."""
        # Always use the current selected_size from core to ensure accuracy
        # This prevents race conditions where async update might have stale data
        current_size = self.core.selected_size
        self.stats_panel.update_stats(balance, yes_balance, no_balance, current_size)
    
    async def _on_countdown_update(self, time_str: str):
        """Handle countdown update from core."""
        # Update immediately on the main thread to avoid glitching
        self.market_panel.update_time_left(time_str)
    
    async def _on_prior_outcomes_update(self, outcomes: list):
        """Handle prior outcomes update from core."""
        outcome_str = ""
        for outcome in outcomes:
            if outcome == "YES":
                outcome_str += "▲"
            elif outcome == "NO":
                outcome_str += "▼"
        if not outcome_str:
            outcome_str = "---"
        self.market_panel.update_prior_outcomes(outcome_str)
    
    async def _on_resolution(self, resolution: Optional[str]):
        """Handle resolution from core."""
        if resolution:
            # Resize overlay to cover entire window
            self.resolution_overlay.setGeometry(self.geometry())
            self.resolution_overlay.show_resolution(resolution)
            # Hide after duration
            QTimer.singleShot(
                int(self.config.resolution_overlay_duration * 1000),
                self.resolution_overlay.hide_resolution
            )
        else:
            self.resolution_overlay.hide_resolution()
    
    def resizeEvent(self, event):
        """Handle window resize - update overlay size."""
        super().resizeEvent(event)
        if self.resolution_overlay.isVisible():
            self.resolution_overlay.setGeometry(self.geometry())
    
    def _on_log(self, message: str):
        """Handle log message from core."""
        self.log_panel.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_panel.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    async def _on_chart_update(self, *args):
        """Handle chart update from core."""
        if not self.graphs_visible:
            return
        
        try:
            if len(args) == 3 and args[2] == 'btc':
                # BTC chart update
                prices, strike_val, _ = args
                self.btc_chart.update_data(prices, strike_val)
            else:
                # Price chart update
                history = args[0]
                self.probability_chart.update_data(history)
        except Exception as e:
            logger.debug(f"Error updating chart: {e}")
    
    # Action handlers
    def buy_yes(self):
        """Place BUY YES order."""
        run_async_task(self.core.place_order('YES'))
    
    def buy_no(self):
        """Place BUY NO order."""
        run_async_task(self.core.place_order('NO'))
    
    def flatten(self):
        """Flatten all positions."""
        run_async_task(self.core.flatten())
    
    def cancel_all(self):
        """Cancel all pending orders."""
        run_async_task(self.core.cancel_all())
    
    def size_up(self):
        """Increase order size."""
        self.core.size_up()
        # Immediately update UI size display using the exact value that will be used for orders
        # This ensures the displayed size always matches what will be submitted
        current_size = self.core.selected_size
        self.stats_panel.update_size_only(current_size)
        # Update full stats in background (for balance, positions, etc.)
        # Note: The async update will also set size, but it uses self.core.selected_size
        # which is the source of truth, so it will match what we just displayed
        run_async_task(self.core.update_account_stats())
    
    def size_down(self):
        """Decrease order size."""
        self.core.size_down()
        # Immediately update UI size display using the exact value that will be used for orders
        # This ensures the displayed size always matches what will be submitted
        current_size = self.core.selected_size
        self.stats_panel.update_size_only(current_size)
        # Update full stats in background (for balance, positions, etc.)
        # Note: The async update will also set size, but it uses self.core.selected_size
        # which is the source of truth, so it will match what we just displayed
        run_async_task(self.core.update_account_stats())
    
    def toggle_graphs(self):
        """Toggle graphs visibility."""
        self.graphs_visible = not self.graphs_visible
        if self.graphs_visible:
            self.charts_widget.show()
            self.core.log_msg("Graphs shown")
        else:
            self.charts_widget.hide()
            self.core.log_msg("Graphs hidden")
    
    def toggle_log(self):
        """Toggle log panel visibility by shrinking/expanding."""
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_panel.setMaximumHeight(self.log_panel_height)
            self.log_panel.setMinimumHeight(self.log_panel_height)
            self.log_panel.show()
            self.core.log_msg("Log shown")
        else:
            self.log_panel.setMaximumHeight(0)
            self.log_panel.setMinimumHeight(0)
            self.log_panel.hide()
            logger.info("Log panel hidden")
    
    def quit_app(self):
        """Quit the application."""
        self.close()
    
    def eventFilter(self, obj, event):
        """Event filter to prevent log panel from consuming shortcut key events."""
        # Only filter events from the log panel
        if obj == self.log_panel and event.type() == QEvent.Type.KeyPress:
            # Check if this is one of our shortcut keys
            key_char = event.text().upper() if event.text() else ""
            key = event.key()
            shortcut_keys = [
                Qt.Key.Key_Y, Qt.Key.Key_N, Qt.Key.Key_F, Qt.Key.Key_C,
                Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus, Qt.Key.Key_Underscore,
                Qt.Key.Key_H, Qt.Key.Key_L, Qt.Key.Key_Q
            ]
            # If it's a shortcut key, prevent the log panel from processing it
            # ApplicationShortcut will handle it at the application level
            if key in shortcut_keys or key_char in ["Y", "N", "F", "C", "+", "=", "-", "_", "H", "L", "Q"]:
                # Let ApplicationShortcut handle it, but prevent log panel from processing
                return True  # Consume the event so log panel doesn't process it
        return super().eventFilter(obj, event)
    
    def show_help(self):
        """Show help dialog with keybindings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("FINGER BLASTER - Keyboard Shortcuts")
        dialog.setStyleSheet("""
            QDialog {
                background-color: #000000;
                color: #00ffff;
            }
            QLabel {
                color: #00ffff;
                background-color: transparent;
            }
        """)
        dialog.setMinimumSize(500, 600)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Title
        title = QLabel("KEYBOARD SHORTCUTS")
        title.setStyleSheet("font-weight: bold; font-size: 18pt; color: #00ffff; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #000000; border: 1px solid #00ffff;")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)
        
        # Keybindings
        keybindings = [
            ("Y / y", "Buy YES"),
            ("N / n", "Buy NO"),
            ("F / f", "Flatten all positions"),
            ("C / c", "Cancel all pending orders"),
            ("+ / =", "Increase order size"),
            ("- / _", "Decrease order size"),
            ("H / h", "Toggle graphs visibility"),
            ("L / l", "Toggle log panel"),
            ("Q / q", "Quit application"),
        ]
        
        for key, description in keybindings:
            row = QHBoxLayout()
            
            key_label = QLabel(f"[{key}]")
            key_label.setStyleSheet("""
                font-weight: bold;
                font-size: 12pt;
                color: #ffff00;
                padding: 5px;
                min-width: 100px;
            """)
            
            desc_label = QLabel(description)
            desc_label.setStyleSheet("""
                font-size: 11pt;
                color: #00ffff;
                padding: 5px;
            """)
            
            row.addWidget(key_label)
            row.addWidget(desc_label)
            row.addStretch()
            
            content_layout.addLayout(row)
        
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Close button
        close_button = QPushButton("CLOSE")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00ffff;
                border: 1px solid #00ffff;
                font-weight: bold;
                font-size: 11pt;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #00ffff;
                color: #000000;
            }
            QPushButton:pressed {
                background-color: #00cccc;
            }
        """)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec()
    
    def closeEvent(self, event):
        """Handle window close event."""
        run_async_task(self.core.shutdown())
        event.accept()


def run_pyqt_app():
    """Run the PyQt6 desktop application."""
    # Setup asyncio event loop for Qt integration
    setup_qt_asyncio()
    
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        logger.error(f"Failed to initialize Qt application: {e}")
        logger.error("\nThis is usually due to missing system libraries.")
        logger.error("Try installing: sudo apt-get install libxcb-cursor0 libxcb-cursor-dev")
        logger.error("Or set QT_QPA_PLATFORM=offscreen for headless mode")
        sys.exit(1)
    
    # Set dark theme
    app.setStyle('Fusion')
    from PyQt6.QtGui import QColor
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(0, 0, 0))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = FingerBlasterPyQtApp()
    window.show()
    
    sys.exit(app.exec())

