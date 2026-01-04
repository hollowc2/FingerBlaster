# Pulse Terminal UI - Real-time Market Dashboard
# Framework: Textual
# Data: Coinbase via PulseCore
# Purpose: Technical analysis signals display

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from pulse.config import PulseConfig, Timeframe, Ticker, IndicatorSnapshot
from pulse.core import PulseCore
from pulse.gui.scoring import compute_signal_score

logger = logging.getLogger("Pulse.GUI")

# Timeframe to card ID mapping
TIMEFRAME_CARD_MAP = {
    Timeframe.TEN_SEC: "ltf-10s",
    Timeframe.ONE_MIN: "ltf-1m",
    Timeframe.FIFTEEN_MIN: "ltf-15m",
    Timeframe.ONE_HOUR: "htf-1h",
    Timeframe.FOUR_HOUR: "htf-4h",
    Timeframe.ONE_DAY: "htf-daily",
}

# Card ID to display timeframe label
CARD_TIMEFRAME_LABELS = {
    "ltf-10s": "10s HFT",
    "ltf-1m": "1m Scalp",
    "ltf-15m": "15m Intraday",
    "htf-1h": "1h",
    "htf-4h": "4h",
    "htf-daily": "Daily",
}

# Data Models
@dataclass
class MarketHeader:
    symbol: str
    price: float
    change_pct: float
    volume_24h: float


@dataclass
class Signal:
    label: str
    score: int
    description: str
    metrics: Dict[str, str]


# Widgets
class CombinedHeaderWidget(Static):
    data: Optional[MarketHeader] = None
    short_term_signal: str = "MIXED"
    long_term_signal: str = "MIXED"
    short_term_type: str = "mixed"  # "bullish", "bearish", or "mixed"
    long_term_type: str = "mixed"

    def _get_signal_color(self, signal_type: str) -> str:
        """Get color markup for signal type."""
        if signal_type == "bullish":
            return "[#10b981]"  # Green
        elif signal_type == "bearish":
            return "[#ef4444]"  # Red
        else:  # mixed
            return "[#f59e0b]"  # Yellow/Orange

    def render(self) -> str:
        if not self.data:
            return "Loading..."
        sign = "+" if self.data.change_pct >= 0 else ""
        # Color code percentage change: green for positive, red for negative
        if self.data.change_pct >= 0:
            pct_color = "[#10b981]"  # Green
        else:
            pct_color = "[#ef4444]"  # Red

        st_color = self._get_signal_color(self.short_term_type)
        lt_color = self._get_signal_color(self.long_term_type)
        return (
            f"{self.data.symbol}  ${self.data.price:,.2f}  "
            f"{pct_color}{sign}{self.data.change_pct:.2f}%[/]  |  "
            f"24h Vol: ${self.data.volume_24h/1e9:.1f}B  |  "
            f"ST: {st_color}{self.short_term_signal}[/]  |  "
            f"LT: {lt_color}{self.long_term_signal}[/]"
        )


class SignalCard(Static):
    signal = reactive(Signal("Loading", 50, "Waiting for data...", {}))

    def __init__(self, title: str, timeframe: str, signal: Signal, id: Optional[str] = None):
        super().__init__(id=id)
        self.title = title
        self.timeframe = (timeframe or "").strip()
        self.series: List[float] = []  # Initialize BEFORE setting signal (reactive)
        self.signal = signal  # This triggers watch_signal() which needs self.series

    def watch_signal(self, signal: Signal) -> None:
        """Reactively update when signal changes."""
        # Remove old score class
        for cls in list(self.classes):
            if cls.startswith("score-"):
                self.remove_class(cls)

        # Add new score class
        self.add_class(self._get_score_class())

        # Re-render
        self.update(self.render())

    def sparkline(self) -> tuple:
        """Generate sparkline with trend info. Returns (sparkline, trend_indicator, trend_color, change_str)."""
        if len(self.series) < 2:
            return ("", "", "", "")

        # Unicode sparkline
        ticks = "▁▂▃▄▅▆▇█"
        mn, mx = min(self.series), max(self.series)
        span = mx - mn if mx != mn else 1
        spark = "".join(ticks[int((v - mn) / span * (len(ticks) - 1))] for v in self.series)

        # Calculate trend (compare last few points to earlier points)
        recent_avg = sum(self.series[-5:]) / min(5, len(self.series))
        earlier_avg = sum(self.series[:5]) / min(5, len(self.series))
        trend = recent_avg - earlier_avg

        # Determine trend indicator and color
        if trend > 0.1:
            trend_indicator = "▲"
            trend_color = "[#10b981]"  # Green
        elif trend < -0.1:
            trend_indicator = "▼"
            trend_color = "[#ef4444]"  # Red
        else:
            trend_indicator = "→"
            trend_color = "[#f59e0b]"  # Yellow

        # Calculate change percentage
        if len(self.series) >= 2:
            current = self.series[-1]
            previous = self.series[-2] if len(self.series) > 1 else self.series[0]
            change = current - previous
            change_pct = (change / abs(previous)) * 100 if previous != 0 else 0
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "0.0%"

        return (spark, trend_indicator, trend_color, change_str)

    def _get_score_class(self) -> str:
        """Get CSS class based on score value."""
        score = self.signal.score
        if score <= 20:
            return "score-0-20"
        elif score <= 40:
            return "score-21-40"
        elif score <= 50:
            return "score-41-50"
        elif score <= 60:
            return "score-51-60"
        elif score <= 70:
            return "score-61-70"
        elif score <= 80:
            return "score-71-80"
        elif score <= 90:
            return "score-81-90"
        else:
            return "score-91-100"

    def on_mount(self) -> None:
        """Update the widget content when mounted."""
        # Apply score-based border color class
        score_class = self._get_score_class()
        self.add_class(score_class)
        content = self.render()
        self.update(content)

    def render(self) -> str:
        # Format the title line with timeframe label
        if self.timeframe:
            title_line = f"[{self.timeframe}] {self.title}"
        else:
            title_line = self.title
        lines = [
            title_line,
            f"Score: {self.signal.score}",
            f"{self.signal.description}",
            ""
        ]

        # Enhanced sparkline with trend
        spark, trend_indicator, trend_color, change_str = self.sparkline()
        if spark:
            # Show sparkline with trend indicator and change
            sparkline_line = f"{spark} {trend_color}{trend_indicator} {change_str}[/]"
            lines.append(sparkline_line)
        else:
            lines.append("")

        lines.append("")

        for k, v in self.signal.metrics.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


# Main Dashboard App
class MarketDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        height: 0;
        display: none;
    }

    Footer {
        height: 0;
        display: none;
    }

    CombinedHeaderWidget {
        height: 1;
        margin: 0;
        padding: 0;
        text-align: center;
        content-align: center middle;
    }

    Grid {
        grid-size: 3 2;
        grid-gutter: 0;
        grid-rows: 1fr 1fr;
        margin: 0;
        padding: 0;
    }

    SignalCard {
        margin: 0;
    }

    SignalCard {
        border: solid $secondary;
        padding: 1 2;
        background: $surface;
    }
    SignalCard:hover {
        border: solid $accent;
    }

    /* Score-based border colors: red (low) to green (high) */
    .score-0-20 { border: solid #ef4444; }      /* Bright red */
    .score-21-40 { border: solid #f87171; }    /* Red */
    .score-41-50 { border: solid #fbbf24; }    /* Yellow */
    .score-51-60 { border: solid #f59e0b; }     /* Orange */
    .score-61-70 { border: solid #84cc16; }     /* Light green */
    .score-71-80 { border: solid #22c55e; }     /* Green */
    .score-81-90 { border: solid #10b981; }    /* Bright green */
    .score-91-100 { border: solid #059669; }   /* Very bright green */

    /* Flashing classes for aligned signals */
    .flash-red { border: solid #ef4444; }
    .flash-green { border: solid #10b981; }
    """

    header_data = reactive(
        MarketHeader("BTC-USD", 0.0, 0.0, 0.0)
    )

    _flash_state: bool = False
    _flash_timer: Optional[Timer] = None

    def __init__(self):
        super().__init__()
        logger.info("MarketDashboard.__init__() called")
        self.core: Optional[PulseCore] = None
        self._indicator_snapshots: Dict[Timeframe, IndicatorSnapshot] = {}
        self._current_ticker: Optional[Ticker] = None
        logger.info("MarketDashboard.__init__() completed")

    def compose(self) -> ComposeResult:
        try:
            logger.info("compose() called")
            yield CombinedHeaderWidget(id="combined_header")
            logger.info("Header widget yielded")

            with Grid():
                logger.info("Grid context entered")
                yield SignalCard(
                    "Loading", "10s HFT",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-10s"
                )
                yield SignalCard(
                    "Loading", "1m Scalp",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-1m"
                )
                yield SignalCard(
                    "Loading", "15m Intraday",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-15m"
                )
                yield SignalCard(
                    "Loading", "1h",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-1h"
                )
                yield SignalCard(
                    "Loading", "4h",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-4h"
                )
                yield SignalCard(
                    "Loading", "Daily",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-daily"
                )
            logger.info("All widgets composed successfully")
        except Exception as e:
            logger.error(f"Error in compose(): {e}", exc_info=True)
            raise

    async def on_mount(self) -> None:
        """Initialize PulseCore and register callbacks."""
        try:
            logger.info("on_mount() started")
            # Configure for exactly the 6 required timeframes
            config = PulseConfig(
                products=["BTC-USD"],
                enabled_timeframes={
                    Timeframe.TEN_SEC,
                    Timeframe.ONE_MIN,
                    Timeframe.FIFTEEN_MIN,
                    Timeframe.ONE_HOUR,
                    Timeframe.FOUR_HOUR,
                    Timeframe.ONE_DAY,
                },
            )
            logger.info("Config created")

            self.core = PulseCore(config=config)
            logger.info("PulseCore instantiated")

            # Register callbacks
            self.core.register_callback('ticker_update', self._on_ticker_update)
            self.core.register_callback('indicator_update', self._on_indicator_update)
            self.core.register_callback('connection_status', self._on_connection_status)
            self.core.register_callback('priming_progress', self._on_priming_progress)
            self.core.register_callback('priming_complete', self._on_priming_complete)
            self.core.register_callback('alert', self._on_alert)
            logger.info("Callbacks registered")

            # Start PulseCore in background
            self.run_worker(self._start_core(), exclusive=True)
            logger.info("Worker started")

            # Start flash timer for alignment effects
            self._flash_timer = self.set_interval(0.5, self._flash_toggle)
            logger.info("on_mount() completed successfully")
        except Exception as e:
            logger.error(f"Error in on_mount: {e}", exc_info=True)
            raise

    async def _start_core(self) -> None:
        """Start PulseCore (runs in worker thread)."""
        try:
            logger.info("Starting PulseCore...")
            await self.core.start()
            logger.info("PulseCore started successfully")
        except Exception as e:
            logger.error(f"Error starting Pulse: {e}")
            self.notify(f"Error starting Pulse: {e}", severity="error")

    async def _on_ticker_update(self, ticker: Ticker) -> None:
        """Handle ticker updates from PulseCore."""
        self._current_ticker = ticker

        # Update header widget
        try:
            header = self.query_one("#combined_header", CombinedHeaderWidget)
            header.data = MarketHeader(
                symbol=ticker.product_id,
                price=ticker.price,
                change_pct=ticker.price_change_pct_24h,
                volume_24h=ticker.volume_24h,
            )

            # Recalculate ST/LT signals based on current card scores
            st_type, st_label = self._calculate_st_signal()
            lt_type, lt_label = self._calculate_lt_signal()
            header.short_term_type = st_type
            header.short_term_signal = st_label
            header.long_term_type = lt_type
            header.long_term_signal = lt_label
            header.refresh()
        except Exception as e:
            logger.debug(f"Error updating ticker: {e}")

    async def _on_indicator_update(
        self,
        product_id: str,
        timeframe: Timeframe,
        snapshot: IndicatorSnapshot
    ) -> None:
        """Handle indicator updates from PulseCore."""
        # Store snapshot
        self._indicator_snapshots[timeframe] = snapshot

        # Get corresponding card ID
        card_id = TIMEFRAME_CARD_MAP.get(timeframe)
        if not card_id:
            return

        # Compute signal score from indicators
        score, label, description = compute_signal_score(snapshot)

        # Build metrics dict for display
        metrics = self._build_metrics_dict(snapshot)

        # Create new Signal
        new_signal = Signal(
            label=label,
            score=score,
            description=description,
            metrics=metrics,
        )

        # Update the SignalCard
        try:
            card = self.query_one(f"#{card_id}", SignalCard)

            # Update signal data
            card.signal = new_signal

            # Update sparkline with latest close prices from candles
            closes = self._get_recent_closes(product_id, timeframe)
            if closes:
                card.series = closes

            # Refresh the card
            card.refresh()

            # Recalculate header signals after any card update
            self._update_header_signals()

        except Exception as e:
            logger.debug(f"Error updating card {card_id}: {e}")

    async def _on_connection_status(self, connected: bool, message: str) -> None:
        """Handle connection status change."""
        if connected:
            logger.info(f"Connected: {message}")
        else:
            logger.warning(f"Disconnected: {message}")

    async def _on_priming_progress(
        self,
        product_id: str,
        timeframe: Timeframe,
        progress: float
    ) -> None:
        """Handle priming progress."""
        logger.info(f"Priming {product_id} {timeframe.value}: {progress*100:.0f}%")

    async def _on_priming_complete(self, product_id: str) -> None:
        """Handle priming completion."""
        logger.info(f"Priming complete for {product_id}")

    async def _on_alert(self, alert) -> None:
        """Handle alerts from indicator engine."""
        logger.info(f"Alert: {alert.message}")

    def _build_metrics_dict(self, snapshot: IndicatorSnapshot) -> Dict[str, str]:
        """Build display metrics from indicator snapshot."""
        metrics = {}

        if snapshot.rsi is not None:
            rsi_label = "OB" if snapshot.rsi > 70 else ("OS" if snapshot.rsi < 30 else "")
            metrics["RSI(14)"] = f"{snapshot.rsi:.1f} {rsi_label}".strip()

        if snapshot.adx is not None:
            metrics["ADX"] = f"{snapshot.adx:.1f}"

        if snapshot.trend_direction:
            arrow = "↑" if snapshot.trend_direction == "UP" else ("↓" if snapshot.trend_direction == "DOWN" else "→")
            metrics["Trend"] = f"{snapshot.trend_direction} {arrow}"

        if snapshot.macd_histogram is not None:
            sign = "+" if snapshot.macd_histogram > 0 else ""
            metrics["MACD Hist"] = f"{sign}{snapshot.macd_histogram:.2f}"

        if snapshot.vwap_deviation is not None:
            sign = "+" if snapshot.vwap_deviation > 0 else ""
            metrics["VWAP Dev"] = f"{sign}{snapshot.vwap_deviation:.2f}%"

        if snapshot.regime:
            metrics["Regime"] = snapshot.regime

        return metrics

    def _get_recent_closes(self, product_id: str, timeframe: Timeframe) -> List[float]:
        """Get recent close prices for sparkline."""
        if not self.core:
            return []

        candles = self.core.get_candles(product_id, timeframe, limit=30)
        if not candles:
            return []

        return [c.close for c in candles]

    def _is_red(self, score: int) -> bool:
        """Check if score is in red range (low scores)."""
        return score <= 40

    def _is_green(self, score: int) -> bool:
        """Check if score is in green range (high scores)."""
        return score >= 70

    def _update_header_signals(self) -> None:
        """Update header ST/LT signals based on card scores."""
        try:
            header = self.query_one("#combined_header", CombinedHeaderWidget)
            st_type, st_label = self._calculate_st_signal()
            lt_type, lt_label = self._calculate_lt_signal()
            header.short_term_type = st_type
            header.short_term_signal = st_label
            header.long_term_type = lt_type
            header.long_term_signal = lt_label
            header.refresh()
        except Exception as e:
            logger.debug(f"Error updating header signals: {e}")

    def _calculate_st_signal(self) -> tuple:
        """Calculate short-term signal type and label. Returns (type, label)."""
        try:
            ltf_10s = self.query_one("#ltf-10s", SignalCard)
            ltf_1m = self.query_one("#ltf-1m", SignalCard)
            ltf_15m = self.query_one("#ltf-15m", SignalCard)

            scores = [
                ltf_10s.signal.score,
                ltf_1m.signal.score,
                ltf_15m.signal.score
            ]

            green_count = sum(1 for s in scores if self._is_green(s))
            red_count = sum(1 for s in scores if self._is_red(s))

            if green_count >= 2:  # Majority bullish
                return ("bullish", "BULLISH")
            elif red_count >= 2:  # Majority bearish
                return ("bearish", "BEARISH")
            else:
                return ("mixed", "MIXED")
        except Exception:
            return ("mixed", "MIXED")

    def _calculate_lt_signal(self) -> tuple:
        """Calculate long-term signal type and label. Returns (type, label)."""
        try:
            htf_1h = self.query_one("#htf-1h", SignalCard)
            htf_4h = self.query_one("#htf-4h", SignalCard)
            htf_daily = self.query_one("#htf-daily", SignalCard)

            scores = [
                htf_1h.signal.score,
                htf_4h.signal.score,
                htf_daily.signal.score
            ]

            green_count = sum(1 for s in scores if self._is_green(s))
            red_count = sum(1 for s in scores if self._is_red(s))

            if green_count >= 2:  # Majority bullish
                return ("bullish", "BULLISH")
            elif red_count >= 2:  # Majority bearish
                return ("bearish", "BEARISH")
            else:
                return ("mixed", "MIXED")
        except Exception:
            return ("mixed", "MIXED")

    def _check_ltf_alignment(self) -> Optional[str]:
        """Check if all LTF signals are aligned. Returns 'red', 'green', or None."""
        try:
            ltf_10s = self.query_one("#ltf-10s", SignalCard)
            ltf_1m = self.query_one("#ltf-1m", SignalCard)
            ltf_15m = self.query_one("#ltf-15m", SignalCard)

            scores = [
                ltf_10s.signal.score,
                ltf_1m.signal.score,
                ltf_15m.signal.score
            ]

            # Check if all are red
            if all(self._is_red(s) for s in scores):
                return "red"
            # Check if all are green
            elif all(self._is_green(s) for s in scores):
                return "green"
            return None
        except Exception:
            return None

    def _check_htf_alignment(self) -> Optional[str]:
        """Check if all HTF signals are aligned. Returns 'red', 'green', or None."""
        try:
            htf_1h = self.query_one("#htf-1h", SignalCard)
            htf_4h = self.query_one("#htf-4h", SignalCard)
            htf_daily = self.query_one("#htf-daily", SignalCard)

            scores = [
                htf_1h.signal.score,
                htf_4h.signal.score,
                htf_daily.signal.score
            ]

            # Check if all are red
            if all(self._is_red(s) for s in scores):
                return "red"
            # Check if all are green
            elif all(self._is_green(s) for s in scores):
                return "green"
            return None
        except Exception:
            return None

    def _update_flashing(self) -> None:
        """Update flashing state for aligned signals."""
        ltf_align = self._check_ltf_alignment()
        htf_align = self._check_htf_alignment()

        # LTF cards
        ltf_ids = ["ltf-10s", "ltf-1m", "ltf-15m"]
        for card_id in ltf_ids:
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Remove flash classes first
                card.remove_class("flash-red", "flash-green")
                # Only add flash class if aligned and in flash state
                if ltf_align == "red" and self._flash_state:
                    card.add_class("flash-red")
                elif ltf_align == "green" and self._flash_state:
                    card.add_class("flash-green")
            except Exception:
                pass

        # HTF cards
        htf_ids = ["htf-1h", "htf-4h", "htf-daily"]
        for card_id in htf_ids:
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Remove flash classes first
                card.remove_class("flash-red", "flash-green")
                # Only add flash class if aligned and in flash state
                if htf_align == "red" and self._flash_state:
                    card.add_class("flash-red")
                elif htf_align == "green" and self._flash_state:
                    card.add_class("flash-green")
            except Exception:
                pass

    def _flash_toggle(self) -> None:
        """Toggle flash state."""
        self._flash_state = not self._flash_state
        self._update_flashing()

    async def on_unmount(self) -> None:
        """Cleanup when app closes."""
        # Stop flash timer
        if self._flash_timer:
            self._flash_timer.stop()

        # Stop PulseCore
        if self.core:
            await self.core.stop()

    def action_quit(self) -> None:
        """Handle quit action."""
        self.run_worker(self._shutdown())

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        if self.core:
            await self.core.stop()
        self.exit()


def run_pulse_app():
    """Entry point for Pulse terminal dashboard."""
    try:
        app = MarketDashboard()
        app.run()
    except Exception as e:
        logger.error(f"Pulse app crashed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_pulse_app()
