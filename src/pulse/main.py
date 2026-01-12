"""
Pulse Standalone Entry Point.

Run with:
    python -m src.pulse
    python -m src.pulse --products BTC-USD ETH-USD
    python -m src.pulse --timeframes 1m 5m 1h
    python main.py --pulse
"""

import argparse
import asyncio
import logging
import select
import signal
import sys
from typing import List, Optional

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

from src.pulse.config import (
    Alert,
    BucketedOrderBook,
    Candle,
    IndicatorSnapshot,
    PulseConfig,
    Ticker,
    Timeframe,
    Trade,
)
from src.pulse.core import PulseCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("Pulse")


def parse_timeframes(timeframe_strs: List[str]) -> set:
    """Parse timeframe strings to Timeframe enum values."""
    timeframes = set()
    tf_map = {tf.value: tf for tf in Timeframe}

    for tf_str in timeframe_strs:
        if tf_str in tf_map:
            timeframes.add(tf_map[tf_str])
        else:
            logger.warning(f"Unknown timeframe: {tf_str}")

    return timeframes if timeframes else {Timeframe.ONE_MIN, Timeframe.FIVE_MIN}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pulse - Real-time Coinbase Market Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m pulse
    python -m pulse --products BTC-USD ETH-USD
    python -m pulse --timeframes 1m 5m 1h
    python -m pulse --verbose
    python -m pulse --streaming  # Use data stream mode instead of GUI

Available timeframes:
    10s  - 10 Second (aggregated locally)
    1m   - 1 Minute
    5m   - 5 Minute
    15m  - 15 Minute
    1h   - 1 Hour
    4h   - 4 Hour
    1d   - Daily
        """
    )

    parser.add_argument(
        '--products', '-p',
        nargs='+',
        default=['BTC-USD'],
        help='Product IDs to track (default: BTC-USD)'
    )

    parser.add_argument(
        '--timeframes', '-t',
        nargs='+',
        default=['1m', '5m'],
        help='Timeframes to enable (default: 1m 5m)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output (only show alerts and errors)'
    )

    parser.add_argument(
        '--no-trades',
        action='store_true',
        help='Disable trade-by-trade output'
    )

    parser.add_argument(
        '--no-candles',
        action='store_true',
        help='Disable candle output'
    )

    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Use data stream mode instead of GUI dashboard'
    )

    return parser.parse_args()


class PulseApp:
    """Standalone Pulse application."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize Pulse app.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.pulse: Optional[PulseCore] = None
        self._shutdown_event = asyncio.Event()
        self._keyboard_task: Optional[asyncio.Task] = None

        # Parse config
        self.config = PulseConfig(
            products=args.products,
            enabled_timeframes=parse_timeframes(args.timeframes),
        )

        # Output control
        self.show_trades = not args.no_trades and not args.quiet
        self.show_candles = not args.no_candles and not args.quiet
        self.show_indicators = not args.quiet

    async def start(self):
        """Start the Pulse application."""
        logger.info("=" * 60)
        logger.info("  PULSE - Real-time Coinbase Market Data Analysis")
        logger.info("=" * 60)
        logger.info(f"Products: {', '.join(self.config.products)}")
        logger.info(f"Timeframes: {', '.join(tf.value for tf in self.config.enabled_timeframes)}")
        logger.info("-" * 60)

        # Create PulseCore
        self.pulse = PulseCore(config=self.config)

        # Register callbacks
        self.pulse.on('candle', self._on_candle)
        self.pulse.on('trade', self._on_trade)
        self.pulse.on('ticker', self._on_ticker)
        self.pulse.on('orderbook', self._on_orderbook)
        self.pulse.on('indicator', self._on_indicator)
        self.pulse.on('alert', self._on_alert)
        self.pulse.on('connection', self._on_connection_status)

        # Start Pulse
        await self.pulse.start()

        # Start keyboard input handler (non-blocking)
        if HAS_TERMIOS and sys.stdin.isatty():
            self._keyboard_task = asyncio.create_task(self._keyboard_input_handler())
            logger.info("Press 'q' or Ctrl+C to quit")
        else:
            logger.info("Press Ctrl+C to quit")

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def stop(self):
        """Stop the Pulse application."""
        logger.info("Shutting down...")

        # Cancel keyboard input task
        if self._keyboard_task and not self._keyboard_task.done():
            self._keyboard_task.cancel()
            try:
                await asyncio.wait_for(self._keyboard_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            finally:
                self._keyboard_task = None

        if self.pulse:
            try:
                await asyncio.wait_for(self.pulse.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Pulse stop() timed out after 5 seconds, forcing shutdown")
            except Exception as e:
                logger.error(f"Error stopping Pulse: {e}", exc_info=True)
            finally:
                self.pulse = None

        self._shutdown_event.set()

    async def _keyboard_input_handler(self):
        """Handle keyboard input in a non-blocking way."""
        old_settings = None
        try:
            # Set terminal to raw mode
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while not self._shutdown_event.is_set():
                # Check if input is available using select (non-blocking)
                loop = asyncio.get_event_loop()
                try:
                    # Wait for input with a short timeout using select
                    ready, _, _ = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: select.select([sys.stdin], [], [], 0.5)
                        ),
                        timeout=0.6
                    )
                    if ready and ready[0]:
                        char = sys.stdin.read(1)
                        if char.lower() == 'q':
                            logger.info("Quit requested via 'q' key")
                            await self.stop()
                            break
                        elif char == '\x03':  # Ctrl+C
                            logger.info("Quit requested via Ctrl+C")
                            await self.stop()
                            break
                except asyncio.TimeoutError:
                    # No input available, continue loop
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Error in keyboard input handler: {e}")
        finally:
            # Restore terminal settings
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    def request_shutdown(self):
        """Request graceful shutdown."""
        # Signal handlers need to be quick, so we schedule the shutdown
        try:
            loop = asyncio.get_running_loop()
            # Schedule the stop coroutine to run
            asyncio.create_task(self.stop())
        except RuntimeError:
            # If no event loop is running, just set the shutdown event
            self._shutdown_event.set()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    async def _on_candle(self, product_id: str, candle: Candle):
        """Handle candle update."""
        if not self.show_candles:
            return

        print(
            f"[CANDLE] {candle.timeframe.value:>4} | "
            f"O:{candle.open:>10.2f} H:{candle.high:>10.2f} "
            f"L:{candle.low:>10.2f} C:{candle.close:>10.2f} "
            f"V:{candle.volume:>12.4f}"
        )

    async def _on_trade(self, trade: Trade):
        """Handle trade update."""
        if not self.show_trades:
            return

        side_color = "BUY " if trade.side == "BUY" else "SELL"
        print(
            f"[TRADE] {side_color} {trade.size:>12.6f} @ {trade.price:>10.2f}"
        )

    async def _on_ticker(self, ticker: Ticker):
        """Handle ticker update."""
        if not self.show_indicators:
            return

        change_str = f"{ticker.price_change_pct_24h:+.2f}%"
        print(
            f"[TICKER] {ticker.product_id} ${ticker.price:,.2f} | "
            f"24h: {change_str} | "
            f"H: ${ticker.high_24h:,.2f} L: ${ticker.low_24h:,.2f}"
        )

    async def _on_orderbook(self, product_id: str, book: BucketedOrderBook):
        """Handle order book update."""
        # Only show occasionally to avoid spam
        pass

    async def _on_indicator(self, snapshot: IndicatorSnapshot):
        """Handle indicator update."""
        if not self.show_indicators:
            return

        parts = [f"[INDICATOR] {snapshot.product_id}"]

        if snapshot.vwap:
            parts.append(f"VWAP: ${snapshot.vwap:,.2f}")

        if snapshot.adx:
            parts.append(f"ADX: {snapshot.adx:.1f}")

        if snapshot.atr:
            parts.append(f"ATR: ${snapshot.atr:.2f}")

        if snapshot.volatility:
            parts.append(f"Vol: {snapshot.volatility:.1f}%")

        #parts.append(f"Trend: {snapshot.trend_direction}")
        #parts.append(f"Regime: {snapshot.regime}")

        print(" | ".join(parts))

    async def _on_alert(self, alert: Alert):
        """Handle alert."""
        severity_prefix = {
            "INFO": "[INFO]",
            "WARNING": "[WARN]",
            "CRITICAL": "[CRIT]",
        }.get(alert.severity, "[ALERT]")

        print(f"\n{'='*60}")
        print(f"{severity_prefix} {alert.alert_type.upper()}")
        print(f"  {alert.message}")
        print(f"  Product: {alert.product_id}")
        if alert.data:
            for k, v in alert.data.items():
                print(f"  {k}: {v}")
        print(f"{'='*60}\n")

    async def _on_connection_status(self, connected: bool, message: str):
        """Handle connection status."""
        status = "CONNECTED" if connected else "DISCONNECTED"
        print(f"[CONNECTION] {status}: {message}")


async def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create app
    app = PulseApp(args)

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.request_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await app.start()
    except KeyboardInterrupt:
        pass
    finally:
        await app.stop()


def run():
    """Run the Pulse application (GUI by default, streaming mode with --streaming flag)."""
    # Parse args early to check for --streaming flag
    import sys as _sys
    args = parse_args()

    if args.streaming:
        # Use streaming mode
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(0)
    else:
        # Use GUI mode (default)
        try:
            from src.pulse.gui import run_pulse_app
            run_pulse_app()
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(0)


if __name__ == "__main__":
    run()
