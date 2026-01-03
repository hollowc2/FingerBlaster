"""
Pulse Standalone Entry Point.

Run with:
    python -m pulse
    python -m pulse --products BTC-USD ETH-USD
    python -m pulse --timeframes 1m 5m 1h
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import List, Optional

from pulse.config import (
    Alert,
    BucketedOrderBook,
    Candle,
    IndicatorSnapshot,
    PulseConfig,
    Ticker,
    Timeframe,
    Trade,
)
from pulse.core import PulseCore

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
        self.pulse.register_callback('candle_update', self._on_candle)
        self.pulse.register_callback('trade_update', self._on_trade)
        self.pulse.register_callback('ticker_update', self._on_ticker)
        self.pulse.register_callback('orderbook_update', self._on_orderbook)
        self.pulse.register_callback('indicator_update', self._on_indicator)
        self.pulse.register_callback('alert', self._on_alert)
        self.pulse.register_callback('connection_status', self._on_connection_status)
        self.pulse.register_callback('priming_progress', self._on_priming_progress)
        self.pulse.register_callback('priming_complete', self._on_priming_complete)

        # Start Pulse
        await self.pulse.start()

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def stop(self):
        """Stop the Pulse application."""
        logger.info("Shutting down...")

        if self.pulse:
            await self.pulse.stop()

        self._shutdown_event.set()

    def request_shutdown(self):
        """Request graceful shutdown."""
        asyncio.create_task(self.stop())

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    async def _on_candle(self, candle: Candle):
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

    async def _on_orderbook(self, book: BucketedOrderBook):
        """Handle order book update."""
        # Only show occasionally to avoid spam
        pass

    async def _on_indicator(self, product_id: str, snapshot: IndicatorSnapshot):
        """Handle indicator update."""
        if not self.show_indicators:
            return

        parts = [f"[INDICATOR] {product_id}"]

        if snapshot.vwap:
            parts.append(f"VWAP: ${snapshot.vwap:,.2f}")

        if snapshot.adx:
            parts.append(f"ADX: {snapshot.adx:.1f}")

        if snapshot.atr:
            parts.append(f"ATR: ${snapshot.atr:.2f}")

        if snapshot.volatility:
            parts.append(f"Vol: {snapshot.volatility:.1f}%")

        parts.append(f"Trend: {snapshot.trend_direction}")
        parts.append(f"Regime: {snapshot.regime}")

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

    async def _on_priming_progress(self, product_id: str, timeframe: Timeframe, progress: float):
        """Handle priming progress."""
        pct = int(progress * 100)
        print(f"[PRIMING] {product_id} {timeframe.value}: {pct}%")

    async def _on_priming_complete(self, product_id: str):
        """Handle priming complete."""
        print(f"[PRIMING] {product_id} complete - Live data active")
        print("-" * 60)


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
    """Run the Pulse application."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)


if __name__ == "__main__":
    run()
