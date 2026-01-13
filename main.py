"""Main entry point for FingerBlaster trading suite.

Supports multiple trading tools:
- Ladder: --ladder
- Pulse: --pulse

Usage:
    python main.py                    # Activetrader terminal UI (default)
    python main.py --ladder           # Ladder tool
    python main.py --pulse            # Pulse dashboard
    python main.py --pulse --timeframes 1m 15m  # Pulse with specific timeframes
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    filename='data/finger_blaster.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FingerBlaster")


def run_activetrader():
    """Run Activetrader tool with terminal UI."""
    # Terminal UI (Textual) - default
    try:
        from src.activetrader.gui.main import run_textual_app
        run_textual_app()
    except ImportError as e:
        logger.error(f"Terminal UI not available: {e}")
        logger.error("Install Textual to use terminal UI.")
        print("ERROR: Terminal UI not available. Install Textual to use terminal UI.")
        sys.exit(1)


def run_ladder():
    """Run Ladder tool."""
    try:
        from src.ladder.ladder import PolyTerm
        from src.activetrader.core import FingerBlasterCore
        
        # Initialize FingerBlasterCore for the ladder
        fb_core = FingerBlasterCore()
        
        # Create the ladder app with FingerBlasterCore
        app = PolyTerm(fb_core)
        
        # Start the app (this will trigger on_mount which starts the update timer)
        app.run()
    except ImportError as e:
        logger.error(f"Ladder tool not available: {e}")
        print("ERROR: Ladder tool not available.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running ladder: {e}", exc_info=True)
        print(f"ERROR: Failed to start ladder: {e}")
        sys.exit(1)


def run_pulse(timeframes=None, products=None):
    """Run Pulse dashboard.

    Args:
        timeframes: List of timeframe strings (e.g., ['1m', '15m'])
        products: List of product IDs (e.g., ['BTC-USD'])
    """
    try:
        from src.pulse.gui.main import run_pulse_app
        from src.pulse.config import PulseConfig, Timeframe

        # Parse timeframes if provided
        if timeframes:
            # Map timeframe strings to Timeframe enum values
            tf_map = {tf.value: tf for tf in Timeframe}
            enabled_timeframes = set()
            for tf_str in timeframes:
                if tf_str in tf_map:
                    enabled_timeframes.add(tf_map[tf_str])
                else:
                    logger.warning(f"Unknown timeframe: {tf_str}")

            # Create config with specified timeframes
            if enabled_timeframes:
                config = PulseConfig(
                    products=products or ["BTC-USD"],
                    enabled_timeframes=enabled_timeframes,
                )
                run_pulse_app(config=config)
            else:
                # No valid timeframes, use default
                run_pulse_app()
        else:
            # No timeframes specified, use default
            run_pulse_app()
    except ImportError as e:
        logger.error(f"Pulse UI not available: {e}")
        print("ERROR: Pulse UI not available.")
        sys.exit(1)


def main():
    """Main entry point - routes to appropriate tool based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="FingerBlaster Trading Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Tool selection (mutually exclusive)
    tool_group = parser.add_mutually_exclusive_group()
    tool_group.add_argument('--ladder', action='store_true', help='Run Ladder tool')
    tool_group.add_argument('--pulse', action='store_true', help='Run Pulse dashboard')
    tool_group.add_argument('--activetrader', action='store_true', help='Run Activetrader (default)')

    # Pulse-specific arguments
    parser.add_argument(
        '--timeframes', '-t',
        nargs='+',
        help='Timeframes for Pulse (e.g., 1m 15m 1h). Available: 10s, 1m, 5m, 15m, 1h, 4h, 1d'
    )
    parser.add_argument(
        '--products', '-p',
        nargs='+',
        help='Product IDs for Pulse (default: BTC-USD)'
    )

    args = parser.parse_args()

    # Route to appropriate tool
    if args.ladder:
        run_ladder()
    elif args.pulse:
        run_pulse(timeframes=args.timeframes, products=args.products)
    elif args.activetrader:
        run_activetrader()
    else:
        # Default to Activetrader if no tool flag specified
        run_activetrader()


if __name__ == "__main__":
    main()
