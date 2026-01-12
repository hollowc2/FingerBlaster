"""Main entry point for FingerBlaster trading suite.

Supports multiple trading tools:
- Ladder: --ladder
- Pulse: --pulse

Usage:
    python main.py                    # Activetrader terminal UI (default)
    python main.py --ladder           # Ladder tool
    python main.py --pulse            # Pulse dashboard
"""

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


def run_pulse():
    """Run Pulse dashboard."""
    try:
        from src.pulse.gui.main import run_pulse_app
        run_pulse_app()
    except ImportError as e:
        logger.error(f"Pulse UI not available: {e}")
        print("ERROR: Pulse UI not available.")
        sys.exit(1)


def main():
    """Main entry point - routes to appropriate tool based on command line arguments."""
    # Check for explicit tool flags
    if "--ladder" in sys.argv:
        run_ladder()
    elif "--pulse" in sys.argv:
        run_pulse()
    elif "--activetrader" in sys.argv:
        run_activetrader()
    else:
        # Default to Activetrader if no tool flag specified
        run_activetrader()


if __name__ == "__main__":
    main()
