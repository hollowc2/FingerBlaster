"""Main entry point for FingerBlaster trading suite.

Supports multiple trading tools:
- Activetrader (original FingerBlaster): --activetrader or default
  - Terminal UI (Textual): --terminal or default
  - Desktop UI (PyQt6): --desktop or --pyqt
  - Web UI: --web
- Ladder: --ladder
- Pulse: --pulse

Usage:
    python main.py                    # Activetrader terminal UI (default)
    python main.py --activetrader     # Activetrader terminal UI
    python main.py --activetrader --desktop  # Activetrader desktop UI
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
    """Run Activetrader tool with appropriate UI mode."""
    # Determine UI mode (default to terminal)
    if "--desktop" in sys.argv or "--pyqt" in sys.argv:
        # Desktop UI (PyQt6)
        try:
            from src.activetrader.gui.desktop.main import run_pyqt_app
            run_pyqt_app()
        except ImportError as e:
            logger.error(f"PyQt6 UI not available: {e}")
            logger.error("Install PyQt6 to use desktop UI.")
            print("ERROR: PyQt6 UI not available. Install PyQt6 to use desktop UI.")
            sys.exit(1)
    elif "--web" in sys.argv:
        # Web UI
        try:
            from src.activetrader.gui.web.main import run_web_app
            run_web_app()
        except ImportError as e:
            logger.error(f"Web UI not available: {e}")
            logger.error("Web UI is not yet implemented.")
            print("ERROR: Web UI is not yet implemented.")
            sys.exit(1)
    else:
        # Terminal UI (Textual) - default
        try:
            from src.activetrader.gui.terminal.main import run_textual_app
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
        app = PolyTerm()
        app.run()
    except ImportError as e:
        logger.error(f"Ladder tool not available: {e}")
        print("ERROR: Ladder tool not available.")
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
        # (but allow UI mode flags like --desktop, --web to work)
        run_activetrader()


if __name__ == "__main__":
    main()
