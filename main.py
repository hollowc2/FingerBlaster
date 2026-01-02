"""Main entry point for FingerBlaster application.

Supports multiple UI interfaces:
- Terminal UI (Textual): --terminal or default
- Dashboard UI (Textual): --dashboard
- Desktop UI (PyQt6): --desktop or --pyqt
- Web UI: --web
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


def main():
    """Main entry point - routes to appropriate UI based on command line arguments."""
    if "--dashboard" in sys.argv:
        # Import and run Dashboard UI
        try:
            from gui.textual_dashboard.dashboard import Dashboard
            Dashboard().run()
        except ImportError as e:
            logger.error(f"Dashboard UI not available: {e}")
            logger.error("Install Textual to use dashboard UI.")
            print("ERROR: Dashboard UI not available. Install Textual to use dashboard UI.")
            sys.exit(1)
    elif "--desktop" in sys.argv or "--pyqt" in sys.argv:
        # Import and run PyQt6 UI
        try:
            from gui.desktop.main import run_pyqt_app
            run_pyqt_app()
        except ImportError as e:
            logger.error(f"PyQt6 UI not available: {e}")
            logger.error("Install PyQt6 to use desktop UI.")
            print("ERROR: PyQt6 UI not available. Install PyQt6 to use desktop UI.")
            sys.exit(1)
    elif "--web" in sys.argv:
        # Import and run Web UI
        try:
            from gui.web.main import run_web_app
            run_web_app()
        except ImportError as e:
            logger.error(f"Web UI not available: {e}")
            logger.error("Web UI is not yet implemented.")
            print("ERROR: Web UI is not yet implemented.")
            sys.exit(1)
    else:
        # Default to Textual terminal UI
        try:
            from gui.terminal.main import run_textual_app
            run_textual_app()
        except ImportError as e:
            logger.error(f"Terminal UI not available: {e}")
            logger.error("Install Textual to use terminal UI.")
            print("ERROR: Terminal UI not available. Install Textual to use terminal UI.")
            sys.exit(1)


if __name__ == "__main__":
    main()
