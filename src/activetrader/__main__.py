"""Entry point for running activetrader module directly.

Usage:
    python -m src.activetrader
    python -m src.activetrader --terminal
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import main

if __name__ == "__main__":
    # Add --activetrader flag to ensure we route to activetrader
    if "--activetrader" not in sys.argv:
        sys.argv.insert(1, "--activetrader")
    main()
