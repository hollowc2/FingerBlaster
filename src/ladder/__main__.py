"""Entry point for running ladder module directly.

Usage:
    python -m src.ladder
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ladder.ladder import PolyTerm

if __name__ == "__main__":
    app = PolyTerm()
    app.run()
