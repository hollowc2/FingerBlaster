"""Position Manager - Terminal UI for viewing and closing positions.

This module provides a Textual-based position manager that displays
open positions in a table format with the ability to close individual positions.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Add project root to Python path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
except NameError:
    project_root = Path.cwd()
    for parent in project_root.parents:
        if (parent / 'src').exists() and (parent / 'src' / 'core.py').exists():
            project_root = parent
            break

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Footer, DataTable, Button
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding

from src.core import FingerBlasterCore

logger = logging.getLogger("FingerBlaster.PositionManager")


@dataclass
class Position:
    """Represents a single position."""
    side: str  # 'Up' or 'Down'
    shares: float
    entry_price: Optional[float]
    current_price: float
    market_value: float
    pnl: float
    pnl_percent: float


class PositionManagerApp(ModalScreen):
    """Position Manager application showing open positions in a table."""
    
    CSS = """
    Screen {
        background: #0D0D0D;
    }
    
    #header {
        height: 3;
        border: solid #262626;
        background: #161616;
        padding: 1;
        align: center middle;
    }
    
    #header-title {
        text-style: bold;
        color: #EAB308;
        width: 100%;
        text-align: center;
    }
    
    #table-container {
        margin: 1;
        border: solid #262626;
        background: #161616;
    }
    
    DataTable {
        width: 100%;
        height: 100%;
    }
    
    #footer-info {
        height: 3;
        border-top: solid #262626;
        background: #161616;
        padding: 1;
    }
    
    .positive-pnl {
        color: $success;
    }
    
    .negative-pnl {
        color: $error;
    }
    
    .close-button {
        width: 8;
    }
    """
    
    BINDINGS = [
        Binding("p", "quit", "Close", priority=True),
        Binding("escape", "quit", "Close", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("enter", "close_selected", "Close Position", priority=True),
    ]
    
    def __init__(self, core: FingerBlasterCore):
        super().__init__()
        self.core = core
        self.positions: List[Position] = []
        self._update_task: Optional[asyncio.Task] = None
    
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        with Vertical():
            with Container(id="header"):
                yield Static("POSITION MANAGER", id="header-title")
            
            with Container(id="table-container"):
                yield DataTable(id="positions-table")
            
            with Container(id="footer-info"):
                yield Static("Press ENTER to close position, 'r' to refresh, 'q' or ESC to close", id="footer-text")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the position manager."""
        table = self.query_one("#positions-table", DataTable)
        
        # Add columns
        table.add_columns(
            "Side",
            "Shares",
            "Entry Price",
            "Current Price",
            "Market Value",
            "PnL",
            "PnL %",
            "Action"
        )
        
        # Start periodic updates
        self.set_interval(2.0, self._update_positions)
        await self._update_positions()
    
    async def _update_positions(self) -> None:
        """Update position data from Polymarket Data API."""
        try:
            # Get current market's token IDs for filtering
            market = await self.core.market_manager.get_market()
            if not market:
                self._clear_table()
                return

            token_map = market.get('token_map', {})
            current_token_ids = set(token_map.values())

            if not current_token_ids:
                self._clear_table()
                return

            # Fetch ALL positions (no API-level filter since condition_id not available)
            api_positions = await self.core.connector.get_positions(
                size_threshold=0.1
            )

            if not api_positions:
                self._clear_table()
                return

            # Convert API response to Position objects, filtering by current market token IDs
            positions: List[Position] = []
            for pos in api_positions:
                # Filter: only show positions from current market
                asset_token = pos.get('asset', '')
                if asset_token not in current_token_ids:
                    continue

                # Determine side from outcome field
                # API may return "YES"/"NO" or "Up"/"Down", normalize to "Up"/"Down"
                outcome = pos.get('outcome', '')
                if not outcome:
                    continue
                
                # Normalize to title case and map legacy YES/NO to Up/Down
                outcome_normalized = outcome[0].upper() + outcome[1:].lower() if len(outcome) > 1 else outcome.upper()
                
                # Map YES/NO to Up/Down for backward compatibility
                if outcome_normalized in ('Up', 'Yes') or outcome_normalized.upper() in ('YES', 'UP'):
                    side = 'Up'
                elif outcome_normalized in ('Down', 'No') or outcome_normalized.upper() in ('NO', 'DOWN'):
                    side = 'Down'
                else:
                    # Skip positions with unrecognized outcome
                    continue

                shares = float(pos.get('size', 0))
                if shares <= 0.1:
                    continue

                entry_price = float(pos.get('avgPrice', 0)) if pos.get('avgPrice') else None
                current_price = float(pos.get('curPrice', 0))
                market_value = float(pos.get('currentValue', 0))
                pnl = float(pos.get('cashPnl', 0))
                pnl_percent = float(pos.get('percentPnl', 0))

                positions.append(Position(
                    side=side,
                    shares=shares,
                    entry_price=entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                ))

            self.positions = positions
            await self._update_table()

        except Exception as e:
            logger.error(f"Error updating positions: {e}", exc_info=True)
    
    async def _update_table(self) -> None:
        """Update the data table with current positions."""
        table = self.query_one("#positions-table", DataTable)
        
        # Clear existing rows
        table.clear()
        
        if not self.positions:
            table.add_row(
                "No positions",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                key="no-positions"
            )
            return
        
        # Add position rows
        for pos in self.positions:
            # Format values
            shares_str = f"{pos.shares:.2f}"
            entry_str = f"${pos.entry_price:.4f}" if pos.entry_price else "N/A"
            current_str = f"${pos.current_price:.4f}"
            value_str = f"${pos.market_value:.2f}"
            
            # Format PnL with color
            pnl_str = f"${pos.pnl:+.2f}"
            pnl_percent_str = f"{pos.pnl_percent:+.2f}%"
            
            # Determine PnL color class
            pnl_class = "positive-pnl" if pos.pnl >= 0 else "negative-pnl"
            
            # Add row with key for identification
            row_key = f"position-{pos.side}"
            table.add_row(
                pos.side,
                shares_str,
                entry_str,
                current_str,
                value_str,
                f"[{pnl_class}]{pnl_str}[/]",
                f"[{pnl_class}]{pnl_percent_str}[/]",
                "[bold #EAB308]Close[/]",
                key=row_key
            )
    
    def _clear_table(self) -> None:
        """Clear the table."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()
        table.add_row(
            "No positions",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            key="no-positions"
        )
    
    async def action_refresh(self) -> None:
        """Refresh positions manually."""
        await self._update_positions()
    
    async def action_close_selected(self) -> None:
        """Close the currently selected position."""
        table = self.query_one("#positions-table", DataTable)
        try:
            # Get row key from cursor coordinate
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            if row_key and row_key.startswith("position-"):
                side = row_key.replace("position-", "")
                await self._close_position(side)
        except Exception as e:
            logger.debug(f"Error getting row key: {e}")
    
    async def action_quit(self) -> None:
        """Close the position manager."""
        self.dismiss()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection - close position when Enter is pressed on a row."""
        row_key = event.row_key
        if row_key and str(row_key.value).startswith("position-"):
            # Extract side from row key
            side = str(row_key.value).replace("position-", "")
            # Close the position
            asyncio.create_task(self._close_position(side))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle cell click - close position when Action column is clicked."""
        # Check if the Action column (index 7) was clicked
        if event.coordinate.column == 7:
            row_key = event.cell_key.row_key
            if row_key and str(row_key.value).startswith("position-"):
                side = str(row_key.value).replace("position-", "")
                asyncio.create_task(self._close_position(side))
    
    async def _close_position(self, side: str) -> None:
        """Close a position for the given side."""
        try:
            self.notify(f"Closing {side} position...", severity="warning")
            await self.core.close_position(side)
            # Refresh positions after closing
            await asyncio.sleep(0.5)
            await self._update_positions()
            self.notify(f"{side} position closed", severity="success")
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            self.notify(f"Error closing position: {e}", severity="error")


def create_position_manager_app(core: FingerBlasterCore) -> PositionManagerApp:
    """Create a position manager app instance.
    
    Args:
        core: FingerBlasterCore instance
        
    Returns:
        PositionManagerApp instance
    """
    return PositionManagerApp(core)

