# FingerBlaster - Polymarket Trading Interface

![FingerBlaster Icon](icon.png)

A standalone terminal-based trading interface for Polymarket's "BTC Up or Down 15m" markets. Features real-time price updates, live charts, and quick order execution.

## Features

- **Real-time Market Data**: Live WebSocket connection to Polymarket order books
- **Interactive UI**: Terminal-based interface with keyboard shortcuts
- **Live Charts**: Optional price history and BTC price charts (can be disabled)
- **Quick Trading**: One-key order placement (Y for YES, N for NO)
- **Position Management**: Flatten positions and cancel orders with single keystrokes
- **Market Context**: Display strike price, time remaining, BTC price, and delta

## Installation

1. Clone or extract this project to a directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root (see `.env.example` for template):
   ```
   PRIVATE_KEY=your_private_key_here
   WALLET_ADDRESS=your_wallet_address_here  # Optional, for proxy wallets
   ```

## Usage

Run the application with the default theme:
```bash
python main.py
```

Run with the purple theme:
```bash
python main.py --theme purple
```

## Keyboard Shortcuts

- `Y` - Buy YES
- `N` - Buy NO
- `F` - Flatten all positions
- `C` - Cancel all orders
- `+` or `=` - Increase order size by $1
- `-` or `_` - Decrease order size by $1
- `H` - Hide graph
- `L` - Toggle log panel visibility
- `Q` - Quit

## Configuration

The application automatically:
- Discovers active "BTC Up or Down 15m" markets
- Connects to WebSocket for real-time updates
- Updates account balances and positions
- Tracks prior market outcomes (last 10)

## Data Files

The application creates a `data/` directory containing:
- `finger_blaster.log` - Application logs
- `prior_outcomes.json` - History of market resolutions

## Requirements

- Python 3.8+
- Valid Polymarket account with API credentials
- Private key for signing transactions
- USDC balance on Polygon for trading

## Notes

- The application uses Binance API for BTC price reference (same as Polymarket)
- Market orders use aggressive pricing to ensure fills
- Order size defaults to $1 and can be adjusted with +/- keys
- Rate limiting: 0.5 seconds between orders

