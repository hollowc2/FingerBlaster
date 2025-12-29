# FingerBlaster - Polymarket Trading Interface

![FingerBlaster Icon](data/images/icon.png)

A high-performance quantitative trading terminal for Polymarket's "BTC Up or Down 15m" binary markets. Features real-time analytics, Black-Scholes fair value pricing, edge detection, and lightning-fast order execution.

## 🎯 Features

### Core Functionality
- **Real-time Market Data**: Live WebSocket connection to Polymarket order books with automatic reconnection
- **Dual UI Modes**: Choose between terminal-based (Textual) or desktop (PyQt6) interface
- **Live Charts**: 
  - Probability chart showing YES price history over time
  - BTC price chart with strike price overlay
  - Charts can be toggled on/off for performance
- **Quick Trading**: One-key order placement (Y for YES, N for NO)
- **Position Management**: Flatten positions and cancel orders with single keystrokes
- **Market Context**: 
  - Strike price display
  - Real-time countdown timer with urgency coloring
  - Live BTC price with delta and basis points
  - Prior outcomes tracking (shows consecutive market results)
- **Resolution Overlay**: Visual notification when markets resolve
- **Account Statistics**: Real-time balance, YES/NO positions with average entry prices, unrealized PnL

### 🧮 Quantitative Analytics Engine (NEW in v3.0)

#### Math & Analytics Layer
- **Basis Points (bps)**: Distance from strike as `((BTC - Strike) / Strike) × 10,000`
- **Binary Fair Value**: Simplified Black-Scholes pricing for binary options (0% risk-free rate)
- **Edge Detection**: Real-time BUY/SELL signals when market price deviates from fair value
- **Z-Score (σ)**: Rolling 15-minute realized volatility with sigma label (e.g., "+1.5σ")

#### Execution & Risk Management
- **Liquidity Depth**: Shows dollar depth at top of book for both YES and NO
- **Real-Time PnL**: Unrealized profit/loss with percentage for open positions
- **Slippage Calculator**: Estimates fill price impact based on order book depth
- **⚠️ PANIC Button**: Press `F` to immediately market-sell ALL positions

#### UX & Urgency Hierarchy
- **Dynamic Timer**:
  - 🟢 **> 5 mins**: Green (Normal trading zone)
  - 🟠 **2-5 mins**: Orange (Watchful - increased gamma exposure)
  - 🔴 **< 2 mins**: Blinking Red + Large Font (Critical - theta/gamma risk)
- **Regime Detection**: Analyzes prior outcomes for trend (e.g., "80% BEARISH")
- **Oracle Lag Monitor**: Compares Chainlink to CEX feeds for front-running opportunities

### Architecture
- **Modular Design**: Shared core logic (`FingerBlasterCore`) used by both UIs
- **Event-Driven**: Callback-based system for UI updates
- **Async/Await**: Fully asynchronous for optimal performance
- **Error Handling**: Comprehensive error handling and logging

## 📸 Screenshots

### Desktop UI (PyQt6)
![Desktop UI](data/images/FingerBlaster_gui.png)

### Terminal UI (Textual)
![Terminal UI](data/images/fingerblaster_terminal.png)

### Desktop UI - Compact Mode
![Terminal UI Compact](data/images/fingerblaster_qt_micro.png)

## 🎬 Demo Videos

### Live Trading Demo
![Live Trading](data/livetrade.gif)

### Fullscreen Interface
![Fullscreen Interface](data/fingerblaster_fullscreen.gif)

### Side-by-Side with Polymarket
![FingerBlaster vs Polymarket](data/fingerblaster_fullscreen_next_to_polymarket.gif)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Valid Polymarket account with API credentials
- Private key for signing transactions
- USDC balance on Polygon for trading

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Desktop UI (Optional)

If you want to use the desktop UI, you may need to install system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libxcb-cursor0 libxcb-cursor-dev
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install libxcb-cursor
```

**Arch Linux:**
```bash
sudo pacman -S libxcb-cursor
```

See [INSTALL_DESKTOP.md](INSTALL_DESKTOP.md) for detailed installation instructions and troubleshooting.

### Step 3: Configuration

Create a `.env` file in the project root:

```env
PRIVATE_KEY=your_private_key_here
WALLET_ADDRESS=your_wallet_address_here  # Optional, for proxy wallets
```

## 💻 Usage

### Terminal UI (Default)

Run the application with the terminal interface:

```bash
python main.py
```

or

```bash
python main.py --textual
```

### Desktop UI

Run the application with the desktop interface:

```bash
python main.py --desktop
```

or

```bash
python main.py --pyqt
```

## ⌨️ Keyboard Shortcuts

### Terminal UI

| Key | Action |
|-----|--------|
| `Y` | Buy YES |
| `N` | Buy NO |
| `F` | ⚠️ **PANIC FLATTEN** - Sell all positions immediately |
| `C` | Cancel all pending orders |
| `+` / `=` | Increase order size by $1 |
| `-` / `_` | Decrease order size by $1 |
| `H` | Toggle graphs visibility |
| `L` | Toggle log panel visibility |
| `Q` | Quit application |

### Desktop UI

| Key | Action |
|-----|--------|
| `F` | ⚠️ **PANIC FLATTEN** - Sell all positions immediately |
| `Ctrl+Y` | Buy YES |
| `Ctrl+N` | Buy NO |
| `Ctrl+C` | Cancel all pending orders |
| `Ctrl++` / `Ctrl+=` | Increase order size by $1 |
| `Ctrl+-` | Decrease order size by $1 |
| `Ctrl+H` | Toggle graphs visibility |
| `Ctrl+L` | Toggle log panel visibility |
| `Ctrl+Q` | Quit application |

### Desktop UI Additional Features
- **Help Dialog**: Click the "HELP" button to view keyboard shortcuts
- **Resizable Windows**: Adjust window size to your preference

## 📊 Analytics Display Guide

### Market Panel
```
══ MARKET CONTEXT ══
STRIKE: 95,500.00
BTC   : $95,623.45
DELTA: ▲$123 (+13bps)      ← Distance from strike in $ and basis points
SIGMA : +0.85σ              ← Z-score: How many std devs from strike
PRIOR : ▲▲▼▲▼▲▲▼▲▲         ← Recent market outcomes
REGIME: 60% BULLISH         ← Trend detection from prior outcomes
ORACLE: 45ms                ← Chainlink vs CEX latency (green = fast)
REMAIN: 08:32               ← Timer with urgency coloring
```

### Price Panel
```
═══ LIVE PRICES ═══
YES
0.58
FV: 0.55 | EDGE: +54bps BUY  ← Fair value + trading signal
DEPTH: $1,250 | SLIP: 12bps  ← Liquidity + slippage estimate
SPREAD: 0.57 / 0.59

NO
0.42
FV: 0.45 | EDGE: -67bps SELL
DEPTH: $890 | SLIP: 18bps
SPREAD: 0.41 / 0.43
```

### Stats Panel
```
══ ACCOUNT ══
CASH: $523.45
SIZE: $10.00
POS : Y:15.2@52c | N:0.0
PnL : +$4.56 (+5.6%)         ← Real-time unrealized PnL
```

## 🎯 Trading Strategy Guide

### Using the Analytics for Decision Making

#### 1. **Edge Detection** - When to Trade
- **Green "BUY" signal**: Market price < Fair Value → Consider buying
- **Red "SELL" signal**: Market price > Fair Value → Consider selling/avoiding
- **Edge > 50bps**: Strong signal worth acting on
- **Edge < 50bps**: Fair pricing, be selective

#### 2. **Z-Score (Sigma)** - Position Confidence
- **|σ| > 2.0**: Strong directional move, high confidence in outcome
- **|σ| < 0.5**: Price near strike, coin-flip territory
- **Positive σ**: BTC above strike (YES favored)
- **Negative σ**: BTC below strike (NO favored)

#### 3. **Timer Urgency** - Risk Management
- **Green (>5 min)**: Normal trading, time for positions to work
- **Orange (2-5 min)**: Gamma increasing, be cautious with new entries
- **Red Blinking (<2 min)**: High gamma/theta risk, consider flattening

#### 4. **Regime Detection** - Trend Following
- **>70% Bullish/Bearish**: Strong trend, consider momentum trades
- **50-70%**: Mixed signals, be selective
- Use regime to bias directional trades in trending conditions

#### 5. **Oracle Lag** - Front-Running Opportunities
- **<100ms**: Prices synchronized, fair game
- **100-500ms**: Minor edge possible with fast execution
- **>500ms (Yellow)**: Potential arbitrage if you have faster CEX data
- **>2000ms (Red)**: Significant lag, potential alpha

#### 6. **Liquidity/Slippage** - Execution Quality
- **Depth**: Higher = better fills for larger orders
- **Slippage >50bps**: Consider scaling into position
- Check depth before using PANIC flatten

### Example Trading Workflow

1. **Market Opens**: Check regime from prior outcomes
2. **Assess Edge**: Look for FV divergence > 50bps
3. **Confirm with σ**: Ensure z-score supports trade direction
4. **Check Liquidity**: Verify depth supports your size
5. **Monitor Timer**: As gamma increases (orange/red), tighten risk
6. **PANIC Button**: If position goes wrong near expiry, press `F`

### Risk Management Rules

1. **Never fight the timer** - Flatten before red zone if uncertain
2. **Respect the edge** - Don't buy overvalued, don't sell undervalued
3. **Size to liquidity** - Check slippage before large trades
4. **Trend is friend** - Regime detection helps avoid counter-trend trades
5. **Oracle lag = opportunity** - Fast information = edge

## ⚙️ Configuration

The application automatically:
- Discovers active "BTC Up or Down 15m" markets
- Connects to WebSocket for real-time price updates
- Updates analytics every 500ms
- Updates account balances, positions, and PnL every 10 seconds
- Updates BTC price via RTDS in real-time (Chainlink prices matching Polymarket resolution)
- Updates market status every 5 seconds
- Tracks prior market outcomes with timestamps (last 10 consecutive markets)
- Shows resolution overlay when markets expire

### Configuration Options

Key settings can be adjusted in `src/config.py`:

#### Trading Settings
- `order_rate_limit_seconds`: Minimum time between orders (default: 0.5s)
- `min_order_size`: Minimum order size (default: $1.00)
- `size_increment`: Order size increment (default: $1.00)
- `market_duration_minutes`: Market duration (default: 15 minutes)

#### Analytics Settings
- `analytics_interval`: How often analytics update (default: 0.5s)
- `timer_critical_minutes`: Red zone threshold (default: 2 min)
- `timer_watchful_minutes`: Orange zone threshold (default: 5 min)
- `default_volatility`: Default BTC volatility for FV calc (default: 60%)
- `edge_threshold_bps`: Threshold for edge signals (default: 50 bps)
- `oracle_lag_warning_ms`: Yellow warning threshold (default: 500ms)
- `oracle_lag_critical_ms`: Red critical threshold (default: 2000ms)

## 📁 Data Files

The application creates a `data/` directory containing:

- `finger_blaster.log` - Application logs with timestamps
- `prior_outcomes.json` - History of market resolutions with timestamps
- `images/` - Application icons and screenshots
- `*.gif` - Demo videos showing the application in action

## 🏗️ Architecture

### Core Components

1. **FingerBlasterCore**: Shared business logic controller
   - Manages market data, history, WebSocket, and order execution
   - Event-driven callback system for UI updates
   - UI-agnostic design

2. **AnalyticsEngine** (NEW): Quantitative analysis module
   - Black-Scholes binary option pricing
   - Rolling volatility calculation
   - Edge detection and z-score
   - Liquidity analysis and slippage estimation
   - Regime detection from prior outcomes
   - Oracle lag monitoring

3. **MarketDataManager**: Handles market discovery and data
   - Finds active markets
   - Calculates mid prices
   - Manages token maps and order books

4. **HistoryManager**: Tracks price and BTC history
   - Maintains YES price history
   - Maintains BTC price history
   - Provides data for charts and volatility

5. **WebSocketManager**: Real-time data connection
   - Connects to Polymarket WebSocket
   - Automatic reconnection on failure
   - Handles ping/pong for connection health

6. **OrderExecutor**: Order placement and management
   - Executes market orders with aggressive pricing
   - Flattens positions (PANIC button)
   - Cancels pending orders

7. **RTDSManager**: Real-time data stream for crypto prices
   - Connects to Polymarket's RTDS WebSocket for BTC prices
   - Provides Chainlink BTC/USD prices (matches Polymarket resolution source)
   - Falls back to Binance API if RTDS unavailable
   - Maintains historical price data for dynamic strike price resolution

8. **UI Components**:
   - **Terminal UI** (`main.py`, `src/ui.py`): Textual-based interface
   - **Desktop UI** (`main_pyqt.py`, `src/ui_pyqt.py`): PyQt6-based interface

### Data Flow

```
Polymarket API/WebSocket + RTDS
    ↓
PolymarketConnector + RTDSManager
    ↓
FingerBlasterCore (Managers)
    ↓
AnalyticsEngine (Calculations)
    ↓
Event Callbacks
    ↓
UI Components (Terminal/Desktop)
```

## 📊 Features in Detail

### Analytics Snapshot

Every 500ms, the analytics engine generates a complete snapshot including:
- Basis points from strike
- Fair values for YES and NO
- Edge direction and magnitude
- Z-score and sigma label
- Liquidity depth at top of book
- Slippage estimates for current order size
- Unrealized PnL for open positions
- Timer urgency level
- Regime detection results
- Oracle lag measurement

### Prior Outcomes Tracking

The application tracks the last 10 consecutive market outcomes, showing them as:
- `▲` for YES outcomes (green)
- `▼` for NO outcomes (red)

Outcomes are matched by timestamp to ensure only consecutive markets are displayed.

### Resolution Overlay

When a market expires, a full-screen overlay appears showing:
- **YES** (green background) if BTC price >= strike price
- **NO** (red background) if BTC price < strike price

The overlay displays for 3 seconds before the application searches for the next market.

### PANIC Flatten

Press `F` at any time to immediately:
1. Log a warning message with emoji
2. Market-sell all YES positions
3. Market-sell all NO positions
4. Update account statistics

This is designed for emergency risk management when you need to exit immediately.

### Charts

**Probability Chart**: Shows YES price history over the market duration with:
- Cyan line for YES price
- X-axis: Time elapsed (0 to 15 minutes)
- Y-axis: Price (0.00 to 1.00)

**BTC Chart**: Shows BTC price history with:
- Cyan line for BTC price
- Yellow line for strike price (if available)
- Automatic scaling with padding

### Real-time Updates

- **Analytics**: Every 500ms (configurable)
- **Price Updates**: Via WebSocket, updates as order book changes
- **BTC Price**: Updates via RTDS using Chainlink prices
- **Countdown**: Updates every 200ms for smooth ticking
- **Account Stats**: Updates every 10 seconds
- **Market Status**: Checks for new markets every 5 seconds

## 🔧 Troubleshooting

### Desktop UI Issues

If the desktop UI fails to launch:

1. **Missing System Libraries**: See [INSTALL_DESKTOP.md](INSTALL_DESKTOP.md)
2. **Wayland**: Try `export QT_QPA_PLATFORM=wayland`
3. **X11 Issues**: Ensure X11 is running and accessible

### WebSocket Connection Issues

- Check internet connection
- Verify Polymarket API is accessible
- Check logs in `data/finger_blaster.log`

### Order Execution Issues

- Verify `.env` file has correct `PRIVATE_KEY`
- Ensure sufficient USDC balance
- Check rate limiting (0.5s between orders)
- Review logs for specific error messages

### Analytics Not Updating

- Ensure market is active (not expired)
- Check that BTC price is updating
- Verify WebSocket connection is alive
- Check logs for analytics errors

## 📝 Notes

- **BTC Price Source**: Uses RTDS Chainlink BTC/USD prices (matches Polymarket resolution), falls back to Binance API
- **Fair Value**: Calculated using Black-Scholes with 60% default volatility, 0% risk-free rate
- **Edge Detection**: 50 bps threshold for BUY/SELL signals (configurable)
- **Market Orders**: Use aggressive pricing (10% above/below mid) to ensure fills
- **Order Size**: Defaults to $1.00 and can be adjusted with +/- keys
- **Rate Limiting**: 0.5 seconds between orders to prevent API throttling
- **PANIC Button**: Press `F` for immediate position exit (no confirmation)

## 🔒 Security

- **Private Keys**: Never commit your `.env` file to version control
- **API Credentials**: Store securely and never share
- **Transactions**: All transactions are signed locally with your private key
- **PANIC Button**: No confirmation dialog - be careful!

## 📄 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- New features are tested
- Documentation is updated
- Both UI modes are considered
- Analytics calculations are validated

---

**Version**: 3.0  
**Last Updated**: 2025  
**New in v3.0**: Quantitative Analytics Engine, PANIC Button, Dynamic Timer Urgency, Edge Detection, Z-Score/Sigma, Regime Detection, Oracle Lag Monitor, Real-Time PnL
