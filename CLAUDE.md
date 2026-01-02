# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FingerBlaster is a high-performance quantitative trading terminal for Polymarket's "BTC Up or Down 15m" binary markets. It features real-time analytics, Black-Scholes fair value pricing, edge detection, and three UI modes: terminal (Textual), desktop (PyQt6), and web (FastAPI + React).

## Running the Application

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your PRIVATE_KEY (required for trading)
```

### Running Different UI Modes
```bash
# Terminal UI (default, Textual-based)
python main.py
python main.py --textual

# Desktop UI (PyQt6)
python main.py --desktop
python main.py --pyqt

# Dashboard UI (Textual alternative layout)
python main.py --dashboard

# Web UI (FastAPI backend + React frontend)
python main.py --web
# Note: Web UI requires separate frontend setup in gui/web/
cd gui/web && npm install && npm run dev
```

### Desktop UI System Dependencies
Desktop mode requires system libraries on Linux:
```bash
# Ubuntu/Debian
sudo apt-get install libxcb-cursor0 libxcb-cursor-dev

# Fedora/RHEL
sudo dnf install libxcb-cursor

# Arch
sudo pacman -S libxcb-cursor
```

## Architecture

### Core Design Pattern
The application uses a **shared core + multiple UI frontends** architecture:
- **FingerBlasterCore** (`src/core.py`) is the central controller containing all business logic
- **Event-driven callbacks** allow UI-agnostic updates via CallbackManager
- Each UI (terminal/desktop/web) registers callbacks and renders updates independently

### Critical Components

**1. FingerBlasterCore (`src/core.py`)**
- Main orchestrator that coordinates all managers
- Event-driven callback system (CALLBACK_EVENTS tuple defines all event types)
- UI-agnostic design - no direct UI references
- Manages lifecycle: market discovery → WebSocket connection → analytics → order execution

**2. AnalyticsEngine (`src/analytics.py`)**
- Black-Scholes binary option pricing with 0% risk-free rate
- Rolling volatility calculation for z-score/sigma
- Edge detection: compares market price vs fair value (threshold: 50bps)
- Liquidity depth analysis and slippage estimation
- Regime detection from prior market outcomes
- Oracle lag monitoring (Chainlink vs CEX)

**3. Manager Classes (`src/engine.py`)**
- **MarketDataManager**: Market discovery, order book state, token mapping
- **HistoryManager**: Maintains price/BTC history using deques (maxlen=10000)
- **WebSocketManager**: CLOB order book connection with auto-reconnect
- **OrderExecutor**: Market order execution with aggressive pricing (10% above/below mid)
- **RTDSManager**: Real-time BTC price from Polymarket's RTDS WebSocket (Chainlink source)

**4. PolymarketConnector (`connectors/polymarket.py`)**
- Wraps py-clob-client for API interactions
- Web3 transaction signing with private key from .env
- Handles order placement, cancellation, balance queries
- Supports both EOA wallets and proxy wallets (Gnosis Safe)

**5. UI Implementations**
- **Terminal** (`gui/terminal/`): Textual-based TUI with plotext charts
- **Desktop** (`gui/desktop/`): PyQt6 with matplotlib charts
- **Web** (`gui/web/`): FastAPI backend + React frontend (in development)

### Data Flow
```
Polymarket API/WebSocket + RTDS
    ↓
PolymarketConnector + RTDSManager
    ↓
FingerBlasterCore
    ├─ MarketDataManager (order books)
    ├─ HistoryManager (price history)
    ├─ WebSocketManager (live updates)
    └─ OrderExecutor (trade execution)
    ↓
AnalyticsEngine (calculations every 500ms)
    ↓
CallbackManager.emit(event, *args)
    ↓
UI Components render updates
```

## Configuration

All configuration is centralized in `src/config.py` via the `AppConfig` dataclass:

### Key Settings
- **Trading**: `order_rate_limit_seconds` (0.5s), `min_order_size` ($1), `size_increment` ($1)
- **Analytics**: `analytics_interval` (0.5s), `default_volatility` (60%), `edge_threshold_bps` (50)
- **Timer Urgency**: `timer_critical_minutes` (2), `timer_watchful_minutes` (5)
- **WebSocket**: CLOB = `ws://ws-subscriptions-clob.polymarket.com`, RTDS = `wss://ws-live-data.polymarket.com`
- **Update Intervals**: market_status (5s), account_stats (10s), countdown (0.2s)

## Critical Implementation Details

### WebSocket Handling
- Two WebSocket connections run concurrently:
  1. **CLOB** (order book): `ws://ws-subscriptions-clob.polymarket.com/ws/market`
  2. **RTDS** (BTC price): `wss://ws-live-data.polymarket.com`
- Both have auto-reconnect with exponential backoff (max 10 attempts)
- Ping/pong every 20 seconds for connection health
- Message size limit: 10MB for security

### Order Execution
- Market orders use **aggressive pricing**: BUY = mid * 1.10, SELL = mid * 0.90
- Rate limiting: 0.5 seconds between orders (configurable)
- All orders require token_id (long hex string, typically 60+ chars)
- Validation decorator `@validate_order_params` prevents invalid API calls
- PANIC flatten (F key): market-sells ALL positions with no confirmation

### Market Lifecycle
1. **Discovery**: Polls Gamma API every 5s for active "BTC Up or Down 15m" markets
2. **Connection**: Subscribes to WebSocket for real-time order book updates
3. **Analytics**: Runs every 500ms generating AnalyticsSnapshot
4. **Resolution**: Shows overlay (3s) when market expires, then searches for next market
5. **Prior Outcomes**: Tracks last 10 consecutive markets with timestamp matching

### Price Sources
- **BTC Price**: Primary = RTDS Chainlink (matches Polymarket resolution oracle), Fallback = Binance API
- **Strike Price**: Dynamic resolution using RTDS historical lookback (2min threshold)
- **Order Book**: Real-time via CLOB WebSocket with incremental updates

### Callback System
Register callbacks via `core.register_callback(event, func)`:
```python
# Event types (see CALLBACK_EVENTS in core.py):
'market_update', 'btc_price_update', 'price_update', 'account_stats_update',
'countdown_update', 'prior_outcomes_update', 'resolution', 'log',
'chart_update', 'analytics_update', 'order_submitted', 'order_filled',
'order_failed', 'flatten_started', 'flatten_completed', 'flatten_failed'
```

## Trading Strategy Context

The analytics engine provides real-time signals for decision-making:
- **Edge Detection**: Green "BUY" when market < fair value, Red "SELL" when market > fair value
- **Z-Score (σ)**: Measures BTC distance from strike in standard deviations
- **Timer Urgency**: Green (>5min) → Orange (2-5min) → Red blinking (<2min) for gamma/theta risk
- **Regime**: Analyzes prior outcomes (e.g., "80% BEARISH" if 8/10 recent were NO)
- **Oracle Lag**: Monitors Chainlink vs CEX latency for front-running opportunities

## Environment Variables

Required in `.env`:
```bash
PRIVATE_KEY=0x...  # Wallet private key for transaction signing (REQUIRED)
WALLET_ADDRESS=0x... # Optional: for proxy wallets (Gnosis Safe)
```

Never commit `.env` to git (already in .gitignore).

## Data Persistence

Created in `data/` directory:
- `finger_blaster.log` - Application logs
- `prior_outcomes.json` - Market resolution history
- `images/` - Screenshots and icons

## Known Issues & Gotchas

### WebSocket Reconnection
If WebSocket disconnects, auto-reconnect logic runs with exponential backoff. Check logs at `data/finger_blaster.log` for connection issues.

### Order Book Edge Cases
- Empty order books default to 0.5 mid price (`DEFAULT_ORDER_BOOK_PRICE`)
- Order book updates are incremental (delta) - full snapshots only on initial subscribe
- Token mapping (YES/NO to token_id) is critical - verify in `market.token_map`

### Desktop UI on Wayland
May need `export QT_QPA_PLATFORM=wayland` if PyQt6 fails to launch.

### Rate Limiting
Polymarket API has undocumented rate limits. Application enforces 0.5s between orders client-side. Excessive requests may result in 429 errors.

### Timezone Handling
All timestamps from Polymarket API are UTC. Application uses pandas Timestamp with explicit `tz_localize('UTC')` to avoid naive datetime issues.

## Performance Considerations

- **History Size**: Capped at 10,000 points to prevent memory bloat (15min market = ~900 data points at 1/sec)
- **Chart Throttling**: UI chart updates throttled to 1s to reduce CPU load
- **Price Cache**: 100ms TTL cache for mid price calculations to avoid redundant computation
- **Lock Contention**: All managers use asyncio.Lock - avoid blocking operations in critical paths

## Adding New UI Modes

To add a new UI (e.g., ncurses, Streamlit):
1. Create `gui/<ui_name>/main.py` with `run_<ui_name>_app()` function
2. Instantiate `FingerBlasterCore` and register callbacks for all relevant events
3. Add command-line argument in `main.py` (e.g., `--ncurses`)
4. Call `core.run()` to start the event loop
5. Implement UI rendering based on callback events (never poll core directly)

## Analytics Snapshot Structure

`AnalyticsSnapshot` dataclass contains all computed values:
- Basis points, fair values (YES/NO), edge direction/magnitude
- Z-score, sigma label, realized volatility
- Liquidity depth, slippage estimates
- Unrealized PnL (total + percentage)
- Timer urgency, regime detection, oracle lag

UI should render from this immutable snapshot (updated every 500ms via `analytics_update` callback).

## Code Style & Patterns

- **Async/await**: All I/O operations are asynchronous (WebSocket, API calls)
- **Type hints**: Comprehensive typing with Optional, Dict, List, Tuple
- **Dataclasses**: Used for immutable data structures (AnalyticsSnapshot, AppConfig)
- **Enums**: For state representation (TimerUrgency, EdgeDirection)
- **Decorators**: `@validate_order_params` for pre-execution validation
- **Logging**: Use module-level logger `logging.getLogger("FingerBlaster.<module>")`
- **Constants**: Extract magic numbers to named constants (TradingConstants, NetworkConstants)

## Debugging

Enable verbose logging in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

Check WebSocket messages:
```python
# In WebSocketManager or RTDSManager, set log level to DEBUG
logger.setLevel(logging.DEBUG)
```

Inspect analytics calculations:
```python
# In AnalyticsEngine, add debug prints in compute_snapshot()
logger.debug(f"Fair value: {fair_value_yes}, Edge: {edge_bps}")
```
