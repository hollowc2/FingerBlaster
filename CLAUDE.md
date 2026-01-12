# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FingerBlaster is a comprehensive trading suite for Polymarket's prediction markets with three specialized tools:

- **Activetrader**: Quantitative trading terminal for "BTC Up or Down 15m" binary markets with real-time analytics, Black-Scholes fair value pricing, edge detection, and one-key trading
- **Ladder**: DOM-style ladder interface for visualizing and executing orders across the full price range (1¢ to 99¢)
- **Pulse**: Real-time market analytics dashboard with multi-timeframe technical indicators for Coinbase markets

All tools feature modern terminal UIs built with [Textual](https://textual.textualize.io/).

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and add your credentials (PRIVATE_KEY, POLY_API_KEY, etc.)
```

### Running Tools
```bash
# Activetrader (default)
python main.py
python main.py --activetrader

# Ladder
python main.py --ladder
python -m src.ladder

# Pulse
python main.py --pulse
python -m src.pulse
python -m src.pulse --products BTC-USD ETH-USD --timeframes 1m 15m 1h
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analytics.py

# Run with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src
```

### Code Quality
```bash
# Type checking
mypy src/

# Linting
flake8 src/
```

## Architecture Overview

### Modular Design Pattern
The suite uses a **shared core + specialized UI** architecture:
- **Tool-specific modules** in `src/<tool_name>/` (activetrader, ladder, pulse)
- **Shared connectors** in `src/connectors/` for API interactions
- **UI-agnostic core logic** with event-driven callback systems

### Key Architectural Concepts

**1. FingerBlasterCore (`src/activetrader/core.py`)**
- Central orchestrator for Activetrader and Ladder tools
- Event-driven callback system via CallbackManager
- Manages lifecycle: market discovery → WebSocket → analytics → execution
- UI-agnostic design - all updates via callbacks (see CALLBACK_EVENTS tuple)
- Coordinates five manager classes: MarketDataManager, HistoryManager, WebSocketManager, OrderExecutor, RTDSManager

**2. Event-Driven Callbacks**
All UI updates happen via registered callbacks:
```python
# Register callbacks for events
core.register_callback('market_update', on_market_update)
core.register_callback('analytics_update', on_analytics_update)
core.register_callback('order_filled', on_order_filled)

# Available events (see CALLBACK_EVENTS in core.py):
# 'market_update', 'btc_price_update', 'price_update', 'account_stats_update',
# 'countdown_update', 'prior_outcomes_update', 'resolution', 'log',
# 'chart_update', 'analytics_update', 'order_submitted', 'order_filled',
# 'order_failed', 'flatten_started', 'flatten_completed', 'flatten_failed',
# 'cancel_started', 'cancel_completed', 'cancel_failed', 'size_changed'
```

**3. AnalyticsEngine (`src/activetrader/analytics.py`)**
Generates AnalyticsSnapshot every 500ms containing:
- Black-Scholes binary option fair value (0% risk-free rate, 60% default volatility)
- Edge detection: compares market price vs fair value (50bps threshold)
- Z-score (σ): BTC distance from strike in standard deviations
- Liquidity depth and slippage estimation
- Timer urgency levels (NORMAL/WATCHFUL/CRITICAL)
- Real-time PnL tracking
- Regime detection from prior outcomes

**4. Dual WebSocket Architecture**
Two concurrent WebSocket connections:
- **CLOB**: Order book updates (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)
- **RTDS**: Real-time BTC price from Chainlink oracle (`wss://ws-live-data.polymarket.com`)
Both have auto-reconnect with exponential backoff (max 10 attempts)

**5. PolymarketConnector (`src/connectors/polymarket.py`)**
- Wraps py-clob-client for Polymarket API
- Web3 transaction signing with PRIVATE_KEY from .env
- Order validation via `@validate_order_params` decorator
- Supports both EOA wallets and proxy wallets (Gnosis Safe)

### Data Flow
```
Polymarket API/WebSocket + RTDS
    ↓
PolymarketConnector + RTDSManager
    ↓
FingerBlasterCore
    ├─ MarketDataManager (order books)
    ├─ HistoryManager (deque, maxlen=10000)
    ├─ WebSocketManager (CLOB connection)
    └─ OrderExecutor (aggressive pricing)
    ↓
AnalyticsEngine (every 500ms)
    ↓
CallbackManager.emit(event, *args)
    ↓
UI Components (Textual TUI)
```

### Ladder Tool Architecture
- **LadderCore** (`src/ladder/core.py`): Controller wrapping FingerBlasterCore
- **LadderDataManager** (`src/ladder/ladder_data.py`): Transforms order books into DOM-style ladder view
- **PolyTerm** (`src/ladder/ladder.py`): Textual UI with cursor navigation and visual order placement
- Cursor-based trading: navigate with arrow keys/vim keys, place orders with single keypress

### Pulse Tool Architecture
- **PulseCore** (`src/pulse/core.py`): Async event bus with non-blocking UI callbacks
- **CoinbaseConnector** (`src/connectors/coinbase.py`): Supports both CDP (JWT) and Legacy (HMAC) API keys
- **IndicatorEngine** (`src/pulse/indicators.py`): Technical indicators (RSI, MACD, ADX, VWAP, Bollinger Bands)
- **TimeframeAggregator** (`src/pulse/aggregators.py`): Multi-timeframe candle aggregation (10s, 1m, 5m, 15m, 1h, 4h, 1d)
- **IndicatorWorker**: CPU-isolated background worker for indicator calculations

## Configuration

All settings centralized in `src/activetrader/config.py` (AppConfig dataclass):

### Critical Settings
- **Trading**: `order_rate_limit_seconds=0.5`, `min_order_size=1.0`, `size_increment=1.0`
- **Analytics**: `analytics_interval=0.5`, `default_volatility=0.60`, `edge_threshold_bps=50.0`
- **Timer Urgency**: `timer_critical_minutes=2`, `timer_watchful_minutes=5`
- **WebSocket**: `ws_ping_interval=20`, `ws_reconnect_delay=5`, `ws_max_reconnect_attempts=10`
- **History**: `max_history_size=10000` (prevents memory bloat)

Pulse configuration in `src/pulse/config.py` via PulseConfig dataclass.

## Critical Implementation Details

### Order Execution
- Market orders use **aggressive pricing**: BUY = mid * 1.10, SELL = mid * 0.90
- Rate limiting: 0.5s minimum between orders (prevents API throttling)
- Token IDs are long hex strings (60+ chars) - never use short IDs
- PANIC flatten (F key): immediate market-sell ALL positions with NO confirmation

### Market Lifecycle (Activetrader)
1. **Discovery**: Poll Gamma API every 5s for active "BTC Up or Down 15m" markets
2. **Connection**: Subscribe to CLOB WebSocket for real-time order book
3. **Analytics**: Generate AnalyticsSnapshot every 500ms
4. **Resolution**: Show overlay for 3s when market expires, search for next market
5. **Prior Outcomes**: Track last 10 markets with timestamp matching (60s tolerance)

### Price Sources
- **BTC Price**: Primary = RTDS Chainlink (matches Polymarket resolution oracle), Fallback = Binance API
- **Strike Price**: Dynamic resolution using RTDS historical lookback (2min threshold)
- **Order Book**: Real-time CLOB WebSocket with incremental delta updates

### WebSocket Considerations
- Order book updates are **incremental deltas** (not full snapshots)
- Empty order books default to 0.5 mid price (`DEFAULT_ORDER_BOOK_PRICE`)
- Ping/pong every 20 seconds to maintain connection health
- 10MB max message size for security

### Async/Threading Model
- All I/O operations are async/await
- Manager classes use asyncio.Lock to prevent race conditions
- Callback execution is fire-and-forget (never blocks)
- Pulse uses IndicatorWorker for CPU isolation (prevents UI blocking)

## Environment Variables

Required in `.env`:
```bash
# REQUIRED - Polymarket
PRIVATE_KEY=0x...                    # Wallet private key for signing
POLY_API_KEY=your_api_key_here       # From https://polymarket.com/settings/api
POLY_API_SECRET=your_api_secret_here
POLY_API_PASSPHRASE=your_passphrase_here

# OPTIONAL - Coinbase (for Pulse)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_API_PASSPHRASE=  # Only for legacy keys, NOT CDP keys
```

Never commit `.env` to version control (already in .gitignore).

## Known Patterns & Gotchas

### Token Mapping
- Markets have YES/NO sides mapped to token_ids
- Token IDs are critical for order execution
- Mapping stored in `market.token_map` (e.g., `{'Up': '0x123...', 'Down': '0x456...'}`)

### Timezone Handling
- All Polymarket timestamps are UTC
- Use pandas Timestamp with explicit `tz_localize('UTC')` to avoid naive datetime issues
- Market start/end times should always be timezone-aware

### Rate Limiting
- Polymarket API has undocumented rate limits
- Client-side enforcement: 0.5s between orders
- Excessive requests may result in 429 errors
- Check logs at `data/finger_blaster.log` for throttling issues

### Empty Order Books
- Default to 0.5 mid price when no orders exist
- Check for None/empty before calculating mid price
- Slippage estimation requires sufficient depth

### Analytics Snapshot Immutability
- AnalyticsSnapshot is a frozen dataclass (immutable)
- UI should render from snapshot, never modify it
- New snapshot generated every 500ms via `analytics_update` callback

## Performance Optimizations

- **History capped at 10,000 points**: Prevents memory bloat (15min market = ~900 points at 1/sec)
- **Chart throttling**: UI updates throttled to 1s to reduce CPU load
- **Price cache**: 100ms TTL cache for mid price calculations
- **Lock contention**: Avoid blocking operations in critical paths
- **Fire-and-forget callbacks**: Callbacks never block event loop

## Testing Strategy

- Unit tests for analytics calculations (Black-Scholes, z-score, edge detection)
- Mock WebSocket connections for integration tests
- Validate order execution with dry-run mode
- Test WebSocket reconnection logic
- Verify callback registration/unregistration

## Extending the Suite

### Adding a New UI for Activetrader
1. Create `src/activetrader/gui/<ui_name>/main.py`
2. Instantiate FingerBlasterCore and register callbacks
3. Implement rendering based on callback events (never poll core)
4. Add command-line flag in `main.py`

### Adding a New Tool
1. Create `src/<tool_name>/` with `__init__.py`, `core.py`, `gui/main.py`
2. Implement tool-specific logic
3. Add entry point in `main.py` with `--<tool_name>` flag
4. Create `src/<tool_name>/__main__.py` for direct execution

### Adding a New Connector
1. Create `src/connectors/<exchange>.py`
2. Inherit from `DataConnector` base class
3. Implement required methods (get_price, place_order, etc.)
4. Use AsyncHttpFetcherMixin for HTTP requests with retry logic

## Debugging

### Enable Verbose Logging
```python
# In main.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### WebSocket Issues
```bash
# Check logs for connection errors
tail -f data/finger_blaster.log | grep -i websocket

# Verify WebSocket URLs are reachable
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" wss://ws-subscriptions-clob.polymarket.com/ws/market
```

### Order Execution Issues
- Verify PRIVATE_KEY format (must start with `0x`)
- Check USDC balance on Polygon
- Review token_id mapping (long hex string, not short ID)
- Ensure rate limiting is not violated
- Check for validation errors in logs

### Analytics Not Updating
- Verify WebSocket connection is alive (check logs)
- Ensure market is active (not expired)
- Check BTC price is updating from RTDS
- Verify analytics_interval in config (default 0.5s)

## Code Style

- **Type hints**: Comprehensive typing with Optional, Dict, List, Tuple
- **Async/await**: All I/O operations are asynchronous
- **Dataclasses**: Immutable data structures (AnalyticsSnapshot, AppConfig)
- **Enums**: State representation (TimerUrgency, EdgeDirection, Timeframe)
- **Decorators**: `@validate_order_params` for pre-execution validation
- **Logging**: Module-level logger: `logging.getLogger("FingerBlaster.<module>")`
- **Constants**: Extract magic numbers to named classes (TradingConstants, NetworkConstants)
