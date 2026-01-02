"""Application configuration constants and CSS styling."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AppConfig:
    """Application configuration constants.
    
    All magic numbers and configuration values are centralized here
    for easy modification and documentation.
    """
    
    # =========================================================================
    # History limits
    # =========================================================================
    # Set high enough to never lose data during a 15-minute market
    # (900 seconds * 10 updates/sec = 9000 max, but we'll use 10000 for safety)
    max_history_size: int = 10000
    max_btc_history_size: int = 100
    
    # =========================================================================
    # Trading limits
    # =========================================================================
    order_rate_limit_seconds: float = 0.5  # Minimum time between orders
    min_order_size: float = 1.0  # Minimum order size in USDC
    max_order_size: float = 1000.0  # Maximum order size in USDC (sanity limit)
    size_increment: float = 1.0  # Size adjustment increment
    
    # =========================================================================
    # Market settings
    # =========================================================================
    market_duration_minutes: int = 15
    market_duration_seconds: int = 900  # 15 * 60
    
    # Dynamic strike resolution settings
    rtds_lookback_threshold_seconds: float = 120.0  # 2 minutes
    prior_outcome_tolerance_seconds: float = 60.0  # 1 minute tolerance for matching
    
    # =========================================================================
    # WebSocket settings (CLOB order book)
    # =========================================================================
    ws_uri: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_reconnect_delay: int = 5  # Base delay between reconnection attempts
    ws_max_reconnect_attempts: int = 10
    ws_ping_interval: int = 20  # Send ping every 20 seconds
    ws_ping_timeout: int = 10  # Wait 10 seconds for pong
    ws_recv_timeout: float = 1.0  # Timeout for receiving messages
    ws_max_message_size: int = 10 * 1024 * 1024  # 10MB max message size
    
    # =========================================================================
    # RTDS (Real Time Data Stream) settings for crypto prices
    # =========================================================================
    rtds_uri: str = "wss://ws-live-data.polymarket.com"
    rtds_reconnect_delay: int = 5
    rtds_max_reconnect_attempts: int = 10
    rtds_ping_interval: int = 20
    rtds_ping_timeout: int = 10
    rtds_recv_timeout: float = 1.0
    rtds_history_retention_ms: int = 3600000  # Keep 1 hour of price history
    
    # =========================================================================
    # UI settings
    # =========================================================================
    time_warning_threshold_minutes: int = 2  # Show warning color when < 2 min left
    resolution_overlay_duration: float = 3.0  # How long to show resolution overlay
    chart_update_throttle_seconds: float = 1.0  # Minimum time between chart updates
    chart_padding_percentage: float = 0.25  # 25% padding around chart Y-axis
    chart_min_points: int = 2  # Minimum data points to render chart
    
    # Price cache settings
    price_cache_ttl_seconds: float = 0.1  # 100ms cache for price calculations
    
    # =========================================================================
    # Data persistence
    # =========================================================================
    data_dir: str = "data"
    log_file: str = "data/finger_blaster.log"
    
    # Prior outcomes (fetched from Polymarket API)
    max_prior_outcomes: int = 10  # Maximum prior outcomes to display
    prior_outcomes_cache_ttl_seconds: float = 60.0  # Cache API results for 60 seconds
    
    # =========================================================================
    # Update intervals (seconds)
    # =========================================================================
    market_status_interval: float = 5.0  # Check for new markets every 5s
    btc_price_interval: float = 3.0  # Fallback BTC price update (RTDS is primary)
    account_stats_interval: float = 10.0  # Refresh account balances every 10s
    countdown_interval: float = 0.2  # Update countdown every 200ms for smooth display
    analytics_interval: float = 0.5  # Analytics update every 500ms
    
    # =========================================================================
    # Analytics settings
    # =========================================================================
    # Timer urgency thresholds (in minutes)
    timer_critical_minutes: int = 2  # Red blinking, gamma/theta risk
    timer_watchful_minutes: int = 5  # Orange, increased attention
    
    # Fair value calculation
    default_volatility: float = 0.60  # 60% annualized BTC volatility default
    edge_threshold_bps: float = 50.0  # Basis points threshold for edge detection
    
    # Slippage estimation
    default_slippage_order_size: float = 10.0  # Default order size for slippage calc
    
    # Oracle lag monitoring
    oracle_lag_warning_ms: int = 500  # Warn if lag > 500ms
    oracle_lag_critical_ms: int = 2000  # Critical if lag > 2000ms


CSS = """
Screen {
    background: #000000;
}

#header {
    background: #000080;
    color: white;
    text-align: center;
    text-style: bold;
}

.box {
    border: none;
    padding: 1;
    margin: 1;
    height: 100%;
}

.title {
    text-style: bold;
    color: #00ffff;
    margin-bottom: 1;
    width: 100%;
    text-align: center;
}

#left_cockpit {
    width: 35%;
    height: 100%;
    margin: 0 1;
    border: solid white;
    background: #1a1a1a;
}

#left_cockpit.no_graphs {
    width: 100%;
}

#charts_panel {
    width: 65%;
    height: 100%;
}

#charts_panel.hidden {
    display: none;
}

.cockpit_widget {
    height: auto;
    padding: 1;
    border-bottom: ascii gray;
    content-align: center middle;
}

/* Analytics-specific styles */
.analytics_label {
    color: #888888;
    margin-top: 0;
}

.edge_undervalued {
    color: #00ff00;
    text-style: bold;
}

.edge_overvalued {
    color: #ff0000;
    text-style: bold;
}

.urgency_critical {
    color: #ff0000;
    text-style: bold blink;
    background: #330000;
}

.urgency_watchful {
    color: #ff8800;
    text-style: bold;
}

.urgency_normal {
    color: #00ff00;
    text-style: bold;
}

#log_panel {
    height: 6;
    border: double white;
    margin-top: 1;
}

#log_panel.hidden {
    display: none;
}

.label_value {
    color: #ffff00;
}

.price_yes {
    color: #00ff00;
    text-style: bold;
    margin: 1 0;
    width: auto;
}

.price_no {
    color: #ff0000;
    text-style: bold;
    margin: 1 0;
    width: auto;
}

.price_label {
    text-style: bold underline;
    width: 100%;
    text-align: center;
}

.chart_label {
    text-style: bold;
    color: #00ffff;
    margin-bottom: 0;
    width: 100%;
    text-align: center;
}

.spread_label {
    color: #888888;
    width: 100%;
    text-align: center;
    margin-top: 1;
}

PlotextPlot {
    height: 1fr;
    width: 100%;
}

ProbabilityChart {
    height: 1fr;
    width: 100%;
    color: #ffffff;
    background: #1a1a1a;
    border: solid white;
}

#price_history_container {
    height: 1fr;
    border: none;
}

#btc_history_container {
    height: 2fr;
}

Digits {
    width: auto;
}

#resolution_overlay {
    layer: overlay;
    width: 100%;
    height: 100%;
    content-align: center middle;
    text-align: center;
    text-style: bold;
}

#resolution_overlay.yes {
    background: #00ff00;
    color: #000000;
}

#resolution_overlay.no {
    background: #ff0000;
    color: #ffffff;
}

#resolution_overlay.hidden {
    display: none;
}
"""

