"""Application configuration constants and CSS styling."""

from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration constants."""
    # History limits
    max_history_size: int = 1000
    max_btc_history_size: int = 100
    
    # Trading limits
    order_rate_limit_seconds: float = 0.5
    min_order_size: float = 1.0
    size_increment: float = 1.0
    
    # Market settings
    market_duration_minutes: int = 15
    market_duration_seconds: int = 900
    
    # WebSocket settings
    ws_uri: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_reconnect_delay: int = 5
    ws_max_reconnect_attempts: int = 10
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    ws_recv_timeout: float = 1.0
    
    # UI settings
    time_warning_threshold_minutes: int = 2
    resolution_overlay_duration: float = 3.0
    chart_update_throttle_seconds: float = 1.0
    chart_padding_percentage: float = 0.25
    
    # Data persistence
    data_dir: str = "data"
    log_file: str = "data/finger_blaster.log"
    prior_outcomes_file: str = "data/prior_outcomes.json"
    max_prior_outcomes: int = 10
    
    # Update intervals (seconds)
    market_status_interval: float = 5.0
    btc_price_interval: float = 3.0
    account_stats_interval: float = 10.0
    countdown_interval: float = 1.0


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
    border: double white;
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
    border: double white;
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

#price_history_container {
    height: 1fr;
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

