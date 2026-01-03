"""
Pulse Configuration and Data Structures.

Contains all dataclasses, enums, and configuration for the Pulse module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any


class Timeframe(Enum):
    """Supported candle timeframes."""
    TEN_SEC = "10s"      # Aggregated locally from trades
    ONE_MIN = "1m"       # REST: ONE_MINUTE
    FIVE_MIN = "5m"      # REST: FIVE_MINUTE
    FIFTEEN_MIN = "15m"  # REST: FIFTEEN_MINUTE
    ONE_HOUR = "1h"      # REST: ONE_HOUR
    FOUR_HOUR = "4h"     # REST: FOUR_HOUR (via TWO_HOUR * 2)
    ONE_DAY = "1d"       # REST: ONE_DAY

    @property
    def seconds(self) -> int:
        """Return timeframe in seconds."""
        mapping = {
            "10s": 10,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return mapping[self.value]

    @property
    def coinbase_granularity(self) -> str:
        """Return Coinbase API granularity value."""
        mapping = {
            "10s": "ONE_MINUTE",  # 10s is local, use 1m as base
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "1h": "ONE_HOUR",
            "4h": "TWO_HOUR",  # Aggregate 2 x 2H candles
            "1d": "ONE_DAY",
        }
        return mapping[self.value]

    @property
    def display_name(self) -> str:
        """Return human-readable timeframe name."""
        mapping = {
            "10s": "10 Second",
            "1m": "1 Minute",
            "5m": "5 Minute",
            "15m": "15 Minute",
            "1h": "1 Hour",
            "4h": "4 Hour",
            "1d": "Daily",
        }
        return mapping[self.value]


@dataclass(frozen=True)
class Candle:
    """Immutable OHLCV candle."""
    timestamp: int          # UNIX timestamp (start of candle)
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe

    @property
    def typical_price(self) -> float:
        """Return typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def hlc3(self) -> float:
        """Alias for typical_price."""
        return self.typical_price

    @property
    def ohlc4(self) -> float:
        """Return OHLC average."""
        return (self.open + self.high + self.low + self.close) / 4.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe.value,
        }


@dataclass
class Trade:
    """Individual trade from market_trades channel."""
    trade_id: str
    product_id: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    timestamp: float  # UNIX timestamp with milliseconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "product_id": self.product_id,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "timestamp": self.timestamp,
        }


@dataclass
class Ticker:
    """24h ticker statistics."""
    product_id: str
    price: float
    volume_24h: float
    low_24h: float
    high_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "low_24h": self.low_24h,
            "high_24h": self.high_24h,
            "price_change_24h": self.price_change_24h,
            "price_change_pct_24h": self.price_change_pct_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class BucketedOrderBook:
    """Order book bucketed into price bands."""
    bids: Dict[float, float]  # price_bucket -> total_size
    asks: Dict[float, float]  # price_bucket -> total_size
    mid_price: float
    spread: float
    best_bid: float
    best_ask: float
    timestamp: float
    bucket_size: float = 100.0

    @property
    def spread_bps(self) -> float:
        """Return spread in basis points."""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0

    def get_depth_at_level(self, price_distance: float, side: str = "bid") -> float:
        """
        Get total depth within price_distance of best price.

        Args:
            price_distance: Distance from best price
            side: "bid" or "ask"

        Returns:
            Total size at that depth level
        """
        total = 0.0
        if side == "bid":
            min_price = self.best_bid - price_distance
            for price, size in self.bids.items():
                if price >= min_price:
                    total += size
        else:
            max_price = self.best_ask + price_distance
            for price, size in self.asks.items():
                if price <= max_price:
                    total += size
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bids": self.bids,
            "asks": self.asks,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "timestamp": self.timestamp,
            "bucket_size": self.bucket_size,
        }


@dataclass
class IndicatorSnapshot:
    """Snapshot of all indicator values for a product and timeframe."""
    product_id: str
    timeframe: "Timeframe"  # Added: timeframe context
    timestamp: float

    # VWAP
    vwap: Optional[float] = None
    vwap_deviation: Optional[float] = None  # Price distance from VWAP

    # Trend indicators
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    trend_direction: str = "SIDEWAYS"  # UP, DOWN, SIDEWAYS

    # Volatility
    atr: Optional[float] = None
    atr_pct: Optional[float] = None  # ATR as % of price
    volatility: Optional[float] = None  # Annualized volatility

    # Market regime
    regime: str = "UNKNOWN"  # TRENDING, RANGING, VOLATILE

    # Momentum indicators
    rsi: Optional[float] = None  # Relative Strength Index (0-100)

    # Trend indicators
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volatility bands
    bb_upper: Optional[float] = None  # Bollinger Band upper
    bb_middle: Optional[float] = None  # Bollinger Band middle (SMA)
    bb_lower: Optional[float] = None  # Bollinger Band lower

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp,
            "vwap": self.vwap,
            "vwap_deviation": self.vwap_deviation,
            "adx": self.adx,
            "plus_di": self.plus_di,
            "minus_di": self.minus_di,
            "trend_direction": self.trend_direction,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "volatility": self.volatility,
            "regime": self.regime,
            "rsi": self.rsi,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
        }


@dataclass
class Alert:
    """Alert generated by the indicator engine."""
    alert_type: str  # regime_change, volume_spike, level_breach
    message: str
    product_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "message": self.message,
            "product_id": self.product_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "severity": self.severity,
        }


@dataclass
class PulseConfig:
    """Configuration for Pulse module."""

    # Products to track
    products: List[str] = field(default_factory=lambda: ["BTC-USD"])

    # Enabled timeframes (lazy loading)
    enabled_timeframes: Set[Timeframe] = field(
        default_factory=lambda: {Timeframe.ONE_MIN, Timeframe.FIVE_MIN}
    )

    # History limits (deque maxlen)
    candle_history_size: int = 300  # 300 bars per timeframe
    trade_history_size: int = 1000  # Last 1000 trades for tape

    # Order book bucketing
    bucket_size_usd: float = 100.0  # $100 price bands

    # WebSocket settings (passed to CoinbaseConnector)
    ws_reconnect_delay: int = 5
    ws_max_reconnect_attempts: int = 10

    # REST settings
    rest_rate_limit_per_sec: int = 10

    # Priming settings
    prime_bars_per_timeframe: int = 300
    prime_parallel: bool = True
    prime_smallest_first: bool = True

    # Indicator settings
    vwap_reset_hour_utc: int = 0  # Reset VWAP at midnight UTC
    adx_period: int = 14
    atr_period: int = 14
    volatility_lookback: int = 20

    # New indicator settings (standard periods)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # Alert thresholds (hardcoded initially)
    volume_spike_threshold: float = 3.0  # 3x average volume
    regime_change_adx_threshold: float = 25.0

    # 10-second candle settings
    ten_sec_wall_clock_aligned: bool = True
    ten_sec_empty_use_prev_close: bool = True

    # Update intervals (seconds)
    indicator_update_interval: float = 1.0  # How often to recalculate indicators

    # Logging
    log_namespace: str = "Pulse"

    def get_enabled_timeframes_list(self) -> List[Timeframe]:
        """Return enabled timeframes sorted by seconds (smallest first)."""
        return sorted(self.enabled_timeframes, key=lambda tf: tf.seconds)


# Callback events emitted by PulseCore
PULSE_EVENTS: tuple = (
    'candle_update',      # (candle: Candle)
    'orderbook_update',   # (book: BucketedOrderBook)
    'trade_update',       # (trade: Trade)
    'ticker_update',      # (ticker: Ticker)
    'vwap_update',        # (product_id: str, timeframe: Timeframe, vwap: float)
    'regime_update',      # (product_id: str, timeframe: Timeframe, snapshot: IndicatorSnapshot)
    'indicator_update',   # (product_id: str, timeframe: Timeframe, snapshot: IndicatorSnapshot)
    'alert',              # (alert: Alert)
    'connection_status',  # (connected: bool, message: str)
    'priming_progress',   # (product_id: str, timeframe: Timeframe, progress: float)
    'priming_complete',   # (product_id: str)
)
