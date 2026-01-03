"""
Pulse Indicators - Technical indicator calculations.

Contains:
- VWAPCalculator: Volume Weighted Average Price
- ADXCalculator: Average Directional Index
- ATRCalculator: Average True Range
- VolatilityCalculator: Rolling volatility
- IndicatorEngine: Coordinates all indicators
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from pulse.config import (
    Alert,
    Candle,
    IndicatorSnapshot,
    PulseConfig,
    Timeframe,
)

logger = logging.getLogger("Pulse.Indicators")


class VWAPCalculator:
    """
    Calculates Volume Weighted Average Price.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Features:
    - Resets at configurable hour (default: midnight UTC)
    - Updates on each candle close
    - Tracks deviation from current price
    """

    def __init__(self, reset_hour_utc: int = 0):
        """
        Initialize VWAP calculator.

        Args:
            reset_hour_utc: Hour (0-23) to reset VWAP daily
        """
        self.reset_hour_utc = reset_hour_utc

        self._cumulative_tp_volume: float = 0.0  # Sum of (typical_price * volume)
        self._cumulative_volume: float = 0.0
        self._last_reset_date: Optional[str] = None
        self._current_vwap: Optional[float] = None

    def _should_reset(self, timestamp: int) -> bool:
        """Check if VWAP should reset based on timestamp."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_date = dt.strftime("%Y-%m-%d")

        if self._last_reset_date is None:
            self._last_reset_date = current_date
            return True

        if current_date != self._last_reset_date and dt.hour >= self.reset_hour_utc:
            self._last_reset_date = current_date
            return True

        return False

    def update(self, candle: Candle) -> float:
        """
        Update VWAP with new candle.

        Args:
            candle: New candle data

        Returns:
            Current VWAP value
        """
        # Check for reset
        if self._should_reset(candle.timestamp):
            self.reset()

        # Calculate typical price and update cumulative values
        typical_price = candle.typical_price
        self._cumulative_tp_volume += typical_price * candle.volume
        self._cumulative_volume += candle.volume

        # Calculate VWAP
        if self._cumulative_volume > 0:
            self._current_vwap = self._cumulative_tp_volume / self._cumulative_volume
        else:
            self._current_vwap = candle.close

        return self._current_vwap

    def reset(self):
        """Reset VWAP calculation."""
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0
        self._current_vwap = None

    @property
    def current_vwap(self) -> Optional[float]:
        """Get current VWAP value."""
        return self._current_vwap

    def get_deviation(self, current_price: float) -> Optional[float]:
        """
        Get price deviation from VWAP.

        Args:
            current_price: Current price

        Returns:
            Deviation as percentage, or None if VWAP not available
        """
        if self._current_vwap is None or self._current_vwap == 0:
            return None
        return ((current_price - self._current_vwap) / self._current_vwap) * 100


class ADXCalculator:
    """
    Calculates Average Directional Index.

    ADX measures trend strength (not direction).
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging

    Also provides +DI and -DI for trend direction.
    """

    def __init__(self, period: int = 14):
        """
        Initialize ADX calculator.

        Args:
            period: Smoothing period (default 14)
        """
        self.period = period

        # Price history for calculations
        self._highs: Deque[float] = deque(maxlen=period + 1)
        self._lows: Deque[float] = deque(maxlen=period + 1)
        self._closes: Deque[float] = deque(maxlen=period + 1)

        # Smoothed values
        self._smoothed_plus_dm: Optional[float] = None
        self._smoothed_minus_dm: Optional[float] = None
        self._smoothed_tr: Optional[float] = None
        self._smoothed_dx: Optional[float] = None

        # Current values
        self._current_adx: Optional[float] = None
        self._current_plus_di: Optional[float] = None
        self._current_minus_di: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update ADX with new candle.

        Args:
            candle: New candle data

        Returns:
            Current ADX value if enough data, None otherwise
        """
        self._highs.append(candle.high)
        self._lows.append(candle.low)
        self._closes.append(candle.close)

        if len(self._highs) < 2:
            return None

        # Calculate True Range
        prev_close = self._closes[-2]
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close)
        )

        # Calculate Directional Movement
        prev_high = self._highs[-2]
        prev_low = self._lows[-2]

        up_move = candle.high - prev_high
        down_move = prev_low - candle.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        # Smoothing
        if self._smoothed_tr is None:
            # First value - need period candles
            if len(self._highs) < self.period + 1:
                return None

            # Initialize with simple sum
            self._smoothed_tr = tr
            self._smoothed_plus_dm = plus_dm
            self._smoothed_minus_dm = minus_dm
        else:
            # Wilder's smoothing
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + tr
            self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm

        # Calculate +DI and -DI
        if self._smoothed_tr > 0:
            self._current_plus_di = 100 * (self._smoothed_plus_dm / self._smoothed_tr)
            self._current_minus_di = 100 * (self._smoothed_minus_dm / self._smoothed_tr)
        else:
            self._current_plus_di = 0.0
            self._current_minus_di = 0.0

        # Calculate DX
        di_sum = self._current_plus_di + self._current_minus_di
        if di_sum > 0:
            dx = 100 * abs(self._current_plus_di - self._current_minus_di) / di_sum
        else:
            dx = 0.0

        # Smooth ADX
        if self._smoothed_dx is None:
            self._smoothed_dx = dx
            self._current_adx = dx
        else:
            self._smoothed_dx = ((self._smoothed_dx * (self.period - 1)) + dx) / self.period
            self._current_adx = self._smoothed_dx

        return self._current_adx

    @property
    def current_adx(self) -> Optional[float]:
        """Get current ADX value."""
        return self._current_adx

    @property
    def plus_di(self) -> Optional[float]:
        """Get current +DI value."""
        return self._current_plus_di

    @property
    def minus_di(self) -> Optional[float]:
        """Get current -DI value."""
        return self._current_minus_di

    @property
    def trend_direction(self) -> str:
        """
        Get trend direction based on +DI/-DI.

        Returns:
            'UP', 'DOWN', or 'SIDEWAYS'
        """
        if self._current_plus_di is None or self._current_minus_di is None:
            return "SIDEWAYS"

        if self._current_adx is not None and self._current_adx < 20:
            return "SIDEWAYS"

        if self._current_plus_di > self._current_minus_di:
            return "UP"
        elif self._current_minus_di > self._current_plus_di:
            return "DOWN"
        else:
            return "SIDEWAYS"

    def reset(self):
        """Reset ADX calculation."""
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._smoothed_plus_dm = None
        self._smoothed_minus_dm = None
        self._smoothed_tr = None
        self._smoothed_dx = None
        self._current_adx = None
        self._current_plus_di = None
        self._current_minus_di = None


class ATRCalculator:
    """
    Calculates Average True Range.

    ATR measures volatility.
    - Higher ATR = Higher volatility
    - Lower ATR = Lower volatility
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR calculator.

        Args:
            period: Smoothing period (default 14)
        """
        self.period = period

        self._prev_close: Optional[float] = None
        self._smoothed_atr: Optional[float] = None
        self._tr_values: Deque[float] = deque(maxlen=period)

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update ATR with new candle.

        Args:
            candle: New candle data

        Returns:
            Current ATR value if enough data, None otherwise
        """
        # Calculate True Range
        if self._prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low - self._prev_close)
            )

        self._prev_close = candle.close
        self._tr_values.append(tr)

        # Need enough values for initial ATR
        if len(self._tr_values) < self.period:
            return None

        # Calculate ATR
        if self._smoothed_atr is None:
            # First ATR is simple average
            self._smoothed_atr = sum(self._tr_values) / self.period
        else:
            # Wilder's smoothing
            self._smoothed_atr = ((self._smoothed_atr * (self.period - 1)) + tr) / self.period

        return self._smoothed_atr

    @property
    def current_atr(self) -> Optional[float]:
        """Get current ATR value."""
        return self._smoothed_atr

    def get_atr_percent(self, current_price: float) -> Optional[float]:
        """
        Get ATR as percentage of price.

        Args:
            current_price: Current price

        Returns:
            ATR as percentage, or None if not available
        """
        if self._smoothed_atr is None or current_price == 0:
            return None
        return (self._smoothed_atr / current_price) * 100

    def reset(self):
        """Reset ATR calculation."""
        self._prev_close = None
        self._smoothed_atr = None
        self._tr_values.clear()


class VolatilityCalculator:
    """
    Calculates rolling volatility.

    Uses log returns standard deviation, annualized.
    """

    def __init__(self, lookback: int = 20, annualize: bool = True):
        """
        Initialize volatility calculator.

        Args:
            lookback: Number of periods for calculation
            annualize: Whether to annualize the volatility
        """
        self.lookback = lookback
        self.annualize = annualize

        self._closes: Deque[float] = deque(maxlen=lookback + 1)
        self._current_volatility: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update volatility with new candle.

        Args:
            candle: New candle data

        Returns:
            Current volatility if enough data, None otherwise
        """
        self._closes.append(candle.close)

        if len(self._closes) < self.lookback + 1:
            return None

        # Calculate log returns
        log_returns = []
        closes_list = list(self._closes)
        for i in range(1, len(closes_list)):
            if closes_list[i - 1] > 0:
                log_return = math.log(closes_list[i] / closes_list[i - 1])
                log_returns.append(log_return)

        if len(log_returns) < 2:
            return None

        # Calculate standard deviation
        mean_return = sum(log_returns) / len(log_returns)
        squared_diffs = [(r - mean_return) ** 2 for r in log_returns]
        variance = sum(squared_diffs) / (len(squared_diffs) - 1)
        std_dev = math.sqrt(variance)

        # Annualize if requested
        if self.annualize:
            # Assume 1-minute candles for annualization
            # 525,600 minutes per year
            periods_per_year = 525600 / 60  # Adjust based on timeframe
            self._current_volatility = std_dev * math.sqrt(periods_per_year) * 100
        else:
            self._current_volatility = std_dev * 100

        return self._current_volatility

    @property
    def current_volatility(self) -> Optional[float]:
        """Get current volatility (as percentage)."""
        return self._current_volatility

    def reset(self):
        """Reset volatility calculation."""
        self._closes.clear()
        self._current_volatility = None


class RSICalculator:
    """
    Calculates Relative Strength Index (RSI).

    RSI measures momentum using gains vs losses over a period.
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    - RSI = 50: Neutral
    """

    def __init__(self, period: int = 14):
        """
        Initialize RSI calculator.

        Args:
            period: Lookback period (default 14)
        """
        self.period = period

        self._closes: Deque[float] = deque(maxlen=period + 1)
        self._current_rsi: Optional[float] = None

        # Smoothed gains/losses (Wilder's smoothing)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update RSI with new candle.

        Args:
            candle: New candle data

        Returns:
            Current RSI value if enough data, None otherwise
        """
        self._closes.append(candle.close)

        if len(self._closes) < 2:
            return None

        # Calculate change
        change = self._closes[-1] - self._closes[-2]
        gain = max(0, change)
        loss = max(0, -change)

        # Initialize averages on first calculation
        if self._avg_gain is None:
            if len(self._closes) < self.period + 1:
                return None

            # Calculate initial average gains/losses
            gains = []
            losses = []
            for i in range(1, len(self._closes)):
                c = self._closes[i] - self._closes[i - 1]
                if c > 0:
                    gains.append(c)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-c)

            self._avg_gain = sum(gains) / self.period
            self._avg_loss = sum(losses) / self.period
        else:
            # Wilder's smoothing
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period

        # Calculate RS and RSI
        if self._avg_loss == 0:
            self._current_rsi = 100.0 if self._avg_gain > 0 else 50.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._current_rsi = 100.0 - (100.0 / (1.0 + rs))

        return self._current_rsi

    @property
    def current_rsi(self) -> Optional[float]:
        """Get current RSI value."""
        return self._current_rsi

    def reset(self):
        """Reset RSI calculation."""
        self._closes.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._current_rsi = None


class MACDCalculator:
    """
    Calculates Moving Average Convergence Divergence (MACD).

    MACD = EMA(12) - EMA(26)
    Signal = EMA(9) of MACD
    Histogram = MACD - Signal
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD calculator.

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal EMA period (default 9)
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal

        self._closes: Deque[float] = deque(maxlen=slow + 1)

        # EMA calculations
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._macd_line: Optional[float] = None
        self._signal_line: Optional[float] = None
        self._macd_histogram: Optional[float] = None

        # Signal line EMA values
        self._macd_values: Deque[float] = deque(maxlen=signal)

    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate EMA for a list of values."""
        if not data or len(data) < period:
            return data[-1] if data else 0.0

        # Multiplier for EMA
        multiplier = 2.0 / (period + 1)

        # Simple average for first value
        ema = sum(data[:period]) / period

        # Calculate EMA for remaining values
        for price in data[period:]:
            ema = price * multiplier + ema * (1 - multiplier)

        return ema

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update MACD with new candle.

        Args:
            candle: New candle data

        Returns:
            Current MACD value if enough data, None otherwise
        """
        self._closes.append(candle.close)

        if len(self._closes) < self.slow:
            return None

        closes_list = list(self._closes)

        # Calculate EMAs using most recent data
        self._ema_fast = self._calculate_ema(closes_list, self.fast)
        self._ema_slow = self._calculate_ema(closes_list, self.slow)

        # Calculate MACD line
        self._macd_line = self._ema_fast - self._ema_slow

        # Track MACD values for signal line
        self._macd_values.append(self._macd_line)

        # Calculate signal line (EMA of MACD)
        if len(self._macd_values) >= self.signal:
            self._signal_line = self._calculate_ema(list(self._macd_values), self.signal)
            self._macd_histogram = self._macd_line - self._signal_line
        else:
            self._signal_line = None
            self._macd_histogram = None

        return self._macd_line

    @property
    def macd_line(self) -> Optional[float]:
        """Get MACD line."""
        return self._macd_line

    @property
    def signal_line(self) -> Optional[float]:
        """Get signal line."""
        return self._signal_line

    @property
    def histogram(self) -> Optional[float]:
        """Get MACD histogram."""
        return self._macd_histogram

    def reset(self):
        """Reset MACD calculation."""
        self._closes.clear()
        self._ema_fast = None
        self._ema_slow = None
        self._macd_line = None
        self._signal_line = None
        self._macd_histogram = None
        self._macd_values.clear()


class BollingerBandsCalculator:
    """
    Calculates Bollinger Bands.

    Middle Band = SMA(period)
    Upper Band = Middle + (std_dev * standard_deviation)
    Lower Band = Middle - (std_dev * standard_deviation)

    Bands contract = low volatility (squeeze)
    Bands expand = high volatility (breakout)
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands calculator.

        Args:
            period: SMA period (default 20)
            std_dev: Number of standard deviations (default 2.0)
        """
        self.period = period
        self.std_dev = std_dev

        self._closes: Deque[float] = deque(maxlen=period)

        # Current values
        self._middle_band: Optional[float] = None
        self._upper_band: Optional[float] = None
        self._lower_band: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update Bollinger Bands with new candle.

        Args:
            candle: New candle data

        Returns:
            Current middle band (SMA) if enough data, None otherwise
        """
        self._closes.append(candle.close)

        if len(self._closes) < self.period:
            return None

        closes_list = list(self._closes)

        # Calculate SMA (middle band)
        self._middle_band = sum(closes_list) / self.period

        # Calculate standard deviation
        variance = sum((x - self._middle_band) ** 2 for x in closes_list) / self.period
        std = math.sqrt(variance)

        # Calculate bands
        self._upper_band = self._middle_band + (self.std_dev * std)
        self._lower_band = self._middle_band - (self.std_dev * std)

        return self._middle_band

    @property
    def middle_band(self) -> Optional[float]:
        """Get middle band (SMA)."""
        return self._middle_band

    @property
    def upper_band(self) -> Optional[float]:
        """Get upper band."""
        return self._upper_band

    @property
    def lower_band(self) -> Optional[float]:
        """Get lower band."""
        return self._lower_band

    def get_band_width(self) -> Optional[float]:
        """
        Get Bollinger Band width (as percentage of middle).

        Returns:
            Width as % of middle band, or None if not available
        """
        if self._upper_band is None or self._lower_band is None or self._middle_band is None:
            return None
        if self._middle_band == 0:
            return None
        return ((self._upper_band - self._lower_band) / self._middle_band) * 100

    def get_position(self, price: float) -> Optional[float]:
        """
        Get price position relative to bands.

        Returns:
            0.0 = at lower band, 0.5 = at middle, 1.0 = at upper band, or None
        """
        if self._upper_band is None or self._lower_band is None:
            return None
        if self._upper_band == self._lower_band:
            return None
        return (price - self._lower_band) / (self._upper_band - self._lower_band)

    def reset(self):
        """Reset Bollinger Bands calculation."""
        self._closes.clear()
        self._middle_band = None
        self._upper_band = None
        self._lower_band = None


class IndicatorEngine:
    """
    Coordinates all indicator calculations per-timeframe.

    Features:
    - Per-product, per-timeframe indicator state
    - All indicators calculated for each timeframe
    - Snapshot creation with timeframe context
    """

    def __init__(
        self,
        config: PulseConfig,
        on_indicator_update: Optional[Callable[[IndicatorSnapshot], Coroutine]] = None,
        on_alert: Optional[Callable[[Alert], Coroutine]] = None,
    ):
        """
        Initialize indicator engine.

        Args:
            config: Pulse configuration
            on_indicator_update: Callback for indicator updates (snapshot per timeframe)
            on_alert: Callback for alerts
        """
        self.config = config
        self.on_indicator_update = on_indicator_update
        self.on_alert = on_alert

        # Per-product, per-timeframe calculators
        # Structure: product_id -> timeframe -> calculator_instance
        self._vwap: Dict[str, Dict[Timeframe, VWAPCalculator]] = {}
        self._adx: Dict[str, Dict[Timeframe, ADXCalculator]] = {}
        self._atr: Dict[str, Dict[Timeframe, ATRCalculator]] = {}
        self._volatility: Dict[str, Dict[Timeframe, VolatilityCalculator]] = {}
        self._rsi: Dict[str, Dict[Timeframe, RSICalculator]] = {}
        self._macd: Dict[str, Dict[Timeframe, MACDCalculator]] = {}
        self._bb: Dict[str, Dict[Timeframe, BollingerBandsCalculator]] = {}

        # Per-product, per-timeframe state
        self._last_snapshots: Dict[str, Dict[Timeframe, IndicatorSnapshot]] = {}
        self._last_prices: Dict[str, Dict[Timeframe, float]] = {}
        self._volume_history: Dict[str, Dict[Timeframe, Deque[float]]] = {}

    def _ensure_calculators(self, product_id: str, timeframe: Timeframe):
        """Ensure calculators exist for product and timeframe."""
        # Ensure product dictionaries exist
        if product_id not in self._vwap:
            self._vwap[product_id] = {}
            self._adx[product_id] = {}
            self._atr[product_id] = {}
            self._volatility[product_id] = {}
            self._rsi[product_id] = {}
            self._macd[product_id] = {}
            self._bb[product_id] = {}
            self._last_snapshots[product_id] = {}
            self._last_prices[product_id] = {}
            self._volume_history[product_id] = {}

        # Ensure timeframe calculators exist
        if timeframe not in self._vwap[product_id]:
            self._vwap[product_id][timeframe] = VWAPCalculator(
                reset_hour_utc=self.config.vwap_reset_hour_utc
            )
            self._adx[product_id][timeframe] = ADXCalculator(period=self.config.adx_period)
            self._atr[product_id][timeframe] = ATRCalculator(period=self.config.atr_period)
            self._volatility[product_id][timeframe] = VolatilityCalculator(
                lookback=self.config.volatility_lookback
            )
            self._rsi[product_id][timeframe] = RSICalculator(period=self.config.rsi_period)
            self._macd[product_id][timeframe] = MACDCalculator(
                fast=self.config.macd_fast,
                slow=self.config.macd_slow,
                signal=self.config.macd_signal,
            )
            self._bb[product_id][timeframe] = BollingerBandsCalculator(
                period=self.config.bb_period,
                std_dev=self.config.bb_std_dev,
            )
            self._volume_history[product_id][timeframe] = deque(maxlen=20)

    async def update(self, product_id: str, candle: Candle) -> IndicatorSnapshot:
        """
        Update all indicators with new candle.

        Args:
            product_id: Product ID
            candle: New candle data (includes timeframe)

        Returns:
            Updated indicator snapshot
        """
        timeframe = candle.timeframe
        self._ensure_calculators(product_id, timeframe)

        # Update all calculators
        vwap = self._vwap[product_id][timeframe].update(candle)
        adx = self._adx[product_id][timeframe].update(candle)
        atr = self._atr[product_id][timeframe].update(candle)
        volatility = self._volatility[product_id][timeframe].update(candle)
        rsi = self._rsi[product_id][timeframe].update(candle)
        macd_line = self._macd[product_id][timeframe].update(candle)
        bb_middle = self._bb[product_id][timeframe].update(candle)

        # Track volume
        self._volume_history[product_id][timeframe].append(candle.volume)
        self._last_prices[product_id][timeframe] = candle.close

        # Calculate additional metrics
        vwap_deviation = None
        if vwap is not None:
            vwap_deviation = self._vwap[product_id][timeframe].get_deviation(candle.close)

        atr_pct = None
        if atr is not None and candle.close > 0:
            atr_pct = (atr / candle.close) * 100

        # Determine trend direction
        trend_direction = self._adx[product_id][timeframe].trend_direction

        # Determine market regime
        regime = self._determine_regime(product_id, timeframe, adx, volatility)

        # Get MACD values
        macd_signal = self._macd[product_id][timeframe].signal_line
        macd_histogram = self._macd[product_id][timeframe].histogram

        # Get Bollinger Bands values
        bb_upper = self._bb[product_id][timeframe].upper_band
        bb_lower = self._bb[product_id][timeframe].lower_band

        # Create snapshot with timeframe context
        snapshot = IndicatorSnapshot(
            product_id=product_id,
            timeframe=timeframe,
            timestamp=time.time(),
            vwap=vwap,
            vwap_deviation=vwap_deviation,
            adx=adx,
            plus_di=self._adx[product_id][timeframe].plus_di,
            minus_di=self._adx[product_id][timeframe].minus_di,
            trend_direction=trend_direction,
            atr=atr,
            atr_pct=atr_pct,
            volatility=volatility,
            regime=regime,
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
        )

        # Check for alerts
        await self._check_alerts(product_id, timeframe, snapshot, candle)

        # Store snapshot
        self._last_snapshots[product_id][timeframe] = snapshot

        # Emit update
        if self.on_indicator_update:
            await self.on_indicator_update(snapshot)

        return snapshot

    def _determine_regime(
        self,
        product_id: str,
        timeframe: Timeframe,
        adx: Optional[float],
        volatility: Optional[float]
    ) -> str:
        """
        Determine market regime.

        Args:
            product_id: Product ID
            timeframe: Timeframe
            adx: Current ADX value
            volatility: Current volatility

        Returns:
            Regime string: TRENDING, RANGING, or VOLATILE
        """
        if adx is None:
            return "UNKNOWN"

        # High ADX = trending
        if adx >= self.config.regime_change_adx_threshold:
            return "TRENDING"

        # High volatility but low ADX = volatile ranging
        if volatility is not None and volatility > 50:
            return "VOLATILE"

        # Low ADX = ranging
        return "RANGING"

    async def _check_alerts(
        self,
        product_id: str,
        timeframe: Timeframe,
        snapshot: IndicatorSnapshot,
        candle: Candle
    ):
        """
        Check for alert conditions.

        Args:
            product_id: Product ID
            timeframe: Timeframe
            snapshot: Current indicator snapshot
            candle: Current candle
        """
        if not self.on_alert:
            return

        # Get previous snapshot for this timeframe
        prev_snapshot = self._last_snapshots.get(product_id, {}).get(timeframe)

        # Check regime change
        if prev_snapshot and prev_snapshot.regime != snapshot.regime:
            if snapshot.regime != "UNKNOWN":
                alert = Alert(
                    alert_type="regime_change",
                    message=f"Market regime changed from {prev_snapshot.regime} to {snapshot.regime} on {timeframe.value}",
                    product_id=product_id,
                    timestamp=time.time(),
                    data={
                        "timeframe": timeframe.value,
                        "previous_regime": prev_snapshot.regime,
                        "new_regime": snapshot.regime,
                        "adx": snapshot.adx,
                    },
                    severity="WARNING" if snapshot.regime == "VOLATILE" else "INFO",
                )
                await self.on_alert(alert)

        # Check volume spike
        volume_history = self._volume_history.get(product_id, {}).get(timeframe, deque())
        if len(volume_history) >= 5 and candle.volume > 0:
            avg_volume = sum(list(volume_history)[:-1]) / (len(volume_history) - 1)
            if avg_volume > 0:
                volume_ratio = candle.volume / avg_volume
                if volume_ratio >= self.config.volume_spike_threshold:
                    alert = Alert(
                        alert_type="volume_spike",
                        message=f"Volume spike on {timeframe.value}: {volume_ratio:.1f}x average",
                        product_id=product_id,
                        timestamp=time.time(),
                        data={
                            "timeframe": timeframe.value,
                            "volume": candle.volume,
                            "avg_volume": avg_volume,
                            "ratio": volume_ratio,
                        },
                        severity="WARNING",
                    )
                    await self.on_alert(alert)

    def get_snapshot(self, product_id: str, timeframe: Optional[Timeframe] = None) -> Optional[IndicatorSnapshot]:
        """
        Get current indicator snapshot for product and timeframe.

        Args:
            product_id: Product ID
            timeframe: Timeframe (if None, returns latest from any timeframe)

        Returns:
            Current snapshot or None
        """
        if product_id not in self._last_snapshots:
            return None

        if timeframe:
            return self._last_snapshots[product_id].get(timeframe)

        # Return latest snapshot (any timeframe)
        snapshots = self._last_snapshots[product_id].values()
        return max(snapshots, key=lambda s: s.timestamp) if snapshots else None

    def get_vwap(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current VWAP for product and timeframe."""
        calc = self._vwap.get(product_id, {}).get(timeframe)
        return calc.current_vwap if calc else None

    def get_adx(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current ADX for product and timeframe."""
        calc = self._adx.get(product_id, {}).get(timeframe)
        return calc.current_adx if calc else None

    def get_atr(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current ATR for product and timeframe."""
        calc = self._atr.get(product_id, {}).get(timeframe)
        return calc.current_atr if calc else None

    def get_volatility(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current volatility for product and timeframe."""
        calc = self._volatility.get(product_id, {}).get(timeframe)
        return calc.current_volatility if calc else None

    def get_rsi(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        """Get current RSI for product and timeframe."""
        calc = self._rsi.get(product_id, {}).get(timeframe)
        return calc.current_rsi if calc else None

    def reset(self, product_id: Optional[str] = None, timeframe: Optional[Timeframe] = None):
        """
        Reset indicator state.

        Args:
            product_id: Specific product to reset, or None for all
            timeframe: Specific timeframe to reset, or None for all
        """
        if product_id and timeframe:
            # Reset specific timeframe for specific product
            if product_id in self._vwap and timeframe in self._vwap[product_id]:
                self._vwap[product_id][timeframe].reset()
            if product_id in self._adx and timeframe in self._adx[product_id]:
                self._adx[product_id][timeframe].reset()
            if product_id in self._atr and timeframe in self._atr[product_id]:
                self._atr[product_id][timeframe].reset()
            if product_id in self._volatility and timeframe in self._volatility[product_id]:
                self._volatility[product_id][timeframe].reset()
            if product_id in self._rsi and timeframe in self._rsi[product_id]:
                self._rsi[product_id][timeframe].reset()
            if product_id in self._macd and timeframe in self._macd[product_id]:
                self._macd[product_id][timeframe].reset()
            if product_id in self._bb and timeframe in self._bb[product_id]:
                self._bb[product_id][timeframe].reset()
            self._last_snapshots[product_id].pop(timeframe, None)
            self._last_prices[product_id].pop(timeframe, None)
            self._volume_history[product_id].pop(timeframe, None)
        elif product_id:
            # Reset all timeframes for specific product
            if product_id in self._vwap:
                for calc in self._vwap[product_id].values():
                    calc.reset()
            if product_id in self._adx:
                for calc in self._adx[product_id].values():
                    calc.reset()
            if product_id in self._atr:
                for calc in self._atr[product_id].values():
                    calc.reset()
            if product_id in self._volatility:
                for calc in self._volatility[product_id].values():
                    calc.reset()
            if product_id in self._rsi:
                for calc in self._rsi[product_id].values():
                    calc.reset()
            if product_id in self._macd:
                for calc in self._macd[product_id].values():
                    calc.reset()
            if product_id in self._bb:
                for calc in self._bb[product_id].values():
                    calc.reset()
            self._last_snapshots[product_id].clear()
            self._last_prices[product_id].clear()
            self._volume_history[product_id].clear()
        else:
            # Reset everything
            for product_cals in self._vwap.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._adx.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._atr.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._volatility.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._rsi.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._macd.values():
                for calc in product_cals.values():
                    calc.reset()
            for product_cals in self._bb.values():
                for calc in product_cals.values():
                    calc.reset()
            self._last_snapshots.clear()
            self._last_prices.clear()
            self._volume_history.clear()
