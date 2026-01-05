# Pulse Terminal UI - Real-time Market Dashboard
# Framework: Textual
# Data: Coinbase via PulseCore
# Purpose: Technical analysis signals display

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from pulse.config import PulseConfig, Timeframe, Ticker, IndicatorSnapshot, Trade, Candle
from pulse.core import PulseCore
from pulse.gui.scoring import compute_signal_score

logger = logging.getLogger("Pulse.GUI")

# Timeframe to card ID mapping
TIMEFRAME_CARD_MAP = {
    Timeframe.TEN_SEC: "ltf-10s",
    Timeframe.ONE_MIN: "ltf-1m",
    Timeframe.FIFTEEN_MIN: "ltf-15m",
    Timeframe.ONE_HOUR: "htf-1h",
    Timeframe.FOUR_HOUR: "htf-4h",
    Timeframe.ONE_DAY: "htf-daily",
}

# Card ID to display timeframe label
CARD_TIMEFRAME_LABELS = {
    "ltf-10s": "10s HFT",
    "ltf-1m": "1m Scalp",
    "ltf-15m": "15m Intraday",
    "htf-1h": "1h",
    "htf-4h": "4h",
    "htf-daily": "Daily",
}

# Data Models
@dataclass
class MarketHeader:
    symbol: str
    price: float
    change_pct: float
    volume_24h: float


@dataclass
class Signal:
    label: str
    score: int
    description: str
    metrics: Dict[str, str]


# Widgets
class CombinedHeaderWidget(Static):
    data: Optional[MarketHeader] = None
    short_term_signal: str = "MIXED"
    long_term_signal: str = "MIXED"
    short_term_type: str = "mixed"  # "bullish", "bearish", or "mixed"
    long_term_type: str = "mixed"

    def _get_signal_color(self, signal_type: str) -> str:
        """Get color markup for signal type."""
        if signal_type == "bullish":
            return "[#10b981]"  # Green
        elif signal_type == "bearish":
            return "[#ef4444]"  # Red
        else:  # mixed
            return "[#f59e0b]"  # Yellow/Orange

    def render(self) -> str:
        if not self.data:
            return "Loading..."
        sign = "+" if self.data.change_pct >= 0 else ""
        # Color code percentage change: green for positive, red for negative
        if self.data.change_pct >= 0:
            pct_color = "[#10b981]"  # Green
        else:
            pct_color = "[#ef4444]"  # Red

        st_color = self._get_signal_color(self.short_term_type)
        lt_color = self._get_signal_color(self.long_term_type)
        return (
            f"{self.data.symbol}  ${self.data.price:,.2f}  "
            f"{pct_color}{sign}{self.data.change_pct:.2f}%[/]  |  "
            f"24h Vol: ${self.data.volume_24h/1e9:.1f}B  |  "
            f"ST: {st_color}{self.short_term_signal}[/]  |  "
            f"LT: {lt_color}{self.long_term_signal}[/]"
        )


class SignalCard(Static):
    signal = reactive(Signal("Loading", 50, "Waiting for data...", {}))

    def __init__(self, title: str, timeframe: str, signal: Signal, id: Optional[str] = None):
        super().__init__(id=id)
        self.title = title
        self.timeframe = (timeframe or "").strip()
        self.series: List[float] = []  # Initialize BEFORE setting signal (reactive)
        self.trades: List[Trade] = []  # Recent trades for order flow visualization
        self.current_candle: Optional[Candle] = None  # Current candle for 1m visualization
        self.recent_candles: List[Candle] = []  # Recent candles for volume comparison
        self.recent_15m_candles: List[Candle] = []  # Recent 15m candles for intraday analysis
        self.recent_1h_candles: List[Candle] = []  # Recent 1h candles for hourly analysis
        self.recent_4h_candles: List[Candle] = []  # Recent 4h candles for swing analysis
        self.recent_daily_candles: List[Candle] = []  # Recent daily candles for market structure analysis
        self.current_price: Optional[float] = None  # Current price for position indicator
        self.indicator_snapshot: Optional[IndicatorSnapshot] = None  # Indicator snapshot for 1h analysis
        self.indicator_snapshot_4h: Optional[IndicatorSnapshot] = None  # Indicator snapshot for 4h analysis
        self.indicator_snapshot_daily: Optional[IndicatorSnapshot] = None  # Indicator snapshot for daily analysis
        self.signal = signal  # This triggers watch_signal() which needs self.series

    def watch_signal(self, signal: Signal) -> None:
        """Reactively update when signal changes."""
        # Ensure series is initialized
        if not hasattr(self, 'series'):
            self.series: List[float] = []
        
        # Remove old score class
        for cls in list(self.classes):
            if cls.startswith("score-"):
                self.remove_class(cls)

        # Add new score class
        self.add_class(self._get_score_class())

        # Re-render
        self.update(self.render())

    def order_flow(self) -> tuple:
        """
        Generate order flow visualization combining:
        1. Buy/Sell Pressure Bar - cumulative buy vs sell volume
        2. Trade Flow Histogram - recent trades as colored bars
        
        Returns: (pressure_bar, histogram, delta_str, delta_color)
        """
        # Ensure trades is initialized
        if not hasattr(self, 'trades'):
            self.trades: List[Trade] = []
        
        if len(self.trades) == 0:
            return ("", "", "", "")
        
        # Get recent trades (last 15 for histogram)
        recent_trades = self.trades[-15:] if len(self.trades) > 15 else self.trades
        
        # Calculate buy/sell pressure
        buy_volume = sum(t.size for t in recent_trades if t.side == "BUY")
        sell_volume = sum(t.size for t in recent_trades if t.side == "SELL")
        total_volume = buy_volume + sell_volume
        
        # Calculate delta (net buy pressure)
        delta = buy_volume - sell_volume
        delta_pct = (delta / total_volume * 100) if total_volume > 0 else 0
        
        # Generate buy/sell pressure bar (20 chars wide)
        bar_width = 20
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
            buy_chars = int(buy_ratio * bar_width)
            sell_chars = bar_width - buy_chars
            pressure_bar = f"[#10b981]{'█' * buy_chars}[/][#ef4444]{'█' * sell_chars}[/]"
        else:
            pressure_bar = "░" * bar_width
        
        # Generate trade flow histogram (one bar per trade, max 15)
        # Use block characters of different heights to show size variation
        histogram_bars = []
        if recent_trades:
            # Normalize sizes for visualization (use max size as reference)
            max_size = max(t.size for t in recent_trades) if recent_trades else 1
            # Use different block characters for size: ▁▂▃▄▅▆▇█
            size_chars = "▁▂▃▄▅▆▇█"
            
            for trade in recent_trades:
                # Calculate bar index (0-7)
                if max_size > 0:
                    size_ratio = trade.size / max_size
                    char_idx = min(int(size_ratio * (len(size_chars) - 1)), len(size_chars) - 1)
                else:
                    char_idx = 0
                bar_char = size_chars[char_idx]
                color = "[#10b981]" if trade.side == "BUY" else "[#ef4444]"
                histogram_bars.append(f"{color}{bar_char}[/]")
            
            histogram = "".join(histogram_bars)  # No spaces for compact display
        else:
            histogram = ""
        
        # Format delta string with color
        if delta > 0:
            delta_color = "[#10b981]"
            delta_str = f"Δ: +{delta:.2f} BTC ({delta_pct:+.1f}%)"
        elif delta < 0:
            delta_color = "[#ef4444]"
            delta_str = f"Δ: {delta:.2f} BTC ({delta_pct:+.1f}%)"
        else:
            delta_color = "[#f59e0b]"
            delta_str = f"Δ: {delta:.2f} BTC (0.0%)"
        
        return (pressure_bar, histogram, delta_str, delta_color)

    def candle_summary(self) -> tuple:
        """
        Generate 1-minute candle summary visualization.
        Shows: OHLC, volume comparison, price position in range.
        
        Returns: (candle_bar, volume_str, position_str, position_color)
        """
        if not self.current_candle or not self.current_price:
            return ("", "", "", "")
        
        candle = self.current_candle
        price = self.current_price
        
        # Calculate price position within candle range (0.0 to 1.0)
        if candle.high != candle.low:
            position_ratio = (price - candle.low) / (candle.high - candle.low)
        else:
            position_ratio = 0.5
        
        # Generate visual candle bar (20 chars wide)
        # Shows: Low (left) to High (right), with Open, Close, and Current price
        bar_width = 20
        if candle.high != candle.low:
            # Calculate positions (0 = low, bar_width-1 = high)
            open_pos = int(((candle.open - candle.low) / (candle.high - candle.low)) * (bar_width - 1))
            close_pos = int(((candle.close - candle.low) / (candle.high - candle.low)) * (bar_width - 1))
            current_pos = int(position_ratio * (bar_width - 1))
            
            # Clamp positions
            open_pos = max(0, min(bar_width - 1, open_pos))
            close_pos = max(0, min(bar_width - 1, close_pos))
            current_pos = max(0, min(bar_width - 1, current_pos))
            
            # Determine candle color (green if close > open, red otherwise)
            is_green = candle.close >= candle.open
            candle_color = "[#10b981]" if is_green else "[#ef4444]"
            
            # Build bar: show range with open/close body and current price marker
            bar_chars = ["░"] * bar_width
            
            # Draw body (between open and close)
            body_start = min(open_pos, close_pos)
            body_end = max(open_pos, close_pos)
            for i in range(body_start, body_end + 1):
                bar_chars[i] = "█"
            
            # Mark open and close positions
            bar_chars[open_pos] = "│"  # Open
            bar_chars[close_pos] = "│"  # Close
            
            # Mark current price (use different char if it overlaps with open/close)
            if current_pos == open_pos or current_pos == close_pos:
                bar_chars[current_pos] = "●"
            else:
                bar_chars[current_pos] = "●"
            
            candle_bar = f"{candle_color}{''.join(bar_chars)}[/]"
        else:
            # Flat candle (high == low)
            candle_bar = f"[#f59e0b]{'─' * bar_width}[/]"
        
        # Enhanced volume comparison (current vs recent average with trend)
        if self.recent_candles and len(self.recent_candles) >= 3:
            # Calculate average volume from previous candles (exclude current)
            prev_candles = self.recent_candles[:-1]
            avg_volume = sum(c.volume for c in prev_candles) / len(prev_candles) if prev_candles else candle.volume
            
            # Calculate volume ratio
            volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine volume trend (comparing last 3 candles)
            if len(prev_candles) >= 2:
                recent_volumes = [c.volume for c in prev_candles[-2:]]
                recent_avg = sum(recent_volumes) / len(recent_volumes)
                older_avg = sum(c.volume for c in prev_candles[:-2]) / max(1, len(prev_candles) - 2) if len(prev_candles) > 2 else recent_avg
                volume_trend = "↑" if recent_avg > older_avg * 1.1 else ("↓" if recent_avg < older_avg * 0.9 else "→")
            else:
                volume_trend = "→"
            
            # Color coding based on volume ratio
            if volume_ratio > 2.0:
                vol_color = "[#10b981]"  # Very high volume
                vol_indicator = "▲▲"
            elif volume_ratio > 1.5:
                vol_color = "[#10b981]"  # High volume
                vol_indicator = "▲"
            elif volume_ratio < 0.3:
                vol_color = "[#ef4444]"  # Very low volume
                vol_indicator = "▼▼"
            elif volume_ratio < 0.5:
                vol_color = "[#ef4444]"  # Low volume
                vol_indicator = "▼"
            else:
                vol_color = "[#f59e0b]"  # Normal volume
                vol_indicator = "→"
            
            # Format: show ratio, trend, and actual volume
            volume_str = f"{vol_color}Vol: {volume_ratio:.1f}x {vol_indicator} {volume_trend} ({candle.volume:.2f} BTC)[/]"
        elif self.recent_candles:
            # Not enough data for comparison, just show volume
            volume_str = f"Vol: {candle.volume:.2f} BTC"
        else:
            volume_str = "Vol: --"
        
        # Price position indicator
        position_pct = position_ratio * 100
        if position_pct > 75:
            position_color = "[#10b981]"
            position_str = f"Price: {position_pct:.0f}% ↑ (near high)"
        elif position_pct < 25:
            position_color = "[#ef4444]"
            position_str = f"Price: {position_pct:.0f}% ↓ (near low)"
        else:
            position_color = "[#f59e0b]"
            position_str = f"Price: {position_pct:.0f}% (mid-range)"
        
        return (candle_bar, volume_str, position_str, position_color)

    def intraday_analysis(self) -> tuple:
        """
        Generate 15-minute intraday analysis combining:
        1. Multi-Candle Pattern Recognition
        2. Intraday Range Analysis
        3. Momentum Oscillator
        
        Returns: (pattern_str, range_str, momentum_str, momentum_color)
        """
        if not self.recent_15m_candles or len(self.recent_15m_candles) < 3:
            return ("", "", "", "")
        
        candles = self.recent_15m_candles[-6:] if len(self.recent_15m_candles) >= 6 else self.recent_15m_candles
        current_price = self.current_price if self.current_price else candles[-1].close
        
        # 1. Multi-Candle Pattern Recognition
        pattern_str = self._detect_pattern(candles)
        
        # 2. Intraday Range Analysis (last 20 candles or available)
        range_candles = self.recent_15m_candles[-20:] if len(self.recent_15m_candles) >= 20 else self.recent_15m_candles
        range_str = self._calculate_range(range_candles, current_price)
        
        # 3. Momentum Oscillator (rate of change over last 3-5 candles)
        momentum_str, momentum_color = self._calculate_momentum(candles)
        
        return (pattern_str, range_str, momentum_str, momentum_color)
    
    def _detect_pattern(self, candles: List[Candle]) -> str:
        """Detect common candlestick patterns from recent candles."""
        if len(candles) < 2:
            return ""
        
        # Count consecutive green/red candles
        consecutive_green = 0
        consecutive_red = 0
        for candle in reversed(candles[-5:]):  # Check last 5
            if candle.close >= candle.open:
                consecutive_green += 1
                consecutive_red = 0
            else:
                consecutive_red += 1
                consecutive_green = 0
        
        # Pattern detection
        if len(candles) >= 3:
            c1, c2, c3 = candles[-3], candles[-2], candles[-1]
            
            # Engulfing patterns
            if (c2.close < c2.open and  # Previous red
                c3.close > c3.open and  # Current green
                c3.open < c2.close and  # Current opens below prev close
                c3.close > c2.open):    # Current closes above prev open
                return "[#10b981]Pattern: Bullish Engulfing ▲[/]"
            
            if (c2.close > c2.open and  # Previous green
                c3.close < c3.open and  # Current red
                c3.open > c2.close and  # Current opens above prev close
                c3.close < c2.open):    # Current closes below prev open
                return "[#ef4444]Pattern: Bearish Engulfing ▼[/]"
            
            # Hammer detection (simplified)
            if len(candles) >= 2:
                c = candles[-1]
                body = abs(c.close - c.open)
                total_range = c.high - c.low
                lower_shadow = min(c.open, c.close) - c.low
                
                if total_range > 0 and lower_shadow > body * 2 and body < total_range * 0.3:
                    if c.close > c.open:
                        return "[#10b981]Pattern: Hammer (Bullish) ▲[/]"
                    else:
                        return "[#f59e0b]Pattern: Hammer (Neutral) →[/]"
        
        # Consecutive pattern
        if consecutive_green >= 3:
            return f"[#10b981]Pattern: {consecutive_green} Green ▲[/]"
        elif consecutive_red >= 3:
            return f"[#ef4444]Pattern: {consecutive_red} Red ▼[/]"
        
        # Mixed pattern
        green_count = sum(1 for c in candles[-3:] if c.close >= c.open)
        if green_count >= 2:
            return "[#10b981]Pattern: Mostly Green →[/]"
        elif green_count <= 1:
            return "[#ef4444]Pattern: Mostly Red →[/]"
        
        return "[#f59e0b]Pattern: Mixed →[/]"
    
    def _calculate_range(self, candles: List[Candle], current_price: float) -> str:
        """Calculate intraday range analysis."""
        if not candles:
            return ""
        
        # Find session high and low
        session_high = max(c.high for c in candles)
        session_low = min(c.low for c in candles)
        session_range = session_high - session_low
        
        if session_range == 0:
            return f"Range: Flat @ ${current_price:,.0f}"
        
        # Calculate position in range (0.0 to 1.0)
        position_ratio = (current_price - session_low) / session_range
        position_pct = position_ratio * 100
        
        # Determine position label
        if position_pct > 80:
            range_color = "[#10b981]"
            position_label = "Near High"
        elif position_pct < 20:
            range_color = "[#ef4444]"
            position_label = "Near Low"
        else:
            range_color = "[#f59e0b]"
            position_label = "Mid-Range"
        
        # Calculate distances
        dist_to_high = ((session_high - current_price) / current_price) * 100
        dist_to_low = ((current_price - session_low) / current_price) * 100
        
        return f"{range_color}Range: {position_pct:.0f}% ({position_label}) | H: ${session_high:,.0f} (+{dist_to_high:.1f}%) | L: ${session_low:,.0f} (-{dist_to_low:.1f}%)[/]"
    
    def _calculate_momentum(self, candles: List[Candle]) -> tuple:
        """Calculate momentum oscillator (rate of change)."""
        if len(candles) < 3:
            return ("", "")
        
        # Calculate momentum over last 3-5 candles
        lookback = min(5, len(candles))
        recent_candles = candles[-lookback:]
        
        # Rate of change: (current_close - past_close) / past_close * 100
        past_close = recent_candles[0].close
        current_close = recent_candles[-1].close
        roc = ((current_close - past_close) / past_close) * 100 if past_close > 0 else 0
        
        # Calculate acceleration (change in momentum)
        if len(recent_candles) >= 4:
            mid_close = recent_candles[len(recent_candles) // 2].close
            first_half_roc = ((mid_close - past_close) / past_close) * 100 if past_close > 0 else 0
            second_half_roc = ((current_close - mid_close) / mid_close) * 100 if mid_close > 0 else 0
            acceleration = second_half_roc - first_half_roc
        else:
            acceleration = 0
        
        # Determine momentum strength and direction
        abs_roc = abs(roc)
        if roc > 0:
            if abs_roc > 2.0:
                momentum_color = "[#10b981]"
                strength = "Strong"
                indicator = "▲▲"
            elif abs_roc > 1.0:
                momentum_color = "[#10b981]"
                strength = "Moderate"
                indicator = "▲"
            else:
                momentum_color = "[#f59e0b]"
                strength = "Weak"
                indicator = "→"
        else:
            if abs_roc > 2.0:
                momentum_color = "[#ef4444]"
                strength = "Strong"
                indicator = "▼▼"
            elif abs_roc > 1.0:
                momentum_color = "[#ef4444]"
                strength = "Moderate"
                indicator = "▼"
            else:
                momentum_color = "[#f59e0b]"
                strength = "Weak"
                indicator = "→"
        
        # Add acceleration indicator
        if abs(acceleration) > 0.5:
            if acceleration > 0:
                accel_indicator = " (accelerating)"
            else:
                accel_indicator = " (decelerating)"
        else:
            accel_indicator = " (steady)"
        
        momentum_str = f"{momentum_color}Momentum: {roc:+.2f}% {indicator} ({strength}){accel_indicator}[/]"
        
        return (momentum_str, momentum_color)

    def hourly_analysis(self) -> tuple:
        """
        Generate 1-hour analysis combining:
        1. Volatility Analysis & Regime Detection
        2. Trend Strength & Quality Score
        3. Breakout Detection
        
        Returns: (volatility_str, trend_str, breakout_str, breakout_color)
        """
        if not self.recent_1h_candles or len(self.recent_1h_candles) < 5:
            return ("", "", "", "")
        
        if not self.indicator_snapshot or not self.current_price:
            return ("", "", "", "")
        
        snapshot = self.indicator_snapshot
        candles = self.recent_1h_candles
        current_price = self.current_price
        
        # 1. Volatility Analysis & Regime Detection
        volatility_str = self._analyze_volatility(candles, snapshot)
        
        # 2. Trend Strength & Quality Score
        trend_str = self._analyze_trend_strength(snapshot)
        
        # 3. Breakout Detection
        breakout_str, breakout_color = self._detect_breakout(candles, snapshot, current_price)
        
        return (volatility_str, trend_str, breakout_str, breakout_color)
    
    def _analyze_volatility(self, candles: List[Candle], snapshot: IndicatorSnapshot) -> str:
        """Analyze volatility and regime."""
        if not snapshot.atr or not snapshot.atr_pct:
            return "Volatility: --"
        
        current_atr_pct = snapshot.atr_pct
        
        # Calculate ATR trend (comparing recent ATR values)
        if len(candles) >= 10:
            # Estimate ATR from recent candles (simplified)
            recent_ranges = [c.high - c.low for c in candles[-5:]]
            current_range_avg = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
            older_ranges = [c.high - c.low for c in candles[-10:-5]] if len(candles) >= 10 else []
            older_range_avg = sum(older_ranges) / len(older_ranges) if older_ranges else current_range_avg
            
            if older_range_avg > 0:
                atr_ratio = current_range_avg / older_range_avg
            else:
                atr_ratio = 1.0
        else:
            atr_ratio = 1.0
        
        # Determine volatility state
        if atr_ratio > 1.3:
            vol_state = "Expanding"
            vol_indicator = "▲"
            vol_color = "[#10b981]"
        elif atr_ratio < 0.7:
            vol_state = "Contracting"
            vol_indicator = "▼"
            vol_color = "[#ef4444]"
        else:
            vol_state = "Stable"
            vol_indicator = "→"
            vol_color = "[#f59e0b]"
        
        # Regime detection
        regime = snapshot.regime if snapshot.regime else "UNKNOWN"
        regime_emoji = {
            "TRENDING": "📈",
            "RANGING": "↔️",
            "VOLATILE": "⚡",
            "UNKNOWN": "❓"
        }.get(regime, "❓")
        
        return f"{vol_color}Volatility: {vol_state} {vol_indicator} (ATR: {current_atr_pct:.2f}% | Regime: {regime} {regime_emoji})[/]"
    
    def _analyze_trend_strength(self, snapshot: IndicatorSnapshot) -> str:
        """Analyze trend strength and quality."""
        if not snapshot.adx:
            return "Trend: --"
        
        adx = snapshot.adx
        trend_dir = snapshot.trend_direction if snapshot.trend_direction else "SIDEWAYS"
        
        # Determine trend strength
        if adx >= 40:
            strength = "Very Strong"
            strength_pct = min(100, int((adx / 50) * 100))
            bar_filled = int(strength_pct / 10)
            bar_color = "[#10b981]"
        elif adx >= 25:
            strength = "Strong"
            strength_pct = int((adx / 40) * 100)
            bar_filled = int(strength_pct / 10)
            bar_color = "[#10b981]"
        elif adx >= 20:
            strength = "Moderate"
            strength_pct = int((adx / 25) * 100)
            bar_filled = int(strength_pct / 10)
            bar_color = "[#f59e0b]"
        else:
            strength = "Weak"
            strength_pct = int((adx / 20) * 100)
            bar_filled = max(1, int(strength_pct / 10))
            bar_color = "[#ef4444]"
        
        # Trend direction indicator
        if trend_dir == "UP":
            trend_indicator = "▲"
            trend_color = "[#10b981]"
        elif trend_dir == "DOWN":
            trend_indicator = "▼"
            trend_color = "[#ef4444]"
        else:
            trend_indicator = "→"
            trend_color = "[#f59e0b]"
        
        # Visual strength bar (10 chars)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        
        return f"{trend_color}Trend: {trend_dir} {trend_indicator} {bar_color}[{bar}][/] {strength_pct}% ({strength}) | ADX: {adx:.1f}[/]"
    
    def _detect_breakout(self, candles: List[Candle], snapshot: IndicatorSnapshot, current_price: float) -> tuple:
        """Detect breakouts above/below key levels."""
        if len(candles) < 10:
            return ("", "")
        
        # Identify key levels: recent swing highs/lows and Bollinger Bands
        recent_candles = candles[-20:] if len(candles) >= 20 else candles
        
        # Find swing highs and lows (local maxima/minima)
        swing_highs = []
        swing_lows = []
        
        for i in range(1, len(recent_candles) - 1):
            if (recent_candles[i].high > recent_candles[i-1].high and 
                recent_candles[i].high > recent_candles[i+1].high):
                swing_highs.append(recent_candles[i].high)
            if (recent_candles[i].low < recent_candles[i-1].low and 
                recent_candles[i].low < recent_candles[i+1].low):
                swing_lows.append(recent_candles[i].low)
        
        # Get nearest resistance (swing high) and support (swing low)
        nearest_resistance = min(swing_highs) if swing_highs else None
        nearest_support = max(swing_lows) if swing_lows else None
        
        # Also check Bollinger Bands
        bb_upper = snapshot.bb_upper
        bb_lower = snapshot.bb_lower
        
        # Determine breakout
        breakout_str = ""
        breakout_color = ""
        
        # Check for breakout above resistance
        if nearest_resistance and current_price > nearest_resistance:
            dist_pct = ((current_price - nearest_resistance) / nearest_resistance) * 100
            # Check if this is a fresh breakout (price was below recently)
            recent_below = any(c.close < nearest_resistance for c in candles[-3:-1])
            if recent_below and dist_pct < 2.0:  # Fresh breakout within 2%
                strength = "Strong" if dist_pct > 0.5 else "Moderate"
                breakout_str = f"Breakout: Above ${nearest_resistance:,.0f} ▲ (+{dist_pct:.2f}% | {strength})"
                breakout_color = "[#10b981]"
        
        # Check for breakout below support
        elif nearest_support and current_price < nearest_support:
            dist_pct = ((nearest_support - current_price) / nearest_support) * 100
            # Check if this is a fresh breakout (price was above recently)
            recent_above = any(c.close > nearest_support for c in candles[-3:-1])
            if recent_above and dist_pct < 2.0:  # Fresh breakout within 2%
                strength = "Strong" if dist_pct > 0.5 else "Moderate"
                breakout_str = f"Breakout: Below ${nearest_support:,.0f} ▼ (-{dist_pct:.2f}% | {strength})"
                breakout_color = "[#ef4444]"
        
        # Check Bollinger Band breakouts
        elif bb_upper and current_price > bb_upper:
            dist_pct = ((current_price - bb_upper) / bb_upper) * 100
            breakout_str = f"Breakout: Above BB Upper ${bb_upper:,.0f} ▲ (+{dist_pct:.2f}%)"
            breakout_color = "[#10b981]"
        elif bb_lower and current_price < bb_lower:
            dist_pct = ((bb_lower - current_price) / bb_lower) * 100
            breakout_str = f"Breakout: Below BB Lower ${bb_lower:,.0f} ▼ (-{dist_pct:.2f}%)"
            breakout_color = "[#ef4444]"
        
        # No breakout, show key levels
        if not breakout_str:
            level_info = []
            if nearest_resistance:
                dist_pct = ((nearest_resistance - current_price) / current_price) * 100
                level_info.append(f"Res: ${nearest_resistance:,.0f} (+{dist_pct:.1f}%)")
            if nearest_support:
                dist_pct = ((current_price - nearest_support) / current_price) * 100
                level_info.append(f"Sup: ${nearest_support:,.0f} (-{dist_pct:.1f}%)")
            
            if level_info:
                breakout_str = " | ".join(level_info)
                breakout_color = "[#f59e0b]"
            else:
                breakout_str = "No breakout detected"
                breakout_color = "[#f59e0b]"
        
        return (breakout_str, breakout_color)

    def swing_analysis(self) -> tuple:
        """
        Generate 4-hour swing analysis combining:
        1. Swing Structure Analysis
        2. Multi-Timeframe Alignment
        3. Trend Continuation Signal
        
        Returns: (structure_str, mtf_str, continuation_str, continuation_color)
        """
        if not self.recent_4h_candles or len(self.recent_4h_candles) < 5:
            return ("", "", "", "")
        
        if not self.indicator_snapshot_4h:
            return ("", "", "", "")
        
        candles = self.recent_4h_candles
        snapshot_4h = self.indicator_snapshot_4h
        snapshot_daily = self.indicator_snapshot_daily
        
        # 1. Swing Structure Analysis
        structure_str = self._analyze_swing_structure(candles)
        
        # 2. Multi-Timeframe Alignment
        mtf_str = self._analyze_mtf_alignment(snapshot_4h, snapshot_daily)
        
        # 3. Trend Continuation Signal
        continuation_str, continuation_color = self._analyze_trend_continuation(candles, snapshot_4h)
        
        return (structure_str, mtf_str, continuation_str, continuation_color)
    
    def _analyze_swing_structure(self, candles: List[Candle]) -> str:
        """Analyze swing structure (HH/HL for uptrend, LH/LL for downtrend)."""
        if len(candles) < 5:
            return ""
        
        # Find swing highs and lows (local maxima/minima)
        swing_highs = []
        swing_lows = []
        
        for i in range(1, len(candles) - 1):
            # Swing high: higher than neighbors
            if (candles[i].high > candles[i-1].high and 
                candles[i].high > candles[i+1].high):
                swing_highs.append((i, candles[i].high))
            # Swing low: lower than neighbors
            if (candles[i].low < candles[i-1].low and 
                candles[i].low < candles[i+1].low):
                swing_lows.append((i, candles[i].low))
        
        if not swing_highs or not swing_lows:
            return "Structure: Insufficient data"
        
        # Get last 2-3 swing highs and lows
        recent_highs = sorted(swing_highs[-3:], key=lambda x: x[1], reverse=True)
        recent_lows = sorted(swing_lows[-3:], key=lambda x: x[1])
        
        # Determine structure type
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for higher highs (HH)
            hh = recent_highs[-1][1] > recent_highs[-2][1] if len(recent_highs) >= 2 else False
            # Check for higher lows (HL)
            hl = recent_lows[-1][1] > recent_lows[-2][1] if len(recent_lows) >= 2 else False
            # Check for lower highs (LH)
            lh = recent_highs[-1][1] < recent_highs[-2][1] if len(recent_highs) >= 2 else False
            # Check for lower lows (LL)
            ll = recent_lows[-1][1] < recent_lows[-2][1] if len(recent_lows) >= 2 else False
            
            if hh and hl:
                structure_type = "HH HL"
                structure_label = "Uptrend Intact"
                structure_color = "[#10b981]"
                structure_indicator = "▲"
                last_swing = f"Last: HH @ ${recent_highs[-1][1]:,.0f}"
            elif lh and ll:
                structure_type = "LH LL"
                structure_label = "Downtrend Intact"
                structure_color = "[#ef4444]"
                structure_indicator = "▼"
                last_swing = f"Last: LL @ ${recent_lows[-1][1]:,.0f}"
            elif hh and not hl:
                structure_type = "HH"
                structure_label = "Uptrend Weakening"
                structure_color = "[#f59e0b]"
                structure_indicator = "→"
                last_swing = f"Last: HH @ ${recent_highs[-1][1]:,.0f}"
            elif ll and not lh:
                structure_type = "LL"
                structure_label = "Downtrend Weakening"
                structure_color = "[#f59e0b]"
                structure_indicator = "→"
                last_swing = f"Last: LL @ ${recent_lows[-1][1]:,.0f}"
            else:
                structure_type = "Mixed"
                structure_label = "Choppy/Ranging"
                structure_color = "[#f59e0b]"
                structure_indicator = "→"
                last_swing = ""
            
            if last_swing:
                return f"{structure_color}Structure: {structure_type} {structure_indicator} ({structure_label}) | {last_swing}[/]"
            else:
                return f"{structure_color}Structure: {structure_type} {structure_indicator} ({structure_label})[/]"
        else:
            return "Structure: Analyzing..."
    
    def _analyze_mtf_alignment(self, snapshot_4h: IndicatorSnapshot, snapshot_daily: Optional[IndicatorSnapshot]) -> str:
        """Analyze multi-timeframe alignment between 4h and daily."""
        if not snapshot_daily:
            return "MTF: Daily data unavailable"
        
        trend_4h = snapshot_4h.trend_direction if snapshot_4h.trend_direction else "SIDEWAYS"
        trend_daily = snapshot_daily.trend_direction if snapshot_daily.trend_direction else "SIDEWAYS"
        
        # Check alignment
        if trend_4h == trend_daily and trend_4h != "SIDEWAYS":
            # Aligned and trending
            if trend_4h == "UP":
                mtf_color = "[#10b981]"
                mtf_indicator = "▲▲"
                alignment = "Aligned"
                strength = "High"
            else:
                mtf_color = "[#ef4444]"
                mtf_indicator = "▼▼"
                alignment = "Aligned"
                strength = "High"
        elif trend_4h != "SIDEWAYS" and trend_daily != "SIDEWAYS" and trend_4h != trend_daily:
            # Diverging trends
            mtf_color = "[#f59e0b]"
            mtf_indicator = "↔️"
            alignment = "Diverging"
            strength = "Low"
        else:
            # At least one is sideways
            mtf_color = "[#f59e0b]"
            mtf_indicator = "→"
            alignment = "Mixed"
            strength = "Moderate"
        
        # Format trend indicators
        trend_4h_arrow = "↑" if trend_4h == "UP" else ("↓" if trend_4h == "DOWN" else "→")
        trend_daily_arrow = "↑" if trend_daily == "UP" else ("↓" if trend_daily == "DOWN" else "→")
        
        return f"{mtf_color}MTF: {alignment} {mtf_indicator} (4h{trend_4h_arrow} Daily{trend_daily_arrow}) | Strength: {strength}[/]"
    
    def _analyze_trend_continuation(self, candles: List[Candle], snapshot: IndicatorSnapshot) -> tuple:
        """Analyze if trend is continuing or weakening."""
        if len(candles) < 8:
            return ("", "")
        
        trend_dir = snapshot.trend_direction if snapshot.trend_direction else "SIDEWAYS"
        
        if trend_dir == "SIDEWAYS":
            return ("Trend: Sideways (No Clear Direction)", "[#f59e0b]")
        
        # Compare recent momentum to earlier momentum
        recent_candles = candles[-4:]  # Last 4 candles (16 hours)
        earlier_candles = candles[-8:-4]  # Previous 4 candles
        
        if not recent_candles or not earlier_candles:
            return ("", "")
        
        recent_start = earlier_candles[0].close
        recent_end = recent_candles[-1].close
        recent_momentum = ((recent_end - recent_start) / recent_start) * 100 if recent_start > 0 else 0
        
        earlier_start = candles[-8].close if len(candles) >= 8 else earlier_candles[0].close
        earlier_end = earlier_candles[-1].close
        earlier_momentum = ((earlier_end - earlier_start) / earlier_start) * 100 if earlier_start > 0 else 0
        
        # Determine continuation strength
        if trend_dir == "UP":
            # Uptrend: check if momentum is maintaining or accelerating
            if recent_momentum > 0 and recent_momentum >= earlier_momentum * 0.8:
                if recent_momentum > earlier_momentum * 1.2:
                    continuation_str = "Trend: Continuing ▲ (Momentum: Accelerating)"
                    continuation_color = "[#10b981]"
                else:
                    continuation_str = "Trend: Continuing ▲ (Momentum: Strong)"
                    continuation_color = "[#10b981]"
            elif recent_momentum > 0:
                continuation_str = "Trend: Weakening → (Momentum: Slowing)"
                continuation_color = "[#f59e0b]"
            else:
                continuation_str = "Trend: Reversing ▼ (Momentum: Negative)"
                continuation_color = "[#ef4444]"
        else:  # DOWN
            # Downtrend: check if momentum is maintaining or accelerating
            if recent_momentum < 0 and recent_momentum <= earlier_momentum * 0.8:
                if recent_momentum < earlier_momentum * 1.2:
                    continuation_str = "Trend: Continuing ▼ (Momentum: Accelerating)"
                    continuation_color = "[#ef4444]"
                else:
                    continuation_str = "Trend: Continuing ▼ (Momentum: Strong)"
                    continuation_color = "[#ef4444]"
            elif recent_momentum < 0:
                continuation_str = "Trend: Weakening → (Momentum: Slowing)"
                continuation_color = "[#f59e0b]"
            else:
                continuation_str = "Trend: Reversing ▲ (Momentum: Positive)"
                continuation_color = "[#10b981]"
        
        # Add ADX context if available
        if snapshot.adx:
            if snapshot.adx >= 25:
                continuation_str += f" | ADX: {snapshot.adx:.1f} (Strong)"
            else:
                continuation_str += f" | ADX: {snapshot.adx:.1f} (Weak)"
        
        return (continuation_str, continuation_color)

    def daily_analysis(self) -> tuple:
        """
        Generate daily analysis combining:
        1. Market Structure Analysis
        2. Major Support/Resistance Levels
        3. Trend Exhaustion Detection
        
        Returns: (structure_str, sr_str, exhaustion_str, exhaustion_color)
        """
        if not self.recent_daily_candles or len(self.recent_daily_candles) < 10:
            return ("", "", "", "")
        
        if not self.indicator_snapshot_daily or not self.current_price:
            return ("", "", "", "")
        
        candles = self.recent_daily_candles
        snapshot = self.indicator_snapshot_daily
        current_price = self.current_price
        
        # 1. Market Structure Analysis
        structure_str = self._analyze_market_structure(candles)
        
        # 2. Major Support/Resistance Levels
        sr_str = self._analyze_major_sr(candles, current_price)
        
        # 3. Trend Exhaustion Detection
        exhaustion_str, exhaustion_color = self._detect_trend_exhaustion(candles, snapshot, current_price)
        
        return (structure_str, sr_str, exhaustion_str, exhaustion_color)
    
    def _analyze_market_structure(self, candles: List[Candle]) -> str:
        """Analyze market structure (bull/bear phases)."""
        if len(candles) < 20:
            return "Market: Insufficient data"
        
        # Find major swing points (higher highs/lower lows)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(candles) - 2):
            # Swing high: higher than 2 neighbors on each side
            if (candles[i].high > candles[i-2].high and 
                candles[i].high > candles[i-1].high and
                candles[i].high > candles[i+1].high and
                candles[i].high > candles[i+2].high):
                swing_highs.append((i, candles[i].high, candles[i].timestamp))
            # Swing low: lower than 2 neighbors on each side
            if (candles[i].low < candles[i-2].low and 
                candles[i].low < candles[i-1].low and
                candles[i].low < candles[i+1].low and
                candles[i].low < candles[i+2].low):
                swing_lows.append((i, candles[i].low, candles[i].timestamp))
        
        if not swing_highs or not swing_lows:
            return "Market: Analyzing structure..."
        
        # Get recent swing points (last 5-6)
        recent_highs = swing_highs[-6:] if len(swing_highs) >= 6 else swing_highs
        recent_lows = swing_lows[-6:] if len(swing_lows) >= 6 else swing_lows
        
        # Determine market phase
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for higher highs and higher lows (bull market)
            hh = recent_highs[-1][1] > recent_highs[-2][1] if len(recent_highs) >= 2 else False
            hl = recent_lows[-1][1] > recent_lows[-2][1] if len(recent_lows) >= 2 else False
            
            # Check for lower highs and lower lows (bear market)
            lh = recent_highs[-1][1] < recent_highs[-2][1] if len(recent_highs) >= 2 else False
            ll = recent_lows[-1][1] < recent_lows[-2][1] if len(recent_lows) >= 2 else False
            
            # Find the lowest low in recent swings (potential market start point)
            lowest_low = min(recent_lows, key=lambda x: x[1])
            highest_high = max(recent_highs, key=lambda x: x[1])
            
            # Calculate duration (days since structure started)
            from datetime import datetime
            current_time = candles[-1].timestamp
            structure_start_time = min(lowest_low[2], highest_high[2])
            days_duration = int((current_time - structure_start_time) / 86400)
            
            if hh and hl:
                market_phase = "Bull Phase"
                phase_color = "[#10b981]"
                phase_indicator = "▲"
                structure_price = lowest_low[1]
                structure_label = f"Since: ${structure_price:,.0f}"
            elif lh and ll:
                market_phase = "Bear Phase"
                phase_color = "[#ef4444]"
                phase_indicator = "▼"
                structure_price = highest_high[1]
                structure_label = f"Since: ${structure_price:,.0f}"
            elif hh and not hl:
                market_phase = "Bull Weakening"
                phase_color = "[#f59e0b]"
                phase_indicator = "→"
                structure_price = lowest_low[1]
                structure_label = f"Since: ${structure_price:,.0f}"
            elif ll and not lh:
                market_phase = "Bear Weakening"
                phase_color = "[#f59e0b]"
                phase_indicator = "→"
                structure_price = highest_high[1]
                structure_label = f"Since: ${structure_price:,.0f}"
            else:
                market_phase = "Accumulation/Distribution"
                phase_color = "[#f59e0b]"
                phase_indicator = "↔️"
                structure_price = (lowest_low[1] + highest_high[1]) / 2
                structure_label = f"Range: ${lowest_low[1]:,.0f}-${highest_high[1]:,.0f}"
            
            return f"{phase_color}Market: {market_phase} {phase_indicator} | {structure_label} | Duration: {days_duration} days[/]"
        else:
            return "Market: Analyzing structure..."
    
    def _analyze_major_sr(self, candles: List[Candle], current_price: float) -> str:
        """Analyze major support/resistance levels."""
        if len(candles) < 20:
            return ""
        
        # Find all-time high and major support levels
        all_time_high = max(c.high for c in candles)
        all_time_low = min(c.low for c in candles)
        
        # Find recent significant swing highs and lows (last 30-60 days)
        analysis_candles = candles[-60:] if len(candles) >= 60 else candles
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(analysis_candles) - 2):
            if (analysis_candles[i].high > analysis_candles[i-2].high and 
                analysis_candles[i].high > analysis_candles[i-1].high and
                analysis_candles[i].high > analysis_candles[i+1].high and
                analysis_candles[i].high > analysis_candles[i+2].high):
                swing_highs.append(analysis_candles[i].high)
            if (analysis_candles[i].low < analysis_candles[i-2].low and 
                analysis_candles[i].low < analysis_candles[i-1].low and
                analysis_candles[i].low < analysis_candles[i+1].low and
                analysis_candles[i].low < analysis_candles[i+2].low):
                swing_lows.append(analysis_candles[i].low)
        
        # Find nearest major support (significant swing low below current price)
        supports = [s for s in swing_lows if s < current_price]
        resistances = [r for r in swing_highs if r > current_price]
        
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        # Calculate distances
        dist_to_ath = ((all_time_high - current_price) / current_price) * 100
        dist_to_atl = ((current_price - all_time_low) / current_price) * 100
        
        # Build support/resistance string
        sr_parts = []
        
        # ATH
        if all_time_high > current_price:
            sr_parts.append(f"ATH: ${all_time_high:,.0f} (+{dist_to_ath:.1f}%)")
        
        # Nearest resistance
        if nearest_resistance:
            dist_to_res = ((nearest_resistance - current_price) / current_price) * 100
            sr_parts.append(f"Res: ${nearest_resistance:,.0f} (+{dist_to_res:.1f}%)")
        
        # Nearest support
        if nearest_support:
            dist_to_sup = ((current_price - nearest_support) / current_price) * 100
            sr_parts.append(f"Sup: ${nearest_support:,.0f} (-{dist_to_sup:.1f}%)")
        
        # ATL
        if all_time_low < current_price:
            sr_parts.append(f"ATL: ${all_time_low:,.0f} (-{dist_to_atl:.1f}%)")
        
        if sr_parts:
            return f"Major S/R: {' | '.join(sr_parts)}"
        else:
            return "Major S/R: Analyzing..."
    
    def _detect_trend_exhaustion(self, candles: List[Candle], snapshot: IndicatorSnapshot, current_price: float) -> tuple:
        """Detect potential trend exhaustion signals."""
        if len(candles) < 10:
            return ("", "")
        
        exhaustion_signals = []
        warning_level = "none"
        
        # 1. RSI extremes
        if snapshot.rsi:
            if snapshot.rsi >= 80:
                exhaustion_signals.append(f"RSI: {snapshot.rsi:.1f} (Extreme Overbought)")
                warning_level = "high"
            elif snapshot.rsi >= 70:
                exhaustion_signals.append(f"RSI: {snapshot.rsi:.1f} (Overbought)")
                if warning_level == "none":
                    warning_level = "moderate"
            elif snapshot.rsi <= 20:
                exhaustion_signals.append(f"RSI: {snapshot.rsi:.1f} (Extreme Oversold)")
                warning_level = "high"
            elif snapshot.rsi <= 30:
                exhaustion_signals.append(f"RSI: {snapshot.rsi:.1f} (Oversold)")
                if warning_level == "none":
                    warning_level = "moderate"
        
        # 2. Volume analysis (declining volume on rallies)
        if len(candles) >= 10:
            recent_candles = candles[-5:]
            earlier_candles = candles[-10:-5]
            
            recent_avg_volume = sum(c.volume for c in recent_candles) / len(recent_candles)
            earlier_avg_volume = sum(c.volume for c in earlier_candles) / len(earlier_candles)
            
            # Check if price is rising but volume is declining
            recent_avg_price = sum(c.close for c in recent_candles) / len(recent_candles)
            earlier_avg_price = sum(c.close for c in earlier_candles) / len(earlier_candles)
            price_rising = recent_avg_price > earlier_avg_price
            
            if price_rising and recent_avg_volume < earlier_avg_volume * 0.7:
                exhaustion_signals.append("Volume: Declining on Rally")
                if warning_level == "none":
                    warning_level = "moderate"
            elif not price_rising and recent_avg_volume < earlier_avg_volume * 0.7:
                exhaustion_signals.append("Volume: Declining on Decline")
                if warning_level == "none":
                    warning_level = "moderate"
        
        # 3. Momentum divergence (price making new highs but momentum weakening)
        if len(candles) >= 15:
            recent_high = max(c.high for c in candles[-5:])
            earlier_high = max(c.high for c in candles[-15:-5])
            
            recent_momentum = ((candles[-1].close - candles[-5].close) / candles[-5].close) * 100
            earlier_momentum = ((candles[-5].close - candles[-10].close) / candles[-10].close) * 100
            
            if recent_high > earlier_high and recent_momentum < earlier_momentum * 0.5:
                exhaustion_signals.append("Momentum: Diverging (Price↑ Momentum↓)")
                if warning_level != "high":
                    warning_level = "moderate"
        
        # 4. ADX weakening (trend losing strength)
        if snapshot.adx:
            if snapshot.adx < 20 and snapshot.trend_direction != "SIDEWAYS":
                exhaustion_signals.append(f"ADX: {snapshot.adx:.1f} (Weak Trend)")
                if warning_level == "none":
                    warning_level = "low"
        
        # Format exhaustion string
        if exhaustion_signals:
            if warning_level == "high":
                exhaustion_color = "[#ef4444]"
                exhaustion_prefix = "⚠️ Exhaustion: High Risk"
            elif warning_level == "moderate":
                exhaustion_color = "[#f59e0b]"
                exhaustion_prefix = "⚠️ Exhaustion: Moderate Risk"
            else:
                exhaustion_color = "[#f59e0b]"
                exhaustion_prefix = "Exhaustion: Watch"
            
            exhaustion_str = f"{exhaustion_prefix} | {' | '.join(exhaustion_signals)}"
        else:
            exhaustion_color = "[#10b981]"
            exhaustion_str = "No Exhaustion Signals | Trend Healthy"
        
        return (exhaustion_str, exhaustion_color)

    def _get_score_class(self) -> str:
        """Get CSS class based on score value."""
        score = self.signal.score
        if score <= 20:
            return "score-0-20"
        elif score <= 40:
            return "score-21-40"
        elif score <= 50:
            return "score-41-50"
        elif score <= 60:
            return "score-51-60"
        elif score <= 70:
            return "score-61-70"
        elif score <= 80:
            return "score-71-80"
        elif score <= 90:
            return "score-81-90"
        else:
            return "score-91-100"

    def on_mount(self) -> None:
        """Update the widget content when mounted."""
        # Apply score-based border color class
        score_class = self._get_score_class()
        self.add_class(score_class)
        content = self.render()
        self.update(content)

    def render(self) -> str:
        # Format the title line with timeframe label only
        if self.timeframe:
            title_line = f"[{self.timeframe}]"
        else:
            title_line = ""  # No label if timeframe not set

        lines = []
        if title_line:  # Only add title line if it exists
            lines.append(title_line)

        lines.extend([
            f"Score: {self.signal.score}",
            f"{self.signal.description}",
            ""
        ])

        # Order flow visualization (only for 10s timeframe - HFT)
        if self.id == "ltf-10s":
            pressure_bar, histogram, delta_str, delta_color = self.order_flow()
            if pressure_bar:
                # Show buy/sell pressure bar
                lines.append(f"Flow: {pressure_bar}")
                # Show trade flow histogram
                if histogram:
                    lines.append(f"Trades: {histogram}")
                # Show delta with color
                lines.append(f"{delta_color}{delta_str}[/]")
            else:
                lines.append("")
        # Candle summary visualization (for 1m timeframe - Scalping)
        elif self.id == "ltf-1m":
            candle_bar, volume_str, position_str, position_color = self.candle_summary()
            if candle_bar:
                # Show OHLC candle bar
                lines.append(f"Candle: {candle_bar}")
                # Show volume comparison
                lines.append(volume_str)
                # Show price position
                lines.append(f"{position_color}{position_str}[/]")
            else:
                lines.append("")
        # Intraday analysis visualization (for 15m timeframe - Intraday Trading)
        elif self.id == "ltf-15m":
            pattern_str, range_str, momentum_str, momentum_color = self.intraday_analysis()
            if pattern_str:
                # Show pattern recognition
                lines.append(pattern_str)
                # Show intraday range
                if range_str:
                    lines.append(range_str)
                # Show momentum oscillator
                if momentum_str:
                    lines.append(momentum_str)
            else:
                lines.append("")
        # Hourly analysis visualization (for 1h timeframe - Swing Trading)
        elif self.id == "htf-1h":
            volatility_str, trend_str, breakout_str, breakout_color = self.hourly_analysis()
            if volatility_str:
                # Show volatility analysis
                lines.append(volatility_str)
                # Show trend strength
                if trend_str:
                    lines.append(trend_str)
                # Show breakout detection
                if breakout_str:
                    lines.append(f"{breakout_color}{breakout_str}[/]")
            else:
                lines.append("")
        # Swing analysis visualization (for 4h timeframe - Position Trading)
        elif self.id == "htf-4h":
            structure_str, mtf_str, continuation_str, continuation_color = self.swing_analysis()
            if structure_str:
                # Show swing structure
                lines.append(structure_str)
                # Show multi-timeframe alignment
                if mtf_str:
                    lines.append(mtf_str)
                # Show trend continuation
                if continuation_str:
                    lines.append(f"{continuation_color}{continuation_str}[/]")
            else:
                lines.append("")
        # Daily analysis visualization (for daily timeframe - Long-term Trading)
        elif self.id == "htf-daily":
            structure_str, sr_str, exhaustion_str, exhaustion_color = self.daily_analysis()
            if structure_str:
                # Show market structure
                lines.append(structure_str)
                # Show major support/resistance
                if sr_str:
                    lines.append(sr_str)
                # Show trend exhaustion
                if exhaustion_str:
                    lines.append(f"{exhaustion_color}{exhaustion_str}[/]")
            else:
                lines.append("")
        else:
            # For other timeframes, skip visualization
            lines.append("")

        lines.append("")

        for k, v in self.signal.metrics.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


# Main Dashboard App
class MarketDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        height: 0;
        display: none;
    }

    Footer {
        height: 0;
        display: none;
    }

    CombinedHeaderWidget {
        height: 1;
        margin: 0;
        padding: 0;
        text-align: center;
        content-align: center middle;
    }

    Grid {
        grid-size: 3 2;
        grid-gutter: 0;
        grid-rows: 1fr 1fr;
        margin: 0;
        padding: 0;
    }

    SignalCard {
        margin: 0;
    }

    SignalCard {
        border: solid $secondary;
        padding: 1 2;
        background: $surface;
    }
    SignalCard:hover {
        border: solid $accent;
    }

    /* Score-based border colors: red (low) to green (high) */
    .score-0-20 { border: solid #ef4444; }      /* Bright red */
    .score-21-40 { border: solid #f87171; }    /* Red */
    .score-41-50 { border: solid #fbbf24; }    /* Yellow */
    .score-51-60 { border: solid #f59e0b; }     /* Orange */
    .score-61-70 { border: solid #84cc16; }     /* Light green */
    .score-71-80 { border: solid #22c55e; }     /* Green */
    .score-81-90 { border: solid #10b981; }    /* Bright green */
    .score-91-100 { border: solid #059669; }   /* Very bright green */

    /* Flashing classes for aligned signals */
    .flash-red { border: solid #ef4444; }
    .flash-green { border: solid #10b981; }
    """

    header_data = reactive(
        MarketHeader("BTC-USD", 0.0, 0.0, 0.0)
    )

    _flash_state: bool = False
    _flash_timer: Optional[Timer] = None

    def __init__(self):
        super().__init__()
        logger.info("MarketDashboard.__init__() called")
        self.core: Optional[PulseCore] = None
        self._indicator_snapshots: Dict[Timeframe, IndicatorSnapshot] = {}
        self._current_ticker: Optional[Ticker] = None
        logger.info("MarketDashboard.__init__() completed")

    def compose(self) -> ComposeResult:
        try:
            logger.info("compose() called")
            yield CombinedHeaderWidget(id="combined_header")
            logger.info("Header widget yielded")

            with Grid():
                logger.info("Grid context entered")
                yield SignalCard(
                    "Loading", "10s HFT",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-10s"
                )
                yield SignalCard(
                    "Loading", "1m Scalp",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-1m"
                )
                yield SignalCard(
                    "Loading", "15m Intraday",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="ltf-15m"
                )
                yield SignalCard(
                    "Loading", "1h",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-1h"
                )
                yield SignalCard(
                    "Loading", "4h",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-4h"
                )
                yield SignalCard(
                    "Loading", "Daily",
                    Signal("Loading", 50, "Initializing...", {}),
                    id="htf-daily"
                )
            logger.info("All widgets composed successfully")
        except Exception as e:
            logger.error(f"Error in compose(): {e}", exc_info=True)
            raise

    async def on_mount(self) -> None:
        """Initialize PulseCore and register callbacks."""
        try:
            logger.info("on_mount() started")
            # Configure for exactly the 6 required timeframes
            config = PulseConfig(
                products=["BTC-USD"],
                enabled_timeframes={
                    Timeframe.TEN_SEC,
                    Timeframe.ONE_MIN,
                    Timeframe.FIFTEEN_MIN,
                    Timeframe.ONE_HOUR,
                    Timeframe.FOUR_HOUR,
                    Timeframe.ONE_DAY,
                },
            )
            logger.info("Config created")

            self.core = PulseCore(config=config)
            logger.info("PulseCore instantiated")

            # Register callbacks
            self.core.register_callback('ticker_update', self._on_ticker_update)
            self.core.register_callback('indicator_update', self._on_indicator_update)
            self.core.register_callback('trade_update', self._on_trade_update)
            self.core.register_callback('connection_status', self._on_connection_status)
            self.core.register_callback('priming_progress', self._on_priming_progress)
            self.core.register_callback('priming_complete', self._on_priming_complete)
            self.core.register_callback('alert', self._on_alert)
            logger.info("Callbacks registered")

            # Start PulseCore in background
            self.run_worker(self._start_core(), exclusive=True)
            logger.info("Worker started")

            # Start flash timer for alignment effects
            self._flash_timer = self.set_interval(0.5, self._flash_toggle)
            logger.info("on_mount() completed successfully")
        except Exception as e:
            logger.error(f"Error in on_mount: {e}", exc_info=True)
            raise

    async def _start_core(self) -> None:
        """Start PulseCore (runs in worker thread)."""
        try:
            logger.info("_start_core() called - Starting PulseCore...")
            if not self.core:
                logger.error("PulseCore is None! Cannot start.")
                self.notify("Error: PulseCore not initialized", severity="error")
                return
            
            logger.info(f"PulseCore instance exists: {type(self.core)}")
            await self.core.start()
            logger.info("PulseCore started successfully - data should be loading now")
        except Exception as e:
            logger.error(f"Error starting Pulse: {e}", exc_info=True)
            self.notify(f"Error starting Pulse: {e}", severity="error")

    async def _on_ticker_update(self, ticker: Ticker) -> None:
        """Handle ticker updates from PulseCore."""
        self._current_ticker = ticker

        # Update header widget
        try:
            header = self.query_one("#combined_header", CombinedHeaderWidget)
            header.data = MarketHeader(
                symbol=ticker.product_id,
                price=ticker.price,
                change_pct=ticker.price_change_pct_24h,
                volume_24h=ticker.volume_24h,
            )

            # Recalculate ST/LT signals based on current card scores
            st_type, st_label = self._calculate_st_signal()
            lt_type, lt_label = self._calculate_lt_signal()
            header.short_term_type = st_type
            header.short_term_signal = st_label
            header.long_term_type = lt_type
            header.long_term_signal = lt_label
            header.refresh()
            
            # Update 1m and 15m cards with current price
            try:
                card_1m = self.query_one("#ltf-1m", SignalCard)
                card_1m.current_price = ticker.price
                card_1m.refresh()
            except Exception:
                pass  # Card might not exist yet
            
            try:
                card_15m = self.query_one("#ltf-15m", SignalCard)
                card_15m.current_price = ticker.price
                card_15m.refresh()
            except Exception:
                pass  # Card might not exist yet
            
            try:
                card_1h = self.query_one("#htf-1h", SignalCard)
                card_1h.current_price = ticker.price
                # Also update indicator snapshot if available
                if self.core:
                    snapshot = self.core.get_indicators(ticker.product_id, Timeframe.ONE_HOUR)
                    if snapshot:
                        card_1h.indicator_snapshot = snapshot
                card_1h.refresh()
            except Exception:
                pass  # Card might not exist yet
            
            try:
                card_4h = self.query_one("#htf-4h", SignalCard)
                card_4h.current_price = ticker.price
                # Also update indicator snapshots if available
                if self.core:
                    snapshot_4h = self.core.get_indicators(ticker.product_id, Timeframe.FOUR_HOUR)
                    snapshot_daily = self.core.get_indicators(ticker.product_id, Timeframe.ONE_DAY)
                    if snapshot_4h:
                        card_4h.indicator_snapshot_4h = snapshot_4h
                    if snapshot_daily:
                        card_4h.indicator_snapshot_daily = snapshot_daily
                card_4h.refresh()
            except Exception:
                pass  # Card might not exist yet
            
            try:
                card_daily = self.query_one("#htf-daily", SignalCard)
                card_daily.current_price = ticker.price
                # Also update indicator snapshot if available
                if self.core:
                    snapshot_daily = self.core.get_indicators(ticker.product_id, Timeframe.ONE_DAY)
                    if snapshot_daily:
                        card_daily.indicator_snapshot_daily = snapshot_daily
                card_daily.refresh()
            except Exception:
                pass  # Card might not exist yet
        except Exception as e:
            logger.debug(f"Error updating ticker: {e}")

    async def _on_indicator_update(
        self,
        product_id: str,
        timeframe: Timeframe,
        snapshot: IndicatorSnapshot
    ) -> None:
        """Handle indicator updates from PulseCore."""
        logger.info(
            f"_on_indicator_update called: {product_id} {timeframe.value} "
            f"RSI={snapshot.rsi}, ADX={snapshot.adx}, MACD={snapshot.macd_histogram}, VWAP={snapshot.vwap}"
        )

        # Store snapshot
        self._indicator_snapshots[timeframe] = snapshot

        # Get corresponding card ID
        card_id = TIMEFRAME_CARD_MAP.get(timeframe)
        if not card_id:
            logger.warning(f"No card_id mapping for timeframe {timeframe.value}")
            return

        # Compute signal score from indicators
        score, label, description = compute_signal_score(snapshot)
        logger.info(f"  → Computed score for {timeframe.value}: {score} ({label}) - {description}")

        # Build metrics dict for display
        metrics = self._build_metrics_dict(snapshot)

        # Create new Signal
        new_signal = Signal(
            label=label,
            score=score,
            description=description,
            metrics=metrics,
        )

        # Update the SignalCard
        try:
            card = self.query_one(f"#{card_id}", SignalCard)
            logger.info(f"  → Found card {card_id}, updating...")

            # Update signal data
            card.signal = new_signal
            logger.info(f"  → Signal updated: score={new_signal.score}")

            # Update order flow with recent trades (only for 10s timeframe)
            if card_id == "ltf-10s":
                recent_trades = self._get_recent_trades(product_id)
                if recent_trades:
                    card.trades = recent_trades
                    logger.info(f"  → Order flow updated with {len(recent_trades)} trades")
            
            # Update candle summary (only for 1m timeframe)
            elif card_id == "ltf-1m":
                # Get recent candles (last 10) - most recent is current
                recent_candles = self.core.get_candles(product_id, Timeframe.ONE_MIN, limit=10)
                if recent_candles:
                    card.recent_candles = recent_candles
                    card.current_candle = recent_candles[-1]  # Most recent is current
                    # Update current price from ticker
                    if self._current_ticker:
                        card.current_price = self._current_ticker.price
                    logger.info(f"  → Candle summary updated for 1m")
            
            # Update intraday analysis (only for 15m timeframe)
            elif card_id == "ltf-15m":
                # Get recent 15m candles (last 20 for range, last 6 for patterns)
                recent_15m_candles = self.core.get_candles(product_id, Timeframe.FIFTEEN_MIN, limit=20)
                if recent_15m_candles:
                    card.recent_15m_candles = recent_15m_candles
                    # Update current price from ticker
                    if self._current_ticker:
                        card.current_price = self._current_ticker.price
                    logger.info(f"  → Intraday analysis updated for 15m")
            
            # Update hourly analysis (only for 1h timeframe)
            elif card_id == "htf-1h":
                # Get recent 1h candles (last 30 for swing analysis)
                recent_1h_candles = self.core.get_candles(product_id, Timeframe.ONE_HOUR, limit=30)
                if recent_1h_candles:
                    card.recent_1h_candles = recent_1h_candles
                    # Get indicator snapshot for 1h
                    snapshot = self.core.get_indicators(product_id, Timeframe.ONE_HOUR)
                    if snapshot:
                        card.indicator_snapshot = snapshot
                    # Update current price from ticker
                    if self._current_ticker:
                        card.current_price = self._current_ticker.price
                    logger.info(f"  → Hourly analysis updated for 1h")
            
            # Update swing analysis (only for 4h timeframe)
            elif card_id == "htf-4h":
                # Get recent 4h candles (last 30 for swing structure analysis)
                recent_4h_candles = self.core.get_candles(product_id, Timeframe.FOUR_HOUR, limit=30)
                if recent_4h_candles:
                    card.recent_4h_candles = recent_4h_candles
                    # Get indicator snapshots for 4h and daily (for MTF alignment)
                    snapshot_4h = self.core.get_indicators(product_id, Timeframe.FOUR_HOUR)
                    snapshot_daily = self.core.get_indicators(product_id, Timeframe.ONE_DAY)
                    if snapshot_4h:
                        card.indicator_snapshot_4h = snapshot_4h
                    if snapshot_daily:
                        card.indicator_snapshot_daily = snapshot_daily
                    # Update current price from ticker
                    if self._current_ticker:
                        card.current_price = self._current_ticker.price
                    logger.info(f"  → Swing analysis updated for 4h")
            
            # Update daily analysis (only for daily timeframe)
            elif card_id == "htf-daily":
                # Get recent daily candles (last 60 for market structure analysis)
                recent_daily_candles = self.core.get_candles(product_id, Timeframe.ONE_DAY, limit=60)
                if recent_daily_candles:
                    card.recent_daily_candles = recent_daily_candles
                    # Get indicator snapshot for daily
                    snapshot_daily = self.core.get_indicators(product_id, Timeframe.ONE_DAY)
                    if snapshot_daily:
                        card.indicator_snapshot_daily = snapshot_daily
                    # Update current price from ticker
                    if self._current_ticker:
                        card.current_price = self._current_ticker.price
                    logger.info(f"  → Daily analysis updated")

            # Refresh the card
            card.refresh()
            logger.info(f"  → Card {card_id} refreshed successfully")

            # Recalculate header signals after any card update
            self._update_header_signals()

        except Exception as e:
            logger.error(f"ERROR updating card {card_id}: {e}", exc_info=True)

    async def _on_trade_update(self, trade: Trade) -> None:
        """Handle trade updates - refresh order flow visualization on 10s card only."""
        try:
            # Order flow is only relevant for 10s timeframe (HFT)
            product_id = trade.product_id
            recent_trades = self._get_recent_trades(product_id, limit=15)
            
            if recent_trades:
                # Only update the 10s card with latest trade data
                try:
                    card = self.query_one("#ltf-10s", SignalCard)
                    card.trades = recent_trades
                    card.refresh()
                except Exception:
                    pass  # Card might not exist yet
        except Exception as e:
            logger.debug(f"Error updating trade flow: {e}")

    async def _on_connection_status(self, connected: bool, message: str) -> None:
        """Handle connection status change."""
        if connected:
            logger.info(f"Connected: {message}")
        else:
            logger.warning(f"Disconnected: {message}")

    async def _on_priming_progress(
        self,
        product_id: str,
        timeframe: Timeframe,
        progress: float
    ) -> None:
        """Handle priming progress."""
        logger.info(f"Priming {product_id} {timeframe.value}: {progress*100:.0f}%")

        # Update the corresponding card to show priming status
        card_id = TIMEFRAME_CARD_MAP.get(timeframe)
        if card_id:
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Update description to show priming
                new_signal = Signal(
                    label="Priming",
                    score=50,
                    description=f"Loading historical data... {progress*100:.0f}%",
                    metrics={}
                )
                card.signal = new_signal
                card.refresh()
            except Exception as e:
                logger.debug(f"Error updating priming progress for {card_id}: {e}")

    async def _on_priming_complete(self, product_id: str) -> None:
        """Handle priming completion."""
        logger.info(f"Priming complete for {product_id}")

        # Update all cards to show they're waiting for live data
        for tf, card_id in TIMEFRAME_CARD_MAP.items():
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Check if we have indicator data for this timeframe
                if tf not in self._indicator_snapshots or self._indicator_snapshots[tf] is None:
                    # No historical data - show waiting message
                    new_signal = Signal(
                        label="Live Mode",
                        score=50,
                        description="Waiting for live market data...",
                        metrics={"Status": "WebSocket Connected"}
                    )
                    card.signal = new_signal
                    card.refresh()
            except Exception as e:
                logger.debug(f"Error updating card {card_id} after priming: {e}")

    async def _on_alert(self, alert) -> None:
        """Handle alerts from indicator engine."""
        logger.info(f"Alert: {alert.message}")

    def _build_metrics_dict(self, snapshot: IndicatorSnapshot) -> Dict[str, str]:
        """Build display metrics from indicator snapshot."""
        metrics = {}

        # For 10s timeframe, show warmup status if indicators aren't ready
        if snapshot.timeframe == Timeframe.TEN_SEC:
            indicators_ready = sum([
                snapshot.rsi is not None,
                snapshot.adx is not None,
                snapshot.macd_histogram is not None,
            ])
            total_indicators = 3

            if indicators_ready == 0 and snapshot.vwap is not None:
                # Only VWAP is ready - show warmup message
                metrics["Status"] = f"Warming up indicators... ({indicators_ready}/{total_indicators})"
            elif indicators_ready < total_indicators:
                metrics["Status"] = f"Indicators warming: {indicators_ready}/{total_indicators}"

        if snapshot.rsi is not None:
            rsi_label = "OB" if snapshot.rsi > 70 else ("OS" if snapshot.rsi < 30 else "")
            metrics["RSI(14)"] = f"{snapshot.rsi:.1f} {rsi_label}".strip()

        if snapshot.adx is not None:
            metrics["ADX"] = f"{snapshot.adx:.1f}"

        if snapshot.trend_direction:
            arrow = "↑" if snapshot.trend_direction == "UP" else ("↓" if snapshot.trend_direction == "DOWN" else "→")
            metrics["Trend"] = f"{snapshot.trend_direction} {arrow}"

        # Only show MACD histogram for higher timeframes (1h, 4h, daily)
        if snapshot.macd_histogram is not None and snapshot.timeframe not in [Timeframe.TEN_SEC, Timeframe.ONE_MIN, Timeframe.FIFTEEN_MIN]:
            sign = "+" if snapshot.macd_histogram > 0 else ""
            metrics["MACD Hist"] = f"{sign}{snapshot.macd_histogram:.2f}"

        # Only show VWAP deviation for lower timeframes (10s, 1m, 15m)
        if snapshot.vwap_deviation is not None and snapshot.timeframe in [Timeframe.TEN_SEC, Timeframe.ONE_MIN, Timeframe.FIFTEEN_MIN]:
            sign = "+" if snapshot.vwap_deviation > 0 else ""
            metrics["VWAP Dev"] = f"{sign}{snapshot.vwap_deviation:.2f}%"

        if snapshot.regime:
            metrics["Regime"] = snapshot.regime

        return metrics

    def _get_recent_trades(self, product_id: str, limit: int = 15) -> List[Trade]:
        """Get recent trades for order flow visualization."""
        if not self.core:
            return []

        trades = self.core.get_recent_trades(product_id, limit=limit)
        return trades

    def _is_red(self, score: int) -> bool:
        """Check if score is in red range (low scores)."""
        return score <= 40

    def _is_green(self, score: int) -> bool:
        """Check if score is in green range (high scores)."""
        return score >= 70

    def _update_header_signals(self) -> None:
        """Update header ST/LT signals based on card scores."""
        try:
            header = self.query_one("#combined_header", CombinedHeaderWidget)
            st_type, st_label = self._calculate_st_signal()
            lt_type, lt_label = self._calculate_lt_signal()
            header.short_term_type = st_type
            header.short_term_signal = st_label
            header.long_term_type = lt_type
            header.long_term_signal = lt_label
            header.refresh()
        except Exception as e:
            logger.debug(f"Error updating header signals: {e}")

    def _calculate_st_signal(self) -> tuple:
        """Calculate short-term signal type and label. Returns (type, label)."""
        try:
            ltf_10s = self.query_one("#ltf-10s", SignalCard)
            ltf_1m = self.query_one("#ltf-1m", SignalCard)
            ltf_15m = self.query_one("#ltf-15m", SignalCard)

            scores = [
                ltf_10s.signal.score,
                ltf_1m.signal.score,
                ltf_15m.signal.score
            ]

            # Count strong bullish (>=70), moderate bullish (60-69), and bearish (<=40)
            strong_bullish = sum(1 for s in scores if s >= 70)
            moderate_bullish = sum(1 for s in scores if 60 <= s < 70)
            bearish = sum(1 for s in scores if s <= 40)
            
            # Calculate average score for better signal quality
            avg_score = sum(scores) / len(scores) if scores else 50

            # Bullish if: 2+ strong bullish, OR (1+ strong + 1+ moderate), OR avg >= 60
            if strong_bullish >= 2 or (strong_bullish >= 1 and moderate_bullish >= 1) or avg_score >= 60:
                return ("bullish", "BULLISH")
            # Bearish if: 2+ bearish scores
            elif bearish >= 2:
                return ("bearish", "BEARISH")
            else:
                return ("mixed", "MIXED")
        except Exception:
            return ("mixed", "MIXED")

    def _calculate_lt_signal(self) -> tuple:
        """Calculate long-term signal type and label. Returns (type, label)."""
        try:
            htf_1h = self.query_one("#htf-1h", SignalCard)
            htf_4h = self.query_one("#htf-4h", SignalCard)
            htf_daily = self.query_one("#htf-daily", SignalCard)

            scores = [
                htf_1h.signal.score,
                htf_4h.signal.score,
                htf_daily.signal.score
            ]

            # Count strong bullish (>=70), moderate bullish (60-69), and bearish (<=40)
            strong_bullish = sum(1 for s in scores if s >= 70)
            moderate_bullish = sum(1 for s in scores if 60 <= s < 70)
            bearish = sum(1 for s in scores if s <= 40)
            
            # Calculate average score for better signal quality
            avg_score = sum(scores) / len(scores) if scores else 50

            # Bullish if: 2+ strong bullish, OR (1+ strong + 1+ moderate), OR avg >= 60
            if strong_bullish >= 2 or (strong_bullish >= 1 and moderate_bullish >= 1) or avg_score >= 60:
                return ("bullish", "BULLISH")
            # Bearish if: 2+ bearish scores
            elif bearish >= 2:
                return ("bearish", "BEARISH")
            else:
                return ("mixed", "MIXED")
        except Exception:
            return ("mixed", "MIXED")

    def _check_ltf_alignment(self) -> Optional[str]:
        """Check if all LTF signals are aligned. Returns 'red', 'green', or None."""
        try:
            ltf_10s = self.query_one("#ltf-10s", SignalCard)
            ltf_1m = self.query_one("#ltf-1m", SignalCard)
            ltf_15m = self.query_one("#ltf-15m", SignalCard)

            scores = [
                ltf_10s.signal.score,
                ltf_1m.signal.score,
                ltf_15m.signal.score
            ]

            # Check if all are red
            if all(self._is_red(s) for s in scores):
                return "red"
            # Check if all are green
            elif all(self._is_green(s) for s in scores):
                return "green"
            return None
        except Exception:
            return None

    def _check_htf_alignment(self) -> Optional[str]:
        """Check if all HTF signals are aligned. Returns 'red', 'green', or None."""
        try:
            htf_1h = self.query_one("#htf-1h", SignalCard)
            htf_4h = self.query_one("#htf-4h", SignalCard)
            htf_daily = self.query_one("#htf-daily", SignalCard)

            scores = [
                htf_1h.signal.score,
                htf_4h.signal.score,
                htf_daily.signal.score
            ]

            # Check if all are red
            if all(self._is_red(s) for s in scores):
                return "red"
            # Check if all are green
            elif all(self._is_green(s) for s in scores):
                return "green"
            return None
        except Exception:
            return None

    def _update_flashing(self) -> None:
        """Update flashing state for aligned signals."""
        ltf_align = self._check_ltf_alignment()
        htf_align = self._check_htf_alignment()

        # LTF cards
        ltf_ids = ["ltf-10s", "ltf-1m", "ltf-15m"]
        for card_id in ltf_ids:
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Remove flash classes first
                card.remove_class("flash-red", "flash-green")
                # Only add flash class if aligned and in flash state
                if ltf_align == "red" and self._flash_state:
                    card.add_class("flash-red")
                elif ltf_align == "green" and self._flash_state:
                    card.add_class("flash-green")
            except Exception:
                pass

        # HTF cards
        htf_ids = ["htf-1h", "htf-4h", "htf-daily"]
        for card_id in htf_ids:
            try:
                card = self.query_one(f"#{card_id}", SignalCard)
                # Remove flash classes first
                card.remove_class("flash-red", "flash-green")
                # Only add flash class if aligned and in flash state
                if htf_align == "red" and self._flash_state:
                    card.add_class("flash-red")
                elif htf_align == "green" and self._flash_state:
                    card.add_class("flash-green")
            except Exception:
                pass

    def _flash_toggle(self) -> None:
        """Toggle flash state."""
        self._flash_state = not self._flash_state
        self._update_flashing()

    async def on_unmount(self) -> None:
        """Cleanup when app closes."""
        # Stop flash timer
        if self._flash_timer:
            self._flash_timer.stop()

        # Stop PulseCore
        if self.core:
            await self.core.stop()

    def action_quit(self) -> None:
        """Handle quit action."""
        self.run_worker(self._shutdown())

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        if self.core:
            await self.core.stop()
        self.exit()


def run_pulse_app():
    """Entry point for Pulse terminal dashboard."""
    try:
        app = MarketDashboard()
        app.run()
    except Exception as e:
        logger.error(f"Pulse app crashed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_pulse_app()
