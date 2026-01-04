"""Signal scoring algorithm for converting technical indicators to trading signals.

Transforms IndicatorSnapshot into a 0-100 score with bullish/bearish label.
"""

from typing import Tuple
from pulse.config import IndicatorSnapshot


def compute_signal_score(snapshot: IndicatorSnapshot) -> Tuple[int, str, str]:
    """
    Compute bullish/bearish score from technical indicators.

    Weights each indicator component and combines into 0-100 scale where:
    - 0-20: Strong Sell (bearish)
    - 21-40: Bearish
    - 41-60: Neutral/Mixed
    - 61-80: Bullish
    - 81-100: Strong Buy (bullish)

    Args:
        snapshot: IndicatorSnapshot with all computed indicators

    Returns:
        Tuple of (score: 0-100, label: str, description: str)
    """
    components = []

    # 1. RSI Component (weight: 25%)
    if snapshot.rsi is not None:
        rsi_score = _compute_rsi_score(snapshot.rsi)
        components.append(("RSI", rsi_score, 0.25))

    # 2. MACD Component (weight: 25%)
    if snapshot.macd_histogram is not None:
        macd_score = _compute_macd_score(snapshot.macd_histogram)
        components.append(("MACD", macd_score, 0.25))

    # 3. Trend Direction from ADX/DI (weight: 25%)
    if snapshot.trend_direction:
        trend_score = _compute_trend_score(snapshot)
        components.append(("Trend", trend_score, 0.25))

    # 4. VWAP Deviation (weight: 25%)
    if snapshot.vwap_deviation is not None:
        vwap_score = _compute_vwap_score(snapshot.vwap_deviation)
        components.append(("VWAP", vwap_score, 0.25))

    # Calculate weighted average
    if not components:
        return (50, "No Data", "Insufficient indicator data")

    total_weight = sum(c[2] for c in components)
    weighted_sum = sum(c[1] * c[2] for c in components)
    score = int(weighted_sum / total_weight)

    # Clamp to 0-100 range
    score = max(0, min(100, score))

    # Generate label and description based on score
    label, description = _get_label_and_description(score)

    return (score, label, description)


def _compute_rsi_score(rsi: float) -> float:
    """
    Compute bullish/bearish score from RSI (0-100).

    - RSI > 70: Overbought = bearish (score 0-30)
    - RSI < 30: Oversold = bullish (score 70-100)
    - 30-70: Neutral zone (score 40-60)
    """
    if rsi > 70:
        # Overbought: higher RSI = more bearish
        # At RSI=70, score=50; at RSI=100, score=0
        return max(0, 50 - (rsi - 70) * (50 / 30))
    elif rsi < 30:
        # Oversold: lower RSI = more bullish
        # At RSI=30, score=50; at RSI=0, score=100
        return min(100, 50 + (30 - rsi) * (50 / 30))
    else:
        # Neutral zone: RSI maps linearly to 40-60
        # RSI=50 -> score=50
        return 40 + (rsi - 30) * (20 / 40)


def _compute_macd_score(macd_histogram: float) -> float:
    """
    Compute bullish/bearish score from MACD histogram.

    - Positive histogram: bullish (score > 50)
    - Negative histogram: bearish (score < 50)
    """
    if macd_histogram > 0:
        # Bullish: stronger positive = higher score
        # Map magnitude to 50-100 range
        return min(100, 50 + abs(macd_histogram) * 100)
    else:
        # Bearish: stronger negative = lower score
        # Map magnitude to 0-50 range
        return max(0, 50 - abs(macd_histogram) * 100)


def _compute_trend_score(snapshot: IndicatorSnapshot) -> float:
    """
    Compute trend score from trend direction and ADX strength.

    - UP trend: score > 50, amplified by ADX
    - DOWN trend: score < 50, amplified by ADX
    - SIDEWAYS: score = 50
    """
    base_scores = {
        "UP": 75,
        "DOWN": 25,
        "SIDEWAYS": 50,
    }

    trend_score = base_scores.get(snapshot.trend_direction, 50)

    # Amplify by ADX strength (0-100)
    if snapshot.adx is not None and snapshot.adx > 0:
        # ADX > 25 = strong trend, amplify deviation from 50
        adx_factor = min(snapshot.adx / 25, 2.0)  # Cap at 2x amplification
        trend_score = 50 + (trend_score - 50) * adx_factor

    return max(0, min(100, trend_score))


def _compute_vwap_score(vwap_deviation: float) -> float:
    """
    Compute score from price position relative to VWAP.

    - Above VWAP: bullish (score > 50)
    - Below VWAP: bearish (score < 50)
    - At VWAP: neutral (score = 50)
    """
    if vwap_deviation > 0:
        # Above VWAP: bullish
        # deviation=1 -> score=60; deviation=5 -> score=100
        return min(100, 50 + vwap_deviation * 10)
    else:
        # Below VWAP: bearish
        # deviation=-1 -> score=40; deviation=-5 -> score=0
        return max(0, 50 + vwap_deviation * 10)


def _get_label_and_description(score: int) -> Tuple[str, str]:
    """Get label and description based on score."""
    if score >= 80:
        return ("Strong Buy", "Multiple bullish signals aligned")
    elif score >= 65:
        return ("Bullish", "Moderate bullish momentum")
    elif score >= 55:
        return ("Lean Bull", "Slight bullish bias")
    elif score >= 45:
        return ("Neutral", "Mixed or choppy conditions")
    elif score >= 35:
        return ("Lean Bear", "Slight bearish bias")
    elif score >= 20:
        return ("Bearish", "Moderate bearish momentum")
    else:
        return ("Strong Sell", "Multiple bearish signals aligned")
