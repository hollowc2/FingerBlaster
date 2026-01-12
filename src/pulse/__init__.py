"""
Pulse - Real-time Coinbase Market Data Analysis Module.

Provides real-time market data collection and technical indicators
for market direction analysis. Can run standalone or integrate with FingerBlaster.

Usage (standalone):
    python -m src.pulse
    python -m src.pulse --products BTC-USD ETH-USD
    python -m src.pulse --timeframes 1m 5m 1h
    python main.py --pulse

Usage (integrated):
    from src.pulse import PulseCore, PulseConfig

    pulse = PulseCore(config=PulseConfig(products=["BTC-USD"]))

    # Register callbacks
    pulse.register_callback('candle_update', my_handler)
    pulse.register_callback('indicator_update', my_indicator_handler)

    await pulse.start()
    # ... run your application ...
    await pulse.stop()
"""

from src.pulse.config import (
    Alert,
    BucketedOrderBook,
    Candle,
    IndicatorSnapshot,
    PULSE_EVENTS,
    PulseConfig,
    Ticker,
    Timeframe,
    Trade,
)
from src.pulse.core import PulseCore

__all__ = [
    # Core
    "PulseCore",
    "PulseConfig",
    # Data types
    "Timeframe",
    "Candle",
    "Trade",
    "Ticker",
    "BucketedOrderBook",
    "IndicatorSnapshot",
    "Alert",
    # Constants
    "PULSE_EVENTS",
]

__version__ = "0.1.0"
