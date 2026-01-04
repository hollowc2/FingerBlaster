"""Synchronous Strategy Data Access

Low-latency direct access to strategy data for use in external trading applications.
This module provides simple functions that can be called directly without async overhead.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("FingerBlaster.StrategyDataSync")


@dataclass
class TradingData:
    """Simplified trading data structure for direct access."""
    # Market basics
    price_to_beat: Optional[float] = None
    btc: Optional[float] = None
    delta: Optional[float] = None
    remain: int = 0
    
    # Volatility
    sigma: Optional[float] = None
    sigma_label: str = ""
    
    # Historical
    prior: list = None
    
    # Regime
    regime: str = ""
    regime_strength: float = 0.0
    
    # Oracle
    oracle: Optional[int] = None
    
    # Up side
    yes_FV: Optional[float] = None
    yes_Edge: Optional[str] = None
    yes_Edge_bps: Optional[float] = None
    
    # Down side
    no_FV: Optional[float] = None
    no_Edge: Optional[str] = None
    no_Edge_bps: Optional[float] = None
    
    def __post_init__(self):
        if self.prior is None:
            self.prior = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class StrategyDataProvider:
    """Provides direct, low-latency access to strategy data.
    
    This class caches the latest analytics snapshot and provides
    synchronous access methods for minimal latency.
    """
    
    def __init__(self, core):
        """Initialize with FingerBlasterCore instance.
        
        Args:
            core: FingerBlasterCore instance
        """
        self.core = core
        self._cached_snapshot = None
        self._cached_market = None
        self._cached_btc_price = None
        self._cached_price_to_beat = None
        self._cached_time_remaining = None
        self._cached_prior_outcomes = None
    
    def update_cache(self, snapshot, market, btc_price, strike, time_remaining, prior_outcomes):
        """Update internal cache with latest data.
        
        This should be called by the core whenever analytics update.
        For minimal latency, this happens synchronously.
        
        Args:
            snapshot: AnalyticsSnapshot
            market: Market dict
            btc_price: Current BTC price
            price_to_beat: Price to beat (float)
            time_remaining: Seconds remaining
            prior_outcomes: List of prior outcomes
        """
        self._cached_snapshot = snapshot
        self._cached_market = market
        self._cached_btc_price = btc_price
        self._cached_price_to_beat = strike
        self._cached_time_remaining = time_remaining
        self._cached_prior_outcomes = prior_outcomes or []
    
    def get_trading_data(self) -> Optional[TradingData]:
        """Get current trading data (synchronous, low-latency).
        
        Returns:
            TradingData object or None if no data available
        """
        if not self._cached_snapshot:
            return None
        
        snapshot = self._cached_snapshot
        
        # Calculate delta
        delta = None
        if self._cached_btc_price and self._cached_price_to_beat:
            delta = self._cached_btc_price - self._cached_price_to_beat
        
        # Convert edge enum to string
        def edge_to_str(edge):
            if edge is None:
                return None
            if hasattr(edge, 'value'):
                return edge.value
            return str(edge)
        
        return TradingData(
            price_to_beat=self._cached_price_to_beat,
            btc=self._cached_btc_price,
            delta=delta,
            remain=self._cached_time_remaining or 0,
            sigma=snapshot.z_score,
            sigma_label=snapshot.sigma_label or "",
            prior=self._cached_prior_outcomes.copy() if self._cached_prior_outcomes else [],
            regime=snapshot.regime_direction or "",
            regime_strength=snapshot.regime_strength or 0.0,
            oracle=snapshot.oracle_lag_ms,
            yes_FV=snapshot.fair_value_yes,
            yes_Edge=edge_to_str(snapshot.edge_yes),
            yes_Edge_bps=snapshot.edge_bps_yes,
            no_FV=snapshot.fair_value_no,
            no_Edge=edge_to_str(snapshot.edge_no),
            no_Edge_bps=snapshot.edge_bps_no,
        )
    
    def get_data_dict(self) -> Optional[Dict[str, Any]]:
        """Get trading data as dictionary (for JSON serialization).
        
        Returns:
            Dictionary or None if no data available
        """
        data = self.get_trading_data()
        return data.to_dict() if data else None


# Global provider instance (set by core)
_provider: Optional[StrategyDataProvider] = None


def set_provider(provider: StrategyDataProvider) -> None:
    """Set the global strategy data provider.
    
    This is called by FingerBlasterCore on initialization.
    
    Args:
        provider: StrategyDataProvider instance
    """
    global _provider
    _provider = provider


def get_trading_data() -> Optional[TradingData]:
    """Get current trading data (global function for easy import).
    
    This is the main function to call from your external trading app.
    It provides direct, low-latency access to the latest data.
    
    Usage:
        from src.strategy_data_sync import get_trading_data
        
        data = get_trading_data()
        if data:
            print(f"Up FV: {data.yes_FV}")
            print(f"Up Edge: {data.yes_Edge} ({data.yes_Edge_bps} bps)")
    
    Returns:
        TradingData object or None if no data available
    """
    if _provider is None:
        return None
    return _provider.get_trading_data()


def get_data_dict() -> Optional[Dict[str, Any]]:
    """Get trading data as dictionary.
    
    Returns:
        Dictionary or None if no data available
    """
    if _provider is None:
        return None
    return _provider.get_data_dict()

