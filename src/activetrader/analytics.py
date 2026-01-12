"""Advanced analytics engine for high-frequency binary options trading.

This module provides quantitative analysis for 15-minute BTC Up/Down markets:
- Basis Points (bps) calculation
- Binary Fair Value using simplified Black-Scholes
- Edge Detection (market vs fair value)
- Z-Score / Sigma with rolling volatility
- Liquidity depth analysis
- Real-time PnL tracking
- Slippage estimation
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

import pandas as pd

logger = logging.getLogger("FingerBlaster.Analytics")


class TimerUrgency(Enum):
    """Timer urgency levels for UI coloring."""
    NORMAL = "normal"      # > 5 minutes, green
    WATCHFUL = "watchful"  # 2-5 minutes, orange
    CRITICAL = "critical"  # < 2 minutes, blinking red


class EdgeDirection(Enum):
    """Edge direction for fair value comparison."""
    UNDERVALUED = "undervalued"  # Market < FV, good to buy
    OVERVALUED = "overvalued"    # Market > FV, good to sell
    FAIR = "fair"                # Market ≈ FV


@dataclass
class AnalyticsSnapshot:
    """Immutable snapshot of all analytics data for UI rendering."""
    # Basis points
    basis_points: Optional[float] = None
    
    # Fair value analysis
    fair_value_yes: Optional[float] = None
    fair_value_no: Optional[float] = None
    edge_yes: Optional[EdgeDirection] = None
    edge_no: Optional[EdgeDirection] = None
    edge_bps_yes: Optional[float] = None  # Edge in basis points
    edge_bps_no: Optional[float] = None
    
    # Volatility / Z-score
    realized_volatility: Optional[float] = None  # 15-min annualized vol
    z_score: Optional[float] = None  # Standard deviations from strike
    sigma_label: str = ""  # e.g., "+1.5σ"
    
    # Liquidity
    yes_bid_depth: Optional[float] = None  # $ at top of book
    yes_ask_depth: Optional[float] = None
    no_bid_depth: Optional[float] = None
    no_ask_depth: Optional[float] = None
    
    # Real-time PnL
    unrealized_pnl_yes: Optional[float] = None
    unrealized_pnl_no: Optional[float] = None
    total_unrealized_pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    
    # Slippage estimation
    estimated_slippage_yes: Optional[float] = None
    estimated_slippage_no: Optional[float] = None
    
    # Timer urgency
    timer_urgency: TimerUrgency = TimerUrgency.NORMAL
    seconds_remaining: int = 0
    
    # Regime detection
    regime_direction: str = ""  # "BULLISH" or "BEARISH"
    regime_strength: float = 0.0  # 0-100%
    
    # Oracle lag
    oracle_lag_ms: Optional[int] = None  # Lag in milliseconds
    chainlink_price: Optional[float] = None
    cex_price: Optional[float] = None


@dataclass
class VolatilityWindow:
    """Rolling window for volatility calculation."""
    prices: deque = field(default_factory=lambda: deque(maxlen=900))  # 15 min of 1-sec data
    timestamps: deque = field(default_factory=lambda: deque(maxlen=900))
    

class AnalyticsEngine:
    """High-performance analytics engine for binary options trading.
    
    Provides real-time calculations for:
    - Basis points distance from strike
    - Fair value using binary option pricing
    - Edge detection
    - Rolling volatility and z-score
    - Liquidity analysis
    - PnL tracking
    - Slippage estimation
    """
    
    # Constants for calculations
    RISK_FREE_RATE = 0.0  # 0% as specified
    BPS_MULTIPLIER = 10000
    SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60
    
    def __init__(self):
        """Initialize the analytics engine."""
        self._vol_window = VolatilityWindow()
        self._last_update = 0.0
        self._update_throttle = 0.1  # 100ms minimum between updates
        
        # Cached values
        self._cached_snapshot: Optional[AnalyticsSnapshot] = None
        self._cache_timestamp = 0.0
        self._cache_ttl = 0.05  # 50ms cache
        
        # Oracle lag tracking
        self._chainlink_price: Optional[float] = None
        self._chainlink_timestamp: Optional[float] = None
        self._cex_price: Optional[float] = None
        self._cex_timestamp: Optional[float] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    # =========================================================================
    # BASIS POINTS CALCULATION
    # =========================================================================
    
    def calculate_basis_points(
        self, 
        current_price: float, 
        price_to_beat: float
    ) -> Optional[float]:
        """Calculate basis points distance from price to beat.
        
        Formula: ((Current BTC Price - Price to Beat) / Price to Beat) * 10,000
        
        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat
            
        Returns:
            Basis points as float, positive = above price to beat
        """
        if price_to_beat <= 0 or current_price <= 0:
            return None
        
        return ((current_price - price_to_beat) / price_to_beat) * self.BPS_MULTIPLIER
    
    # =========================================================================
    # BINARY FAIR VALUE (SIMPLIFIED BLACK-SCHOLES)
    # =========================================================================
    
    def calculate_binary_fair_value(
        self,
        current_price: float,
        price_to_beat: float,
        time_to_expiry_seconds: float,
        volatility: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate fair value of binary Up/Down contract using Black-Scholes.
        
        For a cash-or-nothing binary call (Up pays $1 if S > K):
        FV_YES = N(d2) where d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)
        
        For binary put (Down pays $1 if S < K):
        FV_NO = N(-d2) = 1 - N(d2)
        
        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat
            time_to_expiry_seconds: Time to expiry in seconds
            volatility: Optional volatility override (annualized)
            
        Returns:
            Tuple of (fair_value_yes, fair_value_no) as probabilities [0,1]
        """
        if price_to_beat <= 0 or current_price <= 0 or time_to_expiry_seconds <= 0:
            return None, None
        
        # Use provided volatility or estimate from recent data
        if volatility is None:
            volatility = self._get_realized_volatility()
            if volatility is None or volatility <= 0:
                # Default to 60% annualized vol for BTC if no data
                volatility = 0.60
        
        # Convert time to years
        T = time_to_expiry_seconds / self.SECONDS_IN_YEAR
        
        # Prevent numerical issues with very small T
        if T < 1e-10:
            # At expiry, it's a binary outcome
            return (1.0, 0.0) if current_price >= price_to_beat else (0.0, 1.0)
        
        try:
            sigma_sqrt_t = volatility * math.sqrt(T)
            
            # Prevent division by zero
            if sigma_sqrt_t < 1e-10:
                return (1.0, 0.0) if current_price >= price_to_beat else (0.0, 1.0)
            
            # d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)
            # With r = 0: d2 = (ln(S/K) - σ²T/2) / (σ√T)
            d2 = (math.log(current_price / price_to_beat) - 
                  (volatility ** 2) * T / 2) / sigma_sqrt_t
            
            # N(d2) using standard normal CDF
            fv_yes = self._norm_cdf(d2)
            fv_no = 1.0 - fv_yes
            
            # Clamp to valid probability range
            fv_yes = max(0.01, min(0.99, fv_yes))
            fv_no = max(0.01, min(0.99, fv_no))
            
            return fv_yes, fv_no
            
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.debug(f"Fair value calculation error: {e}")
            return None, None
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function.
        
        Uses error function for accurate calculation.
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    # =========================================================================
    # EDGE DETECTION
    # =========================================================================
    
    def calculate_edge(
        self,
        market_price: float,
        fair_value: float,
        threshold_bps: float = 50.0  # 0.5% threshold for "fair"
    ) -> Tuple[EdgeDirection, float]:
        """Detect edge between market price and fair value.
        
        Args:
            market_price: Current market price
            fair_value: Calculated fair value
            threshold_bps: Basis points threshold for "fair" classification
            
        Returns:
            Tuple of (EdgeDirection, edge_in_bps)
        """
        if fair_value <= 0 or market_price <= 0:
            return EdgeDirection.FAIR, 0.0
        
        # Calculate edge in basis points
        edge_bps = ((fair_value - market_price) / fair_value) * self.BPS_MULTIPLIER
        
        if abs(edge_bps) < threshold_bps:
            return EdgeDirection.FAIR, edge_bps
        elif edge_bps > 0:
            # FV > Market, undervalued (good to buy)
            return EdgeDirection.UNDERVALUED, edge_bps
        else:
            # FV < Market, overvalued (good to sell)
            return EdgeDirection.OVERVALUED, edge_bps
    
    # =========================================================================
    # VOLATILITY & Z-SCORE
    # =========================================================================
    
    async def add_price_sample(self, price: float) -> None:
        """Add a price sample for volatility calculation.
        
        Args:
            price: BTC price sample
        """
        if price <= 0:
            return
        
        async with self._lock:
            now = time.time()
            self._vol_window.prices.append(price)
            self._vol_window.timestamps.append(now)
    
    def _get_realized_volatility(self) -> Optional[float]:
        """Calculate 15-minute realized volatility (annualized).
        
        Uses log returns for calculation.
        
        Returns:
            Annualized volatility as decimal (e.g., 0.60 for 60%)
        """
        prices = list(self._vol_window.prices)
        
        if len(prices) < 10:  # Need at least 10 samples
            return None
        
        try:
            # Calculate log returns
            log_returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0 and prices[i] > 0:
                    log_returns.append(math.log(prices[i] / prices[i-1]))
            
            if len(log_returns) < 5:
                return None
            
            # Standard deviation of returns
            mean_return = sum(log_returns) / len(log_returns)
            variance = sum((r - mean_return) ** 2 for r in log_returns) / len(log_returns)
            std_dev = math.sqrt(variance)
            
            # Annualize: assuming 1-second samples
            # Annualized vol = σ * √(seconds_per_year)
            annualized_vol = std_dev * math.sqrt(self.SECONDS_IN_YEAR)
            
            return annualized_vol
            
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Volatility calculation error: {e}")
            return None
    
    def calculate_z_score(
        self,
        current_price: float,
        price_to_beat: float,
        time_to_expiry_seconds: float,
        volatility: Optional[float] = None
    ) -> Tuple[Optional[float], str]:
        """Calculate how many standard deviations price is from price to beat.
        
        Z = (ln(S/K)) / (σ√T)
        
        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat
            time_to_expiry_seconds: Time remaining
            volatility: Optional volatility override
            
        Returns:
            Tuple of (z_score, sigma_label)
        """
        if price_to_beat <= 0 or current_price <= 0:
            return None, ""
        
        if volatility is None:
            volatility = self._get_realized_volatility()
            if volatility is None or volatility <= 0:
                volatility = 0.60
        
        T = time_to_expiry_seconds / self.SECONDS_IN_YEAR
        
        if T <= 0 or volatility <= 0:
            return None, ""
        
        try:
            sigma_sqrt_t = volatility * math.sqrt(T)
            if sigma_sqrt_t < 1e-10:
                return None, ""
            
            z = math.log(current_price / price_to_beat) / sigma_sqrt_t
            
            # Create sigma label
            sign = "+" if z >= 0 else ""
            label = f"{sign}{z:.2f}σ"
            
            return z, label
            
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Z-score calculation error: {e}")
            return None, ""
    
    # =========================================================================
    # LIQUIDITY ANALYSIS
    # =========================================================================
    
    def _is_order_book_empty(self, order_book: Dict[str, Dict[str, Dict[float, float]]]) -> bool:
        """Check if order book has no data.

        Args:
            order_book: Raw order book data {Up/Down: {bids/asks: {price: size}}}

        Returns:
            True if order book is completely empty (no bids/asks for either side)
        """
        for side in ['Up', 'Down']:
            if side in order_book:
                bids = order_book[side].get('bids', {})
                asks = order_book[side].get('asks', {})
                if bids or asks:
                    return False
        return True

    def calculate_liquidity_depth(
        self,
        order_book: Dict[str, Dict[str, Dict[float, float]]]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Calculate liquidity depth at top of book.

        Args:
            order_book: Raw order book data {Up/Down: {bids/asks: {price: size}}}

        Returns:
            Dictionary with depth in dollars at top of book for each side
            Returns None for depth if order book is not yet populated
        """
        # Check if order book is empty (WebSocket not yet connected)
        if self._is_order_book_empty(order_book):
            return {
                'Up': {'bid_depth': None, 'ask_depth': None},
                'Down': {'bid_depth': None, 'ask_depth': None}
            }

        result = {
            'Up': {'bid_depth': 0.0, 'ask_depth': 0.0},
            'Down': {'bid_depth': 0.0, 'ask_depth': 0.0}
        }

        for side in ['Up', 'Down']:
            if side not in order_book:
                continue

            # Best bid depth (highest price)
            bids = order_book[side].get('bids', {})
            if bids:
                best_bid_price = max(bids.keys())
                best_bid_size = bids[best_bid_price]
                result[side]['bid_depth'] = best_bid_price * best_bid_size

            # Best ask depth (lowest price)
            asks = order_book[side].get('asks', {})
            if asks:
                best_ask_price = min(asks.keys())
                best_ask_size = asks[best_ask_price]
                result[side]['ask_depth'] = best_ask_price * best_ask_size

        return result
    
    # =========================================================================
    # REAL-TIME PNL
    # =========================================================================
    
    def calculate_unrealized_pnl(
        self,
        yes_position: float,
        no_position: float,
        avg_entry_yes: Optional[float],
        avg_entry_no: Optional[float],
        current_yes_price: float,
        current_no_price: float
    ) -> Tuple[float, float, float, Optional[float]]:
        """Calculate unrealized PnL for positions.
        
        Args:
            yes_position: Number of Up shares
            no_position: Number of Down shares
            avg_entry_yes: Average entry price for Up
            avg_entry_no: Average entry price for Down
            current_yes_price: Current mid price for Up
            current_no_price: Current mid price for Down
            
        Returns:
            Tuple of (pnl_yes, pnl_no, total_pnl, pnl_percentage)
        """
        # Use threshold to handle floating point precision issues
        MIN_BALANCE_THRESHOLD = 0.1
        
        pnl_yes = 0.0
        pnl_no = 0.0
        total_cost = 0.0
        
        # Up position PnL - only calculate if position is above threshold
        if yes_position > MIN_BALANCE_THRESHOLD and avg_entry_yes is not None and avg_entry_yes > 0:
            cost_yes = yes_position * avg_entry_yes
            value_yes = yes_position * current_yes_price
            pnl_yes = value_yes - cost_yes
            total_cost += cost_yes
        
        # Down position PnL - only calculate if position is above threshold
        if no_position > MIN_BALANCE_THRESHOLD and avg_entry_no is not None and avg_entry_no > 0:
            cost_no = no_position * avg_entry_no
            value_no = no_position * current_no_price
            pnl_no = value_no - cost_no
            total_cost += cost_no
        
        total_pnl = pnl_yes + pnl_no
        
        # Calculate percentage
        pnl_percentage = None
        if total_cost > 0:
            pnl_percentage = (total_pnl / total_cost) * 100
        
        return pnl_yes, pnl_no, total_pnl, pnl_percentage
    
    # =========================================================================
    # SLIPPAGE ESTIMATION
    # =========================================================================
    
    def estimate_slippage(
        self,
        order_size_usd: float,
        order_book_side: Dict[float, float],
        is_buy: bool
    ) -> Tuple[float, float]:
        """Estimate slippage for a given order size.
        
        Args:
            order_size_usd: Order size in USD
            order_book_side: Order book side to consume (asks for buy, bids for sell)
            is_buy: True for buy orders
            
        Returns:
            Tuple of (expected_fill_price, slippage_bps)
        """
        if not order_book_side or order_size_usd <= 0:
            return 0.0, 0.0
        
        # Sort prices appropriately
        if is_buy:
            # For buys, consume asks in ascending price order
            sorted_levels = sorted(order_book_side.items(), key=lambda x: x[0])
        else:
            # For sells, consume bids in descending price order
            sorted_levels = sorted(order_book_side.items(), key=lambda x: x[0], reverse=True)
        
        if not sorted_levels:
            return 0.0, 0.0
        
        best_price = sorted_levels[0][0]
        remaining_size = order_size_usd
        total_filled = 0.0
        total_cost = 0.0
        
        for price, size in sorted_levels:
            level_value = price * size
            
            if remaining_size <= level_value:
                # Can fill remaining at this level
                shares = remaining_size / price
                total_filled += shares
                total_cost += remaining_size
                remaining_size = 0
                break
            else:
                # Consume entire level
                total_filled += size
                total_cost += level_value
                remaining_size -= level_value
        
        if total_filled <= 0:
            return best_price, 0.0
        
        avg_fill_price = total_cost / total_filled
        
        # Calculate slippage in basis points
        if best_price > 0:
            slippage_bps = abs(avg_fill_price - best_price) / best_price * self.BPS_MULTIPLIER
        else:
            slippage_bps = 0.0
        
        return avg_fill_price, slippage_bps
    
    # =========================================================================
    # TIMER URGENCY
    # =========================================================================
    
    def get_timer_urgency(self, seconds_remaining: int) -> TimerUrgency:
        """Determine timer urgency level.
        
        Args:
            seconds_remaining: Seconds until expiry
            
        Returns:
            TimerUrgency enum value
        """
        if seconds_remaining <= 0:
            return TimerUrgency.CRITICAL
        
        minutes = seconds_remaining / 60
        
        if minutes < 2:
            return TimerUrgency.CRITICAL
        elif minutes < 5:
            return TimerUrgency.WATCHFUL
        else:
            return TimerUrgency.NORMAL
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def detect_regime(
        self,
        prior_outcomes: List[str]
    ) -> Tuple[str, float]:
        """Analyze prior outcomes to detect market regime.
        
        Args:
            prior_outcomes: List of 'Up' or 'Down' outcomes
            
        Returns:
            Tuple of (regime_direction, strength_percentage)
        """
        if not prior_outcomes:
            return "", 0.0
        
        yes_count = sum(1 for o in prior_outcomes if o == 'Up' or (isinstance(o, str) and o.upper() in ('UP', 'YES')))
        total = len(prior_outcomes)
        
        if total == 0:
            return "", 0.0
        
        yes_pct = yes_count / total
        no_pct = 1 - yes_pct
        
        # Determine regime (need > 50% for a direction)
        if yes_pct > 0.5:
            direction = "BULLISH"
            strength = yes_pct * 100
        elif no_pct > 0.5:
            direction = "BEARISH"
            strength = no_pct * 100
        else:
            direction = "NEUTRAL"
            strength = 50.0
        
        return direction, strength
    
    # =========================================================================
    # ORACLE LAG MONITORING
    # =========================================================================
    
    def update_oracle_prices(
        self,
        chainlink_price: Optional[float],
        cex_price: Optional[float],
        chainlink_timestamp: Optional[float] = None,
        cex_timestamp: Optional[float] = None
    ) -> None:
        """Update oracle price tracking for lag calculation.
        
        Args:
            chainlink_price: Price from Chainlink feed
            cex_price: Price from CEX (Binance/Coinbase)
            chainlink_timestamp: Timestamp of Chainlink update
            cex_timestamp: Timestamp of CEX update
        """
        if chainlink_price and chainlink_price > 0:
            self._chainlink_price = chainlink_price
            self._chainlink_timestamp = chainlink_timestamp or time.time()
        
        if cex_price and cex_price > 0:
            self._cex_price = cex_price
            self._cex_timestamp = cex_timestamp or time.time()
    
    def calculate_oracle_lag(self) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """Calculate lag between Chainlink and CEX prices.
        
        Returns:
            Tuple of (lag_ms, chainlink_price, cex_price)
        """
        if (self._chainlink_timestamp is None or 
            self._cex_timestamp is None or
            self._chainlink_price is None or
            self._cex_price is None):
            return None, self._chainlink_price, self._cex_price
        
        # Calculate time lag
        lag_seconds = abs(self._chainlink_timestamp - self._cex_timestamp)
        lag_ms = int(lag_seconds * 1000)
        
        return lag_ms, self._chainlink_price, self._cex_price
    
    # =========================================================================
    # FULL SNAPSHOT GENERATION
    # =========================================================================
    
    async def generate_snapshot(
        self,
        btc_price: float,
        price_to_beat: float,
        time_remaining_seconds: int,
        yes_market_price: float,
        no_market_price: float,
        order_book: Dict[str, Dict[str, Dict[float, float]]],
        yes_position: float,
        no_position: float,
        avg_entry_yes: Optional[float],
        avg_entry_no: Optional[float],
        prior_outcomes: List[str],
        order_size_usd: float = 10.0
    ) -> AnalyticsSnapshot:
        """Generate a complete analytics snapshot.
        
        This is the main method called by the core to get all analytics.
        
        Args:
            btc_price: Current BTC price
            price_to_beat: Market price to beat
            time_remaining_seconds: Seconds until expiry
            yes_market_price: Current Up market price
            no_market_price: Current Down market price
            order_book: Full order book data
            yes_position: Up position size
            no_position: Down position size
            avg_entry_yes: Average entry price for Up
            avg_entry_no: Average entry price for Down
            prior_outcomes: List of prior outcome strings
            order_size_usd: Order size for slippage calculation
            
        Returns:
            Complete AnalyticsSnapshot
        """
        # Add price sample for volatility
        await self.add_price_sample(btc_price)
        
        # Calculate all analytics
        
        # 1. Basis points
        bps = self.calculate_basis_points(btc_price, price_to_beat)
        
        # 2. Get realized volatility
        volatility = self._get_realized_volatility()
        
        # 3. Fair value
        fv_yes, fv_no = self.calculate_binary_fair_value(
            btc_price, price_to_beat, time_remaining_seconds, volatility
        )
        
        # 4. Edge detection
        edge_yes, edge_bps_yes = EdgeDirection.FAIR, 0.0
        edge_no, edge_bps_no = EdgeDirection.FAIR, 0.0
        if fv_yes is not None:
            edge_yes, edge_bps_yes = self.calculate_edge(yes_market_price, fv_yes)
        if fv_no is not None:
            edge_no, edge_bps_no = self.calculate_edge(no_market_price, fv_no)
        
        # 5. Z-score
        z_score, sigma_label = self.calculate_z_score(
            btc_price, price_to_beat, time_remaining_seconds, volatility
        )
        
        # 6. Liquidity depth
        liquidity = self.calculate_liquidity_depth(order_book)
        
        # 7. Unrealized PnL
        pnl_yes, pnl_no, total_pnl, pnl_pct = self.calculate_unrealized_pnl(
            yes_position, no_position,
            avg_entry_yes, avg_entry_no,
            yes_market_price, no_market_price
        )
        
        # 8. Slippage estimation
        yes_asks = order_book.get('Up', {}).get('asks', {})
        no_asks = order_book.get('Down', {}).get('asks', {})
        
        _, slippage_yes = self.estimate_slippage(order_size_usd, yes_asks, is_buy=True)
        _, slippage_no = self.estimate_slippage(order_size_usd, no_asks, is_buy=True)
        
        # 9. Timer urgency
        urgency = self.get_timer_urgency(time_remaining_seconds)
        
        # 10. Regime detection
        regime_dir, regime_strength = self.detect_regime(prior_outcomes)
        
        # 11. Oracle lag
        oracle_lag, cl_price, cex_price = self.calculate_oracle_lag()
        
        return AnalyticsSnapshot(
            # Basis points
            basis_points=bps,
            
            # Fair value
            fair_value_yes=fv_yes,
            fair_value_no=fv_no,
            edge_yes=edge_yes,
            edge_no=edge_no,
            edge_bps_yes=edge_bps_yes,
            edge_bps_no=edge_bps_no,
            
            # Volatility
            realized_volatility=volatility,
            z_score=z_score,
            sigma_label=sigma_label,
            
            # Liquidity
            yes_bid_depth=liquidity['Up']['bid_depth'],
            yes_ask_depth=liquidity['Up']['ask_depth'],
            no_bid_depth=liquidity['Down']['bid_depth'],
            no_ask_depth=liquidity['Down']['ask_depth'],
            
            # PnL
            unrealized_pnl_yes=pnl_yes,
            unrealized_pnl_no=pnl_no,
            total_unrealized_pnl=total_pnl,
            pnl_percentage=pnl_pct,
            
            # Slippage
            estimated_slippage_yes=slippage_yes,
            estimated_slippage_no=slippage_no,
            
            # Timer
            timer_urgency=urgency,
            seconds_remaining=time_remaining_seconds,
            
            # Regime
            regime_direction=regime_dir,
            regime_strength=regime_strength,
            
            # Oracle lag
            oracle_lag_ms=oracle_lag,
            chainlink_price=cl_price,
            cex_price=cex_price
        )
    
    def reset(self) -> None:
        """Reset analytics state for new market."""
        self._vol_window = VolatilityWindow()
        self._cached_snapshot = None
        self._cache_timestamp = 0.0

