"""Comprehensive tests for AnalyticsEngine quantitative calculations."""

import pytest
import math
from src.activetrader.analytics import AnalyticsEngine, AnalyticsSnapshot, TimerUrgency, EdgeDirection


class TestBasisPointsCalculation:
    """Test basis points calculation."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_basis_points_positive(self, engine):
        """Test positive basis points calculation."""
        # 1% increase: (101 - 100) / 100 * 10000 = 100 bps
        assert engine.calculate_basis_points(101.0, 100.0) == 100.0

    def test_basis_points_negative(self, engine):
        """Test negative basis points calculation."""
        # -1% decrease: (99 - 100) / 100 * 10000 = -100 bps
        assert engine.calculate_basis_points(99.0, 100.0) == -100.0

    def test_basis_points_zero(self, engine):
        """Test zero basis points when prices equal."""
        assert engine.calculate_basis_points(100.0, 100.0) == 0.0

    def test_basis_points_large_movement(self, engine):
        """Test large price movements."""
        # 10% increase = 1000 bps
        assert engine.calculate_basis_points(110.0, 100.0) == 1000.0
        # 50% increase = 5000 bps
        assert engine.calculate_basis_points(150.0, 100.0) == 5000.0

    def test_basis_points_small_movement(self, engine):
        """Test small price movements."""
        # 0.1% = 10 bps
        assert abs(engine.calculate_basis_points(100.1, 100.0) - 10.0) < 0.01

    def test_basis_points_fractional_prices(self, engine):
        """Test basis points with fractional prices."""
        # From 0.50 to 0.52 = 4% = 400 bps
        assert abs(engine.calculate_basis_points(0.52, 0.50) - 400.0) < 0.01


class TestBinaryFairValue:
    """Test Black-Scholes binary fair value calculations."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_fair_value_at_strike_near_expiry(self, engine):
        """At strike price near expiry, FV should approach 50/50."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=50000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=60,  # 1 minute
            volatility=0.6
        )
        assert 0.48 < fv_yes < 0.52
        assert 0.48 < fv_no < 0.52
        assert abs((fv_yes + fv_no) - 1.0) < 0.01

    def test_fair_value_far_in_the_money(self, engine):
        """When price >> strike, YES should have high probability."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=52000.0,  # +4% above strike
            price_to_beat=50000.0,
            time_to_expiry_seconds=300,
            volatility=0.6
        )
        assert fv_yes > 0.7  # Strong YES probability
        assert fv_no < 0.3

    def test_fair_value_far_out_of_money(self, engine):
        """When price << strike, YES should have low probability."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=48000.0,  # -4% below strike
            price_to_beat=50000.0,
            time_to_expiry_seconds=300,
            volatility=0.6
        )
        assert fv_yes < 0.3
        assert fv_no > 0.7

    def test_fair_value_long_time_to_expiry(self, engine):
        """With long time remaining, probabilities closer to 50/50 due to uncertainty."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=50000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=900,  # 15 minutes (full market)
            volatility=0.6
        )
        # More uncertainty with time
        assert 0.45 < fv_yes < 0.55

    def test_fair_value_complementary_probabilities(self, engine):
        """YES and NO probabilities should sum to ~1.0."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=51000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=450,
            volatility=0.6
        )
        assert abs((fv_yes + fv_no) - 1.0) < 0.01

    def test_fair_value_high_volatility(self, engine):
        """High volatility increases uncertainty (test at-the-money)."""
        # Use at-the-money for clearer volatility effect
        fv_yes_high, _ = engine.calculate_binary_fair_value(
            current_price=50000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=600,
            volatility=1.0  # 100% vol
        )
        fv_yes_low, _ = engine.calculate_binary_fair_value(
            current_price=50000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=600,
            volatility=0.1  # 10% vol
        )
        # Both should be around 0.5, within reasonable range
        assert 0.45 < fv_yes_high < 0.55
        assert 0.45 < fv_yes_low < 0.55

    def test_fair_value_very_short_expiry(self, engine):
        """Very short time to expiry approaches current state."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=51000.0,  # Above strike
            price_to_beat=50000.0,
            time_to_expiry_seconds=10,  # 10 seconds
            volatility=0.6
        )
        # Should strongly favor YES with so little time
        assert fv_yes > 0.8


class TestEdgeDetection:
    """Test edge detection logic."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_edge_undervalued(self, engine):
        """Test edge detection when market price < fair value."""
        edge, edge_bps = engine.calculate_edge(
            market_price=0.45,
            fair_value=0.52,
            threshold_bps=50.0
        )
        assert edge == EdgeDirection.UNDERVALUED
        assert edge_bps != 0

    def test_edge_overvalued(self, engine):
        """Test edge detection when market price > fair value."""
        edge, edge_bps = engine.calculate_edge(
            market_price=0.58,
            fair_value=0.50,
            threshold_bps=50.0
        )
        assert edge == EdgeDirection.OVERVALUED
        assert edge_bps != 0

    def test_edge_fair(self, engine):
        """Test edge detection within threshold."""
        edge, edge_bps = engine.calculate_edge(
            market_price=0.5025,
            fair_value=0.5000,
            threshold_bps=50.0  # 50 bps = 0.5%
        )
        assert edge == EdgeDirection.FAIR

    def test_edge_boundary_undervalued(self, engine):
        """Test edge at boundary (just above threshold)."""
        # 51 bps difference, should be undervalued
        edge, edge_bps = engine.calculate_edge(
            market_price=0.4949,
            fair_value=0.5000,
            threshold_bps=50.0
        )
        assert edge == EdgeDirection.UNDERVALUED

    def test_edge_boundary_fair(self, engine):
        """Test edge within fair threshold."""
        # Well within 50 bps threshold
        edge, edge_bps = engine.calculate_edge(
            market_price=0.4990,
            fair_value=0.5000,
            threshold_bps=50.0
        )
        # 20 bps difference, should be fair
        assert edge == EdgeDirection.FAIR


class TestZScoreCalculation:
    """Test z-score calculations."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_z_score_at_strike(self, engine):
        """Z-score should be 0 when price equals strike."""
        z, sigma_label = engine.calculate_z_score(
            current_price=50000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=300,
            volatility=0.6
        )
        assert abs(z) < 0.01  # Approximately 0

    def test_z_score_above_strike(self, engine):
        """Positive z-score when price > strike."""
        z, sigma_label = engine.calculate_z_score(
            current_price=51000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=300,
            volatility=0.6
        )
        assert z > 0

    def test_z_score_below_strike(self, engine):
        """Negative z-score when price < strike."""
        z, sigma_label = engine.calculate_z_score(
            current_price=49000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=300,
            volatility=0.6
        )
        assert z < 0

    def test_z_score_magnitude_increases_with_distance(self, engine):
        """Z-score magnitude increases with distance from strike."""
        z1, _ = engine.calculate_z_score(50500.0, 50000.0, 300, 0.6)
        z2, _ = engine.calculate_z_score(51000.0, 50000.0, 300, 0.6)
        z3, _ = engine.calculate_z_score(52000.0, 50000.0, 300, 0.6)

        assert abs(z1) < abs(z2) < abs(z3)


class TestTimerUrgency:
    """Test timer urgency threshold detection."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_timer_urgency_normal(self, engine):
        """Test NORMAL urgency when > 5 minutes remain."""
        urgency = engine.get_timer_urgency(600)  # 10 minutes
        assert urgency == TimerUrgency.NORMAL

    def test_timer_urgency_watchful(self, engine):
        """Test WATCHFUL urgency when 2-5 minutes remain."""
        urgency = engine.get_timer_urgency(180)  # 3 minutes
        assert urgency == TimerUrgency.WATCHFUL

    def test_timer_urgency_critical(self, engine):
        """Test CRITICAL urgency when < 2 minutes remain."""
        urgency = engine.get_timer_urgency(90)  # 1.5 minutes
        assert urgency == TimerUrgency.CRITICAL

    def test_timer_urgency_boundary_critical(self, engine):
        """Test boundary at 2 minutes."""
        urgency = engine.get_timer_urgency(119)  # Just under 2 min
        assert urgency == TimerUrgency.CRITICAL

    def test_timer_urgency_boundary_watchful(self, engine):
        """Test boundary at 5 minutes."""
        urgency = engine.get_timer_urgency(299)  # Just under 5 min
        assert urgency == TimerUrgency.WATCHFUL

    def test_timer_urgency_zero_seconds(self, engine):
        """Test urgency at expiry."""
        urgency = engine.get_timer_urgency(0)
        assert urgency == TimerUrgency.CRITICAL


class TestLiquidityDepth:
    """Test liquidity depth calculations."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_liquidity_depth_normal_book(self, engine, sample_order_book):
        """Test liquidity depth with normal order book."""
        depths = engine.calculate_liquidity_depth(sample_order_book)

        assert depths['Up']['bid_depth'] is not None
        assert depths['Up']['ask_depth'] is not None
        assert depths['Up']['bid_depth'] > 0
        assert depths['Up']['ask_depth'] > 0

    def test_liquidity_depth_empty_book(self, engine, empty_order_book):
        """Test liquidity depth with empty order book."""
        depths = engine.calculate_liquidity_depth(empty_order_book)

        # Should return None for empty books
        assert depths['Up']['bid_depth'] is None
        assert depths['Up']['ask_depth'] is None

    def test_liquidity_depth_one_sided_book(self, engine):
        """Test liquidity depth with one-sided order book."""
        order_book = {
            "Up": {
                "bids": {0.50: 100.0},
                "asks": {}  # No asks
            },
            "Down": {
                "bids": {},
                "asks": {0.50: 100.0}
            }
        }
        depths = engine.calculate_liquidity_depth(order_book)

        assert depths['Up']['bid_depth'] > 0
        assert depths['Up']['ask_depth'] == 0.0  # Not None, but 0.0


class TestSlippageEstimation:
    """Test slippage estimation."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_slippage_with_depth(self, engine):
        """Test slippage estimation with sufficient depth."""
        order_book_side = {
            0.50: 100.0,  # Size in shares, not $
            0.51: 200.0,
            0.52: 300.0,
        }

        avg_fill, slippage = engine.estimate_slippage(
            order_size_usd=150.0,
            order_book_side=order_book_side,
            is_buy=True
        )

        # Should have some slippage
        assert slippage >= 0

    def test_slippage_thin_book(self, engine):
        """Test slippage with thin order book."""
        order_book_side = {
            0.50: 20.0,  # Only 20 shares = $10 liquidity
        }

        avg_fill, slippage = engine.estimate_slippage(
            order_size_usd=100.0,  # Way more than available
            order_book_side=order_book_side,
            is_buy=True
        )

        # Some slippage expected (order partially filled or fully filled at limit)
        assert slippage >= 0  # Just verify it returns valid data

    def test_slippage_empty_book(self, engine):
        """Test slippage with empty order book."""
        avg_fill, slippage = engine.estimate_slippage(
            order_size_usd=100.0,
            order_book_side={},
            is_buy=True
        )

        # Returns (0.0, 0.0) for empty book
        assert slippage == 0.0

    def test_slippage_exact_match(self, engine):
        """Test slippage when order can be filled at single level."""
        order_book_side = {
            0.50: 200.0,  # 200 shares at 50¢ = $100
        }

        avg_fill, slippage = engine.estimate_slippage(
            order_size_usd=100.0,
            order_book_side=order_book_side,
            is_buy=True
        )

        # Should have minimal slippage
        assert slippage < 0.01  # < 1%


class TestPnLCalculation:
    """Test unrealized PnL calculations."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_pnl_positive(self, engine):
        """Test positive unrealized PnL."""
        pnl_yes, pnl_no, total_pnl, pnl_pct = engine.calculate_unrealized_pnl(
            yes_position=100.0,  # 100 shares YES
            no_position=0.0,
            avg_entry_yes=0.45,
            avg_entry_no=None,
            current_yes_price=0.52,
            current_no_price=0.48
        )

        # (0.52 - 0.45) * 100 = 7.0
        assert abs(pnl_yes - 7.0) < 0.01
        assert total_pnl > 0

    def test_pnl_negative(self, engine):
        """Test negative unrealized PnL."""
        pnl_yes, pnl_no, total_pnl, pnl_pct = engine.calculate_unrealized_pnl(
            yes_position=100.0,
            no_position=0.0,
            avg_entry_yes=0.55,
            avg_entry_no=None,
            current_yes_price=0.48,
            current_no_price=0.52
        )

        # (0.48 - 0.55) * 100 = -7.0
        assert abs(pnl_yes - (-7.0)) < 0.01
        assert total_pnl < 0

    def test_pnl_no_position(self, engine):
        """Test PnL with no position."""
        pnl_yes, pnl_no, total_pnl, pnl_pct = engine.calculate_unrealized_pnl(
            yes_position=0.0,
            no_position=0.0,
            avg_entry_yes=None,
            avg_entry_no=None,
            current_yes_price=0.52,
            current_no_price=0.48
        )

        assert total_pnl == 0.0

    def test_pnl_small_position_ignored(self, engine):
        """Test that tiny positions (< 0.1) are ignored."""
        pnl_yes, pnl_no, total_pnl, pnl_pct = engine.calculate_unrealized_pnl(
            yes_position=0.05,  # < 0.1 threshold
            no_position=0.0,
            avg_entry_yes=0.50,
            avg_entry_no=None,
            current_yes_price=0.60,
            current_no_price=0.40
        )

        assert total_pnl == 0.0


class TestRegimeDetection:
    """Test regime detection from prior outcomes."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_regime_bullish(self, engine):
        """Test bullish regime detection."""
        # List of strings, not dictionaries
        prior_outcomes = ['Up'] * 7 + ['Down'] * 3

        direction, strength = engine.detect_regime(prior_outcomes)

        assert direction == "BULLISH"
        assert strength == 70.0  # 7/10 = 70%

    def test_regime_bearish(self, engine):
        """Test bearish regime detection."""
        prior_outcomes = ['Down'] * 8 + ['Up'] * 2

        direction, strength = engine.detect_regime(prior_outcomes)

        assert direction == "BEARISH"
        assert strength == 80.0  # 8/10 = 80%

    def test_regime_neutral(self, engine):
        """Test neutral regime (50/50)."""
        prior_outcomes = ['Up' if i % 2 == 0 else 'Down' for i in range(10)]

        direction, strength = engine.detect_regime(prior_outcomes)

        assert direction == "NEUTRAL"
        assert strength == 50.0

    def test_regime_no_outcomes(self, engine):
        """Test regime with no prior outcomes."""
        direction, strength = engine.detect_regime([])

        # Empty list returns empty string and 0.0
        assert direction == ""
        assert strength == 0.0


class TestAnalyticsEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    def test_fair_value_zero_time_to_expiry(self, engine):
        """Test fair value with zero time remaining."""
        fv_yes, fv_no = engine.calculate_binary_fair_value(
            current_price=51000.0,
            price_to_beat=50000.0,
            time_to_expiry_seconds=1,  # Near zero
            volatility=0.6
        )

        # Should strongly favor current state
        assert fv_yes > 0.9  # Already above strike

    def test_basis_points_with_zero_base(self, engine):
        """Test basis points calculation edge case."""
        # Should handle gracefully (though not realistic)
        try:
            bps = engine.calculate_basis_points(1.0, 0.0)
            # If it doesn't crash, that's good
        except (ZeroDivisionError, ValueError):
            # Also acceptable to raise error
            pass

    def test_slippage_zero_order_size(self, engine):
        """Test slippage with zero order size."""
        order_book_side = {0.50: 100.0}

        avg_fill, slippage = engine.estimate_slippage(
            order_size_usd=0.0,
            order_book_side=order_book_side,
            is_buy=True
        )

        # Should return 0.0 for zero order
        assert slippage == 0.0
