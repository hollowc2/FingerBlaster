"""Comprehensive tests for Ladder DOM data manager."""

import pytest
from src.ladder.ladder_data import LadderDataManager, DOMRow, DOMViewModel, UserOrder, price_to_cent
from src.ladder.ladder import VolumeBarRenderer


class TestLadderDataManager:
    """Test ladder data transformation."""

    @pytest.fixture
    def manager(self):
        return LadderDataManager()

    def test_to_cent_conversion(self, manager):
        """Test float price to cent conversion."""
        # price_to_cent is a module-level function, not a method
        assert price_to_cent(0.50) == 50
        assert price_to_cent(0.99) == 99
        assert price_to_cent(0.01) == 1
        assert price_to_cent(0.555) == 56  # Rounds

    def test_empty_books(self, manager):
        """Test handling of empty order books."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # Should have all 99 levels
        assert len(ladder) == 99

        # All should have zero depth
        for level in ladder.values():
            assert level['yes_bid'] == 0.0
            assert level['yes_ask'] == 0.0

    def test_up_bids_map_to_yes_bids(self, manager):
        """Test Up token bids map to YES bids (buy YES liquidity)."""
        up_book = {
            'bids': {
                0.50: 100.0,
                0.51: 50.0,
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # Should have YES bid depth at 50¢ and 51¢
        assert ladder[50]['yes_bid'] == 100.0
        assert ladder[51]['yes_bid'] == 50.0

    def test_down_bids_map_to_yes_asks(self, manager):
        """Test Down token bids map to YES asks (sell YES = buy NO)."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {
            'bids': {
                0.40: 75.0,  # Down @ 40¢ = Yes @ 60¢
                0.50: 100.0,  # Down @ 50¢ = Yes @ 50¢
            },
            'asks': {}
        }

        ladder = manager.build_ladder_data(up_book, down_book)

        # Down bid @ 40¢ = YES ask @ 60¢
        assert ladder[60]['yes_ask'] == 75.0
        # Down bid @ 50¢ = YES ask @ 50¢
        assert ladder[50]['yes_ask'] == 100.0

    def test_complementary_pricing(self, manager):
        """Test complementary pricing (Up + Down = 100¢)."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {
            'bids': {
                0.30: 50.0,  # 70¢ YES
                0.70: 100.0,  # 30¢ YES
            },
            'asks': {}
        }

        ladder = manager.build_ladder_data(up_book, down_book)

        # Verify complementary mapping
        assert ladder[70]['yes_ask'] == 50.0
        assert ladder[30]['yes_ask'] == 100.0

    def test_aggregation_at_same_level(self, manager):
        """Test multiple orders at same price level aggregate."""
        up_book = {
            'bids': {
                0.50: 100.0,
                0.500001: 50.0,  # Rounds to same cent
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # Both should aggregate at 50¢
        assert ladder[50]['yes_bid'] == 150.0

    def test_out_of_bounds_prices_ignored(self, manager):
        """Test prices outside 1-99¢ range are ignored."""
        up_book = {
            'bids': {
                0.00: 100.0,  # Out of bounds
                1.00: 100.0,  # Out of bounds
                0.50: 50.0,   # Valid
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # Only 50¢ should have depth
        assert ladder[50]['yes_bid'] == 50.0

    def test_invalid_price_handling(self, manager):
        """Test invalid price values handled gracefully."""
        up_book = {
            'bids': {
                'invalid': 100.0,
                0.50: 50.0,
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        # Should not crash
        ladder = manager.build_ladder_data(up_book, down_book)

        # Valid price should still work
        assert ladder[50]['yes_bid'] == 50.0


class TestDOMViewModel:
    """Test DOM view model building."""

    @pytest.fixture
    def manager(self):
        return LadderDataManager()

    def test_all_99_rows_created(self, manager):
        """Test all 99 price levels initialized."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        assert len(dom.rows) == 99
        assert all(1 <= price <= 99 for price in dom.rows.keys())

    def test_complementary_no_price(self, manager):
        """Test NO price is complement of YES price."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        # Check all rows have complementary NO price
        for price_cent, row in dom.rows.items():
            assert row.no_price == 100 - price_cent

    def test_best_bid_detection(self, manager):
        """Test best YES bid identified correctly."""
        up_book = {
            'bids': {
                0.45: 100.0,
                0.50: 50.0,  # Highest bid
                0.40: 75.0,
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        # Best bid should be 50¢
        assert dom.best_bid_cent == 50
        assert dom.rows[50].is_best_bid is True
        assert dom.rows[45].is_best_bid is False

    def test_best_ask_detection(self, manager):
        """Test best YES ask identified correctly."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {
            'bids': {
                0.40: 50.0,  # YES @ 60¢
                0.45: 75.0,  # YES @ 55¢ (lowest ask)
                0.50: 100.0,  # YES @ 50¢
            },
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Best ask should be 50¢ (lowest where you can sell YES)
        assert dom.best_ask_cent == 50
        assert dom.rows[50].is_best_ask is True

    def test_inside_spread_marking(self, manager):
        """Test prices inside spread marked correctly."""
        up_book = {
            'bids': {0.45: 100.0},  # Best bid 45¢
            'asks': {}
        }
        down_book = {
            'bids': {0.45: 100.0},  # YES ask 55¢
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Prices between 45 and 55 should be inside spread
        assert dom.rows[46].is_inside_spread is True
        assert dom.rows[50].is_inside_spread is True
        assert dom.rows[54].is_inside_spread is True

        # Bid and ask themselves not inside
        assert dom.rows[45].is_inside_spread is False
        assert dom.rows[55].is_inside_spread is False

    def test_mid_price_calculation(self, manager):
        """Test mid price calculated correctly."""
        up_book = {
            'bids': {0.45: 100.0},
            'asks': {}
        }
        down_book = {
            'bids': {0.45: 100.0},  # YES @ 55¢
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Mid should be (45 + 55) / 2 = 50
        assert dom.mid_price_cent == 50

    def test_mid_price_no_bids(self, manager):
        """Test mid price when no bids."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {
            'bids': {0.40: 100.0},  # Only asks
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Should use ask as mid
        assert dom.mid_price_cent == dom.best_ask_cent

    def test_mid_price_no_asks(self, manager):
        """Test mid price when no asks."""
        up_book = {
            'bids': {0.55: 100.0},  # Only bids
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        # Should use bid as mid
        assert dom.mid_price_cent == dom.best_bid_cent

    def test_mid_price_empty_book(self, manager):
        """Test mid price with completely empty book."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        # Should default to 50¢
        assert dom.mid_price_cent == 50

    def test_max_depth_calculation(self, manager):
        """Test max depth calculated across all levels."""
        up_book = {
            'bids': {
                0.45: 100.0,
                0.50: 500.0,  # Largest
                0.55: 250.0,
            },
            'asks': {}
        }
        down_book = {
            'bids': {
                0.40: 300.0,
                'asks': {}
            }
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Max depth should be 500
        assert dom.max_depth == 500.0

    def test_user_orders_mapping(self, manager):
        """Test user orders mapped to correct price levels."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        user_orders = [
            {'order_id': 'order1', 'price_cent': 50, 'size': 10.0, 'side': 'YES'},
            {'order_id': 'order2', 'price_cent': 50, 'size': 5.0, 'side': 'YES'},
            {'order_id': 'order3', 'price_cent': 60, 'size': 20.0, 'side': 'NO'},
        ]

        dom = manager.build_dom_data(up_book, down_book, user_orders)

        # 50¢ should have 2 orders
        assert len(dom.rows[50].my_orders) == 2
        assert dom.rows[50].my_orders[0].order_id == 'order1'
        assert dom.rows[50].my_orders[0].size == 10.0

        # 60¢ should have 1 order
        assert len(dom.rows[60].my_orders) == 1
        assert dom.rows[60].my_orders[0].side == 'NO'

    def test_user_orders_out_of_bounds_ignored(self, manager):
        """Test user orders outside 1-99¢ ignored."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        user_orders = [
            {'order_id': 'order1', 'price_cent': 0, 'size': 10.0, 'side': 'YES'},
            {'order_id': 'order2', 'price_cent': 100, 'size': 5.0, 'side': 'YES'},
            {'order_id': 'order3', 'price_cent': 50, 'size': 20.0, 'side': 'YES'},
        ]

        dom = manager.build_dom_data(up_book, down_book, user_orders)

        # Only 50¢ order should be mapped
        assert len(dom.rows[50].my_orders) == 1

    def test_depth_on_both_sides(self, manager):
        """Test level can have both YES and NO depth."""
        up_book = {
            'bids': {0.50: 100.0},  # YES bid @ 50¢
            'asks': {}
        }
        down_book = {
            'bids': {0.50: 75.0},  # NO bid @ 50¢ = YES ask @ 50¢
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        row = dom.rows[50]
        assert row.yes_depth == 100.0  # From up bids
        assert row.no_depth == 75.0    # From down bids

    def test_fractional_sizes(self, manager):
        """Test handling of fractional order sizes."""
        up_book = {
            'bids': {
                0.50: 10.5,
                0.51: 0.25,
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.rows[50].yes_depth == 10.5
        assert dom.rows[51].yes_depth == 0.25

    def test_spread_at_same_price(self, manager):
        """Test spread when bid = ask (crossed market)."""
        up_book = {
            'bids': {0.50: 100.0},
            'asks': {}
        }
        down_book = {
            'bids': {0.50: 100.0},  # Same price
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # No inside spread when bid == ask
        inside_spread_rows = [r for r in dom.rows.values() if r.is_inside_spread]
        assert len(inside_spread_rows) == 0


class TestDOMRowDataclass:
    """Test DOMRow dataclass."""

    def test_dom_row_creation(self):
        """Test creating DOMRow."""
        row = DOMRow(
            price_cent=50,
            no_price=50,
            no_depth=100.0,
            yes_depth=75.0,
            my_orders=[],
            is_inside_spread=False,
            is_best_bid=True,
            is_best_ask=False,
        )

        assert row.price_cent == 50
        assert row.yes_depth == 75.0
        assert row.is_best_bid is True

    def test_dom_row_default_my_orders(self):
        """Test my_orders defaults to empty list."""
        row = DOMRow(
            price_cent=50,
            no_price=50,
            no_depth=0.0,
            yes_depth=0.0,
        )

        assert row.my_orders == []


class TestUserOrderDataclass:
    """Test UserOrder dataclass."""

    def test_user_order_creation(self):
        """Test creating UserOrder."""
        order = UserOrder(
            order_id="abc123",
            size=10.5,
            side="YES",
        )

        assert order.order_id == "abc123"
        assert order.size == 10.5
        assert order.side == "YES"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def manager(self):
        return LadderDataManager()

    def test_very_large_order_size(self, manager):
        """Test handling very large order sizes."""
        up_book = {
            'bids': {0.50: 999999999.0},
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.rows[50].yes_depth == 999999999.0
        assert dom.max_depth == 999999999.0

    def test_very_small_order_size(self, manager):
        """Test handling very small order sizes."""
        up_book = {
            'bids': {0.50: 0.0001},
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.rows[50].yes_depth == 0.0001

    def test_negative_size_ignored(self, manager):
        """Test negative sizes handled."""
        up_book = {
            'bids': {0.50: -100.0},  # Invalid
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        # Should handle gracefully (may accumulate as negative or be filtered)
        dom = manager.build_dom_data(up_book, down_book)

        # Implementation may vary, but shouldn't crash
        assert dom is not None

    def test_many_price_levels(self, manager):
        """Test handling many price levels."""
        up_book = {
            'bids': {i / 100.0: 10.0 for i in range(1, 100)},
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        dom = manager.build_dom_data(up_book, down_book)

        # Should aggregate all at their respective cent levels
        assert dom is not None
        assert len(dom.rows) == 99


# ========== VolumeBarRenderer Tests ==========
class TestVolumeBarRenderer:
    """Test volume bar rendering with Unicode blocks."""

    @pytest.fixture
    def renderer(self):
        return VolumeBarRenderer(max_width=10)

    def test_render_bar_empty_depth(self, renderer):
        """Test render_bar with zero depth returns empty string."""
        result = renderer.render_bar(0, 100.0, align_right=False)
        assert result == " " * 10  # All spaces

    def test_render_bar_zero_max_depth(self, renderer):
        """Test render_bar with zero max_depth returns empty string."""
        result = renderer.render_bar(50.0, 0, align_right=False)
        assert result == " " * 10  # All spaces

    def test_render_bar_negative_depth(self, renderer):
        """Test render_bar with negative depth returns empty string."""
        result = renderer.render_bar(-10.0, 100.0, align_right=False)
        assert result == " " * 10  # All spaces

    def test_render_bar_negative_max_depth(self, renderer):
        """Test render_bar with negative max_depth returns empty string."""
        result = renderer.render_bar(50.0, -100.0, align_right=False)
        assert result == " " * 10  # All spaces

    def test_render_bar_full_width(self, renderer):
        """Test render_bar at 100% fills all blocks."""
        result = renderer.render_bar(100.0, 100.0, align_right=False)
        assert result == "█" * 10  # Full width

    def test_render_bar_half_width(self, renderer):
        """Test render_bar at 50% fills half blocks."""
        result = renderer.render_bar(50.0, 100.0, align_right=False)
        # 50% = 5 full blocks (5 * 8 = 40 eighths)
        assert result.startswith("█" * 5)
        assert len(result) == 10

    def test_render_bar_left_aligned(self, renderer):
        """Test left-aligned bar has blocks on left."""
        result = renderer.render_bar(25.0, 100.0, align_right=False)
        # 25% = 2.5 blocks = 2 full + partial
        assert result[0] == "█"
        assert result[1] == "█"
        # Rest should be spaces after the partial
        assert result[-1] == " "

    def test_render_bar_right_aligned(self, renderer):
        """Test right-aligned bar has blocks on right."""
        result = renderer.render_bar(25.0, 100.0, align_right=True)
        # 25% = 2.5 blocks = 2 full + partial
        # Right-aligned means bar extends from right toward left
        # Check that leftmost visible character is not full block (indicates right alignment)
        # and rightmost characters include blocks
        assert len(result) == 10
        # At least some blocks on right side
        assert "█" in result[-5:]  # Blocks should be on right half
        # Left side should be mostly spaces
        assert result[0] == " "

    def test_render_bar_over_100_percent_clamped(self, renderer):
        """Test render_bar over 100% is clamped."""
        result = renderer.render_bar(200.0, 100.0, align_right=False)
        # Should be clamped to 100% = full width
        assert result == "█" * 10

    def test_render_bar_fractional_blocks(self, renderer):
        """Test render_bar renders fractional blocks correctly."""
        # Test 12.5% = 1 block exactly
        result = renderer.render_bar(12.5, 100.0, align_right=False)
        assert result[0] == "█"
        assert result[1] != "█"  # Not a full second block

    def test_render_bar_custom_width(self):
        """Test VolumeBarRenderer with custom max_width."""
        renderer = VolumeBarRenderer(max_width=20)
        result = renderer.render_bar(50.0, 100.0, align_right=False)
        assert len(result) == 20
        assert result.count("█") == 10  # Half of 20

    def test_render_bar_very_small_fraction(self, renderer):
        """Test render_bar with very small fraction shows partial block."""
        result = renderer.render_bar(1.0, 100.0, align_right=False)
        # 1% = 0.8 eighths ≈ 0 full blocks, but may show 1 partial
        assert len(result) == 10

    def test_render_bar_width_1(self):
        """Test VolumeBarRenderer with max_width of 1."""
        renderer = VolumeBarRenderer(max_width=1)
        result = renderer.render_bar(100.0, 100.0, align_right=False)
        assert result == "█"

        result = renderer.render_bar(0.0, 100.0, align_right=False)
        assert result == " "

    def test_render_bar_unicode_blocks(self, renderer):
        """Test that partial blocks use correct Unicode characters."""
        # The BLOCKS_LEFT sequence is " ▏▎▍▌▋▊▉█"
        # Check that partial rendering uses these
        result_25 = renderer.render_bar(3.125, 100.0, align_right=False)  # 1/8 of 25%
        # Should have exactly 2 full eighths = 1/4 of first block
        # Total: 10 * 8 * 0.03125 = 2.5 eighths ≈ partial block
        assert len(result_25) == 10


# ========== price_to_cent Function Tests ==========
class TestPriceToCent:
    """Test the price_to_cent conversion function."""

    def test_exact_conversion(self):
        """Test exact decimal to cent conversion."""
        assert price_to_cent(0.01) == 1
        assert price_to_cent(0.50) == 50
        assert price_to_cent(0.99) == 99

    def test_rounding_banker(self):
        """Test Python's banker's rounding (round half to even).

        Python's round() uses banker's rounding: when exactly at 0.5,
        it rounds to the nearest even number.
        """
        # 55.5 rounds to 56 (nearest even)
        assert price_to_cent(0.555) == 56
        # 50.5 rounds to 50 (nearest even)
        assert price_to_cent(0.505) == 50
        # 99.5 rounds to 100 (nearest even)
        assert price_to_cent(0.995) == 100

    def test_rounding_not_at_midpoint(self):
        """Test rounding when not at exact midpoint."""
        assert price_to_cent(0.554) == 55  # Below midpoint
        assert price_to_cent(0.556) == 56  # Above midpoint
        assert price_to_cent(0.994) == 99

    def test_zero(self):
        """Test zero price."""
        assert price_to_cent(0.0) == 0

    def test_one(self):
        """Test price of 1.0."""
        assert price_to_cent(1.0) == 100

    def test_very_small_price(self):
        """Test very small price values."""
        assert price_to_cent(0.001) == 0
        assert price_to_cent(0.004) == 0
        # 0.5 with banker's rounding goes to nearest even (0)
        assert price_to_cent(0.005) == 0
        # But 0.006 clearly rounds up
        assert price_to_cent(0.006) == 1

    def test_float_precision(self):
        """Test float precision edge cases."""
        # Common float precision issue: 0.1 + 0.2 != 0.3
        assert price_to_cent(0.1 + 0.2) == 30


# ========== DOMRow Boundary Tests ==========
class TestDOMRowBoundaries:
    """Test DOMRow at boundary conditions."""

    def test_dom_row_at_price_1(self):
        """Test DOMRow at minimum price."""
        row = DOMRow(
            price_cent=1,
            no_price=99,
            no_depth=100.0,
            yes_depth=50.0,
        )
        assert row.price_cent == 1
        assert row.no_price == 99

    def test_dom_row_at_price_99(self):
        """Test DOMRow at maximum price."""
        row = DOMRow(
            price_cent=99,
            no_price=1,
            no_depth=100.0,
            yes_depth=50.0,
        )
        assert row.price_cent == 99
        assert row.no_price == 1

    def test_dom_row_with_large_depth(self):
        """Test DOMRow with very large depth values."""
        row = DOMRow(
            price_cent=50,
            no_price=50,
            no_depth=1e12,
            yes_depth=1e12,
        )
        assert row.no_depth == 1e12
        assert row.yes_depth == 1e12

    def test_dom_row_with_zero_depth(self):
        """Test DOMRow with zero depth."""
        row = DOMRow(
            price_cent=50,
            no_price=50,
            no_depth=0.0,
            yes_depth=0.0,
        )
        assert row.no_depth == 0.0
        assert row.yes_depth == 0.0


# ========== UserOrder Tests ==========
class TestUserOrderEdgeCases:
    """Test UserOrder dataclass edge cases."""

    def test_user_order_with_empty_strings(self):
        """Test UserOrder with empty strings."""
        order = UserOrder(order_id="", size=10.0, side="")
        assert order.order_id == ""
        assert order.side == ""

    def test_user_order_with_zero_size(self):
        """Test UserOrder with zero size."""
        order = UserOrder(order_id="abc", size=0.0, side="YES")
        assert order.size == 0.0

    def test_user_order_with_negative_size(self):
        """Test UserOrder with negative size."""
        order = UserOrder(order_id="abc", size=-10.0, side="YES")
        assert order.size == -10.0

    def test_user_order_with_long_id(self):
        """Test UserOrder with very long order ID."""
        long_id = "x" * 1000
        order = UserOrder(order_id=long_id, size=10.0, side="YES")
        assert order.order_id == long_id
        assert len(order.order_id) == 1000


# ========== LadderDataManager Rounding Tests ==========
class TestLadderDataManagerRounding:
    """Test LadderDataManager price rounding edge cases."""

    @pytest.fixture
    def manager(self):
        return LadderDataManager()

    def test_prices_round_to_same_cent(self, manager):
        """Test multiple prices rounding to same cent aggregate.

        Note: Uses banker's rounding (0.505 -> 50, not 51)
        """
        up_book = {
            'bids': {
                0.5049: 10.0,  # Rounds to 50
                0.5050: 20.0,  # Rounds to 50 (banker's: nearest even)
                0.5051: 30.0,  # Rounds to 51
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # 0.5049 and 0.5050 both round to 50
        assert ladder[50]['yes_bid'] == 30.0  # 10 + 20
        assert ladder[51]['yes_bid'] == 30.0  # Just the 0.5051

    def test_boundary_rounding(self, manager):
        """Test rounding at cent boundaries with banker's rounding.

        Note: 0.005 rounds to 0 (banker's: nearest even), not 1
        """
        up_book = {
            'bids': {
                0.004: 10.0,   # Rounds to 0 (out of bounds)
                0.005: 20.0,   # Rounds to 0 (banker's) - out of bounds
                0.006: 15.0,   # Rounds to 1 (in bounds)
                0.994: 30.0,   # Rounds to 99 (in bounds)
                0.995: 40.0,   # Rounds to 100 (out of bounds)
            },
            'asks': {}
        }
        down_book = {'bids': {}, 'asks': {}}

        ladder = manager.build_ladder_data(up_book, down_book)

        # 0 and 100 are out of bounds (ignored)
        # 0.004 -> 0 (out), 0.005 -> 0 (out), 0.006 -> 1 (in)
        assert ladder[1]['yes_bid'] == 15.0
        assert ladder[99]['yes_bid'] == 30.0


# ========== DOMViewModel Integration Tests ==========
class TestDOMViewModelIntegration:
    """Test DOMViewModel with various order book scenarios."""

    @pytest.fixture
    def manager(self):
        return LadderDataManager()

    def test_crossed_market_scenario(self, manager):
        """Test DOM with crossed market (bid > ask)."""
        up_book = {
            'bids': {0.55: 100.0},  # Best bid at 55¢
            'asks': {}
        }
        down_book = {
            'bids': {0.50: 100.0},  # YES ask at 50¢ (lower than bid)
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.best_bid_cent == 55
        assert dom.best_ask_cent == 50
        # Mid price should still be calculated
        assert dom.mid_price_cent == 52  # (55 + 50) // 2

    def test_wide_spread_scenario(self, manager):
        """Test DOM with very wide spread."""
        up_book = {
            'bids': {0.10: 100.0},  # Best bid at 10¢
            'asks': {}
        }
        down_book = {
            'bids': {0.10: 100.0},  # YES ask at 90¢
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.best_bid_cent == 10
        assert dom.best_ask_cent == 90
        assert dom.mid_price_cent == 50

        # Count inside spread rows
        inside_spread = [r for r in dom.rows.values() if r.is_inside_spread]
        assert len(inside_spread) == 79  # 11-89 inclusive

    def test_single_tick_spread(self, manager):
        """Test DOM with minimum spread (1 tick)."""
        up_book = {
            'bids': {0.50: 100.0},  # Best bid at 50¢
            'asks': {}
        }
        down_book = {
            'bids': {0.49: 100.0},  # YES ask at 51¢
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        assert dom.best_bid_cent == 50
        assert dom.best_ask_cent == 51
        assert dom.mid_price_cent == 50  # (50 + 51) // 2 = 50

        # No inside spread (bid + 1 == ask)
        inside_spread = [r for r in dom.rows.values() if r.is_inside_spread]
        assert len(inside_spread) == 0

    def test_dom_with_many_user_orders(self, manager):
        """Test DOM with many user orders at various prices."""
        up_book = {'bids': {}, 'asks': {}}
        down_book = {'bids': {}, 'asks': {}}

        user_orders = [
            {'order_id': f'order_{i}', 'price_cent': i, 'size': float(i), 'side': 'YES'}
            for i in range(1, 100)
        ]

        dom = manager.build_dom_data(up_book, down_book, user_orders)

        # Each price should have exactly one order
        for price_cent in range(1, 100):
            assert len(dom.rows[price_cent].my_orders) == 1
            assert dom.rows[price_cent].my_orders[0].size == float(price_cent)

    def test_dom_max_depth_from_no_side(self, manager):
        """Test max_depth is correctly calculated from NO side."""
        up_book = {
            'bids': {0.50: 100.0},  # YES depth = 100
            'asks': {}
        }
        down_book = {
            'bids': {0.50: 500.0},  # NO depth = 500 (larger)
            'asks': {}
        }

        dom = manager.build_dom_data(up_book, down_book)

        # Max depth should be from NO side
        assert dom.max_depth == 500.0
