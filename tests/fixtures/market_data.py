"""Sample market data fixtures for testing."""

import pandas as pd
from typing import Dict, Any, List

# Sample market structure matching Polymarket API response
SAMPLE_MARKET = {
    "id": "test_market_12345",
    "question": "Will BTC be up or down in 15 minutes?",
    "market_slug": "btc-up-or-down-15m-test",
    "end_date_iso": "2026-01-11T12:15:00Z",
    "game_start_time": "2026-01-11T12:00:00Z",
    "seconds_delay": 0,
    "active": True,
    "closed": False,
    "archived": False,
    "new": False,
    "featured": False,
    "submitted_by": "Polymarket",
    "slug": "btc-up-or-down-15m-test",
    "end_date": "2026-01-11T12:15:00Z",
    "tags": ["BTC Up or Down 15m"],
    "clob_token_ids": [
        "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",  # Up token
        "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",  # Down token
    ],
    "tokens": [
        {
            "token_id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "outcome": "Up",
            "price": 0.52,
        },
        {
            "token_id": "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
            "outcome": "Down",
            "price": 0.48,
        },
    ],
    "market_type": "binary",
    "enable_order_book": True,
    "description": "This market will resolve to \"Up\" if BTC price is higher at end time, \"Down\" otherwise.",
}

# Sample order book with realistic depth
SAMPLE_ORDERBOOK = {
    "Up": {
        "bids": {
            0.51: 100.0,  # $100 at 51¢
            0.50: 250.0,  # $250 at 50¢
            0.49: 500.0,  # $500 at 49¢
            0.48: 750.0,  # $750 at 48¢
            0.47: 1000.0,  # $1000 at 47¢
        },
        "asks": {
            0.52: 150.0,  # $150 at 52¢
            0.53: 300.0,  # $300 at 53¢
            0.54: 450.0,  # $450 at 54¢
            0.55: 600.0,  # $600 at 55¢
            0.56: 900.0,  # $900 at 56¢
        },
    },
    "Down": {
        "bids": {
            0.47: 150.0,
            0.46: 300.0,
            0.45: 450.0,
            0.44: 600.0,
            0.43: 900.0,
        },
        "asks": {
            0.48: 100.0,
            0.49: 250.0,
            0.50: 500.0,
            0.51: 750.0,
            0.52: 1000.0,
        },
    },
}

# Thin order book for slippage testing
THIN_ORDERBOOK = {
    "Up": {
        "bids": {
            0.50: 10.0,  # Only $10 liquidity
        },
        "asks": {
            0.52: 15.0,  # Only $15 liquidity
        },
    },
    "Down": {
        "bids": {
            0.48: 15.0,
        },
        "asks": {
            0.50: 10.0,
        },
    },
}

# Empty order book for edge case testing
EMPTY_ORDERBOOK = {
    "Up": {
        "bids": {},
        "asks": {},
    },
    "Down": {
        "bids": {},
        "asks": {},
    },
}

# Sample WebSocket order book update message (CLOB format)
SAMPLE_WS_ORDERBOOK_UPDATE = {
    "event_type": "book",
    "market": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "asset_id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "hash": "0xabcdef1234567890",
    "price": "0.52",
    "size": "150.0",
    "side": "BUY",
    "timestamp": 1704960000,
}

# Sample RTDS WebSocket message (BTC price update)
SAMPLE_RTDS_MESSAGE = {
    "event_type": "price_update",
    "asset": "BTC",
    "price": 50000.0,
    "timestamp": 1704960000,
    "source": "chainlink",
}


def generate_market_with_times(
    start_time: str = "2026-01-11T12:00:00Z",
    end_time: str = "2026-01-11T12:15:00Z",
    market_id: str = "test_market_12345",
) -> Dict[str, Any]:
    """Generate a market with custom start/end times."""
    market = SAMPLE_MARKET.copy()
    market["id"] = market_id
    market["game_start_time"] = start_time
    market["end_date_iso"] = end_time
    market["end_date"] = end_time
    return market


def generate_orderbook(
    up_mid: float = 0.52,
    spread: float = 0.02,
    depth_per_level: float = 100.0,
    num_levels: int = 5,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Generate a synthetic order book with configurable parameters.

    Args:
        up_mid: Mid price for Up side
        spread: Bid-ask spread
        depth_per_level: Dollar depth at each price level
        num_levels: Number of price levels on each side

    Returns:
        Order book dictionary
    """
    half_spread = spread / 2
    tick = 0.01  # 1 cent price increment

    up_bids = {}
    up_asks = {}

    # Generate Up side bids (below mid)
    for i in range(num_levels):
        price = round(up_mid - half_spread - (i * tick), 2)
        if price > 0:
            up_bids[price] = depth_per_level * (i + 1)

    # Generate Up side asks (above mid)
    for i in range(num_levels):
        price = round(up_mid + half_spread + (i * tick), 2)
        if price < 1.0:
            up_asks[price] = depth_per_level * (i + 1)

    # Down side is complement of Up
    down_mid = 1.0 - up_mid
    down_bids = {}
    down_asks = {}

    for i in range(num_levels):
        price = round(down_mid - half_spread - (i * tick), 2)
        if price > 0:
            down_bids[price] = depth_per_level * (i + 1)

    for i in range(num_levels):
        price = round(down_mid + half_spread + (i * tick), 2)
        if price < 1.0:
            down_asks[price] = depth_per_level * (i + 1)

    return {
        "Up": {"bids": up_bids, "asks": up_asks},
        "Down": {"bids": down_bids, "asks": down_asks},
    }


def generate_prior_outcomes(num_outcomes: int = 10) -> List[Dict[str, Any]]:
    """
    Generate prior market outcomes for regime detection testing.

    Args:
        num_outcomes: Number of prior outcomes to generate

    Returns:
        List of prior outcome dictionaries
    """
    outcomes = []
    base_time = pd.Timestamp("2026-01-11T10:00:00Z")
    base_strike = 50000.0

    for i in range(num_outcomes):
        # Alternate between Up and Down outcomes
        outcome = "Up" if i % 2 == 0 else "Down"
        strike = base_strike + (i * 100)  # Slight upward drift
        actual = strike + 50 if outcome == "Up" else strike - 50

        outcomes.append({
            "end_time": base_time + pd.Timedelta(minutes=15 * (i + 1)),
            "strike_price": strike,
            "actual_price": actual,
            "outcome": outcome,
            "market_id": f"prior_market_{i}",
        })

    return outcomes
