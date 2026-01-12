"""Mock PolymarketConnector for testing without real API calls."""

import asyncio
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime


class MockPolymarketConnector:
    """
    Mock PolymarketConnector that simulates Polymarket API behavior without making real API calls.

    Usage:
        connector = MockPolymarketConnector()
        connector.set_market_data([SAMPLE_MARKET])
        markets = await connector.get_active_markets("BTC Up or Down 15m")
    """

    def __init__(self):
        """Initialize mock connector with default state."""
        # Controlled test data
        self.markets: List[Dict] = []
        self.orders: Dict[str, Dict] = {}  # order_id -> order details
        self.positions: Dict[str, float] = {}  # token_id -> position size
        self.balances: Dict[str, float] = {"USDC": 1000.0}  # Default test balance

        # Mock external dependencies
        self.clob_client = MagicMock()
        self.w3 = MagicMock()
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.api_passphrase = "test_passphrase"
        self.private_key = "0x0000000000000000000000000000000000000000000000000000000000000001"

        # Behavior configuration
        self.should_fail_orders = False  # Set to True to simulate order failures
        self.fail_after_n_orders = None  # Fail after N successful orders
        self.order_counter = 0
        self.api_call_delay = 0.0  # Simulate network latency

        # Price mocks
        self.mock_btc_price = 50000.0
        self.mock_strike_price = 50000.0

    def set_market_data(self, markets: List[Dict]):
        """Inject market data for testing."""
        self.markets = markets

    def set_balances(self, balances: Dict[str, float]):
        """Set mock balances."""
        self.balances = balances

    def set_positions(self, positions: Dict[str, float]):
        """Set mock positions."""
        self.positions = positions

    def configure_failure(self, should_fail: bool = True, fail_after_n: Optional[int] = None):
        """Configure order placement failures for testing error handling."""
        self.should_fail_orders = should_fail
        self.fail_after_n_orders = fail_after_n

    async def get_active_markets(self, tag: str) -> List[Dict]:
        """
        Return pre-configured market list filtered by tag.

        Args:
            tag: Market tag to filter by

        Returns:
            List of markets matching the tag
        """
        await asyncio.sleep(self.api_call_delay)
        return [m for m in self.markets if tag in m.get("tags", [])]

    async def get_balances(self, wallet_address: Optional[str] = None) -> Dict[str, float]:
        """
        Return mock balances including positions.

        Returns:
            Dictionary of token -> balance
        """
        await asyncio.sleep(self.api_call_delay)
        combined = self.balances.copy()
        combined.update(self.positions)
        return combined

    async def get_token_balance(self, token_id: str) -> float:
        """
        Get balance for specific token.

        Args:
            token_id: Token ID to query

        Returns:
            Balance for that token
        """
        await asyncio.sleep(self.api_call_delay)
        return self.positions.get(token_id, 0.0)

    async def place_market_order(
        self,
        token_id: str,
        amount: float,
        side: str,
        price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Simulate order placement.

        Args:
            token_id: Token to trade
            amount: Dollar amount
            side: BUY or SELL
            price: Limit price (optional for market orders)

        Returns:
            Order ID if successful, None if failed
        """
        await asyncio.sleep(self.api_call_delay)

        # Check failure conditions
        if self.should_fail_orders:
            return None

        if self.fail_after_n_orders is not None:
            if self.order_counter >= self.fail_after_n_orders:
                return None

        # Validate parameters
        if not token_id or amount <= 0 or side not in ["BUY", "SELL"]:
            return None

        # Check balance for BUY orders
        if side == "BUY" and self.balances.get("USDC", 0) < amount:
            return None

        # Create order
        order_id = f"mock_order_{self.order_counter}"
        self.orders[order_id] = {
            "order_id": order_id,
            "token_id": token_id,
            "amount": amount,
            "side": side,
            "price": price,
            "status": "submitted",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.order_counter += 1

        # Update balances (simulate fill)
        if side == "BUY":
            self.balances["USDC"] -= amount
            # Assume buying at 0.5 for simplicity
            shares = amount / 0.5
            self.positions[token_id] = self.positions.get(token_id, 0.0) + shares
        else:  # SELL
            shares = self.positions.get(token_id, 0.0)
            if shares > 0:
                # Sell all shares
                proceeds = shares * 0.5  # Assume selling at 0.5
                self.balances["USDC"] += proceeds
                self.positions[token_id] = 0.0

        return order_id

    async def create_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        fee_rate_bps: int = 0,
    ) -> Optional[Dict]:
        """
        Create limit order (more detailed than market order).

        Returns:
            Order details if successful
        """
        await asyncio.sleep(self.api_call_delay)

        if self.should_fail_orders:
            return None

        order_id = f"mock_limit_order_{self.order_counter}"
        order = {
            "order_id": order_id,
            "token_id": token_id,
            "side": side,
            "size": size,
            "price": price,
            "fee_rate_bps": fee_rate_bps,
            "status": "open",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.orders[order_id] = order
        self.order_counter += 1
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: Order to cancel

        Returns:
            True if successful
        """
        await asyncio.sleep(self.api_call_delay)

        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        await asyncio.sleep(self.api_call_delay)

        cancelled = 0
        for order in self.orders.values():
            if order["status"] in ["open", "submitted"]:
                order["status"] = "cancelled"
                cancelled += 1
        return cancelled

    async def get_orders(self, market_id: Optional[str] = None) -> List[Dict]:
        """
        Get all orders, optionally filtered by market.

        Args:
            market_id: Optional market filter

        Returns:
            List of orders
        """
        await asyncio.sleep(self.api_call_delay)
        return list(self.orders.values())

    async def get_price_from_binance(self, symbol: str = "BTCUSD") -> Optional[float]:
        """
        Return mock BTC price from Binance.

        Args:
            symbol: Trading pair symbol

        Returns:
            Mock price
        """
        await asyncio.sleep(self.api_call_delay)
        return self.mock_btc_price

    async def get_chainlink_price(self, timestamp: Optional[int] = None) -> Optional[float]:
        """
        Return mock Chainlink price.

        Args:
            timestamp: Optional timestamp for historical price

        Returns:
            Mock price
        """
        await asyncio.sleep(self.api_call_delay)
        return self.mock_btc_price

    def set_btc_price(self, price: float):
        """Set mock BTC price for testing."""
        self.mock_btc_price = price

    def set_strike_price(self, price: float):
        """Set mock strike price for testing."""
        self.mock_strike_price = price

    def get_last_order(self) -> Optional[Dict]:
        """Get the most recently placed order."""
        if not self.orders:
            return None
        last_order_id = f"mock_order_{self.order_counter - 1}"
        return self.orders.get(last_order_id)

    def reset(self):
        """Reset mock state to defaults."""
        self.markets = []
        self.orders = {}
        self.positions = {}
        self.balances = {"USDC": 1000.0}
        self.order_counter = 0
        self.should_fail_orders = False
        self.fail_after_n_orders = None
        self.mock_btc_price = 50000.0
        self.mock_strike_price = 50000.0
