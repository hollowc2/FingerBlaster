"""Comprehensive tests for OrderExecutor (order placement, rate limiting, position management)."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from src.activetrader.engine import OrderExecutor
from src.activetrader.config import AppConfig


class TestOrderExecutorValidation:
    """Test order validation logic."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0  # No rate limit for tests
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_market_order = AsyncMock(return_value={'orderID': 'test_123'})
        connector.create_order = MagicMock(return_value={'orderID': 'test_456'})
        connector.flatten_market = AsyncMock(return_value=[{'orderID': 'flatten_1'}])
        connector.cancel_all_orders = AsyncMock(return_value=5)
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {
            'Up': '0x' + '1' * 64,
            'Down': '0x' + '2' * 64
        }

    @pytest.mark.asyncio
    async def test_execute_order_valid(self, executor, token_map):
        """Test executing valid order."""
        result = await executor.execute_order('Up', 10.0, token_map)

        assert result is not None
        assert result['orderID'] == 'test_123'

    @pytest.mark.asyncio
    async def test_execute_order_invalid_side(self, executor, token_map):
        """Test order with invalid side returns None."""
        result = await executor.execute_order('Invalid', 10.0, token_map)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_order_size_too_small(self, executor, token_map):
        """Test order size below minimum."""
        result = await executor.execute_order('Up', 0.5, token_map)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_order_zero_size(self, executor, token_map):
        """Test order with zero size."""
        result = await executor.execute_order('Up', 0.0, token_map)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_order_negative_size(self, executor, token_map):
        """Test order with negative size."""
        result = await executor.execute_order('Up', -10.0, token_map)

        assert result is None


class TestOrderExecutorRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.5  # 500ms rate limit
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_market_order = AsyncMock(return_value={'orderID': 'test_123'})
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, executor, token_map):
        """Test that rate limit delays are enforced."""
        start_time = time.time()

        # Place two orders rapidly
        await executor.execute_order('Up', 10.0, token_map)
        await executor.execute_order('Up', 10.0, token_map)

        elapsed = time.time() - start_time

        # Second order should wait for rate limit
        assert elapsed >= 0.5  # At least 500ms delay

    @pytest.mark.asyncio
    async def test_rate_limit_allows_after_delay(self, executor, token_map):
        """Test orders allowed after rate limit period."""
        # Place first order
        await executor.execute_order('Up', 10.0, token_map)

        # Wait for rate limit period
        await asyncio.sleep(0.6)

        # Place second order (should not delay)
        start_time = time.time()
        await executor.execute_order('Up', 10.0, token_map)
        elapsed = time.time() - start_time

        # Should execute immediately (no additional delay)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_orders(self, executor, token_map):
        """Test rate limit with concurrent order attempts."""
        # Try to place multiple orders concurrently
        tasks = [
            executor.execute_order('Up', 10.0, token_map)
            for _ in range(3)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # All should succeed
        assert all(r is not None for r in results)

        # Total time should respect rate limiting (3 orders * 0.5s spacing ≈ 1.0s)
        assert elapsed >= 1.0


class TestOrderExecutorMarketOrders:
    """Test market order execution."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_market_order = AsyncMock(return_value={'orderID': 'market_123'})
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_market_order_buy_up(self, executor, token_map, mock_connector):
        """Test market order for Up side."""
        await executor.execute_order('Up', 100.0, token_map, price=None)

        # Verify connector called correctly
        mock_connector.create_market_order.assert_called_once()
        call_args = mock_connector.create_market_order.call_args
        assert call_args[0][0] == token_map['Up']  # Token ID
        assert call_args[0][1] == 100.0  # Size
        assert call_args[0][2] == 'BUY'  # Side

    @pytest.mark.asyncio
    async def test_market_order_buy_down(self, executor, token_map, mock_connector):
        """Test market order for Down side."""
        await executor.execute_order('Down', 50.0, token_map, price=None)

        mock_connector.create_market_order.assert_called_once()
        call_args = mock_connector.create_market_order.call_args
        assert call_args[0][0] == token_map['Down']


class TestOrderExecutorLimitOrders:
    """Test limit order execution."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_order = MagicMock(return_value={'orderID': 'limit_123'})
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_limit_order_with_price(self, executor, token_map):
        """Test limit order with specified price."""
        result = await executor.execute_order('Up', 100.0, token_map, price=0.55)

        assert result is not None
        assert result['orderID'] == 'limit_123'


class TestOrderExecutorPositionManagement:
    """Test position flattening and order cancellation."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.flatten_market = AsyncMock(return_value=[
            {'orderID': 'flatten_up'},
            {'orderID': 'flatten_down'}
        ])
        connector.cancel_all_orders = AsyncMock(return_value=5)
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_flatten_positions(self, executor, token_map, mock_connector):
        """Test flattening all positions."""
        results = await executor.flatten_positions(token_map)

        assert len(results) == 2
        assert results[0]['orderID'] == 'flatten_up'
        assert results[1]['orderID'] == 'flatten_down'
        mock_connector.flatten_market.assert_called_once_with(token_map)

    @pytest.mark.asyncio
    async def test_flatten_positions_no_positions(self, executor, token_map, mock_connector):
        """Test flattening when no positions exist."""
        mock_connector.flatten_market.return_value = []

        results = await executor.flatten_positions(token_map)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_flatten_positions_error(self, executor, token_map, mock_connector):
        """Test flatten handling errors gracefully."""
        mock_connector.flatten_market.side_effect = Exception("Flatten error")

        results = await executor.flatten_positions(token_map)

        # Should return empty list on error
        assert results == []

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, executor, mock_connector):
        """Test canceling all orders."""
        result = await executor.cancel_all_orders()

        assert result is True
        mock_connector.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_orders_none_pending(self, executor, mock_connector):
        """Test canceling when no orders pending."""
        mock_connector.cancel_all_orders.return_value = 0  # Returns falsy

        result = await executor.cancel_all_orders()

        assert result is False  # bool(0) = False


class TestOrderExecutorErrorHandling:
    """Test error handling in order execution."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_order_connector_exception(self, config, token_map):
        """Test order execution when connector raises exception."""
        mock_connector = MagicMock()
        mock_connector.create_market_order = AsyncMock(side_effect=Exception("API error"))

        executor = OrderExecutor(config, mock_connector)
        result = await executor.execute_order('Up', 10.0, token_map)

        # Should return None on exception
        assert result is None

    @pytest.mark.asyncio
    async def test_order_connector_returns_none(self, config, token_map):
        """Test order execution when connector returns None."""
        mock_connector = MagicMock()
        mock_connector.create_market_order = AsyncMock(return_value=None)

        executor = OrderExecutor(config, mock_connector)
        result = await executor.execute_order('Up', 10.0, token_map)

        assert result is None

    @pytest.mark.asyncio
    async def test_order_connector_returns_invalid_response(self, config, token_map):
        """Test order execution with invalid response format."""
        mock_connector = MagicMock()
        mock_connector.create_market_order = AsyncMock(return_value={'status': 'failed'})  # No orderID

        executor = OrderExecutor(config, mock_connector)
        result = await executor.execute_order('Up', 10.0, token_map)

        # Should return None if no orderID
        assert result is None


class TestOrderExecutorEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_market_order = AsyncMock(return_value={'orderID': 'test_123'})
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

    @pytest.mark.asyncio
    async def test_order_exact_minimum_size(self, executor, token_map):
        """Test order at exact minimum size."""
        result = await executor.execute_order('Up', 1.0, token_map)

        assert result is not None

    @pytest.mark.asyncio
    async def test_order_just_below_minimum(self, executor, token_map):
        """Test order just below minimum size."""
        result = await executor.execute_order('Up', 0.999, token_map)

        assert result is None

    @pytest.mark.asyncio
    async def test_order_large_size(self, executor, token_map):
        """Test order with large size."""
        result = await executor.execute_order('Up', 10000.0, token_map)

        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_token_map(self, executor):
        """Test order with empty token map."""
        result = await executor.execute_order('Up', 10.0, {})

        assert result is None

    @pytest.mark.asyncio
    async def test_order_with_missing_token(self, executor):
        """Test order when token not in map."""
        incomplete_map = {'Up': '0x' + '1' * 64}  # Missing 'Down'

        result = await executor.execute_order('Down', 10.0, incomplete_map)

        assert result is None
