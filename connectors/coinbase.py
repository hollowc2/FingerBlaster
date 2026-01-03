"""
Coinbase Advanced Trade API Connector.

Provides REST and WebSocket access to Coinbase Advanced Trade API for:
- Historical candlestick data (multiple timeframes)
- Real-time order book (L2) data
- Real-time trades (market_trades)
- 24h ticker statistics

This connector is designed to be reusable by any module (e.g., Pulse)
and follows the same patterns as polymarket.py.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from connectors.async_http_mixin import AsyncHttpFetcherMixin
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Pulse.CoinbaseConnector")


class CoinbaseGranularity(Enum):
    """Coinbase candle granularity values."""
    ONE_MINUTE = "ONE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    TWO_HOUR = "TWO_HOUR"
    SIX_HOUR = "SIX_HOUR"
    ONE_DAY = "ONE_DAY"

    @property
    def seconds(self) -> int:
        """Return granularity in seconds."""
        mapping = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }
        return mapping[self.value]


@dataclass
class CoinbaseConfig:
    """Configuration for Coinbase connector."""

    # API endpoints
    rest_base_url: str = "https://api.coinbase.com/api/v3/brokerage"
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"

    # Rate limiting
    rest_rate_limit_per_sec: int = 10
    rest_request_timeout: int = 10

    # WebSocket settings
    ws_reconnect_delay: int = 5
    ws_max_reconnect_attempts: int = 10
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    ws_message_size_limit: int = 10 * 1024 * 1024  # 10MB

    # Default product
    default_product_id: str = "BTC-USD"


class CoinbaseWebSocketManager:
    """
    Manages WebSocket connection to Coinbase Advanced Trade.

    Handles:
    - level2 (order book)
    - market_trades (individual trades)
    - ticker (24h stats)

    Features:
    - JWT authentication per message
    - Auto-reconnect with exponential backoff
    - Callback-based message dispatch
    """

    def __init__(
        self,
        config: CoinbaseConfig,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        on_trade: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        on_l2_update: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        on_ticker: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        on_connection_status: Optional[Callable[[bool, str], Coroutine]] = None,
    ):
        """
        Initialize WebSocket manager.

        Args:
            config: Coinbase configuration
            api_key: Coinbase API key (optional for public data)
            api_secret: Coinbase API secret (optional for public data)
            on_trade: Callback for trade messages
            on_l2_update: Callback for L2 order book updates
            on_ticker: Callback for ticker updates
            on_connection_status: Callback for connection status changes
        """
        self.config = config
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")

        self.on_trade = on_trade
        self.on_l2_update = on_l2_update
        self.on_ticker = on_ticker
        self.on_connection_status = on_connection_status

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_attempts = 0
        self._subscribed_products: Set[str] = set()
        self._subscribed_channels: Set[str] = set()
        self._lock = asyncio.Lock()
        self._connect_task: Optional[asyncio.Task] = None

    def _generate_signature(self, timestamp: str, channel: str, product_ids: List[str]) -> str:
        """
        Generate HMAC-SHA256 signature for WebSocket authentication.

        Args:
            timestamp: Unix timestamp as string
            channel: Channel name
            product_ids: List of product IDs

        Returns:
            Hex-encoded signature
        """
        if not self.api_secret:
            return ""

        message = f"{timestamp}{channel}{','.join(product_ids)}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _build_subscribe_message(
        self,
        channel: str,
        product_ids: List[str],
        subscribe: bool = True
    ) -> Dict[str, Any]:
        """
        Build subscribe/unsubscribe message with authentication.

        Args:
            channel: Channel name (level2, market_trades, ticker)
            product_ids: List of product IDs
            subscribe: True to subscribe, False to unsubscribe

        Returns:
            Message dictionary
        """
        timestamp = str(int(time.time()))

        message = {
            "type": "subscribe" if subscribe else "unsubscribe",
            "product_ids": product_ids,
            "channel": channel,
        }

        # Add authentication if credentials available
        if self.api_key and self.api_secret:
            message["api_key"] = self.api_key
            message["timestamp"] = timestamp
            message["signature"] = self._generate_signature(timestamp, channel, product_ids)

        return message

    async def start(self, product_ids: Optional[List[str]] = None, channels: Optional[List[str]] = None):
        """
        Start WebSocket connection and subscribe to channels.

        Args:
            product_ids: Product IDs to subscribe to (default: BTC-USD)
            channels: Channels to subscribe to (default: all)
        """
        if self._running:
            logger.warning("WebSocket already running")
            return

        self._running = True
        self._subscribed_products = set(product_ids or [self.config.default_product_id])
        self._subscribed_channels = set(channels or ["level2", "market_trades", "ticker"])

        self._connect_task = asyncio.create_task(self._connect_loop())

    async def stop(self):
        """Stop WebSocket connection."""
        self._running = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass

    async def subscribe(self, product_ids: List[str], channels: List[str]):
        """
        Subscribe to additional products/channels.

        Args:
            product_ids: Product IDs to subscribe to
            channels: Channels to subscribe to
        """
        async with self._lock:
            if not self._ws:
                logger.warning("WebSocket not connected, queueing subscription")
                self._subscribed_products.update(product_ids)
                self._subscribed_channels.update(channels)
                return

            for channel in channels:
                try:
                    msg = self._build_subscribe_message(channel, product_ids, subscribe=True)
                    await self._ws.send(json.dumps(msg))
                    logger.info(f"Subscribed to {channel} for {product_ids}")
                except Exception as e:
                    logger.error(f"Error subscribing to {channel}: {e}")

            self._subscribed_products.update(product_ids)
            self._subscribed_channels.update(channels)

    async def unsubscribe(self, product_ids: List[str], channels: List[str]):
        """
        Unsubscribe from products/channels.

        Args:
            product_ids: Product IDs to unsubscribe from
            channels: Channels to unsubscribe from
        """
        async with self._lock:
            if not self._ws:
                return

            for channel in channels:
                try:
                    msg = self._build_subscribe_message(channel, product_ids, subscribe=False)
                    await self._ws.send(json.dumps(msg))
                    logger.info(f"Unsubscribed from {channel} for {product_ids}")
                except Exception as e:
                    logger.error(f"Error unsubscribing from {channel}: {e}")

            self._subscribed_products -= set(product_ids)
            self._subscribed_channels -= set(channels)

    async def _connect_loop(self):
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except InvalidStatusCode as e:
                logger.error(f"WebSocket invalid status: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if not self._running:
                break

            # Reconnect with exponential backoff
            self._reconnect_attempts += 1
            if self._reconnect_attempts > self.config.ws_max_reconnect_attempts:
                logger.error("Max reconnection attempts reached")
                if self.on_connection_status:
                    await self.on_connection_status(False, "Max reconnection attempts reached")
                break

            wait_time = self.config.ws_reconnect_delay * (2 ** min(self._reconnect_attempts - 1, 3))
            logger.info(f"Reconnecting in {wait_time}s (attempt {self._reconnect_attempts})")
            await asyncio.sleep(wait_time)

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages."""
        logger.info(f"Connecting to Coinbase WebSocket: {self.config.ws_url}")

        async with websockets.connect(
            self.config.ws_url,
            ping_interval=self.config.ws_ping_interval,
            ping_timeout=self.config.ws_ping_timeout,
            max_size=self.config.ws_message_size_limit,
        ) as ws:
            self._ws = ws
            self._reconnect_attempts = 0

            logger.info("WebSocket connected")
            if self.on_connection_status:
                await self.on_connection_status(True, "Connected")

            # Subscribe to channels
            for channel in self._subscribed_channels:
                msg = self._build_subscribe_message(
                    channel, list(self._subscribed_products), subscribe=True
                )
                await ws.send(json.dumps(msg))
                logger.info(f"Sent subscription for {channel}")

            # Process messages
            async for message in ws:
                await self._process_message(message)

    async def _process_message(self, raw_message: str):
        """
        Process incoming WebSocket message.

        Args:
            raw_message: Raw JSON message string
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON message: {e}")
            return

        channel = data.get("channel")
        msg_type = data.get("type")

        # Handle subscription confirmations
        if msg_type == "subscriptions":
            logger.debug(f"Subscription confirmed: {data.get('channels', [])}")
            return

        # Handle errors
        if msg_type == "error":
            logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
            return

        # Dispatch to appropriate callback
        events = data.get("events", [])
        for event in events:
            event_type = event.get("type")

            if channel == "l2_data" or channel == "level2":
                if self.on_l2_update:
                    try:
                        await self.on_l2_update(event)
                    except Exception as e:
                        logger.error(f"Error in L2 callback: {e}")

            elif channel == "market_trades":
                if self.on_trade:
                    trades = event.get("trades", [])
                    for trade in trades:
                        try:
                            await self.on_trade(trade)
                        except Exception as e:
                            logger.error(f"Error in trade callback: {e}")

            elif channel == "ticker":
                if self.on_ticker:
                    tickers = event.get("tickers", [])
                    for ticker in tickers:
                        try:
                            await self.on_ticker(ticker)
                        except Exception as e:
                            logger.error(f"Error in ticker callback: {e}")


class CoinbaseConnector(AsyncHttpFetcherMixin):
    """
    Coinbase Advanced Trade API Connector.

    Provides:
    - REST API access for historical data (candles)
    - WebSocket for real-time data (trades, order book, ticker)

    Usage:
        connector = CoinbaseConnector()
        await connector.start()

        # Get historical candles
        candles = await connector.get_candles("BTC-USD", CoinbaseGranularity.ONE_MINUTE)

        # Register callbacks for real-time data
        connector.on_trade = my_trade_handler

        await connector.stop()
    """

    def __init__(
        self,
        config: Optional[CoinbaseConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize Coinbase connector.

        Args:
            config: Optional configuration (uses defaults if None)
            api_key: Coinbase API key (reads from env if None)
            api_secret: Coinbase API secret (reads from env if None)
        """
        self.config = config or CoinbaseConfig()
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")

        # Async session (lazy initialization)
        self.async_session: Optional[aiohttp.ClientSession] = None

        # WebSocket manager
        self._ws_manager: Optional[CoinbaseWebSocketManager] = None

        # Rate limiting
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()

        # Callbacks for WebSocket data
        self.on_trade: Optional[Callable[[Dict[str, Any]], Coroutine]] = None
        self.on_l2_update: Optional[Callable[[Dict[str, Any]], Coroutine]] = None
        self.on_ticker: Optional[Callable[[Dict[str, Any]], Coroutine]] = None
        self.on_connection_status: Optional[Callable[[bool, str], Coroutine]] = None

    async def _ensure_async_session(self):
        """Lazy initialization of async session."""
        if self.async_session is None or self.async_session._client.closed:
            self.async_session = await self._create_async_session(max_retries=3)

    async def _rate_limit(self):
        """Enforce rate limiting for REST requests."""
        async with self._request_lock:
            now = time.time()
            min_interval = 1.0 / self.config.rest_rate_limit_per_sec
            elapsed = now - self._last_request_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time = time.time()

    def _generate_auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Generate authentication headers for REST API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            body: Request body (for POST)

        Returns:
            Headers dictionary
        """
        if not self.api_key or not self.api_secret:
            return {}

        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}{body}"

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def start(
        self,
        product_ids: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        start_websocket: bool = True
    ):
        """
        Start the connector.

        Args:
            product_ids: Product IDs to subscribe to (default: BTC-USD)
            channels: WebSocket channels (default: all)
            start_websocket: Whether to start WebSocket connection
        """
        await self._ensure_async_session()

        if start_websocket:
            self._ws_manager = CoinbaseWebSocketManager(
                config=self.config,
                api_key=self.api_key,
                api_secret=self.api_secret,
                on_trade=self._handle_trade,
                on_l2_update=self._handle_l2_update,
                on_ticker=self._handle_ticker,
                on_connection_status=self._handle_connection_status,
            )
            await self._ws_manager.start(product_ids, channels)

    async def stop(self):
        """Stop the connector and cleanup resources."""
        if self._ws_manager:
            await self._ws_manager.stop()
            self._ws_manager = None

        if self.async_session and not self.async_session._client.closed:
            await self.async_session.close()

    async def _handle_trade(self, trade: Dict[str, Any]):
        """Internal trade handler that forwards to user callback."""
        if self.on_trade:
            await self.on_trade(trade)

    async def _handle_l2_update(self, update: Dict[str, Any]):
        """Internal L2 handler that forwards to user callback."""
        if self.on_l2_update:
            await self.on_l2_update(update)

    async def _handle_ticker(self, ticker: Dict[str, Any]):
        """Internal ticker handler that forwards to user callback."""
        if self.on_ticker:
            await self.on_ticker(ticker)

    async def _handle_connection_status(self, connected: bool, message: str):
        """Internal connection status handler that forwards to user callback."""
        if self.on_connection_status:
            await self.on_connection_status(connected, message)

    async def get_candles(
        self,
        product_id: str,
        granularity: CoinbaseGranularity,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles from REST API.

        Args:
            product_id: Product ID (e.g., "BTC-USD")
            granularity: Candle granularity
            start: Start timestamp (UNIX seconds)
            end: End timestamp (UNIX seconds)
            limit: Maximum candles to fetch (max 300 per request)

        Returns:
            List of candle dictionaries with keys:
            - start: Candle start time (UNIX timestamp)
            - open: Open price
            - high: High price
            - low: Low price
            - close: Close price
            - volume: Volume
        """
        await self._ensure_async_session()
        await self._rate_limit()

        path = f"/products/{product_id}/candles"
        url = f"{self.config.rest_base_url}{path}"

        params = {
            "granularity": granularity.value,
            "limit": min(limit, 300),
        }

        if start:
            params["start"] = str(start)
        if end:
            params["end"] = str(end)

        headers = self._generate_auth_headers("GET", path)

        try:
            async with self.async_session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                candles = data.get("candles", [])

                # Convert to standard format
                result = []
                for candle in candles:
                    result.append({
                        "start": int(candle.get("start", 0)),
                        "open": float(candle.get("open", 0)),
                        "high": float(candle.get("high", 0)),
                        "low": float(candle.get("low", 0)),
                        "close": float(candle.get("close", 0)),
                        "volume": float(candle.get("volume", 0)),
                    })

                return result

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching candles for {product_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {e}")
            return []

    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get product details.

        Args:
            product_id: Product ID (e.g., "BTC-USD")

        Returns:
            Product details dictionary or None
        """
        await self._ensure_async_session()
        await self._rate_limit()

        path = f"/products/{product_id}"
        url = f"{self.config.rest_base_url}{path}"

        headers = self._generate_auth_headers("GET", path)

        try:
            async with self.async_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching product {product_id}: {e}")
            return None

    async def get_ticker(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker for a product.

        Args:
            product_id: Product ID (e.g., "BTC-USD")

        Returns:
            Ticker dictionary with price, volume, etc.
        """
        await self._ensure_async_session()
        await self._rate_limit()

        path = f"/products/{product_id}/ticker"
        url = f"{self.config.rest_base_url}{path}"

        headers = self._generate_auth_headers("GET", path)

        try:
            async with self.async_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching ticker for {product_id}: {e}")
            return None

    async def prime_timeframe(
        self,
        product_id: str,
        granularity: CoinbaseGranularity,
        bars: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Prime a single timeframe with historical data.

        Fetches up to `bars` candles, handling pagination if needed.

        Args:
            product_id: Product ID
            granularity: Candle granularity
            bars: Number of bars to fetch

        Returns:
            List of candles (oldest first)
        """
        all_candles = []
        remaining = bars
        end_time = int(time.time())

        while remaining > 0:
            batch_size = min(remaining, 300)
            # Calculate start time based on granularity
            start_time = end_time - (batch_size * granularity.seconds)

            candles = await self.get_candles(
                product_id,
                granularity,
                start=start_time,
                end=end_time,
                limit=batch_size
            )

            if not candles:
                break

            all_candles.extend(candles)
            remaining -= len(candles)

            # Move end time back for next batch
            if candles:
                end_time = min(c["start"] for c in candles) - 1

            # Avoid tight loop
            await asyncio.sleep(0.1)

        # Sort by timestamp (oldest first)
        all_candles.sort(key=lambda c: c["start"])

        logger.info(f"Primed {len(all_candles)} candles for {product_id} {granularity.value}")
        return all_candles

    async def prime_all_timeframes(
        self,
        product_id: str,
        granularities: List[CoinbaseGranularity],
        bars_per_tf: int = 300,
        parallel: bool = True,
        smallest_first: bool = True
    ) -> Dict[CoinbaseGranularity, List[Dict[str, Any]]]:
        """
        Prime all timeframes with historical data.

        Args:
            product_id: Product ID
            granularities: List of granularities to prime
            bars_per_tf: Bars per timeframe
            parallel: Whether to fetch in parallel
            smallest_first: Sort by smallest granularity first

        Returns:
            Dictionary mapping granularity to candle list
        """
        results: Dict[CoinbaseGranularity, List[Dict[str, Any]]] = {}

        # Sort granularities
        sorted_grans = sorted(granularities, key=lambda g: g.seconds)
        if not smallest_first:
            sorted_grans = sorted_grans[::-1]

        if parallel:
            # Parallel fetch with semaphore for rate limiting
            semaphore = asyncio.Semaphore(5)

            async def fetch_with_limit(gran: CoinbaseGranularity):
                async with semaphore:
                    candles = await self.prime_timeframe(product_id, gran, bars_per_tf)
                    return gran, candles

            tasks = [fetch_with_limit(g) for g in sorted_grans]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed:
                if isinstance(result, Exception):
                    logger.error(f"Priming failed: {result}")
                    continue
                gran, candles = result
                results[gran] = candles
        else:
            # Sequential fetch
            for gran in sorted_grans:
                candles = await self.prime_timeframe(product_id, gran, bars_per_tf)
                results[gran] = candles

        return results

    async def subscribe(self, product_ids: List[str], channels: List[str]):
        """
        Subscribe to WebSocket channels.

        Args:
            product_ids: Product IDs to subscribe to
            channels: Channels to subscribe to
        """
        if self._ws_manager:
            await self._ws_manager.subscribe(product_ids, channels)

    async def unsubscribe(self, product_ids: List[str], channels: List[str]):
        """
        Unsubscribe from WebSocket channels.

        Args:
            product_ids: Product IDs to unsubscribe from
            channels: Channels to unsubscribe from
        """
        if self._ws_manager:
            await self._ws_manager.unsubscribe(product_ids, channels)
