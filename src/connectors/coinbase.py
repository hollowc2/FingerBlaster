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
from websockets.asyncio.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from src.connectors.async_http_mixin import AsyncHttpFetcherMixin
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Pulse.CoinbaseConnector")

# Try to import JWT libraries for CDP API key support
try:
    import jwt
    from cryptography.hazmat.primitives import serialization
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning(
        "JWT libraries not available. Install with: pip install PyJWT cryptography "
        "to use CDP API keys (JWT authentication)"
    )


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
        api_passphrase: Optional[str] = None,
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
            api_passphrase: Coinbase API passphrase (optional for public data)
            on_trade: Callback for trade messages
            on_l2_update: Callback for L2 order book updates
            on_ticker: Callback for ticker updates
            on_connection_status: Callback for connection status changes
        """
        self.config = config
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
        self.api_passphrase = api_passphrase or os.getenv("COINBASE_API_PASSPHRASE")

        # Detect if this is a CDP API key (JWT-based) or legacy key (HMAC-based)
        self.is_cdp_key = self._detect_cdp_key()

        self.on_trade = on_trade
        self.on_l2_update = on_l2_update
        self.on_ticker = on_ticker
        self.on_connection_status = on_connection_status

        self._ws: Optional[ClientConnection] = None
        self._running = False
        self._reconnect_attempts = 0
        self._subscribed_products: Set[str] = set()
        self._subscribed_channels: Set[str] = set()
        self._lock = asyncio.Lock()
        self._connect_task: Optional[asyncio.Task] = None

    def _detect_cdp_key(self) -> bool:
        """
        Detect if this is a CDP API key (JWT-based) or legacy key (HMAC-based).

        CDP keys have:
        - API key format: organizations/{org_id}/apiKeys/{key_id}
        - Private key in PEM format: -----BEGIN EC PRIVATE KEY-----

        Returns:
            True if CDP key, False if legacy key
        """
        if not self.api_key or not self.api_secret:
            return False

        # Check if API key matches CDP format
        if self.api_key.startswith("organizations/") and "/apiKeys/" in self.api_key:
            return True

        # Check if secret is a PEM-format EC private key
        if "-----BEGIN EC PRIVATE KEY-----" in self.api_secret:
            return True

        return False

    def _generate_jwt_token(self, channel: str, product_ids: List[str]) -> str:
        """
        Generate JWT token for WebSocket authentication with CDP API keys.

        Based on Coinbase Advanced Trade WebSocket documentation.

        Args:
            channel: Channel name (e.g., "level2", "market_trades", "ticker")
            product_ids: List of product IDs to subscribe to

        Returns:
            JWT token string

        Raises:
            ImportError: If JWT libraries not available
            ValueError: If API credentials invalid
        """
        if not JWT_AVAILABLE:
            raise ImportError(
                "JWT libraries required for CDP API keys. "
                "Install with: pip install PyJWT cryptography"
            )

        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for JWT generation")

        # Parse the private key
        try:
            # Handle newlines in the key (replace \n with actual newlines)
            key_secret = self.api_secret.replace('\\n', '\n')
            private_key_bytes = key_secret.encode('utf-8')
            private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        except Exception as e:
            logger.error(f"Failed to parse EC private key: {e}")
            raise ValueError(f"Invalid private key format: {e}")

        # Build JWT payload for WebSocket
        # Format: similar to REST but for WebSocket channel subscription
        jwt_payload = {
            'sub': self.api_key,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,  # 2 minutes expiration
        }

        # Generate nonce
        nonce = secrets.token_hex(16)

        # Sign and encode JWT
        jwt_token = jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': self.api_key, 'nonce': nonce},
        )

        logger.debug(f"Generated JWT token for WebSocket {channel} (length: {len(jwt_token)})")
        return jwt_token

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
        
        # Try base64 decoding first (common format for Coinbase)
        try:
            import base64
            secret_bytes = base64.b64decode(self.api_secret)
        except Exception:
            # If that fails, use the secret as-is
            secret_bytes = self.api_secret.encode('utf-8')

        signature = hmac.new(
            secret_bytes,
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

        Supports both CDP (JWT) and legacy (HMAC) authentication.

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
            if self.is_cdp_key:
                # JWT authentication for CDP API keys
                try:
                    jwt_token = self._generate_jwt_token(channel, product_ids)
                    message["jwt"] = jwt_token
                    logger.debug(f"Using JWT authentication for {channel} subscription")
                except Exception as e:
                    logger.error(f"Failed to generate JWT token: {e}")
                    # Fall back to no auth (public data only)
            else:
                # HMAC signature authentication for legacy API keys
                message["api_key"] = self.api_key
                message["timestamp"] = timestamp
                message["signature"] = self._generate_signature(timestamp, channel, product_ids)
                logger.debug(f"Using HMAC authentication for {channel} subscription")

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
                await asyncio.wait_for(self._ws.close(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket close() timed out, forcing shutdown")
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await asyncio.wait_for(self._connect_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket connect task cancellation timed out")
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

        async with connect(
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
                if not self._running:
                    break
                msg = self._build_subscribe_message(
                    channel, list(self._subscribed_products), subscribe=True
                )
                await ws.send(json.dumps(msg))
                logger.info(f"Sent subscription for {channel}")

            # Process messages
            try:
                async for message in ws:
                    if not self._running:
                        logger.info("Shutdown requested, closing WebSocket connection")
                        break
                    await self._process_message(message)
            except asyncio.CancelledError:
                logger.debug("WebSocket message loop cancelled")
                raise

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
            error_msg = data.get('message', 'Unknown error')
            error_reason = data.get('reason', '')
            
            # Check if it's an authentication error
            if 'authentication' in error_msg.lower() or 'auth' in error_msg.lower():
                missing = []
                if not self.api_key:
                    missing.append("COINBASE_API_KEY")
                if not self.api_secret:
                    missing.append("COINBASE_API_SECRET")
                # Passphrase only needed for legacy keys, not CDP keys
                if not self.is_cdp_key and not self.api_passphrase:
                    missing.append("COINBASE_API_PASSPHRASE")
                
                if missing:
                    if self.is_cdp_key:
                        logger.error(
                            f"Coinbase WebSocket authentication failure: {error_msg} - "
                            f"Missing required credentials: {', '.join(missing)}. "
                            f"CDP API keys require: API key and private key (passphrase NOT needed)."
                        )
                    else:
                        logger.error(
                            f"Coinbase WebSocket authentication failure: {error_msg} - "
                            f"Missing required credentials: {', '.join(missing)}. "
                            f"Legacy API keys require: API key, secret, and passphrase."
                        )
                else:
                    if self.is_cdp_key:
                        logger.error(
                            f"Coinbase WebSocket authentication failure: {error_msg} - "
                            f"Invalid or expired API credentials. Please verify: "
                            f"1) API key and private key are correct "
                            f"2) API key has proper permissions "
                            f"3) System clock is synchronized"
                        )
                    else:
                        logger.error(
                            f"Coinbase WebSocket authentication failure: {error_msg} - "
                            f"Invalid or expired API credentials. Please verify: "
                            f"1) API key, secret, and passphrase are correct "
                            f"2) API key has proper permissions "
                            f"3) System clock is synchronized"
                        )
            else:
                logger.error(f"Coinbase WebSocket error: {error_msg} (reason: {error_reason})")
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
        api_passphrase: Optional[str] = None,
    ):
        """
        Initialize Coinbase connector.

        Args:
            config: Optional configuration (uses defaults if None)
            api_key: Coinbase API key (reads from env if None)
            api_secret: Coinbase API secret (reads from env if None)
            api_passphrase: Coinbase API passphrase (reads from env if None)
        """
        self.config = config or CoinbaseConfig()
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
        self.api_passphrase = api_passphrase or os.getenv("COINBASE_API_PASSPHRASE")
        
        # Detect if this is a CDP API key (JWT-based) or legacy key (HMAC-based)
        self.is_cdp_key = self._detect_cdp_key()
        
        # Diagnostic logging (without exposing secrets)
        if self.api_key:
            key_type = "CDP (JWT)" if self.is_cdp_key else "Legacy (HMAC)"
            logger.info(f"Coinbase API key loaded ({key_type}): {self.api_key[:8]}...{self.api_key[-4:] if len(self.api_key) > 12 else '***'} (length: {len(self.api_key)})")
        else:
            logger.warning("Coinbase API key not found in environment")
        
        if self.api_secret:
            logger.info(f"Coinbase API secret loaded: {'*' * (len(self.api_secret) - 8)}{self.api_secret[-4:]} (length: {len(self.api_secret)})")
        else:
            logger.warning("Coinbase API secret not found in environment")
        
        if self.is_cdp_key:
            if not JWT_AVAILABLE:
                logger.error(
                    "CDP API key detected but JWT libraries not available! "
                    "Install with: pip install PyJWT==2.8.0 cryptography==42.0.5"
                )
            else:
                logger.info("CDP API key detected - using JWT authentication (no passphrase required)")
        else:
            # Legacy HMAC keys may require passphrase
            if self.api_passphrase:
                logger.info(f"Coinbase API passphrase loaded: {'*' * (len(self.api_passphrase) - 4)}{self.api_passphrase[-2:]} (length: {len(self.api_passphrase)})")
            else:
                logger.warning("Legacy API key detected but passphrase not found - may cause 401 errors for legacy keys")
        
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
    
    def _detect_cdp_key(self) -> bool:
        """
        Detect if this is a CDP API key (JWT-based) or legacy key (HMAC-based).
        
        CDP keys have:
        - API key format: organizations/{org_id}/apiKeys/{key_id}
        - Private key in PEM format: -----BEGIN EC PRIVATE KEY-----
        
        Returns:
            True if CDP key, False if legacy key
        """
        if not self.api_key or not self.api_secret:
            return False
        
        # Check if API key matches CDP format
        if self.api_key.startswith("organizations/") and "/apiKeys/" in self.api_key:
            return True
        
        # Check if secret is a PEM-format EC private key
        if "-----BEGIN EC PRIVATE KEY-----" in self.api_secret:
            return True
        
        # Check if secret is a PEM-format RSA private key (less common but possible)
        if "-----BEGIN PRIVATE KEY-----" in self.api_secret:
            return True
        
        return False

    async def _ensure_async_session(self):
        """Lazy initialization of async session."""
        if self.async_session is None:
            self.async_session = await self._create_async_session(max_retries=3)
        else:
            # Check if session is closed (RetryClient wraps ClientSession)
            try:
                if hasattr(self.async_session, 'client_session'):
                    # RetryClient from mixin
                    if self.async_session.client_session._closed:
                        self.async_session = await self._create_async_session(max_retries=3)
                elif hasattr(self.async_session, '_client'):
                    # Direct ClientSession
                    if self.async_session._client.closed:
                        self.async_session = await self._create_async_session(max_retries=3)
            except AttributeError:
                # If we can't check, recreate to be safe
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

    def _generate_jwt_token(self, method: str, path: str) -> str:
        """
        Generate JWT token for CDP API key authentication.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            
        Returns:
            JWT token string
        """
        if not JWT_AVAILABLE:
            raise ImportError("JWT libraries required for CDP API keys. Install with: pip install PyJWT cryptography")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for JWT generation")
        
        # Parse the private key
        try:
            # Handle newlines in the key (replace \n with actual newlines)
            key_secret = self.api_secret.replace('\\n', '\n')
            private_key_bytes = key_secret.encode('utf-8')
            private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        except Exception as e:
            logger.error(f"Failed to parse EC private key: {e}")
            raise ValueError(f"Invalid private key format: {e}")
        
        # Build JWT payload
        request_host = "api.coinbase.com"
        uri = f"{method} {request_host}{path}"
        
        jwt_payload = {
            'sub': self.api_key,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,  # 2 minutes expiration
            'uri': uri,
        }
        
        # Generate nonce
        nonce = secrets.token_hex(16)
        
        # Sign and encode JWT
        jwt_token = jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': self.api_key, 'nonce': nonce},
        )
        
        return jwt_token

    def _generate_auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Generate authentication headers for REST API.
        
        Supports two authentication methods:
        1. CDP API keys (JWT-based): Uses Authorization Bearer token
        2. Legacy API keys (HMAC-based): Uses CB-ACCESS-* headers

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            body: Request body (for POST)

        Returns:
            Headers dictionary
        """
        if not self.api_key or not self.api_secret:
            return {}

        headers = {
            "Content-Type": "application/json",
        }
        
        # Use JWT authentication for CDP keys
        if self.is_cdp_key:
            if not JWT_AVAILABLE:
                logger.error(
                    f"CDP API key detected but JWT libraries not available! "
                    f"Install with: pip install PyJWT==2.8.0 cryptography==42.0.5"
                )
                return {}
            try:
                jwt_token = self._generate_jwt_token(method, path)
                headers["Authorization"] = f"Bearer {jwt_token}"
                logger.info(f"Generated JWT token for {method} {path} (token length: {len(jwt_token)})")
            except Exception as e:
                logger.error(f"Failed to generate JWT token for {method} {path}: {e}", exc_info=True)
                return {}
        else:
            # Use HMAC authentication for legacy keys
            timestamp = str(int(time.time()))
            # Prehash string: timestamp + method + requestPath + body
            message = f"{timestamp}{method}{path}{body}"

            # The API secret might be base64 encoded - try both
            secret_bytes = None
            secret_format = "unknown"
            try:
                # Try base64 decoding first (common format for Coinbase)
                import base64
                secret_bytes = base64.b64decode(self.api_secret)
                secret_format = "base64-decoded"
            except Exception:
                # If that fails, use the secret as-is (raw string)
                try:
                    secret_bytes = self.api_secret.encode('utf-8')
                    secret_format = "raw-utf8"
                except Exception as e:
                    logger.error(f"Failed to encode API secret: {e}")
                    return {}

            signature = hmac.new(
                secret_bytes,
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Debug logging (without exposing secrets)
            logger.debug(
                f"Auth header generation (HMAC): method={method}, path={path}, "
                f"timestamp={timestamp}, secret_format={secret_format}, "
                f"has_passphrase={bool(self.api_passphrase)}, "
                f"signature_length={len(signature)}"
            )

            headers.update({
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
            })
            
            # Add passphrase if available
            if self.api_passphrase:
                headers["CB-ACCESS-PASSPHRASE"] = self.api_passphrase
            else:
                logger.debug(
                    "CB-ACCESS-PASSPHRASE not provided - attempting without it. "
                    "If authentication fails, the passphrase is usually shown AFTER you create the API key, "
                    "or you may need to set it during the creation process."
                )

        return headers

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
                api_passphrase=self.api_passphrase,
                on_trade=self._handle_trade,
                on_l2_update=self._handle_l2_update,
                on_ticker=self._handle_ticker,
                on_connection_status=self._handle_connection_status,
            )
            await self._ws_manager.start(product_ids, channels)

    async def stop(self):
        """Stop the connector and cleanup resources."""
        # Stop WebSocket manager if it exists
        if hasattr(self, '_ws_manager') and self._ws_manager:
            try:
                await self._ws_manager.stop()
            except Exception as e:
                logger.debug(f"Error stopping WebSocket manager: {e}")
            finally:
                self._ws_manager = None

        # Close async session if it exists
        if hasattr(self, 'async_session') and self.async_session:
            try:
                # RetryClient has a close method
                if hasattr(self.async_session, 'close'):
                    await self.async_session.close()
                # Or if it's a ClientSession directly
                elif hasattr(self.async_session, '_client') and not self.async_session._client.closed:
                    await self.async_session.close()
            except Exception as e:
                logger.debug(f"Error closing async session: {e}")
            finally:
                self.async_session = None

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

        # For JWT (CDP keys), use full API path without query params in URI
        # For HMAC (legacy keys), include query params in signature
        if self.is_cdp_key:
            # JWT URI needs full path: /api/v3/brokerage/products/BTC-USD/candles
            # The path from get_candles is relative, so add the API prefix
            auth_path = f"/api/v3/brokerage{path}"
        else:
            # HMAC requires params in signature
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            auth_path = f"{path}?{query_string}" if query_string else path

        headers = self._generate_auth_headers("GET", auth_path)

        try:
            async with self.async_session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                # Check for authentication errors
                if response.status == 401:
                    # Try to get error details from response
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get('message', '') or error_data.get('error', '')
                        if self.is_cdp_key:
                            logger.error(
                                f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                f"Coinbase API error: {error_msg}. "
                                f"Please verify: 1) API key and private key are correct "
                                f"2) API key has proper permissions (View permission required) "
                                f"3) System clock is synchronized "
                                f"4) Private key format is correct (PEM format with BEGIN/END lines)"
                            )
                        else:
                            logger.error(
                                f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                f"Coinbase API error: {error_msg}. "
                                f"Please verify: 1) API key, secret, and passphrase are correct "
                                f"2) API key has proper permissions (View permission required) "
                                f"3) System clock is synchronized "
                                f"4) API secret format (try base64 decoded if using raw, or vice versa)"
                            )
                    except Exception:
                        missing = []
                        if not self.api_key:
                            missing.append("COINBASE_API_KEY")
                        if not self.api_secret:
                            missing.append("COINBASE_API_SECRET")
                        if not self.is_cdp_key and not self.api_passphrase:
                            missing.append("COINBASE_API_PASSPHRASE")
                        
                        if missing:
                            if self.is_cdp_key:
                                logger.error(
                                    f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                    f"Missing required credentials: {', '.join(missing)}. "
                                    f"CDP API keys require: API key and private key (passphrase NOT needed)."
                                )
                            else:
                                logger.error(
                                    f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                    f"Missing required credentials: {', '.join(missing)}. "
                                    f"Legacy API keys require: API key, secret, and passphrase."
                                )
                        else:
                            if self.is_cdp_key:
                                logger.error(
                                    f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                    f"Invalid or expired Coinbase API credentials. Please verify: "
                                    f"1) API key and private key are correct "
                                    f"2) API key has proper permissions "
                                    f"3) System clock is synchronized"
                                )
                            else:
                                logger.error(
                                    f"Error fetching candles for {product_id}: 401 Unauthorized - "
                                    f"Invalid or expired Coinbase API credentials. Please verify: "
                                    f"1) API key, secret, and passphrase are correct "
                                    f"2) API key has proper permissions "
                                    f"3) System clock is synchronized (timestamps must be accurate)"
                                )
                    return []
                
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

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                # Already handled above, but catch here too for safety
                return []
            logger.error(f"Error fetching candles for {product_id}: {e.status}, message='{e.message}', url='{e.request_info.url}'")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching candles for {product_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {e}")
            return []

    async def get_public_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get product details including 24h volume, price changes, and other market data.
        Note: Despite the name, this endpoint requires authentication.

        Args:
            product_id: Product ID (e.g., "BTC-USD")

        Returns:
            Product details dictionary with volume_24h, price, etc.
        """
        await self._ensure_async_session()
        await self._rate_limit()

        path = f"/products/{product_id}"
        url = f"{self.config.rest_base_url}{path}"

        # For JWT, use full API path; for HMAC, use relative path
        if self.is_cdp_key:
            auth_path = f"/api/v3/brokerage{path}"
        else:
            auth_path = path

        headers = self._generate_auth_headers("GET", auth_path)

        try:
            async with self.async_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                if response.status == 401:
                    if not self.api_key or not self.api_secret:
                        logger.error(
                            f"Error fetching product {product_id}: 401 Unauthorized - "
                            f"Coinbase API credentials are required. Set COINBASE_API_KEY and COINBASE_API_SECRET in .env"
                        )
                    else:
                        logger.error(
                            f"Error fetching product {product_id}: 401 Unauthorized - "
                            f"Invalid or expired Coinbase API credentials."
                        )
                    return None

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                return None  # Already logged above
            logger.error(f"Error fetching product {product_id}: {e.status}, message='{e.message}'")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching product {product_id}: {e}")
            return None

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

        # For JWT, use full API path; for HMAC, use relative path
        if self.is_cdp_key:
            auth_path = f"/api/v3/brokerage{path}"
        else:
            auth_path = path

        headers = self._generate_auth_headers("GET", auth_path)

        try:
            async with self.async_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                if response.status == 401:
                    if not self.api_key or not self.api_secret:
                        logger.error(
                            f"Error fetching product {product_id}: 401 Unauthorized - "
                            f"Coinbase API credentials are required. Set COINBASE_API_KEY and COINBASE_API_SECRET in .env"
                        )
                    else:
                        logger.error(
                            f"Error fetching product {product_id}: 401 Unauthorized - "
                            f"Invalid or expired Coinbase API credentials."
                        )
                    return None

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                return None  # Already logged above
            logger.error(f"Error fetching product {product_id}: {e.status}, message='{e.message}'")
            return None
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

        # For JWT, use full API path; for HMAC, use relative path
        if self.is_cdp_key:
            auth_path = f"/api/v3/brokerage{path}"
        else:
            auth_path = path

        headers = self._generate_auth_headers("GET", auth_path)

        try:
            async with self.async_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.rest_request_timeout)
            ) as response:
                if response.status == 401:
                    if not self.api_key or not self.api_secret:
                        logger.error(
                            f"Error fetching ticker for {product_id}: 401 Unauthorized - "
                            f"Coinbase API credentials are required. Set COINBASE_API_KEY and COINBASE_API_SECRET in .env"
                        )
                    else:
                        logger.error(
                            f"Error fetching ticker for {product_id}: 401 Unauthorized - "
                            f"Invalid or expired Coinbase API credentials."
                        )
                    return None
                
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                return None  # Already logged above
            logger.error(f"Error fetching ticker for {product_id}: {e.status}, message='{e.message}'")
            return None
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
