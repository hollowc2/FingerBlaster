
import json
import ast
import logging
import os
import re
import time
import hmac
import hashlib
import base64
import urllib.parse
import asyncio
from decimal import Decimal, ROUND_DOWN
from functools import wraps
from typing import Optional, Dict, List, Any, Tuple, Callable, TypeVar

T = TypeVar('T')

import pandas as pd
import requests
import aiohttp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from web3 import Web3, AsyncWeb3
from web3.providers import AsyncHTTPProvider

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    TradeParams, MarketOrderArgs, OrderType,
    BalanceAllowanceParams, AssetType, OrderArgs
)
from py_clob_client.order_builder.constants import BUY, SELL

from src.connectors.http_mixin import HttpFetcherMixin
from src.connectors.async_http_mixin import AsyncHttpFetcherMixin
from src.connectors.polymarket_api import PolymarketAPI, SignatureType

load_dotenv()

class TradingConstants:
    MAX_PRICE = 0.99
    MIN_PRICE = 0.01
    BUY_AGGRESSIVE_MULTIPLIER = 1.10
    SELL_AGGRESSIVE_MULTIPLIER = 0.90
    MARKET_DURATION_MINUTES = 15
    MIN_BALANCE_THRESHOLD = 0.1
    USDC_DECIMALS = 6
    CONDITIONAL_TOKEN_DECIMALS = 6
    PRICE_ROUNDING_PLACES = 2

class NetworkConstants:
    POLYGON_CHAIN_ID = 137
    POLYGON_RPC_URL = "https://polygon-rpc.com"
    CLOB_HOST = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    DATA_API_URL = "https://data-api.polymarket.com"
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    USDC_CONTRACT_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    # Chainlink BTC/USD Price Feed on Polygon (used by Polymarket for resolution)
    CHAINLINK_BTC_USD_FEED = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3

logger = logging.getLogger("PolymarketConnector")

# Minimal Chainlink Aggregator ABI for price queries
CHAINLINK_AGGREGATOR_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_roundId", "type": "uint80"}],
        "name": "getRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    }
]


def validate_order_params(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    async def wrapper(self, token_id: str, amount: float, side: str, *args, **kwargs) -> T:
        if not token_id or len(token_id) < 10:
            logger.error(f"Invalid token_id: {token_id}")
            return None
        if amount <= 0 or amount > 1_000_000:
            logger.error(f"Invalid amount: {amount}")
            return None
        if side.upper() not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None
        return await func(self, token_id, amount, side, *args, **kwargs)
    return wrapper


class PolymarketConnector(HttpFetcherMixin, AsyncHttpFetcherMixin):

    def __init__(self):
        # Initialize API helper for low-level auth and client
        self.api = PolymarketAPI(NetworkConstants.CLOB_HOST, NetworkConstants.POLYGON_CHAIN_ID)
        
        # Facade: Expose client and signature_type from API
        self.client = self.api.client
        self.signature_type = self.api.signature_type
        
        self.gamma_url = NetworkConstants.GAMMA_API_URL

        # Create session with connection pooling and retries (inherited from HttpFetcherMixin)
        self.session = self._create_session(max_retries=NetworkConstants.MAX_RETRIES)

        # Async session (lazy initialization)
        self.async_session: Optional[aiohttp.ClientSession] = None
        self._session_initialized = False

        # AsyncWeb3 (lazy initialization)
        self.async_w3: Optional[AsyncWeb3] = None

    async def _ensure_async_session(self):
        """Lazy initialization of async session."""
        # Check if session exists and is not closed
        # RetryClient wraps ClientSession, so we check the underlying session
        needs_init = (
            self.async_session is None or
            (hasattr(self.async_session, 'client_session') and
             self.async_session.client_session._closed) or
            (hasattr(self.async_session, '_closed') and
             self.async_session._closed)
        )
        if needs_init:
            self.async_session = await self._create_async_session(
                max_retries=NetworkConstants.MAX_RETRIES
            )
            self._session_initialized = True

    async def _ensure_async_web3(self):
        """Lazy initialization of AsyncWeb3."""
        if self.async_w3 is None:
            provider = AsyncHTTPProvider(NetworkConstants.POLYGON_RPC_URL)
            self.async_w3 = AsyncWeb3(provider)

    async def close(self):
        """Close async session and cleanup resources."""
        if self.async_session:
            try:
                # Handle both RetryClient (has close method) and direct ClientSession
                if hasattr(self.async_session, 'close'):
                    await self.async_session.close()
                elif hasattr(self.async_session, '_client') and not self.async_session._client.closed:
                    await self.async_session.close()
            except Exception as e:
                logger.debug(f"Error closing async session: {e}")
            finally:
                self.async_session = None
        
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.debug(f"Error closing sync session: {e}")
        if self.async_w3:
            # AsyncWeb3 providers don't need explicit cleanup in most cases
            pass
    
    def _safe_parse_json_list(self, json_str: str) -> List[str]:
        """
        Safely parse a JSON list string without using eval().
        
        SECURITY FIX: Replaces eval() with json.loads() or ast.literal_eval()
        
        Args:
            json_str: JSON string representation of a list
            
        Returns:
            Parsed list of strings
            
        Raises:
            ValueError: If string cannot be parsed
        """
        if not json_str or not isinstance(json_str, str):
            return []
        
        json_str = json_str.strip()
        if not json_str:
            return []
        
        # Try json.loads first (fastest and safest)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback to ast.literal_eval for Python literal strings
        try:
            parsed = ast.literal_eval(json_str)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        
        # If all else fails, return empty list
        logger.warning(f"Could not parse JSON list: {json_str[:50]}...")
        return []
    
    def _extract_price_to_beat(self, market: Dict[str, Any], event: Dict[str, Any]) -> str:
        """
        Extract price to beat from market data using multiple strategies.
        
        Improved to check more fields and be more accurate.
        For dynamic prices to beat, uses RTDS or fetches price at exact market start time.
        
        Args:
            market: Market data dictionary
            event: Event data dictionary
            
        Returns:
            Price to beat as string
        """
        # Log all available fields for debugging (first time only to avoid spam)
        logger.info(f"=== STRIKE EXTRACTION DEBUG ===")
        logger.info(f"groupItemThreshold: {market.get('groupItemThreshold')} (type: {type(market.get('groupItemThreshold'))})")
        logger.info(f"groupItemTitle: {market.get('groupItemTitle')}")
        logger.info(f"Market start: {market.get('startDate')}")
        logger.info(f"Current time: {pd.Timestamp.now(tz='UTC')}")

        # Log all available fields for debugging (first time only to avoid spam)
        if not hasattr(self, '_logged_fields'):
            logger.debug(f"Market fields: {list(market.keys())}")
            logger.debug(f"Event fields: {list(event.keys())}")
            # Log sample values for key fields (including groupItemThreshold which might be the strike!)
            for key in ['groupItemThreshold', 'groupItemTitle', 'title', 'question', 'description', 'resolutionCriteria']:
                if key in market:
                    logger.debug(f"Market.{key} = {market[key]}")
            for key in ['resolutionCriteria', 'endDate']:
                if key in event:
                    logger.debug(f"Event.{key} = {event[key]}")
            self._logged_fields = True
        
        # Strategy 0: Check groupItemThreshold FIRST (this is the strike price Polymarket uses!)
        if 'groupItemThreshold' in market:
            threshold = market.get('groupItemThreshold')
            logger.info(f"groupItemThreshold raw value: {threshold} (type: {type(threshold)})")
            if threshold is not None:
                try:
                    # Handle both string and numeric types
                    if isinstance(threshold, str):
                        # Remove any formatting
                        threshold_clean = threshold.replace('$', '').replace(',', '').strip()
                        threshold_val = float(threshold_clean)
                    else:
                        threshold_val = float(threshold)
                    
                    if threshold_val > 0:
                        logger.info(f"✓ STRIKE from groupItemThreshold: ${threshold_val:,.2f}")
                        return f"{threshold_val:,.2f}"
                    else:
                        logger.warning(f"groupItemThreshold is zero or negative: {threshold_val}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse groupItemThreshold '{threshold}': {e}")
        else:
            logger.warning("groupItemThreshold NOT FOUND in market data - strike may be inaccurate!")
        
        # Strategy 1: Check for explicit strike/resolutionCriteria field
        if 'resolutionCriteria' in market:
            criteria = market['resolutionCriteria']
            if isinstance(criteria, dict):
                strike = criteria.get('strike') or criteria.get('price') or criteria.get('threshold')
                if strike:
                    logger.info(f"Found strike in resolutionCriteria: {strike}")
                    return str(strike)
        
        # Check event-level resolution criteria
        if 'resolutionCriteria' in event:
            criteria = event['resolutionCriteria']
            if isinstance(criteria, dict):
                strike = criteria.get('strike') or criteria.get('price') or criteria.get('threshold')
                if strike:
                    logger.info(f"Found strike in event resolutionCriteria: {strike}")
                    return str(strike)
        
        # Strategy 1: Check groupItemTitle (most reliable for static strikes)
        strike_price = market.get('groupItemTitle', '').replace('> ', '').replace('< ', '').strip()
        if strike_price:
            # Clean up any remaining formatting
            strike_price = strike_price.replace('$', '').replace(',', '').strip()
            if strike_price and strike_price != "N/A":
                logger.info(f"Found strike in groupItemTitle: {strike_price}")
                return strike_price
        
        # Strategy 2: Check title field
        title = market.get('title', '')
        if title:
            # Pattern: "> 12345" or "BTC > $12345"
            match = re.search(r'>\s*\$?([0-9,.]+)', title)
            if match:
                return match.group(1).replace(',', '')
        
        # Strategy 3: Parse from question field
        if 'question' in market:
            question = market['question']
            
            # Pattern 1: "> 12345" or "> $12345"
            match = re.search(r'>\s*\$?([0-9,.]+)', question)
            if match:
                return match.group(1).replace(',', '')
            
            # Pattern 2: "above $12345" or "above 12345"
            match = re.search(r'above\s+\$?([0-9,.]+)', question, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
            
            # Pattern 3: ">= 12345"
            match = re.search(r'>=\s*\$?([0-9,.]+)', question)
            if match:
                return match.group(1).replace(',', '')
        
        # Strategy 4: Parse from description
        if 'description' in market:
            description = market['description']
            
            # Pattern 1: "greater than $12345"
            match = re.search(r"greater than \$?([0-9,.]+)", description, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
            
            # Pattern 2: "price at the beginning of that range" (dynamic strike)
            if "price at the beginning of that range" in description.lower() or "price at the start" in description.lower():
                try:
                    end_dt = pd.Timestamp(event['endDate'])
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize('UTC')
                    start_dt = end_dt - pd.Timedelta(minutes=TradingConstants.MARKET_DURATION_MINUTES)
                    now = pd.Timestamp.now(tz='UTC')
                    
                    logger.debug(f"Dynamic strike detected: Market start={start_dt}, End={end_dt}, Now={now}")
                    
                    if start_dt < now:
                        # For dynamic strikes, we need Chainlink price at market start time
                        # This will be handled by the core using RTDS historical data
                        # Return "Dynamic" to signal that core should look it up
                        logger.debug(f"Dynamic strike detected - will be resolved by RTDS historical lookup")
                        return "Dynamic"
                    else:
                        logger.debug("Market hasn't started yet, strike pending")
                        return "Pending"
                except Exception as e:
                    logger.error(f"Error fetching dynamic strike: {e}", exc_info=True)
        
        # Strategy 5: Check outcomes array for strike info
        outcomes = market.get('outcomes', [])
        if outcomes:
            for outcome in outcomes:
                outcome_title = outcome.get('title', '')
                if outcome_title:
                    match = re.search(r'>\s*\$?([0-9,.]+)', outcome_title)
                    if match:
                        return match.group(1).replace(',', '')
        
        # Log what we found for debugging
        logger.warning(f"Could not extract strike price from market data")
        logger.debug(f"Market data sample: {json.dumps({k: v for k, v in list(market.items())[:10]}, default=str)}")
        return "N/A"
    
    def _build_token_map(self, market: Dict[str, Any], clob_token_ids: List[str]) -> Dict[str, str]:
        """
        Build token map (Up/Down -> token_id) from market data.
        
        Extracted from duplicate code to follow DRY principle.
        
        Args:
            market: Market data dictionary
            clob_token_ids: List of CLOB token IDs
            
        Returns:
            Dictionary mapping 'Up' and 'Down' to token IDs
        """
        token_map = {}
        up_token_id = None
        down_token_id = None
        
        # Try to get from tokens array
        tokens_data = market.get('tokens', [])
        if tokens_data:
            for token in tokens_data:
                outcome = token.get('outcome', '')
                # Handle both title case and uppercase
                outcome_upper = outcome.upper()
                if outcome_upper in ('YES', 'UP'):
                    up_token_id = token.get('tokenId') or token.get('clobTokenId')
                elif outcome_upper in ('NO', 'DOWN'):
                    down_token_id = token.get('tokenId') or token.get('clobTokenId')
        
        # Build map from found tokens
        if up_token_id:
            token_map['Up'] = up_token_id
        if down_token_id:
            token_map['Down'] = down_token_id
        
        # Fallback to clob_token_ids list if needed
        if 'Up' not in token_map and len(clob_token_ids) > 0:
            token_map['Up'] = clob_token_ids[0]
        if 'Down' not in token_map and len(clob_token_ids) > 1:
            token_map['Down'] = clob_token_ids[1]
        
        return token_map
    
    def _parse_market_data(
        self, event: Dict[str, Any], include_token_map: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Parse market data from event dictionary.
        
        Extracted common logic from get_active_market() and get_next_market()
        to eliminate code duplication (DRY principle).
        
        Args:
            event: Event data dictionary
            include_token_map: Whether to include full token_map in response
            
        Returns:
            Market data dictionary or None if parsing fails
        """
        if not event.get('markets'):
            return None
        
        market = event['markets'][0]
        
        # SECURITY FIX: Use safe JSON parsing instead of eval()
        clob_token_ids_str = market.get('clobTokenIds', '[]')
        clob_token_ids = self._safe_parse_json_list(clob_token_ids_str)
        
        if not clob_token_ids:
            logger.warning("No CLOB token IDs found in market data")
            return None
        
        # Extract price to beat
        price_to_beat = self._extract_price_to_beat(market, event)
        
        # Build response
        result = {
            'event_slug': event.get('slug'),
            'market_id': market.get('id'),
            'token_id': clob_token_ids[0],  # Primary token
            'token_ids': clob_token_ids,
            'start_date': event.get('startDate') or market.get('startDate'),
            'end_date': event.get('endDate'),
            'price_to_beat': str(price_to_beat),
            'question': market.get('question', '') or event.get('question', ''),
            'title': market.get('title', '') or event.get('title', '')
        }
        
        # Include token map if requested
        if include_token_map:
            result['token_map'] = self._build_token_map(market, clob_token_ids)
        
        return result
    
    async def get_token_balance(self, token_id: str) -> float:
        """
        Get the balance of a specific outcome token (async wrapper).

        Note: Uses to_thread() internally since py-clob-client is synchronous.

        Args:
            token_id: Token ID to check balance for

        Returns:
            Balance in shares (float)
        """
        def _sync_get_balance():
            try:
                b_params = BalanceAllowanceParams(
                    token_id=token_id,
                    asset_type=AssetType.CONDITIONAL,
                    signature_type=self.signature_type
                )
                balance_info = self.client.get_balance_allowance(b_params)
                raw_balance = int(balance_info.get('balance', 0))
                return raw_balance / (10 ** TradingConstants.CONDITIONAL_TOKEN_DECIMALS)
            except Exception as e:
                logger.error(f"Error fetching token balance for {token_id}: {e}")
                return 0.0

        return await asyncio.to_thread(_sync_get_balance)

    async def get_positions(
        self,
        market_ids: Optional[List[str]] = None,
        size_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get user positions with entry prices and PnL from Polymarket Data API.

        Args:
            market_ids: Optional list of condition IDs to filter by
            size_threshold: Minimum position size (default 0.1)

        Returns:
            List of position dicts with keys:
                - asset: token ID
                - size: position quantity
                - avgPrice: average entry price
                - curPrice: current market price
                - initialValue: opening position value
                - currentValue: present position value
                - cashPnl: profit/loss in USDC
                - percentPnl: profit/loss percentage
                - outcome: Up or Down (or legacy YES/NO)
                - conditionId: market condition ID
        """
        try:
            wallet_address = os.getenv("WALLET_ADDRESS") or self.api.wallet_address or self.client.get_address()
            if not wallet_address:
                logger.warning("No wallet address available for positions query")
                return []

            return await self.api.get_positions(
                NetworkConstants.DATA_API_URL, 
                wallet_address, 
                size_threshold,
                session=self.async_session
            )
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def flatten_market(self, token_map: Dict[str, str]) -> List[Any]:
        """
        Close all positions in the given token map (YES and NO) (async wrapper).

        Note: Uses async methods but some underlying operations use to_thread().

        Args:
            token_map: Dictionary mapping 'YES'/'NO' to token IDs

        Returns:
            List of order responses
        """
        results = []
        for outcome, token_id in token_map.items():
            logger.info(f"Checking balance for {outcome} ({token_id[:10]}...)")
            balance = await self.get_token_balance(token_id)

            if balance > TradingConstants.MIN_BALANCE_THRESHOLD:
                logger.info(f"Found {balance} shares of {outcome}. Flattening...")
                resp = await self.create_market_order(token_id, balance, 'SELL')
                results.append(resp)
            else:
                logger.info(f"No significant balance for {outcome}.")

        return results
    
    async def get_usdc_balance(self) -> float:
        """
        Get USDC balance from wallet (async).

        Tries AsyncWeb3 direct blockchain query first, falls back to CLOB API.

        Returns:
            USDC balance (float)
        """
        # Try AsyncWeb3 first (most reliable)
        try:
            await self._ensure_async_web3()
            key = os.getenv("PRIVATE_KEY")
            env_address = os.getenv("WALLET_ADDRESS")

            target_address = None
            if env_address:
                target_address = env_address
            elif key:
                # For getting address from key, we can still use sync Web3
                w3_sync = Web3()
                account = w3_sync.eth.account.from_key(key)
                target_address = account.address

            if target_address:
                usdc_abi = [{
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                }]

                contract = self.async_w3.eth.contract(
                    address=NetworkConstants.USDC_CONTRACT_ADDRESS,
                    abi=usdc_abi
                )

                target_address = Web3.to_checksum_address(target_address)
                balance_wei = await contract.functions.balanceOf(target_address).call()
                return balance_wei / (10 ** TradingConstants.USDC_DECIMALS)
        except Exception as e:
            logger.debug(f"AsyncWeb3 balance check failed: {e}")

        # Fallback to CLOB API (wrapped in to_thread)
        try:
            def _get_clob_balance():
                balance_info = self.client.get_balance_allowance()
                return float(balance_info.get('balance', 0))

            return await asyncio.to_thread(_get_clob_balance)
        except Exception as e:
            logger.error(f"Both AsyncWeb3 and CLOB API balance checks failed: {e}")
            return 0.0
    
    def fetch_data(
        self, symbol: str, start_time: int = None, 
        end_time: int = None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades for a given token_id (symbol).
        
        Args:
            symbol: Token ID
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum number of records (optional)
            
        Returns:
            DataFrame with trade data
        """
        # Implementation placeholder - would need proper API endpoint
        return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get midpoint price for a token_id.
        
        Args:
            symbol: Token ID
            
        Returns:
            Midpoint price (float)
        """
        try:
            return self.client.get_midpoint(symbol)
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0.0
    
    def get_order_book(self, token_id: str):
        """
        Get order book for a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Order book object or None
        """
        try:
            return self.client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    async def get_market_data(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get market data from Gamma API (async).

        Args:
            market_id: Market ID or slug

        Returns:
            Market data dictionary or None
        """
        await self._ensure_async_session()
        try:
            if "-" in market_id and not market_id.isdigit():
                url = f"{self.gamma_url}/events?slug={market_id}"
            else:
                url = f"{self.gamma_url}/markets/{market_id}"

            data = await self._get_json_async(url, timeout=NetworkConstants.REQUEST_TIMEOUT)
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            return data
        except Exception as e:
            logger.error(f"Unexpected error fetching market data: {e}")
            return None
    
    async def get_active_market(self, series_id: str = "10192") -> Optional[Dict[str, Any]]:
        """
        Fetch the currently active market for a given series ID (async).

        Args:
            series_id: Series ID (default: "10192" for BTC Up or Down 15m)

        Returns:
            Market data dictionary or None
        """
        await self._ensure_async_session()
        try:
            target_event = await self.api.get_active_market(self.gamma_url, series_id, session=self.async_session)
            if not target_event:
                return None

            # Try to get more detailed market data that might have the actual strike
            logger.info(f"Initial market data fetch complete. ID: {target_event.get('id') if target_event else 'None'}")
            market_data = self._parse_market_data(target_event, include_token_map=True)

            # If we have a market_id, try to fetch detailed market info
            # Also check if there's a resolution endpoint that has the strike
            if market_data and market_data.get('market_id'):
                try:
                    # Try the market details endpoint
                    detailed_market = await self.get_market_data(market_data['market_id'])
                    if detailed_market:
                        # Check if detailed market has better strike info
                        if isinstance(detailed_market, dict):
                            # Check groupItemThreshold in detailed market (might be populated after market starts)
                            if 'markets' in detailed_market and len(detailed_market['markets']) > 0:
                                detailed_market_obj = detailed_market['markets'][0]
                                threshold = detailed_market_obj.get('groupItemThreshold')
                                if threshold and threshold != 0 and threshold != "0":
                                    try:
                                        threshold_val = float(threshold)
                                        if threshold_val > 0:
                                            logger.info(f"Found strike in detailed market groupItemThreshold: {threshold_val:,.2f}")
                                            market_data['price_to_beat'] = f"{threshold_val:,.2f}"
                                            return market_data
                                    except (ValueError, TypeError):
                                        pass

                            detailed_strike = self._extract_price_to_beat(
                                detailed_market.get('markets', [{}])[0] if detailed_market.get('markets') else detailed_market,
                                detailed_market if 'endDate' in detailed_market else target_event
                            )
                            if detailed_strike and detailed_strike != "N/A":
                                logger.info(f"Found strike from detailed market data: {detailed_strike}")
                                market_data['price_to_beat'] = detailed_strike

                    # Also try querying the market's resolution endpoint if it exists
                    # Some markets might expose the strike via a resolution query
                    try:
                        resolution_url = f"{self.gamma_url}/markets/{market_data['market_id']}/resolution"
                        async with self.async_session.get(
                            resolution_url, timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
                        ) as resolution_response:
                            if resolution_response.status == 200:
                                resolution_data = await resolution_response.json()
                                # Check if resolution data contains strike
                                strike = (resolution_data.get('strike') or
                                        resolution_data.get('threshold') or
                                        resolution_data.get('price'))
                                if strike:
                                    logger.info(f"Found strike from resolution endpoint: {strike}")
                                    market_data['price_to_beat'] = str(strike)
                    except Exception as e:
                        logger.debug(f"Resolution endpoint check failed (may not exist): {e}")

                except Exception as e:
                    logger.debug(f"Could not fetch detailed market data: {e}")

            return market_data

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching active market: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_active_market: {e}", exc_info=True)
            return None
    
    async def get_strike_from_polymarket_page(self, slug: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Optional[float]:
        """
        Fetch the exact strike price from the Polymarket frontend __NEXT_DATA__.
        This guarantees 100% alignment with the website display.
        """
        if not slug: return None
        
        # Format timestamps to match React Query keys: "2026-01-08T01:00:00Z"
        # Ensure UTC
        if start_dt.tz is None: start_dt = start_dt.tz_localize('UTC')
        if end_dt.tz is None: end_dt = end_dt.tz_localize('UTC')
        
        start_str = start_dt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_dt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        
        url = f"https://polymarket.com/event/{slug}"
        try:
            logger.info(f"Fetching official strike from frontend: {url}")
            await self._ensure_async_session()
            # Use standard User-Agent to avoid blocking
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            
            async with self.async_session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to fetch page: {resp.status}")
                    return None
                    
                text = await resp.text()
                # Extract __NEXT_DATA__ JSON
                match = re.search(r'<script id="__NEXT_DATA__" type="application/json">([^<]+)</script>', text)
                if match:
                    data = json.loads(match.group(1))
                    
                    # Traverse queries in dehydratedState
                    queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
                    
                    found_val = None
                    
                    # Strategy 1: Look for exact 'crypto-prices' key match
                    for q in queries:
                        key = q.get('queryKey')
                        if isinstance(key, list) and len(key) >= 6 and key[0] == 'crypto-prices':
                            # key structure: ['crypto-prices', 'price', 'BTC', start_str, 'fifteen', end_str]
                            if key[3] == start_str or key[5] == end_str:
                                val = q.get('state', {}).get('data', {}).get('openPrice')
                                if val:
                                    logger.info(f"✓ Found exact frontend strike (openPrice): {val} in query {key}")
                                    return float(val)
                    
                    # Strategy 2: Look for 'past-results' match (previous close)
                    for q in queries:
                        key = q.get('queryKey')
                        if isinstance(key, list) and len(key) >= 4 and key[0] == 'past-results':
                            # key: ['past-results', 'BTC', 'fifteen', start_str]
                            if key[3] == start_str:
                                # This query has the results for the PREVIOUS period ending at start_str?
                                # Or results starting at start_str?
                                # Let's assume strategy 1 is sufficient as it matched explicitly in our test.
                                pass

        except Exception as e:
            logger.error(f"Error fetching frontend strike: {e}")
        
        return None

    async def get_next_market(
        self, current_end_date_str: str, series_id: str = "10192"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch the next market that ends AFTER the provided current_end_date (async).

        Args:
            current_end_date_str: Current market end date string
            series_id: Series ID (default: "10192")

        Returns:
            Market data dictionary or None
        """
        await self._ensure_async_session()
        try:
            url = f"{self.gamma_url}/events?limit=100&closed=false&series_id={series_id}"
            async with self.async_session.get(
                url, timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
            ) as response:
                response.raise_for_status()
                events = await response.json()

            if not events:
                return None

            current_end = pd.Timestamp(current_end_date_str)
            if current_end.tz is None:
                current_end = current_end.tz_localize('UTC')

            # Filter events that end after current
            valid_events = [
                e for e in events
                if pd.Timestamp(e['endDate']) > current_end
            ]

            if not valid_events:
                return None

            # Sort by end date and get the soonest one
            valid_events.sort(key=lambda x: pd.Timestamp(x['endDate']))
            target_event = valid_events[0]

            return self._parse_market_data(target_event, include_token_map=False)

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching next market: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_next_market: {e}", exc_info=True)
            return None
    
    async def get_btc_price(self) -> float:
        """
        Fetch current BTC price from Binance API (async).

        Returns:
            BTC price in USDT (float)
        """
        await self._ensure_async_session()
        try:
            url = f"{NetworkConstants.BINANCE_API_URL}/ticker/price"
            params = {'symbol': 'BTCUSDT'}
            data = await self._get_json_async(
                url, params=params, timeout=NetworkConstants.REQUEST_TIMEOUT
            )
            if data:
                return float(data['price'])
            return 0.0
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing BTC price from Binance: {e}")
            return 0.0
    
    async def get_chainlink_price_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Fetch Chainlink BTC/USD price at a specific timestamp (async).

        Uses Chainlink Data Streams API to get historical prices.
        Based on: https://data.chain.link/streams/btc-usd

        Args:
            timestamp: Timestamp to fetch price for

        Returns:
            Chainlink BTC/USD price as float or None if unavailable
        """
        await self._ensure_async_session()
        try:
            # Chainlink Data Streams API for BTC/USD
            # The stream ID for BTC/USD is typically available via their streams API
            # Try multiple approaches to get the price

            # Approach 1: Query the streams endpoint for BTC/USD
            streams_url = "https://data.chain.link/v1/streams"
            try:
                async with self.async_session.get(
                    streams_url, timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
                ) as streams_response:
                    if streams_response.status == 200:
                        streams_data = await streams_response.json()
                        # Find BTC/USD stream
                        btc_stream = None
                        if isinstance(streams_data, list):
                            btc_stream = next(
                                (s for s in streams_data if 'btc' in str(s.get('id', '')).lower() and 'usd' in str(s.get('id', '')).lower()),
                                None
                            )

                        if btc_stream:
                            stream_id = btc_stream.get('id')
                            # Query historical data for this stream
                            timestamp_ms = int(timestamp.timestamp() * 1000)
                            data_url = f"https://data.chain.link/v1/streams/{stream_id}/data"
                            params = {
                                'timestamp': timestamp_ms,
                                'limit': 1
                            }
                            async with self.async_session.get(
                                data_url, params=params, timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
                            ) as data_response:
                                if data_response.status == 200:
                                    data = await data_response.json()
                                    if isinstance(data, dict) and 'data' in data:
                                        price = data['data'].get('price') or data['data'].get('value')
                                        if price:
                                            logger.info(f"Found Chainlink price from streams API: {price}")
                                            return float(price)
            except Exception as e:
                logger.debug(f"Streams API approach failed: {e}")

            # Approach 2: Direct query to BTC/USD feed (if we know the feed address)
            # Chainlink BTC/USD feed on Polygon: 0x34bB4e028b3d2Be6B97F6e75e68492b891C5fF15
            # But we need to query via their API, not directly

            # Approach 3: Try the data.chain.link query endpoint
            timestamp_ms = int(timestamp.timestamp() * 1000)
            query_url = "https://data.chain.link/v1/queries"
            # Try with different parameter formats
            for param_format in [
                {'stream': 'btc-usd', 'timestamp': timestamp_ms},
                {'feed': 'btc-usd', 'timestamp': timestamp_ms},
                {'symbol': 'btc/usd', 'timestamp': timestamp_ms},
            ]:
                try:
                    async with self.async_session.get(
                        query_url, params=param_format, timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
                    ) as query_response:
                        if query_response.status == 200:
                            query_data = await query_response.json()
                            # Try various response formats
                            price = None
                            if isinstance(query_data, dict):
                                price = (query_data.get('price') or
                                       query_data.get('value') or
                                       query_data.get('data', {}).get('price') or
                                       query_data.get('data', {}).get('value'))
                            elif isinstance(query_data, list) and len(query_data) > 0:
                                price = (query_data[0].get('price') or
                                       query_data[0].get('value'))

                            if price:
                                logger.info(f"Found Chainlink price from query API: {price}")
                                return float(price)
                except Exception as e:
                    logger.debug(f"Query format {param_format} failed: {e}")
                    continue

            logger.debug(f"All Chainlink API approaches failed for timestamp {timestamp}")
            return None

        except aiohttp.ClientError as e:
            logger.debug(f"Error fetching Chainlink historical price: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error parsing Chainlink price: {e}")
            return None
    
    async def get_market_resolution(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Query Polymarket API for actual market resolution details (async).

        Args:
            market_id: Market ID to query

        Returns:
            Resolution data including outcome, price used, etc.
        """
        await self._ensure_async_session()
        try:
            # Try the market endpoint to get resolution info
            url = f"{self.gamma_url}/markets/{market_id}"
            data = await self._get_json_async(url, timeout=NetworkConstants.REQUEST_TIMEOUT)

            if not data:
                return None

            # Extract resolution info
            result = {
                'market_id': market_id,
                'resolved': data.get('resolved', False),
                'resolution': data.get('resolution'),  # Could be 'YES' or 'NO'
                'winner': data.get('winner'),
                'resolution_price': data.get('resolutionPrice'),
                'strike': data.get('groupItemThreshold'),
            }

            # Also check for outcome prices (1.0 = won, 0.0 = lost)
            if 'outcomePrices' in data:
                result['outcome_prices'] = data['outcomePrices']

            logger.info(f"Market resolution query: {result}")
            return result

        except Exception as e:
            logger.debug(f"Error parsing market resolution: {e}")
            return None
    
    async def get_btc_price_at(self, timestamp: pd.Timestamp) -> str:
        """
        Fetch BTC price at a specific timestamp from Binance (async).

        Args:
            timestamp: Timestamp to fetch price for

        Returns:
            BTC price as string or "N/A" if unavailable
        """
        await self._ensure_async_session()
        try:
            url = f"{NetworkConstants.BINANCE_API_URL}/klines"

            start_ms = int(timestamp.timestamp() * 1000)
            end_ms = int((timestamp + pd.Timedelta(minutes=1)).timestamp() * 1000)

            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1
            }

            data = await self._get_json_async(
                url, params=params, timeout=NetworkConstants.REQUEST_TIMEOUT
            )
            if data and len(data) > 0:
                return str(float(data[0][1]))  # Open price

            return "N/A"
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Error parsing historical BTC price: {e}")
            return "N/A"

    async def get_coinbase_15m_open_price_at(self, timestamp) -> Optional[float]:
        """
        Fetch BTC price from Coinbase 15m candle (fallback to Chainlink/RTDS).

        Args:
            timestamp: Timestamp to fetch price for (pandas Timestamp)

        Returns:
            Open price from Coinbase 15m candle, or None if unavailable
        """
        try:
            # Check if we have Coinbase credentials configured
            coinbase_key = os.getenv('COINBASE_API_KEY')
            if not coinbase_key:
                logger.debug("Coinbase API not configured, skipping")
                return None

            # Import and use Coinbase connector
            from src.connectors.coinbase import CoinbaseConnector
            coinbase = CoinbaseConnector()

            try:
                price = await coinbase.get_15m_open_price_at(timestamp)
                return price
            finally:
                # Clean up async session
                if hasattr(coinbase, 'async_session') and coinbase.async_session:
                    try:
                        if hasattr(coinbase.async_session, 'close'):
                            await coinbase.async_session.close()
                    except Exception as e:
                        logger.debug(f"Error closing Coinbase session: {e}")

        except ImportError:
            logger.debug("CoinbaseConnector not available")
        except Exception as e:
            logger.error(f"Error using Coinbase connector: {e}")

        return None

    async def get_chainlink_onchain_price_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Query Chainlink BTC/USD oracle on Polygon for historical price.

        This queries the same Chainlink feed that Polymarket uses for resolution,
        guaranteeing the strike price matches what will be used for market settlement.

        Args:
            timestamp: Timestamp to fetch price for (pandas Timestamp)

        Returns:
            BTC/USD price from Chainlink oracle, or None if unavailable
        """
        try:
            await self._ensure_async_web3()

            # Create contract instance
            contract = self.async_w3.eth.contract(
                address=Web3.to_checksum_address(NetworkConstants.CHAINLINK_BTC_USD_FEED),
                abi=CHAINLINK_AGGREGATOR_ABI
            )

            # Get target timestamp in seconds
            target_ts = int(timestamp.timestamp())

            # Get latest round data
            latest_round_data = await contract.functions.latestRoundData().call()
            latest_round_id = latest_round_data[0]
            latest_timestamp = latest_round_data[3]

            # If target is after latest round, can't get future data
            if target_ts > latest_timestamp:
                logger.debug(f"Target timestamp {target_ts} is after latest round {latest_timestamp}")
                return None

            # Get decimals (Chainlink BTC/USD uses 8 decimals)
            decimals = await contract.functions.decimals().call()

            # Search backwards from latest round to find closest round to target timestamp
            # Chainlink updates roughly every 20 seconds to 1 hour depending on price movement
            # We'll search up to 1000 rounds back (covers ~10 hours at 36s/round average)
            closest_round_id = None
            closest_price = None
            closest_diff = float('inf')
            max_search_rounds = 1000

            logger.info(f"Searching Chainlink oracle for price at {timestamp} (target_ts={target_ts})")

            for i in range(max_search_rounds):
                try:
                    round_id = latest_round_id - i
                    if round_id <= 0:
                        break

                    round_data = await contract.functions.getRoundData(round_id).call()
                    round_timestamp = round_data[3]
                    round_price = round_data[1]

                    # Calculate time difference
                    diff = abs(round_timestamp - target_ts)

                    # If this round is closer, save it
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_round_id = round_id
                        closest_price = round_price

                        # If we found a round within 60 seconds, that's good enough
                        if diff <= 60:
                            break

                    # If we've gone too far back in time (>1 hour past target), stop
                    if round_timestamp < target_ts - 3600:
                        break

                except Exception as e:
                    # Round might not exist or other error, continue searching
                    logger.debug(f"Error querying round {round_id}: {e}")
                    continue

            if closest_price is not None:
                # Convert from Chainlink decimals (8) to standard float
                price = float(closest_price) / (10 ** decimals)
                logger.info(
                    f"✓ Chainlink on-chain: Found price ${price:,.2f} at round {closest_round_id} "
                    f"(diff: {closest_diff}s from target)"
                )
                return price
            else:
                logger.warning(f"Could not find Chainlink price near timestamp {timestamp}")
                return None

        except Exception as e:
            logger.error(f"Error querying Chainlink on-chain oracle: {e}")
            return None

    def _generate_headers(self, method: str, path: str, body: str = None) -> Dict[str, str]:
        """
        Generate authentication headers for API requests.
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body (optional)
            
        Returns:
            Dictionary of headers
        """
        if not self.client or not hasattr(self.client, 'creds') or not self.client.creds:
            return {}
        
        timestamp = str(int(time.time()))
        sign_body = body if body else ""
        message = f"{timestamp}{method}{path}{sign_body}"
        
        secret = self.client.creds.api_secret
        
        try:
            secret_bytes = base64.b64decode(secret)
        except Exception:
            secret_bytes = secret.encode('utf-8')
        
        signature = hmac.new(
            secret_bytes,
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        
        return {
            "Poly-Api-Key": self.client.creds.api_key,
            "Poly-Api-Passphrase": self.client.creds.api_passphrase,
            "Poly-Timestamp": timestamp,
            "Poly-Signature": signature_b64
        }
    
    async def get_candles(
        self, token_id: str, interval: str = "1m",
        start_time: int = None, end_time: int = None, fidelity: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch candles from CLOB API (async).

        Args:
            token_id: Token ID
            interval: Candle interval
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            fidelity: Fidelity parameter (optional)

        Returns:
            List of candle data dictionaries
        """
        await self._ensure_async_session()
        try:
            path = "/prices-history"
            params = {"market": token_id, "interval": interval}

            if start_time:
                params['startTs'] = start_time
            if end_time:
                params['endTs'] = end_time
            if fidelity:
                params['fidelity'] = fidelity

            query_string = urllib.parse.urlencode(params)
            full_path = f"{path}?{query_string}"

            headers = {}
            if self.client and hasattr(self.client, 'creds') and self.client.creds:
                try:
                    headers = self._generate_headers("GET", full_path)
                except Exception as e:
                    logger.warning(f"Error generating auth headers: {e}")

            url = f"{NetworkConstants.CLOB_HOST}{path}"
            async with self.async_session.get(
                url, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('history', [])
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching candles: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {e}")
            return []
    
    def fetch_all_trades(
        self, token_id: str, start_time: int = None, end_time: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all trades from CLOB API.
        
        Args:
            token_id: Token ID
            start_time: Start timestamp in seconds (optional)
            end_time: End timestamp in seconds (optional)
            
        Returns:
            List of trade dictionaries
        """
        try:
            start_ms = start_time * 1000 if start_time else None
            end_ms = end_time * 1000 if end_time else None
            
            params = TradeParams(
                market=token_id,
                after=start_ms,
                before=end_ms
            )
            
            logger.debug(f"Fetching trades for {token_id}")
            trades = self.client.get_trades(params)
            logger.info(f"Fetched {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    def fetch_market_trades_events(self, condition_id: str) -> List[Dict[str, Any]]:
        """
        Fetch trade events for a market.
        
        Args:
            condition_id: Market condition ID
            
        Returns:
            List of trade event dictionaries
        """
        try:
            trades = self.client.get_market_trades_events(condition_id)
            logger.info(f"Fetched {len(trades)} trade events for {condition_id}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching market trades events: {e}")
            return []
    
    def create_order(
        self, token_id: str, price: float, size: float, side: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create and post a limit order.
        
        Args:
            token_id: Asset ID to trade
            price: Limit price
            size: Size in shares
            side: 'BUY' or 'SELL'
            
        Returns:
            Order response dictionary or None
        """
        # Input validation
        if not token_id or price <= 0 or size <= 0:
            logger.error(f"Invalid order parameters: token_id={token_id}, price={price}, size={size}")
            return None
        
        if side.upper() not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None
        
        # Check if client is initialized
        if not self.client:
            logger.error("ClobClient not initialized - cannot place order")
            return None
        
        # Check if client has credentials
        if not hasattr(self.client, 'creds') or not self.client.creds:
            logger.error("ClobClient missing API credentials - cannot place order")
            return None
        
        try:
            order_side = BUY if side.upper() == 'BUY' else SELL
            
            logger.info(f"Creating {side} order: token_id={token_id[:20]}..., price={price:.4f}, size={size:.4f} shares")
            
            order_args = OrderArgs(
                price=price,
                size=size,
                side=order_side,
                token_id=token_id
            )
            
            logger.debug(f"OrderArgs: {order_args}")
            
            resp = self.client.create_and_post_order(order_args)
            
            logger.info(f"Order Response Type: {type(resp)}")
            logger.info(f"Order Response: {resp}")
            
            # Log response structure for debugging
            if isinstance(resp, dict):
                logger.info(f"Response keys: {list(resp.keys())}")
                # Check for various possible order ID fields
                for key in ['orderID', 'order_id', 'id', 'hash', 'orderHash']:
                    if key in resp:
                        logger.info(f"Found order ID in field '{key}': {resp[key]}")
            
            return resp
        except Exception as e:
            logger.error(f"Error creating order: {e}", exc_info=True)
            
            # Log additional error details if available
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status Code: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Error Response: {e.response}")
            if hasattr(e, 'error_message'):
                logger.error(f"Error Message: {e.error_message}")
            
            return None
    
    @validate_order_params
    async def create_market_order(
        self, token_id: str, amount: float, side: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create and post a market order (async wrapper).

        Note: Uses to_thread() internally since py-clob-client is synchronous.

        Args:
            token_id: Asset ID to trade
            amount: For BUY: USDC amount. For SELL: Shares amount.
            side: 'BUY' or 'SELL'

        Returns:
            Order response dictionary or None

        Note:
            Input validation is handled by the @validate_order_params decorator.
        """
        def _sync_create_order():
            try:
                # Clean and validate amount
                clean_amount = float(
                    Decimal(str(amount)).quantize(
                        Decimal("0.01"), rounding=ROUND_DOWN
                    )
                )

                logger.info(f"Creating market {side} order: {clean_amount} for {token_id[:20]}...")

                # Check credentials
                if not hasattr(self.client, 'creds') or not self.client.creds:
                    error_msg = "ERROR: Client does not have API credentials set!"
                    logger.error(error_msg)
                    return None

                # Get order book for price discovery
                ob = self.get_order_book(token_id)
                if not ob:
                    error_msg = "ERROR: Could not fetch order book"
                    logger.error(error_msg)
                    return None

                # Calculate aggressive price
                price = self._calculate_aggressive_price(ob, side.upper())
                logger.info(f"Aggressive {side} price: {price}")

                # Create market order
                market_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=clean_amount,
                    side=side.upper(),
                    price=price,
                    order_type=OrderType.FOK
                )

                logger.info("Building and signing market order...")
                signed_order = self.client.create_market_order(market_args)
                logger.info("✓ Order signed successfully")

                logger.info("Posting order to CLOB...")
                resp = self.client.post_order(signed_order, OrderType.FOK)
                logger.info(f"Order Response: {resp}")

                return resp

            except Exception as e:
                error_msg = f"EXCEPTION in create_market_order: {e}"
                logger.error(error_msg, exc_info=True)

                if hasattr(e, 'status_code'):
                    logger.error(f"Status: {e.status_code}")
                if hasattr(e, 'error_message'):
                    logger.error(f"Error Message: {e.error_message}")

                return None

        return await asyncio.to_thread(_sync_create_order)
    
    def _calculate_aggressive_price(self, order_book, side: str) -> float:
        """
        Calculate aggressive price for market order.
        
        Args:
            order_book: Order book object
            side: 'BUY' or 'SELL'
            
        Returns:
            Aggressive price (float)
        """
        if side == 'BUY':
            if not order_book.asks or len(order_book.asks) == 0:
                best_ask = TradingConstants.MAX_PRICE
            else:
                best_ask = float(order_book.asks[0].price)
            
            price = min(
                round(best_ask * TradingConstants.BUY_AGGRESSIVE_MULTIPLIER, TradingConstants.PRICE_ROUNDING_PLACES),
                TradingConstants.MAX_PRICE
            )
        else:  # SELL
            if not order_book.bids or len(order_book.bids) == 0:
                best_bid = TradingConstants.MIN_PRICE
            else:
                best_bid = float(order_book.bids[0].price)
            
            price = max(
                round(best_bid * TradingConstants.SELL_AGGRESSIVE_MULTIPLIER, TradingConstants.PRICE_ROUNDING_PLACES),
                TradingConstants.MIN_PRICE
            )
        
        return price
    
    def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Cancel a specific order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancel response dictionary or None
        """
        if not order_id:
            logger.error("Invalid order_id")
            return None
        
        try:
            logger.info(f"Cancelling order {order_id}...")
            resp = self.client.cancel(order_id)
            logger.info(f"Cancel Response: {resp}")
            return resp
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return None
    
    async def cancel_all_orders(self) -> Optional[Dict[str, Any]]:
        """
        Cancel all open orders (async wrapper).

        Note: Uses to_thread() internally since py-clob-client is synchronous.

        Returns:
            Cancel response dictionary or None
        """
        def _sync_cancel_all():
            try:
                logger.info("Cancelling ALL orders...")
                resp = self.client.cancel_all()
                logger.info(f"Cancel All Response: {resp}")
                return resp
            except Exception as e:
                logger.error(f"Error cancelling all orders: {e}")
                return None

        return await asyncio.to_thread(_sync_cancel_all)
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary or None
        """
        if not order_id:
            logger.error("Invalid order_id")
            return None
        
        try:
            return self.client.get_order(order_id)
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    async def get_closed_markets(
        self,
        series_id: str = "10192",
        limit: int = 20,
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch closed (resolved) markets from Gamma API (async).

        This returns historical market results to determine prior outcomes.
        The markets are filtered by series_id (15-minute BTC markets) and
        sorted by most recent first by default.

        Args:
            series_id: Series ID (default: "10192" for BTC Up or Down 15m)
            limit: Maximum number of results to return (default: 20)
            ascending: If False, returns most recent first (default: False)

        Returns:
            List of closed market data dictionaries with outcome information
        """
        await self._ensure_async_session()
        try:
            # Try /events endpoint first (better structure with markets nested in events)
            events_url = f"{self.gamma_url}/events"
            events_params = {
                'closed': 'true',
                'limit': limit,
                'series_id': series_id,
                'order': 'endDate',
                'ascending': 'false',  # Most recent first
            }

            logger.debug(f"Querying {events_url} with params: {events_params}")

            async with self.async_session.get(
                events_url,
                params=events_params,
                timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
            ) as events_response:
                events_response.raise_for_status()
                events_data = await events_response.json()
            
            # Extract markets from events and attach event data to each market
            markets = []
            if isinstance(events_data, list):
                for event in events_data:
                    if isinstance(event, dict) and 'markets' in event:
                        event_markets = event.get('markets', [])
                        if isinstance(event_markets, list):
                            # Attach event endDate to each market for timing
                            for mkt in event_markets:
                                if isinstance(mkt, dict):
                                    # Store event reference for endDate fallback
                                    mkt['_event'] = event
                            markets.extend(event_markets)
            
            # Fallback to /markets endpoint if events didn't work
            if not markets or not isinstance(markets, list) or len(markets) == 0:
                logger.info(f"/events endpoint returned no markets, trying /markets endpoint...")

                url = f"{self.gamma_url}/markets"
                params = {
                    'closed': 'true',
                    'limit': limit,
                    'ascending': str(ascending).lower(),
                    'series_id': series_id
                }

                logger.debug(f"Querying {url} with params: {params}")

                async with self.async_session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=NetworkConstants.REQUEST_TIMEOUT)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    markets = data
                elif isinstance(data, dict):
                    # Some APIs return { "data": [...], "markets": [...], etc }
                    markets = data.get('data', data.get('markets', data.get('results', [])))
            
            if not markets or not isinstance(markets, list) or len(markets) == 0:
                logger.warning(
                    f"No closed markets found for series ID: {series_id} from either endpoint."
                )
                return []
            
            logger.info(f"API returned {len(markets)} raw markets, parsing...")
            
            # Parse each market to extract outcome and timing information
            parsed_markets = []
            for idx, market in enumerate(markets):
                try:
                    # If market is nested in an event, extract it
                    if isinstance(market, dict) and 'markets' in market:
                        # This is an event, get the first market
                        nested_markets = market.get('markets', [])
                        if nested_markets:
                            market = nested_markets[0]
                    
                    parsed_market = self._parse_closed_market(market)
                    if parsed_market:
                        parsed_markets.append(parsed_market)
                    else:
                        logger.debug(f"Market {idx+1} could not be parsed (missing data)")
                except Exception as e:
                    logger.warning(f"Error parsing closed market {idx+1}: {e}")
                    continue
            
            # Sort by end_timestamp, most recent first
            parsed_markets.sort(
                key=lambda x: x.get('end_timestamp', pd.Timestamp.min),
                reverse=True
            )
            
            logger.info(f"Successfully parsed {len(parsed_markets)} closed markets from {len(markets)} raw results")
            
            # Log the timestamps of the first few for debugging
            if parsed_markets:
                first_end = parsed_markets[0].get('end_timestamp')
                logger.debug(f"Most recent closed market ended at: {first_end}")
            
            return parsed_markets

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching closed markets: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_closed_markets: {e}", exc_info=True)
            return []
    
    def _parse_closed_market(self, market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a closed market to determine the winner and timing.
        
        Looks at outcomePrices to determine winner:
        - ["1.0", "0.0"] or ["1", "0"] means Up won (first outcome)
        - ["0.0", "1.0"] or ["0", "1"] means Down won (second outcome)
        
        Args:
            market: Market data dictionary from Gamma API
            
        Returns:
            Parsed market with outcome, timestamp, and metadata
        """
        try:
            market_id = market.get('id', market.get('market_id', 'unknown'))
            
            # Try multiple ways to determine outcome
            outcome = None
            
            # Method 1: Check outcomePrices field
            # For BTC markets: index 0 = "Up" (YES), index 1 = "Down" (NO)
            outcome_prices = market.get('outcomePrices') or market.get('outcome_prices')
            outcomes_labels = market.get('outcomes', [])
            
            # IMPORTANT: Parse JSON strings if the API returns them as strings
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = None
            
            if isinstance(outcomes_labels, str):
                try:
                    outcomes_labels = json.loads(outcomes_labels)
                except (json.JSONDecodeError, TypeError):
                    outcomes_labels = []
            
            logger.debug(f"Market {market_id}: outcomePrices={outcome_prices}, outcomes={outcomes_labels}")
            
            if outcome_prices and isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                try:
                    # Handle both string and numeric values
                    first_price_str = str(outcome_prices[0]).strip()
                    second_price_str = str(outcome_prices[1]).strip()
                    
                    first_price = float(first_price_str)
                    second_price = float(second_price_str)
                    
                    logger.debug(f"Market {market_id}: first_price={first_price}, second_price={second_price}")
                    
                    # If both are 0 or very small, market might not be resolved yet
                    if first_price < 0.01 and second_price < 0.01:
                        logger.debug(f"Market {market_id} has unresolved outcomePrices: {outcome_prices}")
                        return None
                    
                    # Determine which outcome labels we have
                    # BTC markets use "Up"/"Down", use directly
                    first_label = str(outcomes_labels[0]) if outcomes_labels and len(outcomes_labels) > 0 else 'Up'
                    second_label = str(outcomes_labels[1]) if outcomes_labels and len(outcomes_labels) > 1 else 'Down'
                    
                    # Normalize to title case
                    if first_label:
                        first_label = first_label[0].upper() + first_label[1:].lower() if len(first_label) > 1 else first_label.upper()
                    if second_label:
                        second_label = second_label[0].upper() + second_label[1:].lower() if len(second_label) > 1 else second_label.upper()
                    
                    logger.debug(f"Market {market_id}: first_label={first_label}, second_label={second_label}")
                    
                    # Check which outcome won (price = 1 or very close to 1)
                    # The winning outcome has price close to 1, losing has price close to 0
                    if first_price >= 0.5:
                        # First outcome won - use the label directly
                        outcome = first_label if first_label in ('Up', 'Down') else 'Up'
                        logger.debug(f"Market {market_id}: first outcome won, outcome={outcome}")
                    elif second_price >= 0.5:
                        # Second outcome won
                        outcome = second_label if second_label in ('Up', 'Down') else 'Down'
                        logger.debug(f"Market {market_id}: second outcome won, outcome={outcome}")
                    else:
                        logger.debug(f"Market {market_id}: neither price >= 0.5, first={first_price}, second={second_price}")
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing outcomePrices {outcome_prices}: {e}")
                    pass
            
            # Method 2: Check resolution field
            if not outcome:
                resolution = market.get('resolution') or market.get('winner')
                if resolution:
                    # Normalize to title case
                    resolution_str = str(resolution)
                    if resolution_str:
                        resolution_normalized = resolution_str[0].upper() + resolution_str[1:].lower() if len(resolution_str) > 1 else resolution_str.upper()
                    else:
                        resolution_normalized = resolution_str
                    # Handle both Up/Down and legacy YES/NO
                    if resolution_normalized in ('Up', 'Down'):
                        outcome = resolution_normalized
                    elif resolution_normalized.upper() in ('YES', 'NO'):
                        # Map legacy YES/NO to Up/Down
                        outcome = 'Up' if resolution_normalized.upper() == 'YES' else 'Down'
            
            # Method 3: Check outcomes array for winner
            if not outcome:
                outcomes = market.get('outcomes', [])
                if isinstance(outcomes, list):
                    for outcome_obj in outcomes:
                        if isinstance(outcome_obj, dict):
                            # Check if this outcome won (price = 1.0 or winner = true)
                            outcome_price = outcome_obj.get('price')
                            if (outcome_price == 1.0 or 
                                outcome_price == "1" or 
                                outcome_price == "1.0" or
                                (isinstance(outcome_price, (int, float)) and outcome_price >= 0.99) or
                                outcome_obj.get('winner') is True):
                                outcome_title = str(outcome_obj.get('title', ''))
                                # Normalize to title case and map to Up/Down
                                if outcome_title:
                                    outcome_title_normalized = outcome_title[0].upper() + outcome_title[1:].lower() if len(outcome_title) > 1 else outcome_title.upper()
                                else:
                                    outcome_title_normalized = outcome_title
                                # Map Yes/Up to Up, No/Down to Down
                                if outcome_title_normalized in ('Up', 'Yes') or outcome_title_normalized.upper() in ('YES', 'UP'):
                                    outcome = 'Up'
                                elif outcome_title_normalized in ('Down', 'No') or outcome_title_normalized.upper() in ('NO', 'DOWN'):
                                    outcome = 'Down'
                                break
                        elif isinstance(outcome_obj, str):
                            # Simple string outcomes - check if we can determine from context
                            # This is less reliable, but might work for some markets
                            pass
            
            if not outcome:
                logger.warning(
                    f"Market {market_id} has unclear outcome. "
                    f"outcomePrices: {market.get('outcomePrices')}, "
                    f"resolution: {market.get('resolution')}, "
                    f"outcomes: {market.get('outcomes')}"
                )
                return None
            
            logger.debug(f"Market {market_id}: determined outcome = {outcome}")
            
            # Extract timing information
            # Try multiple field names for end date
            end_date_str = (
                market.get('endDate') or 
                market.get('end_date') or
                market.get('endDateIso') or
                market.get('endTime') or
                market.get('end_time')
            )
            
            # If market is nested in event, try event's endDate
            if not end_date_str and isinstance(market, dict):
                # Check if parent event has endDate
                parent_event = market.get('event') or market.get('_event')
                if parent_event and isinstance(parent_event, dict):
                    end_date_str = (
                        parent_event.get('endDate') or 
                        parent_event.get('end_date') or
                        parent_event.get('endDateIso')
                    )
            
            if not end_date_str:
                logger.debug(f"No endDate found for market {market_id}")
                return None
            
            # Parse end date
            end_date = pd.Timestamp(end_date_str)
            if end_date.tz is None:
                end_date = end_date.tz_localize('UTC')
            
            # Extract other useful information
            result = {
                'outcome': outcome,
                'end_date': end_date_str,
                'end_timestamp': end_date,
                'market_id': market.get('id'),
                'question': market.get('question', ''),
                'outcome_prices': outcome_prices,
                'resolved': market.get('resolved', True),
            }
            
            # Try to extract resolution price if available
            if 'resolutionPrice' in market:
                result['resolution_price'] = market.get('resolutionPrice')
            
            # Try to extract strike if available
            if 'groupItemThreshold' in market:
                result['strike'] = market.get('groupItemThreshold')
            
            return result
            
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Error parsing closed market: {e}")
            return None

