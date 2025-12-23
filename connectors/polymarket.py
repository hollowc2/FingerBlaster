
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
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from web3 import Web3

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    TradeParams, MarketOrderArgs, OrderType, 
    BalanceAllowanceParams, AssetType, OrderArgs
)
from py_clob_client.order_builder.constants import BUY, SELL

from connectors.base import DataConnector

load_dotenv()

# Constants - Extract magic numbers to named constants
class TradingConstants:
    """Trading-related constants."""
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
    """Network and API constants."""
    POLYGON_CHAIN_ID = 137
    POLYGON_RPC_URL = "https://polygon-rpc.com"
    CLOB_HOST = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    USDC_CONTRACT_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3

class SignatureType:
    """Signature type constants."""
    EOA = 0
    GNOSIS_SAFE = 2

# Configure logger
logger = logging.getLogger("PolymarketConnector")


class PolymarketConnector(DataConnector):

    def __init__(self):
        """Initialize the Polymarket connector with proper error handling."""
        self.signature_type = SignatureType.EOA
        self.client: Optional[ClobClient] = None
        self.gamma_url = NetworkConstants.GAMMA_API_URL
        
        # Create session with connection pooling and retries
        self.session = self._create_session()
        
        # Initialize client
        self._initialize_client()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with connection pooling and retry strategy.
        
        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=NetworkConstants.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _initialize_client(self) -> None:
        """
        Initialize the ClobClient with proper authentication.
        
        Separated from __init__ to follow SRP and improve testability.
        """
        key = os.getenv("PRIVATE_KEY")
        
        if not key:
            logger.info("No private key found, initializing public client")
            self.client = ClobClient(NetworkConstants.CLOB_HOST)
            self.signature_type = SignatureType.EOA
            return
        
        try:
            logger.info("Initializing authenticated ClobClient...")
            api_creds = self._derive_api_credentials(key)
            signature_type, funder_address = self._determine_signature_type(key)
            
            self.client = self._create_authenticated_client(
                key, api_creds, signature_type, funder_address
            )
            self.signature_type = signature_type
            
            self._configure_client_headers()
            self._check_initial_balance()
            
        except Exception as e:
            logger.error(f"Error initializing authenticated client: {e}", exc_info=True)
            logger.warning("Falling back to client without L2 credentials")
            self.client = ClobClient(
                NetworkConstants.CLOB_HOST,
                key=key,
                chain_id=NetworkConstants.POLYGON_CHAIN_ID,
                signature_type=SignatureType.EOA
            )
            self.signature_type = SignatureType.EOA
    
    def _derive_api_credentials(self, key: str):
        """
        Derive API credentials for L2 authentication.
        
        Args:
            key: Private key string
            
        Returns:
            API credentials object
            
        Raises:
            Exception: If credential derivation fails
        """
        logger.info("Step 1: Deriving API credentials for L2 authentication...")
        temp_client = ClobClient(
            NetworkConstants.CLOB_HOST,
            key=key,
            chain_id=NetworkConstants.POLYGON_CHAIN_ID
        )
        
        try:
            api_creds = temp_client.create_or_derive_api_creds()
            logger.info(f"✓ API Credentials derived successfully! Key: {api_creds.api_key[:10]}...")
            return api_creds
        except Exception as e:
            logger.error(f"✗ Error deriving API creds: {e}")
            raise
    
    def _determine_signature_type(self, key: str) -> Tuple[int, Optional[str]]:
        """
        Determine signature type and funder address based on configuration.
        
        Args:
            key: Private key string
            
        Returns:
            Tuple of (signature_type, funder_address)
        """
        logger.info("Step 2: Determining signature type...")
        
        w3 = Web3(Web3.HTTPProvider(NetworkConstants.POLYGON_RPC_URL))
        account = w3.eth.account.from_key(key)
        signer_address = account.address
        
        env_address = os.getenv("WALLET_ADDRESS")
        signature_type = SignatureType.EOA
        funder_address = None
        
        if env_address and env_address.lower() != signer_address.lower():
            try:
                funder_address = Web3.to_checksum_address(env_address)
                signature_type = SignatureType.GNOSIS_SAFE
                
                logger.info(f"Detected Proxy Setup:")
                logger.info(f"  - Signer (Private Key): {signer_address}")
                logger.info(f"  - Proxy (Funds):        {funder_address}")
                logger.info("  - Signature Type: 2 (GNOSIS_SAFE)")
                logger.warning("  ⚠ NOTE: Signer EOA must have POL for gas!")
            except Exception as e:
                logger.error(f"✗ Error setting up proxy address: {e}")
                signature_type = SignatureType.EOA
                funder_address = None
        else:
            logger.info(f"Detected Direct EOA Setup:")
            logger.info(f"  - Address: {signer_address}")
            logger.info("  - Signature Type: 0 (EOA)")
        
        return signature_type, funder_address
    
    def _create_authenticated_client(
        self, key: str, api_creds, signature_type: int, funder_address: Optional[str]
    ) -> ClobClient:
        """
        Create authenticated ClobClient instance.
        
        Args:
            key: Private key
            api_creds: API credentials
            signature_type: Signature type (EOA or GNOSIS_SAFE)
            funder_address: Optional funder address for proxy wallets
            
        Returns:
            Configured ClobClient instance
        """
        logger.info("Step 3: Initializing ClobClient with L1 + L2 authentication...")
        
        client_kwargs = {
            "host": NetworkConstants.CLOB_HOST,
            "key": key,
            "chain_id": NetworkConstants.POLYGON_CHAIN_ID,
            "creds": api_creds,
            "signature_type": signature_type
        }
        
        if funder_address:
            client_kwargs["funder"] = funder_address
        
        client = ClobClient(**client_kwargs)
        logger.info("✓ ClobClient initialized with L1 + L2 authentication!")
        
        return client
    
    def _configure_client_headers(self) -> None:
        """Configure custom headers to avoid Cloudflare 403 errors."""
        try:
            if not hasattr(self.client, 'http_client') or not hasattr(self.client.http_client, 'session'):
                return
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://polymarket.com/",
                "Origin": "https://polymarket.com",
                "Sec-Ch-Ua": '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site"
            }
            
            self.client.http_client.session.headers.update(headers)
            logger.info("✓ Updated ClobClient headers with full browser set.")
        except Exception as e:
            logger.warning(f"Could not update headers: {e}")
    
    def _check_initial_balance(self) -> None:
        """Check and log initial balance/allowances."""
        try:
            b_params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self.signature_type
            )
            balance_info = self.client.get_balance_allowance(b_params)
            logger.info(f"✓ Wallet Balance (API): {balance_info}")
        except Exception as e:
            logger.warning(f"Could not fetch balance via API: {e}")
    
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
    
    def _extract_strike_price(self, market: Dict[str, Any], event: Dict[str, Any]) -> str:
        """
        Extract strike price from market data using multiple strategies.
        
        Improved to check more fields and be more accurate.
        For dynamic strikes, uses RTDS or fetches price at exact market start time.
        
        Args:
            market: Market data dictionary
            event: Event data dictionary
            
        Returns:
            Strike price as string
        """
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
        
        # Strategy 0: Check groupItemThreshold FIRST (this is likely the strike price Polymarket uses!)
        if 'groupItemThreshold' in market:
            threshold = market.get('groupItemThreshold')
            logger.debug(f"Checking groupItemThreshold value: {threshold} (type: {type(threshold)})")
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
                        logger.info(f"✓ Found strike in groupItemThreshold: {threshold_val:,.2f}")
                        return f"{threshold_val:,.2f}"
                    else:
                        logger.debug(f"groupItemThreshold is zero or negative: {threshold_val}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse groupItemThreshold '{threshold}': {e}")
        else:
            logger.debug("groupItemThreshold not found in market data")
        
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
        Build token map (YES/NO -> token_id) from market data.
        
        Extracted from duplicate code to follow DRY principle.
        
        Args:
            market: Market data dictionary
            clob_token_ids: List of CLOB token IDs
            
        Returns:
            Dictionary mapping 'YES' and 'NO' to token IDs
        """
        token_map = {}
        yes_token_id = None
        no_token_id = None
        
        # Try to get from tokens array
        tokens_data = market.get('tokens', [])
        if tokens_data:
            for token in tokens_data:
                outcome = token.get('outcome', '').upper()
                if outcome == 'YES':
                    yes_token_id = token.get('tokenId') or token.get('clobTokenId')
                elif outcome == 'NO':
                    no_token_id = token.get('tokenId') or token.get('clobTokenId')
        
        # Build map from found tokens
        if yes_token_id:
            token_map['YES'] = yes_token_id
        if no_token_id:
            token_map['NO'] = no_token_id
        
        # Fallback to clob_token_ids list if needed
        if 'YES' not in token_map and len(clob_token_ids) > 0:
            token_map['YES'] = clob_token_ids[0]
        if 'NO' not in token_map and len(clob_token_ids) > 1:
            token_map['NO'] = clob_token_ids[1]
        
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
        
        # Extract strike price
        strike_price = self._extract_strike_price(market, event)
        
        # Build response
        result = {
            'event_slug': event.get('slug'),
            'market_id': market.get('id'),
            'token_id': clob_token_ids[0],  # Primary token
            'token_ids': clob_token_ids,
            'end_date': event.get('endDate'),
            'strike_price': str(strike_price)
        }
        
        # Include token map if requested
        if include_token_map:
            result['token_map'] = self._build_token_map(market, clob_token_ids)
        
        return result
    
    def get_token_balance(self, token_id: str) -> float:
        """
        Get the balance of a specific outcome token.
        
        Args:
            token_id: Token ID to check balance for
            
        Returns:
            Balance in shares (float)
        """
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
    
    def flatten_market(self, token_map: Dict[str, str]) -> List[Any]:
        """
        Close all positions in the given token map (YES and NO).
        
        Args:
            token_map: Dictionary mapping 'YES'/'NO' to token IDs
            
        Returns:
            List of order responses
        """
        results = []
        for outcome, token_id in token_map.items():
            logger.info(f"Checking balance for {outcome} ({token_id[:10]}...)")
            balance = self.get_token_balance(token_id)
            
            if balance > TradingConstants.MIN_BALANCE_THRESHOLD:
                logger.info(f"Found {balance} shares of {outcome}. Flattening...")
                resp = self.create_market_order(token_id, balance, 'SELL')
                results.append(resp)
            else:
                logger.info(f"No significant balance for {outcome}.")
        
        return results
    
    def get_usdc_balance(self) -> float:
        """
        Get USDC balance from wallet.
        
        Tries Web3 direct blockchain query first, falls back to CLOB API.
        
        Returns:
            USDC balance (float)
        """
        # Try Web3 first (most reliable)
        try:
            w3 = Web3(Web3.HTTPProvider(NetworkConstants.POLYGON_RPC_URL))
            key = os.getenv("PRIVATE_KEY")
            env_address = os.getenv("WALLET_ADDRESS")
            
            target_address = None
            if env_address:
                target_address = env_address
            elif key:
                account = w3.eth.account.from_key(key)
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
                
                contract = w3.eth.contract(
                    address=NetworkConstants.USDC_CONTRACT_ADDRESS,
                    abi=usdc_abi
                )
                
                target_address = Web3.to_checksum_address(target_address)
                balance_wei = contract.functions.balanceOf(target_address).call()
                return balance_wei / (10 ** TradingConstants.USDC_DECIMALS)
        except Exception as e:
            logger.debug(f"Web3 balance check failed: {e}")
        
        # Fallback to CLOB API
        try:
            balance_info = self.client.get_balance_allowance()
            return float(balance_info.get('balance', 0))
        except Exception as e:
            logger.error(f"Both Web3 and CLOB API balance checks failed: {e}")
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
    
    def get_market_data(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get market data from Gamma API.
        
        Args:
            market_id: Market ID or slug
            
        Returns:
            Market data dictionary or None
        """
        try:
            if "-" in market_id and not market_id.isdigit():
                url = f"{self.gamma_url}/events?slug={market_id}"
            else:
                url = f"{self.gamma_url}/markets/{market_id}"
            
            response = self.session.get(url, timeout=NetworkConstants.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return data
        except requests.RequestException as e:
            logger.error(f"Error fetching market data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching market data: {e}")
            return None
    
    def get_active_market(self, series_id: str = "10192") -> Optional[Dict[str, Any]]:
        """
        Fetch the currently active market for a given series ID.
        
        Args:
            series_id: Series ID (default: "10192" for BTC Up or Down 15m)
            
        Returns:
            Market data dictionary or None
        """
        try:
            url = f"{self.gamma_url}/events?limit=100&closed=false&series_id={series_id}"
            response = self.session.get(url, timeout=NetworkConstants.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            events = response.json()
            if not events:
                logger.info(f"No active events found for series ID: {series_id}")
                return None
            
            # Filter valid future events
            now = pd.Timestamp.now(tz='UTC')
            valid_events = [
                e for e in events
                if pd.Timestamp(e['endDate']) > now
            ]
            
            if not valid_events:
                logger.info("No valid future events found.")
                return None
            
            # Sort by end date and get the soonest one
            valid_events.sort(key=lambda x: pd.Timestamp(x['endDate']))
            target_event = valid_events[0]
            
            # Try to get more detailed market data that might have the actual strike
            market_data = self._parse_market_data(target_event, include_token_map=True)
            
            # If we have a market_id, try to fetch detailed market info
            # Also check if there's a resolution endpoint that has the strike
            if market_data and market_data.get('market_id'):
                try:
                    # Try the market details endpoint
                    detailed_market = self.get_market_data(market_data['market_id'])
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
                                            market_data['strike_price'] = f"{threshold_val:,.2f}"
                                            return market_data
                                    except (ValueError, TypeError):
                                        pass
                            
                            detailed_strike = self._extract_strike_price(
                                detailed_market.get('markets', [{}])[0] if detailed_market.get('markets') else detailed_market,
                                detailed_market if 'endDate' in detailed_market else target_event
                            )
                            if detailed_strike and detailed_strike != "N/A":
                                logger.info(f"Found strike from detailed market data: {detailed_strike}")
                                market_data['strike_price'] = detailed_strike
                    
                    # Also try querying the market's resolution endpoint if it exists
                    # Some markets might expose the strike via a resolution query
                    try:
                        resolution_url = f"{self.gamma_url}/markets/{market_data['market_id']}/resolution"
                        resolution_response = self.session.get(
                            resolution_url, timeout=NetworkConstants.REQUEST_TIMEOUT
                        )
                        if resolution_response.status_code == 200:
                            resolution_data = resolution_response.json()
                            # Check if resolution data contains strike
                            strike = (resolution_data.get('strike') or 
                                    resolution_data.get('threshold') or
                                    resolution_data.get('price'))
                            if strike:
                                logger.info(f"Found strike from resolution endpoint: {strike}")
                                market_data['strike_price'] = str(strike)
                    except Exception as e:
                        logger.debug(f"Resolution endpoint check failed (may not exist): {e}")
                        
                except Exception as e:
                    logger.debug(f"Could not fetch detailed market data: {e}")
            
            return market_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching active market: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_active_market: {e}", exc_info=True)
            return None
    
    def get_next_market(
        self, current_end_date_str: str, series_id: str = "10192"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch the next market that ends AFTER the provided current_end_date.
        
        Args:
            current_end_date_str: Current market end date string
            series_id: Series ID (default: "10192")
            
        Returns:
            Market data dictionary or None
        """
        try:
            url = f"{self.gamma_url}/events?limit=100&closed=false&series_id={series_id}"
            response = self.session.get(url, timeout=NetworkConstants.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            events = response.json()
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
            
        except requests.RequestException as e:
            logger.error(f"Error fetching next market: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_next_market: {e}", exc_info=True)
            return None
    
    def get_btc_price(self) -> float:
        """
        Fetch current BTC price from Binance API.
        
        Returns:
            BTC price in USDT (float)
        """
        try:
            url = f"{NetworkConstants.BINANCE_API_URL}/ticker/price"
            params = {'symbol': 'BTCUSDT'}
            response = self.session.get(
                url, params=params, timeout=NetworkConstants.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.error(f"Error fetching BTC price from Binance: {e}")
            return 0.0
    
    def get_chainlink_price_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Fetch Chainlink BTC/USD price at a specific timestamp.
        
        Uses Chainlink Data Streams API to get historical prices.
        Based on: https://data.chain.link/streams/btc-usd
        
        Args:
            timestamp: Timestamp to fetch price for
            
        Returns:
            Chainlink BTC/USD price as float or None if unavailable
        """
        try:
            # Chainlink Data Streams API for BTC/USD
            # The stream ID for BTC/USD is typically available via their streams API
            # Try multiple approaches to get the price
            
            # Approach 1: Query the streams endpoint for BTC/USD
            streams_url = "https://data.chain.link/v1/streams"
            try:
                streams_response = self.session.get(
                    streams_url, timeout=NetworkConstants.REQUEST_TIMEOUT
                )
                if streams_response.status_code == 200:
                    streams_data = streams_response.json()
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
                        data_response = self.session.get(
                            data_url, params=params, timeout=NetworkConstants.REQUEST_TIMEOUT
                        )
                        if data_response.status_code == 200:
                            data = data_response.json()
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
                    query_response = self.session.get(
                        query_url, params=param_format, timeout=NetworkConstants.REQUEST_TIMEOUT
                    )
                    if query_response.status_code == 200:
                        query_data = query_response.json()
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
            
        except requests.RequestException as e:
            logger.debug(f"Error fetching Chainlink historical price: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error parsing Chainlink price: {e}")
            return None
    
    def get_btc_price_at(self, timestamp: pd.Timestamp) -> str:
        """
        Fetch BTC price at a specific timestamp from Binance.
        
        Args:
            timestamp: Timestamp to fetch price for
            
        Returns:
            BTC price as string or "N/A" if unavailable
        """
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
            
            response = self.session.get(
                url, params=params, timeout=NetworkConstants.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                return str(float(data[0][1]))  # Open price
            
            return "N/A"
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            logger.error(f"Error fetching historical BTC price: {e}")
            return "N/A"
    
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
    
    def get_candles(
        self, token_id: str, interval: str = "1m",
        start_time: int = None, end_time: int = None, fidelity: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch candles from CLOB API.
        
        Args:
            token_id: Token ID
            interval: Candle interval
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            fidelity: Fidelity parameter (optional)
            
        Returns:
            List of candle data dictionaries
        """
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
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get('history', [])
        except requests.RequestException as e:
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
            logger.error("Invalid order parameters")
            return None
        
        if side.upper() not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None
        
        try:
            order_side = BUY if side.upper() == 'BUY' else SELL
            
            logger.info(f"Creating {side} order for {size} shares at {price}")
            
            order_args = OrderArgs(
                price=price,
                size=size,
                side=order_side,
                token_id=token_id
            )
            
            resp = self.client.create_and_post_order(order_args)
            logger.info(f"Order Response: {resp}")
            return resp
        except Exception as e:
            logger.error(f"Error creating order: {e}", exc_info=True)
            return None
    
    def create_market_order(
        self, token_id: str, amount: float, side: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create and post a market order.
        
        Args:
            token_id: Asset ID to trade
            amount: For BUY: USDC amount. For SELL: Shares amount.
            side: 'BUY' or 'SELL'
            
        Returns:
            Order response dictionary or None
        """
        # Input validation
        if not token_id or amount <= 0:
            logger.error(f"Invalid market order parameters: token_id={token_id}, amount={amount}")
            return None
        
        if side.upper() not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None
        
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
    
    def cancel_all_orders(self) -> Optional[Dict[str, Any]]:
        """
        Cancel all open orders.
        
        Returns:
            Cancel response dictionary or None
        """
        try:
            logger.info("Cancelling ALL orders...")
            resp = self.client.cancel_all()
            logger.info(f"Cancel All Response: {resp}")
            return resp
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return None
    
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

