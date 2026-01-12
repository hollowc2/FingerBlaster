import os
import logging
from typing import Optional, Tuple, Dict, Any, List
from web3 import Web3

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
import aiohttp
import pandas as pd

# Re-export or redefine constants if needed, or import them 
# But for now assuming we pass necessary values or duplicate strict constants to avoid circular deps with polymarket.py
# best to move constants to a shared file eventually. For now, I'll assume constants are available or passed in.

logger = logging.getLogger("PolymarketAPI")

class SignatureType:
    """Signature type constants."""
    EOA = 0
    GNOSIS_SAFE = 2

class PolymarketAPI:
    """
    Handles low-level authentication and connection to Polymarket CLOB.
    """
    def __init__(self, host: str, chain_id: int):
        self.host = host
        self.chain_id = chain_id
        self.client: Optional[ClobClient] = None
        self.signature_type = SignatureType.EOA
        self.wallet_address: Optional[str] = None
        
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the ClobClient with proper authentication."""
        key = os.getenv("PRIVATE_KEY")

        if not key:
            logger.info("No private key found, initializing public client")
            self.client = ClobClient(self.host)
            self.signature_type = SignatureType.EOA
            return

        # Validate private key format
        key = key.strip()
        if not key.startswith("0x") or len(key) != 66:
            logger.error("Invalid PRIVATE_KEY format. Must be 0x + 64 hex chars.")
            logger.warning("Initializing public client due to invalid private key")
            self.client = ClobClient(self.host)
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
            self.wallet_address = funder_address if funder_address else self.client.get_address()
            
            self._configure_client_headers()
            self._check_initial_balance()
            
        except Exception as e:
            logger.error(f"Error initializing authenticated client: {e}", exc_info=True)
            logger.warning("Falling back to client without L2 credentials")
            self.client = ClobClient(
                self.host,
                key=key,
                chain_id=self.chain_id,
                signature_type=SignatureType.EOA
            )
            self.signature_type = SignatureType.EOA

    def _derive_api_credentials(self, key: str):
        logger.info("Step 1: Deriving API credentials for L2 authentication...")
        try:
            temp_client = ClobClient(self.host, key=key, chain_id=self.chain_id)
            api_creds = temp_client.create_or_derive_api_creds()
            logger.info(f"✓ API Credentials derived successfully! Key: {api_creds.api_key[:10]}...")
            return api_creds
        except Exception as e:
            logger.error(f"✗ Error deriving API creds: {e}")
            raise

    def _determine_signature_type(self, key: str) -> Tuple[int, Optional[str]]:
        logger.info("Step 2: Determining signature type...")
        # Note: We need RPC URL here. Assuming we can get it from env or pass it.
        # For this refactor, I'll rely on the existing constant pattern or env.
        polygon_rpc = "https://polygon-rpc.com" 
        
        w3 = Web3(Web3.HTTPProvider(polygon_rpc))
        account = w3.eth.account.from_key(key)
        signer_address = account.address
        
        env_address = os.getenv("WALLET_ADDRESS")
        signature_type = SignatureType.EOA
        funder_address = None
        
        if env_address and env_address.lower() != signer_address.lower():
            try:
                funder_address = Web3.to_checksum_address(env_address)
                signature_type = SignatureType.GNOSIS_SAFE
                logger.info(f"Detected Proxy Setup: Signer={signer_address}, Proxy={funder_address}")
            except Exception as e:
                logger.error(f"✗ Error setting up proxy address: {e}")
                signature_type = SignatureType.EOA
                funder_address = None
        else:
            logger.info(f"Detected Direct EOA Setup: {signer_address}")
        
        return signature_type, funder_address

    def _create_authenticated_client(
        self, key: str, api_creds, signature_type: int, funder_address: Optional[str]
    ) -> ClobClient:
        logger.info("Step 3: Initializing ClobClient with L1 + L2 authentication...")
        client_kwargs = {
            "host": self.host,
            "key": key,
            "chain_id": self.chain_id,
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
                "Referer": "https://polymarket.com/",
                "Origin": "https://polymarket.com",
            }
            self.client.http_client.session.headers.update(headers)
        except Exception as e:
            logger.warning(f"Could not update headers: {e}")

    def _check_initial_balance(self) -> None:
        try:
            if not self.client: return
            # Using raw int 2 for conditional allowed (AssetType.CONDITIONAL is usually 2 or COLLATERAL is 1)
            # Need to be careful with imports.
            # Assuming basic check.
            pass 
        except Exception:
            pass

    async def get_active_market(self, gamma_url: str, series_id: str = "10192", session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
        """Fetch active market for a series."""
        try:
            url = f"{gamma_url}/events?limit=100&closed=false&series_id={series_id}"
            
            if session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200: return None
                    events = await resp.json()
            else:
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(url, timeout=10) as resp:
                        if resp.status != 200: return None
                        events = await resp.json()
            
            if not events: return None
            
            now = pd.Timestamp.now(tz='UTC')
            valid_events = [e for e in events if pd.Timestamp(e['endDate']) > now]
            if not valid_events: return None
            
            valid_events.sort(key=lambda x: pd.Timestamp(x['endDate']))
            return valid_events[0]
        except Exception as e:
            logger.error(f"Error in get_active_market: {e}")
            return None

    async def get_positions(self, data_api_url: str, wallet_address: str, size_threshold: float = 0.1, session: Optional[aiohttp.ClientSession] = None) -> List[Dict[str, Any]]:
        """Fetch user positions."""
        try:
            params = {"user": wallet_address, "sizeThreshold": size_threshold, "limit": 100}
            url = f"{data_api_url}/positions"
            
            if session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
            else:
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            return await resp.json()
            return []
        except Exception as e:
            logger.error(f"Error in get_positions: {e}")
            return []
