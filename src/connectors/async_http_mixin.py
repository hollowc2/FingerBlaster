"""Async HTTP fetcher mixin using aiohttp."""

import logging
import aiohttp
from typing import Dict, List, Optional, Any
from aiohttp_retry import RetryClient, ExponentialRetry

logger = logging.getLogger("FingerBlaster.AsyncHttpMixin")

DEFAULT_REQUEST_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 3


class AsyncHttpFetcherMixin:
    """Async HTTP request/response handling using aiohttp."""

    async def _create_async_session(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES
    ) -> aiohttp.ClientSession:
        """Create aiohttp session with retry strategy."""
        timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)

        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )

        # Wrap with retry client
        retry_options = ExponentialRetry(
            attempts=max_retries,
            statuses=[429, 500, 502, 503, 504]
        )
        return RetryClient(client_session=session, retry_options=retry_options)

    async def _get_json_async(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        raise_for_status: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Execute async GET request and parse JSON."""
        try:
            async with self.async_session.get(url, params=params) as response:
                if raise_for_status:
                    response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request to {url} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    async def _get_json_with_fallback_async(
        self,
        url: str,
        param_variants: List[Dict[str, Any]],
        timeout: int = DEFAULT_REQUEST_TIMEOUT
    ) -> Optional[Dict[str, Any]]:
        """Try multiple parameter formats until one succeeds (async)."""
        for i, params in enumerate(param_variants, 1):
            try:
                async with self.async_session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
            except aiohttp.ClientError:
                continue
        return None
