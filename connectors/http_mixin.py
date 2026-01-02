"""HTTP fetcher mixin providing common HTTP request/response handling utilities."""

import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Any, Union, Literal

logger = logging.getLogger("FingerBlaster.HttpMixin")

# Default constants - can be overridden by subclass
DEFAULT_REQUEST_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 3


class HttpFetchError(Exception):
    """Base exception for HTTP fetcher errors."""
    pass


class ApiTimeoutError(HttpFetchError):
    """Request exceeded timeout."""
    pass


class ApiResponseError(HttpFetchError):
    """Invalid response format."""
    pass


class HttpFetcherMixin:
    """Provides common HTTP request/response handling utilities for connectors.

    This mixin consolidates duplicated HTTP patterns found across the codebase:
    - Session creation with retry strategy
    - GET request + JSON parsing (13+ occurrences)
    - Error handling and logging
    - Multi-format response parsing with fallbacks

    Usage:
        class MyConnector(DataConnector, HttpFetcherMixin):
            def __init__(self):
                super().__init__()
                self.session = self._create_session()
    """

    def _create_session(self, max_retries: int = DEFAULT_MAX_RETRIES) -> requests.Session:
        """
        Create a requests session with connection pooling and retry strategy.

        Retry strategy retries on:
        - Network errors (connection failures, timeouts)
        - HTTP 429 (Too Many Requests)
        - HTTP 500+ (Server errors: 500, 502, 503, 504)

        Uses exponential backoff (backoff_factor=1):
        - Retry 1: 0s
        - Retry 2: 2s
        - Retry 3: 4s

        Args:
            max_retries: Maximum number of retries (default: DEFAULT_MAX_RETRIES)

        Returns:
            Configured requests.Session instance with retry adapter
        """
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=max_retries,
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

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        raise_for_status: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Execute GET request and parse JSON response.

        Consolidates the most common pattern found in 13+ locations:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

        Args:
            url: The URL to fetch
            params: Optional query parameters
            timeout: Request timeout in seconds (default: from NetworkConstants)
            raise_for_status: Whether to raise on HTTP errors (default: True)

        Returns:
            Parsed JSON response as dictionary, or None if request fails

        Raises:
            ApiTimeoutError: If request times out
            ApiResponseError: If JSON parsing fails
            requests.HTTPError: If raise_for_status=True and HTTP error occurs
        """
        try:
            response = self.session.get(url, params=params, timeout=timeout)

            if raise_for_status:
                response.raise_for_status()

            try:
                return response.json()
            except ValueError as e:
                logger.error(f"Failed to parse JSON response from {url}: {e}")
                raise ApiResponseError(f"Invalid JSON response: {e}")

        except requests.Timeout as e:
            logger.error(f"Request to {url} timed out after {timeout}s: {e}")
            raise ApiTimeoutError(f"Request timed out: {e}")
        except requests.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def _get_json_with_fallback(
        self,
        url: str,
        param_variants: List[Dict[str, Any]],
        timeout: int = DEFAULT_REQUEST_TIMEOUT
    ) -> Optional[Dict[str, Any]]:
        """
        Try multiple parameter formats until one succeeds.

        Consolidates the retry logic pattern found in get_chainlink_price_at()
        where multiple API parameter formats are attempted sequentially.

        Args:
            url: The URL to fetch
            param_variants: List of parameter dictionaries to try in order
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response from first successful request, or None if all fail

        Example:
            data = self._get_json_with_fallback(
                query_url,
                [
                    {'timestamp': ts1, 'format': 'v1'},
                    {'timestamp': ts2, 'format': 'v2'},
                    {'timestamp': ts3, 'format': 'v3'}
                ]
            )
        """
        last_error = None

        for i, params in enumerate(param_variants, 1):
            try:
                logger.debug(f"Attempting request with params variant {i}/{len(param_variants)}: {params}")
                response = self.session.get(url, params=params, timeout=timeout)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.debug(f"Successfully fetched data with variant {i}")
                        return data
                    except ValueError as e:
                        logger.warning(f"Variant {i} returned 200 but invalid JSON: {e}")
                        last_error = e
                        continue
                else:
                    logger.debug(f"Variant {i} returned status {response.status_code}")
                    last_error = f"HTTP {response.status_code}"

            except requests.RequestException as e:
                logger.debug(f"Variant {i} request failed: {e}")
                last_error = e
                continue

        logger.error(f"All {len(param_variants)} parameter variants failed. Last error: {last_error}")
        return None

    def _parse_json_response(
        self,
        data: Union[Dict, List, Any],
        expected_format: Literal['single', 'list', 'dict'] = 'single'
    ) -> Union[Dict[str, Any], List[Dict], Any]:
        """
        Parse JSON with format validation and fallbacks.

        Handles multiple response format patterns found in the codebase:
        - Direct dict: {'field': 'value'}
        - List response: [{'field': 'value'}]
        - Nested dict: {'data': {'field': 'value'}} or {'markets': [...]}

        Args:
            data: The parsed JSON data
            expected_format: Expected format ('single', 'list', 'dict')

        Returns:
            Processed data in the expected format

        Example:
            # Handle list responses that might be wrapped
            markets = self._parse_json_response(data, expected_format='list')
        """
        if expected_format == 'list':
            # If data is already a list, return it
            if isinstance(data, list):
                return data
            # If it's a dict, try to extract list from common keys
            if isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    return data['data']
                if 'markets' in data and isinstance(data['markets'], list):
                    return data['markets']
                if 'results' in data and isinstance(data['results'], list):
                    return data['results']
            # Fallback: wrap in list
            return [data]

        elif expected_format == 'dict':
            # If data is a list with items, return first item
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            # If it's already a dict, return it
            if isinstance(data, dict):
                return data
            # Fallback: return as-is
            return data

        else:  # expected_format == 'single'
            # Return as-is for single item responses
            return data
