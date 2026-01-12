"""Mock Web3 provider for testing transaction signing without blockchain interaction."""

from unittest.mock import MagicMock
from typing import Dict, Any


class MockWeb3Provider:
    """
    Mock Web3 provider that simulates transaction signing.

    Usage:
        w3 = MockWeb3Provider()
        signed_tx = w3.eth.account.sign_transaction(tx_dict, private_key)
    """

    def __init__(self):
        """Initialize mock Web3 provider."""
        self.eth = MagicMock()
        self.eth.account = MagicMock()

        # Mock signing to return deterministic result
        self.eth.account.sign_transaction = MagicMock(
            return_value={
                'rawTransaction': b'0xmockedrawTransaction1234567890abcdef',
                'hash': b'0xmockedhash1234567890abcdef',
                'r': 12345,
                's': 67890,
                'v': 27,
            }
        )

        # Mock address conversion
        self.to_checksum_address = MagicMock(side_effect=lambda addr: addr.upper())

    def to_checksum_address(self, address: str) -> str:
        """
        Mock checksum address conversion.

        Args:
            address: Ethereum address

        Returns:
            Uppercase version (mock checksum)
        """
        return address.upper()


def create_mock_web3() -> MockWeb3Provider:
    """
    Factory function to create a mock Web3 provider.

    Returns:
        Mock Web3 provider instance
    """
    return MockWeb3Provider()
