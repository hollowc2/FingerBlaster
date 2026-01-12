import pytest
import json
from src.connectors.polymarket import PolymarketConnector

class TestPolymarketConnectorParsing:
    """Test parsing logic in PolymarketConnector."""

    def test_safe_parse_json_list_valid(self):
        """Test parsing valid JSON list strings."""
        connector = PolymarketConnector()
        # We don't need full init for this util method test, 
        # but if __init__ is heavy we might need to mock it.
        # Based on code review, __init__ sets simple attributes.
        
        valid_json = '["tag1", "tag2"]'
        result = connector._safe_parse_json_list(valid_json)
        assert result == ["tag1", "tag2"]

    def test_safe_parse_json_list_invalid(self):
        """Test parsing invalid JSON returns empty list (and logs warning)."""
        connector = PolymarketConnector()
        invalid_json = '["unclosed string'
        
        # Implementation captures exceptions and returns empty list
        assert connector._safe_parse_json_list(invalid_json) == []

    def test_safe_parse_json_list_empty(self):
        """Test parsing empty string."""
        connector = PolymarketConnector()
        assert connector._safe_parse_json_list("") == []
