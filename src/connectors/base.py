from abc import ABC, abstractmethod
import pandas as pd

class DataConnector(ABC):
    @abstractmethod
    def fetch_data(self, symbol: str, start_time: int = None, end_time: int = None, limit: int = None) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol.
        """
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """
        Gets the latest price for a given symbol.
        """
        pass

