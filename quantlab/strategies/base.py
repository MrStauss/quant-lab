"""Base strategy class for QuantLab."""
from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all QuantLab strategies.

    All strategy implementations must inherit from this class and
    override the :meth:`generate_signal` method. The returned signal
    dictionary must follow the unified signal schema defined in the
    QuantLab specification.
    """

    name: str = "base"
    family: str = ""

    @abstractmethod
    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate a trading signal.

        Args:
            features: DataFrame of pre-computed features (index=Date).
            prices: DataFrame of OHLCV close prices (index=Date, columns=tickers).

        Returns:
            Unified signal dictionary with the following keys:
                - date: ISO date string.
                - strategy: Strategy identifier string.
                - family: Asset family name.
                - weights: Mapping of ticker symbol to target weight (sum ~1.0).
                - confidence: Float in [0.0, 1.0].
                - risk_state: One of "low", "medium", "high", "extreme".
                - reason: Human-readable explanation string.
        """
        ...

    def _latest_date(self, features: pd.DataFrame, prices: pd.DataFrame) -> str:
        """Return the most recent common date as an ISO string."""
        common_idx = features.index.intersection(prices.index)
        if len(common_idx) == 0:
            # Fallback: use the latest available date
            latest = max(features.index.max(), prices.index.max())
        else:
            latest = common_idx.max()
        return pd.Timestamp(latest).strftime("%Y-%m-%d")
