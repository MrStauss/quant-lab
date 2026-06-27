"""QuantLab data layer.

Provides data ingestion from Yahoo Finance, persistent storage via
DuckDB/SQLite, and data-quality validation.
"""

from quantlab.data.ingestion import (
    download_latest_data,
    download_ohlcv,
    get_all_family_tickers,
)
from quantlab.data.storage import DataStore
from quantlab.data.validation import ValidationReport, validate_ohlcv

__all__ = [
    "download_ohlcv",
    "download_latest_data",
    "get_all_family_tickers",
    "DataStore",
    "validate_ohlcv",
    "ValidationReport",
]
