"""Yahoo Finance data ingestion for QuantLab.

Provides functions to download OHLCV market data for asset families
used by trading strategies. Handles single-ticker and multi-ticker
downloads, returning consistently-shaped MultiIndex DataFrames.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from quantlab.config import QuantLabConfig

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"


def download_ohlcv(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        interval: Data interval (``"1d"``, ``"1wk"``, ``"1mo"``).

    Returns:
        DataFrame with MultiIndex columns ``(ticker, field)`` and
        ``Date`` index.  Fields: ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume``.

    Raises:
        ValueError: If *tickers* is empty.
        RuntimeError: If the download fails entirely (no data returned).

    """
    if not tickers:
        raise ValueError("tickers list must not be empty")

    tickers = [t.upper().strip() for t in tickers]
    logger.info(
        "Downloading %d ticker(s) from %s to %s [%s]",
        len(tickers),
        start_date,
        end_date,
        interval,
    )

    try:
        raw = yf.download(
            tickers=tickers if len(tickers) > 1 else tickers[0],
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    # ------------------------------------------------------------------
    # Normalise to MultiIndex (ticker, field) regardless of input shape
    # ------------------------------------------------------------------
    if len(tickers) == 1:
        # Single ticker → yfinance returns a flat DataFrame
        raw = raw.copy()
        raw.columns = pd.MultiIndex.from_product(
            [tickers, raw.columns],
            names=["ticker", "field"],
        )
    else:
        # Multi-ticker → yfinance already returns MultiIndex but columns
        # are (field, ticker); swap levels to (ticker, field).
        if raw.columns.nlevels == 2:
            raw = raw.swaplevel(axis=1)
            raw.columns.names = ["ticker", "field"]

    # Ensure standard field names (capitalised)
    raw = raw.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj Close",
        },
        level="field",
    )

    logger.info(
        "Downloaded %d rows x %d columns",
        len(raw),
        raw.shape[1],
    )
    return raw


def get_all_family_tickers(config: "QuantLabConfig") -> List[str]:
    """Collect all unique tickers from all asset families + benchmark.

    Args:
        config: Runtime configuration containing *asset_families*.

    Returns:
        Sorted list of unique ticker symbols.
    """
    tickers: set = set()
    for family in config.asset_families.values():
        tickers.update(family.values())
    tickers.add(getattr(config, "benchmark_ticker", BENCHMARK_TICKER))
    return sorted(tickers)


def download_latest_data(config: "QuantLabConfig") -> pd.DataFrame:
    """Download latest data for all configured asset families.

    Convenience wrapper around :func:`download_ohlcv` that resolves
    the full ticker list from *config* and uses the configured date
    range.

    Args:
        config: Runtime configuration.

    Returns:
        OHLCV DataFrame with MultiIndex columns.
    """
    tickers = get_all_family_tickers(config)
    return download_ohlcv(
        tickers=tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )
