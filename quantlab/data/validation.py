"""Data quality validation for QuantLab OHLCV data.

Provides comprehensive sanity checks to detect gaps, stale prices,
extreme returns, and anomalous volume patterns before the data is
fed into the feature engine and strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Report produced by :func:`validate_ohlcv`.

    Attributes
    ----------
    is_valid :
        ``True`` when no critical issues were detected.
    missing_dates :
        ISO date strings of unexpectedly missing trading dates.
    stale_tickers :
        Ticker symbols whose prices have been unchanged for too long.
    bad_returns :
        Ticker symbols that exhibited extreme single-day moves.
    zero_volume_days :
        Mapping from ticker symbol to list of dates with zero volume.
    messages :
        Human-readable descriptions of every issue found.
    """

    is_valid: bool
    missing_dates: List[str] = field(default_factory=list)
    stale_tickers: List[str] = field(default_factory=list)
    bad_returns: List[str] = field(default_factory=list)
    zero_volume_days: Dict[str, List[str]] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


def validate_ohlcv(
    data: pd.DataFrame,
    max_gap: int = 5,
    stale_window: int = 3,
    return_threshold: float = 0.50,
) -> ValidationReport:
    """Validate OHLCV data quality.

    Checks performed:

    1. **Missing dates** – detect unexpectedly large gaps in the
       trading calendar (weekends / holidays are expected and ignored).
    2. **Stale prices** – flag tickers whose *Close* price has not
       changed for *stale_window* or more consecutive days.
    3. **Bad returns** – flag tickers with a single-day absolute return
       exceeding *return_threshold*.
    4. **Zero volume days** – record any dates where volume is zero or
       missing.

    Parameters
    ----------
    data :
        OHLCV DataFrame with MultiIndex columns ``(ticker, field)``.
    max_gap :
        Maximum allowed number of consecutive missing trading days
        before a gap is reported.
    stale_window :
        Number of consecutive unchanged *Close* prices that triggers a
        stale-price flag.
    return_threshold :
        Absolute-return threshold (e.g. ``0.50`` = 50 %) above which a
        daily move is considered extreme.

    Returns
    -------
    ValidationReport
        Aggregated report containing all detected issues.
    """
    messages: List[str] = []

    if data is None or data.empty:
        messages.append("Data is empty or None")
        return ValidationReport(is_valid=False, messages=messages)

    # ------------------------------------------------------------------
    # 1. Missing dates
    # ------------------------------------------------------------------
    missing_dates = detect_missing_dates(data, max_gap=max_gap)
    if missing_dates:
        messages.append(
            f"Missing dates detected ({len(missing_dates)} gaps > {max_gap} days): "
            f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
        )

    # ------------------------------------------------------------------
    # 2. Stale prices
    # ------------------------------------------------------------------
    stale_tickers = detect_stale_prices(data, window=stale_window)
    if stale_tickers:
        messages.append(f"Stale prices detected for: {stale_tickers}")

    # ------------------------------------------------------------------
    # 3. Bad returns
    # ------------------------------------------------------------------
    returns = _extract_close_prices(data).pct_change().dropna(how="all")
    bad_returns = detect_bad_returns(returns, threshold=return_threshold)
    if bad_returns:
        messages.append(f"Extreme returns detected for: {bad_returns}")

    # ------------------------------------------------------------------
    # 4. Zero volume days
    # ------------------------------------------------------------------
    zero_volume_days: Dict[str, List[str]] = {}
    if ("Volume", "") in data.columns or any(
        field == "Volume" for _, field in data.columns
    ):
        volume = _extract_field(data, "Volume")
        for ticker in volume.columns:
            zero_mask = (volume[ticker] == 0) | (volume[ticker].isna())
            zero_dates = volume.index[zero_mask]
            if len(zero_dates) > 0:
                zero_volume_days[ticker] = zero_dates.strftime("%Y-%m-%d").tolist()
                messages.append(
                    f"Zero/missing volume on {len(zero_dates)} day(s) for {ticker}"
                )

    is_valid = not any(
        [missing_dates, stale_tickers, bad_returns, zero_volume_days]
    )

    return ValidationReport(
        is_valid=is_valid,
        missing_dates=missing_dates,
        stale_tickers=stale_tickers,
        bad_returns=bad_returns,
        zero_volume_days=zero_volume_days,
        messages=messages,
    )


def detect_missing_dates(data: pd.DataFrame, max_gap: int = 5) -> List[str]:
    """Detect unexpectedly missing trading dates.

    Weekends (Saturday / Sunday) and standard gaps up to *max_gap*
    trading days are tolerated.  Only gaps **exceeding** *max_gap* are
    reported.

    Parameters
    ----------
    data :
        OHLCV DataFrame with ``Date`` index.
    max_gap :
        Maximum tolerated gap length in **calendar** days.

    Returns
    -------
    List[str]
        ISO date strings for the start of each excessive gap.
    """
    if data.empty:
        return []

    dates = pd.to_datetime(data.index).sort_values()
    if len(dates) < 2:
        return []

    gaps = dates.to_series().diff().dt.days.iloc[1:]
    excessive = gaps[gaps > max_gap]

    # Ignore weekend-only gaps (gap of 1-2 days across weekends is normal)
    missing_starts: List[str] = []
    for gap_start_date, gap_days in excessive.items():
        # Skip if the gap is just a weekend (Sat-Sun = 2 calendar days)
        if gap_days <= 2:
            continue
        missing_starts.append(gap_start_date.strftime("%Y-%m-%d"))

    return missing_starts


def detect_stale_prices(data: pd.DataFrame, window: int = 3) -> List[str]:
    """Detect tickers with stale (unchanged) close prices.

    A ticker is flagged when its *Close* price does not change for
    *window* or more **consecutive** trading days.

    Parameters
    ----------
    data :
        OHLCV DataFrame with MultiIndex columns ``(ticker, field)``.
    window :
        Number of consecutive unchanged prices required to flag.

    Returns
    -------
    List[str]
        Ticker symbols with stale prices.
    """
    close = _extract_close_prices(data)
    if close.empty:
        return []

    stale_tickers: List[str] = []
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < window + 1:
            continue

        # Count consecutive unchanged prices using diff
        unchanged = prices.diff().abs() == 0
        # Rolling sum of consecutive True values
        max_consecutive = 0
        current = 0
        for val in unchanged:
            if val:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        if max_consecutive >= window - 1:  # diff() loses 1 observation
            stale_tickers.append(ticker)

    return stale_tickers


def detect_bad_returns(
    returns: pd.DataFrame, threshold: float = 0.50
) -> List[str]:
    """Detect tickers with extreme single-day returns.

    Parameters
    ----------
    returns :
        Daily returns DataFrame (tickers as columns, Date as index).
    threshold :
        Absolute-return threshold (e.g. ``0.50`` = 50 %).

    Returns
    -------
    List[str]
        Ticker symbols with at least one extreme daily return.
    """
    if returns is None or returns.empty:
        return []

    extreme = returns.abs() > threshold
    flagged = extreme.any(axis=0)
    return flagged[flagged].index.tolist()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Extract Close prices from a MultiIndex OHLCV DataFrame.

    Supports both ``(ticker, field)`` and ``(field, ticker)`` column
    orderings, as well as flat single-ticker DataFrames.
    """
    if data.columns.nlevels == 2:
        # Try (ticker, field) first
        if "Close" in data.columns.get_level_values(1):
            return data.xs("Close", level=1, axis=1)
        # Try (field, ticker)
        if "Close" in data.columns.get_level_values(0):
            return data.xs("Close", level=0, axis=1)
    elif data.columns.nlevels == 1:
        # Flat DataFrame – try common close column names
        for name in ("Close", "close", "Adj Close", "adj close"):
            if name in data.columns:
                return data[[name]].rename(columns={name: "Close"})
    return pd.DataFrame()


def _extract_field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract a specific field (e.g. Volume) from a MultiIndex DataFrame."""
    if data.columns.nlevels == 2:
        if field in data.columns.get_level_values(1):
            return data.xs(field, level=1, axis=1)
        if field in data.columns.get_level_values(0):
            return data.xs(field, level=0, axis=1)
    elif data.columns.nlevels == 1:
        if field in data.columns:
            return data[[field]]
    return pd.DataFrame()
