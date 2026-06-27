"""Benchmark comparison utilities for backtesting."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_benchmark_returns(
    prices: pd.DataFrame, ticker: str = "SPY"
) -> pd.Series:
    """
    Calculate benchmark returns from price data.

    Args:
        prices: DataFrame with Date index and ticker columns (Close prices),
                or MultiIndex columns (ticker, field) from OHLCV data.
        ticker: Benchmark ticker symbol. Defaults to "SPY".

    Returns:
        pd.Series of daily benchmark returns indexed by date.
    """
    if prices.empty:
        logger.warning("Empty prices DataFrame in calculate_benchmark_returns")
        return pd.Series(dtype=float)

    # Handle MultiIndex columns (OHLCV format)
    if isinstance(prices.columns, pd.MultiIndex):
        if "Close" in prices.columns.get_level_values(1):
            close_prices = prices.xs("Close", level=1, axis=1)
        else:
            close_prices = prices
    else:
        close_prices = prices

    # Get benchmark prices
    if ticker in close_prices.columns:
        bench_prices = close_prices[ticker]
    else:
        # Fallback: use first available column
        logger.warning(
            "Benchmark ticker %s not found in prices. Using first available column.",
            ticker,
        )
        bench_prices = close_prices.iloc[:, 0]

    # Calculate daily returns
    returns = bench_prices.pct_change().fillna(0.0)
    returns.name = f"{ticker}_return"
    return returns


def compare_to_benchmark(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> dict:
    """
    Generate comparison metrics between strategy and benchmark.

    Args:
        strategy_returns: Series of strategy daily returns.
        benchmark_returns: Series of benchmark daily returns.

    Returns:
        Dictionary with:
        - active_return: Annualized excess return over benchmark
        - tracking_error: Annualized standard deviation of return differences
        - information_ratio: Active return / tracking error
        - beta: Strategy beta relative to benchmark
        - alpha: Annualized alpha (Jensen's alpha)
        - correlation: Correlation between strategy and benchmark returns
    """
    # Align series by index
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        logger.warning("Insufficient data for benchmark comparison")
        return {
            "active_return": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "beta": np.nan,
            "alpha": np.nan,
            "correlation": np.nan,
        }

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    # Active return: annualized mean of daily return differences
    active_daily = strat - bench
    active_return = float(active_daily.mean() * 252)

    # Tracking error: annualized std of daily return differences
    tracking_error = float(active_daily.std() * np.sqrt(252))

    # Information ratio
    if tracking_error > 1e-12:
        information_ratio = active_return / tracking_error
    else:
        information_ratio = np.nan

    # Beta: covariance(strategy, benchmark) / variance(benchmark)
    bench_var = bench.var()
    if bench_var > 1e-12:
        beta = float(strat.cov(bench) / bench_var)
    else:
        beta = np.nan

    # Alpha: annualized strategy return - beta * annualized benchmark return
    ann_strat_return = float(strat.mean() * 252)
    ann_bench_return = float(bench.mean() * 252)
    alpha = ann_strat_return - beta * ann_bench_return

    # Correlation
    if strat.std() > 1e-12 and bench.std() > 1e-12:
        correlation = float(strat.corr(bench))
    else:
        correlation = np.nan

    return {
        "active_return": active_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "correlation": correlation,
    }
