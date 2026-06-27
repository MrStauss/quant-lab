"""Performance analytics for QuantLab.

All functions accept pandas Series of daily returns and produce scalar
metrics.  The ``calculate_all_metrics`` wrapper aggregates every metric
into a human-readable dictionary suitable for reports.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calculate_cagr(returns: pd.Series) -> float:
    """Calculate the Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series (decimal form, e.g. 0.01 for 1%).

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.1523 for 15.23%).
    """
    if returns.empty or returns.isna().all():
        return 0.0
    # Drop NaNs
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    total_return = (1 + clean).prod()
    n_years = len(clean) / 252.0
    if n_years <= 0:
        return 0.0
    return float(total_return ** (1.0 / n_years) - 1)


def calculate_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calculate the annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    risk_free : float, optional
        Annual risk-free rate as a decimal (default 0.0).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free / 252.0
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def calculate_sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calculate the annualized Sortino ratio.

    Uses downside deviation (standard deviation of negative returns only).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    risk_free : float, optional
        Annual risk-free rate as a decimal (default 0.0).

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free / 252.0
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(252))


def calculate_calmar(returns: pd.Series) -> float:
    """Calculate the Calmar ratio (CAGR / max drawdown).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.

    Returns
    -------
    float
        Calmar ratio.  Returns 0.0 if there is no drawdown.
    """
    cagr = calculate_cagr(returns)
    mdd = calculate_max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    return float(cagr / abs(mdd))


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum peak-to-trough drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. -0.15 for -15%).
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate the percentage of positive return days.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.

    Returns
    -------
    float
        Win rate as a decimal (e.g. 0.55 for 55%).
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return float((clean > 0).sum() / len(clean))


def calculate_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Calculate portfolio beta relative to a benchmark.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns.
    benchmark : pd.Series
        Benchmark daily returns.

    Returns
    -------
    float
        Beta coefficient.
    """
    if returns.empty or benchmark.empty:
        return 0.0
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    bench_var = bench.var()
    if bench_var == 0 or np.isnan(bench_var):
        return 0.0
    return float(port.cov(bench) / bench_var)


def calculate_alpha(
    returns: pd.Series, benchmark: pd.Series, risk_free: float = 0.0
) -> float:
    """Calculate annualized Jensen's alpha.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns.
    benchmark : pd.Series
        Benchmark daily returns.
    risk_free : float, optional
        Annual risk-free rate as a decimal (default 0.0).

    Returns
    -------
    float
        Annualized alpha as a decimal.
    """
    if returns.empty or benchmark.empty:
        return 0.0
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    beta = calculate_beta(port, bench)
    port_ann = port.mean() * 252
    bench_ann = bench.mean() * 252
    return float(port_ann - risk_free - beta * (bench_ann - risk_free))


def calculate_var(returns: pd.Series, level: float = 0.05) -> float:
    """Calculate historical Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    level : float, optional
        Confidence level (default 0.05 for 5% VaR).

    Returns
    -------
    float
        VaR as a negative decimal (e.g. -0.02 for -2%).
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(np.percentile(clean, level * 100))


def calculate_cvar(returns: pd.Series, level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    level : float, optional
        Confidence level (default 0.05 for 5% CVaR).

    Returns
    -------
    float
        CVaR as a negative decimal.
    """
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    var = calculate_var(clean, level)
    return float(clean[clean <= var].mean())


def calculate_turnover(weights_df: pd.DataFrame) -> float:
    """Calculate average portfolio turnover.

    Turnover is half the sum of absolute weight changes across all
    rebalancing periods, averaged per period.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame with dates as index and tickers as columns.  Each row
        must sum to approximately 1.0.

    Returns
    -------
    float
        Average turnover per rebalancing period.
    """
    if weights_df.empty or len(weights_df) < 2:
        return 0.0
    clean = weights_df.dropna()
    if len(clean) < 2:
        return 0.0
    diffs = clean.diff().abs().sum(axis=1).iloc[1:]
    if diffs.empty:
        return 0.0
    return float(diffs.mean() / 2.0)


# ---------------------------------------------------------------------------
# Comprehensive wrapper
# ---------------------------------------------------------------------------

def calculate_all_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    weights_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate all performance metrics at once.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (decimal form).
    benchmark : pd.Series, optional
        Daily benchmark returns for beta/alpha calculations.
    weights_df : pd.DataFrame, optional
        Historical weights for turnover calculation.

    Returns
    -------
    dict
        Dictionary mapping metric names to formatted string values.
    """
    clean = returns.dropna()
    total_days = int(len(clean))

    cagr = calculate_cagr(clean)
    sharpe = calculate_sharpe(clean)
    sortino = calculate_sortino(clean)
    calmar = calculate_calmar(clean)
    mdd = calculate_max_drawdown(clean)
    win_rate = calculate_win_rate(clean)
    var5 = calculate_var(clean, level=0.05)
    cvar5 = calculate_cvar(clean, level=0.05)

    turnover = calculate_turnover(weights_df) if weights_df is not None else 0.0

    result: Dict[str, Any] = {
        "Total Days": f"{total_days}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Calmar Ratio": f"{calmar:.3f}",
        "Max Drawdown": f"{mdd:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "VaR (5%)": f"{var5:.2%}",
        "CVaR (5%)": f"{cvar5:.2%}",
        "Turnover": f"{turnover:.3f}",
    }

    if benchmark is not None:
        beta = calculate_beta(clean, benchmark)
        alpha = calculate_alpha(clean, benchmark)
        result["Beta"] = f"{beta:.3f}"
        result["Alpha"] = f"{alpha:.3f}"

    return result
