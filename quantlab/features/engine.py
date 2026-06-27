"""Feature engineering for QuantLab strategies.

The :class:`FeatureEngine` computes a comprehensive set of technical
and statistical features from price data.  All methods are designed to
be **ticker-agnostic** — they operate on every column of the input
price matrix simultaneously via vectorised pandas/numpy operations.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute all features needed by QuantLab strategies.

    Parameters
    ----------
    prices :
        DataFrame of **Close** prices.  Columns are ticker symbols,
        index is a ``DatetimeIndex``.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        if prices is None or prices.empty:
            raise ValueError("prices DataFrame must not be None or empty")
        self.prices: pd.DataFrame = prices.sort_index()
        self._returns: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------
    def compute_all(self) -> pd.DataFrame:
        """Compute and return **all** features as a single DataFrame.

        The returned DataFrame has a ``DatetimeIndex`` and MultiIndex
        columns organised by feature group:

        * ``return`` – daily percentage returns
        * ``vol_20`` – 20-day annualised volatility
        * ``drawdown`` – peak-to-trough drawdown
        * ``rsi_14`` – 14-day Relative Strength Index
        * ``trend_z20`` – z-score distance from 20-day moving average
        * ``mom_5`` / ``mom_10`` / ``mom_20`` / ``mom_60`` – momentum
        * ``vol_regime`` – volatility regime label (0=low, 1=med, 2=high)
        * ``rolling_corr`` – stacked rolling correlation panels

        Returns
        -------
        pd.DataFrame
            Combined feature matrix.
        """
        features: Dict[str, pd.DataFrame] = {}

        # 1. Daily returns
        features["return"] = self.daily_returns()

        # 2. Rolling volatility (20-day)
        features["vol_20"] = self.rolling_volatility(window=20)

        # 3. Drawdown
        features["drawdown"] = self.drawdown()

        # 4. RSI (14-day)
        features["rsi_14"] = self.rsi(window=14)

        # 5. Trend distance (20-day z-score)
        features["trend_z20"] = self.trend_distance(window=20)

        # 6. Momentum (multiple windows)
        mom = self.momentum(windows=[5, 10, 20, 60])
        for col in mom.columns:
            features[col] = mom[[col]]

        # 7. Volatility regime
        features["vol_regime"] = self.volatility_regime(n_regimes=3)

        # Combine all features side-by-side
        combined = pd.concat(features.values(), axis=1)

        # Flatten MultiIndex columns if present – keep as (feature, ticker)
        if combined.columns.nlevels == 1:
            # Ensure we have a sensible naming convention
            combined.columns = pd.MultiIndex.from_product(
                [[""], combined.columns],
            )

        logger.info(
            "compute_all: produced %d rows x %d columns",
            len(combined),
            combined.shape[1],
        )
        return combined

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------
    def daily_returns(self) -> pd.DataFrame:
        """Daily percentage returns.

        Returns
        -------
        pd.DataFrame
            Percentage change of *prices* (first row is NaN).
        """
        returns = self.prices.pct_change()
        returns.columns = pd.MultiIndex.from_product(
            [["return"], returns.columns],
            names=["feature", "ticker"],
        )
        return returns

    def rolling_volatility(self, window: int = 20) -> pd.DataFrame:
        """Annualised rolling volatility.

        Computed as the rolling standard deviation of daily returns
        multiplied by :math:`\\sqrt{252}`.

        Parameters
        ----------
        window :
            Rolling look-back window in trading days.

        Returns
        -------
        pd.DataFrame
            Annualised volatility for each ticker.
        """
        returns = self._get_returns()
        vol = returns.rolling(window=window, min_periods=max(2, window // 2)).std() * np.sqrt(252)
        vol.columns = pd.MultiIndex.from_product(
            [[f"vol_{window}"], vol.columns],
            names=["feature", "ticker"],
        )
        return vol

    def rolling_covariance(self, window: int = 60) -> pd.DataFrame:
        """Rolling covariance matrix.

        Parameters
        ----------
        window :
            Rolling look-back window in trading days.

        Returns
        -------
        pd.DataFrame
            Stacked / multi-index DataFrame with levels
            ``(date, ticker_i, ticker_j)`` and a single ``cov`` column.
        """
        returns = self._get_returns()
        tickers = returns.columns.tolist()

        cov_frames: List[pd.DataFrame] = []
        min_periods = max(10, window // 3)

        for i, start in enumerate(returns.index):
            if i < min_periods - 1:
                continue
            end_idx = i
            start_idx = max(0, end_idx - window + 1)
            sub = returns.iloc[start_idx : end_idx + 1]
            if len(sub) < min_periods:
                continue
            cov_mat = sub.cov()
            # Stack to long form
            stacked = cov_mat.stack()
            stacked.index.names = ["ticker_i", "ticker_j"]
            df = stacked.reset_index()
            df["date"] = start
            df = df.set_index(["date", "ticker_i", "ticker_j"])
            df.columns = ["cov"]
            cov_frames.append(df)

        if not cov_frames:
            idx = pd.MultiIndex.from_product(
                [[], [], []], names=["date", "ticker_i", "ticker_j"]
            )
            return pd.DataFrame(index=idx, columns=["cov"])

        return pd.concat(cov_frames)

    def rolling_correlation(self, window: int = 60) -> pd.DataFrame:
        """Rolling correlation matrix.

        Parameters
        ----------
        window :
            Rolling look-back window in trading days.

        Returns
        -------
        pd.DataFrame
            Stacked / multi-index DataFrame with levels
            ``(date, ticker_i, ticker_j)`` and a single ``corr`` column.
        """
        returns = self._get_returns()
        min_periods = max(10, window // 3)

        corr_frames: List[pd.DataFrame] = []
        for i, start in enumerate(returns.index):
            if i < min_periods - 1:
                continue
            end_idx = i
            start_idx = max(0, end_idx - window + 1)
            sub = returns.iloc[start_idx : end_idx + 1]
            if len(sub) < min_periods:
                continue
            corr_mat = sub.corr()
            stacked = corr_mat.stack()
            stacked.index.names = ["ticker_i", "ticker_j"]
            df = stacked.reset_index()
            df["date"] = start
            df = df.set_index(["date", "ticker_i", "ticker_j"])
            df.columns = ["corr"]
            corr_frames.append(df)

        if not corr_frames:
            idx = pd.MultiIndex.from_product(
                [[], [], []], names=["date", "ticker_i", "ticker_j"]
            )
            return pd.DataFrame(index=idx, columns=["corr"])

        return pd.concat(corr_frames)

    def rsi(self, window: int = 14) -> pd.DataFrame:
        """Relative Strength Index.

        .. math::

            \\text{RSI} = 100 - \\frac{100}{1 + RS}

        where :math:`RS = \\text{avg_gain} / \\text{avg_loss}` and the
        averages are simple rolling means over *window* days.

        Parameters
        ----------
        window :
            Look-back period for the RSI calculation.

        Returns
        -------
        pd.DataFrame
            RSI values bounded in ``[0, 100]``.
        """
        delta = self.prices.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi.columns = pd.MultiIndex.from_product(
            [[f"rsi_{window}"], rsi.columns],
            names=["feature", "ticker"],
        )
        return rsi

    def drawdown(self) -> pd.DataFrame:
        """Current drawdown from peak.

        Computed as the percentage decline from the running maximum
        price.

        Returns
        -------
        pd.DataFrame
            Negative values representing drawdown (0 = no drawdown).
        """
        rolling_max = self.prices.cummax()
        dd = (self.prices - rolling_max) / rolling_max
        dd.columns = pd.MultiIndex.from_product(
            [["drawdown"], dd.columns],
            names=["feature", "ticker"],
        )
        return dd

    def trend_distance(self, window: int = 20) -> pd.DataFrame:
        """Distance from *window*-day moving average in z-score terms.

        .. math::

            Z_t = \\frac{P_t - \\text{MA}_t}{\\sigma_t}

        where :math:`\\sigma_t` is the rolling standard deviation over
        the same *window*.

        Parameters
        ----------
        window :
            Moving-average look-back period.

        Returns
        -------
        pd.DataFrame
            Z-scores; positive = price above MA, negative = below.
        """
        ma = self.prices.rolling(window=window, min_periods=max(2, window // 2)).mean()
        std = self.prices.rolling(window=window, min_periods=max(2, window // 2)).std()

        # Guard against zero std (e.g. stale prices)
        std_safe = std.replace(0, np.nan)
        zscore = (self.prices - ma) / std_safe

        zscore.columns = pd.MultiIndex.from_product(
            [[f"trend_z{window}"], zscore.columns],
            names=["feature", "ticker"],
        )
        return zscore

    def momentum(self, windows: List[int] = None) -> pd.DataFrame:
        """Momentum features: returns over various look-back windows.

        Parameters
        ----------
        windows :
            List of day-counts over which to compute total returns.
            Default: ``[5, 10, 20, 60]``.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns ``(mom_N, ticker)`` where *N* is each
            window length.
        """
        if windows is None:
            windows = [5, 10, 20, 60]

        frames: List[pd.DataFrame] = []
        for w in windows:
            # Total return over w days: P_t / P_{t-w} - 1
            mom = self.prices.pct_change(periods=w)
            mom.columns = pd.MultiIndex.from_product(
                [[f"mom_{w}"], mom.columns],
                names=["feature", "ticker"],
            )
            frames.append(mom)

        return pd.concat(frames, axis=1)

    def volatility_regime(self, n_regimes: int = 3) -> pd.DataFrame:
        """Label each date with a volatility regime.

        Uses **tercile-based** classification on 20-day annualised
        volatility:

        * Regime 0 = low volatility  (bottom tercile)
        * Regime 1 = medium volatility (middle tercile)
        * Regime 2 = high volatility   (top tercile)

        Parameters
        ----------
        n_regimes :
            Number of equal-frequency bins to create.

        Returns
        -------
        pd.DataFrame
            Integer regime labels (0 … *n_regimes* - 1) per ticker.
        """
        returns = self._get_returns()
        vol_20 = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)

        # Use the *mean* volatility across all tickers as the market-wide
        # volatility measure for regime classification.
        mean_vol = vol_20.mean(axis=1)

        # Tercile-based bins using pandas qcut
        try:
            regime = pd.qcut(mean_vol, q=n_regimes, labels=False, duplicates="drop")
        except ValueError:
            # Not enough unique values for qcut – fallback to all zeros
            regime = pd.Series(0, index=mean_vol.index)

        # Broadcast the single regime series to all tickers
        tickers = self.prices.columns.tolist()
        regime_df = pd.DataFrame(
            np.tile(regime.values[:, None], len(tickers)),
            index=regime.index,
            columns=pd.MultiIndex.from_product(
                [["vol_regime"], tickers],
                names=["feature", "ticker"],
            ),
        )
        return regime_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_returns(self) -> pd.DataFrame:
        """Return cached daily returns, computing if necessary."""
        if self._returns is None:
            self._returns = self.prices.pct_change()
        return self._returns
