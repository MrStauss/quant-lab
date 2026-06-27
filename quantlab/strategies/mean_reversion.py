"""Mean-reversion strategy for semiconductor ETFs.

Uses the z-score of SOXX price relative to its 20-day moving average
to determine SOXL vs SOXS positioning.  Extreme negative z-scores
suggest mean-reversion upward → overweight SOXL; extreme positive
z-scores suggest mean-reversion downward → overweight SOXS.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantlab.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion model for the SOXX/SOXL/SOXS trio.

    Computes the z-score of the base ETF (SOXX) price against its
    N-day moving average.  When the z-score exceeds a threshold in
    either direction, the strategy takes a contrarian position:

        zscore < -threshold → long SOXL (reversion up)
        zscore > +threshold → long SOXS (reversion down)

    Position size scales with the magnitude of the z-score, and a
    blend of GLD/BIL provides ballast during high-volatility regimes.

    Parameters
    ----------
    family_name : str, optional
        Asset family name (default "Semiconductors").
    assets : dict or None
        Ticker mapping.  Defaults to the Semiconductors family.
    zscore_threshold : float, optional
        Z-score threshold for triggering mean-reversion signals
        (default 1.5).
    zscore_window : int, optional
        Look-back window for the moving average (default 20).
    """

    name = "mean_reversion"
    family = "Semiconductors"

    def __init__(
        self,
        family_name: str = "Semiconductors",
        assets: Optional[Dict[str, str]] = None,
        zscore_threshold: float = 1.5,
        zscore_window: int = 20,
    ) -> None:
        self.family_name = family_name
        if assets is None:
            self.assets = {
                "bull": "SOXL",
                "base": "SOXX",
                "bear": "SOXS",
                "safe_haven": "GLD",
                "cash": "BIL",
            }
        else:
            self.assets = assets
        self.zscore_threshold = zscore_threshold
        self.zscore_window = zscore_window
        self.family = family_name

    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate mean-reversion signal for the latest date.

        Parameters
        ----------
        features : pd.DataFrame
            Pre-computed feature matrix (index=Date).
        prices : pd.DataFrame
            Close prices with ticker columns (index=Date).

        Returns
        -------
        dict
            Unified signal dictionary.
        """
        base_ticker = self.assets["base"]    # SOXX
        bull_ticker = self.assets["bull"]    # SOXL
        bear_ticker = self.assets["bear"]    # SOXS
        safe_ticker = self.assets["safe_haven"]  # GLD
        cash_ticker = self.assets.get("cash", "BIL")

        if base_ticker not in prices.columns:
            return self._fallback_signal(
                f"Base ticker {base_ticker} missing from prices", prices, features
            )

        # Compute z-score: (price - MA) / std
        price = prices[base_ticker]
        ma = price.rolling(window=self.zscore_window, min_periods=self.zscore_window).mean()
        std = price.rolling(window=self.zscore_window, min_periods=self.zscore_window).std()
        zscore = ((price - ma) / (std + 1e-12)).dropna()

        if zscore.empty:
            return self._fallback_signal(
                "Insufficient data to compute z-score", prices, features
            )

        latest_z = float(zscore.iloc[-1])

        # Volatility regime for GLD/BIL blending
        returns = prices.pct_change().dropna()
        base_vol = (
            returns[base_ticker]
            .rolling(self.zscore_window)
            .std()
            .dropna()
        )
        latest_vol = float(base_vol.iloc[-1]) if not base_vol.empty else 0.0

        # Build weights
        weights: Dict[str, float] = {
            bull_ticker: 0.0,
            base_ticker: 0.0,
            bear_ticker: 0.0,
            safe_ticker: 0.0,
            cash_ticker: 0.0,
        }

        abs_z = abs(latest_z)

        if abs_z < self.zscore_threshold:
            # No strong mean-reversion signal → hold cash
            weights[cash_ticker] = 1.0
            risk_state = "low"
            reason = (
                f"SOXX z-score={latest_z:.2f} within threshold "
                f"(+/-{self.zscore_threshold}) — no mean-reversion signal"
            )
            confidence = 0.3
        elif latest_z < -self.zscore_threshold:
            # Oversold → long SOXL (mean reversion up)
            # Scale position: larger |zscore| → larger position (capped)
            scale = min(1.0, abs_z / (2.0 * self.zscore_threshold))
            weights[bull_ticker] = scale

            # Blend remainder into GLD/BIL based on vol regime
            vol_percentile = self._vol_percentile(base_vol)
            if vol_percentile > 0.7:
                # High vol → more cash
                weights[cash_ticker] = 1.0 - scale
            else:
                weights[safe_ticker] = 1.0 - scale

            risk_state = "medium"
            reason = (
                f"SOXX oversold (z={latest_z:.2f}) — mean-reversion long "
                f"SOXL ({scale:.1%})"
            )
            confidence = min(1.0, abs_z / (2.0 * self.zscore_threshold))
        else:
            # Overbought → long SOXS (mean reversion down)
            scale = min(1.0, abs_z / (2.0 * self.zscore_threshold))
            weights[bear_ticker] = scale

            vol_percentile = self._vol_percentile(base_vol)
            if vol_percentile > 0.7:
                weights[cash_ticker] = 1.0 - scale
            else:
                weights[safe_ticker] = 1.0 - scale

            risk_state = "high"
            reason = (
                f"SOXX overbought (z={latest_z:.2f}) — mean-reversion short "
                f"via SOXS ({scale:.1%})"
            )
            confidence = min(1.0, abs_z / (2.0 * self.zscore_threshold))

        # Round weights for cleanliness
        weights = {k: round(v, 4) for k, v in weights.items()}

        return {
            "date": self._latest_date(features, prices),
            "strategy": self.name,
            "family": self.family_name,
            "weights": weights,
            "confidence": round(float(confidence), 4),
            "risk_state": risk_state,
            "reason": reason,
        }

    @staticmethod
    def _vol_percentile(vol_series: pd.Series) -> float:
        """Return the latest volatility as a percentile of recent history.

        Parameters
        ----------
        vol_series : pd.Series
            Rolling volatility series.

        Returns
        -------
        float
            Percentile in [0.0, 1.0].
        """
        if vol_series.empty or vol_series.isna().all():
            return 0.5
        latest = vol_series.iloc[-1]
        hist = vol_series.dropna()
        if len(hist) < 2:
            return 0.5
        return float((hist < latest).mean())

    def _fallback_signal(
        self,
        reason: str,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return a conservative cash signal."""
        cash_ticker = self.assets.get("cash", "BIL")
        weights: Dict[str, float] = {cash_ticker: 1.0}
        for key in ("bull", "base", "bear", "safe_haven"):
            ticker = self.assets.get(key)
            if ticker and ticker not in weights:
                weights[ticker] = 0.0

        return {
            "date": self._latest_date(features, prices),
            "strategy": self.name,
            "family": self.family_name,
            "weights": weights,
            "confidence": 0.0,
            "risk_state": "extreme",
            "reason": f"Fallback: {reason}",
        }

    def _latest_date(self, features: pd.DataFrame, prices: pd.DataFrame) -> str:
        """Return the most recent common date as an ISO string."""
        common_idx = features.index.intersection(prices.index)
        if len(common_idx) == 0:
            latest = max(features.index.max(), prices.index.max())
        else:
            latest = common_idx.max()
        return pd.Timestamp(latest).strftime("%Y-%m-%d")
