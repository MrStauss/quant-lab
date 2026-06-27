"""Composer RSI strategy.

Refactored Composer Trade Symphony using RSI signals for
semiconductor ETFs.  A decision tree routes capital among
SOXL, SOXS, GLD, and BIL based on RSI14 readings for SPY,
SOXX, VIXM and SOXS.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantlab.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Tickers required by theComposer decision tree
_COMPOSER_TICKERS = ["SOXL", "SOXS", "SOXX", "SPY", "VIXM", "GLD", "BIL"]


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index for a price series.

    Parameters
    ----------
    series : pd.Series
        Price series (e.g. Close prices).
    window : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class ComposerRSIStrategy(BaseStrategy):
    """Composer-style RSI decision-tree strategy.

    Evaluates a fixed decision tree using RSI14 levels:

    * SPY RSI14 > 32 ?
      * SOXX RSI14 < 69 ?
        * VIXM RSI14 < 50  → 100% SOXL
        * SOXX RSI14 > 70  → 100% SOXS
        * VIXM RSI14 > 50  → 50% SOXS, 50% GLD
      * SOXX RSI14 >= 69 ?
        * SOXS RSI14 < 31  → 100% SOXS
        * else              → 100% BIL (cash)

    Parameters
    ----------
    assets : dict or None
        Optional asset mapping.  Defaults to the Semiconductors family.
    """

    name = "composer_rsi"
    family = "Semiconductors"

    def __init__(self, assets: Optional[Dict[str, str]] = None) -> None:
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

    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate Composer RSI signal for the latest date.

        Parameters
        ----------
        features : pd.DataFrame
            Pre-computed feature matrix (index=Date).  May contain
            RSI columns; if not they are computed on the fly.
        prices : pd.DataFrame
            Close prices with ticker columns (index=Date).

        Returns
        -------
        dict
            Unified signal dictionary.
        """
        # Compute RSI9 and RSI14 for all relevant tickers
        rsi_data = self._compute_all_rsi(prices)

        if rsi_data.empty:
            return self._fallback_signal("No RSI data could be computed", prices, features)

        # Extract latest RSI values
        latest = rsi_data.iloc[-1]

        # Apply decision tree
        weights = self._apply_rules(latest)

        # Compute confidence based on how "clear-cut" the RSI readings are
        confidence = self._compute_confidence(latest)

        # Determine risk state from the chosen allocation
        risk_state = self._weights_to_risk_state(weights)

        # Build reason string
        reason = self._build_reason(latest, weights)

        return {
            "date": self._latest_date(features, prices),
            "strategy": self.name,
            "family": self.family,
            "weights": weights,
            "confidence": round(confidence, 4),
            "risk_state": risk_state,
            "reason": reason,
        }

    def _compute_all_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI9 and RSI14 for required tickers.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices (index=Date, columns=tickers).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns like ``SPY_rsi14``, ``SOXX_rsi9``, …
        """
        rsi_df = pd.DataFrame(index=prices.index)
        for ticker in _COMPOSER_TICKERS:
            if ticker in prices.columns:
                rsi_df[f"{ticker}_rsi14"] = _compute_rsi(prices[ticker], window=14)
                rsi_df[f"{ticker}_rsi9"] = _compute_rsi(prices[ticker], window=9)
        return rsi_df.dropna()

    def _apply_rules(self, rsi_row: pd.Series) -> Dict[str, float]:
        """Apply the RSI decision tree to a single row of RSI values.

        Parameters
        ----------
        rsi_row : pd.Series
            Series containing ``{ticker}_rsi14`` (and optionally _rsi9)
            values.

        Returns
        -------
        dict
            Weight dictionary for SOXL, SOXS, GLD, BIL.
        """
        # Default: 100% cash
        w: Dict[str, float] = {"SOXL": 0.0, "SOXS": 0.0, "GLD": 0.0, "BIL": 1.0}

        spy_rsi14 = rsi_row.get("SPY_rsi14", np.nan)
        soxx_rsi14 = rsi_row.get("SOXX_rsi14", np.nan)
        vixm_rsi14 = rsi_row.get("VIXM_rsi14", np.nan)
        soxs_rsi14 = rsi_row.get("SOXS_rsi14", np.nan)

        # Guard against missing data
        if np.isnan(spy_rsi14) or np.isnan(soxx_rsi14):
            return w

        if spy_rsi14 > 32.0:
            if soxx_rsi14 < 69.0:
                if not np.isnan(vixm_rsi14) and vixm_rsi14 < 50.0:
                    w = {"SOXL": 1.0, "SOXS": 0.0, "GLD": 0.0, "BIL": 0.0}
                elif not np.isnan(soxx_rsi14) and soxx_rsi14 > 70.0:
                    # Dead-code in original logic (can't be <69 and >70)
                    # Preserved for exact backward compatibility
                    w = {"SOXL": 0.0, "SOXS": 1.0, "GLD": 0.0, "BIL": 0.0}
                elif not np.isnan(vixm_rsi14) and vixm_rsi14 > 50.0:
                    w = {"SOXL": 0.0, "SOXS": 0.5, "GLD": 0.5, "BIL": 0.0}
            else:
                # SOXX RSI14 >= 69
                if not np.isnan(soxs_rsi14) and soxs_rsi14 < 31.0:
                    w = {"SOXL": 0.0, "SOXS": 1.0, "GLD": 0.0, "BIL": 0.0}
                # else: keep default 100% BIL

        return w

    def _compute_confidence(self, rsi_row: pd.Series) -> float:
        """Compute confidence based on how clear-cut the RSI readings are.

        Confidence is higher when RSI values are far from the decision
        thresholds (32, 69, 50, 31), making the tree routing unambiguous.

        Parameters
        ----------
        rsi_row : pd.Series
            Latest RSI values.

        Returns
        -------
        float
            Confidence in [0.0, 1.0].
        """
        thresholds = {
            "SPY_rsi14": 32.0,
            "SOXX_rsi14": 69.0,
            "VIXM_rsi14": 50.0,
            "SOXS_rsi14": 31.0,
        }

        margins = []
        for col, thresh in thresholds.items():
            val = rsi_row.get(col, np.nan)
            if not np.isnan(val):
                # Normalised distance from threshold (0 = at threshold, 1 = far)
                margin = abs(val - thresh) / 50.0  # scale roughly
                margins.append(min(1.0, margin))

        if not margins:
            return 0.5

        # Average margin clipped to [0, 1]
        confidence = float(np.clip(np.mean(margins), 0.0, 1.0))
        return confidence

    @staticmethod
    def _weights_to_risk_state(weights: Dict[str, float]) -> str:
        """Map output weights to a risk-state label."""
        if weights.get("SOXL", 0.0) > 0.5:
            return "medium"  # leveraged long = medium risk
        elif weights.get("SOXS", 0.0) > 0.5:
            return "high"  # inverse leveraged = high risk
        elif weights.get("GLD", 0.0) > 0.0:
            return "low"  # gold / mixed defensive
        else:
            return "low"  # cash

    def _build_reason(
        self, rsi_row: pd.Series, weights: Dict[str, float]
    ) -> str:
        """Build a human-readable reason string."""
        parts = []
        for t in ("SPY", "SOXX", "VIXM", "SOXS"):
            col = f"{t}_rsi14"
            val = rsi_row.get(col)
            if val is not None and not pd.isna(val):
                parts.append(f"{t}_RSI14={val:.1f}")

        primary = max(weights, key=weights.get)
        return (
            f"Composer RSI: {' | '.join(parts)} → "
            f"primary allocation {primary} ({weights.get(primary, 0):.0%})"
        )

    def _fallback_signal(
        self,
        reason: str,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return a conservative cash signal."""
        return {
            "date": self._latest_date(features, prices),
            "strategy": self.name,
            "family": self.family,
            "weights": {"SOXL": 0.0, "SOXS": 0.0, "GLD": 0.0, "BIL": 1.0},
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
