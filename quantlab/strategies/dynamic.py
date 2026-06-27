"""Dynamic Covariance Allocation strategy.

Allocates between the bull ETF and safe-haven ETF based on the
rolling correlation between them.  High correlation implies the
safe haven is providing less diversification benefit, so we reduce
bull exposure; low correlation implies diversification is working,
so we increase bull exposure.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quantlab.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class DynamicCovarianceStrategy(BaseStrategy):
    """Dynamic allocation driven by rolling bull/safe-haven correlation.

    Computes the rolling correlation between the bull ETF and the
    safe-haven ETF over a configurable window.  The correlation is
    normalised to [0, 1] and used to set weights:

        bull_weight  = 1 - normalized_corr
        safe_weight  = normalized_corr

    When correlation is high the safe haven is not diversifying well,
    so we shelter in it.  When correlation is low diversification works,
    so we lean into the bull ETF.

    Parameters
    ----------
    family_name : str
        Name of the asset family (e.g. "Semiconductors").
    assets : dict
        Mapping with keys ``bull``, ``base``, ``bear``, ``safe_haven``,
        ``cash`` pointing to ticker symbols.
    rolling_window : int, optional
        Look-back window in trading days for correlation (default 60).
    """

    name = "dynamic_covariance"

    def __init__(
        self,
        family_name: str,
        assets: Dict[str, str],
        rolling_window: int = 60,
    ) -> None:
        self.family_name = family_name
        self.assets = assets
        self.rolling_window = rolling_window
        self.family = family_name

    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate dynamic covariance allocation signal.

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
        bull_ticker = self.assets["bull"]
        safe_ticker = self.assets["safe_haven"]
        base_ticker = self.assets["base"]
        cash_ticker = self.assets.get("cash", "BIL")

        returns = prices.pct_change().dropna()

        if bull_ticker not in returns.columns or safe_ticker not in returns.columns:
            return self._fallback_signal(
                f"Missing tickers: {bull_ticker} or {safe_ticker}", prices, features
            )

        # Rolling correlation between bull and safe haven
        bull_safe_corr = (
            returns[bull_ticker]
            .rolling(self.rolling_window)
            .corr(returns[safe_ticker])
        )

        if bull_safe_corr.isna().all():
            return self._fallback_signal(
                "No correlation data available (insufficient history)", prices, features
            )

        # Normalise correlation from [-1, 1] → [0, 1]
        latest_corr = bull_safe_corr.dropna().iloc[-1]
        normalized_corr = (latest_corr + 1.0) / 2.0

        # Clip to safe bounds
        normalized_corr = float(np.clip(normalized_corr, 0.0, 1.0))

        bull_weight = 1.0 - normalized_corr
        safe_weight = normalized_corr

        # Build unified weights dict
        weights: Dict[str, float] = {
            bull_ticker: round(bull_weight, 4),
            safe_ticker: round(safe_weight, 4),
            base_ticker: 0.0,
            cash_ticker: 0.0,
        }

        # Confidence: higher when correlation is stable (low vol of correlation)
        corr_vol = bull_safe_corr.dropna().std()
        # Inverse: high corr volatility → low confidence
        confidence = 1.0 / (1.0 + 5.0 * corr_vol)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        risk_state = self._corr_to_risk_state(latest_corr)
        reason = (
            f"Rolling {self.rolling_window}d corr({bull_ticker},{safe_ticker})="
            f"{latest_corr:.3f} → bull={bull_weight:.2%}, safe={safe_weight:.2%}"
        )

        return {
            "date": self._latest_date(features, prices),
            "strategy": self.name,
            "family": self.family_name,
            "weights": weights,
            "confidence": round(confidence, 4),
            "risk_state": risk_state,
            "reason": reason,
        }

    @staticmethod
    def _corr_to_risk_state(corr: float) -> str:
        """Map correlation level to risk state label.

        High positive correlation means assets move together → less
        diversification → higher effective risk.
        """
        if corr > 0.7:
            return "high"
        elif corr > 0.3:
            return "medium"
        else:
            return "low"

    def _fallback_signal(
        self,
        reason: str,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return a conservative cash signal when the strategy cannot run."""
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
