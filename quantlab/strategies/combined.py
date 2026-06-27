"""Combined ensemble strategy.

Blends the HMM Regime Rotation and Dynamic Covariance Allocation
strategies with configurable weights (default 50/50).  The ensemble
averages sub-strategy confidence, merges their weight vectors, and
concatenates their reasoning.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantlab.strategies.base import BaseStrategy
from quantlab.strategies.regime import HMMRegimeStrategy
from quantlab.strategies.dynamic import DynamicCovarianceStrategy

logger = logging.getLogger(__name__)


class CombinedStrategy(BaseStrategy):
    """Ensemble combining regime rotation and dynamic covariance.

    Runs both sub-strategies, normalises their weight vectors to sum
    to 1.0, then blends:

        final_weight = w_regime * regime_weight + w_dynamic * dynamic_weight

    Parameters
    ----------
    family_name : str
        Name of the asset family (e.g. "Semiconductors").
    assets : dict
        Mapping with keys ``bull``, ``base``, ``bear``, ``safe_haven``,
        ``cash`` pointing to ticker symbols.
    weights : dict or None
        Blend weights for the two sub-strategies.  Expected keys:
        ``regime_rotation`` and ``dynamic_allocation``.  Defaults to
        50/50.
    """

    name = "combined_ensemble"

    def __init__(
        self,
        family_name: str,
        assets: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.family_name = family_name
        self.assets = assets
        self.family = family_name

        if weights is None:
            self._blend_weights = {
                "regime_rotation": 0.50,
                "dynamic_allocation": 0.50,
            }
        else:
            self._blend_weights = {
                "regime_rotation": weights.get("regime_rotation", 0.50),
                "dynamic_allocation": weights.get("dynamic_allocation", 0.50),
            }

        # Normalise blend weights to sum to 1.0
        total = sum(self._blend_weights.values())
        if total > 0:
            self._blend_weights = {
                k: v / total for k, v in self._blend_weights.items()
            }
        else:
            self._blend_weights = {"regime_rotation": 0.50, "dynamic_allocation": 0.50}

        # Instantiate sub-strategies
        self._regime_strategy = HMMRegimeStrategy(family_name, assets)
        self._dynamic_strategy = DynamicCovarianceStrategy(family_name, assets)

    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate blended ensemble signal.

        Parameters
        ----------
        features : pd.DataFrame
            Pre-computed feature matrix (index=Date).
        prices : pd.DataFrame
            Close prices with ticker columns (index=Date).

        Returns
        -------
        dict
            Unified signal dictionary with blended weights.
        """
        # Run sub-strategies
        regime_signal = self._regime_strategy.generate_signal(features, prices)
        dynamic_signal = self._dynamic_strategy.generate_signal(features, prices)

        # Blend weight vectors
        blended_weights = self._blend_weights_vectors(
            regime_signal["weights"],
            dynamic_signal["weights"],
            self._blend_weights["regime_rotation"],
            self._blend_weights["dynamic_allocation"],
        )

        # Average confidence
        avg_confidence = (
            self._blend_weights["regime_rotation"] * regime_signal["confidence"]
            + self._blend_weights["dynamic_allocation"] * dynamic_signal["confidence"]
        )

        # Combine reasons
        combined_reason = (
            f"[Regime] {regime_signal['reason']} | "
            f"[Dynamic] {dynamic_signal['reason']}"
        )

        # Risk state: take the more conservative of the two
        risk_state = self._merge_risk_states(
            regime_signal["risk_state"], dynamic_signal["risk_state"]
        )

        return {
            "date": regime_signal.get("date", self._latest_date(features, prices)),
            "strategy": self.name,
            "family": self.family_name,
            "weights": blended_weights,
            "confidence": round(float(avg_confidence), 4),
            "risk_state": risk_state,
            "reason": combined_reason,
        }

    @staticmethod
    def _blend_weights_vectors(
        w1: Dict[str, float],
        w2: Dict[str, float],
        alpha: float,
        beta: float,
    ) -> Dict[str, float]:
        """Blend two weight dictionaries with coefficients alpha and beta.

        Both weight dicts are first normalised to sum to 1.0 (ignoring
        negative values), then combined.  The result is re-normalised.

        Parameters
        ----------
        w1, w2 : dict
            Ticker → weight mappings.
        alpha, beta : float
            Blend coefficients (should sum to ~1.0).

        Returns
        -------
        dict
            Blended and normalised weights.
        """
        all_tickers = set(w1.keys()) | set(w2.keys())

        # Normalise each vector
        def _normalise(w: Dict[str, float]) -> Dict[str, float]:
            clipped = {k: max(0.0, v) for k, v in w.items()}
            total = sum(clipped.values())
            if total > 0:
                return {k: v / total for k, v in clipped.items()}
            return {k: 0.0 for k in clipped}

        nw1 = _normalise(w1)
        nw2 = _normalise(w2)

        blended: Dict[str, float] = {}
        for ticker in all_tickers:
            blended[ticker] = alpha * nw1.get(ticker, 0.0) + beta * nw2.get(ticker, 0.0)

        # Final normalisation
        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total, 4) for k, v in blended.items()}
        else:
            # Fallback to cash if everything collapsed
            cash = "BIL"
            blended = {ticker: 0.0 for ticker in all_tickers}
            blended[cash] = 1.0

        return blended

    @staticmethod
    def _merge_risk_states(rs1: str, rs2: str) -> str:
        """Return the more conservative of two risk states."""
        order = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        v1 = order.get(rs1, 1)
        v2 = order.get(rs2, 1)
        merged = max(v1, v2)
        for label, val in order.items():
            if val == merged:
                return label
        return "medium"

    def _latest_date(self, features: pd.DataFrame, prices: pd.DataFrame) -> str:
        """Return the most recent common date as an ISO string."""
        common_idx = features.index.intersection(prices.index)
        if len(common_idx) == 0:
            latest = max(features.index.max(), prices.index.max())
        else:
            latest = common_idx.max()
        return pd.Timestamp(latest).strftime("%Y-%m-%d")
