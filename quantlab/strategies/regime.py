"""HMM Regime Rotation strategy.

Implements Hidden Markov Model-based regime detection that maps
identified market states to appropriate ETF positions within an
asset family.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from quantlab.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class HMMRegimeStrategy(BaseStrategy):
    """Hidden Markov Model regime detection and rotation.

    Uses a Gaussian HMM to identify three market regimes from
    base-ETF features (returns, rolling volatility, drawdown).
    States are ordered by volatility and mapped to:
        - State 0 (low vol / Bull)  → bull ETF
        - State 1 (med vol / Neutral) → base ETF
        - State 2 (high vol / Bear)  → safe haven ETF

    Parameters
    ----------
    family_name : str
        Name of the asset family (e.g. "Semiconductors").
    assets : dict
        Mapping with keys ``bull``, ``base``, ``bear``, ``safe_haven``,
        ``cash`` pointing to ticker symbols.
    n_regimes : int, optional
        Number of HMM hidden states (default 3).
    """

    name = "hmm_regime_rotation"

    def __init__(
        self,
        family_name: str,
        assets: Dict[str, str],
        n_regimes: int = 3,
    ) -> None:
        self.family_name = family_name
        self.assets = assets
        self.n_regimes = n_regimes
        self.family = family_name

    def generate_signal(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate regime-based rotation signal for the latest date.

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
        base_ticker = self.assets["base"]
        bull_ticker = self.assets["bull"]
        safe_ticker = self.assets["safe_haven"]
        cash_ticker = self.assets.get("cash", "BIL")

        # Compute returns from prices
        returns = prices.pct_change().dropna()

        if base_ticker not in returns.columns:
            return self._fallback_signal(
                f"Base ticker {base_ticker} not in returns data", prices, features
            )

        # Build feature array for HMM: [returns, rolling_vol, drawdown]
        base_ret = returns[base_ticker].values.reshape(-1, 1)
        rolling_vol = (
            returns[base_ticker].rolling(20).std().values.reshape(-1, 1)
        )

        cumulative = (1 + returns[base_ticker]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).values.reshape(-1, 1)

        # Stack and skip initial NaN rows from rolling windows
        min_offset = 20
        features_raw = np.column_stack(
            [base_ret[min_offset:], rolling_vol[min_offset:], drawdown[min_offset:]]
        )

        if len(features_raw) < self.n_regimes * 2:
            return self._fallback_signal(
                "Insufficient data for HMM fitting", prices, features
            )

        # Standardise features
        feat_mean = features_raw.mean(axis=0)
        feat_std = features_raw.std(axis=0) + 1e-8
        features_scaled = (features_raw - feat_mean) / feat_std

        # Fit HMM and predict states
        states = self._fit_hmm(features_scaled)

        # Map states by average volatility (low→0, med→1, high→2)
        remapped_states = self._map_states_by_vol(states, features_raw)

        # Latest state determines allocation
        latest_state = int(remapped_states[-1])

        # Build weights based on latest state
        weights: Dict[str, float] = {bull_ticker: 0.0, base_ticker: 0.0, safe_ticker: 0.0, cash_ticker: 0.0}
        if latest_state == 0:
            # Bull regime → 100% bull ETF
            weights[bull_ticker] = 1.0
            risk_state = "low"
            reason = f"Bull regime detected (HMM state {latest_state}) — risk-on"
        elif latest_state == 1:
            # Neutral regime → 100% base ETF
            weights[base_ticker] = 1.0
            risk_state = "medium"
            reason = f"Neutral regime detected (HMM state {latest_state}) — base position"
        else:
            # Bear regime → 100% safe haven
            weights[safe_ticker] = 1.0
            risk_state = "high"
            reason = f"Bear regime detected (HMM state {latest_state}) — risk-off"

        # Confidence based on state persistence/stability
        confidence = self._compute_confidence(remapped_states, latest_state)

        signal_date = self._latest_date(features, prices)

        return {
            "date": signal_date,
            "strategy": self.name,
            "family": self.family_name,
            "weights": weights,
            "confidence": confidence,
            "risk_state": risk_state,
            "reason": reason,
        }

    def _fit_hmm(self, features_array: np.ndarray) -> np.ndarray:
        """Fit Gaussian HMM and return state sequence.

        Falls back to volatility tercile classification if hmmlearn
        is unavailable or the HMM fit fails to converge.

        Parameters
        ----------
        features_array : np.ndarray
            Standardised feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Integer state sequence of length n_samples.
        """
        n_samples = features_array.shape[0]

        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not installed; falling back to vol terciles")
            return self._vol_tercile_fallback(features_array)

        try:
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            model.fit(features_array)
            states = model.predict(features_array)
        except Exception as exc:
            logger.warning("HMM fit failed (%s); falling back to vol terciles", exc)
            states = self._vol_tercile_fallback(features_array)

        return states

    def _vol_tercile_fallback(self, features_array: np.ndarray) -> np.ndarray:
        """Classify regimes using volatility terciles.

        Uses the rolling-volatility column (index 1) of the raw feature
        array to partition observations into low / medium / high vol.

        Parameters
        ----------
        features_array : np.ndarray
            Feature matrix; column 1 must be rolling volatility.

        Returns
        -------
        np.ndarray
            Integer state labels (0=low, 1=med, 2=high).
        """
        vol_series = pd.Series(features_array[:, 1])
        terciles = vol_series.quantile([0.33, 0.67]).values

        states = np.zeros(len(vol_series), dtype=int)
        states[vol_series <= terciles[0]] = 0
        states[
            (vol_series > terciles[0]) & (vol_series <= terciles[1])
        ] = 1
        states[vol_series > terciles[1]] = 2
        return states

    def _map_states_by_vol(
        self, states: np.ndarray, features_raw: np.ndarray
    ) -> np.ndarray:
        """Remap HMM states so that lower state ID = lower volatility.

        The HMM assigns arbitrary integer labels; we reorder them by
        the average rolling volatility observed in each state.

        Parameters
        ----------
        states : np.ndarray
            Raw HMM state sequence.
        features_raw : np.ndarray
            Unscaled features; column 1 is rolling volatility.

        Returns
        -------
        np.ndarray
            Remapped state sequence.
        """
        vol_by_state = []
        for i in range(self.n_regimes):
            mask = states == i
            if mask.sum() > 0:
                avg_vol = features_raw[mask, 1].mean()
                vol_by_state.append((i, avg_vol))

        # Sort by avg vol ascending
        vol_by_state.sort(key=lambda x: x[1])
        state_mapping = {old: new for new, (old, _) in enumerate(vol_by_state)}
        return np.array([state_mapping[s] for s in states])

    def _compute_confidence(
        self, remapped_states: np.ndarray, latest_state: int, window: int = 20
    ) -> float:
        """Compute signal confidence from state persistence.

        Confidence is higher when the current regime has persisted for
        several consecutive days without flipping.

        Parameters
        ----------
        remapped_states : np.ndarray
            Full remapped state sequence.
        latest_state : int
            Current (latest) state ID.
        window : int
            Number of recent days to examine.

        Returns
        -------
        float
            Confidence value in [0.0, 1.0].
        """
        recent = remapped_states[-window:]
        state_ratio = (recent == latest_state).mean()
        # Scale: require at least 60% persistence to get above 0.5 confidence
        confidence = min(1.0, max(0.0, state_ratio))
        return round(float(confidence), 4)

    def _fallback_signal(
        self,
        reason: str,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return a conservative cash signal when the strategy cannot run."""
        cash_ticker = self.assets.get("cash", "BIL")
        weights = {cash_ticker: 1.0}
        # Add zero weights for other family tickers if available
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
