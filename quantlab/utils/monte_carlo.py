"""Monte Carlo simulation for forward-looking performance analysis.

The ``MonteCarloSimulator`` generates synthetic return paths via
bootstrap (sampling with replacement) or regime-aware sampling, then
derives forward-return distribution statistics.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Monte Carlo simulation engine for portfolio forward analysis.

    Parameters
    ----------
    n_simulations : int
        Number of independent simulation paths to generate.
    n_periods : int
        Number of trading days in each simulated path.

    Attributes
    ----------
    n_simulations : int
        Number of simulation paths.
    n_periods : int
        Number of periods per path.
    rng : np.random.Generator
        NumPy random generator instance.
    """

    def __init__(self, n_simulations: int = 2500, n_periods: int = 504):
        self.n_simulations = n_simulations
        self.n_periods = n_periods
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Simulation methods
    # ------------------------------------------------------------------

    def bootstrap_simulation(
        self, historical_returns: pd.Series
    ) -> np.ndarray:
        """Bootstrap simulation: resample historical returns with replacement.

        Each simulation path is constructed by randomly sampling (with
        replacement) from ``historical_returns`` for ``n_periods`` days,
        then computing the cumulative product.

        Parameters
        ----------
        historical_returns : pd.Series
            Daily historical returns (decimal form).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_simulations, n_periods)`` containing
            cumulative portfolio values starting at 1.0.
        """
        clean = historical_returns.dropna().values
        if len(clean) == 0:
            return np.ones((self.n_simulations, self.n_periods))

        simulations = np.zeros((self.n_simulations, self.n_periods))
        for i in range(self.n_simulations):
            random_returns = self.rng.choice(
                clean, size=self.n_periods, replace=True
            )
            simulations[i] = (1.0 + random_returns).cumprod()

        return simulations

    def regime_aware_simulation(
        self,
        historical_returns: pd.Series,
        regimes: pd.Series,
    ) -> np.ndarray:
        """Regime-aware simulation: sample returns conditional on regime.

        The *current* regime (last value in ``regimes``) determines which
        historical returns are eligible for resampling.  This preserves
        volatility-clustering and regime-specific distributional features.

        Parameters
        ----------
        historical_returns : pd.Series
            Daily historical returns (decimal form).
        regimes : pd.Series
            Regime labels aligned with ``historical_returns`` index.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_simulations, n_periods)`` containing
            cumulative portfolio values starting at 1.0.
        """
        clean_returns = historical_returns.dropna()
        clean_regimes = regimes.reindex(clean_returns.index).dropna()

        if len(clean_returns) == 0 or len(clean_regimes) == 0:
            return np.ones((self.n_simulations, self.n_periods))

        current_regime = clean_regimes.iloc[-1]
        mask = clean_regimes == current_regime
        regime_returns = clean_returns[mask].values

        # Fallback: if regime has too few observations, use all returns
        if len(regime_returns) < 10:
            regime_returns = clean_returns.values

        simulations = np.zeros((self.n_simulations, self.n_periods))
        for i in range(self.n_simulations):
            random_returns = self.rng.choice(
                regime_returns, size=self.n_periods, replace=True
            )
            simulations[i] = (1.0 + random_returns).cumprod()

        return simulations

    # ------------------------------------------------------------------
    # Distribution analysis
    # ------------------------------------------------------------------

    def forward_distribution(
        self, simulations: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate forward return distribution statistics.

        Parameters
        ----------
        simulations : np.ndarray
            Array of shape ``(n_simulations, n_periods)`` with cumulative
            portfolio values (starting at 1.0).

        Returns
        -------
        dict
            Dictionary with:

            - ``percentiles`` : dict with keys 5, 25, 50, 75, 95
            - ``mean`` : mean final value
            - ``std`` : standard deviation of final values
            - ``prob_profit`` : probability final value > 1.0
            - ``prob_loss`` : probability final value < 1.0
            - ``prob_loss_20pct`` : probability final value < 0.8
            - ``stress_paths`` : worst 5% simulation paths
        """
        if simulations.size == 0:
            return {
                "percentiles": {5: 1.0, 25: 1.0, 50: 1.0, 75: 1.0, 95: 1.0},
                "mean": 1.0,
                "std": 0.0,
                "prob_profit": 0.5,
                "prob_loss": 0.5,
                "prob_loss_20pct": 0.0,
                "stress_paths": np.array([]),
            }

        final_values = simulations[:, -1]

        percentiles = {
            5: float(np.percentile(final_values, 5)),
            25: float(np.percentile(final_values, 25)),
            50: float(np.percentile(final_values, 50)),
            75: float(np.percentile(final_values, 75)),
            95: float(np.percentile(final_values, 95)),
        }

        mean_val = float(final_values.mean())
        std_val = float(final_values.std())
        prob_profit = float((final_values > 1.0).sum() / len(final_values))
        prob_loss = 1.0 - prob_profit
        prob_loss_20pct = float((final_values < 0.8).sum() / len(final_values))

        # Stress paths: bottom 5% of simulations
        p5 = percentiles[5]
        stress_mask = final_values <= p5
        stress_paths = simulations[stress_mask] if stress_mask.any() else np.array([])

        return {
            "percentiles": percentiles,
            "mean": mean_val,
            "std": std_val,
            "prob_profit": prob_profit,
            "prob_loss": prob_loss,
            "prob_loss_20pct": prob_loss_20pct,
            "stress_paths": stress_paths,
        }
