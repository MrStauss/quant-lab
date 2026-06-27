"""Walk-forward backtesting engine with transaction costs and slippage."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantlab.config import QuantLabConfig
    from quantlab.signals.engine import SignalEngine
    from quantlab.risk.manager import RiskManager
    from quantlab.portfolio.allocator import PortfolioAllocator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtesting with transaction costs and slippage."""

    def __init__(self, config: "QuantLabConfig") -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Runtime configuration containing transaction_cost_bps,
                    slippage_bps, initial_capital, and other parameters.
        """
        self.config = config
        self.tc_bps: float = getattr(config, "transaction_cost_bps", 5.0)
        self.slippage_bps: float = getattr(config, "slippage_bps", 2.0)
        self.initial_capital: float = getattr(config, "initial_capital", 100_000.0)
        self.benchmark_ticker: str = getattr(config, "benchmark_ticker", "SPY")

    def run(
        self,
        prices: pd.DataFrame,
        signal_engine: "SignalEngine",
        risk_manager: "RiskManager",
        allocator: "PortfolioAllocator",
        rebalance_freq: str = "W",
    ) -> pd.DataFrame:
        """
        Run walk-forward backtest.

        Walk-forward logic:
        1. Initialize portfolio: cash = initial_capital, positions = {{}}
        2. For each rebalance date (weekly or daily per rebalance_freq):
           a. Compute features from prices up to that date
           b. Run signal_engine.run_all(features, prices)
           c. Risk check each signal
           d. allocator.allocate(approved_signals, features)
           e. Calculate trades needed (old_weights -> new_weights)
           f. Apply costs: cost = turnover * portfolio_value * (tc_bps + slippage_bps) / 10000
           g. Update portfolio value with strategy returns between rebalance dates
           h. Record: date, return, benchmark_return, equity, drawdown, weights, turnover

        Args:
            prices: OHLCV close prices (DataFrame with Date index, ticker columns).
            signal_engine: Configured with all strategies.
            risk_manager: Risk guardrails.
            allocator: Portfolio construction.
            rebalance_freq: "D" for daily or "W" for weekly rebalancing.

        Returns:
            DataFrame with columns:
            - date, strategy_return, benchmark_return, cumulative_value,
              drawdown, weights, turnover
        """
        if prices.empty:
            logger.warning("Empty prices DataFrame passed to BacktestEngine.run()")
            return pd.DataFrame(
                columns=[
                    "date",
                    "strategy_return",
                    "benchmark_return",
                    "cumulative_value",
                    "drawdown",
                    "weights",
                    "turnover",
                ]
            )

        # Ensure prices is a DataFrame with Date index and ticker columns
        if isinstance(prices.columns, pd.MultiIndex):
            # Extract Close prices from OHLCV MultiIndex
            if "Close" in prices.columns.get_level_values(1):
                prices = prices.xs("Close", level=1, axis=1)
            else:
                prices = prices.iloc[:, :]

        # Drop rows with all NaN
        prices = prices.dropna(how="all")

        # Generate rebalance dates
        if rebalance_freq.upper() == "D":
            rebalance_dates = prices.index
        else:
            # Weekly - use the last trading day of each week
            rebalance_dates = prices.resample("W").last().dropna().index

        # Filter to dates within prices range
        rebalance_dates = rebalance_dates[
            (rebalance_dates >= prices.index[0]) & (rebalance_dates <= prices.index[-1])
        ]

        # Calculate daily returns for all tickers
        daily_returns = prices.pct_change().fillna(0.0)

        # Calculate benchmark returns
        benchmark_returns = self._calculate_benchmark_returns(daily_returns)

        # Initialize portfolio state
        cash = self.initial_capital
        positions: Dict[str, float] = {}
        portfolio_value = self.initial_capital
        old_weights: Dict[str, float] = {}

        records: List[Dict[str, Any]] = []
        prev_rebalance_date: Optional[pd.Timestamp] = None

        for i, reb_date in enumerate(rebalance_dates):
            # Find the actual trading date closest to reb_date
            if reb_date in prices.index:
                trade_date = reb_date
            else:
                # Find nearest previous trading day
                valid_dates = prices.index[prices.index <= reb_date]
                if len(valid_dates) == 0:
                    continue
                trade_date = valid_dates[-1]

            # Step g: Update portfolio value with strategy returns since last rebalance
            if prev_rebalance_date is not None and old_weights:
                # Get the period between rebalances
                period_mask = (daily_returns.index > prev_rebalance_date) & (
                    daily_returns.index <= trade_date
                )
                period_returns = daily_returns.loc[period_mask]

                if not period_returns.empty:
                    # Calculate weighted portfolio return for each day
                    for _, day_rets in period_returns.iterrows():
                        port_ret = sum(
                            old_weights.get(ticker, 0.0) * day_rets.get(ticker, 0.0)
                            for ticker in old_weights
                        )
                        portfolio_value *= 1.0 + port_ret

            # Step a: Compute features from prices up to trade_date
            prices_up_to_date = prices.loc[prices.index <= trade_date]
            if len(prices_up_to_date) < 2:
                continue

            # Build features using the feature engine
            try:
                from quantlab.features.engine import FeatureEngine

                feat_engine = FeatureEngine(prices_up_to_date)
                features = feat_engine.compute_all()
            except Exception as exc:
                logger.warning("Feature computation failed at %s: %s", trade_date, exc)
                # Fallback: use returns as features
                features = daily_returns.loc[daily_returns.index <= trade_date].copy()

            # Step b: Run signal engine
            try:
                signals = signal_engine.run_all(features, prices_up_to_date)
            except Exception as exc:
                logger.warning("Signal engine failed at %s: %s", trade_date, exc)
                signals = []

            # Step c: Risk check each signal
            approved_signals: List[Dict[str, Any]] = []
            portfolio_state = {
                "cash": cash,
                "positions": positions,
                "equity": portfolio_value,
                "last_update": trade_date.isoformat(),
            }
            for signal in signals:
                try:
                    risk_result = risk_manager.check(signal, portfolio_state)
                    if risk_result.get("approved", False):
                        modified = risk_result.get("modified_signal", signal)
                        approved_signals.append(modified)
                except Exception as exc:
                    logger.warning("Risk check failed for signal: %s", exc)

            # Step d: Allocate
            if approved_signals:
                try:
                    allocation = allocator.allocate(approved_signals, features)
                    new_weights = allocation.get("final_weights", {})
                except Exception as exc:
                    logger.warning("Allocator failed at %s: %s", trade_date, exc)
                    new_weights = old_weights.copy() if old_weights else {}
            else:
                # No approved signals: hold cash
                new_weights = {}

            # Ensure weights sum to ~1.0 (allow for small numerical error)
            weight_sum = sum(new_weights.values())
            if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
                new_weights = {
                    k: v / weight_sum for k, v in new_weights.items()
                }
            elif weight_sum == 0:
                # All cash
                new_weights = {}

            # Step e: Calculate turnover
            turnover = self._calculate_turnover(old_weights, new_weights)

            # Step f: Apply transaction costs
            trade_changes = pd.Series({
                ticker: new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)
                for ticker in set(new_weights) | set(old_weights)
            })
            cost = self._apply_costs(trade_changes)
            portfolio_value -= cost

            # Update cash and positions
            cash = portfolio_value * (1.0 - sum(new_weights.values()))
            positions = {
                ticker: portfolio_value * weight
                for ticker, weight in new_weights.items()
                if weight > 0
            }
            old_weights = new_weights.copy()

            # Calculate period return
            if prev_rebalance_date is not None:
                period_mask = (daily_returns.index > prev_rebalance_date) & (
                    daily_returns.index <= trade_date
                )
                period_bench = benchmark_returns.loc[period_mask]
                bench_ret = (1 + period_bench).prod() - 1 if not period_bench.empty else 0.0
            else:
                bench_ret = 0.0

            # Calculate drawdown
            peak_value = max(
                (r["cumulative_value"] for r in records),
                default=self.initial_capital,
            )
            drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0

            # Calculate strategy return for this period
            if prev_rebalance_date is not None and records:
                prev_value = records[-1]["cumulative_value"]
                strat_ret = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            else:
                strat_ret = 0.0

            record = {
                "date": trade_date,
                "strategy_return": strat_ret,
                "benchmark_return": bench_ret,
                "cumulative_value": portfolio_value,
                "drawdown": drawdown,
                "weights": json.dumps(new_weights),
                "turnover": turnover,
            }
            records.append(record)
            prev_rebalance_date = trade_date

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df.set_index("date", inplace=True)
        return result_df

    def _apply_costs(self, trades: pd.Series) -> float:
        """
        Apply transaction costs and slippage.

        Total cost = sum(|trade|) * portfolio_value * (tc_bps + slippage_bps) / 10000
        Since trades are in weight-change terms, sum of abs changes is turnover * 2.
        We compute: turnover * portfolio_value * (tc_bps + slippage_bps) / 10000.

        Args:
            trades: Series of weight changes per ticker.

        Returns:
            Total cost in currency units.
        """
        turnover = trades.abs().sum() / 2.0
        total_bps = self.tc_bps + self.slippage_bps
        # Cost is calculated as a fraction of portfolio value
        # We return the cost as a fraction (caller multiplies by portfolio_value)
        cost_fraction = turnover * total_bps / 10_000.0
        return cost_fraction

    def _calculate_turnover(
        self, old_weights: Dict[str, float], new_weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover.

        Turnover = sum of absolute weight changes / 2.

        Args:
            old_weights: Previous portfolio weights (ticker -> weight).
            new_weights: New portfolio weights (ticker -> weight).

        Returns:
            Turnover as a fraction (0.0 to 1.0).
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        total_change = sum(
            abs(new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0))
            for ticker in all_tickers
        )
        return total_change / 2.0

    def _calculate_benchmark_returns(
        self, daily_returns: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate benchmark returns from daily returns DataFrame.

        Args:
            daily_returns: DataFrame with Date index, ticker columns of daily returns.

        Returns:
            Series of benchmark daily returns.
        """
        if self.benchmark_ticker in daily_returns.columns:
            return daily_returns[self.benchmark_ticker]
        # Fallback: equal-weighted average of all tickers
        return daily_returns.mean(axis=1)
