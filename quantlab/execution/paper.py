"""Paper trading engine for QuantLab.

Simulates execution without a real broker. Tracks equity curve,
maintains trade blotter, and persists state via PortfolioRepository.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantlab.config import QuantLabConfig
    from quantlab.storage.repository import PortfolioRepository

logger = logging.getLogger(__name__)


class PaperTrader:
    """Paper trading engine. Simulates execution without real broker."""

    def __init__(
        self, config: "QuantLabConfig", repository: "PortfolioRepository"
    ) -> None:
        """
        Initialize the paper trader.

        Args:
            config: Runtime configuration with transaction_cost_bps,
                    initial_capital, etc.
            repository: PortfolioRepository for persisting state.
        """
        self.config = config
        self.repository = repository
        self.tc_bps: float = getattr(config, "transaction_cost_bps", 5.0)
        self.initial_capital: float = getattr(config, "initial_capital", 100_000.0)

        # In-memory state tracking
        self.current_cash: float = self.initial_capital
        self.current_positions: Dict[str, float] = {}  # ticker -> shares
        self.equity_history: List[Dict[str, Any]] = []  # [{date, equity}, ...]
        self.trade_log: List[Dict[str, Any]] = []  # All trade blotter entries
        self.allocation_history: List[Dict[str, Any]] = []  # Weight history

        # Track previous portfolio value for daily PnL
        self._previous_portfolio_value: float = self.initial_capital

    def execute_signal(
        self, signal: Dict[str, Any], prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate executing a signal.

        Steps:
        1. Get current prices for all tickers in signal
        2. Calculate shares for each position (weight * portfolio_value / price)
        3. Record fills: {ticker: {"price": price, "shares": shares}}
        4. Calculate fees: sum(|shares| * price) * tc_bps / 10000
        5. Update portfolio value, daily P&L
        6. Save to repository
        7. Return trade blotter entry dict

        Args:
            signal: Signal dict with "date", "weights" (ticker -> weight),
                    "strategy", "family", "confidence", etc.
            prices: DataFrame with Date index and ticker columns (Close prices),
                    or MultiIndex columns from OHLCV.

        Returns:
            Trade blotter entry dict:
            {
                "date": str,
                "signal": dict,
                "fills": {ticker: {"price": float, "shares": float}},
                "fees": float,
                "portfolio_value": float,
                "daily_pnl": float
            }
        """
        signal_date = signal.get("date", datetime.now().strftime("%Y-%m-%d"))
        weights: Dict[str, float] = signal.get("weights", {})

        # Extract close prices
        if isinstance(prices.columns, pd.MultiIndex):
            if "Close" in prices.columns.get_level_values(1):
                close_prices = prices.xs("Close", level=1, axis=1)
            else:
                close_prices = prices
        else:
            close_prices = prices

        # Get the latest available prices on or before signal_date
        if isinstance(close_prices.index, pd.DatetimeIndex):
            mask = close_prices.index <= pd.Timestamp(signal_date)
            available_dates = close_prices.index[mask]
            if len(available_dates) > 0:
                latest_date = available_dates[-1]
                current_prices = close_prices.loc[latest_date]
            else:
                current_prices = close_prices.iloc[-1] if len(close_prices) > 0 else pd.Series()
        else:
            current_prices = close_prices.iloc[-1] if len(close_prices) > 0 else pd.Series()

        # Calculate current portfolio value before execution
        old_position_values = {
            ticker: self.current_positions.get(ticker, 0.0) * (
                current_prices.get(ticker, 0.0) if hasattr(current_prices, "get") else 0.0
            )
            for ticker in self.current_positions
        }
        old_portfolio_value = self.current_cash + sum(old_position_values.values())
        if old_portfolio_value <= 0:
            old_portfolio_value = self._previous_portfolio_value

        # Calculate target shares for new positions
        fills: Dict[str, Dict[str, float]] = {}
        new_positions: Dict[str, float] = {}
        total_order_value = 0.0

        for ticker, weight in weights.items():
            if weight <= 0:
                continue
            price = (
                current_prices.get(ticker, np.nan)
                if hasattr(current_prices, "get")
                else np.nan
            )
            if np.isnan(price) or price <= 0:
                logger.warning("No valid price for %s on %s, skipping", ticker, signal_date)
                continue

            target_value = old_portfolio_value * weight
            shares = target_value / price
            new_positions[ticker] = shares
            fills[ticker] = {"price": float(price), "shares": float(shares)}
            total_order_value += abs(shares) * price

        # Calculate fees
        fees = total_order_value * self.tc_bps / 10_000.0

        # Update cash: subtract invested amount and fees
        invested = sum(
            fills[t]["price"] * fills[t]["shares"] for t in fills
        )
        self.current_cash = old_portfolio_value - invested - fees

        # Update positions
        self.current_positions = new_positions.copy()

        # Calculate new portfolio value
        new_position_values = {
            ticker: self.current_positions.get(ticker, 0.0) * (
                current_prices.get(ticker, 0.0) if hasattr(current_prices, "get") else 0.0
            )
            for ticker in self.current_positions
        }
        new_portfolio_value = self.current_cash + sum(new_position_values.values())

        # Daily PnL
        daily_pnl = new_portfolio_value - self._previous_portfolio_value

        # Update tracking
        self._previous_portfolio_value = new_portfolio_value

        # Build trade blotter entry
        trade_entry = {
            "date": signal_date,
            "signal": signal,
            "fills": fills,
            "fees": fees,
            "portfolio_value": new_portfolio_value,
            "daily_pnl": daily_pnl,
        }

        # Record equity history
        self.equity_history.append(
            {"date": signal_date, "equity": new_portfolio_value}
        )

        # Record trade log
        self.trade_log.append(trade_entry)

        # Record allocation history
        self.allocation_history.append(
            {
                "date": signal_date,
                "weights": weights,
                "strategy": signal.get("strategy", ""),
                "family": signal.get("family", ""),
                "confidence": signal.get("confidence", 0.0),
            }
        )

        # Persist to repository
        try:
            self.repository.save_trade(trade_entry)
            self.repository.save_portfolio_state(
                {
                    "cash": self.current_cash,
                    "positions": self.current_positions,
                    "equity": new_portfolio_value,
                    "last_update": signal_date,
                }
            )
        except Exception as exc:
            logger.warning("Failed to persist trade/state to repository: %s", exc)

        return trade_entry

    def get_equity_curve(self) -> pd.Series:
        """
        Get portfolio equity over time.

        Returns:
            pd.Series with Date index and portfolio values.
        """
        if not self.equity_history:
            return pd.Series(dtype=float, name="equity")

        df = pd.DataFrame(self.equity_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date").sort_index()
        return df["equity"]

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get all simulated trades.

        Returns:
            pd.DataFrame with trade blotter entries.
        """
        if not self.trade_log:
            return pd.DataFrame(
                columns=["date", "signal", "fills", "fees", "portfolio_value", "daily_pnl"]
            )

        df = pd.DataFrame(self.trade_log)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def get_allocation_history(self) -> pd.DataFrame:
        """
        Get allocation history over time.

        Returns:
            pd.DataFrame with columns: date, weights, strategy, family, confidence.
        """
        if not self.allocation_history:
            return pd.DataFrame(
                columns=["date", "weights", "strategy", "family", "confidence"]
            )

        df = pd.DataFrame(self.allocation_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df
