"""Portfolio repository for persisting signals, trades, allocations, and state.

Uses DuckDB (primary) with SQLite fallback for structured data persistence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# SQL schema definitions
_CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    date TEXT,
    strategy TEXT,
    family TEXT,
    weights_json TEXT,
    confidence REAL,
    risk_state TEXT,
    reason TEXT
);
"""

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    date TEXT,
    signal_json TEXT,
    fills_json TEXT,
    fees REAL,
    portfolio_value REAL,
    daily_pnl REAL
);
"""

_CREATE_ALLOCATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS allocations (
    date TEXT,
    weights_json TEXT,
    family_rankings_json TEXT,
    risk_score TEXT,
    confidence REAL,
    reason TEXT
);
"""

_CREATE_PORTFOLIO_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY,
    cash REAL,
    positions_json TEXT,
    equity REAL,
    last_update TEXT
);
"""

_INIT_PORTFOLIO_STATE = """
INSERT OR IGNORE INTO portfolio_state (id, cash, positions_json, equity, last_update)
VALUES (1, 0.0, '{}', 0.0, '');
"""


class PortfolioRepository:
    """
    Persist signals, trades, allocations, and portfolio state.
    Uses DuckDB (primary) with SQLite fallback.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the repository.

        Args:
            db_path: Path to the database file. If None, uses in-memory SQLite.
        """
        self.db_path = db_path or Path(":memory:")
        self._using_duckdb = False
        self._conn: Any = None

        # Try DuckDB first, fall back to SQLite
        try:
            import duckdb

            self._conn = duckdb.connect(str(self.db_path))
            self._using_duckdb = True
            logger.info("Using DuckDB at %s", self.db_path)
        except ImportError:
            logger.info("DuckDB not available, falling back to SQLite")
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        self._initialize_schema()

    def _execute(self, sql: str, parameters: Optional[tuple] = None) -> Any:
        """Execute SQL with optional parameters."""
        if parameters:
            return self._conn.execute(sql, parameters)
        return self._conn.execute(sql)

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        self._execute(_CREATE_SIGNALS_TABLE)
        self._execute(_CREATE_TRADES_TABLE)
        self._execute(_CREATE_ALLOCATIONS_TABLE)
        self._execute(_CREATE_PORTFOLIO_STATE_TABLE)
        self._execute(_INIT_PORTFOLIO_STATE)

        # Commit for SQLite
        if not self._using_duckdb:
            self._conn.commit()

    def save_signal(self, signal: Dict[str, Any]) -> None:
        """
        Insert a signal into the signals table.

        Args:
            signal: Signal dict with "date", "strategy", "family",
                    "weights", "confidence", "risk_state", "reason".
        """
        sql = """
        INSERT INTO signals (date, strategy, family, weights_json, confidence, risk_state, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            signal.get("date", ""),
            signal.get("strategy", ""),
            signal.get("family", ""),
            json.dumps(signal.get("weights", {})),
            signal.get("confidence", 0.0),
            signal.get("risk_state", ""),
            signal.get("reason", ""),
        )
        self._execute(sql, params)
        if not self._using_duckdb:
            self._conn.commit()

    def load_signals(
        self,
        date: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load signals with optional filters.

        Args:
            date: Filter by date (ISO format string). If None, load all dates.
            strategy: Filter by strategy name. If None, load all strategies.

        Returns:
            DataFrame with signal records.
        """
        sql = "SELECT * FROM signals WHERE 1=1"
        params: list = []

        if date is not None:
            sql += " AND date = ?"
            params.append(date)
        if strategy is not None:
            sql += " AND strategy = ?"
            params.append(strategy)

        sql += " ORDER BY date"

        result = self._execute(sql, tuple(params) if params else None)
        rows = result.fetchall()
        columns = [desc[0] for desc in result.description] if result.description else []

        if not rows:
            return pd.DataFrame(
                columns=[
                    "date", "strategy", "family", "weights_json",
                    "confidence", "risk_state", "reason",
                ]
            )

        df = pd.DataFrame(rows, columns=columns)
        return df

    def save_trade(self, trade: Dict[str, Any]) -> None:
        """
        Insert a trade into the trades table.

        Args:
            trade: Trade dict with "date", "signal", "fills",
                   "fees", "portfolio_value", "daily_pnl".
        """
        sql = """
        INSERT INTO trades (date, signal_json, fills_json, fees, portfolio_value, daily_pnl)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            trade.get("date", ""),
            json.dumps(trade.get("signal", {})),
            json.dumps(trade.get("fills", {})),
            trade.get("fees", 0.0),
            trade.get("portfolio_value", 0.0),
            trade.get("daily_pnl", 0.0),
        )
        self._execute(sql, params)
        if not self._using_duckdb:
            self._conn.commit()

    def save_allocation(self, allocation: Dict[str, Any]) -> None:
        """
        Insert an allocation into the allocations table.

        Args:
            allocation: Allocation dict with "date", "final_weights",
                        "family_rankings", "risk_score", "confidence", "reason".
        """
        sql = """
        INSERT INTO allocations (date, weights_json, family_rankings_json, risk_score, confidence, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            allocation.get("date", ""),
            json.dumps(allocation.get("final_weights", {})),
            json.dumps(allocation.get("family_rankings", [])),
            allocation.get("risk_score", ""),
            allocation.get("confidence", 0.0),
            allocation.get("reason", ""),
        )
        self._execute(sql, params)
        if not self._using_duckdb:
            self._conn.commit()

    def get_latest_allocation(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent allocation.

        Returns:
            Most recent allocation dict or None if no allocations exist.
        """
        sql = """
        SELECT * FROM allocations
        ORDER BY date DESC
        LIMIT 1
        """
        result = self._execute(sql)
        row = result.fetchone()

        if row is None:
            return None

        columns = [desc[0] for desc in result.description] if result.description else []
        row_dict = dict(zip(columns, row))

        return {
            "date": row_dict.get("date", ""),
            "final_weights": json.loads(row_dict.get("weights_json", "{}")),
            "family_rankings": json.loads(row_dict.get("family_rankings_json", "[]")),
            "risk_score": row_dict.get("risk_score", ""),
            "confidence": row_dict.get("confidence", 0.0),
            "reason": row_dict.get("reason", ""),
        }

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state (cash, positions, equity).

        Returns:
            Dict with "cash", "positions", "equity", "last_update".
        """
        sql = "SELECT * FROM portfolio_state WHERE id = 1"
        result = self._execute(sql)
        row = result.fetchone()

        if row is None:
            return {
                "cash": 0.0,
                "positions": {},
                "equity": 0.0,
                "last_update": "",
            }

        columns = [desc[0] for desc in result.description] if result.description else []
        row_dict = dict(zip(columns, row))

        positions_json = row_dict.get("positions_json", "{}")
        try:
            positions = json.loads(positions_json) if positions_json else {}
        except json.JSONDecodeError:
            positions = {}

        return {
            "cash": row_dict.get("cash", 0.0),
            "positions": positions,
            "equity": row_dict.get("equity", 0.0),
            "last_update": row_dict.get("last_update", ""),
        }

    def save_portfolio_state(self, state: Dict[str, Any]) -> None:
        """
        Upsert portfolio state.

        Args:
            state: Dict with "cash", "positions", "equity", "last_update".
        """
        positions = state.get("positions", {})
        if isinstance(positions, dict):
            positions_json = json.dumps(positions)
        else:
            positions_json = json.dumps(dict(positions))

        sql = """
        INSERT INTO portfolio_state (id, cash, positions_json, equity, last_update)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            cash = excluded.cash,
            positions_json = excluded.positions_json,
            equity = excluded.equity,
            last_update = excluded.last_update
        """
        params = (
            state.get("cash", 0.0),
            positions_json,
            state.get("equity", 0.0),
            state.get("last_update", ""),
        )
        self._execute(sql, params)
        if not self._using_duckdb:
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()

    def __enter__(self) -> "PortfolioRepository":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
