"""DuckDB/SQLite storage with Parquet archive for QuantLab.

The :class:`DataStore` class provides persistent storage for OHLCV
market data.  DuckDB is used as the primary engine when available;
otherwise the implementation transparently falls back to the SQLite3
standard library module.  Daily snapshots can additionally be written
to / read from Parquet files for offline analysis.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import DuckDB; fall back to None so we can use SQLite.
try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _HAS_DUCKDB = False

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_OHLCV_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    date     DATE NOT NULL,
    ticker   TEXT NOT NULL,
    "Open"   DOUBLE,
    "High"   DOUBLE,
    "Low"    DOUBLE,
    "Close"  DOUBLE,
    "Volume" BIGINT,
    PRIMARY KEY (date, ticker)
)
"""

_SQLITE_OHLCV_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    date     TEXT NOT NULL,
    ticker   TEXT NOT NULL,
    "Open"   REAL,
    "High"   REAL,
    "Low"    REAL,
    "Close"  REAL,
    "Volume" INTEGER,
    PRIMARY KEY (date, ticker)
)
"""


class DataStore:
    """Unified data store backed by DuckDB (primary) with Parquet archive.

    Falls back to the std-lib :mod:`sqlite3` module when DuckDB is not
    installed.

    Parameters
    ----------
    db_path :
        Path to the database file.  When ``None`` a default location
        ``<package_root>/../../data_store/quantlab.duckdb`` is used.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            # Resolve default relative to this file:
            #   quantlab/data/storage.py  ->  quantlab  ->  data_store/
            db_path = Path(__file__).parent.parent / ".." / "data_store" / "quantlab.duckdb"
        self.db_path: Path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = None
        self._backend: str = "duckdb" if _HAS_DUCKDB else "sqlite"

        self._connect()
        self._ensure_schema()
        logger.info("DataStore initialised (%s) -> %s", self._backend, self.db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Open the underlying database connection."""
        if self._backend == "duckdb":
            self._conn = duckdb.connect(str(self.db_path))
        else:
            self._conn = sqlite3.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        """Create the OHLCV table if it does not already exist."""
        ddl = _OHLCV_TABLE_DDL if self._backend == "duckdb" else _SQLITE_OHLCV_TABLE_DDL
        self._conn.execute(ddl)
        self._conn.commit()

    def _execute(self, sql: str, params: Optional[tuple] = None):
        """Execute *sql* and return the cursor ( DuckDB / SQLite compatible)."""
        if params is None:
            return self._conn.execute(sql)
        return self._conn.execute(sql, params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_ohlcv(self, data: pd.DataFrame) -> None:
        """Upsert OHLCV data.

        Overlapping ``(date, ticker)`` rows are replaced so that
        re-running a download for the same date range is idempotent.

        Parameters
        ----------
        data :
            DataFrame with MultiIndex columns ``(ticker, field)`` and
            ``Date`` index (as produced by :func:`quantlab.data.ingestion.download_ohlcv`).
        """
        if data is None or data.empty:
            logger.warning("save_ohlcv called with empty data; nothing to do")
            return

        # Normalise to long form: one row per (date, ticker)
        long_df = self._melt_to_long(data)
        if long_df.empty:
            logger.warning("save_ohlcv: no valid rows after melting")
            return

        # Delete overlapping keys, then insert fresh rows
        tickers = long_df["ticker"].unique().tolist()
        dates = [str(d) for d in long_df["date"].unique().tolist()]

        placeholders_tickers = ",".join("?" * len(tickers))
        placeholders_dates = ",".join("?" * len(dates))

        delete_sql = (
            f'DELETE FROM ohlcv WHERE ticker IN ({placeholders_tickers}) '
            f'AND date IN ({placeholders_dates})'
        )
        self._execute(delete_sql, tuple(tickers) + tuple(dates))

        # Insert new rows
        self._insert_long_df(long_df)
        self._conn.commit()
        logger.info(
            "save_ohlcv: upserted %d rows for %d tickers",
            len(long_df),
            len(tickers),
        )

    def load_ohlcv(
        self,
        tickers: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data, optionally filtered.

        Parameters
        ----------
        tickers :
            Restrict to these ticker symbols.  ``None`` = all tickers.
        start_date :
            Inclusive start date filter (YYYY-MM-DD).
        end_date :
            Inclusive end date filter (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex columns ``(ticker, field)`` and
            ``Date`` index.  Empty DataFrame if no data matches.
        """
        conditions: list = []
        params: list = []

        if tickers:
            placeholders = ",".join("?" * len(tickers))
            conditions.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f'SELECT * FROM ohlcv {where_clause} ORDER BY date, ticker'

        df = pd.read_sql_query(sql, self._conn, params=params or None)
        if df.empty:
            return pd.DataFrame(
                columns=pd.MultiIndex.from_product([[], []], names=["ticker", "field"]),
            )

        return self._pivot_to_wide(df)

    def archive_to_parquet(self, data: pd.DataFrame, date_str: str) -> None:
        """Save a daily snapshot to the Parquet archive.

        Falls back to CSV format when pyarrow / fastparquet is not
        installed.

        Parameters
        ----------
        data :
            OHLCV DataFrame (MultiIndex columns).
        date_str :
            Snapshot date string used in the filename
            (e.g. ``"2024-01-15"``).
        """
        archive_dir = self.db_path.parent / "parquet"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Flatten columns for archive compatibility: "AAPL_Close", etc.
        flat = data.copy()
        flat.columns = [f"{ticker}_{field}" for ticker, field in flat.columns]
        flat = flat.reset_index()

        out_path = archive_dir / f"ohlcv_{date_str}.parquet"
        csv_path = archive_dir / f"ohlcv_{date_str}.csv"

        try:
            flat.to_parquet(out_path, index=False)
            logger.info("archive_to_parquet: wrote %s", out_path)
        except ImportError:
            logger.warning(
                "pyarrow not available; falling back to CSV archive: %s", csv_path
            )
            flat.to_csv(csv_path, index=False)

    def load_parquet_archive(self, date_str: str) -> pd.DataFrame:
        """Load a specific Parquet archive.

        Also checks for a CSV fallback file when the Parquet file is
        not present.

        Parameters
        ----------
        date_str :
            Snapshot date string (e.g. ``"2024-01-15"``).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with MultiIndex columns, or empty DataFrame
            if the file does not exist.
        """
        archive_dir = self.db_path.parent / "parquet"
        in_path = archive_dir / f"ohlcv_{date_str}.parquet"
        csv_path = archive_dir / f"ohlcv_{date_str}.csv"

        if not in_path.exists() and not csv_path.exists():
            logger.warning("load_parquet_archive: file not found: %s", in_path)
            return pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [[], []], names=["ticker", "field"]
                ),
            )

        if in_path.exists():
            try:
                flat = pd.read_parquet(in_path)
            except ImportError:
                logger.warning("pyarrow not available; cannot load %s", in_path)
                return pd.DataFrame(
                    columns=pd.MultiIndex.from_product(
                        [[], []], names=["ticker", "field"]
                    ),
                )
        else:
            flat = pd.read_csv(csv_path)

        if "Date" in flat.columns:
            flat = flat.set_index("Date")
        elif "date" in flat.columns:
            flat = flat.set_index("date")

        # Rebuild MultiIndex columns from flattened names
        parsed: list = []
        for col in flat.columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                parsed.append(tuple(parts))
            else:
                parsed.append((col, ""))
        flat.columns = pd.MultiIndex.from_tuples(parsed, names=["ticker", "field"])

        return flat

    def get_latest_date(self) -> Optional[str]:
        """Return the most recent date stored in the database.

        Returns ``None`` when the table is empty.
        """
        cursor = self._execute('SELECT MAX(date) FROM ohlcv')
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("DataStore connection closed")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Shape conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _melt_to_long(data: pd.DataFrame) -> pd.DataFrame:
        """Convert a wide MultiIndex DataFrame to long form.

        Input:  columns = (ticker, field), index = Date
        Output: columns = [date, ticker, Open, High, Low, Close, Volume]
        """
        if data.columns.nlevels != 2:
            raise ValueError(
                f"Expected 2-level MultiIndex columns, got {data.columns.nlevels}"
            )

        # Stack tickers from columns to rows -> index = (Date, ticker)
        stacked = data.stack(level=0, future_stack=True)
        stacked.index.names = ["date", "ticker"]
        stacked = stacked.reset_index()

        # Convert Timestamp to ISO date strings for SQLite compatibility
        stacked["date"] = pd.to_datetime(stacked["date"]).dt.strftime("%Y-%m-%d")

        # Ensure standard columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in stacked.columns:
                stacked[col] = None  # type: ignore[assignment]

        return stacked[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def _pivot_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a long-form DataFrame back to wide MultiIndex form.

        Input:  columns = [date, ticker, Open, High, Low, Close, Volume]
        Output: columns = (ticker, field), index = Date
        """
        long_df = long_df.copy()
        long_df["date"] = pd.to_datetime(long_df["date"])
        long_df = long_df.set_index(["date", "ticker"])

        # Select only the OHLCV field columns
        field_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in long_df.columns]
        wide = long_df[field_cols].unstack(level="ticker")
        # Swap levels so columns are (ticker, field)
        wide = wide.swaplevel(axis=1)
        wide.columns.names = ["ticker", "field"]
        # Sort ticker level for determinism
        wide = wide.sort_index(axis=1)
        return wide

    def _insert_long_df(self, long_df: pd.DataFrame) -> None:
        """Insert a long-form DataFrame into the database."""
        if self._backend == "duckdb":
            # DuckDB can ingest DataFrames directly via register + INSERT
            self._conn.register("_temp_df", long_df)
            self._conn.execute(
                'INSERT INTO ohlcv SELECT * FROM _temp_df'
            )
            self._conn.unregister("_temp_df")
        else:
            # SQLite -> use executemany with to_numpy/tolist
            rows = long_df[[
                "date", "ticker", "Open", "High", "Low", "Close", "Volume"
            ]].values.tolist()
            self._conn.executemany(
                'INSERT INTO ohlcv (date, ticker, "Open", "High", "Low", "Close", "Volume") '
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
