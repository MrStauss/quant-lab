"""Global configuration for QuantLab."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / ".." / "data_store"
DB_PATH = DATA_DIR / "quantlab.duckdb"
PARQUET_DIR = DATA_DIR / "parquet"
REPORTS_DIR = BASE_DIR / ".." / "reports"
LOGS_DIR = BASE_DIR / ".." / "logs"

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2026-06-26"

# ---------------------------------------------------------------------------
# Asset families
# ---------------------------------------------------------------------------
ASSET_FAMILIES: Dict[str, Dict[str, str]] = {
    "Semiconductors": {
        "bull": "SOXL",
        "base": "SOXX",
        "bear": "SOXS",
        "safe_haven": "GLD",
        "cash": "BIL",
    },
    "Tech": {
        "bull": "TECL",
        "base": "XLK",
        "bear": "TECS",
        "safe_haven": "GLD",
        "cash": "BIL",
    },
    "SP500": {
        "bull": "UPRO",
        "base": "SPY",
        "bear": "SPXU",
        "safe_haven": "SH",
        "cash": "BIL",
    },
    "Russell2000": {
        "bull": "TNA",
        "base": "IWM",
        "bear": "TZA",
        "safe_haven": "GLD",
        "cash": "BIL",
    },
    "Energy": {
        "bull": "ERX",
        "base": "XLE",
        "bear": "ERY",
        "safe_haven": "GLD",
        "cash": "BIL",
    },
    "Financials": {
        "bull": "FAS",
        "base": "XLF",
        "bear": "FAZ",
        "safe_haven": "GLD",
        "cash": "BIL",
    },
}

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 60  # days for covariance calculations
REBALANCE_FREQ = "W"  # Weekly rebalancing
INITIAL_CAPITAL = 100_000

# ---------------------------------------------------------------------------
# Monte Carlo parameters
# ---------------------------------------------------------------------------
MC_SIMULATIONS = 2_500
MC_PERIODS = 504  # ~2 years forward

# ---------------------------------------------------------------------------
# Combined strategy weights
# ---------------------------------------------------------------------------
COMBINED_STRATEGY_WEIGHTS = {
    "regime_rotation": 0.50,
    "dynamic_allocation": 0.50,
}

# ---------------------------------------------------------------------------
# Risk management parameters
# ---------------------------------------------------------------------------
MAX_POSITION_SIZE = 0.60  # Max 60% in single position
MAX_LEVERAGED_ETF_EXPOSURE = 0.80  # Max 80% in leveraged ETFs
TARGET_VOLATILITY = 0.15  # 15% annualized target vol
MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown guardrail
CASH_FALLBACK_TICKER = "BIL"
KILL_SWITCH_ENABLED = True

# ---------------------------------------------------------------------------
# Backtesting parameters
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 5  # 5 basis points per trade
SLIPPAGE_BPS = 2  # 2 basis points slippage
BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------
PAPER_MODE_DEFAULT = True

# ---------------------------------------------------------------------------
# Feature parameters
# ---------------------------------------------------------------------------
RSI_SHORT_WINDOW = 9
RSI_LONG_WINDOW = 14
VOLATILITY_REGIME_BINS = 3  # low, medium, high
TREND_ZSCORE_WINDOW = 20


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class QuantLabConfig:
    """Runtime configuration container for QuantLab.

    Attributes
    ----------
    start_date : str
        Backtest start date (YYYY-MM-DD).
    end_date : str
        Backtest end date (YYYY-MM-DD).
    asset_families : dict
        Mapping of asset family names to their constituent tickers.
    initial_capital : float
        Starting portfolio value in USD.
    rebalance_freq : str
        Rebalancing frequency: "D" daily, "W" weekly, "M" monthly.
    paper_mode : bool
        If True, trades are simulated (no real broker execution).
    max_position_size : float
        Maximum weight for any single position (0.0 - 1.0).
    max_leveraged_exposure : float
        Maximum aggregate weight in leveraged ETFs.
    target_volatility : float
        Annualized volatility target (e.g., 0.15 for 15%).
    max_drawdown_pct : float
        Maximum allowable drawdown before kill-switch triggers.
    transaction_cost_bps : float
        One-way transaction cost in basis points.
    slippage_bps : float
        Slippage assumption in basis points.
    mc_simulations : int
        Number of Monte Carlo simulation paths.
    mc_periods : int
        Number of periods (days) in each MC simulation.
    db_path : Path
        Path to the DuckDB database file.
    data_dir : Path
        Root directory for data storage.
    reports_dir : Path
        Directory where reports are written.
    """

    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    asset_families: Dict = field(default_factory=lambda: ASSET_FAMILIES.copy())
    initial_capital: float = INITIAL_CAPITAL
    rebalance_freq: str = REBALANCE_FREQ
    paper_mode: bool = PAPER_MODE_DEFAULT
    max_position_size: float = MAX_POSITION_SIZE
    max_leveraged_exposure: float = MAX_LEVERAGED_ETF_EXPOSURE
    target_volatility: float = TARGET_VOLATILITY
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS
    mc_simulations: int = MC_SIMULATIONS
    mc_periods: int = MC_PERIODS
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR
