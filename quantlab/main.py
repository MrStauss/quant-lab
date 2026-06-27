"""CLI entry point for QuantLab.

Provides a unified command-line interface with four modes:

- **backtest** — Run a walk-forward backtest over historical data.
- **paper** — Run the daily pipeline once in paper-trading mode.
- **daemon** — Run the daily pipeline once (V1.0, no scheduling).
- **report** — Generate the latest HTML report from saved data.

Usage::

    python -m quantlab.main backtest
    python -m quantlab.main paper
    python -m quantlab.main daemon
    python -m quantlab.main report
    python -m quantlab.main --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Imports — guarded for graceful degradation
# ---------------------------------------------------------------------------


def _import_quantlab_modules() -> Dict[str, Any]:
    """Import QuantLab modules with graceful error handling.

    Returns:
        Dictionary mapping module names to imported objects.
    """
    modules: Dict[str, Any] = {}

    try:
        from quantlab.config import QuantLabConfig, ASSET_FAMILIES
        modules["QuantLabConfig"] = QuantLabConfig
        modules["ASSET_FAMILIES"] = ASSET_FAMILIES
    except ImportError as e:
        logger.error("Failed to import quantlab.config: %s", e)
        modules["QuantLabConfig"] = None
        modules["ASSET_FAMILIES"] = {}

    try:
        from quantlab.data.ingestion import download_ohlcv, download_latest_data
        modules["download_ohlcv"] = download_ohlcv
        modules["download_latest_data"] = download_latest_data
    except ImportError as e:
        logger.warning("Failed to import data.ingestion: %s", e)
        modules["download_ohlcv"] = None
        modules["download_latest_data"] = None

    try:
        from quantlab.data.validation import validate_ohlcv
        modules["validate_ohlcv"] = validate_ohlcv
    except ImportError as e:
        logger.warning("Failed to import data.validation: %s", e)
        modules["validate_ohlcv"] = None

    try:
        from quantlab.data.storage import DataStore
        modules["DataStore"] = DataStore
    except ImportError as e:
        logger.warning("Failed to import data.storage: %s", e)
        modules["DataStore"] = None

    try:
        from quantlab.features.engine import FeatureEngine
        modules["FeatureEngine"] = FeatureEngine
    except ImportError as e:
        logger.warning("Failed to import features.engine: %s", e)
        modules["FeatureEngine"] = None

    try:
        from quantlab.signals.engine import SignalEngine
        modules["SignalEngine"] = SignalEngine
    except ImportError as e:
        logger.warning("Failed to import signals.engine: %s", e)
        modules["SignalEngine"] = None

    try:
        from quantlab.risk.manager import RiskManager
        modules["RiskManager"] = RiskManager
    except ImportError as e:
        logger.warning("Failed to import risk.manager: %s", e)
        modules["RiskManager"] = None

    try:
        from quantlab.portfolio.allocator import PortfolioAllocator
        modules["PortfolioAllocator"] = PortfolioAllocator
    except ImportError as e:
        logger.warning("Failed to import portfolio.allocator: %s", e)
        modules["PortfolioAllocator"] = None

    try:
        from quantlab.execution.paper import PaperTrader
        modules["PaperTrader"] = PaperTrader
    except ImportError as e:
        logger.warning("Failed to import execution.paper: %s", e)
        modules["PaperTrader"] = None

    try:
        from quantlab.storage.repository import PortfolioRepository
        modules["PortfolioRepository"] = PortfolioRepository
    except ImportError as e:
        logger.warning("Failed to import storage.repository: %s", e)
        modules["PortfolioRepository"] = None

    try:
        from quantlab.backtesting.engine import BacktestEngine
        modules["BacktestEngine"] = BacktestEngine
    except ImportError as e:
        logger.warning("Failed to import backtesting.engine: %s", e)
        modules["BacktestEngine"] = None

    try:
        from quantlab.backtesting.benchmarks import calculate_benchmark_returns
        modules["calculate_benchmark_returns"] = calculate_benchmark_returns
    except ImportError as e:
        logger.warning("Failed to import backtesting.benchmarks: %s", e)
        modules["calculate_benchmark_returns"] = None

    try:
        from quantlab.utils.metrics import calculate_all_metrics
        modules["calculate_all_metrics"] = calculate_all_metrics
    except ImportError as e:
        logger.warning("Failed to import utils.metrics: %s", e)
        modules["calculate_all_metrics"] = None

    try:
        from quantlab.utils.reporting import generate_daily_report
        modules["generate_daily_report"] = generate_daily_report
    except ImportError as e:
        logger.warning("Failed to import utils.reporting: %s", e)
        modules["generate_daily_report"] = None

    try:
        from quantlab.daemon.scheduler import DailyDaemon
        modules["DailyDaemon"] = DailyDaemon
    except ImportError as e:
        logger.warning("Failed to import daemon.scheduler: %s", e)
        modules["DailyDaemon"] = None

    try:
        from quantlab.strategies.regime import HMMRegimeStrategy
        modules["HMMRegimeStrategy"] = HMMRegimeStrategy
    except ImportError as e:
        logger.warning("Failed to import strategies.regime: %s", e)
        modules["HMMRegimeStrategy"] = None

    try:
        from quantlab.strategies.dynamic import DynamicCovarianceStrategy
        modules["DynamicCovarianceStrategy"] = DynamicCovarianceStrategy
    except ImportError as e:
        logger.warning("Failed to import strategies.dynamic: %s", e)
        modules["DynamicCovarianceStrategy"] = None

    try:
        from quantlab.strategies.combined import CombinedStrategy
        modules["CombinedStrategy"] = CombinedStrategy
    except ImportError as e:
        logger.warning("Failed to import strategies.combined: %s", e)
        modules["CombinedStrategy"] = None

    try:
        from quantlab.strategies.composer import ComposerRSIStrategy
        modules["ComposerRSIStrategy"] = ComposerRSIStrategy
    except ImportError as e:
        logger.warning("Failed to import strategies.composer: %s", e)
        modules["ComposerRSIStrategy"] = None

    try:
        from quantlab.strategies.mean_reversion import MeanReversionStrategy
        modules["MeanReversionStrategy"] = MeanReversionStrategy
    except ImportError as e:
        logger.warning("Failed to import strategies.mean_reversion: %s", e)
        modules["MeanReversionStrategy"] = None

    return modules


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace, modules: Dict[str, Any]) -> int:
    """Run walk-forward backtest.

    Args:
        args: Parsed command-line arguments.
        modules: Imported QuantLab modules.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting backtest mode")

    QuantLabConfig = modules.get("QuantLabConfig")
    if QuantLabConfig is None:
        print("ERROR: Cannot import QuantLabConfig", file=sys.stderr)
        return 1

    config = QuantLabConfig()

    # Download data
    download_ohlcv = modules.get("download_ohlcv")
    if download_ohlcv is None:
        print("ERROR: Cannot import download_ohlcv", file=sys.stderr)
        return 1

    try:
        from quantlab.config import ASSET_FAMILIES

        all_tickers: List[str] = []
        for family in ASSET_FAMILIES.values():
            all_tickers.extend(family.values())
        all_tickers = list(set(all_tickers))
        if "SPY" not in all_tickers:
            all_tickers.append("SPY")

        logger.info("Downloading data for %d tickers...", len(all_tickers))
        raw_data = download_ohlcv(
            tickers=all_tickers,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        logger.info("Downloaded %d rows of data", len(raw_data))
    except Exception as exc:
        logger.exception("Failed to download data: %s", exc)
        print(f"ERROR: Failed to download data: {exc}", file=sys.stderr)
        return 1

    # Extract close prices
    import pandas as pd

    if isinstance(raw_data.columns, pd.MultiIndex):
        close_prices = raw_data.xs("Close", level=1, axis=1)
    else:
        close_prices = raw_data.copy()

    # Setup components
    FeatureEngine = modules.get("FeatureEngine")
    SignalEngine = modules.get("SignalEngine")
    RiskManager = modules.get("RiskManager")
    PortfolioAllocator = modules.get("PortfolioAllocator")
    BacktestEngine = modules.get("BacktestEngine")

    if any(m is None for m in [FeatureEngine, SignalEngine, RiskManager, PortfolioAllocator, BacktestEngine]):
        print("ERROR: Required modules not available for backtest", file=sys.stderr)
        return 1

    # Build feature engine
    feature_engine = FeatureEngine(prices=close_prices)
    features = feature_engine.compute_all()

    # Build signal engine with strategies
    signal_engine = SignalEngine(config=config)

    # Register strategies for each family
    for family_name, assets in ASSET_FAMILIES.items():
        if modules.get("HMMRegimeStrategy"):
            signal_engine.register_strategy(modules["HMMRegimeStrategy"](family_name=family_name, assets=assets))
        if modules.get("DynamicCovarianceStrategy"):
            signal_engine.register_strategy(modules["DynamicCovarianceStrategy"](family_name=family_name, assets=assets))
        if modules.get("CombinedStrategy"):
            signal_engine.register_strategy(modules["CombinedStrategy"](family_name=family_name, assets=assets))

    # Global strategies
    if modules.get("ComposerRSIStrategy"):
        signal_engine.register_strategy(modules["ComposerRSIStrategy"]())
    if modules.get("MeanReversionStrategy"):
        signal_engine.register_strategy(modules["MeanReversionStrategy"]())

    # Risk manager and allocator
    risk_manager = RiskManager(config=config)
    allocator = PortfolioAllocator(config=config)

    # Run backtest
    backtest_engine = BacktestEngine(config=config)
    results = backtest_engine.run(
        prices=close_prices,
        signal_engine=signal_engine,
        risk_manager=risk_manager,
        allocator=allocator,
        rebalance_freq=config.rebalance_freq,
    )

    # Calculate and print metrics
    calculate_all_metrics = modules.get("calculate_all_metrics")
    if calculate_all_metrics and not results.empty:
        strategy_returns = results["strategy_return"].dropna()
        benchmark_returns = results["benchmark_return"].dropna() if "benchmark_return" in results.columns else None

        metrics = calculate_all_metrics(strategy_returns, benchmark=benchmark_returns)

        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Period: {config.start_date} to {config.end_date}")
        print(f"Total days: {len(results)}")
        print(f"Final portfolio value: ${results['cumulative_value'].iloc[-1]:,.2f}")
        print("\n--- Performance Metrics ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ("max_drawdown", "volatility", "var_95", "cvar_95"):
                    print(f"  {key}: {value * 100:.2f}%")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Save results
        reports_dir = Path(getattr(config, "reports_dir", "./reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)
        results_file = reports_dir / f"backtest_results_{config.start_date}_{config.end_date}.csv"
        results.to_csv(results_file)
        print(f"\nResults saved to: {results_file}")

    return 0


def cmd_paper(args: argparse.Namespace, modules: Dict[str, Any]) -> int:
    """Run paper trading for today.

    Args:
        args: Parsed command-line arguments.
        modules: Imported QuantLab modules.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting paper trading mode")

    DailyDaemon = modules.get("DailyDaemon")
    if DailyDaemon is None:
        print("ERROR: Cannot import DailyDaemon", file=sys.stderr)
        return 1

    QuantLabConfig = modules.get("QuantLabConfig")
    if QuantLabConfig is None:
        print("ERROR: Cannot import QuantLabConfig", file=sys.stderr)
        return 1

    config = QuantLabConfig()
    daemon = DailyDaemon(config)
    summary = daemon.run_pipeline()

    print("\n" + "=" * 50)
    print("PAPER TRADING RESULTS")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if summary.get("errors"):
        print("\nWARNINGS/ERRORS:")
        for err in summary["errors"]:
            print(f"  - {err}")

    return 0 if not summary.get("errors") else 1


def cmd_daemon(args: argparse.Namespace, modules: Dict[str, Any]) -> int:
    """Run daily pipeline once (V1.0 — no actual scheduling).

    Args:
        args: Parsed command-line arguments.
        modules: Imported QuantLab modules.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting daemon mode (single run)")
    # In V1.0, daemon mode is identical to paper mode
    return cmd_paper(args, modules)


def cmd_report(args: argparse.Namespace, modules: Dict[str, Any]) -> int:
    """Generate latest HTML report.

    Args:
        args: Parsed command-line arguments.
        modules: Imported QuantLab modules.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting report generation mode")

    QuantLabConfig = modules.get("QuantLabConfig")
    PortfolioRepository = modules.get("PortfolioRepository")
    generate_daily_report = modules.get("generate_daily_report")

    if any(m is None for m in [QuantLabConfig, PortfolioRepository, generate_daily_report]):
        print("ERROR: Required modules not available for report generation", file=sys.stderr)
        return 1

    config = QuantLabConfig()
    repository = PortfolioRepository(db_path=config.db_path)

    # Load latest allocation
    allocation = repository.get_latest_allocation()
    if allocation is None:
        print("ERROR: No allocation found. Run paper/daemon mode first.", file=sys.stderr)
        return 1

    # Load signals
    try:
        signals_df = repository.load_signals(date=allocation.get("date"))
        signals = signals_df.to_dict("records") if not signals_df.empty else []
    except Exception as exc:
        logger.warning("Could not load signals: %s", exc)
        signals = []

    # Load features from data store
    FeatureEngine = modules.get("FeatureEngine")
    features = pd.DataFrame()
    try:
        DataStore = modules.get("DataStore")
        if DataStore is not None:
            data_store = DataStore(db_path=config.db_path)
            raw_data = data_store.load_ohlcv()
            if not raw_data.empty and FeatureEngine is not None:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    close_prices = raw_data.xs("Close", level=1, axis=1)
                else:
                    close_prices = raw_data.copy()
                feature_engine = FeatureEngine(prices=close_prices)
                features = feature_engine.compute_all()
    except Exception as exc:
        logger.warning("Could not compute features for report: %s", exc)

    # Get portfolio state
    portfolio_state = repository.get_portfolio_state()

    # Get metrics
    metrics: Dict[str, Any] = {}
    calculate_all_metrics = modules.get("calculate_all_metrics")
    if calculate_all_metrics:
        try:
            from quantlab.execution.paper import PaperTrader
            trader = PaperTrader(config=config, repository=repository)
            equity_curve = trader.get_equity_curve()
            if len(equity_curve) > 1:
                returns = equity_curve.pct_change().dropna()
                metrics = calculate_all_metrics(returns)
        except Exception as exc:
            logger.warning("Could not calculate metrics: %s", exc)

    # Generate report
    reports_dir = Path(getattr(config, "reports_dir", "./reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"daily_report_{allocation.get('date', 'latest')}.html"

    report_file = generate_daily_report(
        allocation=allocation,
        signals=signals,
        features=features,
        portfolio_state=portfolio_state,
        metrics=metrics,
        output_path=report_path,
    )

    print(f"\nReport generated: {report_file}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="quantlab",
        description="QuantLab — quantitative trading research platform (v%s)" % _VERSION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quantlab.main backtest    # Run walk-forward backtest
  python -m quantlab.main paper       # Run paper trading for today
  python -m quantlab.main daemon      # Run daily pipeline once
  python -m quantlab.main report      # Generate latest HTML report

Safety note: QuantLab V1.0 operates in paper mode only. No live trading.
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run walk-forward backtest",
        description="Run a walk-forward backtest over historical data.",
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD). Overrides config.",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD). Overrides config.",
    )
    backtest_parser.add_argument(
        "--rebalance-freq",
        type=str,
        default="W",
        choices=["D", "W"],
        help="Rebalancing frequency: D=daily, W=weekly (default: W)",
    )

    # paper command
    paper_parser = subparsers.add_parser(
        "paper",
        help="Run paper trading for today",
        description="Run the daily pipeline once in paper-trading mode.",
    )
    paper_parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save HTML report after execution (default: True)",
    )

    # daemon command
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Run daily pipeline once (no scheduling in V1.0)",
        description=(
            "Run the daily pipeline once. In V1.0 this is identical to "
            "paper mode. Full scheduling will be added in V1.5."
        ),
    )

    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate latest HTML report",
        description="Generate an HTML report from the latest saved allocation and signals.",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for the report.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Import all modules
    modules = _import_quantlab_modules()

    # Route to command handler
    command_handlers = {
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "daemon": cmd_daemon,
        "report": cmd_report,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, modules)


if __name__ == "__main__":
    sys.exit(main())
