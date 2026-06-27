"""Daily daemon scheduler for QuantLab.

Implements the :class:`DailyDaemon` that orchestrates the full end-to-end
pipeline: data ingestion, validation, feature computation, signal
generation, risk management, portfolio allocation, persistence, and
reporting.

The ``schedule()`` method is intentionally a stub for V1.0 — the daemon
is designed to be invoked manually (or by an external scheduler such as
cron / systemd timers).  Full APScheduler integration is planned for
V1.5.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# QuantLab imports — all guarded so the file is always importable
# ---------------------------------------------------------------------------
try:
    from quantlab.config import QuantLabConfig, ASSET_FAMILIES, REPORTS_DIR
except ImportError as _err:  # pragma: no cover
    QuantLabConfig = None  # type: ignore[misc,assignment]
    ASSET_FAMILIES = {}  # type: ignore[assignment]
    REPORTS_DIR = Path(".")  # type: ignore[assignment]
    logging.getLogger(__name__).warning("Cannot import quantlab.config: %s", _err)

try:
    from quantlab.data.ingestion import download_latest_data
except ImportError as _err:  # pragma: no cover
    download_latest_data = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("Cannot import data.ingestion: %s", _err)

try:
    from quantlab.data.validation import validate_ohlcv
except ImportError as _err:  # pragma: no cover
    validate_ohlcv = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("Cannot import data.validation: %s", _err)

try:
    from quantlab.data.storage import DataStore
except ImportError as _err:  # pragma: no cover
    DataStore = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import data.storage: %s", _err)

try:
    from quantlab.features.engine import FeatureEngine
except ImportError as _err:  # pragma: no cover
    FeatureEngine = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import features.engine: %s", _err)

try:
    from quantlab.signals.engine import SignalEngine
except ImportError as _err:  # pragma: no cover
    SignalEngine = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import signals.engine: %s", _err)

try:
    from quantlab.risk.manager import RiskManager
except ImportError as _err:  # pragma: no cover
    RiskManager = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import risk.manager: %s", _err)

try:
    from quantlab.portfolio.allocator import PortfolioAllocator
except ImportError as _err:  # pragma: no cover
    PortfolioAllocator = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import portfolio.allocator: %s", _err)

try:
    from quantlab.execution.paper import PaperTrader
except ImportError as _err:  # pragma: no cover
    PaperTrader = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import execution.paper: %s", _err)

try:
    from quantlab.storage.repository import PortfolioRepository
except ImportError as _err:  # pragma: no cover
    PortfolioRepository = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import storage.repository: %s", _err)

try:
    from quantlab.strategies.regime import HMMRegimeStrategy
except ImportError as _err:  # pragma: no cover
    HMMRegimeStrategy = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import strategies.regime: %s", _err)

try:
    from quantlab.strategies.dynamic import DynamicCovarianceStrategy
except ImportError as _err:  # pragma: no cover
    DynamicCovarianceStrategy = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import strategies.dynamic: %s", _err)

try:
    from quantlab.strategies.composer import ComposerRSIStrategy
except ImportError as _err:  # pragma: no cover
    ComposerRSIStrategy = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import strategies.composer: %s", _err)

try:
    from quantlab.strategies.mean_reversion import MeanReversionStrategy
except ImportError as _err:  # pragma: no cover
    MeanReversionStrategy = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import strategies.mean_reversion: %s", _err)

try:
    from quantlab.strategies.combined import CombinedStrategy
except ImportError as _err:  # pragma: no cover
    CombinedStrategy = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).warning("Cannot import strategies.combined: %s", _err)

try:
    from quantlab.utils.reporting import generate_daily_report
except ImportError as _err:  # pragma: no cover
    generate_daily_report = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("Cannot import utils.reporting: %s", _err)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VERSION = "1.0.0"


class DailyDaemon:
    """Daily after-close execution pipeline for QuantLab.

    The daemon ties together every major subsystem and runs them in the
    correct order.  It is fully self-contained — constructing a
    ``DailyDaemon`` with a :class:`QuantLabConfig` is sufficient to run
    the pipeline.

    Example::

        config = QuantLabConfig()
        daemon = DailyDaemon(config)
        summary = daemon.run_pipeline()
        print(summary)

    .. note::
        The ``schedule()`` method is a **stub** for V1.0.  Use
        :meth:`run_pipeline` directly (e.g. via cron) until V1.5 adds
        built-in APScheduler support.
    """

    def __init__(self, config: QuantLabConfig) -> None:  # type: ignore[valid-type]
        """Initialise the daemon and all subsystems.

        Args:
            config: Runtime configuration container.
        """
        self.config = config
        self._version: str = _VERSION

        # Initialise data layer
        self.data_store: DataStore = DataStore(db_path=config.db_path)  # type: ignore[call-arg]

        # Initialise feature engine placeholder
        self.feature_engine: Optional[FeatureEngine] = None  # type: ignore[valid-type]

        # Initialise signal engine
        self.signal_engine: SignalEngine = SignalEngine(config=config)  # type: ignore[call-arg]

        # Initialise risk manager
        self.risk_manager: RiskManager = RiskManager(config=config)  # type: ignore[call-arg]

        # Initialise portfolio allocator
        self.allocator: PortfolioAllocator = PortfolioAllocator(config=config)  # type: ignore[call-arg]

        # Initialise paper trader
        self.repository: PortfolioRepository = PortfolioRepository(db_path=config.db_path)  # type: ignore[call-arg]
        self.paper_trader: PaperTrader = PaperTrader(config=config, repository=self.repository)  # type: ignore[call-arg]

        # Register strategies
        self._register_strategies()

        logger.info(
            "DailyDaemon v%s initialised — %d families, %d strategies registered",
            self._version,
            len(config.asset_families),
            len(self.signal_engine._strategies),
        )

    # ------------------------------------------------------------------
    # Strategy registration
    # ------------------------------------------------------------------

    def _register_strategies(self) -> None:
        """Register all strategies for each asset family."""
        families = self.config.asset_families

        for family_name, assets in families.items():
            # HMM Regime Strategy
            if HMMRegimeStrategy is not None:
                try:
                    hmm = HMMRegimeStrategy(family_name=family_name, assets=assets)
                    self.signal_engine.register_strategy(hmm)
                    logger.debug("Registered HMMRegimeStrategy for %s", family_name)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to register HMMRegimeStrategy for %s: %s", family_name, exc)

            # Dynamic Covariance Strategy
            if DynamicCovarianceStrategy is not None:
                try:
                    dyn = DynamicCovarianceStrategy(family_name=family_name, assets=assets)
                    self.signal_engine.register_strategy(dyn)
                    logger.debug("Registered DynamicCovarianceStrategy for %s", family_name)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to register DynamicCovarianceStrategy for %s: %s", family_name, exc)

            # Combined Strategy (ensemble)
            if CombinedStrategy is not None:
                try:
                    combined = CombinedStrategy(family_name=family_name, assets=assets)
                    self.signal_engine.register_strategy(combined)
                    logger.debug("Registered CombinedStrategy for %s", family_name)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to register CombinedStrategy for %s: %s", family_name, exc)

        # Global strategies (not family-specific)
        if ComposerRSIStrategy is not None:
            try:
                composer = ComposerRSIStrategy()
                self.signal_engine.register_strategy(composer)
                logger.debug("Registered ComposerRSIStrategy")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to register ComposerRSIStrategy: %s", exc)

        if MeanReversionStrategy is not None:
            try:
                mr = MeanReversionStrategy()
                self.signal_engine.register_strategy(mr)
                logger.debug("Registered MeanReversionStrategy")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to register MeanReversionStrategy: %s", exc)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the full daily pipeline.

        Steps:
            1. Download latest market data.
            2. Validate data quality.
            3. Store data in the database.
            4. Compute features.
            5. Run all registered strategies.
            6. Apply risk checks to each signal.
            7. Allocate portfolio.
            8. Save recommendation.
            9. Generate report.

        Returns:
            Summary dictionary with pipeline results, including:

            - ``date`` — ISO date string of the pipeline run.
            - ``data_rows`` — number of OHLCV rows downloaded.
            - ``validation_passed`` — whether data validation succeeded.
            - ``n_signals`` — number of signals generated.
            - ``n_approved`` — number of signals approved by risk manager.
            - ``allocation`` — final portfolio allocation dict.
            - ``report_path`` — path to the generated HTML report.
            - ``errors`` — list of any errors encountered.
            - ``duration_sec`` — pipeline execution time in seconds.
        """
        start_time = datetime.now(timezone.utc)
        errors: List[str] = []
        summary: Dict[str, Any] = {
            "date": start_time.strftime("%Y-%m-%d"),
            "timestamp": start_time.isoformat(),
            "version": self._version,
            "data_rows": 0,
            "validation_passed": False,
            "n_signals": 0,
            "n_approved": 0,
            "allocation": {},
            "report_path": None,
            "errors": errors,
            "duration_sec": 0.0,
        }

        try:
            # ----------------------------------------------------------
            # 1. Download latest data
            # ----------------------------------------------------------
            logger.info("Step 1/9 — Downloading latest data …")
            if download_latest_data is None:
                raise RuntimeError("download_latest_data is not available")

            raw_data: pd.DataFrame = download_latest_data(self.config)
            summary["data_rows"] = len(raw_data)
            logger.info("Downloaded %d rows of OHLCV data", summary["data_rows"])

            # ----------------------------------------------------------
            # 2. Validate data
            # ----------------------------------------------------------
            logger.info("Step 2/9 — Validating data …")
            if validate_ohlcv is None:
                raise RuntimeError("validate_ohlcv is not available")

            validation_report = validate_ohlcv(raw_data)
            summary["validation_passed"] = validation_report.is_valid

            if not validation_report.is_valid:
                for msg in validation_report.messages:
                    logger.warning("Validation issue: %s", msg)
                    errors.append(f"validation: {msg}")

                # Proceed with caution if only warnings
                if validation_report.missing_dates or validation_report.bad_returns:
                    logger.error("Critical validation failures — aborting pipeline")
                    summary["errors"] = errors
                    return summary

            # ----------------------------------------------------------
            # 3. Store data
            # ----------------------------------------------------------
            logger.info("Step 3/9 — Storing data …")
            self.data_store.save_ohlcv(raw_data)
            logger.info("Data stored successfully")

            # ----------------------------------------------------------
            # 4. Compute features
            # ----------------------------------------------------------
            logger.info("Step 4/9 — Computing features …")
            if FeatureEngine is None:
                raise RuntimeError("FeatureEngine is not available")

            # Extract Close prices for feature computation
            close_prices = self._extract_close_prices(raw_data)
            self.feature_engine = FeatureEngine(prices=close_prices)
            features = self.feature_engine.compute_all()
            logger.info("Computed features with shape %s", features.shape)

            # ----------------------------------------------------------
            # 5. Run strategies
            # ----------------------------------------------------------
            logger.info("Step 5/9 — Running strategies …")
            if SignalEngine is None:
                raise RuntimeError("SignalEngine is not available")

            signals = self.signal_engine.run_all(features=features, prices=close_prices)
            summary["n_signals"] = len(signals)
            logger.info("Generated %d signals from %d strategies", summary["n_signals"], len(self.signal_engine._strategies))

            # ----------------------------------------------------------
            # 6. Apply risk checks
            # ----------------------------------------------------------
            logger.info("Step 6/9 — Applying risk checks …")
            if RiskManager is None:
                raise RuntimeError("RiskManager is not available")

            approved_signals: List[Dict[str, Any]] = []
            portfolio_state = self.repository.get_portfolio_state()

            for signal in signals:
                risk_result = self.risk_manager.check(signal, portfolio_state=portfolio_state)
                if risk_result.get("approved", False):
                    modified = risk_result.get("modified_signal", signal)
                    approved_signals.append(modified)
                    logger.debug("Signal approved: %s/%s", signal.get("strategy"), signal.get("family"))
                else:
                    logger.info("Signal rejected: %s/%s — %s", signal.get("strategy"), signal.get("family"), risk_result.get("reasons", []))

                # Log kill switch trigger
                if risk_result.get("kill_switch", False):
                    logger.critical("KILL SWITCH ACTIVATED — halting trading")
                    errors.append("kill_switch_activated")
                    summary["errors"] = errors
                    return summary

            summary["n_approved"] = len(approved_signals)
            logger.info("Approved %d of %d signals", summary["n_approved"], summary["n_signals"])

            # ----------------------------------------------------------
            # 7. Allocate portfolio
            # ----------------------------------------------------------
            logger.info("Step 7/9 — Allocating portfolio …")
            if PortfolioAllocator is None:
                raise RuntimeError("PortfolioAllocator is not available")

            allocation = self.allocator.allocate(
                signals=approved_signals if approved_signals else signals,
                features=features,
            )
            summary["allocation"] = allocation
            logger.info("Portfolio allocated — risk_score=%s confidence=%.2f", allocation.get("risk_score"), allocation.get("confidence", 0))

            # ----------------------------------------------------------
            # 8. Save recommendation
            # ----------------------------------------------------------
            logger.info("Step 8/9 — Saving recommendation …")
            if PortfolioRepository is not None:
                self.repository.save_allocation(allocation)
                for signal in approved_signals:
                    self.repository.save_signal(signal)
                logger.info("Allocation and signals saved")

            # ----------------------------------------------------------
            # 9. Generate report
            # ----------------------------------------------------------
            logger.info("Step 9/9 — Generating report …")
            if generate_daily_report is not None:
                reports_dir = Path(getattr(self.config, "reports_dir", REPORTS_DIR))
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_filename = f"daily_report_{summary['date']}.html"
                report_path = reports_dir / report_filename

                # Gather metrics if available
                try:
                    from quantlab.utils.metrics import calculate_all_metrics

                    equity_curve = self.paper_trader.get_equity_curve()
                    if len(equity_curve) > 1:
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_all_metrics(returns)
                    else:
                        metrics = {}
                except Exception as exc:
                    logger.warning("Could not compute metrics: %s", exc)
                    metrics = {}

                portfolio_state = self.repository.get_portfolio_state()

                generate_daily_report(
                    allocation=allocation,
                    signals=signals,
                    features=features,
                    portfolio_state=portfolio_state,
                    metrics=metrics,
                    output_path=report_path,
                )
                summary["report_path"] = str(report_path)
                logger.info("Report saved to %s", report_path)
            else:
                logger.warning("Report generation not available")

        except Exception as exc:  # pragma: no cover
            logger.exception("Pipeline failed: %s", exc)
            errors.append(str(exc))
            summary["errors"] = errors

        finally:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            summary["duration_sec"] = round(elapsed, 2)
            logger.info("Pipeline completed in %.2f seconds", elapsed)

        return summary

    # ------------------------------------------------------------------
    # Scheduler stub (V1.5)
    # ------------------------------------------------------------------

    def schedule(self) -> None:
        """Schedule daily run at 4:15 PM ET.

        .. note::
            This is a **stub** for V1.0.  Full APScheduler integration
            is planned for V1.5.  Until then use ``run_pipeline()``
            directly (e.g. via cron or systemd timers).
        """
        print("Scheduler not implemented in V1.0. Use run_pipeline() for manual execution.")
        logger.info("schedule() called — scheduler is stub in V1.0")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_close_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
        """Extract Close prices from a MultiIndex OHLCV DataFrame.

        Args:
            raw_data: DataFrame with MultiIndex columns ``(ticker, field)``.

        Returns:
            DataFrame with columns=tickers and Date index containing Close
            prices only.
        """
        if isinstance(raw_data.columns, pd.MultiIndex):
            if "Close" in raw_data.columns.get_level_values(1):
                close = raw_data.xs("Close", level=1, axis=1)
            else:
                close = raw_data.copy()
        else:
            close = raw_data.copy()

        return close.sort_index()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    """Quick CLI entry-point to run the daemon pipeline once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if QuantLabConfig is None:
        print("ERROR: Cannot import QuantLabConfig", file=sys.stderr)
        sys.exit(1)

    config = QuantLabConfig()
    daemon = DailyDaemon(config)
    summary = daemon.run_pipeline()

    print("\n=== Pipeline Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
