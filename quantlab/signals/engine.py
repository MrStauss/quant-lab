"""Unified signal engine for QuantLab.

Orchestrates all strategies and produces normalized signals
with validated structure and weight normalization.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantlab.config import QuantLabConfig

logger = logging.getLogger(__name__)

_FALLBACK_MAX_POSITION_SIZE = 0.60
_FALLBACK_MAX_LEVERAGED_ETF_EXPOSURE = 0.80


@runtime_checkable
class BaseStrategy(Protocol):
    """Protocol describing a strategy object acceptable to SignalEngine."""

    name: str = "base"
    family: str = ""

    def generate_signal(self, features: pd.DataFrame, prices: pd.DataFrame) -> dict:
        """Generate a signal dict."""
        ...


class SignalEngine:
    """Orchestrates all strategies and produces normalized signals.

    Each registered strategy is run over the same *features* and *prices*
    data.  Returned signals are validated, normalised, and collected.
    """

    def __init__(self, config: Optional["QuantLabConfig"] = None) -> None:
        """Initialise the engine.

        Args:
            config: QuantLabConfig instance.  When *None* sensible
                defaults are used.
        """
        self._strategies: List[BaseStrategy] = []
        self.config = config
        self._max_position: float = getattr(
            config, "max_position_size", _FALLBACK_MAX_POSITION_SIZE
        )
        self._max_lev_exp: float = getattr(
            config, "max_leveraged_exposure", _FALLBACK_MAX_LEVERAGED_ETF_EXPOSURE
        )
        logger.info("SignalEngine initialised (strategies=%d)", len(self._strategies))

    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the engine registry.

        Args:
            strategy: Object implementing the BaseStrategy protocol.

        Raises:
            TypeError: If strategy does not conform to the protocol.
            ValueError: If a strategy with the same name is already registered.
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(
                f"Strategy {strategy!r} does not implement BaseStrategy protocol"
            )
        existing = [s for s in self._strategies if s.name == strategy.name]
        if existing:
            raise ValueError(f"Strategy '{strategy.name}' is already registered")
        self._strategies.append(strategy)
        logger.info("Registered strategy '%s' (family=%s)", strategy.name, strategy.family)

    def run_all(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Run all registered strategies and return list of signals.

        Args:
            features: DataFrame with computed features (index=Date).
            prices: DataFrame with Close prices (columns=tickers, index=Date).

        Returns:
            List of signal dictionaries.
        """
        if not self._strategies:
            logger.warning("No strategies registered — returning empty signal list")
            return []

        signals: List[Dict[str, Any]] = []
        for strategy in self._strategies:
            try:
                raw_signal = strategy.generate_signal(features, prices)
                if raw_signal is None:
                    logger.warning("Strategy '%s' returned None — skipping", strategy.name)
                    continue
                normalized = self.normalize_weights(raw_signal)
                if self.validate_signal(normalized):
                    signals.append(normalized)
                    logger.info(
                        "Strategy '%s' produced valid signal (confidence=%.3f)",
                        strategy.name,
                        normalized.get("confidence", 0.0),
                    )
                else:
                    logger.warning(
                        "Strategy '%s' signal failed validation — skipping",
                        strategy.name,
                    )
            except Exception as exc:
                logger.error(
                    "Strategy '%s' raised %s: %s",
                    strategy.name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                continue

        logger.info(
            "SignalEngine.run_all completed — %d/%d signals valid",
            len(signals),
            len(self._strategies),
        )
        return signals

    def normalize_weights(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure weights sum to 1.0 and clip negative values to 0.

        Operates on a deep copy so the original is never mutated.

        Args:
            signal: Raw signal dictionary.

        Returns:
            New signal dictionary with weights normalised.
        """
        out = copy.deepcopy(signal)
        weights: Dict[str, float] = out.get("weights", {})
        if not weights:
            out["weights"] = {"BIL": 1.0}
            return out

        # Clip negatives to 0
        clipped = {ticker: max(0.0, float(w)) for ticker, w in weights.items()}
        # Remove zeroed tickers
        clipped = {t: w for t, w in clipped.items() if w > 0.0}

        if not clipped:
            out["weights"] = {"BIL": 1.0}
            return out

        # Renormalise to sum exactly 1.0
        total = sum(clipped.values())
        if total > 0.0:
            normalised = {ticker: w / total for ticker, w in clipped.items()}
        else:
            normalised = {"BIL": 1.0}

        out["weights"] = normalised
        return out

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal structure and value ranges.

        Checks:
        - Required top-level keys exist.
        - weights is non-empty, numeric, sums to ~1.0.
        - No individual weight exceeds MAX_POSITION_SIZE.
        - confidence is in [0.0, 1.0].
        - risk_state is one of allowed literals.
        - reason is non-empty string.

        Args:
            signal: Signal dictionary to validate.

        Returns:
            True when the signal passes all checks.
        """
        required_keys = {
            "date", "strategy", "family", "weights",
            "confidence", "risk_state", "reason",
        }
        missing = required_keys - set(signal.keys())
        if missing:
            logger.debug("Signal validation failed — missing keys: %s", missing)
            return False

        # weights checks
        weights = signal.get("weights")
        if not isinstance(weights, dict) or len(weights) == 0:
            logger.debug("Signal validation failed — invalid weights")
            return False

        try:
            weight_values = [float(v) for v in weights.values()]
        except (TypeError, ValueError):
            logger.debug("Signal validation failed — non-numeric weight values")
            return False

        weight_sum = sum(weight_values)
        if abs(weight_sum - 1.0) > 1e-4:
            logger.debug("Signal validation failed — weights sum to %.6f", weight_sum)
            return False

        max_w = max(weight_values)
        if max_w > self._max_position + 1e-6:
            logger.debug("Signal validation failed — max weight %.4f exceeds limit", max_w)
            return False

        # confidence check
        try:
            conf = float(signal.get("confidence", -1))
        except (TypeError, ValueError):
            logger.debug("Signal validation failed — non-numeric confidence")
            return False
        if not (0.0 <= conf <= 1.0):
            logger.debug("Signal validation failed — confidence %.4f out of range", conf)
            return False

        # risk_state check
        if signal.get("risk_state") not in {"low", "medium", "high", "extreme"}:
            logger.debug("Signal validation failed — invalid risk_state")
            return False

        # reason check
        if not isinstance(signal.get("reason"), str) or not signal.get("reason"):
            logger.debug("Signal validation failed — empty reason")
            return False

        return True
