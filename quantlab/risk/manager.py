"""Risk management layer for QuantLab.

Applies guardrails to signals before they reach the portfolio allocator.
All checks are defensive — when a limit is breached the signal is either
capped, scaled down, or converted to the cash fallback ticker.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirrored from config so the module is self-contained
# ---------------------------------------------------------------------------
_MAX_POSITION_SIZE = 0.60
_MAX_LEVERAGED_ETF_EXPOSURE = 0.80
_TARGET_VOLATILITY = 0.15
_MAX_DRAWDOWN_PCT = 0.15
_CASH_FALLBACK_TICKER = "BIL"
_KILL_SWITCH_ENABLED = True

_LEVERAGED_ETFS: Set[str] = {
    "SOXL", "SOXS", "TECL", "TECS", "UPRO", "SPXU",
    "TNA", "TZA", "ERX", "ERY", "FAS", "FAZ",
}

# Number of trading days in a year for annualisation
_TRADING_DAYS_PER_YEAR = 252


class RiskManager:
    """Risk management layer.

    Applies the following guardrails (in order) via :meth:`check`:

    1. **Kill switch** — if enabled and extreme conditions are detected,
       all trading is halted.
    2. **No-trade conditions** — if market conditions are too extreme,
       convert the signal to 100% cash.
    3. **Max position** — no single ticker may exceed ``MAX_POSITION_SIZE``.
    4. **Leveraged ETF exposure** — total weight in leveraged ETFs must
       stay below ``MAX_LEVERAGED_ETF_EXPOSURE``.
    5. **Volatility target** — estimated portfolio vol is scaled to
       ``TARGET_VOLATILITY``.
    6. **Drawdown guardrail** — if portfolio drawdown exceeds
       ``MAX_DRAWDOWN_PCT``, convert to cash.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the risk manager.

        Args:
            config: ``QuantLabConfig`` instance (or *None* for defaults).
        """
        self.config = config
        self.max_position_size: float = getattr(
            config, "max_position_size", _MAX_POSITION_SIZE
        )
        self.max_leveraged_exposure: float = getattr(
            config, "max_leveraged_exposure", _MAX_LEVERAGED_ETF_EXPOSURE
        )
        self.target_volatility: float = getattr(
            config, "target_volatility", _TARGET_VOLATILITY
        )
        self.max_drawdown_pct: float = getattr(
            config, "max_drawdown_pct", _MAX_DRAWDOWN_PCT
        )
        self.cash_fallback: str = getattr(
            config, "cash_fallback_ticker", _CASH_FALLBACK_TICKER
        )
        self.kill_switch_enabled: bool = getattr(
            config, "kill_switch_enabled", _KILL_SWITCH_ENABLED
        )

        logger.info(
            "RiskManager initialised — max_pos=%.2f max_lev=%.2f target_vol=%.2f "
            "max_dd=%.2f kill=%s",
            self.max_position_size,
            self.max_leveraged_exposure,
            self.target_volatility,
            self.max_drawdown_pct,
            self.kill_switch_enabled,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        signal: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply all risk checks to a signal.

        The checks are applied sequentially — each step may mutate the
        working signal.  The method never raises; on unexpected errors
        it falls back to the cash position.

        Args:
            signal: Normalised signal dictionary (see ``SignalEngine``).
            portfolio_state: Optional dict with at least:
                ``equity``, ``peak_equity``, ``drawdown_pct``.

        Returns:
            Dict with keys:
            - ``approved`` (*bool*)
            - ``modified_signal`` (*dict*) — the post-risk signal
            - ``reasons`` (*list[str]*) — log of every decision
            - ``kill_switch`` (*bool*)
        """
        import copy

        result: Dict[str, Any] = {
            "approved": False,
            "modified_signal": copy.deepcopy(signal),
            "reasons": [],
            "kill_switch": False,
        }

        try:
            self._apply_checks(result, signal, portfolio_state)
        except Exception as exc:
            logger.error(
                "Unexpected error in RiskManager.check: %s — falling back to cash",
                exc,
                exc_info=True,
            )
            fallback = self.apply_cash_fallback(copy.deepcopy(signal))
            result["modified_signal"] = fallback
            result["approved"] = True
            result["reasons"].append(
                f"Exception during risk check ({type(exc).__name__}) — cash fallback enforced"
            )

        return result

    def _apply_checks(
        self,
        result: Dict[str, Any],
        original_signal: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]],
    ) -> None:
        """Internal sequential check pipeline."""
        import copy

        working = copy.deepcopy(original_signal)
        reasons: List[str] = []

        # ---- 1. Kill switch ------------------------------------------------
        if self.kill_switch_enabled:
            kill_active, kill_reason = self._evaluate_kill_switch(working, portfolio_state)
            if kill_active:
                result["kill_switch"] = True
                result["approved"] = False
                fallback = self.apply_cash_fallback(working)
                result["modified_signal"] = fallback
                reasons.append(f"KILL SWITCH ACTIVE: {kill_reason}")
                result["reasons"] = reasons
                logger.critical("Kill switch triggered — all trading halted")
                return

        # ---- 2. No-trade conditions ----------------------------------------
        features = working.get("_features")
        if features is not None and isinstance(features, pd.DataFrame):
            if self.no_trade_conditions(features):
                reasons.append("No-trade conditions detected — converting to cash fallback")
                working = self.apply_cash_fallback(working)
                # Skip remaining checks — we are already in cash
                result["approved"] = True
                result["modified_signal"] = working
                result["reasons"] = reasons
                return

        # ---- 3. Max position check -----------------------------------------
        weights: Dict[str, float] = dict(working.get("weights", {}))
        capped_weights, pos_reasons = self.check_max_position(weights)
        reasons.extend(pos_reasons)
        working["weights"] = capped_weights

        # ---- 4. Leveraged ETF exposure check -------------------------------
        lev_weights, lev_reasons = self.check_leveraged_exposure(working["weights"])
        reasons.extend(lev_reasons)
        working["weights"] = lev_weights

        # ---- 5. Volatility target check ------------------------------------
        if features is not None and isinstance(features, pd.DataFrame):
            vol_weights, vol_reasons = self.check_volatility_target(
                features, working["weights"]
            )
            reasons.extend(vol_reasons)
            working["weights"] = vol_weights

        # ---- 6. Drawdown guardrail -----------------------------------------
        if portfolio_state is not None:
            dd_weights, dd_reasons = self.check_drawdown_guardrail(
                portfolio_state, working["weights"]
            )
            if dd_reasons and "fallback" in dd_reasons[-1].lower():
                # Drawdown triggered cash fallback
                working = self.apply_cash_fallback(working)
                reasons.extend(dd_reasons)
                result["approved"] = True
                result["modified_signal"] = working
                result["reasons"] = reasons
                return
            reasons.extend(dd_reasons)
            working["weights"] = dd_weights

        # ---- Final approval ------------------------------------------------
        # Ensure weights still sum to 1.0 after all modifications
        working["weights"] = self._renormalise(working["weights"])

        if not reasons:
            reasons.append("All risk checks passed — signal unchanged")

        result["approved"] = True
        result["modified_signal"] = working
        result["reasons"] = reasons

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_max_position(self, weights: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """Ensure no single position exceeds ``MAX_POSITION_SIZE``.

        When a ticker breaches the limit its weight is capped and the
        excess is distributed proportionally across all other positions.

        Args:
            weights: Ticker → weight mapping.

        Returns:
            Tuple of (capped_weights, reasons).
        """
        reasons: List[str] = []
        if not weights:
            return {self.cash_fallback: 1.0}, reasons

        capped = dict(weights)
        excess_total = 0.0
        capped_tickers: List[str] = []

        for ticker, w in capped.items():
            if w > self.max_position_size:
                excess = w - self.max_position_size
                excess_total += excess
                capped[ticker] = self.max_position_size
                capped_tickers.append(f"{ticker}({w:.3f})")

        if capped_tickers:
            reasons.append(
                f"Max position cap applied to: {', '.join(capped_tickers)} "
                f"-> capped at {self.max_position_size:.2f}"
            )
            # Redistribute excess proportionally to uncapped positions
            uncapped = {t: w for t, w in capped.items() if w < self.max_position_size}
            if uncapped:
                uncapped_total = sum(uncapped.values())
                for t in uncapped:
                    if uncapped_total > 0:
                        capped[t] += excess_total * (uncapped[t] / uncapped_total)

        return self._renormalise(capped), reasons

    def check_leveraged_exposure(
        self, weights: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Ensure total leveraged ETF exposure is within limit.

        Leveraged ETFs: SOXL, SOXS, TECL, TECS, UPRO, SPXU, TNA, TZA,
        ERX, ERY, FAS, FAZ.

        If total exposure exceeds the limit, all leveraged positions are
        scaled down proportionally and the freed capital is moved to the
        cash fallback.

        Args:
            weights: Ticker → weight mapping.

        Returns:
            Tuple of (adjusted_weights, reasons).
        """
        reasons: List[str] = []
        if not weights:
            return {self.cash_fallback: 1.0}, reasons

        lev_total = sum(w for t, w in weights.items() if self.is_leveraged_etf(t))

        if lev_total <= self.max_leveraged_exposure + 1e-9:
            return dict(weights), reasons

        reasons.append(
            f"Leveraged ETF exposure {lev_total:.3f} exceeds limit "
            f"{self.max_leveraged_exposure:.2f} — scaling down"
        )

        adjusted = dict(weights)
        scale = self.max_leveraged_exposure / lev_total if lev_total > 0 else 0.0
        freed = 0.0

        for t in adjusted:
            if self.is_leveraged_etf(t):
                old_w = adjusted[t]
                adjusted[t] = old_w * scale
                freed += old_w - adjusted[t]

        # Add freed capital to cash fallback
        if freed > 0:
            adjusted[self.cash_fallback] = adjusted.get(self.cash_fallback, 0.0) + freed

        return self._renormalise(adjusted), reasons

    def check_volatility_target(
        self,
        features: pd.DataFrame,
        weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str]]:
        """Estimate portfolio volatility and scale weights if above target.

        Portfolio vol is approximated as the weighted sum of individual
        annualised volatilities (conservative upper bound).  When the
        estimate exceeds ``TARGET_VOLATILITY`` all risky positions are
        scaled down and the remainder is allocated to cash.

        Args:
            features: Feature DataFrame.  Must contain a ``vol`` or
                ``volatility`` column level, or columns ending in
                ``_vol`` / ``_volatility``.
            weights: Ticker → weight mapping.

        Returns:
            Tuple of (scaled_weights, reasons).
        """
        reasons: List[str] = []
        if not weights or features is None or features.empty:
            return dict(weights), reasons

        # Try to extract per-ticker volatilities
        vols = self._extract_volatilities(features, list(weights.keys()))
        if vols is None:
            return dict(weights), reasons

        # Portfolio vol as weighted sum (upper-bound approximation)
        port_vol = sum(weights.get(t, 0.0) * vols.get(t, 0.0) for t in weights)

        if port_vol <= self.target_volatility + 1e-9:
            return dict(weights), reasons

        # Scale down all non-cash positions
        scale = self.target_volatility / port_vol if port_vol > 0 else 1.0
        reasons.append(
            f"Portfolio vol {port_vol:.3f} exceeds target {self.target_volatility:.2f} "
            f"-> scaling by {scale:.3f}"
        )

        adjusted = dict(weights)
        freed = 0.0
        for t in adjusted:
            if t != self.cash_fallback:
                old_w = adjusted[t]
                adjusted[t] = old_w * scale
                freed += old_w - adjusted[t]

        if freed > 0:
            adjusted[self.cash_fallback] = adjusted.get(self.cash_fallback, 0.0) + freed

        return self._renormalise(adjusted), reasons

    def check_drawdown_guardrail(
        self,
        portfolio_state: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Trigger if portfolio drawdown exceeds maximum.

        When drawdown > ``MAX_DRAWDOWN_PCT`` the signal is rejected and
        converted to 100% cash.

        Args:
            portfolio_state: Dict with ``drawdown_pct`` or ``equity`` +
                ``peak_equity`` keys.
            weights: Current weights (returned unchanged if no breach).

        Returns:
            Tuple of (weights_or_fallback, reasons).
        """
        reasons: List[str] = []
        if weights is None:
            weights = {}

        dd = self._extract_drawdown(portfolio_state)
        if dd is None:
            return dict(weights), reasons

        if dd > self.max_drawdown_pct:
            reasons.append(
                f"Drawdown {dd:.3f} exceeds max {self.max_drawdown_pct:.2f} — "
                f"cash fallback enforced"
            )
            # Return empty weights — caller will apply_cash_fallback
            return {}, reasons

        return dict(weights), reasons

    def apply_cash_fallback(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert signal to 100% cash fallback ticker.

        Args:
            signal: Signal dictionary to mutate.

        Returns:
            Deep-copied signal with ``weights`` set to 100% cash.
        """
        import copy

        out = copy.deepcopy(signal)
        out["weights"] = {self.cash_fallback: 1.0}
        out["confidence"] = 0.0
        out["risk_state"] = "extreme"
        out["reason"] = f"Risk fallback — 100% {self.cash_fallback} (cash)"
        return out

    def no_trade_conditions(self, features: pd.DataFrame) -> bool:
        """Check if market conditions prohibit trading.

        Conditions treated as no-trade:
        - All features are NaN (missing data).
        - More than 50% of recent vol values are NaN.
        - Any single ticker shows a one-day return magnitude > 25%
          (extreme gap / halt).

        Args:
            features: Latest feature DataFrame.

        Returns:
            *True* if trading should be suspended.
        """
        if features is None or features.empty:
            logger.warning("no_trade_conditions: empty features — suspending trade")
            return True

        # Check for entirely NaN data
        if features.isna().all().all():
            logger.warning("no_trade_conditions: all values are NaN — suspending trade")
            return True

        # Check for extreme returns in the most recent row
        if not features.empty:
            last_row = features.iloc[-1]
            # Look for return-like columns
            ret_cols = [c for c in features.columns if "return" in str(c).lower()]
            if ret_cols:
                recent_rets = last_row[ret_cols]
                if (recent_rets.abs() > 0.25).any():
                    extreme = recent_rets[recent_rets.abs() > 0.25]
                    logger.warning(
                        "no_trade_conditions: extreme return detected — %s", extreme
                    )
                    return True

        # Check for excessive missing data in the last 5 rows
        recent = features.tail(5)
        if not recent.empty:
            nan_frac = recent.isna().mean().mean()
            if nan_frac > 0.5:
                logger.warning(
                    "no_trade_conditions: %.1f%% NaN in recent features — suspending",
                    nan_frac * 100,
                )
                return True

        return False

    def is_leveraged_etf(self, ticker: str) -> bool:
        """Check if ticker is a leveraged ETF.

        Args:
            ticker: Ticker symbol.

        Returns:
            *True* for known leveraged ETFs.
        """
        return ticker.upper() in _LEVERAGED_ETFS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_kill_switch(
        self,
        signal: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Determine whether the kill switch should fire.

        The kill switch fires when **all** of the following are true:
        - Portfolio drawdown exceeds 20% (well past the normal 15% guardrail).
        - The signal risk_state is ``extreme``.

        Returns:
            (active, reason)
        """
        if not self.kill_switch_enabled:
            return False, ""

        # Check drawdown
        if portfolio_state is not None:
            dd = self._extract_drawdown(portfolio_state)
            if dd is not None and dd > 0.20:
                return True, f"portfolio drawdown {dd:.3f} exceeds 20% kill threshold"

        # Check signal risk state
        if signal.get("risk_state") == "extreme":
            return True, "signal risk_state is 'extreme'"

        return False, ""

    def _extract_drawdown(self, portfolio_state: Dict[str, Any]) -> Optional[float]:
        """Extract drawdown percentage from portfolio_state.

        Looks for ``drawdown_pct`` directly, or computes from
        ``equity`` / ``peak_equity``.
        """
        dd_raw = portfolio_state.get("drawdown_pct")
        if dd_raw is not None:
            try:
                return float(dd_raw)
            except (TypeError, ValueError):
                pass

        equity = portfolio_state.get("equity")
        peak = portfolio_state.get("peak_equity")
        if equity is not None and peak is not None:
            try:
                e, p = float(equity), float(peak)
                if p > 0:
                    return 1.0 - (e / p)
            except (TypeError, ValueError):
                pass

        return None

    def _extract_volatilities(
        self,
        features: pd.DataFrame,
        tickers: List[str],
    ) -> Optional[Dict[str, float]]:
        """Best-effort extraction of per-ticker volatilities from features.

        Tries, in order:
        1. Columns with ``_vol`` suffix.
        2. Columns with ``_volatility`` suffix.
        3. Columns containing ``vol`` in the name.
        4. Computes from daily returns if available.

        Returns:
            Dict mapping ticker → annualised volatility, or *None* if
            no volatilities could be extracted.
        """
        if features is None or features.empty:
            return None

        vols: Dict[str, float] = {}
        last_row = features.iloc[-1]

        for ticker in tickers:
            if ticker == self.cash_fallback:
                vols[ticker] = 0.0
                continue

            val = None
            # Try _vol suffix
            col_candidates = [c for c in features.columns if str(c) == f"{ticker}_vol"]
            if col_candidates:
                val = last_row.get(col_candidates[0])
            # Try _volatility suffix
            if val is None or pd.isna(val):
                col_candidates = [
                    c for c in features.columns if str(c) == f"{ticker}_volatility"
                ]
                if col_candidates:
                    val = last_row.get(col_candidates[0])
            # Try any column containing ticker and 'vol'
            if val is None or pd.isna(val):
                col_candidates = [
                    c
                    for c in features.columns
                    if ticker in str(c) and "vol" in str(c).lower()
                ]
                if col_candidates:
                    val = last_row.get(col_candidates[0])

            if val is not None and not pd.isna(val):
                v = float(val)
                # If vol looks like a daily value, annualise it
                if 0.0 < v < 0.10:
                    v *= np.sqrt(_TRADING_DAYS_PER_YEAR)
                vols[ticker] = v

        return vols if vols else None

    @staticmethod
    def _renormalise(weights: Dict[str, float]) -> Dict[str, float]:
        """Renormalise weights so they sum to 1.0.

        Negative values are clipped to 0.  If all weights are zero
        (or empty) the result is 100% cash.
        """
        if not weights:
            return {_CASH_FALLBACK_TICKER: 1.0}

        clipped = {t: max(0.0, float(w)) for t, w in weights.items()}
        clipped = {t: w for t, w in clipped.items() if w > 1e-12}
        total = sum(clipped.values())
        if total <= 1e-12:
            return {_CASH_FALLBACK_TICKER: 1.0}
        return {t: w / total for t, w in clipped.items()}
