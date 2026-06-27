"""Portfolio allocator for QuantLab.

Combines multiple strategy signals into a single final allocation,
applies risk caps, and produces family rankings.
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
_CASH_FALLBACK_TICKER = "BIL"

_LEVERAGED_ETFS: Set[str] = {
    "SOXL", "SOXS", "TECL", "TECS", "UPRO", "SPXU",
    "TNA", "TZA", "ERX", "ERY", "FAS", "FAZ",
}


class PortfolioAllocator:
    """Combines strategy signals into the final target allocation.

    The allocation pipeline follows these steps:

    1. **Filter** to risk-manager-approved signals only.
    2. **Family aggregation** — for each asset family pick the best
       (highest-confidence) signal.
    3. **Confidence weighting** — blend weight vectors using normalised
       confidences as weights.
    4. **Normalisation** — ensure final weights sum to 1.0.
    5. **Risk caps** — apply position limits and leveraged ETF exposure
       caps (mirroring :class:`RiskManager` logic).
    6. **Family ranking** — sort families by risk-adjusted opportunity.
    7. **Final target** — package everything into the canonical output
       dictionary.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the allocator.

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
        self.cash_fallback: str = getattr(
            config, "cash_fallback_ticker", _CASH_FALLBACK_TICKER
        )

        logger.info(
            "PortfolioAllocator initialised — max_pos=%.2f max_lev=%.2f",
            self.max_position_size,
            self.max_leveraged_exposure,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def allocate(
        self,
        signals: List[Dict[str, Any]],
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Combine multiple strategy signals into a final allocation.

        Steps:
        1. Filter to approved signals only.
        2. For each family, pick the best (highest-confidence) signal.
        3. Weight signals by confidence and blend weight vectors.
        4. Normalise to sum 1.0.
        5. Apply risk caps (position limits + leveraged exposure).
        6. Rank families by risk-adjusted opportunity.
        7. Produce final target.

        Args:
            signals: List of signal dictionaries (post-risk-manager).
            features: Latest feature DataFrame.

        Returns:
            Dict with keys:
            - ``date`` (*str*)
            - ``final_weights`` (*dict*)
            - ``family_rankings`` (*list[dict]*)
            - ``risk_score`` (*str*)
            - ``confidence`` (*float*)
            - ``reason`` (*str*)
        """
        import copy

        # ---- Step 1: filter to approved signals --------------------------
        approved = self._filter_approved(signals)
        if not approved:
            logger.warning("No approved signals — returning 100%% cash allocation")
            today = self._extract_date(signals)
            return self._cash_allocation(today)

        # ---- Step 2: pick best signal per family ------------------------
        family_best = self._pick_best_per_family(approved)

        # ---- Step 3: blend by confidence --------------------------------
        blended_weights = self._blend_by_confidence(family_best)

        # ---- Step 4: normalise ------------------------------------------
        blended_weights = self._renormalise(blended_weights)

        # ---- Step 5: apply risk caps ------------------------------------
        final_weights = self.apply_risk_caps(blended_weights)

        # ---- Step 6: rank families --------------------------------------
        family_rankings = self.rank_families(approved)

        # ---- Step 7: compute aggregate metadata -------------------------
        avg_confidence = np.mean([s.get("confidence", 0.0) for s in approved])
        risk_states = [s.get("risk_state", "medium") for s in approved]
        risk_score = self._aggregate_risk_score(risk_states)

        worst_reasons = [
            s.get("reason", "") for s in approved if s.get("risk_state") in ("high", "extreme")
        ]
        if worst_reasons:
            reason = (
                f"Blended allocation from {len(approved)} signals; "
                f"risk={risk_score}; concerns: {' | '.join(worst_reasons[:3])}"
            )
        else:
            reason = (
                f"Blended allocation from {len(approved)} signals; "
                f"risk={risk_score}; all checks passed"
            )

        today = self._extract_date(signals)

        return {
            "date": today,
            "final_weights": final_weights,
            "family_rankings": family_rankings,
            "risk_score": risk_score,
            "confidence": float(avg_confidence),
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Family ranking
    # ------------------------------------------------------------------

    def rank_families(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank asset families by risk-adjusted opportunity score.

        The score for each family is computed as::

            score = confidence * expected_return_proxy / (1 + risk_penalty)

        where *risk_penalty* maps ``risk_state`` to a numeric weight
        (low=0, medium=0.5, high=1.0, extreme=2.0) and
        *expected_return_proxy* is estimated from the signal's bullish
        tilt (fraction of weight allocated to bull ETFs minus bear ETFs).

        Args:
            signals: List of signal dictionaries.

        Returns:
            Sorted list of family ranking dicts, highest score first.
        """
        from collections import defaultdict

        # Group signals by family
        by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in signals:
            by_family[s.get("family", "Unknown")].append(s)

        rankings: List[Dict[str, Any]] = []
        for family, fam_signals in by_family.items():
            # Aggregate: average confidence, worst risk state
            avg_conf = np.mean([s.get("confidence", 0.0) for s in fam_signals])
            worst_risk = self._worst_risk_state(
                [s.get("risk_state", "medium") for s in fam_signals]
            )
            # Expected return proxy: net bullish tilt
            net_bull = self._net_bullish_tilt(fam_signals)

            risk_penalty = {"low": 0.0, "medium": 0.5, "high": 1.0, "extreme": 2.0}.get(
                worst_risk, 0.5
            )

            score = avg_conf * net_bull / (1.0 + risk_penalty)

            rankings.append(
                {
                    "family": family,
                    "score": float(score),
                    "confidence": float(avg_conf),
                    "risk_state": worst_risk,
                    "net_bullish_tilt": float(net_bull),
                }
            )

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Risk caps
    # ------------------------------------------------------------------

    def apply_risk_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position and leveraged ETF exposure caps.

        Mirrors the logic in :class:`RiskManager` so the allocator can
        independently enforce limits on the blended weights.

        Args:
            weights: Blended ticker → weight mapping.

        Returns:
            Capped and renormalised weights.
        """
        if not weights:
            return {self.cash_fallback: 1.0}

        w = dict(weights)

        # 1. Max position cap
        w = self._cap_max_position(w)

        # 2. Leveraged ETF exposure cap
        w = self._cap_leveraged_exposure(w)

        # 3. Renormalise
        return self._renormalise(w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_approved(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only signals that passed the risk manager.

        A signal is considered approved when it carries an
        ``_risk_result`` dict with ``approved=True``, **or** when no
        ``_risk_result`` key exists (legacy / pre-risk signals).
        """
        approved: List[Dict[str, Any]] = []
        for s in signals:
            rr = s.get("_risk_result")
            if isinstance(rr, dict):
                if rr.get("approved", False):
                    approved.append(s)
            else:
                # No risk result attached — assume approved
                approved.append(s)
        return approved

    @staticmethod
    def _pick_best_per_family(
        signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """For each family, select the signal with highest confidence."""
        from collections import defaultdict

        by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in signals:
            by_family[s.get("family", "Unknown")].append(s)

        best: List[Dict[str, Any]] = []
        for fam_sigs in by_family.values():
            top = max(fam_sigs, key=lambda x: x.get("confidence", 0.0))
            best.append(top)
        return best

    def _blend_by_confidence(
        self, signals: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Blend weight vectors using normalised confidences as weights.

        Args:
            signals: One signal per family (the best of each).

        Returns:
            Merged ticker → weight mapping.
        """
        if not signals:
            return {self.cash_fallback: 1.0}

        if len(signals) == 1:
            return dict(signals[0].get("weights", {self.cash_fallback: 1.0}))

        # Normalise confidences to sum to 1.0
        confidences = np.array([s.get("confidence", 0.0) for s in signals])
        conf_sum = confidences.sum()
        if conf_sum <= 1e-12:
            # All zero confidence — equal weight
            blend_weights = np.ones(len(signals)) / len(signals)
        else:
            blend_weights = confidences / conf_sum

        # Collect all tickers
        all_tickers: Set[str] = set()
        for s in signals:
            all_tickers.update(s.get("weights", {}).keys())

        # Weighted sum
        blended: Dict[str, float] = {t: 0.0 for t in all_tickers}
        for s, bw in zip(signals, blend_weights):
            for ticker, w in s.get("weights", {}).items():
                blended[ticker] = blended.get(ticker, 0.0) + w * bw

        return blended

    def _cap_max_position(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Cap individual positions at ``max_position_size``."""
        if not weights:
            return weights

        capped = dict(weights)
        excess_total = 0.0

        for ticker, w in capped.items():
            if w > self.max_position_size:
                excess = w - self.max_position_size
                excess_total += excess
                capped[ticker] = self.max_position_size

        if excess_total > 0:
            uncapped = {t: w for t, w in capped.items() if w < self.max_position_size}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for t in uncapped:
                    capped[t] += excess_total * (uncapped[t] / uncapped_total)

        return capped

    def _cap_leveraged_exposure(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Cap total leveraged ETF exposure at ``max_leveraged_exposure``."""
        lev_total = sum(w for t, w in weights.items() if t.upper() in _LEVERAGED_ETFS)

        if lev_total <= self.max_leveraged_exposure + 1e-9:
            return weights

        adjusted = dict(weights)
        scale = self.max_leveraged_exposure / lev_total if lev_total > 0 else 0.0
        freed = 0.0

        for t in adjusted:
            if t.upper() in _LEVERAGED_ETFS:
                old_w = adjusted[t]
                adjusted[t] = old_w * scale
                freed += old_w - adjusted[t]

        if freed > 0:
            adjusted[self.cash_fallback] = adjusted.get(self.cash_fallback, 0.0) + freed

        return adjusted

    @staticmethod
    def _renormalise(weights: Dict[str, float]) -> Dict[str, float]:
        """Renormalise weights so they sum to 1.0."""
        if not weights:
            return {_CASH_FALLBACK_TICKER: 1.0}

        clipped = {t: max(0.0, float(w)) for t, w in weights.items()}
        clipped = {t: w for t, w in clipped.items() if w > 1e-12}
        total = sum(clipped.values())
        if total <= 1e-12:
            return {_CASH_FALLBACK_TICKER: 1.0}
        return {t: w / total for t, w in clipped.items()}

    @staticmethod
    def _worst_risk_state(states: List[str]) -> str:
        """Return the most severe risk state from a list."""
        order = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        best = max(states, key=lambda s: order.get(s, 1))
        return best

    @staticmethod
    def _aggregate_risk_score(states: List[str]) -> str:
        """Derive a single risk score from a list of risk states.

        Uses majority vote biased toward the worst state.
        """
        if not states:
            return "medium"

        order = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        # Simple heuristic: if any extreme -> extreme; if >50% high -> high
        if any(s == "extreme" for s in states):
            return "extreme"
        if sum(1 for s in states if s == "high") / len(states) > 0.5:
            return "high"
        if sum(1 for s in states if s in ("medium", "high", "extreme")) / len(states) > 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _net_bullish_tilt(signals: List[Dict[str, Any]]) -> float:
        """Compute average net bullish tilt across signals.

        Bullish tilt = weight in bull ETF - weight in bear ETF.
        Returns the average across all signals in the list.
        """
        if not signals:
            return 0.0

        # Common bull/bear suffixes
        bull_suffixes = (" bull", "_bull", "-bull")
        bear_suffixes = (" bear", "_bear", "-bear")

        tilts: List[float] = []
        for s in signals:
            weights = s.get("weights", {})
            bull_w = 0.0
            bear_w = 0.0
            # Heuristic: tickers ending in common leveraged bull patterns
            for ticker, w in weights.items():
                t = ticker.upper()
                if t in ("SOXL", "TECL", "UPRO", "TNA", "ERX", "FAS"):
                    bull_w += w
                elif t in ("SOXS", "TECS", "SPXU", "TZA", "ERY", "FAZ"):
                    bear_w += w
            tilts.append(bull_w - bear_w)

        return float(np.mean(tilts))

    @staticmethod
    def _extract_date(signals: List[Dict[str, Any]]) -> str:
        """Extract the most common date from a list of signals."""
        if not signals:
            import datetime as _dt
            return _dt.date.today().isoformat()
        dates = [s.get("date", "") for s in signals]
        # Return the most common date
        from collections import Counter
        if dates:
            return Counter(dates).most_common(1)[0][0]
        return ""

    def _cash_allocation(self, date: str) -> Dict[str, Any]:
        """Produce a 100% cash allocation when no signals are available."""
        return {
            "date": date,
            "final_weights": {self.cash_fallback: 1.0},
            "family_rankings": [],
            "risk_score": "extreme",
            "confidence": 0.0,
            "reason": "No approved signals — 100% cash fallback",
        }
