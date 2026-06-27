"""Daily report generation for QuantLab.

Provides :func:`generate_daily_report` for rich HTML reports and
:func:`generate_markdown_report` for quick text-based summaries.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# HTML template (Jinja2)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLab Daily Report - {{ date }}</title>
    <style>
        :root {
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #212529;
            --muted-color: #6c757d;
            --border-color: #dee2e6;
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --extreme: #7c3aed;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        .header h1 { font-size: 1.8rem; margin-bottom: 8px; }
        .header .meta { opacity: 0.9; font-size: 0.9rem; }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--primary);
            color: var(--primary);
        }
        .card h3 {
            font-size: 1rem;
            margin: 16px 0 8px;
            color: var(--text-color);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 0.9rem;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: #f1f5f9;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        tr:hover { background: #f8fafc; }
        .risk-low { color: var(--success); font-weight: bold; }
        .risk-medium { color: var(--warning); font-weight: bold; }
        .risk-high { color: var(--danger); font-weight: bold; }
        .risk-extreme { color: var(--extreme); font-weight: bold; }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-low { background: #dcfce7; color: var(--success); }
        .badge-medium { background: #fef3c7; color: var(--warning); }
        .badge-high { background: #fee2e2; color: var(--danger); }
        .badge-extreme { background: #ede9fe; color: var(--extreme); }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
        }
        .chart-container {
            text-align: center;
            margin: 16px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        .metric-box {
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }
        .metric-label {
            font-size: 0.8rem;
            color: var(--muted-color);
            margin-top: 4px;
        }
        .footer {
            text-align: center;
            color: var(--muted-color);
            font-size: 0.85rem;
            margin-top: 30px;
            padding: 20px;
        }
        .allocation-list {
            list-style: none;
        }
        .allocation-list li {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .allocation-list .ticker {
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }
        .allocation-list .weight {
            color: var(--primary);
            font-weight: 600;
        }
        .reason-box {
            background: #f0f9ff;
            border-left: 4px solid var(--primary);
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>QuantLab Daily Report</h1>
            <div class="meta">
                Date: {{ date }} &nbsp;|&nbsp;
                Version: {{ version }} &nbsp;|&nbsp;
                Generated: {{ timestamp }}
            </div>
        </div>

        <!-- Risk Score Banner -->
        <div class="card" style="border-left: 6px solid {{ risk_color }};">
            <h2>Overall Risk Assessment</h2>
            <div style="display: flex; align-items: center; gap: 16px;">
                <span class="badge badge-{{ risk_score }}">
                    {{ risk_score | upper }}
                </span>
                <span style="font-size: 1.1rem;">
                    Confidence: <strong>{{ confidence_pct }}%</strong>
                    &nbsp;|&nbsp;
                    Top Family: <strong>{{ top_family }}</strong>
                </span>
            </div>
        </div>

        <!-- Current Market State -->
        <div class="card">
            <h2>Current Market State</h2>
            {% if market_state %}
            <div class="metric-grid">
                {% for key, value in market_state.items() %}
                <div class="metric-box">
                    <div class="metric-value">{{ value }}</div>
                    <div class="metric-label">{{ key }}</div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="color: var(--muted-color);">No market state data available.</p>
            {% endif %}
        </div>

        <div class="grid-2">
            <!-- Suggested Allocation -->
            <div class="card">
                <h2>Suggested Allocation</h2>
                {% if allocation_chart %}
                <div class="chart-container">
                    <img src="{{ allocation_chart }}" alt="Allocation Pie Chart">
                </div>
                {% endif %}
                <ul class="allocation-list">
                    {% for ticker, weight in final_weights.items() %}
                    <li>
                        <span class="ticker">{{ ticker }}</span>
                        <span class="weight">{{ (weight * 100) | round(2) }}%</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>

            <!-- Regime Probabilities -->
            <div class="card">
                <h2>Regime Probabilities</h2>
                {% if regime_chart %}
                <div class="chart-container">
                    <img src="{{ regime_chart }}" alt="Regime Probabilities">
                </div>
                {% endif %}
                {% if regime_probs %}
                <table>
                    <thead>
                        <tr><th>Regime</th><th>Probability</th></tr>
                    </thead>
                    <tbody>
                        {% for regime, prob in regime_probs.items() %}
                        <tr>
                            <td>{{ regime }}</td>
                            <td>{{ (prob * 100) | round(1) }}%</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <p style="color: var(--muted-color);">No regime data available.</p>
                {% endif %}
            </div>
        </div>

        <!-- Latest Signals -->
        <div class="card">
            <h2>Latest Signals</h2>
            {% if signals_table %}
            <div style="overflow-x: auto;">
                {{ signals_table | safe }}
            </div>
            {% else %}
            <p style="color: var(--muted-color);">No signals available.</p>
            {% endif %}
        </div>

        <!-- Top Asset Family Rankings -->
        <div class="card">
            <h2>Top Asset Family Rankings</h2>
            {% if family_rankings %}
            <table>
                <thead>
                    <tr><th>Rank</th><th>Family</th><th>Score</th></tr>
                </thead>
                <tbody>
                    {% for entry in family_rankings %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ entry.family }}</td>
                        <td>{{ "%.3f" | format(entry.score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p style="color: var(--muted-color);">No family rankings available.</p>
            {% endif %}
        </div>

        <!-- Performance Metrics -->
        <div class="card">
            <h2>Performance Metrics</h2>
            {% if metrics %}
            <div class="metric-grid">
                {% for key, value in metrics.items() %}
                <div class="metric-box">
                    <div class="metric-value">{{ value }}</div>
                    <div class="metric-label">{{ key }}</div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="color: var(--muted-color);">No performance metrics available.</p>
            {% endif %}
        </div>

        <!-- Reason Codes -->
        <div class="card">
            <h2>Allocation Reason</h2>
            <div class="reason-box">
                {{ reason }}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            QuantLab v{{ version }} - Generated at {{ timestamp }}<br>
            <em>This report is for informational purposes only. Not investment advice.</em>
        </div>
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _create_allocation_pie_chart(weights: Dict[str, float]) -> Optional[str]:
    """Create a pie chart of portfolio allocation and return as base64 string.

    Args:
        weights: Ticker to weight mapping.

    Returns:
        Base64-encoded PNG data URI, or None if weights are empty.
    """
    if not weights:
        return None

    # Filter out zero weights and sort
    filtered = {k: v for k, v in weights.items() if v > 0.001}
    if not filtered:
        return None

    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    labels = [ticker for ticker, _ in sorted_items]
    sizes = [weight for _, weight in sorted_items]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 9},
    )
    ax.set_title("Suggested Portfolio Allocation", fontsize=12, fontweight="bold")
    plt.setp(autotexts, size=8, weight="bold")
    plt.tight_layout()

    return _fig_to_base64(fig)


def _create_regime_bar_chart(regime_probs: Dict[str, float]) -> Optional[str]:
    """Create a bar chart of regime probabilities and return as base64 string.

    Args:
        regime_probs: Regime name to probability mapping.

    Returns:
        Base64-encoded PNG data URI, or None if input is empty.
    """
    if not regime_probs:
        return None

    regimes = list(regime_probs.keys())
    probs = [regime_probs[r] * 100 for r in regimes]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#16a34a", "#d97706", "#dc2626"][: len(regimes)]
    bars = ax.bar(regimes, probs, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_title("HMM Regime Probabilities", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{prob:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(loc="upper right")
    plt.tight_layout()

    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_market_state(features: pd.DataFrame) -> Dict[str, str]:
    """Extract latest market state summary from features.

    Args:
        features: Feature DataFrame with MultiIndex columns.

    Returns:
        Dictionary of market state labels to formatted values.
    """
    state: Dict[str, str] = {}

    try:
        if features is None or features.empty:
            return state

        latest = features.iloc[-1]

        # Try to extract prices for key tickers
        close_cols = [c for c in features.columns if isinstance(c, tuple) and c[1] == "Close"]
        if not close_cols:
            close_cols = [c for c in features.columns if "close" in str(c).lower()]

        for col in close_cols[:6]:
            ticker = col[0] if isinstance(col, tuple) else str(col)
            val = latest.get(col)
            if pd.notna(val):
                state[f"{ticker} Close"] = f"{val:.2f}"

        # Add VIX if available
        if any("VIX" in str(c) for c in features.columns):
            vix_col = [c for c in features.columns if "VIX" in str(c)][0]
            vix_val = latest.get(vix_col)
            if pd.notna(vix_val):
                state["VIX Level"] = f"{vix_val:.2f}"

        # Add volatility summary
        vol_cols = [c for c in features.columns if "vol_20" in str(c)]
        if vol_cols:
            avg_vol = np.mean([latest.get(c) for c in vol_cols if pd.notna(latest.get(c))])
            if pd.notna(avg_vol):
                state["Avg Volatility"] = f"{avg_vol * 100:.1f}%"

        # Trend summary
        trend_cols = [c for c in features.columns if "trend_z20" in str(c)]
        if trend_cols:
            avg_trend = np.mean([latest.get(c) for c in trend_cols if pd.notna(latest.get(c))])
            if pd.notna(avg_trend):
                direction = "Up" if avg_trend > 0.5 else "Down" if avg_trend < -0.5 else "Neutral"
                state["Trend Direction"] = direction

    except Exception as exc:
        logger.warning("Could not extract market state: %s", exc)

    return state


def _extract_regime_probs(signals: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract regime probabilities from HMM strategy signals.

    Args:
        signals: List of signal dictionaries.

    Returns:
        Dictionary mapping regime names to probabilities.
    """
    probs: Dict[str, float] = {}

    for signal in signals:
        if "hmm" in signal.get("strategy", "").lower():
            metadata = signal.get("metadata", {})
            if "regime_probs" in metadata:
                rp = metadata["regime_probs"]
                for regime, prob in rp.items():
                    probs[regime] = probs.get(regime, 0.0) + prob
                break
            elif "regime" in signal:
                regime = signal["regime"]
                probs[regime] = 1.0
                break

    # If no HMM signals found, use a default neutral distribution
    if not probs:
        probs = {"Bull": 0.33, "Neutral": 0.34, "Bear": 0.33}

    # Normalise
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def _risk_score_to_color(risk_score: str) -> str:
    """Map risk score string to CSS color hex."""
    mapping = {
        "low": "#16a34a",
        "medium": "#d97706",
        "high": "#dc2626",
        "extreme": "#7c3aed",
    }
    return mapping.get(risk_score.lower(), "#6b7280")


def _format_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format numeric metrics for display.

    Args:
        metrics: Raw metrics dictionary.

    Returns:
        Formatted metrics with string values.
    """
    formatted: Dict[str, str] = {}

    key_mapping = {
        "cagr": "CAGR",
        "sharpe_ratio": "Sharpe",
        "sortino_ratio": "Sortino",
        "calmar_ratio": "Calmar",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "volatility": "Volatility",
        "alpha": "Alpha",
        "beta": "Beta",
        "var_95": "VaR (95%)",
        "cvar_95": "CVaR (95%)",
        "turnover": "Turnover",
    }

    for raw_key, display_key in key_mapping.items():
        if raw_key in metrics:
            val = metrics[raw_key]
            if isinstance(val, (int, float, np.floating, np.integer)):
                if raw_key in ("max_drawdown", "volatility", "var_95", "cvar_95"):
                    formatted[display_key] = f"{val * 100:.2f}%"
                elif raw_key == "win_rate":
                    formatted[display_key] = f"{val * 100:.1f}%"
                elif raw_key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha", "beta", "turnover"):
                    formatted[display_key] = f"{val:.2f}"
                else:
                    formatted[display_key] = f"{val:.2%}"

    return formatted


def _build_signals_table_html(signals: List[Dict[str, Any]]) -> str:
    """Build an HTML table from a list of signals.

    Args:
        signals: List of signal dictionaries.

    Returns:
        HTML string containing a table of signals.
    """
    if not signals:
        return ""

    rows = []
    for sig in signals:
        weights_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in sig.get("weights", {}).items() if v > 0.001
        )
        risk_class = f"risk-{sig.get('risk_state', 'medium')}"
        rows.append(
            f"<tr>"
            f"<td>{sig.get('date', '')}</td>"
            f"<td>{sig.get('strategy', '')}</td>"
            f"<td>{sig.get('family', '')}</td>"
            f"<td>{weights_str}</td>"
            f'<td class="{risk_class}">{sig.get("risk_state", "")}</td>'
            f"<td>{sig.get('confidence', 0.0):.2f}</td>"
            f"<td>{sig.get('reason', '')[:80]}</td>"
            f"</tr>"
        )

    thead = (
        "<thead><tr>"
        "<th>Date</th><th>Strategy</th><th>Family</th>"
        "<th>Weights</th><th>Risk</th><th>Conf</th><th>Reason</th>"
        "</tr></thead>"
    )

    return f"<table>{thead}<tbody>{''.join(rows)}</tbody></table>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_daily_report(
    allocation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    features: pd.DataFrame,
    portfolio_state: Dict[str, Any],
    metrics: Dict[str, Any],
    output_path: Path,
) -> str:
    """Generate a professional daily HTML report.

    Parameters
    ----------
    allocation :
        Final portfolio allocation dict with ``final_weights``,
        ``family_rankings``, ``risk_score``, ``confidence``, and
        ``reason`` keys.
    signals :
        List of all signal dictionaries generated by strategies.
    features :
        Feature DataFrame from :meth:`FeatureEngine.compute_all`.
    portfolio_state :
        Current portfolio state (cash, positions, equity).
    metrics :
        Performance metrics dict from
        :func:`calculate_all_metrics`.
    output_path :
        Path where the HTML file should be written.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    try:
        import jinja2
    except ImportError as exc:
        logger.error("Jinja2 is required for HTML report generation: %s", exc)
        # Fallback to markdown
        md_content = generate_markdown_report(allocation, signals, metrics)
        md_path = output_path.with_suffix(".md")
        md_path.write_text(md_content, encoding="utf-8")
        return str(md_path)

    now = datetime.now(timezone.utc)
    date_str = allocation.get("date", now.strftime("%Y-%m-%d"))

    # Extract data for template
    final_weights = allocation.get("final_weights", {})
    risk_score = allocation.get("risk_score", "unknown")
    confidence = allocation.get("confidence", 0.0)
    family_rankings = allocation.get("family_rankings", [])
    reason = allocation.get("reason", "No reason provided.")

    # Generate charts
    allocation_chart = _create_allocation_pie_chart(final_weights)
    regime_probs = _extract_regime_probs(signals)
    regime_chart = _create_regime_bar_chart(regime_probs)

    # Extract market state
    market_state = _extract_market_state(features)

    # Build signals table HTML
    signals_table = _build_signals_table_html(signals)

    # Format metrics
    formatted_metrics = _format_metrics(metrics)

    # Top family info
    top_family = family_rankings[0].get("family", "N/A") if family_rankings else "N/A"

    # Render template
    env = jinja2.Environment()
    template = env.from_string(_HTML_TEMPLATE)

    html_content = template.render(
        date=date_str,
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        version=_VERSION,
        risk_score=risk_score,
        risk_color=_risk_score_to_color(risk_score),
        confidence_pct=f"{confidence * 100:.1f}",
        top_family=top_family,
        market_state=market_state,
        final_weights=final_weights,
        allocation_chart=allocation_chart,
        regime_probs=regime_probs,
        regime_chart=regime_chart,
        signals_table=signals_table,
        family_rankings=family_rankings,
        metrics=formatted_metrics,
        reason=reason,
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    logger.info("Daily HTML report saved to %s", output_path)
    return str(output_path)


def generate_markdown_report(
    allocation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> str:
    """Generate a markdown summary for quick viewing.

    Parameters
    ----------
    allocation :
        Final portfolio allocation dict.
    signals :
        List of signal dictionaries.
    metrics :
        Performance metrics dict.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    now = datetime.now(timezone.utc)
    date_str = allocation.get("date", now.strftime("%Y-%m-%d"))
    risk_score = allocation.get("risk_score", "unknown")
    confidence = allocation.get("confidence", 0.0)
    reason = allocation.get("reason", "No reason provided.")
    final_weights = allocation.get("final_weights", {})
    family_rankings = allocation.get("family_rankings", [])

    lines: List[str] = [
        "# QuantLab Daily Report",
        "",
        f"**Date:** {date_str}  ",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Version:** {_VERSION}",
        "",
        "---",
        "",
        "## Risk Assessment",
        "",
        f"- **Risk Score:** {risk_score.upper()}",
        f"- **Confidence:** {confidence * 100:.1f}%",
        "",
        "---",
        "",
        "## Suggested Allocation",
        "",
        "| Ticker | Weight |",
        "|--------|--------|",
    ]

    for ticker, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.001:
            lines.append(f"| {ticker} | {weight * 100:.2f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Signals Summary",
        "",
        "| Strategy | Family | Risk State | Confidence |",
        "|----------|--------|------------|------------|",
    ])

    for sig in signals:
        lines.append(
            f"| {sig.get('strategy', '')} "
            f"| {sig.get('family', '')} "
            f"| {sig.get('risk_state', '')} "
            f"| {sig.get('confidence', 0.0):.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Family Rankings",
        "",
    ])

    for i, entry in enumerate(family_rankings, 1):
        lines.append(f"{i}. **{entry.get('family', 'N/A')}** (score: {entry.get('score', 0):.3f})")

    lines.extend([
        "",
        "---",
        "",
        "## Performance Metrics",
        "",
    ])

    if metrics:
        formatted = _format_metrics(metrics)
        for key, value in formatted.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("No metrics available.")

    lines.extend([
        "",
        "---",
        "",
        "## Reason",
        "",
        reason,
        "",
        "---",
        "",
        "*QuantLab v" + _VERSION + " - This report is for informational purposes only. Not investment advice.*",
        "",
    ])

    return "\n".join(lines)
