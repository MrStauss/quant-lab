# quant-lab

> Modular quantitative trading research platform: HMM regime detection, walk-forward backtesting, risk management and paper trading for asset-family ETF strategies.

## Overview

quant-lab (formerly `QuantLab-V1.0`) is a research platform for asset-family-based strategy development. It provides a modular pipeline from data ingestion (Yahoo Finance) through features, strategies, signals and risk guardrails to portfolio allocation, paper execution and daily reporting. Versions are tracked with tags/releases rather than in the repository name; the current version is **1.0.0**.

## Features

- **Multi-Asset-Family Strategies**: Supports Semiconductors, Tech, S&P 500, Russell 2000, Energy, and Financials
- **Regime Detection**: Hidden Markov Model (HMM) for bull/neutral/bear regime classification
- **Dynamic Allocation**: Rolling covariance-based portfolio construction
- **Risk Management**: Kill switch, position limits, drawdown guardrails, volatility targeting
- **Paper Trading**: Full simulation without real broker connection
- **Daily Reporting**: HTML and Markdown report generation
- **Jupyter Dashboard**: Interactive notebook for market visualization

## Quickstart

```bash
# Clone the repository
git clone https://github.com/MrStauss/quant-lab.git
cd quant-lab

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r quantlab/requirements.txt
```

### Dependencies

- Python >= 3.10
- yfinance >= 0.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- hmmlearn >= 0.3.0
- duckdb >= 0.9.0
- pyarrow >= 12.0.0
- jinja2 >= 3.1.0

### Usage

### CLI Commands

QuantLab provides a unified CLI with four modes:

```bash
# Run walk-forward backtest
python -m quantlab.main backtest

# Run paper trading for today
python -m quantlab.main paper

# Run daily pipeline once (no scheduling in V1.0)
python -m quantlab.main daemon

# Generate latest HTML report
python -m quantlab.main report

# Show help
python -m quantlab.main --help
```

### Backtest Options

```bash
# Custom date range
python -m quantlab.main backtest --start-date 2022-01-01 --end-date 2024-12-31

# Daily rebalancing
python -m quantlab.main backtest --rebalance-freq D
```

### Jupyter Dashboard

```bash
# Start Jupyter
cd quantlab/notebooks
jupyter notebook dashboard.ipynb
```

Run each cell in order to see:
- Current market state with latest prices
- Strategy signals table
- Suggested allocation pie chart
- Regime probabilities bar chart
- Risk score indicator
- Correlation heatmap
- Equity curve
- Drawdown chart

### Programmatic Usage

```python
from quantlab.config import QuantLabConfig
from quantlab.daemon.scheduler import DailyDaemon

# Initialize configuration
config = QuantLabConfig()

# Create and run the daily daemon
daemon = DailyDaemon(config)
summary = daemon.run_pipeline()

print(f"Signals: {summary['n_signals']}")
print(f"Approved: {summary['n_approved']}")
print(f"Allocation: {summary['allocation']}")
```

## Project structure

```
quantlab/
├── config.py              # Global configuration & constants
├── main.py                # CLI entry point
├── daemon/
│   └── scheduler.py       # Daily execution pipeline
├── data/
│   ├── ingestion.py       # Yahoo Finance data download
│   ├── storage.py         # DuckDB/SQLite persistence
│   └── validation.py      # Data quality checks
├── features/
│   └── engine.py          # Feature computation
├── strategies/
│   ├── regime.py          # HMM regime rotation
│   ├── dynamic.py         # Dynamic covariance allocation
│   ├── composer.py        # Composer RSI strategy
│   ├── mean_reversion.py  # Mean-reversion strategy
│   └── combined.py        # Ensemble strategy
├── signals/
│   └── engine.py          # Signal normalization
├── risk/
│   └── manager.py         # Risk management
├── portfolio/
│   └── allocator.py       # Portfolio construction
├── backtesting/
│   ├── engine.py          # Walk-forward backtest
│   └── benchmarks.py      # Benchmark comparison
├── execution/
│   └── paper.py           # Paper trading
├── storage/
│   └── repository.py      # State persistence
├── utils/
│   ├── metrics.py         # Performance analytics
│   ├── monte_carlo.py     # Monte Carlo simulations
│   └── reporting.py       # Report generation
└── notebooks/
    └── dashboard.ipynb    # Interactive dashboard
```

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `config` | Central configuration dataclass with defaults for all parameters |
| `daemon` | End-to-end daily pipeline orchestrator |
| `data.ingestion` | Yahoo Finance OHLCV data download |
| `data.storage` | DuckDB/SQLite data persistence with Parquet archive |
| `data.validation` | Data quality checks (stale prices, gaps, bad returns) |
| `features.engine` | Technical feature computation (RSI, volatility, momentum, etc.) |
| `strategies.*` | Trading strategies for regime detection and allocation |
| `signals.engine` | Signal normalization and validation |
| `risk.manager` | Risk guardrails (position limits, drawdown, kill switch) |
| `portfolio.allocator` | Final portfolio construction from signals |
| `backtesting.engine` | Walk-forward backtesting with costs |
| `execution.paper` | Paper trading simulation |
| `storage.repository` | Portfolio state persistence |
| `utils.metrics` | Performance analytics (CAGR, Sharpe, max drawdown, etc.) |
| `utils.monte_carlo` | Forward-looking simulation tools |
| `utils.reporting` | HTML/Markdown report generation |

## Data Flow

```
[Yahoo Finance] -> [data/ingestion] -> [data/storage] -> [features/engine]
                                                            |
[data/validation] <- quality checks <- [data/validation]    |
                                                            v
[signals/engine] <- [strategies/*] <- run all strategies
  |
[risk/manager] <- risk guardrails
  |
[portfolio/allocator] <- final allocation
  |
[execution/paper] <- simulated execution -> [storage/repository]
  |
[utils/reporting] <- daily report
  |
[notebooks/dashboard] <- visualization
```

## Asset Families

| Family | Bull | Base | Bear | Safe Haven | Cash |
|--------|------|------|------|------------|------|
| Semiconductors | SOXL | SOXX | SOXS | GLD | BIL |
| Tech | TECL | XLK | TECS | GLD | BIL |
| S&P 500 | UPRO | SPY | SPXU | SH | BIL |
| Russell 2000 | TNA | IWM | TZA | GLD | BIL |
| Energy | ERX | XLE | ERY | GLD | BIL |
| Financials | FAS | XLF | FAZ | GLD | BIL |

## Results / Screenshots

Backtest equity curves, drawdown charts and daily HTML reports are produced locally by `python -m quantlab.main backtest` / `report` and the Jupyter dashboard. No results are committed to this repository.

## Roadmap

- Scheduled daemon mode (V1.0 runs the daily pipeline once)
- Transaction cost and slippage model refinements
- More asset families and strategy ensembles
- Tagged releases for each version

## Disclaimer

**Research and educational use only. Not financial advice.** Nothing in this repository is a recommendation to buy or sell any security. Backtested and paper-trading results are hypothetical and do not guarantee future performance. Leveraged and inverse ETFs carry substantial risk. Always do your own research and consult a qualified financial advisor before making investment decisions.

### Safety features

**quant-lab (v1.0) operates in paper mode only.** No live broker connections are supported. The platform is designed for research and educational purposes. Always consult a qualified financial advisor before making investment decisions.

Key safety features:
- Paper mode is the default and only execution mode
- Kill switch can halt all trading
- Config-based broker toggle exists but is disabled
- Maximum position size limits (default: 60%)
- Maximum leveraged ETF exposure limits (default: 80%)
- Drawdown guardrails (default: 15%)

## License

Copyright (c) 2026 MrStauss. **All rights reserved.** No license is granted to copy, modify or redistribute this code. You are welcome to read it for reference.
