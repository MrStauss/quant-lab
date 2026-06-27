"""Backtesting module for QuantLab."""

from quantlab.backtesting.engine import BacktestEngine
from quantlab.backtesting.benchmarks import calculate_benchmark_returns, compare_to_benchmark

__all__ = [
    "BacktestEngine",
    "calculate_benchmark_returns",
    "compare_to_benchmark",
]
