"""QuantLab strategy modules.

All strategies implement the :class:`BaseStrategy` interface and
return a unified signal dictionary.
"""
from quantlab.strategies.base import BaseStrategy
from quantlab.strategies.regime import HMMRegimeStrategy
from quantlab.strategies.dynamic import DynamicCovarianceStrategy
from quantlab.strategies.composer import ComposerRSIStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.combined import CombinedStrategy

__all__ = [
    "BaseStrategy",
    "HMMRegimeStrategy",
    "DynamicCovarianceStrategy",
    "ComposerRSIStrategy",
    "MeanReversionStrategy",
    "CombinedStrategy",
]
