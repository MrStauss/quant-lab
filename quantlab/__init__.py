"""QuantLab -- Modular Quantitative Trading System."""

__version__ = "1.0.0"

try:
    from quantlab.config import QuantLabConfig, ASSET_FAMILIES
except ImportError:
    from config import QuantLabConfig, ASSET_FAMILIES

__all__ = ["QuantLabConfig", "ASSET_FAMILIES"]
