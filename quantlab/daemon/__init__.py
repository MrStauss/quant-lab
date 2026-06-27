"""Daily daemon for QuantLab."""

try:
    from quantlab.daemon.scheduler import DailyDaemon
except ImportError:
    from daemon.scheduler import DailyDaemon

__all__ = ["DailyDaemon"]
