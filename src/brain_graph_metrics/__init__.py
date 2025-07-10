"""
brain_graph_metrics package

This package provides a CLI-callable wrapper of networkX to compute graph theoretical metrics on brain connectivity networks.
"""

from importlib.metadata import version, PackageNotFoundError
from brain_graph_metrics.brain_graph_metrics import *

try:
    __version__ = version("brain_graph_metrics")
except PackageNotFoundError:
    # If the package metadata isn't available (e.g., during development),
    # fall back to a default version.
    __version__ = "unknown"
