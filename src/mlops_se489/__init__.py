"""Retail Demand Forecasting.

Weekly retail product demand forecasting MLOps pipeline
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlops_se489")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Stack Overflowers"
__email__ = "aahme147@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
