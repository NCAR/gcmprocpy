"""gpigen — build TIEGCM GPI (geophysical indices) NetCDF files from GFZ data.

Quick start
-----------
>>> from gcmprocpy import gpigen
>>> ds = gpigen.generate_gpi(start="2024-01-01", end="2024-06-01")
>>> path = gpigen.save_gpi(ds, output_dir=".")

The dataset holds, on a daily grid (``ndays``): ``year_day`` (YYYYDDD),
``f107d`` (daily 10.7 cm flux), ``f107a`` (running-average flux), and ``kp``
(``ndays x 8`` 3-hourly Kp).
"""

from .core import generate_gpi
from .dataset import build_dataset, gpi_filename, save_gpi
from .plotting import make_plots

__all__ = [
    "generate_gpi",
    "save_gpi",
    "build_dataset",
    "gpi_filename",
    "make_plots",
    "__version__",
]

__version__ = "0.1.0"
