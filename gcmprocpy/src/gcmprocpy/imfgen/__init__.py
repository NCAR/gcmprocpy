"""imfgen -- build TIEGCM IMF / solar-wind boundary NetCDF files.

Quick start
-----------
>>> from gcmprocpy import imfgen
>>> ds = imfgen.generate_imf(start="1982-01-01", end="1982-12-31")
>>> path = imfgen.save_imf(ds, output_dir=".")

The dataset holds, on a per-minute grid (``ndata``): ``bx``/``by``/``bz`` (nT),
``swden`` (cm^-3), ``swvel`` (km/s) -- each with a 0/1 ``*Mask`` quality flag
(0 = linearly interpolated) -- plus ``date`` (``YYYYDDD.frac``) and an ISO
``timestamp`` string.
"""

from .core import generate_imf, generate_imf_years
from .dataset import build_dataset, imf_filename, save_imf

# Vendored into gcmprocpy: this subpackage is no longer installed under its own
# distribution name, so pin the version here rather than reading package metadata.
__version__ = "0.1.0"

__all__ = [
    "generate_imf",
    "generate_imf_years",
    "save_imf",
    "build_dataset",
    "imf_filename",
    "__version__",
]
