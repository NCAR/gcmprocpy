"""Assemble processed channels into an xarray Dataset and write NetCDF.

The variable set, dimension name (``ndata``) and attributes reproduce the
original ``imf_create.py`` / ``bcwind_imf.py`` output exactly (one extra pair of
convenience attributes -- ``yearday_beg`` / ``yearday_end`` -- is added so the
filename can be derived without re-deriving it from the data).
"""

import os
from datetime import datetime

import numpy as np
import xarray as xr

from .processing import CHANNELS

# data-variable name + mask-variable name for each channel
VAR_NAMES = {"bx": "bx", "by": "by", "bz": "bz", "swden": "swden", "swvel": "swvel"}
MASK_NAMES = {"bx": "bxMask", "by": "byMask", "bz": "bzMask",
              "swden": "denMask", "swvel": "velMask"}
UNITS = {"bx": "nT", "by": "nT", "bz": "nT", "swden": "cm^{-3}", "swvel": "km/s"}
LONG_NAMES = {"bx": "IMF Bx", "by": "IMF By", "bz": "IMF Bz",
              "swden": "solar wind density", "swvel": "solar wind velocity"}
MASK_LONG_NAME = "Quality flag: 0=data derived from linear interpolation."

DEFAULT_PREFIX = {"omni": "imf_OMNI", "bcwind": "imf_bcwind"}

_SOURCE_ATTRS = {
    "omni": {
        "Description": ("10-minute average of OMNI data trailed by 1 minutes. "
                        "Sampled to minute output"),
        "Source": "Hourly OMNI combined 1AU IP Data",
        "url_reference": "https://omniweb.gsfc.nasa.gov/ow_min.html",
    },
    "bcwind": {
        "Description": "BCWIND.h5 to minute output IMF data",
        "Source": "bcwind.h5",
        "url_reference": "https://github.com/AnonNick/IMF",
    },
}


def build_dataset(processed, dates, timestamps, source="omni", source_path=None):
    """Build the IMF ``xarray.Dataset``.

    ``processed`` maps each channel in :data:`CHANNELS` to ``(values, mask)``.
    ``dates`` is the ``YYYYDDD.frac`` float array; ``timestamps`` the ISO strings.
    """
    dates = np.asarray(dates)
    ndata = len(dates)
    data_vars = {}
    for name in CHANNELS:
        values, mask = processed[name]
        data_vars[VAR_NAMES[name]] = (
            "ndata", np.asarray(values, dtype=float),
            {"units": UNITS[name], "long_name": LONG_NAMES[name]},
        )
        data_vars[MASK_NAMES[name]] = (
            "ndata", np.asarray(mask, dtype="int8"),
            {"units": "boolean", "long_name": MASK_LONG_NAME},
        )
    data_vars["date"] = (
        "ndata", dates,
        {"long_name": "year-day plus fractional day: yyyyddd.frac"},
    )
    data_vars["timestamp"] = (
        "ndata", np.asarray(timestamps),
        {"long_name": "Timestamp of the data: YYYY-MM-DDTHH:MM:SS"},
    )

    ds = xr.Dataset(data_vars, coords={"ndata": np.arange(ndata)})

    attrs = dict(_SOURCE_ATTRS[source])
    if source == "bcwind" and source_path:
        attrs["Source"] = str(source_path)
    attrs["CreationTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    attrs["Version"] = "1.0.0"
    attrs["CreatedBy"] = "nikhilr"
    attrs["data_source"] = source
    attrs["yearday_beg"] = int(dates[0])
    attrs["yearday_end"] = int(dates[-1])
    # url_reference last, matching the originals' ordering
    attrs["url_reference"] = _SOURCE_ATTRS[source]["url_reference"]
    ds.attrs.update(attrs)
    return ds


def imf_filename(ds, prefix=None):
    """``<prefix>_<begYYYYDDD>-<endYYYYDDD>.nc`` from the dataset's bounds."""
    if prefix is None:
        prefix = DEFAULT_PREFIX.get(ds.attrs.get("data_source", "omni"), "imf")
    beg = int(ds.attrs["yearday_beg"])
    end = int(ds.attrs["yearday_end"])
    return f"{prefix}_{beg}-{end}.nc"


def save_imf(ds, output_dir=".", prefix=None, path=None):
    """Write ``ds`` to NetCDF and return the path written.

    ``path`` overrides the auto-generated ``<prefix>_<beg>-<end>.nc`` name. (For
    per-year output, generate each year with :func:`imfgen.generate_imf_years`
    and call this once per dataset -- see ``imfgen --split-years``.)
    """
    if path is None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, imf_filename(ds, prefix))
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    ds.to_netcdf(path=path)
    return path
