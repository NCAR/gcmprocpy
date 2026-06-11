"""Top-level orchestration: fetch -> process -> Dataset."""

from datetime import datetime

import numpy as np

from .dataset import build_dataset
from .dates import coerce_datetime, date_value, iso_timestamp, yesterday
from .processing import process_channels
from .sources import bcwind_samples, omni_samples, omni_samples_hapi

DEFAULT_OMNI_START = "1982-01-01"


def generate_imf(
    start=None,
    end=None,
    source="omni",
    window=10,
    cache_dir=None,
    bcwind_path=None,
    download=True,
    omni_access="hapi",
    verbose=False,
):
    """Generate an IMF ``xarray.Dataset``.

    Parameters
    ----------
    start, end : date-like or None
        Inclusive bounds (``YYYY-MM-DD``, ``YYYYDDD``, ISO, or ``datetime``). For
        ``omni`` they default to ``1982-01-01`` and yesterday. For ``bcwind`` they
        optionally filter the file (default: the file's full span).
    source : {"omni", "bcwind"}
        OMNI 1-minute data (default) or a BCWIND HDF5 file.
    window : int
        Trailing-average window in minutes for the OMNI pipeline (default 10).
        Ignored for ``bcwind`` (raw pass-through).
    cache_dir : str or None
        Directory holding / receiving ``omni_min<year>.asc`` files
        (``omni_access="asc"`` only).
    bcwind_path : str or None
        Path to the BCWIND HDF5 file (required when ``source="bcwind"``).
    download : bool
        Fetch missing OMNI year files over FTP (``omni_access="asc"`` only;
        default True).
    omni_access : {"hapi", "asc"}
        How to obtain OMNI data. ``"hapi"`` (default) fetches only the requested
        window from CDAWeb's HAPI server (no whole-year download) -- best for
        short ranges. ``"asc"`` downloads/parses the SPDF ``omni_min<year>.asc``
        files and reproduces the legacy per-year output exactly. Ignored for
        ``bcwind``.
    verbose : bool
        Print progress.

    Returns
    -------
    xarray.Dataset
        Use :func:`imfgen.save_imf` to write it to NetCDF.
    """
    if source == "omni":
        start_dt = coerce_datetime(start if start is not None else DEFAULT_OMNI_START)
        end_dt = (coerce_datetime(end, end_of_day=True)
                  if end is not None else yesterday())
        if verbose:
            print(f"Generating OMNI IMF {start_dt.date()} -> {end_dt.date()} "
                  f"(window={window}, access={omni_access})")
        if omni_access == "hapi":
            samples = omni_samples_hapi(start_dt, end_dt, window=window,
                                        verbose=verbose)
        elif omni_access == "asc":
            samples = omni_samples(start_dt, end_dt, window=window,
                                   cache_dir=cache_dir, download=download,
                                   verbose=verbose)
        else:
            raise ValueError(
                f"Unknown omni_access {omni_access!r}; expected 'hapi' or 'asc'."
            )
        source_path = None
    elif source == "bcwind":
        if not bcwind_path:
            raise ValueError("source='bcwind' requires bcwind_path=<file.h5>.")
        start_dt = coerce_datetime(start) if start is not None else None
        end_dt = (coerce_datetime(end, end_of_day=True)
                  if end is not None else None)
        if verbose:
            print(f"Converting BCWIND file {bcwind_path}")
        samples = bcwind_samples(bcwind_path, start_dt=start_dt, end_dt=end_dt)
        source_path = bcwind_path
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'omni' or 'bcwind'.")

    timestamps = samples["timestamps"]
    n_out = len(timestamps)
    processed = process_channels(
        samples["channels"], samples["window"], n_out, samples["interpolate"]
    )
    dates = np.array([date_value(t) for t in timestamps])
    iso = np.array([iso_timestamp(t) for t in timestamps])

    ds = build_dataset(processed, dates, iso, source=source, source_path=source_path)
    if verbose:
        print(f"Built IMF dataset: {ds.attrs['yearday_beg']} -> "
              f"{ds.attrs['yearday_end']} ({n_out} minutes)")
    return ds


def generate_imf_years(start=None, end=None, window=10, cache_dir=None,
                       download=True, omni_access="hapi", verbose=False):
    """Yield one OMNI ``Dataset`` per calendar year in ``[start, end]``.

    Each year is generated **independently** (its own within-year interpolation),
    so the per-year files reproduce the legacy ``imf_OMNI_YYYY001-YYYYddd.nc``
    files. This is what ``imfgen --split-years`` writes. (BCWIND files are a
    single span and are not split.) For bit-for-bit reproduction of the legacy
    files use ``omni_access="asc"``.
    """
    start_dt = coerce_datetime(start if start is not None else DEFAULT_OMNI_START)
    end_dt = (coerce_datetime(end, end_of_day=True)
              if end is not None else yesterday())
    for year in range(start_dt.year, end_dt.year + 1):
        y_start = max(start_dt, datetime(year, 1, 1))
        y_end = min(end_dt, datetime(year, 12, 31, 23, 59))
        yield generate_imf(
            start=y_start, end=y_end, source="omni", window=window,
            cache_dir=cache_dir, download=download, omni_access=omni_access,
            verbose=verbose,
        )
