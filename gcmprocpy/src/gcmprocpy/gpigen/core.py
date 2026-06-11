"""Top-level orchestration: fetch -> process -> Dataset."""

from .dates import coerce_datetime, yesterday
from .processing import (
    build_kp,
    compute_end_trims,
    fill_fobs,
    running_average,
    trim,
)
from .dataset import build_dataset
from .sources import fetch
from datetime import timedelta


def generate_gpi(
    start="1960-01-01",
    end=None,
    source="json",
    window=81,
    centered=True,
    status="def",
    cache_dir=None,
    verbose=False,
):
    """Generate a GPI ``xarray.Dataset`` for ``[start, end]``.

    Parameters
    ----------
    start, end : date-like or None
        Inclusive bounds (``YYYY-MM-DD``, ``YYYYDDD``, ISO, or ``datetime``).
        ``end=None`` defaults to yesterday. The output begins at ``start`` but,
        for a centered window, ends ``window // 2`` days before ``end`` (a
        centered average needs future data that does not yet exist).
    source : {"json", "textfile"}
        GFZ JSON API (default) or the locally-parsed 1932-onward text file.
    window : int
        Averaging window in days (default 81).
    centered : bool
        Centered (default) vs trailing average for ``f107a``.
    status : str
        GFZ ``status`` query param for the JSON API (default ``"def"``).
    cache_dir : str or None
        Where to drop the downloaded text file (``textfile`` source only).
    verbose : bool
        Print progress.

    Returns
    -------
    xarray.Dataset
        Use :func:`gpigen.save_gpi` to write it to NetCDF.
    """
    start_dt = coerce_datetime(start)
    end_dt = coerce_datetime(end, end_of_day=True) if end is not None else yesterday()

    half = window // 2
    start_remove = half if centered else window
    avg_end_invalid = half if centered else 0

    # Fetch extra lead-in days so the average is complete at `start`.
    fetch_start = start_dt - timedelta(days=start_remove)
    if verbose:
        print(
            f"Fetching {source} data {fetch_start.date()} -> {end_dt.date()} "
            f"(window={window}, {'centered' if centered else 'trailing'})"
        )
    fobs_data, kp_data = fetch(
        source, fetch_start, end_dt, status=status, cache_dir=cache_dir, verbose=verbose
    )

    if not fobs_data["Fobs"] or not kp_data["Kp"]:
        raise ValueError(
            f"No GPI data available for {start_dt.date()} -> {end_dt.date()} "
            f"(source={source!r}): the GFZ {source} source returned no records for "
            f"this range. Check the dates fall within the published archive and are "
            f"not in the future."
        )

    year_day, f107d, missing_dates = fill_fobs(fobs_data)
    if verbose and missing_dates:
        print(f"Filled {len(missing_dates)} missing F10.7 day(s): {missing_dates}")

    f107a = running_average(f107d, window, centered)
    kp, unique_dates = build_kp(kp_data)

    fobs_end, kp_end = compute_end_trims(year_day, unique_dates, avg_end_invalid)

    year_day = trim(year_day, start_remove, fobs_end)
    f107d = trim(f107d, start_remove, fobs_end)
    f107a = trim(f107a, start_remove, fobs_end)
    kp = trim(kp, start_remove, kp_end)

    if not (len(year_day) == len(f107d) == len(f107a) == len(kp)):
        raise RuntimeError(
            "Array length mismatch after trimming "
            f"(year_day={len(year_day)}, f107d={len(f107d)}, "
            f"f107a={len(f107a)}, kp={len(kp)})."
        )

    if len(year_day) == 0:
        kind = "centered" if centered else "trailing"
        raise ValueError(
            f"No GPI output days for {start_dt.date()} -> {end_dt.date()}: the "
            f"available data is shorter than the {window}-day {kind} averaging "
            f"window. Request a wider date range or a smaller window."
        )

    ds = build_dataset(year_day, f107d, f107a, kp, window, centered, missing_dates)
    if verbose:
        print(
            f"Built GPI dataset: {ds.attrs['yearday_beg']} -> "
            f"{ds.attrs['yearday_end']} ({len(year_day)} days)"
        )
    return ds
