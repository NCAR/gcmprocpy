"""Core data-processing pipeline shared by both original scripts.

This is the unified version of the gap-fill / interpolation / averaging /
edge-trim logic that used to be duplicated (and to diverge) between
``gpi_create.py`` and ``gpi_create_27avg.py``.
"""

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from .dates import date_format

MISSING = -1  # missing-value sentinel for f107d throughout


def interpolate(data):
    """Linear-fill remaining ``-1`` sentinels in a 1-D array."""
    data = np.array(data, dtype=float)
    indices = np.arange(len(data))
    valid = data != MISSING
    out = np.copy(data)
    out[~valid] = np.interp(indices[~valid], indices[valid], data[valid])
    return out


def find_missing_dates(yearday_array):
    """Detect skipped calendar days in a ``YYYYDDD`` sequence.

    Returns ``(new_dates, missing_dates, missing_date_index)`` where
    ``new_dates`` is the gap-filled day sequence, ``missing_dates`` the inserted
    days, and ``missing_date_index`` the index (into the *original* array) after
    which each missing day belongs.
    """
    dates = [datetime.strptime(str(d), "%Y%j") for d in yearday_array]
    missing_dates = []
    new_dates = []
    missing_date_index = []
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        delta_days = (next_date - current_date).days
        new_dates.append(int(current_date.strftime("%Y%j")))
        for day in range(1, delta_days):
            missing_date = current_date + timedelta(days=day)
            missing_dates.append(int(missing_date.strftime("%Y%j")))
            new_dates.append(int(missing_date.strftime("%Y%j")))
            missing_date_index.append(i)
        if i == len(dates) - 2:
            new_dates.append(int(next_date.strftime("%Y%j")))
    return new_dates, missing_dates, missing_date_index


def fill_fobs(fobs_data):
    """Build ``(year_day, f107d, missing_dates)`` from a Fobs dict.

    Fills calendar gaps (inserting the mean of the neighbouring valid days for
    f107d) then linearly interpolates any remaining ``-1`` sentinels.
    """
    year_day = np.array([date_format(dt) for dt in fobs_data["datetime"]])
    f107d = np.array(fobs_data["Fobs"], dtype=float)

    year_day, missing_dates, missing_date_index = find_missing_dates(year_day)

    for i, l_index in enumerate(missing_date_index):
        mi = i + l_index + 1
        # The not-yet-inserted next real day sits at index ``mi``; the previous
        # day is at ``mi - 1``. (The original scripts read ``mi + 1`` for the
        # upper neighbour, which skipped the true next day -- corrected here.)
        lower = _nearest_valid(f107d, mi - 1, -1)
        upper = _nearest_valid(f107d, mi, +1)
        f107d = np.insert(f107d, mi, (lower + upper) / 2)

    f107d = interpolate(f107d)
    return np.array(year_day), f107d, missing_dates


def _nearest_valid(arr, start, step):
    """First non-``-1`` value scanning from ``start`` in direction ``step``.

    Falls back to scanning the opposite direction if nothing is found, so an
    insertion adjacent to a run of sentinels still yields a usable value.
    """
    j = start
    while 0 <= j < len(arr):
        if arr[j] != MISSING:
            return arr[j]
        j += step
    j = start - step
    while 0 <= j < len(arr):
        if arr[j] != MISSING:
            return arr[j]
        j -= step
    return 0.0


def running_average(f107d, window, centered):
    """Running mean of ``f107d``.

    - centered: ``half = window // 2``; ``mean(f107d[i-half : i+half+1])`` for
      ``i`` in ``[half, n-half)`` (window=81 -> the original symmetric average).
    - trailing: ``mean(f107d[i-window : i])`` for ``i`` in ``[window, n)``
      (window=27 -> the original trailing average).

    Boundary samples that lack a full window are left at 0 and trimmed later.
    """
    f107a = np.zeros_like(f107d)
    n = len(f107d)
    if centered:
        half = window // 2
        for i in range(half, n - half):
            f107a[i] = np.mean(f107d[i - half : i + half + 1])
    else:
        for i in range(window, n):
            f107a[i] = np.mean(f107d[i - window : i])
    return f107a


def build_kp(kp_data):
    """Group 3-hourly Kp into one ``(ndays, 8)`` row per unique date.

    Returns ``(kp_array, unique_dates)`` where ``unique_dates`` are sorted
    ``YYYY-MM-DD`` strings aligned with the rows.
    """
    unique_dates = sorted({dt[:10] for dt in kp_data["datetime"]})
    groups = defaultdict(list)
    for stamp, value in zip(kp_data["datetime"], kp_data["Kp"]):
        groups[stamp[:10]].append(value)
    kp = np.array([groups[d][:8] for d in unique_dates])
    return kp, unique_dates


def compute_end_trims(year_day, unique_dates, avg_end_invalid):
    """How many days to drop from the end of the Fobs grid vs the Kp grid.

    ``avg_end_invalid`` is the number of trailing days whose *average* is
    incomplete (``half`` for a centered window, ``0`` for trailing). When the
    Kp and Fobs series end on different days, the trims are adjusted so the two
    grids come out the same length.

    Length-matching constraint (derived):
        len(year_day) - fobs_end == len(unique_dates) - kp_end
    which, with ``E = avg_end_invalid``, gives:
        - Fobs ends later by ``d``:  fobs_end = max(E, d), kp_end = fobs_end - d
        - Kp   ends later by ``d``:  fobs_end = E,          kp_end = E + d
        - aligned:                   fobs_end = E,          kp_end = E
    """
    e = avg_end_invalid
    extra_kp = extra_fobs = 0
    if len(unique_dates) != len(year_day):
        fobs_last = datetime.strptime(str(year_day[-1]), "%Y%j").date()
        kp_last = datetime.strptime(unique_dates[-1], "%Y-%m-%d").date()
        delta = (kp_last - fobs_last).days
        if delta > 0:
            extra_kp = delta
        elif delta < 0:
            extra_fobs = -delta

    if extra_fobs:
        fobs_end = max(e, extra_fobs)
        kp_end = fobs_end - extra_fobs
    elif extra_kp:
        fobs_end = e
        kp_end = e + extra_kp
    else:
        fobs_end = e
        kp_end = e
    return fobs_end, kp_end


def trim(array, start, end):
    """Slice ``array[start : len-end]``, treating ``end == 0`` as ``[start:]``."""
    return array[start : (-end if end else None)]
