"""Source-agnostic processing shared by the OMNI and BCWIND pipelines.

A *source* (see :mod:`imfgen.sources`) returns five raw per-sample channel
arrays plus the output timestamps. This module turns those into the final
masked / averaged / interpolated channels, identically to the original
``imf_create.py`` (OMNI) and ``bcwind_imf.py`` (BCWIND) scripts:

1. ``apply_range_masks`` -- out-of-range values (fill flags, bad data) -> NaN.
2. ``trailing_average``  -- per output minute, the NaN-aware mean of the
   ``window`` samples ``channel[k:k+window]`` (window=1 is a pass-through).
3. ``interpolate_and_mask`` -- linearly fill the NaN gaps and emit a 0/1 quality
   flag (0 = the value was filled). Skipped for raw pass-through sources, whose
   masks stay all-ones (NaNs preserved).

The five channels and the order they are written:
"""

import warnings

import numpy as np

CHANNELS = ("bx", "by", "bz", "swden", "swvel")

# Upper bounds above which a value is treated as missing and set to NaN. These
# are the OMNI thresholds from ``imf_create.py``; the BCWIND script used 1e4 for
# the B components, but real |B| << 1000 nT so the two are equivalent on data --
# unified here to a single set. (swvel=1e4 also masks BCWIND's Va field: Va is
# usually > 1e4 so most BCWIND swvel samples drop to NaN while the few
# sub-threshold ones pass through, matching the existing output.)
RANGE_LIMITS = {"bx": 1e3, "by": 1e3, "bz": 1e3, "swden": 500.0, "swvel": 1e4}


def apply_range_masks(channels):
    """Set out-of-range samples to NaN, in place, and return ``channels``.

    ``channels`` maps each name in :data:`CHANNELS` to a float ndarray. The
    comparison is ``value > limit`` (strict), exactly as the originals.
    """
    for name, arr in channels.items():
        limit = RANGE_LIMITS[name]
        arr[arr > limit] = np.nan
    return channels


def trailing_average(arr, window, n_out):
    """NaN-aware trailing mean: ``out[k] = nanmean(arr[k:k+window])``.

    ``arr`` must have at least ``n_out + window - 1`` samples. An all-NaN window
    yields NaN (matching the originals' explicit guard). ``window == 1`` is an
    exact pass-through of the first ``n_out`` samples.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < n_out + window - 1:
        raise ValueError(
            f"need >= {n_out + window - 1} samples for window={window}, "
            f"n_out={n_out}; got {len(arr)}."
        )
    if window == 1:
        return arr[:n_out].copy()
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)[:n_out]
    with warnings.catch_warnings():
        # An all-NaN window is expected (data gaps) and returns NaN; the
        # original code guarded this case explicitly to avoid the warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(windows, axis=1)


def interpolate_and_mask(values):
    """Linearly interpolate NaN gaps in ``values``; return ``(filled, mask)``.

    ``mask`` is int8, ``0`` where a value was filled by interpolation and ``1``
    where it is a genuine average. Mirrors the per-channel ``np.interp`` block in
    ``imf_create.py`` (interpolating over integer sample indices).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mask = np.ones(n, dtype="int8")
    nan = np.isnan(values)
    if not nan.any():
        return values.copy(), mask
    mask[nan] = 0
    valid = ~nan
    if not valid.any():
        # No anchor points to interpolate from -- cannot happen on real data.
        # Warn (the original would have raised) and pass the all-NaN channel
        # through with an all-zero mask rather than crashing the whole run.
        warnings.warn(
            "channel has no valid samples; leaving it all-NaN with mask all-zeros.",
            stacklevel=2,
        )
        return values.copy(), mask
    idx = np.arange(n)
    filled = values.copy()
    filled[nan] = np.interp(idx[nan], idx[valid], values[valid])
    return filled, mask


def process_channels(channels, window, n_out, interpolate):
    """Run the full per-channel pipeline; return ``{name: (values, mask)}``.

    ``channels`` are the raw source arrays (length ``>= n_out + window - 1``).
    With ``interpolate=False`` the masks are all-ones and NaNs are preserved
    (the BCWIND pass-through behaviour).
    """
    apply_range_masks(channels)
    out = {}
    for name in CHANNELS:
        avg = trailing_average(channels[name], window, n_out)
        if interpolate:
            out[name] = interpolate_and_mask(avg)
        else:
            out[name] = (avg, np.ones(n_out, dtype="int8"))
    return out
