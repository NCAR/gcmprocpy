"""Tests for gcmprocpy.imfgen.processing -- the shared transforms."""

import numpy as np
import pytest

from gcmprocpy.imfgen.processing import (
    CHANNELS,
    RANGE_LIMITS,
    apply_range_masks,
    interpolate_and_mask,
    process_channels,
    trailing_average,
)


# --- apply_range_masks ---------------------------------------------------

def test_apply_range_masks_sets_out_of_range_to_nan():
    ch = {
        "bx": np.array([5.0, 9999.99]),
        "by": np.array([1.0, 1e4]),       # 1e4 is NOT > 1e3? it is > 1e3 -> nan
        "bz": np.array([-3.0, 2000.0]),
        "swden": np.array([6.0, 600.0]),
        "swvel": np.array([400.0, 5e4]),
    }
    out = apply_range_masks({k: v.copy() for k, v in ch.items()})
    assert np.isnan(out["bx"][1]) and out["bx"][0] == 5.0
    assert np.isnan(out["by"][1])
    assert np.isnan(out["bz"][1])
    assert np.isnan(out["swden"][1])
    assert np.isnan(out["swvel"][1])


def test_range_limit_keeps_real_bcwind_va_below_threshold():
    # Va ~ 36000 (>1e4) is masked; a real flow speed ~400 is kept.
    assert RANGE_LIMITS["swvel"] == 1e4


def test_apply_range_masks_mutates_in_place():
    # Documented contract: masking happens in place and the same dict is returned.
    arr = np.array([5.0, 9999.99])
    ch = {"bx": arr, "by": np.array([1.0]), "bz": np.array([1.0]),
          "swden": np.array([6.0]), "swvel": np.array([400.0])}
    out = apply_range_masks(ch)
    assert out is ch
    assert np.isnan(arr[1])          # caller's own array was modified


# --- trailing_average ----------------------------------------------------

def test_trailing_average_window_one_is_passthrough():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(trailing_average(a, 1, 4), a)


def test_trailing_average_matches_manual_nanmean():
    a = np.arange(20, dtype=float)
    out = trailing_average(a, 10, 5)
    for k in range(5):
        assert out[k] == pytest.approx(np.nanmean(a[k:k + 10]))


def test_trailing_average_ignores_nan_in_window():
    a = np.array([np.nan, 2.0, 4.0, np.nan, 6.0, 8.0])
    out = trailing_average(a, 3, 3)
    assert out[0] == pytest.approx(3.0)   # nanmean([nan, 2, 4])
    assert out[1] == pytest.approx(3.0)   # nanmean([2, 4, nan])
    assert out[2] == pytest.approx(5.0)   # nanmean([4, nan, 6])


def test_trailing_average_all_nan_window_is_nan():
    a = np.array([np.nan, np.nan, np.nan, 5.0])
    out = trailing_average(a, 3, 2)
    assert np.isnan(out[0])               # window [nan,nan,nan]
    assert out[1] == pytest.approx(5.0)   # window [nan,nan,5]


def test_trailing_average_too_short_raises():
    with pytest.raises(ValueError):
        trailing_average(np.arange(5.0), 10, 5)


# --- interpolate_and_mask ------------------------------------------------

def test_interpolate_fills_interior_gap():
    filled, mask = interpolate_and_mask(np.array([1.0, np.nan, np.nan, 4.0]))
    assert np.allclose(filled, [1, 2, 3, 4])
    assert list(mask) == [1, 0, 0, 1]


def test_interpolate_extrapolates_flat_at_edges():
    filled, mask = interpolate_and_mask(np.array([np.nan, 2.0, 4.0, np.nan]))
    assert filled[0] == 2.0 and filled[-1] == 4.0   # np.interp clamps to ends
    assert list(mask) == [0, 1, 1, 0]


def test_interpolate_no_nan_keeps_mask_ones():
    filled, mask = interpolate_and_mask(np.array([5.0, 6.0, 7.0]))
    assert np.array_equal(filled, [5, 6, 7])
    assert np.all(mask == 1)


def test_interpolate_all_nan_returns_nan_and_zero_mask():
    with pytest.warns(UserWarning, match="no valid samples"):
        filled, mask = interpolate_and_mask(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(filled))
    assert np.all(mask == 0)


def test_mask_dtype_is_int8():
    _, mask = interpolate_and_mask(np.array([1.0, np.nan, 3.0]))
    assert mask.dtype == np.int8


# --- process_channels ----------------------------------------------------

def _channels(n_lead_plus_out):
    return {name: np.linspace(1.0, 2.0, n_lead_plus_out) for name in CHANNELS}


def test_process_channels_omni_like_interpolates():
    ch = _channels(14)  # window 10 + n_out 5 - 1 = 14
    out = process_channels(ch, window=10, n_out=5, interpolate=True)
    assert set(out) == set(CHANNELS)
    for name in CHANNELS:
        values, mask = out[name]
        assert len(values) == 5 and len(mask) == 5


def test_process_channels_passthrough_keeps_masks_ones():
    ch = {name: np.array([1.0, np.nan, 3.0]) for name in CHANNELS}
    out = process_channels(ch, window=1, n_out=3, interpolate=False)
    for name in CHANNELS:
        values, mask = out[name]
        assert np.all(mask == 1)             # raw pass-through: no interpolation
        assert np.isnan(values[1])           # NaN preserved
