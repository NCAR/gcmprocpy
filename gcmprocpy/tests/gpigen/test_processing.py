"""Tests for gcmprocpy.gpigen.processing — the core pipeline transforms."""

import numpy as np
import pytest

from gcmprocpy.gpigen.processing import (
    build_kp,
    compute_end_trims,
    fill_fobs,
    find_missing_dates,
    interpolate,
    running_average,
    trim,
)


# --- interpolate ---------------------------------------------------------

def test_interpolate_fills_sentinels():
    out = interpolate(np.array([1.0, -1, -1, 4.0]))
    assert np.allclose(out, [1, 2, 3, 4])


def test_interpolate_leaves_valid_untouched():
    out = interpolate(np.array([5.0, 6.0, 7.0]))
    assert np.allclose(out, [5, 6, 7])


# --- find_missing_dates --------------------------------------------------

def test_find_missing_dates_no_gap():
    new, missing, idx = find_missing_dates([2024001, 2024002, 2024003])
    assert new == [2024001, 2024002, 2024003]
    assert missing == [] and idx == []


def test_find_missing_dates_single_gap():
    new, missing, idx = find_missing_dates([2024001, 2024002, 2024004])
    assert new == [2024001, 2024002, 2024003, 2024004]
    assert missing == [2024003]
    assert idx == [1]


def test_find_missing_dates_multi_day_gap():
    new, missing, idx = find_missing_dates([2024001, 2024005])
    assert new == [2024001, 2024002, 2024003, 2024004, 2024005]
    assert missing == [2024002, 2024003, 2024004]
    assert idx == [0, 0, 0]


# --- fill_fobs (gap fill + interpolate) ----------------------------------

def test_fill_fobs_no_gaps():
    fobs = {
        "datetime": [f"2024-01-0{d}T00:00:00Z" for d in (1, 2, 3)],
        "Fobs": [100.0, 110.0, 120.0],
    }
    yd, f107d, missing = fill_fobs(fobs)
    assert list(yd) == [2024001, 2024002, 2024003]
    assert np.allclose(f107d, [100, 110, 120])
    assert missing == []


def test_fill_fobs_single_gap_uses_immediate_neighbours():
    # Day 2024003 missing between 110 and 130 -> filled with the midpoint 120,
    # NOT mean(110, <day-after>). This guards the off-by-one fix.
    fobs = {
        "datetime": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
                     "2024-01-04T00:00:00Z", "2024-01-05T00:00:00Z"],
        "Fobs": [100.0, 110.0, 130.0, 140.0],
    }
    yd, f107d, missing = fill_fobs(fobs)
    assert list(yd) == [2024001, 2024002, 2024003, 2024004, 2024005]
    assert missing == [2024003]
    assert f107d[2] == 120.0  # (110 + 130) / 2, not (110 + 140) / 2


def test_fill_fobs_multi_day_gap_sequential_midpoints():
    fobs = {
        "datetime": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
                     "2024-01-05T00:00:00Z", "2024-01-06T00:00:00Z"],
        "Fobs": [100.0, 110.0, 140.0, 150.0],
    }
    yd, f107d, missing = fill_fobs(fobs)
    assert missing == [2024003, 2024004]
    assert f107d[2] == 125.0  # (110 + 140) / 2
    assert f107d[3] == 132.5  # (125 + 140) / 2


def test_fill_fobs_interpolates_sentinels():
    fobs = {
        "datetime": [f"2024-01-0{d}T00:00:00Z" for d in (1, 2, 3)],
        "Fobs": [100.0, -1.0, 120.0],
    }
    _, f107d, _ = fill_fobs(fobs)
    assert f107d[1] == 110.0


# --- running_average -----------------------------------------------------

def test_running_average_centered_matches_slice():
    x = np.arange(200, dtype=float)
    a = running_average(x, 81, centered=True)
    assert a[100] == pytest.approx(np.mean(x[60:141]))
    # first/last 40 are incomplete -> left at 0
    assert a[39] == 0 and a[40] != 0
    assert a[159] != 0 and a[160] == 0


def test_running_average_trailing_matches_slice():
    x = np.arange(200, dtype=float)
    a = running_average(x, 27, centered=False)
    assert a[100] == pytest.approx(np.mean(x[73:100]))
    assert a[26] == 0 and a[27] != 0
    assert a[199] != 0  # trailing average is complete at the end


# --- build_kp ------------------------------------------------------------

def test_build_kp_groups_eight_per_day():
    kp_data = {"datetime": [], "Kp": []}
    for d in range(1, 4):
        for h in range(0, 24, 3):
            kp_data["datetime"].append(f"2024-01-0{d}T{h:02d}:00:00Z")
            kp_data["Kp"].append(float(d * 10 + h))
    kp, unique = build_kp(kp_data)
    assert kp.shape == (3, 8)
    assert unique == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert list(kp[0]) == [10, 13, 16, 19, 22, 25, 28, 31]


# --- compute_end_trims ---------------------------------------------------

@pytest.fixture
def aligned_grids():
    yd = list(range(2024001, 2024031))
    ud = [f"2024-01-{d:02d}" for d in range(1, 31)]
    return yd, ud


def test_end_trims_aligned_centered(aligned_grids):
    yd, ud = aligned_grids
    assert compute_end_trims(yd, ud, 40) == (40, 40)


def test_end_trims_aligned_trailing(aligned_grids):
    yd, ud = aligned_grids
    assert compute_end_trims(yd, ud, 0) == (0, 0)


def test_end_trims_fobs_ends_later_small(aligned_grids):
    yd, ud = aligned_grids
    # Fobs 3 days longer than Kp, centered E=40 -> drop 40 fobs, 37 kp.
    assert compute_end_trims(yd, ud[:-3], 40) == (40, 37)


def test_end_trims_kp_ends_later(aligned_grids):
    yd, ud = aligned_grids
    ud_long = ud + ["2024-01-31", "2024-02-01"]
    assert compute_end_trims(yd, ud_long, 40) == (40, 42)


def test_end_trims_fobs_ends_later_trailing(aligned_grids):
    yd, ud = aligned_grids
    # Trailing E=0: drop the 3 extra fobs days, keep all kp.
    assert compute_end_trims(yd, ud[:-3], 0) == (3, 0)


def test_end_trims_length_match_invariant(aligned_grids):
    # After trimming, both grids must come out equal length.
    yd, ud = aligned_grids
    for ud_variant in (ud, ud[:-3], ud[:-1], ud + ["2024-01-31"]):
        for E in (0, 40):
            fobs_end, kp_end = compute_end_trims(yd, ud_variant, E)
            assert len(yd) - fobs_end == len(ud_variant) - kp_end


# --- trim ----------------------------------------------------------------

def test_trim_with_end():
    a = np.arange(10)
    assert list(trim(a, 2, 3)) == [2, 3, 4, 5, 6]


def test_trim_zero_end_keeps_tail():
    a = np.arange(10)
    assert list(trim(a, 2, 0)) == list(range(2, 10))
