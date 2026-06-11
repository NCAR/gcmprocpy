"""Tests for gcmprocpy.imfgen.sources -- OMNI parsing/fill-detection and BCWIND reading."""

from datetime import datetime

import numpy as np
import pytest

from gcmprocpy.imfgen import sources
from gcmprocpy.imfgen.sources import (
    _first_run_start,
    _row_all_fill,
    bcwind_samples,
    fetch,
    last_valid_count,
    omni_samples,
    read_bcwind,
)


# --- fill detection ------------------------------------------------------

def test_row_all_fill_flags_fill_rows_only():
    # one all-fill row (all 9999.99) and one valid row (constant 2.0).
    fill_row = np.full((1, 41), 9999.99)
    valid_row = np.full((1, 41), 2.0)
    rows = np.vstack([fill_row, valid_row])
    flags = _row_all_fill(rows)
    assert list(flags) == [True, False]


def test_row_all_fill_partial_fill_is_not_flagged():
    # a row that is fill everywhere except one real column is NOT all-fill.
    row = np.full((1, 41), 9999.99)
    row[0, 3] = 2.0
    assert _row_all_fill(row)[0] == np.False_


def test_row_all_fill_real_heterogeneous_omni_fill_pattern():
    # The real OMNI fill flags span several magnitudes; ALL must be detected as
    # "fill" (their value/9 is a repunit) for a row to be flagged.
    real_fill = [99, 99, 999, 999, 999, 999999, 999999, 99.99, 999999,
                 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99,
                 9999.99, 99999.9, 99999.9, 99999.9, 99999.9, 999.99, 9999999.0]
    row = np.array([real_fill + [9999.99] * (41 - len(real_fill))])
    assert _row_all_fill(row)[0]
    # one realistic in-range B value (not a fill repunit) -> not all-fill
    row2 = row.copy()
    row2[0, 9] = -5.3
    assert not _row_all_fill(row2)[0]


def test_first_run_start():
    f = np.array([0, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert _first_run_start(f, 3) == 1      # run of 3 starts at index 1
    assert _first_run_start(f, 4) == -1     # no run of 4
    assert _first_run_start(np.ones(5, bool), 5) == 0


def test_last_valid_count_complete_keeps_all(fake_omni):
    cache_dir, writer = fake_omni
    writer(1999, 2)  # 2 complete days, no dead tail
    rows = sources.load_omni_year(1999, cache_dir)
    assert last_valid_count(rows) == len(rows)


def test_last_valid_count_truncates_at_dead_day(fake_omni):
    cache_dir, writer = fake_omni
    # 3 days; the last full day is dead -> keep only the first 2 days.
    writer(1999, 3, dead_tail=1440)
    rows = sources.load_omni_year(1999, cache_dir)
    assert last_valid_count(rows) == 2 * 1440


# --- omni_samples (offline, real tiny files) -----------------------------

def test_omni_samples_assembles_lead_in_and_output(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 2)  # leap-year prior file (lead-in)
    writer(2001, 2)  # target
    s = omni_samples(datetime(2001, 1, 1), datetime(2001, 1, 2, 23, 59),
                     window=10, cache_dir=cache_dir, download=False)
    assert s["window"] == 10 and s["interpolate"] is True
    n_out = len(s["timestamps"])
    assert n_out == 2 * 1440
    # channels carry `window` lead-in samples beyond n_out.
    for arr in s["channels"].values():
        assert len(arr) == n_out + 10
    assert s["timestamps"][0] == datetime(2001, 1, 1, 0, 0)
    assert s["timestamps"][-1] == datetime(2001, 1, 2, 23, 59)


def test_omni_samples_edge_trim_drops_dead_tail(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)                       # lead-in year
    writer(2001, 3, dead_tail=1440)       # day 3 dead
    s = omni_samples(datetime(2001, 1, 1), datetime(2001, 1, 3, 23, 59),
                     window=10, cache_dir=cache_dir, download=False)
    assert len(s["timestamps"]) == 2 * 1440         # day 3 trimmed
    assert s["timestamps"][-1] == datetime(2001, 1, 2, 23, 59)


def test_omni_samples_missing_files_raises(tmp_path, patch_download):
    with pytest.raises(FileNotFoundError):
        omni_samples(datetime(2001, 1, 1), datetime(2001, 1, 2),
                     cache_dir=str(tmp_path), download=False)


def test_omni_samples_missing_interior_year_raises(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)   # present
    # 2001 deliberately absent
    writer(2002, 1)   # present -> 2001 is a missing interior year
    with pytest.raises(FileNotFoundError):
        omni_samples(datetime(2000, 1, 1), datetime(2002, 1, 1, 23, 59),
                     cache_dir=cache_dir, download=False)


def test_omni_samples_mid_year_start_lead_in_from_same_year(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2001, 6)
    s = omni_samples(datetime(2001, 1, 5), datetime(2001, 1, 6, 23, 59),
                     window=10, cache_dir=cache_dir, download=False)
    assert s["timestamps"][0] == datetime(2001, 1, 5, 0, 0)
    assert len(s["channels"]["bx"]) == len(s["timestamps"]) + 10  # full lead-in


def test_omni_samples_record_start_pads_lead_in_with_warning(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2001, 2)   # no prior-year file -> lead-in padded
    with pytest.warns(UserWarning, match="lead-in"):
        s = omni_samples(datetime(2001, 1, 1), datetime(2001, 1, 1, 23, 59),
                         window=10, cache_dir=cache_dir, download=False)
    assert np.isnan(s["channels"]["bx"][:10]).all()   # padded with NaN


def test_fetch_omni_matches_omni_samples(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 1)
    kw = dict(start_dt=datetime(2001, 1, 1), end_dt=datetime(2001, 1, 1, 23, 59),
              window=10, cache_dir=cache_dir, download=False)
    a = fetch("omni", **kw)
    b = omni_samples(**kw)
    assert np.array_equal(a["channels"]["bx"], b["channels"]["bx"], equal_nan=True)
    assert a["timestamps"] == b["timestamps"]


# --- BCWIND --------------------------------------------------------------

def test_read_bcwind_returns_expected_keys(fake_bcwind):
    path = fake_bcwind(n=10)
    raw = read_bcwind(path)
    assert set(raw) == {"Bx", "By", "Bz", "D", "Va", "UT"}
    assert len(raw["Bx"]) == 10


def test_bcwind_samples_passthrough(fake_bcwind):
    path = fake_bcwind(n=50)
    s = bcwind_samples(path)
    assert s["window"] == 1 and s["interpolate"] is False
    assert len(s["timestamps"]) == 50
    assert s["timestamps"][0] == datetime(2024, 5, 9, 18, 0)
    assert s["source_path"] == path


def test_bcwind_samples_date_filter(fake_bcwind):
    path = fake_bcwind(n=120, start="2024-05-09 18:00:00")
    s = bcwind_samples(path, start_dt=datetime(2024, 5, 9, 18, 30),
                       end_dt=datetime(2024, 5, 9, 18, 59))
    assert len(s["timestamps"]) == 30
    assert s["timestamps"][0] == datetime(2024, 5, 9, 18, 30)


def test_bcwind_samples_empty_range_raises(fake_bcwind):
    path = fake_bcwind(n=10)
    with pytest.raises(ValueError):
        bcwind_samples(path, start_dt=datetime(2030, 1, 1))


def test_bcwind_date_tracks_real_timestamps_across_year_boundary(fake_bcwind):
    # 3 minutes straddling 2024-12-31 23:59 -> 2025-01-01 00:00. The date must
    # follow each sample's true year/day (the original counter produced 2024367).
    from gcmprocpy.imfgen.dates import date_value
    path = fake_bcwind(n=3, start="2024-12-31 23:58:00")
    s = bcwind_samples(path)
    dates = [date_value(t) for t in s["timestamps"]]
    assert dates[0] == date_value(datetime(2024, 12, 31, 23, 58))  # 2024366.x
    assert dates[2] == 2025001.0                                   # rolls over correctly
    assert int(dates[2]) // 1000 == 2025


# --- dispatch ------------------------------------------------------------

def test_fetch_unknown_source():
    with pytest.raises(ValueError):
        fetch("nope")


def test_fetch_dispatches_to_bcwind(fake_bcwind):
    path = fake_bcwind(n=5)
    s = fetch("bcwind", path=path)
    assert len(s["timestamps"]) == 5
