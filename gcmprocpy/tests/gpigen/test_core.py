"""Integration tests for gcmprocpy.gpigen.core.generate_gpi (network mocked)."""

import numpy as np
import pytest

from gcmprocpy.gpigen import generate_gpi


def test_generate_centered_trims_window_each_end(patch_fetch, fake_gfz):
    # 160 fetched days starting 2023-11-22; request 2024-01-01..(end aligned).
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    ds = generate_gpi(start="2024-01-01", end="2024-04-30", source="json", verbose=False)
    # Output starts at requested start; centered window ends 40 days before end.
    assert int(ds.attrs["yearday_beg"]) == 2024001
    assert int(ds.attrs["yearday_end"]) == 2024081  # day 121 (Apr 30) - 40
    # f107a complete (non-zero) across the whole trimmed output.
    assert np.all(ds["f107a"].values != 0)
    assert ds["kp"].shape == (ds.sizes["ndays"], 8)


def test_generate_trailing_no_end_trim(patch_fetch, fake_gfz):
    patch_fetch(fake_gfz(start="2023-12-05", n_days=148))
    ds = generate_gpi(
        start="2024-01-01", end="2024-04-30", window=27, centered=False
    )
    assert int(ds.attrs["yearday_beg"]) == 2024001
    assert int(ds.attrs["yearday_end"]) == 2024121  # full range, no end trim
    assert np.all(ds["f107a"].values != 0)


def test_generate_fetches_lead_in_days(patch_fetch, fake_gfz):
    state = patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    generate_gpi(start="2024-01-01", end="2024-04-30")
    call = state["calls"][0]
    # Centered 81 -> fetch start is 40 days before the requested start.
    assert call["start"].date().isoformat() == "2023-11-22"


def test_generate_default_end_is_yesterday(patch_fetch, fake_gfz, monkeypatch):
    from datetime import datetime
    import gcmprocpy.gpigen.core as core

    fixed = datetime(2024, 6, 10, 12, 0, 0)

    class _DT(datetime):
        @classmethod
        def now(cls):
            return fixed

    monkeypatch.setattr(core, "yesterday", lambda: datetime(2024, 6, 9, 23, 59, 59))
    patch_fetch(fake_gfz(start="2024-01-01", n_days=200))
    state = patch_fetch.state
    generate_gpi(start="2024-02-01", end=None)
    assert state["calls"][0]["end"].date().isoformat() == "2024-06-09"


def test_generate_dataset_attrs_and_vars(patch_fetch, fake_gfz):
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    ds = generate_gpi(start="2024-01-01", end="2024-04-30")
    for var in ("year_day", "f107d", "f107a", "kp"):
        assert var in ds
    assert ds.attrs["averaging_window_days"] == 81
    assert ds.attrs["averaging_kind"] == "centered"
    assert ds.attrs["data_source_url"] == "https://kp.gfz.de/"
    assert "F107_missing" in ds.attrs


def test_generate_reports_missing_days(patch_fetch, fake_gfz):
    # Drop one interior day from Fobs -> recorded in F107_missing.
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161, drop_fobs_days=[80]))
    ds = generate_gpi(start="2024-01-01", end="2024-04-30")
    assert len(ds.attrs["F107_missing"]) == 1


def test_generate_all_grids_equal_length(patch_fetch, fake_gfz):
    # Misaligned ends: Fobs ends 2 days after Kp -> still consistent lengths.
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161, kp_short_days=2))
    ds = generate_gpi(start="2024-01-01", end="2024-04-30")
    n = ds.sizes["ndays"]
    assert ds["f107d"].size == n == ds["f107a"].size == ds["kp"].shape[0]


def test_generate_raises_on_empty_data(patch_fetch):
    # GFZ source returns no records (e.g. a future / out-of-archive range):
    # a clear error, not an opaque numpy failure deeper in the pipeline.
    patch_fetch(({"datetime": [], "Fobs": []}, {"datetime": [], "Kp": []}))
    with pytest.raises(ValueError, match="No GPI data available"):
        generate_gpi(start="2099-01-01", end="2099-02-01", verbose=False)


def test_generate_raises_when_shorter_than_window(patch_fetch, fake_gfz):
    # Far fewer days than the 81-day centered window -> everything trims away.
    # Expect a clear message rather than an empty-dataset crash downstream.
    patch_fetch(fake_gfz(start="2024-01-01", n_days=10))
    with pytest.raises(ValueError, match="shorter than the 81-day"):
        generate_gpi(start="2024-01-05", end="2024-01-09", verbose=False)
