"""Integration tests for gcmprocpy.imfgen.core (offline, synthetic data)."""

from datetime import datetime

import numpy as np
import pytest

from gcmprocpy.imfgen import generate_imf, generate_imf_years


# --- OMNI ----------------------------------------------------------------

def test_generate_omni_basic(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 2)
    ds = generate_imf(start="2001-01-01", end="2001-01-02", source="omni",
                      cache_dir=cache_dir, download=False, omni_access="asc")
    assert ds.sizes["ndata"] == 2 * 1440
    assert int(ds.attrs["yearday_beg"]) == 2001001
    assert int(ds.attrs["yearday_end"]) == 2001002
    for var in ("bx", "by", "bz", "swden", "swvel", "date", "timestamp",
                "bxMask", "denMask", "velMask"):
        assert var in ds
    assert ds.attrs["data_source"] == "omni"
    assert ds["timestamp"].values[0] == "2001-01-01T00:00:00"


def test_generate_omni_default_start_used(fake_omni, patch_download, monkeypatch):
    cache_dir, writer = fake_omni
    # Default start is 1982-01-01; supply 1981 (lead-in) + a tiny 1982.
    writer(1981, 1)
    writer(1982, 1)
    import gcmprocpy.imfgen.core as core
    monkeypatch.setattr(core, "yesterday", lambda: datetime(1982, 1, 1, 23, 59))
    ds = generate_imf(source="omni", cache_dir=cache_dir, download=False,
                      omni_access="asc")
    assert int(ds.attrs["yearday_beg"]) == 1982001


def test_generate_omni_edge_trim(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 3, dead_tail=1440)
    ds = generate_imf(start="2001-01-01", end="2001-12-31", source="omni",
                      cache_dir=cache_dir, download=False, omni_access="asc")
    assert int(ds.attrs["yearday_end"]) == 2001002   # day 3 trimmed


def test_generate_omni_invokes_download_when_enabled(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 1)
    generate_imf(start="2001-01-01", end="2001-01-01", source="omni",
                 cache_dir=cache_dir, download=True, omni_access="asc")
    assert patch_download, "download_omni_files should have been called"


def test_generate_omni_masks_and_interpolates(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    # mark a block of minutes missing inside day 1 of 2001 -> interpolated.
    writer(2001, 1, missing=range(500, 540))
    ds = generate_imf(start="2001-01-01", end="2001-01-01", source="omni",
                      cache_dir=cache_dir, download=False, omni_access="asc")
    assert (ds["bxMask"].values == 0).any()          # some interpolated
    assert np.all(np.isfinite(ds["bx"].values))      # gaps filled


# --- OMNI via CDAWeb HAPI -------------------------------------------------

def test_generate_omni_hapi_basic(patch_hapi):
    ds = generate_imf(start="2020-01-01", end="2020-01-02", source="omni",
                      omni_access="hapi")
    assert ds.sizes["ndata"] == 2 * 1440      # 2020-01-01 00:00 .. 01-02 23:59
    assert int(ds.attrs["yearday_beg"]) == 2020001
    assert int(ds.attrs["yearday_end"]) == 2020002
    assert ds["timestamp"].values[0] == "2020-01-01T00:00:00"
    assert ds.attrs["data_source"] == "omni"
    for var in ("bx", "by", "bz", "swden", "swvel", "bxMask"):
        assert var in ds
    assert np.all(np.isfinite(ds["bx"].values))


def test_generate_omni_defaults_to_hapi(patch_hapi):
    # No cache_dir / download needed -> confirms hapi is the default access mode.
    ds = generate_imf(start="2020-06-01", end="2020-06-01", source="omni")
    assert ds.sizes["ndata"] == 1440
    assert patch_hapi["calls"], "hapi() should have been called"


def test_generate_omni_hapi_requests_window_lead_in(patch_hapi):
    generate_imf(start="2020-03-10", end="2020-03-10", source="omni",
                 window=10, omni_access="hapi")
    start_str, stop_str = patch_hapi["calls"][0]
    # lead-in = window minutes before start; stop = last minute + 1 (exclusive).
    assert start_str == "2020-03-09T23:50:00Z"
    assert stop_str == "2020-03-11T00:00:00Z"


def test_generate_omni_hapi_masks_and_interpolates(patch_hapi):
    # Inject a fill block into the fetched stream -> masked to NaN -> interpolated.
    patch_hapi["missing"] = set(range(500, 540))
    ds = generate_imf(start="2020-01-01", end="2020-01-01", source="omni",
                      omni_access="hapi")
    assert (ds["bxMask"].values == 0).any()          # some interpolated
    assert np.all(np.isfinite(ds["bx"].values))      # gaps filled


def test_generate_omni_hapi_missing_dependency(monkeypatch):
    from gcmprocpy.imfgen import sources
    monkeypatch.setattr(sources, "_hapi", None)
    with pytest.raises(ImportError, match="hapiclient"):
        generate_imf(start="2020-01-01", end="2020-01-01", source="omni",
                     omni_access="hapi")


def test_generate_omni_unknown_access(patch_hapi):
    with pytest.raises(ValueError, match="omni_access"):
        generate_imf(start="2020-01-01", end="2020-01-01", source="omni",
                     omni_access="bogus")


@pytest.mark.live
def test_generate_omni_hapi_live_small_window():
    # Hits the real CDAWeb HAPI server; opt in with --run-live.
    pytest.importorskip("hapiclient")
    ds = generate_imf(start="2020-01-01", end="2020-01-01", source="omni",
                      omni_access="hapi")
    assert ds.sizes["ndata"] == 1440
    assert np.isfinite(ds["bx"].values).any()


# --- BCWIND --------------------------------------------------------------

def test_generate_bcwind(fake_bcwind):
    path = fake_bcwind(n=100)
    ds = generate_imf(source="bcwind", bcwind_path=path)
    assert ds.sizes["ndata"] == 100
    assert ds.attrs["data_source"] == "bcwind"
    assert np.all(ds["bxMask"].values == 1)          # no interpolation
    assert ds["timestamp"].values[0] == "2024-05-09T18:00:00"


def test_generate_bcwind_masks_high_va(fake_bcwind):
    # Va base above the 1e4 threshold -> swvel masked -> all NaN (mask stays 1).
    path = fake_bcwind(n=30, va_base=2e4)
    ds = generate_imf(source="bcwind", bcwind_path=path)
    assert np.all(np.isnan(ds["swvel"].values))
    assert np.all(ds["velMask"].values == 1)


def test_generate_bcwind_swvel_threshold_straddle(fake_bcwind):
    # Va ramps 9990..10019; only Va <= 1e4 survives (the golden's mixed pattern).
    path = fake_bcwind(n=30, va_base=9990.0)
    ds = generate_imf(source="bcwind", bcwind_path=path)
    sv = ds["swvel"].values
    assert np.isfinite(sv).any() and np.isnan(sv).any()       # genuinely mixed
    assert int(np.sum(np.isfinite(sv))) == 11                  # 9990..10000 kept
    assert np.allclose(sv[np.isfinite(sv)], np.arange(9990.0, 10001.0))
    assert np.all(ds["velMask"].values == 1)                   # no interpolation


def test_generate_bcwind_requires_path():
    with pytest.raises(ValueError):
        generate_imf(source="bcwind")


def test_generate_unknown_source():
    with pytest.raises(ValueError):
        generate_imf(source="nope")


# --- per-year ------------------------------------------------------------

def test_generate_imf_years_yields_one_per_year(fake_omni, patch_download):
    cache_dir, writer = fake_omni
    writer(2000, 1)   # lead-in for 2001
    writer(2001, 1)
    writer(2002, 1)
    out = list(generate_imf_years(start="2001-01-01", end="2002-12-31",
                                  cache_dir=cache_dir, download=False,
                                  omni_access="asc"))
    assert len(out) == 2
    assert int(out[0].attrs["yearday_beg"]) == 2001001
    assert int(out[1].attrs["yearday_beg"]) == 2002001
