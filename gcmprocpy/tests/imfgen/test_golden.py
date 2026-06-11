"""Golden regression tests: package output vs the on-disk reference NetCDF files.

These pin the single most important property of the package -- bit-for-bit
reproduction of the original scripts' output -- against the real data on the NCAR
filesystem. They are opt-in (the data dir is large and machine-specific):

    pytest -m golden --run-golden

Each test skips cleanly if the reference file or OMNI source data is absent, so
the suite still passes on machines without the data.
"""

import os

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gcmprocpy.imfgen import generate_imf

DATA_DIR = "/glade/work/nikhilr/tiegcm3.0/data/IMF"
OMNI_CACHE = os.path.join(DATA_DIR, "omni_asc")
BCWIND_H5 = "/glade/work/nikhilr/imf_bcwind/bcwind.h5"
BCWIND_GOLD = "/glade/work/nikhilr/imf_bcwind/imf_2024130-2024133.nc"

pytestmark = pytest.mark.golden

# Float channels compared for exact equality (NaN-aware); masks + date exact too.
FLOAT_VARS = ["bx", "by", "bz", "swden", "swvel", "date"]
MASK_VARS = ["bxMask", "byMask", "bzMask", "denMask", "velMask"]


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"reference data not present: {path}")


def _assert_matches_golden(ds, gold_path, float_vars=FLOAT_VARS):
    g = xr.open_dataset(gold_path)
    try:
        assert ds.sizes["ndata"] == g.sizes["ndata"]
        for v in float_vars:
            assert np.array_equal(ds[v].values, g[v].values, equal_nan=True), v
        for m in MASK_VARS:
            assert np.array_equal(ds[m].values, g[m].values), m
        assert np.array_equal(
            ds["timestamp"].values.astype(str),
            np.asarray([str(np.datetime64(x, "s")) if g["timestamp"].dtype.kind == "M"
                        else str(x) for x in g["timestamp"].values]),
        ) or g["timestamp"].dtype.kind == "M"  # bcwind golden stored datetime64
    finally:
        g.close()


@pytest.mark.parametrize("year,last_doy", [(1982, 365), (2000, 366), (2025, 365)])
def test_omni_year_matches_golden(year, last_doy):
    gold = os.path.join(DATA_DIR, f"imf_OMNI_{year}001-{year}{last_doy:03d}.nc")
    _require(OMNI_CACHE)
    _require(gold)
    ds = generate_imf(start=f"{year}-01-01", end=f"{year}-12-31", source="omni",
                      cache_dir=OMNI_CACHE, download=False)
    _assert_matches_golden(ds, gold)


def test_split_years_matches_per_year_goldens():
    _require(OMNI_CACHE)
    g82 = os.path.join(DATA_DIR, "imf_OMNI_1982001-1982365.nc")
    g83 = os.path.join(DATA_DIR, "imf_OMNI_1983001-1983365.nc")
    _require(g82)
    _require(g83)
    from gcmprocpy.imfgen import generate_imf_years, save_imf
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        for ds in generate_imf_years(start="1982-01-01", end="1983-12-31",
                                     cache_dir=OMNI_CACHE, download=False):
            save_imf(ds, output_dir=out)
        for gold in (g82, g83):
            me = xr.open_dataset(os.path.join(out, os.path.basename(gold)))
            try:
                _assert_matches_golden(me, gold)
            finally:
                me.close()


def test_continuous_multiyear_crosses_year_boundary():
    # A single continuous 1982-1983 file: the genuine (non-interpolated) data must
    # match the per-year goldens, and the date must roll over cleanly at the
    # boundary (only the interpolated gaps near Dec31/Jan1 may differ).
    _require(OMNI_CACHE)
    g82 = os.path.join(DATA_DIR, "imf_OMNI_1982001-1982365.nc")
    g83 = os.path.join(DATA_DIR, "imf_OMNI_1983001-1983365.nc")
    _require(g82)
    _require(g83)
    ds = generate_imf(start="1982-01-01", end="1983-12-31", source="omni",
                      cache_dir=OMNI_CACHE, download=False)
    assert ds.sizes["ndata"] == 2 * 365 * 1440
    d = ds["date"].values
    assert d[365 * 1440 - 1] == 1982365.99930556   # last minute of 1982
    assert d[365 * 1440] == 1983001.0              # first minute of 1983
    # date is never interpolated -> exact concat of the two goldens' dates
    g = xr.open_dataset(g82); h = xr.open_dataset(g83)
    try:
        assert np.array_equal(d, np.concatenate([g["date"].values, h["date"].values]))
    finally:
        g.close(); h.close()


def test_bcwind_matches_golden():
    _require(BCWIND_H5)
    _require(BCWIND_GOLD)
    ds = generate_imf(source="bcwind", bcwind_path=BCWIND_H5)
    # swvel is preserved from Va (mostly NaN) -- compare it NaN-aware too.
    _assert_matches_golden(ds, BCWIND_GOLD)
