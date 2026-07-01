"""Tests for gcmprocpy.imfgen.dataset -- build, filename, write."""

import os

import numpy as np
import pytest
import xarray as xr

from gcmprocpy.imfgen.dataset import build_dataset, imf_filename, save_imf
from gcmprocpy.imfgen.processing import CHANNELS


def _processed(n=5):
    return {name: (np.linspace(1.0, 2.0, n), np.ones(n, dtype="int8"))
            for name in CHANNELS}


def _sample_ds(n=5, source="omni"):
    dates = np.array([1982001.0 + i / 1440 for i in range(n)])
    timestamps = np.array([f"1982-01-01T00:0{i}:00" for i in range(n)])
    return build_dataset(_processed(n), dates, timestamps, source=source)


def test_build_dataset_has_all_vars_and_dim():
    ds = _sample_ds()
    expected = {"bx", "bxMask", "by", "byMask", "bz", "bzMask",
                "swden", "denMask", "swvel", "velMask", "date", "timestamp"}
    assert expected <= set(ds.data_vars)
    assert ds.sizes["ndata"] == 5
    assert ds["bxMask"].dtype == np.int8


def test_build_dataset_attrs_by_source():
    omni = _sample_ds(source="omni")
    assert omni.attrs["data_source"] == "omni"
    assert "OMNI" in omni.attrs["Description"]
    assert omni.attrs["url_reference"].endswith("ow_min.html")
    assert omni.attrs["yearday_beg"] == 1982001

    bc = build_dataset(_processed(3),
                       np.array([2024130.75, 2024130.75069, 2024130.7514]),
                       np.array(["2024-05-09T18:00:00"] * 3),
                       source="bcwind", source_path="/x/bcwind.h5")
    assert bc.attrs["data_source"] == "bcwind"
    assert bc.attrs["Source"] == "/x/bcwind.h5"


def test_imf_filename_default_prefix_by_source():
    assert imf_filename(_sample_ds(source="omni")) == "imf_OMNI_1982001-1982001.nc"
    bc = build_dataset(_processed(2), np.array([2024130.75, 2024133.99]),
                       np.array(["2024-05-09T18:00:00", "2024-05-12T23:58:00"]),
                       source="bcwind")
    assert imf_filename(bc) == "imf_bcwind_2024130-2024133.nc"


def test_imf_filename_custom_prefix():
    assert imf_filename(_sample_ds(), prefix="imf_x") == "imf_x_1982001-1982001.nc"


def test_save_imf_auto_name(tmp_path):
    path = save_imf(_sample_ds(), output_dir=str(tmp_path))
    assert path.endswith("imf_OMNI_1982001-1982001.nc")
    assert os.path.exists(path)


def test_save_imf_explicit_path(tmp_path):
    target = str(tmp_path / "sub" / "custom.nc")
    path = save_imf(_sample_ds(), path=target)
    assert path == target and os.path.exists(target)


def test_save_imf_roundtrip(tmp_path):
    ds = _sample_ds()
    path = save_imf(ds, output_dir=str(tmp_path))
    reloaded = xr.open_dataset(path)
    assert np.array_equal(reloaded["date"].values, ds["date"].values)
    assert list(reloaded["timestamp"].values) == list(ds["timestamp"].values)
    assert reloaded["bxMask"].dtype == np.int8
    reloaded.close()


# --- WACCM-X model format ------------------------------------------------

def _sample_waccmx_ds(n=5):
    from datetime import datetime
    dates = np.array([1982001.0 + i / 1440 for i in range(n)])
    iso = np.array([f"1982-01-01T00:0{i}:00" for i in range(n)])
    dts = [datetime(1982, 1, 1, 0, i, 0) for i in range(n)]
    return build_dataset(_processed(n), dates, iso, source="omni",
                         model="waccmx", datetimes=dts)


def test_build_dataset_waccmx_format():
    ds = _sample_waccmx_ds()
    assert "time" in ds.dims and "ndata" not in ds.dims
    assert "timestamp" not in ds.data_vars           # WACCM-X drops the ISO string
    assert {"date", "datefrac", "datesec"} <= set(ds.data_vars)
    assert ds["date"].dtype == np.int32 and ds["datesec"].dtype == np.int32
    assert ds["datefrac"].dtype == np.float64
    assert ds["date"].values[0] == 19820101          # YYYYMMDD int
    assert list(ds["datesec"].values[:3]) == [0, 60, 120]   # exact seconds-of-day
    assert abs(ds["datefrac"].values[0] - 1982001.0) < 1e-9  # preserves yyyyddd.frac
    assert ds.attrs["model"] == "waccmx"


def test_build_dataset_waccmx_requires_datetimes():
    with pytest.raises(ValueError, match="datetimes"):
        build_dataset(_processed(3), np.array([1982001.0, 1982001.001, 1982001.002]),
                      np.array(["x", "y", "z"]), source="omni", model="waccmx")


def test_imf_filename_waccmx_tag():
    assert imf_filename(_sample_waccmx_ds()) == "imf_OMNI_WACCMX_1982001-1982001.nc"
    # an explicit prefix is used verbatim (no auto WACCMX tag)
    assert imf_filename(_sample_waccmx_ds(), prefix="myimf") == "myimf_1982001-1982001.nc"


def test_tiegcm_output_unchanged_no_model_attr():
    # TIE-GCM output must stay attr-identical (no new 'model' attr) for fidelity.
    assert "model" not in _sample_ds().attrs


def test_save_imf_waccmx_unlimited_time(tmp_path):
    import netCDF4 as nc
    path = save_imf(_sample_waccmx_ds(), output_dir=str(tmp_path))
    assert path.endswith("imf_OMNI_WACCMX_1982001-1982001.nc")
    d = nc.Dataset(path)
    assert d.dimensions["time"].isunlimited()
    d.close()


def test_waccmx_is_reformat_of_tiegcm():
    # Same inputs -> the two formats carry identical channel/mask values; only the
    # date encoding and dimension differ (so waccmx is a faithful reformat of the
    # golden-validated tiegcm output).
    from datetime import datetime
    n = 5
    processed = _processed(n)
    dates = np.array([1982001.0 + i / 1440 for i in range(n)])
    iso = np.array([f"1982-01-01T00:0{i}:00" for i in range(n)])
    dts = [datetime(1982, 1, 1, 0, i, 0) for i in range(n)]
    tg = build_dataset(processed, dates, iso, source="omni")
    wx = build_dataset(processed, dates, iso, source="omni",
                       model="waccmx", datetimes=dts)
    for name in CHANNELS:
        assert np.array_equal(tg[name].values, wx[name].values)      # channels identical
    assert np.array_equal(wx["datefrac"].values, tg["date"].values)  # datefrac == tiegcm date
    assert list(wx["date"].values) == [19820101] * n
