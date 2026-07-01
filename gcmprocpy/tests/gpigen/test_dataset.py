"""Tests for gcmprocpy.gpigen.dataset — build, filename, write."""

import numpy as np
import pytest

from gcmprocpy.gpigen.dataset import build_dataset, gpi_filename, save_gpi


def _sample_ds(window=81, centered=True):
    n = 5
    year_day = np.array([2024001 + i for i in range(n)])
    f107d = np.linspace(150, 160, n)
    f107a = np.linspace(155, 156, n)
    kp = np.tile(np.arange(8.0), (n, 1))
    return build_dataset(year_day, f107d, f107a, kp, window, centered, [2024003])


def test_build_dataset_shapes_and_attrs():
    ds = _sample_ds()
    assert ds["kp"].shape == (5, 8)
    assert ds.attrs["yearday_beg"] == 2024001
    assert ds.attrs["yearday_end"] == 2024005
    assert ds.attrs["F107_missing"] == [2024003]


def test_build_dataset_label_reflects_window():
    ds = _sample_ds(window=27, centered=False)
    assert "27-day trailing" in ds["f107a"].attrs["long_name"]
    assert ds.attrs["averaging_kind"] == "trailing"


def test_gpi_filename():
    ds = _sample_ds()
    assert gpi_filename(ds) == "gpi_2024001-2024005.nc"
    assert gpi_filename(ds, prefix="gpi_27avg") == "gpi_27avg_2024001-2024005.nc"


def test_save_gpi_auto_name(tmp_path):
    ds = _sample_ds()
    path = save_gpi(ds, output_dir=str(tmp_path))
    assert path.endswith("gpi_2024001-2024005.nc")
    import os
    assert os.path.exists(path)


def test_save_gpi_explicit_path(tmp_path):
    ds = _sample_ds()
    target = str(tmp_path / "sub" / "custom.nc")
    path = save_gpi(ds, path=target)
    assert path == target
    import os
    assert os.path.exists(target)


def test_save_gpi_roundtrip(tmp_path):
    import xarray as xr
    ds = _sample_ds()
    path = save_gpi(ds, output_dir=str(tmp_path))
    reloaded = xr.open_dataset(path)
    assert list(reloaded["year_day"].values) == list(ds["year_day"].values)
    assert reloaded["kp"].shape == (5, 8)
    reloaded.close()


# --- WACCM-X model format ------------------------------------------------

def _sample_waccmx_ds(window=81, centered=True):
    n = 5
    year_day = np.array([2024001 + i for i in range(n)])
    f107d = np.linspace(150.0, 160.0, n)          # first day = 150.0
    f107a = np.linspace(155.0, 156.0, n)
    kp = np.tile(np.arange(8.0), (n, 1))          # Kp = 0,1,...,7 each day
    return build_dataset(year_day, f107d, f107a, kp, window, centered, [2024003],
                         model="waccmx")


def test_build_dataset_waccmx_3hourly_flatten():
    ds = _sample_waccmx_ds()
    assert ds.sizes["time"] == 40                        # 5 days x 8
    assert "ndays" not in ds.dims and "nkp" not in ds.dims
    assert {"date", "datesec", "f107", "f107a", "kp", "ap"} == set(ds.data_vars)
    assert list(ds["datesec"].values[:8]) == [i * 10800.0 for i in range(8)]
    assert ds["datesec"].values[0] == 0.0                # not NaN (fixes ref bug)
    assert (ds["date"].values[:8] == 20240101.0).all()   # YYYYMMDD, repeated 8x/day
    assert (ds["f107"].values[:8] == 150.0).all()        # daily value repeated
    assert "units" not in ds["kp"].attrs                 # kp is dimensionless (ref bug fixed)
    assert ds.attrs["model"] == "waccmx"


def test_build_dataset_waccmx_ap_from_kp():
    from gcmprocpy.gpigen.indices import kp_to_ap
    ds = _sample_waccmx_ds()
    np.testing.assert_array_equal(ds["ap"].values,
                                  kp_to_ap(ds["kp"].values).astype(float))
    # spot-check the official table
    assert kp_to_ap(0) == 0 and kp_to_ap(1.0) == 4 and kp_to_ap(9.0) == 400


def test_gpi_filename_waccmx_tag():
    assert gpi_filename(_sample_waccmx_ds()) == "gpi_WACCMX_2024001-2024005.nc"


def test_gpi_tiegcm_output_unchanged_no_model_attr():
    assert "model" not in _sample_ds().attrs


def test_save_gpi_waccmx_unlimited_time(tmp_path):
    import netCDF4 as nc
    path = save_gpi(_sample_waccmx_ds(), output_dir=str(tmp_path))
    assert path.endswith("gpi_WACCMX_2024001-2024005.nc")
    d = nc.Dataset(path)
    assert d.dimensions["time"].isunlimited()
    d.close()


def test_waccmx_is_reformat_of_tiegcm():
    # Same inputs -> the WACCM-X 3-hourly series is the exact flatten of the
    # TIE-GCM daily arrays: daily f107/f107a repeated across the 8 slots, kp
    # flattened row-major, ap = kp_to_ap(kp).
    from gcmprocpy.gpigen.indices import kp_to_ap
    tg = _sample_ds()
    wx = _sample_waccmx_ds()
    nkp = tg["kp"].shape[1]
    assert np.array_equal(wx["f107"].values, np.repeat(tg["f107d"].values, nkp))
    assert np.array_equal(wx["f107a"].values, np.repeat(tg["f107a"].values, nkp))
    assert np.array_equal(wx["kp"].values, tg["kp"].values.reshape(-1))
    assert np.array_equal(wx["ap"].values, kp_to_ap(tg["kp"].values.reshape(-1)).astype(float))
