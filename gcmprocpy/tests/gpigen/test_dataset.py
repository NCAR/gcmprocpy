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
