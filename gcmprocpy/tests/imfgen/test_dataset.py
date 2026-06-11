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
