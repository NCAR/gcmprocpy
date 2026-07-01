"""Tests for gcmprocpy.data_magnetic (geographic<->magnetic coordinate support)."""
import sys

import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcmprocpy import data_magnetic as dm


class _FakeMDS:
    """Minimal ModelDataset stand-in (just needs .ds/.model/.filename)."""
    def __init__(self, ds):
        self.ds = ds
        self.model = 'TIE-GCM'
        self.filename = 'test_sech.nc'


def _native_datasets():
    mlat = np.arange(-80.0, 81.0, 10.0)
    mlon = np.arange(-180.0, 180.0, 20.0)
    mlev = np.arange(4.0)
    data = np.arange(mlev.size * mlat.size * mlon.size, dtype=float).reshape(
        mlev.size, mlat.size, mlon.size)
    ds = xr.Dataset(
        {'POTEN': (['mlev', 'mlat', 'mlon'], data,
                   {'units': 'V', 'long_name': 'electric potential'})},
        coords={'mlev': mlev, 'mlat': mlat, 'mlon': mlon})
    return [_FakeMDS(ds)], mlat, mlon, data

apexpy = pytest.importorskip("apexpy")


def test_is_magnetic_var():
    ds = xr.Dataset({
        'ZMAG': (['mlat', 'mlon'], np.zeros((5, 4))),
        'TN':   (['lat', 'lon'], np.zeros((3, 4))),
    })
    assert dm.is_magnetic_var(ds, 'ZMAG') is True
    assert dm.is_magnetic_var(ds, 'TN') is False
    assert dm.is_magnetic_var(ds, 'NOPE') is False


def test_decimal_year():
    assert dm.decimal_year(2003.2) == 2003.2
    assert abs(dm.decimal_year(np.datetime64('2003-01-01T00:00:00')) - 2003.0) < 1e-6
    mid = dm.decimal_year(np.datetime64('2003-07-02T12:00:00'))
    assert 2003.49 < mid < 2003.51
    # numpy scalar years are decimal years, not epoch seconds
    assert dm.decimal_year(np.int64(2015)) == 2015.0
    assert dm.decimal_year(np.float32(2015.5)) == pytest.approx(2015.5)


def test_default_mag_grid():
    mlat, mlon = dm.default_mag_grid()
    assert mlat.min() > -90 and mlat.max() < 90
    assert mlon.min() == -180.0 and mlon.max() < 180.0


def test_require_apexpy_present():
    assert dm.require_apexpy() is apexpy


def test_require_apexpy_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'apexpy', None)      # makes `import apexpy` raise
    with pytest.raises(ImportError, match=r"gcmprocpy\[magnetic\]"):
        dm.require_apexpy()


def test_geo_to_qd_grid_constant():
    glats = np.arange(-87.5, 90, 2.5)
    glons = np.arange(-180, 180, 5.0)
    values = np.full((len(glats), len(glons)), 42.0)
    mlat, mlon, out = dm.geo_to_qd_grid(values, glats, glons, 300.0, 2003.5)
    assert out.shape == (len(mlat), len(mlon))
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.allclose(finite, 42.0)          # constant field -> constant regrid


def test_geo_to_qd_grid_matches_apexpy_inverse():
    # field value == geographic latitude; a QD node's regridded value should equal
    # the geographic latitude that qd2geo maps that node to. Validates the whole
    # inverse-map + bilinear-sample pipeline against apexpy end-to-end.
    glats = np.arange(-87.5, 90, 2.5)
    glons = np.arange(-180, 180, 5.0)
    field = np.repeat(glats[:, None], len(glons), axis=1)     # values == glat
    height = 300.0
    mlat, mlon, out = dm.geo_to_qd_grid(field, glats, glons, height, 2003.5)
    A = apexpy.Apex(date=2003.5)
    checked = 0
    for i in (10, 20, 40, 60):
        for j in (10, 30, 50, 70):
            gl, go, _ = A.qd2geo(mlat[i], mlon[j], height)
            if np.isfinite(out[i, j]) and -85 < gl < 85:
                assert abs(out[i, j] - gl) < 3.0     # interpolation tolerance
                checked += 1
    assert checked > 5, "too few interior QD nodes validated"


# --- native-magnetic extraction + plotting --------------------------------

def test_extract_mag_lat_lon():
    datasets, mlat, mlon, data = _native_datasets()
    mlt, mln, vals, meta = dm.extract_mag_lat_lon(datasets, 'POTEN', time=None, level='mean')
    assert vals.shape == (mlat.size, mlon.size)
    assert np.array_equal(mlt, mlat) and np.array_equal(mln, mlon)
    assert meta['units'] == 'V' and meta['long_name'] == 'electric potential'
    assert np.allclose(vals, data.mean(axis=0))          # 'mean' averages over mlev


def test_extract_mag_lat_lon_rejects_geographic():
    ds = xr.Dataset({'TN': (['lat', 'lon'], np.zeros((3, 4)))},
                    coords={'lat': [0, 1, 2], 'lon': [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="magnetic"):
        dm.extract_mag_lat_lon([_FakeMDS(ds)], 'TN', time=None)


def test_plt_mag_lat_lon_native():
    from gcmprocpy.plot_gen import plt_mag_lat_lon
    datasets, *_ = _native_datasets()
    fig = plt_mag_lat_lon(datasets, variable_name='POTEN',
                          time=np.datetime64('2003-01-01'), level='mean')
    fig = fig[0] if isinstance(fig, tuple) else fig
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Magnetic Longitude (Deg)'
    assert ax.get_ylabel() == 'Magnetic Latitude (Deg)'
    plt.close('all')


def test_plt_mag_lat_lon_geographic(tiegcm_datasets):
    # geographic TN -> quasi-dipole regrid via apexpy
    from gcmprocpy.plot_gen import plt_mag_lat_lon
    fig = plt_mag_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0)
    fig = fig[0] if isinstance(fig, tuple) else fig
    assert fig.axes[0].get_ylabel() == 'Magnetic Latitude (Deg)'
    plt.close('all')


def test_plt_mag_lat_lon_geographic_needs_level(tiegcm_datasets):
    from gcmprocpy.plot_gen import plt_mag_lat_lon
    with pytest.raises(ValueError, match="level is required"):
        plt_mag_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00')
    with pytest.raises(ValueError, match="level is required"):     # 'mean' also rejected (needs altitude)
        plt_mag_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level='mean')
