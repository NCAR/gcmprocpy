"""Shared fixtures for gcmprocpy tests."""
import numpy as np
import xarray as xr
import pytest
from gcmprocpy.containers import ModelDataset


@pytest.fixture
def tiegcm_dataset():
    """Create a minimal TIE-GCM-like xarray dataset for testing."""
    times = np.array(['2003-03-20T00:00:00', '2003-03-20T01:00:00'], dtype='datetime64[ns]')
    lats = np.array([-87.5, -82.5, -2.5, 2.5, 82.5, 87.5])
    lons = np.array([-150.0, -90.0, -30.0, 30.0, 90.0, 150.0])
    levs = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
    ilevs = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 7.25])
    mtime = np.array([[80, 0, 0, 0], [80, 1, 0, 0]])

    np.random.seed(42)
    tn_data = np.random.uniform(200, 1500, size=(2, len(levs), len(lats), len(lons)))
    ne_data = np.random.uniform(1e8, 1e12, size=(2, len(ilevs), len(lats), len(lons)))
    un_data = np.random.uniform(-100, 100, size=(2, len(levs), len(lats), len(lons)))
    vn_data = np.random.uniform(-50, 50, size=(2, len(levs), len(lats), len(lons)))

    # ZG: geometric height on ilev in cm, increasing with level index
    zg_base = np.linspace(80e5, 700e5, len(ilevs))  # 80–700 km in cm
    zg_data = np.broadcast_to(
        zg_base[np.newaxis, :, np.newaxis, np.newaxis],
        (2, len(ilevs), len(lats), len(lons)),
    ).copy()

    ds = xr.Dataset(
        {
            'TN': (['time', 'lev', 'lat', 'lon'], tn_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
            'NE': (['time', 'ilev', 'lat', 'lon'], ne_data, {'units': 'cm-3', 'long_name': 'ELECTRON DENSITY'}),
            'UN': (['time', 'lev', 'lat', 'lon'], un_data, {'units': 'cm/s', 'long_name': 'NEUTRAL ZONAL WIND'}),
            'VN': (['time', 'lev', 'lat', 'lon'], vn_data, {'units': 'cm/s', 'long_name': 'NEUTRAL MERIDIONAL WIND'}),
            'ZG': (['time', 'ilev', 'lat', 'lon'], zg_data, {'units': 'cm', 'long_name': 'GEOMETRIC HEIGHT'}),
            'mtime': (['time', 'mtimedim'], mtime),
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons,
            'lev': xr.DataArray(levs, dims='lev', attrs={'units': 'ln(p0/p)'}),
            'ilev': xr.DataArray(ilevs, dims='ilev', attrs={'units': 'ln(p0/p)'}),
        },
    )
    return ds


@pytest.fixture
def waccmx_dataset():
    """Create a minimal WACCM-X-like xarray dataset for testing."""
    times = np.array(['2003-03-20T00:00:00', '2003-03-20T01:00:00'], dtype='datetime64[ns]')
    lats = np.array([-87.5, -2.5, 2.5, 87.5])
    lons = np.array([-180.0, 0.0, 175.0])
    levs = np.array([1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 500.0, 1000.0])
    ilevs = np.array([1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 500.0, 1000.0, 1013.0])

    np.random.seed(42)
    tn_data = np.random.uniform(200, 1500, size=(2, len(levs), len(lats), len(lons)))

    # Z3: geometric height on lev in m, increasing with level index (lower pressure = higher)
    z3_base = np.linspace(0, 500e3, len(levs))  # 0–500 km in m
    z3_data = np.broadcast_to(
        z3_base[np.newaxis, :, np.newaxis, np.newaxis],
        (2, len(levs), len(lats), len(lons)),
    ).copy()

    ds = xr.Dataset(
        {
            'TN': (['time', 'lev', 'lat', 'lon'], tn_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
            'Z3': (['time', 'lev', 'lat', 'lon'], z3_data, {'units': 'm', 'long_name': 'GEOMETRIC HEIGHT'}),
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons,
            'lev': xr.DataArray(levs, dims='lev', attrs={'units': 'hPa'}),
            'ilev': xr.DataArray(ilevs, dims='ilev', attrs={'units': 'hPa'}),
        },
    )
    return ds


@pytest.fixture
def tiegcm_datasets(tiegcm_dataset):
    """Wrap a TIE-GCM dataset as a list of ModelDataset objects."""
    return [ModelDataset(ds=tiegcm_dataset, filename='test_tiegcm.nc', model='TIE-GCM')]


@pytest.fixture
def waccmx_datasets(waccmx_dataset):
    """Wrap a WACCM-X dataset as a list of ModelDataset objects."""
    return [ModelDataset(ds=waccmx_dataset, filename='test_waccmx.nc', model='WACCM-X')]
