"""Shared fixtures for gcmprocpy tests."""
import numpy as np
import xarray as xr
import pytest


@pytest.fixture
def tiegcm_dataset():
    """Create a minimal TIE-GCM-like xarray dataset for testing."""
    times = np.array(['2003-03-20T00:00:00', '2003-03-20T01:00:00'], dtype='datetime64[ns]')
    lats = np.array([-87.5, -82.5, -2.5, 2.5, 82.5, 87.5])
    lons = np.array([-180.0, -90.0, 0.0, 90.0, 175.0])
    levs = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
    ilevs = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 7.25])
    mtime = np.array([[80, 0, 0, 0], [80, 1, 0, 0]])

    np.random.seed(42)
    tn_data = np.random.uniform(200, 1500, size=(2, len(levs), len(lats), len(lons)))
    ne_data = np.random.uniform(1e8, 1e12, size=(2, len(ilevs), len(lats), len(lons)))

    ds = xr.Dataset(
        {
            'TN': (['time', 'lev', 'lat', 'lon'], tn_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
            'NE': (['time', 'ilev', 'lat', 'lon'], ne_data, {'units': 'cm-3', 'long_name': 'ELECTRON DENSITY'}),
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

    ds = xr.Dataset(
        {
            'TN': (['time', 'lev', 'lat', 'lon'], tn_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
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
    """Wrap a TIE-GCM dataset in the list-of-lists format used throughout gcmprocpy."""
    return [[tiegcm_dataset, 'test_tiegcm.nc', 'TIE-GCM']]


@pytest.fixture
def waccmx_datasets(waccmx_dataset):
    """Wrap a WACCM-X dataset in the list-of-lists format used throughout gcmprocpy."""
    return [[waccmx_dataset, 'test_waccmx.nc', 'WACCM-X']]
