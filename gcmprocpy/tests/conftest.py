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

    shape_lev = (2, len(levs), len(lats), len(lons))
    shape_ilev = (2, len(ilevs), len(lats), len(lons))

    np.random.seed(42)
    tn_data = np.random.uniform(200, 1500, size=shape_lev)
    ne_data = np.random.uniform(1e8, 1e12, size=shape_ilev)
    un_data = np.random.uniform(-100, 100, size=shape_lev)
    vn_data = np.random.uniform(-50, 50, size=shape_lev)

    # Species for emissions / OH / EP flux
    o1_data = np.random.uniform(1e7, 1e10, size=shape_lev)
    no_data = np.random.uniform(1e5, 1e8, size=shape_lev)
    co2_data = np.random.uniform(1e8, 1e11, size=shape_lev)
    o2_data = np.random.uniform(1e10, 1e13, size=shape_lev)
    n2_data = np.random.uniform(1e11, 1e14, size=shape_lev)
    h_data = np.random.uniform(1e4, 1e7, size=shape_lev)
    o3_data = np.random.uniform(1e5, 1e8, size=shape_lev)
    ho2_data = np.random.uniform(1e3, 1e6, size=shape_lev)

    # W (vertical wind on ilev, s⁻¹ for TIE-GCM)
    w_data = np.random.uniform(-1e-4, 1e-4, size=shape_ilev)

    # ZG: geometric height on ilev in cm, increasing with level index
    zg_base = np.linspace(80e5, 700e5, len(ilevs))  # 80–700 km in cm
    zg_data = np.broadcast_to(
        zg_base[np.newaxis, :, np.newaxis, np.newaxis],
        shape_ilev,
    ).copy()

    ds = xr.Dataset(
        {
            'TN': (['time', 'lev', 'lat', 'lon'], tn_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
            'NE': (['time', 'ilev', 'lat', 'lon'], ne_data, {'units': 'cm-3', 'long_name': 'ELECTRON DENSITY'}),
            'UN': (['time', 'lev', 'lat', 'lon'], un_data, {'units': 'cm/s', 'long_name': 'NEUTRAL ZONAL WIND'}),
            'VN': (['time', 'lev', 'lat', 'lon'], vn_data, {'units': 'cm/s', 'long_name': 'NEUTRAL MERIDIONAL WIND'}),
            'O1': (['time', 'lev', 'lat', 'lon'], o1_data, {'units': 'cm-3', 'long_name': 'ATOMIC OXYGEN'}),
            'NO': (['time', 'lev', 'lat', 'lon'], no_data, {'units': 'cm-3', 'long_name': 'NITRIC OXIDE'}),
            'CO2': (['time', 'lev', 'lat', 'lon'], co2_data, {'units': 'cm-3', 'long_name': 'CARBON DIOXIDE'}),
            'O2': (['time', 'lev', 'lat', 'lon'], o2_data, {'units': 'cm-3', 'long_name': 'MOLECULAR OXYGEN'}),
            'N2': (['time', 'lev', 'lat', 'lon'], n2_data, {'units': 'cm-3', 'long_name': 'MOLECULAR NITROGEN'}),
            'H': (['time', 'lev', 'lat', 'lon'], h_data, {'units': 'cm-3', 'long_name': 'ATOMIC HYDROGEN'}),
            'O3': (['time', 'lev', 'lat', 'lon'], o3_data, {'units': 'cm-3', 'long_name': 'OZONE'}),
            'HO2': (['time', 'lev', 'lat', 'lon'], ho2_data, {'units': 'cm-3', 'long_name': 'HYDROPEROXYL'}),
            'OMEGA': (['time', 'ilev', 'lat', 'lon'], w_data, {'units': 's-1', 'long_name': 'VERTICAL WIND'}),
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

    shape_lev = (2, len(levs), len(lats), len(lons))

    np.random.seed(42)
    t_data = np.random.uniform(200, 1500, size=shape_lev)
    u_data = np.random.uniform(-50, 50, size=shape_lev)
    v_data = np.random.uniform(-30, 30, size=shape_lev)

    # Species
    o_data = np.random.uniform(1e7, 1e10, size=shape_lev)
    no_data = np.random.uniform(1e5, 1e8, size=shape_lev)
    co2_data = np.random.uniform(1e8, 1e11, size=shape_lev)
    o2_data = np.random.uniform(1e10, 1e13, size=shape_lev)
    n2_data = np.random.uniform(1e11, 1e14, size=shape_lev)
    h_data = np.random.uniform(1e4, 1e7, size=shape_lev)
    o3_data = np.random.uniform(1e5, 1e8, size=shape_lev)
    ho2_data = np.random.uniform(1e3, 1e6, size=shape_lev)
    w_data = np.random.uniform(-0.01, 0.01, size=shape_lev)

    # Z3: geometric height on lev in m, increasing with level index (lower pressure = higher)
    z3_base = np.linspace(0, 500e3, len(levs))  # 0–500 km in m
    z3_data = np.broadcast_to(
        z3_base[np.newaxis, :, np.newaxis, np.newaxis],
        shape_lev,
    ).copy()

    ds = xr.Dataset(
        {
            # 'TN' kept for backward compat with existing height tests
            'TN': (['time', 'lev', 'lat', 'lon'], t_data, {'units': 'K', 'long_name': 'NEUTRAL TEMPERATURE'}),
            'T': (['time', 'lev', 'lat', 'lon'], t_data, {'units': 'K', 'long_name': 'TEMPERATURE'}),
            'U': (['time', 'lev', 'lat', 'lon'], u_data, {'units': 'm/s', 'long_name': 'ZONAL WIND'}),
            'V': (['time', 'lev', 'lat', 'lon'], v_data, {'units': 'm/s', 'long_name': 'MERIDIONAL WIND'}),
            'W': (['time', 'lev', 'lat', 'lon'], w_data, {'units': 'm/s', 'long_name': 'VERTICAL WIND'}),
            'O': (['time', 'lev', 'lat', 'lon'], o_data, {'units': 'cm-3', 'long_name': 'ATOMIC OXYGEN'}),
            'NO': (['time', 'lev', 'lat', 'lon'], no_data, {'units': 'cm-3', 'long_name': 'NITRIC OXIDE'}),
            'CO2': (['time', 'lev', 'lat', 'lon'], co2_data, {'units': 'cm-3', 'long_name': 'CARBON DIOXIDE'}),
            'O2': (['time', 'lev', 'lat', 'lon'], o2_data, {'units': 'cm-3', 'long_name': 'MOLECULAR OXYGEN'}),
            'N2': (['time', 'lev', 'lat', 'lon'], n2_data, {'units': 'cm-3', 'long_name': 'MOLECULAR NITROGEN'}),
            'H': (['time', 'lev', 'lat', 'lon'], h_data, {'units': 'cm-3', 'long_name': 'ATOMIC HYDROGEN'}),
            'O3': (['time', 'lev', 'lat', 'lon'], o3_data, {'units': 'cm-3', 'long_name': 'OZONE'}),
            'HO2': (['time', 'lev', 'lat', 'lon'], ho2_data, {'units': 'cm-3', 'long_name': 'HYDROPEROXYL'}),
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
