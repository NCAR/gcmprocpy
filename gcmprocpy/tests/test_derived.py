"""Tests for the derive-if-missing resolver, derivable fields, and persistence."""
import os

import numpy as np
import pytest
import xarray as xr

import gcmprocpy as gy
from gcmprocpy.containers import ModelDataset, resolve_derivable
from gcmprocpy.data_derived import ensure_field
from gcmprocpy.io import load_datasets, save_derived

TIME = '2003-03-20T00:00:00'


def _tiegcm_ds(include=('O2', 'O1', 'TN'), n2=None, cm3=False):
    """Minimal TIE-GCM-like dataset; N2 omitted unless *n2* is given."""
    t = np.array([TIME], dtype='datetime64[ns]')
    lat = np.array([-30., 0., 30.])
    lon = np.array([0., 120., 240.])
    lev = np.array([-3., 1., 5.])
    shp = (1, 3, 3, 3)
    rng = np.random.default_rng(0)
    if cm3:
        pool = {
            'O2': (rng.uniform(1e11, 1e13, shp), 'cm-3'),
            'O1': (rng.uniform(1e10, 1e12, shp), 'cm-3'),
            'TN': (rng.uniform(200, 1500, shp), 'K'),
            'H':  (rng.uniform(1e4, 1e7, shp), 'cm-3'),
            'O3': (rng.uniform(1e5, 1e8, shp), 'cm-3'),
            'HO2': (rng.uniform(1e3, 1e6, shp), 'cm-3'),
            'NO': (rng.uniform(1e5, 1e8, shp), 'cm-3'),
        }
    else:
        pool = {
            'O2': (rng.uniform(0.05, 0.2, shp), 'MMR'),
            'O1': (rng.uniform(0.1, 0.4, shp), 'MMR'),
            'TN': (rng.uniform(200, 1500, shp), 'K'),
        }
    data = {k: (['time', 'lev', 'lat', 'lon'], v, {'units': u})
            for k, (v, u) in pool.items() if k in include}
    data['mtime'] = (['time', 'mtimedim'], np.array([[80, 0, 0, 0]]))
    if n2 is not None:
        data['N2'] = (['time', 'lev', 'lat', 'lon'], np.full(shp, n2), {'units': 'MMR'})
    return xr.Dataset(
        data,
        coords={'time': t, 'lat': lat, 'lon': lon,
                'lev': xr.DataArray(lev, dims='lev', attrs={'units': 'ln(p0/p)'})},
    )


# --- registry ------------------------------------------------------------

def test_registry_has_n2_and_ratios():
    assert resolve_derivable('N2') is not None
    assert resolve_derivable('O/N2') is not None
    assert resolve_derivable('O_N2') is not None        # GUI-safe alias
    assert resolve_derivable('O/O2+N2') is not None
    assert resolve_derivable('O_O2pN2') is not None      # alias
    assert resolve_derivable('NOT_A_FIELD') is None


# --- N2 residual ---------------------------------------------------------

def test_n2_residual_matches_formula():
    ds = _tiegcm_ds()
    mds = ModelDataset(ds=ds, filename='t.nc', model='TIE-GCM')
    assert 'N2' not in mds.ds.variables
    assert ensure_field(mds, 'N2') is True
    expect = np.clip(1.0 - ds['O2'].values - ds['O1'].values, 1e-5, None)
    assert np.allclose(mds.ds['N2'].values, expect)
    assert mds.ds['N2'].attrs['units'] == 'MMR'


def test_raw_field_takes_priority_over_derivable():
    # A real N2 in the file must be used as-is, never overwritten by the residual.
    mds = ModelDataset(ds=_tiegcm_ds(n2=0.5), filename='t.nc', model='TIE-GCM')
    ensure_field(mds, 'N2')
    assert np.allclose(mds.ds['N2'].values, 0.5)


def test_not_present_not_derivable_returns_false():
    mds = ModelDataset(ds=_tiegcm_ds(), filename='t.nc', model='TIE-GCM')
    assert ensure_field(mds, 'NONEXISTENT') is False


# --- derive-if-missing through the extractors ----------------------------

def test_arr_lat_lon_derives_missing_n2():
    mds = ModelDataset(ds=_tiegcm_ds(), filename='t.nc', model='TIE-GCM')
    pd = gy.arr_lat_lon([mds], 'N2', TIME, selected_lev_ilev=1.0, plot_mode=True)
    assert pd is not None
    assert pd.values.shape == (3, 3)
    assert pd.variable_unit == 'MMR'


def test_ratio_derives_on_derived():
    # O/N2 needs the derived N2 -> exercises the recursive derive chain.
    mds = ModelDataset(ds=_tiegcm_ds(), filename='t.nc', model='TIE-GCM')
    pd = gy.arr_lat_lon([mds], 'O_N2', TIME, selected_lev_ilev=1.0, plot_mode=True)
    assert pd is not None and pd.values.shape == (3, 3)
    o = mds.ds['O1'].sel(time=TIME, lev=1.0).values
    n2 = mds.ds['N2'].sel(time=TIME, lev=1.0).values
    assert np.allclose(pd.values, o / n2)


# --- OH model runs without N2 in the file (the keystone gap) -------------

def test_oh83_runs_without_n2(tiegcm_dataset):
    ds = tiegcm_dataset.drop_vars('N2')
    mds = ModelDataset(ds=ds, filename='t.nc', model='TIE-GCM')
    pd = gy.arr_mkeoh83([mds], 'OH83', TIME, selected_lev_ilev=1.0, plot_mode=True)
    assert pd is not None and np.all(np.isfinite(pd.values))


def test_full_oh_model_runs_without_n2(tiegcm_dataset):
    ds = tiegcm_dataset.drop_vars('N2')
    mds = ModelDataset(ds=ds, filename='t.nc', model='TIE-GCM')
    pd = gy.arr_mkoh_band([mds], 'OH_8_3', TIME, selected_lev_ilev=1.0, plot_mode=True)
    assert pd is not None and pd.values.shape == ds['TN'].isel(time=0, lev=0).shape


# --- persistence (in-place append) ---------------------------------------

def test_save_derived_appends_and_reloads(tmp_path):
    path = str(tmp_path / 'hist.nc')
    _tiegcm_ds().to_netcdf(path)
    datasets = load_datasets(path)
    assert 'N2' not in datasets[0].ds.variables

    written = save_derived(datasets, 'N2')
    assert any(w.endswith(':N2') for w in written)
    # in-memory dataset now has it (reopened from the augmented file)
    assert 'N2' in datasets[0].ds.variables

    # a fresh load reads N2 straight from disk -> no recomputation needed
    fresh = load_datasets(path)
    assert 'N2' in fresh[0].ds.variables
    raw = xr.open_dataset(path)
    expect = np.clip(1.0 - raw['O2'].values - raw['O1'].values, 1e-5, None)
    assert np.allclose(fresh[0].ds['N2'].values, expect)
    assert fresh[0].ds['N2'].attrs.get('units') == 'MMR'


def test_save_derived_skips_existing(tmp_path):
    path = str(tmp_path / 'hist.nc')
    _tiegcm_ds(n2=0.5).to_netcdf(path)
    datasets = load_datasets(path)
    written = save_derived(datasets, 'N2')   # already on disk -> skipped
    assert written == []


def test_save_derived_readonly_raises(tmp_path):
    path = str(tmp_path / 'ro.nc')
    _tiegcm_ds().to_netcdf(path)
    os.chmod(path, 0o444)
    datasets = load_datasets(path)
    try:
        with pytest.raises(PermissionError):
            save_derived(datasets, 'N2')
    finally:
        os.chmod(path, 0o644)


def test_save_derived_rejects_non_derivable(tmp_path):
    path = str(tmp_path / 'hist.nc')
    _tiegcm_ds().to_netcdf(path)
    datasets = load_datasets(path)
    with pytest.raises(ValueError):
        save_derived(datasets, 'TOTALLY_MADE_UP')
