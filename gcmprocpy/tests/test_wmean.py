"""Tests for the cos(lat) area-weighted mean reduction ('wmean')."""
import numpy as np

import gcmprocpy as gy

TIME = '2003-03-20T00:00:00'


def _coslat_weighted(values, lats, axis):
    """Reference cos(lat)-weighted, NaN-aware mean over *axis*."""
    w = np.cos(np.deg2rad(np.asarray(lats, dtype=float)))
    shape = [1] * values.ndim
    shape[axis] = len(lats)
    w = w.reshape(shape)
    valid = np.isfinite(values)
    num = np.nansum(np.where(valid, values, 0.0) * w, axis=axis)
    den = np.sum(np.where(valid, w, 0.0), axis=axis)
    return num / den


def test_arr_var_lon_wmean_is_coslat_weighted(tiegcm_datasets):
    # variable-vs-longitude collapses LATITUDE -> cos-lat weighting applies.
    wm = gy.arr_var_lon(tiegcm_datasets, 'TN', TIME, selected_lev_ilev=1.0,
                        selected_lat='wmean', plot_mode=True)
    pm = gy.arr_var_lon(tiegcm_datasets, 'TN', TIME, selected_lev_ilev=1.0,
                        selected_lat='mean', plot_mode=True)
    sl = gy.arr_lat_lon(tiegcm_datasets, 'TN', TIME, selected_lev_ilev=1.0,
                        plot_mode=True)
    expected = _coslat_weighted(sl.values, sl.lats, axis=0)
    np.testing.assert_allclose(wm.values, expected, rtol=1e-10)
    # heavily-polar fixture -> the weighted mean must differ from the plain mean
    assert not np.allclose(wm.values, pm.values)


def test_arr_var_lat_wmean_equals_mean_over_lon(tiegcm_datasets):
    # variable-vs-latitude collapses LONGITUDE -> weighting is a no-op.
    wm = gy.arr_var_lat(tiegcm_datasets, 'TN', TIME, selected_lev_ilev=1.0,
                        selected_lon='wmean', plot_mode=True)
    pm = gy.arr_var_lat(tiegcm_datasets, 'TN', TIME, selected_lev_ilev=1.0,
                        selected_lon='mean', plot_mode=True)
    np.testing.assert_allclose(wm.values, pm.values, rtol=1e-12)


def test_arr_lev_lon_wmean_differs_from_mean(tiegcm_datasets):
    wm = gy.arr_lev_lon(tiegcm_datasets, 'TN', TIME, selected_lat='wmean',
                        plot_mode=True)
    pm = gy.arr_lev_lon(tiegcm_datasets, 'TN', TIME, selected_lat='mean',
                        plot_mode=True)
    assert wm is not None and pm is not None
    assert not np.allclose(wm.values, pm.values)


def test_arr_lev_lat_wmean_equals_mean(tiegcm_datasets):
    # lev-vs-lat collapses LONGITUDE -> weighting is a no-op.
    wm = gy.arr_lev_lat(tiegcm_datasets, 'TN', TIME, selected_lon='wmean',
                        plot_mode=True)
    pm = gy.arr_lev_lat(tiegcm_datasets, 'TN', TIME, selected_lon='mean',
                        plot_mode=True)
    np.testing.assert_allclose(wm.values, pm.values, rtol=1e-12)


def test_arr_lev_var_global_wmean(tiegcm_datasets):
    # global area-weighted mean: collapse lon (plain) and lat (cos-weighted).
    wm = gy.arr_lev_var(tiegcm_datasets, 'TN', TIME, selected_lat='wmean',
                        selected_lon='mean', plot_mode=True)
    pm = gy.arr_lev_var(tiegcm_datasets, 'TN', TIME, selected_lat='mean',
                        selected_lon='mean', plot_mode=True)
    assert wm is not None and pm is not None
    assert wm.values.shape == pm.values.shape
    assert not np.allclose(wm.values, pm.values)


def test_arr_lev_var_wmean_matches_manual_global(tiegcm_dataset):
    # exact check: global cos-lat mean over (lat, lon) at each level.
    from gcmprocpy.containers import ModelDataset
    mds = ModelDataset(ds=tiegcm_dataset, filename='t.nc', model='TIE-GCM')
    wm = gy.arr_lev_var([mds], 'TN', TIME, selected_lat='wmean',
                        selected_lon='mean', plot_mode=True)
    tn = tiegcm_dataset['TN'].sel(time=TIME)            # (lev, lat, lon)
    w = np.cos(np.deg2rad(tiegcm_dataset['lat'].values))
    num = (tn.values * w[None, :, None]).sum(axis=(1, 2))
    den = w.sum() * tn.shape[2]
    expected = num / den
    # arr_lev_var drops all-NaN levels; TN here is fully finite, so shapes match
    np.testing.assert_allclose(np.sort(wm.values), np.sort(expected), rtol=1e-10)
