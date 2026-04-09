"""Tests for height interpolation functions in data_parse."""
import numpy as np
import pytest
from gcmprocpy.data_parse import (
    _get_height_var, height_to_pres_level, interpolate_to_height,
)


class TestGetHeightVar:
    def test_tiegcm_returns_zg(self, tiegcm_dataset):
        name, dim, scale = _get_height_var(tiegcm_dataset)
        assert name == 'ZG'
        assert dim == 'ilev'
        assert scale == 1e-5

    def test_waccmx_returns_z3(self, waccmx_dataset):
        name, dim, scale = _get_height_var(waccmx_dataset)
        assert name == 'Z3'
        assert dim == 'lev'
        assert scale == 1e-3

    def test_no_height_var(self, tiegcm_dataset):
        ds = tiegcm_dataset.drop_vars('ZG')
        name, dim, scale = _get_height_var(ds)
        assert name is None
        assert dim is None
        assert scale is None


class TestHeightToPresLevel:
    def test_tiegcm_returns_ilev_value(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        # ZG spans 80–700 km linearly on 9 ilevs; pick ~middle height
        result = height_to_pres_level(tiegcm_datasets, time, 400.0)
        ilevs = tiegcm_datasets[0].ds['ilev'].values
        assert result in ilevs

    def test_tiegcm_low_height_returns_lowest_ilev(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = height_to_pres_level(tiegcm_datasets, time, 80.0)
        assert result == tiegcm_datasets[0].ds['ilev'].values[0]

    def test_tiegcm_high_height_returns_highest_ilev(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = height_to_pres_level(tiegcm_datasets, time, 700.0)
        assert result == tiegcm_datasets[0].ds['ilev'].values[-1]

    def test_waccmx_returns_lev_value(self, waccmx_datasets):
        time = '2003-03-20T00:00:00'
        result = height_to_pres_level(waccmx_datasets, time, 250.0)
        levs = waccmx_datasets[0].ds['lev'].values
        assert result in levs

    def test_with_lat_lon(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = height_to_pres_level(
            tiegcm_datasets, time, 400.0, latitude=2.5, longitude=30.0)
        ilevs = tiegcm_datasets[0].ds['ilev'].values
        assert result in ilevs

    def test_returns_float(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = height_to_pres_level(tiegcm_datasets, time, 300.0)
        assert isinstance(result, float)

    def test_raises_on_missing_height_var(self, tiegcm_datasets):
        # Drop ZG so no height var is available
        tiegcm_datasets[0].ds = tiegcm_datasets[0].ds.drop_vars('ZG')
        with pytest.raises(ValueError, match="Could not find height variable"):
            height_to_pres_level(tiegcm_datasets, '2003-03-20T00:00:00', 300.0)


class TestInterpolateToHeight:
    def test_tiegcm_output_shape(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        ds = tiegcm_datasets[0].ds
        # Get TN at first timestep: (nlev, nlat, nlon) -> average over lon -> (nlev, nlat)
        tn = ds['TN'].sel(time=np.datetime64(time, 'ns')).values
        tn_2d = tn.mean(axis=2)  # (nlev, nlat)
        levs = ds['lev'].values

        result, heights = interpolate_to_height(
            tiegcm_datasets, tn_2d, levs, time, n_heights=20)
        assert result.shape == (20, tn_2d.shape[1])
        assert len(heights) == 20

    def test_waccmx_output_shape(self, waccmx_datasets):
        time = '2003-03-20T00:00:00'
        ds = waccmx_datasets[0].ds
        tn = ds['TN'].sel(time=np.datetime64(time, 'ns')).values
        tn_2d = tn.mean(axis=2)  # (nlev, nlat)
        levs = ds['lev'].values

        result, heights = interpolate_to_height(
            waccmx_datasets, tn_2d, levs, time, n_heights=15)
        assert result.shape == (15, tn_2d.shape[1])
        assert len(heights) == 15

    def test_custom_target_heights(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        ds = tiegcm_datasets[0].ds
        tn = ds['TN'].sel(time=np.datetime64(time, 'ns')).values
        tn_2d = tn.mean(axis=2)
        levs = ds['lev'].values

        targets = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result, heights = interpolate_to_height(
            tiegcm_datasets, tn_2d, levs, time, target_heights=targets)
        np.testing.assert_array_equal(heights, targets)
        assert result.shape[0] == 5

    def test_heights_monotonically_increasing(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        ds = tiegcm_datasets[0].ds
        tn = ds['TN'].sel(time=np.datetime64(time, 'ns')).values
        tn_2d = tn.mean(axis=2)
        levs = ds['lev'].values

        _, heights = interpolate_to_height(
            tiegcm_datasets, tn_2d, levs, time, n_heights=30)
        assert np.all(np.diff(heights) > 0)

    def test_log_interp_runs(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        ds = tiegcm_datasets[0].ds
        # Use NE (positive values) for log interp test
        ne = ds['NE'].sel(time=np.datetime64(time, 'ns')).values
        ne_2d = ne.mean(axis=2)  # (nilev, nlat)
        ilevs = ds['ilev'].values

        result, heights = interpolate_to_height(
            tiegcm_datasets, ne_2d, ilevs, time, n_heights=20, log_interp=True)
        assert result.shape == (20, ne_2d.shape[1])
        # Log interp should produce positive values where data exists
        valid = ~np.isnan(result)
        assert np.all(result[valid] > 0)

    def test_raises_on_missing_height_var(self, tiegcm_datasets):
        tiegcm_datasets[0].ds = tiegcm_datasets[0].ds.drop_vars('ZG')
        dummy_data = np.ones((8, 6))
        levs = np.arange(8, dtype=float)
        with pytest.raises(ValueError, match="Could not find height variable"):
            interpolate_to_height(tiegcm_datasets, dummy_data, levs, '2003-03-20T00:00:00')
