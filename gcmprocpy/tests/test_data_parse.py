"""Tests for data_parse module."""
import numpy as np
import pytest
from gcmprocpy.data_parse import (
    time_list, var_list, level_list, lon_list, lat_list,
    dim_list, var_info, dim_info, check_var_dims,
    level_log_transform, get_mtime, get_time,
)


class TestTimeList:
    def test_returns_all_timestamps(self, tiegcm_datasets):
        times = time_list(tiegcm_datasets)
        assert len(times) == 2
        assert times[0] == np.datetime64('2003-03-20T00:00:00', 'ns')
        assert times[1] == np.datetime64('2003-03-20T01:00:00', 'ns')


class TestVarList:
    def test_returns_sorted_variables(self, tiegcm_datasets):
        variables = var_list(tiegcm_datasets)
        assert 'TN' in variables
        assert 'NE' in variables
        assert variables == sorted(variables)

    def test_excludes_coordinates(self, tiegcm_datasets):
        variables = var_list(tiegcm_datasets)
        assert 'lat' not in variables
        assert 'lon' not in variables


class TestLevelList:
    def test_returns_sorted_levels(self, tiegcm_datasets):
        levels = level_list(tiegcm_datasets)
        assert levels == sorted(levels)
        assert len(levels) > 0

    def test_combines_lev_and_ilev(self, tiegcm_datasets):
        levels = level_list(tiegcm_datasets)
        # ilev has 7.25 which is not in lev
        assert 7.25 in levels


class TestLonList:
    def test_returns_sorted_longitudes(self, tiegcm_datasets):
        lons = lon_list(tiegcm_datasets)
        assert lons == sorted(lons)
        assert -180.0 in lons
        assert 175.0 in lons


class TestLatList:
    def test_returns_sorted_latitudes(self, tiegcm_datasets):
        lats = lat_list(tiegcm_datasets)
        assert lats == sorted(lats)
        assert -87.5 in lats
        assert 87.5 in lats


class TestDimList:
    def test_returns_dimensions(self, tiegcm_datasets):
        dims = dim_list(tiegcm_datasets)
        assert 'time' in dims
        assert 'lat' in dims
        assert 'lon' in dims
        assert 'lev' in dims
        assert 'ilev' in dims


class TestVarInfo:
    def test_returns_attributes(self, tiegcm_datasets):
        info = var_info(tiegcm_datasets, 'TN')
        assert 'test_tiegcm.nc' in info
        assert info['test_tiegcm.nc']['attributes']['units'] == 'K'
        assert info['test_tiegcm.nc']['attributes']['long_name'] == 'NEUTRAL TEMPERATURE'

    def test_missing_variable(self, tiegcm_datasets):
        info = var_info(tiegcm_datasets, 'NONEXISTENT')
        assert info['test_tiegcm.nc'] is None


class TestDimInfo:
    def test_returns_dimension_size(self, tiegcm_datasets):
        info = dim_info(tiegcm_datasets, 'lat')
        assert 'test_tiegcm.nc' in info
        assert info['test_tiegcm.nc']['size'] == 6


class TestCheckVarDims:
    def test_lev_variable(self, tiegcm_dataset):
        assert check_var_dims(tiegcm_dataset, 'TN') == 'lev'

    def test_ilev_variable(self, tiegcm_dataset):
        assert check_var_dims(tiegcm_dataset, 'NE') == 'ilev'

    def test_missing_variable(self, tiegcm_dataset):
        assert check_var_dims(tiegcm_dataset, 'NONEXISTENT') == 'Variable not found in dataset'


class TestLevelLogTransform:
    def test_tiegcm_log_level_true_noop(self):
        """TIE-GCM with log_level=True should not transform (already ln(p0/p))."""
        arr = np.array([-7.0, -3.0, 0.0, 3.0, 7.0])
        result = level_log_transform(arr, 'TIE-GCM', log_level=True)
        np.testing.assert_array_equal(result, arr)

    def test_tiegcm_log_level_false_exp(self):
        """TIE-GCM with log_level=False should apply exp()."""
        arr = np.array([0.0, 1.0])
        result = level_log_transform(arr, 'TIE-GCM', log_level=False)
        np.testing.assert_allclose(result, np.exp(arr))

    def test_waccmx_log_level_true_log(self):
        """WACCM-X with log_level=True should apply log()."""
        arr = np.array([1.0, 10.0, 100.0])
        result = level_log_transform(arr, 'WACCM-X', log_level=True)
        np.testing.assert_allclose(result, np.log(arr))

    def test_waccmx_log_level_false_noop(self):
        """WACCM-X with log_level=False should not transform (already hPa)."""
        arr = np.array([1.0, 10.0, 100.0])
        result = level_log_transform(arr, 'WACCM-X', log_level=False)
        np.testing.assert_array_equal(result, arr)


class TestGetMtime:
    def test_returns_mtime_for_timestamp(self, tiegcm_datasets):
        ds = tiegcm_datasets[0][0]
        time = np.datetime64('2003-03-20T00:00:00', 'ns')
        mtime = get_mtime(ds, time)
        assert mtime is not None

    def test_returns_mtime_array(self, tiegcm_datasets):
        ds = tiegcm_datasets[0][0]
        time = np.datetime64('2003-03-20T01:00:00', 'ns')
        mtime = get_mtime(ds, time)
        # mtime should be [day, hour, min, sec]
        assert len(mtime) == 4


class TestGetTime:
    def test_returns_time_for_mtime(self, tiegcm_datasets):
        mtime = [80, 0, 0, 0]
        time = get_time(tiegcm_datasets, mtime)
        assert time == np.datetime64('2003-03-20T00:00:00', 'ns')
