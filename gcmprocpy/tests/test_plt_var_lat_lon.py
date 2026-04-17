"""Tests for plt_var_lat / plt_var_lon (1D meridional and zonal line plots)
and their underlying arr_var_lat / arr_var_lon extractors."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcmprocpy.data_parse import arr_var_lat, arr_var_lon
from gcmprocpy.plot_gen import plt_var_lat, plt_var_lon


class TestArrVarLat:
    def test_shape_matches_lat_axis(self, tiegcm_datasets):
        result = arr_var_lat(tiegcm_datasets, 'TN',
                             time='2003-03-20T00:00:00',
                             selected_lev_ilev=1.0, selected_lon=30.0,
                             plot_mode=True)
        assert result.values.shape == result.lats.shape
        assert result.values.ndim == 1
        assert result.selected_lon == pytest.approx(30.0)

    def test_zonal_mean_lon(self, tiegcm_datasets):
        result = arr_var_lat(tiegcm_datasets, 'TN',
                             time='2003-03-20T00:00:00',
                             selected_lev_ilev=1.0, selected_lon='mean',
                             plot_mode=True)
        assert result.selected_lon == 'mean'
        assert result.values.ndim == 1

    def test_returns_none_for_missing_time(self, tiegcm_datasets):
        result = arr_var_lat(tiegcm_datasets, 'TN',
                             time='1999-01-01T00:00:00',
                             selected_lev_ilev=1.0, selected_lon=0.0,
                             plot_mode=True)
        assert result is None


class TestArrVarLon:
    def test_shape_matches_lon_axis(self, tiegcm_datasets):
        result = arr_var_lon(tiegcm_datasets, 'TN',
                             time='2003-03-20T00:00:00',
                             selected_lev_ilev=1.0, selected_lat=2.5,
                             plot_mode=True)
        assert result.values.shape == result.lons.shape
        assert result.values.ndim == 1
        assert result.selected_lat == pytest.approx(2.5)

    def test_meridional_mean_lat(self, tiegcm_datasets):
        result = arr_var_lon(tiegcm_datasets, 'TN',
                             time='2003-03-20T00:00:00',
                             selected_lev_ilev=1.0, selected_lat='mean',
                             plot_mode=True)
        assert result.selected_lat == 'mean'
        assert result.values.ndim == 1


class TestPltVarLat:
    def test_returns_figure(self, tiegcm_datasets):
        fig = plt_var_lat(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', longitude=30.0)
        assert fig is not None
        plt.close(fig)

    def test_zonal_mean_returns_figure(self, tiegcm_datasets):
        fig = plt_var_lat(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', longitude='mean')
        assert fig is not None
        plt.close(fig)

    def test_latitude_window_applied(self, tiegcm_datasets):
        fig = plt_var_lat(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', longitude=30.0,
                          latitude_minimum=-50, latitude_maximum=50)
        ax = fig.gca()
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(-50)
        assert xmax == pytest.approx(50)
        plt.close(fig)


class TestPltVarLon:
    def test_returns_figure(self, tiegcm_datasets):
        fig = plt_var_lon(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', latitude=2.5)
        assert fig is not None
        plt.close(fig)

    def test_meridional_mean_returns_figure(self, tiegcm_datasets):
        fig = plt_var_lon(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', latitude='mean')
        assert fig is not None
        plt.close(fig)

    def test_longitude_window_applied(self, tiegcm_datasets):
        fig = plt_var_lon(tiegcm_datasets, 'TN', level=1.0,
                          time='2003-03-20T00:00:00', latitude=2.5,
                          longitude_minimum=-90, longitude_maximum=90)
        ax = fig.gca()
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(-90)
        assert xmax == pytest.approx(90)
        plt.close(fig)
