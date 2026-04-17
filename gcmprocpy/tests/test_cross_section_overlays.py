"""Tests for wind / EP-flux overlays on plt_lev_lat and plt_lev_lon."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcmprocpy.plot_gen import plt_lev_lat, plt_lev_lon


def _count_quivers(fig):
    return sum(1 for ax in fig.axes
               for c in ax.collections
               if type(c).__name__ == 'Quiver')


class TestPltLevLatWindOverlay:
    def test_no_overlay_by_default(self, tiegcm_datasets):
        fig = plt_lev_lat(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          longitude=30.0)
        assert fig is not None
        assert _count_quivers(fig) == 0
        plt.close(fig)

    def test_wind_overlay_adds_quiver(self, tiegcm_datasets):
        fig = plt_lev_lat(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          longitude=30.0, wind=True, wind_density=2)
        assert _count_quivers(fig) == 1
        plt.close(fig)

    def test_epflux_overlay_adds_quiver(self, tiegcm_datasets):
        fig = plt_lev_lat(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          longitude=30.0, epflux=True, wind_density=2)
        assert _count_quivers(fig) == 1
        plt.close(fig)

    def test_wind_with_custom_color(self, tiegcm_datasets):
        fig = plt_lev_lat(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          longitude=30.0, wind=True, wind_density=2,
                          wind_color='red', wind_scale=1000)
        assert _count_quivers(fig) == 1
        plt.close(fig)


class TestPltLevLonWindOverlay:
    def test_no_overlay_by_default(self, tiegcm_datasets):
        fig = plt_lev_lon(tiegcm_datasets, 'TN', latitude=2.5,
                          time='2003-03-20T00:00:00')
        assert _count_quivers(fig) == 0
        plt.close(fig)

    def test_wind_overlay_adds_quiver(self, tiegcm_datasets):
        fig = plt_lev_lon(tiegcm_datasets, 'TN', latitude=2.5,
                          time='2003-03-20T00:00:00',
                          wind=True, wind_density=2)
        assert _count_quivers(fig) == 1
        plt.close(fig)

    def test_wind_with_custom_color(self, tiegcm_datasets):
        fig = plt_lev_lon(tiegcm_datasets, 'TN', latitude=2.5,
                          time='2003-03-20T00:00:00',
                          wind=True, wind_density=2,
                          wind_color='cyan', wind_scale=500)
        assert _count_quivers(fig) == 1
        plt.close(fig)
