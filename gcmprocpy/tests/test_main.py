"""Tests for main module (plot_routine) and plot functions."""
import sys
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock


class TestPlotRoutine:
    def test_invalid_plot_type(self, capsys):
        """plot_routine should log an error for invalid plot types."""
        import logging
        logging.basicConfig(level=logging.DEBUG)

        from gcmprocpy.main import plot_routine
        args = MagicMock()
        args.plot_type = 'invalid_type'

        plot_routine(args)
        # The error is logged, not printed

    def test_loads_datasets_from_cached(self, tiegcm_datasets):
        """plot_routine should use cached_datasets when provided."""
        from gcmprocpy.main import plot_routine
        args = MagicMock()
        args.plot_type = 'lat_lon'
        args.time = None
        args.mtime = None

        # This will call plt_lat_lon which needs proper args.
        # We just verify it doesn't try to load from directory.
        args.directory = None
        args.dataset = None

        with patch('gcmprocpy.main.plt_lat_lon') as mock_plot:
            mock_plot.return_value = MagicMock()
            plot_routine(args, cached_datasets=tiegcm_datasets, multiple_output=True)
            mock_plot.assert_called_once()

    def test_no_datasets_no_crash(self):
        """plot_routine with no datasets should not crash."""
        from gcmprocpy.main import plot_routine
        args = MagicMock()
        args.plot_type = 'lat_lon'
        args.time = None
        args.mtime = None
        args.directory = None
        args.dataset = None

        with patch('gcmprocpy.main.plt_lat_lon') as mock_plot:
            mock_plot.return_value = MagicMock()
            plot_routine(args)


class TestWindOverlay:
    def test_mercator_with_wind(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0,
                          wind=True, wind_density=1)
        assert fig is not None
        plt.close(fig)

    def test_orthographic_with_wind(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0,
                          projection='orthographic', wind=True, wind_density=1)
        assert fig is not None
        plt.close(fig)

    def test_mollweide_with_wind(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0,
                          projection='mollweide', wind=True, wind_density=1)
        assert fig is not None
        plt.close(fig)

    def test_north_polar_with_wind(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0,
                          projection='north_polar', wind=True, wind_density=1)
        assert fig is not None
        plt.close(fig)

    def test_no_wind_no_arrows(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0)
        assert fig is not None
        plt.close(fig)

    def test_wind_color_and_scale(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lat_lon
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', level=5.0,
                          wind=True, wind_density=1, wind_color='red', wind_scale=500)
        assert fig is not None
        plt.close(fig)


class TestPltLonTime:
    def test_creates_figure(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lon_time
        fig = plt_lon_time(tiegcm_datasets, 'TN', latitude=2.5, level=5.0)
        assert fig is not None
        plt.close(fig)

    def test_with_lat_mean(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lon_time
        fig = plt_lon_time(tiegcm_datasets, 'TN', latitude='mean', level=5.0)
        assert fig is not None
        plt.close(fig)

    def test_clean_plot(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_lon_time
        fig = plt_lon_time(tiegcm_datasets, 'TN', latitude=2.5, level=5.0, clean_plot=True)
        assert fig is not None
        plt.close(fig)


class TestPltVarTime:
    def test_creates_figure(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_var_time
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=30.0, level=5.0)
        assert fig is not None
        plt.close(fig)

    def test_without_level(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_var_time
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=30.0)
        assert fig is not None
        plt.close(fig)

    def test_clean_plot(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_var_time
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=30.0, level=5.0, clean_plot=True)
        assert fig is not None
        plt.close(fig)
