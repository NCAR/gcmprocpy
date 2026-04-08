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
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=0.0, level=5.0)
        assert fig is not None
        plt.close(fig)

    def test_without_level(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_var_time
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=0.0)
        assert fig is not None
        plt.close(fig)

    def test_clean_plot(self, tiegcm_datasets):
        from gcmprocpy.plot_gen import plt_var_time
        fig = plt_var_time(tiegcm_datasets, 'TN', latitude=2.5, longitude=0.0, level=5.0, clean_plot=True)
        assert fig is not None
        plt.close(fig)
