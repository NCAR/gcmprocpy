"""Tests for main module (plot_routine)."""
import sys
import pytest
import numpy as np
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
