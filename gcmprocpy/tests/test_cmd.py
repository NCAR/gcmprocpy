"""Tests for CLI command modules."""
import sys
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock


class TestCmdLatLon:
    """Tests for cmd_lat_lon CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lat_lon import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp'])
        assert args.filename == 'test'
        assert args.variable_name == 'TN'
        assert args.directory == '/tmp'

    def test_parser_defaults(self):
        from gcmprocpy.cmd.cmd_lat_lon import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp'])
        assert args.output_format == 'jpg'
        assert args.projection == 'mercator'
        assert args.center_longitude == 0
        assert args.central_latitude == 0
        assert args.wind is False
        assert args.wind_density == 15
        assert args.wind_scale is None
        assert args.wind_color == 'black'
        assert args.clean_plot is False
        assert args.coastlines is False
        assert args.nightshade is False
        assert args.gm_equator is False
        assert args.line_color == 'white'

    def test_parser_wind_args(self):
        from gcmprocpy.cmd.cmd_lat_lon import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-wind', '-wd', '5', '-ws', '300.0', '-wc', 'red'])
        assert args.wind is True
        assert args.wind_density == 5
        assert args.wind_scale == 300.0
        assert args.wind_color == 'red'

    def test_parser_projection_args(self):
        from gcmprocpy.cmd.cmd_lat_lon import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-proj', 'north_polar', '-clon', '45.0', '-clat', '30.0'])
        assert args.projection == 'north_polar'
        assert args.center_longitude == 45.0
        assert args.central_latitude == 30.0

    @patch('gcmprocpy.cmd.cmd_lat_lon.save_output')
    @patch('gcmprocpy.cmd.cmd_lat_lon.plt_lat_lon')
    @patch('gcmprocpy.cmd.cmd_lat_lon.load_datasets')
    def test_cmd_calls_plot_with_wind(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lat_lon import cmd_plt_lat_lon
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lat_lon', '-dir', '/tmp', '-o_file', 'out',
                                         '-var', 'TN', '-wind', '-proj', 'polar']):
            cmd_plt_lat_lon()
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs['wind'] is True
        assert call_kwargs['projection'] == 'polar'
        assert call_kwargs['variable_name'] == 'TN'
        mock_save.assert_called_once()


class TestCmdLevVar:
    """Tests for cmd_lev_var CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lev_var import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0'])
        assert args.variable_name == 'TN'
        assert args.latitude == 30.0

    def test_parser_optional_args(self):
        from gcmprocpy.cmd.cmd_lev_var import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0',
                                  '-t', '2022-01-01T12:00:00', '-lon', '45.0',
                                  '-lvl_min', '-5.0', '-lvl_max', '5.0'])
        assert args.time == '2022-01-01T12:00:00'
        assert args.longitude == 45.0
        assert args.level_minimum == -5.0
        assert args.level_maximum == 5.0

    @patch('gcmprocpy.cmd.cmd_lev_var.save_output')
    @patch('gcmprocpy.cmd.cmd_lev_var.plt_lev_var')
    @patch('gcmprocpy.cmd.cmd_lev_var.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lev_var import cmd_plt_lev_var
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lev_var', '-dir', '/tmp', '-o_file', 'out',
                                         '-o_format', 'png', '-var', 'TN', '-lat', '30.0']):
            cmd_plt_lev_var()
        mock_load.assert_called_once_with('/tmp', None)
        mock_plot.assert_called_once()
        mock_save.assert_called_once()


class TestCmdLevLon:
    """Tests for cmd_lev_lon CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lev_lon import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0'])
        assert args.variable_name == 'TN'
        assert args.latitude == 30.0

    @patch('gcmprocpy.cmd.cmd_lev_lon.save_output')
    @patch('gcmprocpy.cmd.cmd_lev_lon.plt_lev_lon')
    @patch('gcmprocpy.cmd.cmd_lev_lon.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lev_lon import cmd_plt_lev_lon
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lev_lon', '-dir', '/tmp', '-o_file', 'out',
                                         '-o_format', 'png', '-var', 'TN', '-lat', '30.0']):
            cmd_plt_lev_lon()
        mock_plot.assert_called_once()
        mock_save.assert_called_once()


class TestCmdLevLat:
    """Tests for cmd_lev_lat CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lev_lat import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp'])
        assert args.variable_name == 'TN'

    @patch('gcmprocpy.cmd.cmd_lev_lat.save_output')
    @patch('gcmprocpy.cmd.cmd_lev_lat.plt_lev_lat')
    @patch('gcmprocpy.cmd.cmd_lev_lat.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lev_lat import cmd_plt_lev_lat
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lev_lat', '-dir', '/tmp', '-o_file', 'out',
                                         '-o_format', 'png', '-var', 'TN']):
            cmd_plt_lev_lat()
        mock_plot.assert_called_once()
        mock_save.assert_called_once()


class TestCmdLevTime:
    """Tests for cmd_lev_time CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lev_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0'])
        assert args.variable_name == 'TN'
        assert args.latitude == 30.0

    def test_parser_time_range(self):
        from gcmprocpy.cmd.cmd_lev_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0',
                                  '--mtime_minimum', '1.0', '--mtime_maximum', '10.0'])
        assert args.mtime_minimum == 1.0
        assert args.mtime_maximum == 10.0

    @patch('gcmprocpy.cmd.cmd_lev_time.save_output')
    @patch('gcmprocpy.cmd.cmd_lev_time.plt_lev_time')
    @patch('gcmprocpy.cmd.cmd_lev_time.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lev_time import cmd_plt_lev_time
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lev_time', '-dir', '/tmp', '-o_file', 'out',
                                         '-o_format', 'png', '-var', 'TN', '-lat', '30.0']):
            cmd_plt_lev_time()
        mock_plot.assert_called_once()
        mock_save.assert_called_once()


class TestCmdLatTime:
    """Tests for cmd_lat_time CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lat_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-o_format', 'png', '-var', 'TN', '-dir', '/tmp'])
        assert args.variable_name == 'TN'

    @patch('gcmprocpy.cmd.cmd_lat_time.save_output')
    @patch('gcmprocpy.cmd.cmd_lat_time.plt_lat_time')
    @patch('gcmprocpy.cmd.cmd_lat_time.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lat_time import cmd_plt_lat_time
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lat_time', '-dir', '/tmp', '-o_file', 'out',
                                         '-o_format', 'png', '-var', 'TN']):
            cmd_plt_lat_time()
        mock_plot.assert_called_once()
        mock_save.assert_called_once()


class TestCmdLonTime:
    """Tests for cmd_lon_time CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_lon_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0'])
        assert args.variable_name == 'TN'
        assert args.latitude == 30.0

    def test_parser_defaults(self):
        from gcmprocpy.cmd.cmd_lon_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0'])
        assert args.contour_intervals == 10
        assert args.line_color == 'white'
        assert args.clean_plot is False
        assert args.longitude_minimum is None
        assert args.longitude_maximum is None

    def test_parser_all_args(self):
        from gcmprocpy.cmd.cmd_lon_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp', '-lat', '30.0',
                                  '-lvl', '5.0', '-ci', '20', '-lon_min', '-90.0', '-lon_max', '90.0',
                                  '--mtime_minimum', '1.0', '--mtime_maximum', '5.0', '-clean'])
        assert args.level == 5.0
        assert args.contour_intervals == 20
        assert args.longitude_minimum == -90.0
        assert args.longitude_maximum == 90.0
        assert args.mtime_minimum == 1.0
        assert args.mtime_maximum == 5.0
        assert args.clean_plot is True

    @patch('gcmprocpy.cmd.cmd_lon_time.save_output')
    @patch('gcmprocpy.cmd.cmd_lon_time.plt_lon_time')
    @patch('gcmprocpy.cmd.cmd_lon_time.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_lon_time import cmd_plt_lon_time
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['lon_time', '-dir', '/tmp', '-o_file', 'out',
                                         '-var', 'TN', '-lat', '30.0']):
            cmd_plt_lon_time()
        mock_load.assert_called_once_with('/tmp', None)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs['variable_name'] == 'TN'
        assert call_kwargs['latitude'] == 30.0
        mock_save.assert_called_once()


class TestCmdVarTime:
    """Tests for cmd_var_time CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_var_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-lat', '30.0', '-lon', '45.0'])
        assert args.variable_name == 'TN'
        assert args.latitude == 30.0
        assert args.longitude == 45.0

    def test_parser_defaults(self):
        from gcmprocpy.cmd.cmd_var_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-lat', '30.0', '-lon', '45.0'])
        assert args.level is None
        assert args.variable_unit is None
        assert args.mtime_minimum is None
        assert args.mtime_maximum is None
        assert args.clean_plot is False

    def test_parser_all_args(self):
        from gcmprocpy.cmd.cmd_var_time import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-lat', '30.0', '-lon', '45.0', '-lvl', '5.0',
                                  '-unit', 'K', '--mtime_minimum', '1.0',
                                  '--mtime_maximum', '10.0', '-clean'])
        assert args.level == 5.0
        assert args.variable_unit == 'K'
        assert args.mtime_minimum == 1.0
        assert args.mtime_maximum == 10.0
        assert args.clean_plot is True

    @patch('gcmprocpy.cmd.cmd_var_time.save_output')
    @patch('gcmprocpy.cmd.cmd_var_time.plt_var_time')
    @patch('gcmprocpy.cmd.cmd_var_time.load_datasets')
    def test_cmd_calls_plot(self, mock_load, mock_plot, mock_save):
        from gcmprocpy.cmd.cmd_var_time import cmd_plt_var_time
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['var_time', '-dir', '/tmp', '-o_file', 'out',
                                         '-var', 'TN', '-lat', '30.0', '-lon', '45.0']):
            cmd_plt_var_time()
        mock_load.assert_called_once_with('/tmp', None)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs['variable_name'] == 'TN'
        assert call_kwargs['latitude'] == 30.0
        assert call_kwargs['longitude'] == 45.0
        mock_save.assert_called_once()


class TestCmdSatTrack:
    """Tests for cmd_sat_track CLI."""

    def test_parser_required_args(self):
        from gcmprocpy.cmd.cmd_sat_track import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-sat_file', '/tmp/sat.csv'])
        assert args.variable_name == 'TN'
        assert args.satellite_file == '/tmp/sat.csv'

    def test_parser_defaults(self):
        from gcmprocpy.cmd.cmd_sat_track import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-sat_file', '/tmp/sat.csv'])
        assert args.level is None
        assert args.contour_intervals == 10
        assert args.line_color == 'white'
        assert args.clean_plot is False

    def test_parser_all_args(self):
        from gcmprocpy.cmd.cmd_sat_track import cmd_parser
        parser = cmd_parser()
        args = parser.parse_args(['-o_file', 'test', '-var', 'TN', '-dir', '/tmp',
                                  '-sat_file', '/tmp/sat.csv', '-lvl', '5.0',
                                  '-ci', '20', '-cmc', 'inferno', '-clean'])
        assert args.level == 5.0
        assert args.contour_intervals == 20
        assert args.cmap_color == 'inferno'
        assert args.clean_plot is True

    @patch('gcmprocpy.cmd.cmd_sat_track.save_output')
    @patch('gcmprocpy.cmd.cmd_sat_track.plt_sat_track')
    @patch('gcmprocpy.cmd.cmd_sat_track.load_datasets')
    def test_cmd_reads_csv_and_calls_plot(self, mock_load, mock_plot, mock_save, tmp_path):
        from gcmprocpy.cmd.cmd_sat_track import cmd_plt_sat_track
        # Create a test CSV file
        csv_file = tmp_path / 'sat.csv'
        csv_file.write_text(
            'time,lat,lon\n'
            '2003-03-20T00:00:00,30.0,45.0\n'
            '2003-03-20T00:30:00,35.0,50.0\n'
            '2003-03-20T01:00:00,40.0,55.0\n'
        )
        mock_load.return_value = 'datasets'
        mock_plot.return_value = 'figure'
        with patch.object(sys, 'argv', ['sat_track', '-dir', '/tmp', '-o_file', 'out',
                                         '-var', 'TN', '-sat_file', str(csv_file)]):
            cmd_plt_sat_track()
        mock_load.assert_called_once_with('/tmp', None)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs['variable_name'] == 'TN'
        assert len(call_kwargs['sat_time']) == 3
        assert len(call_kwargs['sat_lat']) == 3
        assert len(call_kwargs['sat_lon']) == 3
        assert call_kwargs['sat_lat'][0] == 30.0
        assert call_kwargs['sat_lon'][2] == 55.0
        mock_save.assert_called_once()
