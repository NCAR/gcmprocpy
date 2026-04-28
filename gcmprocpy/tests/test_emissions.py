"""Tests for emissions functions in data_emissions module."""
import numpy as np
import pytest
from gcmprocpy.data_emissions import mkeno53, mkeco215, mkeoh83, arr_mkeno53, arr_mkeco215, arr_mkeoh83
from gcmprocpy.containers import PlotData


# ---------------------------------------------------------------------------
# Pure math functions
# ---------------------------------------------------------------------------

class TestMkeno53:
    def test_returns_positive_values(self):
        temp = np.array([500.0, 1000.0, 1500.0])
        o = np.array([1e9, 1e9, 1e9])
        no = np.array([1e7, 1e7, 1e7])
        result = mkeno53(temp, o, no)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_increases_with_temperature(self):
        o = np.full(5, 1e9)
        no = np.full(5, 1e7)
        temps = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0])
        result = mkeno53(temps, o, no)
        assert np.all(np.diff(result) > 0)

    def test_zero_oxygen_gives_zero(self):
        result = mkeno53(np.array([1000.0]), np.array([0.0]), np.array([1e7]))
        assert result[0] == 0.0

    def test_2d_array(self):
        temp = np.random.uniform(300, 1500, (4, 6))
        o = np.random.uniform(1e8, 1e10, (4, 6))
        no = np.random.uniform(1e5, 1e8, (4, 6))
        result = mkeno53(temp, o, no)
        assert result.shape == (4, 6)
        assert np.all(result > 0)


class TestMkeco215:
    def test_returns_positive_values(self):
        temp = np.array([500.0, 1000.0])
        o = np.array([1e9, 1e9])
        co2 = np.array([1e10, 1e10])
        result = mkeco215(temp, o, co2)
        assert result.shape == (2,)
        assert np.all(result > 0)

    def test_increases_with_temperature(self):
        o = np.full(5, 1e9)
        co2 = np.full(5, 1e10)
        temps = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0])
        result = mkeco215(temps, o, co2)
        assert np.all(np.diff(result) > 0)

    def test_zero_co2_gives_zero(self):
        result = mkeco215(np.array([1000.0]), np.array([1e9]), np.array([0.0]))
        assert result[0] == 0.0


class TestMkeoh83:
    def test_returns_positive_values(self):
        temp = np.array([200.0, 300.0])
        o = np.array([1e9, 1e9])
        o2 = np.array([1e12, 1e12])
        n2 = np.array([1e13, 1e13])
        result = mkeoh83(temp, o, o2, n2)
        assert result.shape == (2,)
        assert np.all(result > 0)

    def test_zero_oxygen_gives_zero(self):
        result = mkeoh83(np.array([300.0]), np.array([0.0]),
                         np.array([1e12]), np.array([1e13]))
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# Array / dataset functions
# ---------------------------------------------------------------------------

class TestArrMkeno53:
    def test_raw_mode_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeno53(tiegcm_datasets, 'NO53', time,
                             selected_lev_ilev=1.0, plot_mode=False)
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0)

    def test_plot_mode_returns_plotdata(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeno53(tiegcm_datasets, 'NO53', time,
                             selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'photons cm-3 sec-1'
        assert result.variable_long_name == '5.3-micron NO'
        assert result.values.shape == (len(result.lats), len(result.lons))

    def test_waccmx(self, waccmx_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeno53(waccmx_datasets, 'NO53', time,
                             selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.model == 'WACCM-X'


class TestArrMkeco215:
    def test_raw_mode_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeco215(tiegcm_datasets, 'CO215', time,
                              selected_lev_ilev=1.0, plot_mode=False)
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0)

    def test_plot_mode_returns_plotdata(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeco215(tiegcm_datasets, 'CO215', time,
                              selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'photons cm-3 sec-1'
        assert result.variable_long_name == '15-micron CO2'


class TestArrMkeoh83:
    def test_raw_mode_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeoh83(tiegcm_datasets, 'OH83', time,
                             selected_lev_ilev=1.0, plot_mode=False)
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0)

    def test_plot_mode_returns_plotdata(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkeoh83(tiegcm_datasets, 'OH83', time,
                             selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'photons cm-3 sec-1'
        assert result.variable_long_name == 'OH v(8,3)'
