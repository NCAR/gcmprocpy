"""Tests for convert_units module."""
import numpy as np
import pytest
from gcmprocpy.convert_units import convert_units


class TestConvertUnits:
    """Tests for the convert_units function."""

    def test_same_unit_returns_unchanged(self):
        data = np.array([1.0, 2.0, 3.0])
        result, unit = convert_units(data, 'K', 'K')
        np.testing.assert_array_equal(result, data)
        assert unit == 'K'

    def test_simple_factor_conversion(self):
        data = np.array([100.0])
        result, unit = convert_units(data, 'cm/s', 'm/s')
        np.testing.assert_allclose(result, [1.0])
        assert unit == 'm/s'

    def test_cm_to_m(self):
        data = np.array([100.0, 200.0])
        result, unit = convert_units(data, 'cm', 'm')
        np.testing.assert_allclose(result, [1.0, 2.0])
        assert unit == 'm'

    def test_km_to_m(self):
        data = np.array([1.0])
        result, unit = convert_units(data, 'km', 'm')
        np.testing.assert_allclose(result, [1000.0])
        assert unit == 'm'

    def test_kelvin_to_celsius(self):
        data = np.array([273.15, 373.15])
        result, unit = convert_units(data, 'K', 'C')
        np.testing.assert_allclose(result, [0.0, 100.0])
        assert unit == 'C'

    def test_kelvin_to_fahrenheit(self):
        data = np.array([273.15])
        result, unit = convert_units(data, 'K', 'F')
        np.testing.assert_allclose(result, [32.0], atol=0.1)
        assert unit == 'F'

    def test_density_conversion(self):
        data = np.array([1e6])
        result, unit = convert_units(data, 'cm-3', 'm-3')
        np.testing.assert_allclose(result, [1e12])
        assert unit == 'm-3'

    def test_unknown_conversion_returns_original(self):
        data = np.array([1.0, 2.0])
        result, unit = convert_units(data, 'unknown_unit', 'm/s')
        np.testing.assert_array_equal(result, data)
        assert unit == 'unknown_unit'

    def test_scalar_input(self):
        result, unit = convert_units(5.0, 'km', 'm')
        assert result == 5000.0
        assert unit == 'm'

    def test_erg_to_joule(self):
        data = np.array([1e7])
        result, unit = convert_units(data, 'erg/g/s', 'J/kg/s')
        np.testing.assert_allclose(result, [1.0])
        assert unit == 'J/kg/s'
