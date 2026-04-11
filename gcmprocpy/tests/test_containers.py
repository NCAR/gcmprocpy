"""Tests for derived variable registry in containers module."""
import pytest
from gcmprocpy.containers import (
    get_species_names, register_derived, resolve_derived, DERIVED_VARIABLES,
)


class TestGetSpeciesNames:
    def test_tiegcm_returns_correct_mapping(self):
        sp = get_species_names('TIE-GCM')
        assert sp['temp'] == 'TN'
        assert sp['o'] == 'O1'
        assert sp['o2'] == 'O2'
        assert sp['n2'] == 'N2'
        assert sp['no'] == 'NO'
        assert sp['co2'] == 'CO2'

    def test_waccmx_returns_correct_mapping(self):
        sp = get_species_names('WACCM-X')
        assert sp['temp'] == 'T'
        assert sp['o'] == 'O'
        assert sp['o2'] == 'O2'
        assert sp['n2'] == 'N2'

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_species_names('FAKE-MODEL')


class TestResolvedDerived:
    def test_exact_match(self):
        handler, found = resolve_derived('NO53')
        assert found is True
        assert handler is not None

    def test_case_insensitive(self):
        handler, found = resolve_derived('no53')
        assert found is True

    def test_pattern_match_oh(self):
        handler, found = resolve_derived('OH_8_3')
        assert found is True

    def test_pattern_match_oh_total(self):
        handler, found = resolve_derived('OH_TOTAL')
        assert found is True

    def test_pattern_match_oh_vib(self):
        handler, found = resolve_derived('OH_VIB_5')
        assert found is True

    def test_epflux_components(self):
        for name in ('EPVY', 'EPVZ', 'EPVDIV'):
            handler, found = resolve_derived(name)
            assert found is True, f"{name} should resolve"

    def test_regular_variable_not_derived(self):
        handler, found = resolve_derived('TN')
        assert found is False
        assert handler is None

    def test_empty_string_not_derived(self):
        handler, found = resolve_derived('')
        assert found is False


class TestRegisterDerived:
    def test_register_and_resolve(self):
        def _dummy(datasets, variable_name, time, **kwargs):
            return None

        register_derived('_TEST_VAR', _dummy)
        handler, found = resolve_derived('_TEST_VAR')
        assert found is True
        assert handler is _dummy

        # Cleanup
        del DERIVED_VARIABLES['_TEST_VAR']

    def test_register_with_plot_types(self):
        def _dummy2(datasets, variable_name, time, **kwargs):
            return None

        register_derived('_TEST_VAR2', _dummy2, plot_types={'lat_lon'})
        assert DERIVED_VARIABLES['_TEST_VAR2']['plot_types'] == {'lat_lon'}

        # Cleanup
        del DERIVED_VARIABLES['_TEST_VAR2']
