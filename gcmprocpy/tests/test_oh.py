"""Tests for OH Meinel band model in data_oh module."""
import numpy as np
import pytest
from gcmprocpy.data_oh import ohrad, arr_mkoh_band, OH_BANDS
from gcmprocpy.containers import PlotData


# ---------------------------------------------------------------------------
# Pure physics function
# ---------------------------------------------------------------------------

class TestOhrad:
    @pytest.fixture
    def typical_inputs(self):
        """Physically plausible mesopause-region values."""
        shape = (4, 6)
        np.random.seed(99)
        return dict(
            temp=np.random.uniform(180, 220, shape),
            o2=np.random.uniform(1e12, 5e13, shape),
            o=np.random.uniform(1e9, 1e11, shape),
            n2=np.random.uniform(1e13, 5e14, shape),
            h=np.random.uniform(1e6, 1e8, shape),
            o3=np.random.uniform(1e7, 1e9, shape),
            ho2=np.random.uniform(1e5, 1e7, shape),
        )

    def test_returns_vib_pop_and_band_emission(self, typical_inputs):
        vib_pop, band_em = ohrad(**typical_inputs)
        assert vib_pop.shape == (4, 6, 10)  # 10 vibrational levels
        assert isinstance(band_em, dict)
        assert len(band_em) == len(OH_BANDS)

    def test_vib_populations_non_negative(self, typical_inputs):
        vib_pop, _ = ohrad(**typical_inputs)
        assert np.all(vib_pop >= 0)

    def test_band_emissions_non_negative(self, typical_inputs):
        _, band_em = ohrad(**typical_inputs)
        for key, val in band_em.items():
            assert np.all(val >= 0), f"Band {key} has negative values"

    def test_all_39_bands_present(self, typical_inputs):
        _, band_em = ohrad(**typical_inputs)
        assert len(band_em) == 39

    def test_higher_v_has_lower_population(self, typical_inputs):
        """In general, higher vibrational levels should be less populated."""
        vib_pop, _ = ohrad(**typical_inputs)
        # Compare mean population of v=1 vs v=9
        mean_v1 = vib_pop[..., 1].mean()
        mean_v9 = vib_pop[..., 9].mean()
        assert mean_v1 > mean_v9

    def test_scalar_inputs(self):
        """Single-point inputs should work."""
        vib_pop, band_em = ohrad(
            temp=np.array(200.0), o2=np.array(1e13), o=np.array(1e10),
            n2=np.array(1e14), h=np.array(1e7), o3=np.array(1e8),
            ho2=np.array(1e6),
        )
        assert vib_pop.shape[-1] == 10


# ---------------------------------------------------------------------------
# Array / dataset functions
# ---------------------------------------------------------------------------

class TestArrMkohBand:
    def test_specific_band_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkoh_band(tiegcm_datasets, 'OH_8_3', time,
                               selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert 'OH(8-3)' in result.variable_long_name
        assert result.variable_unit == 'photons cm-3 sec-1'
        assert np.all(result.values >= 0)

    def test_total_emission(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkoh_band(tiegcm_datasets, 'OH_TOTAL', time,
                               selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert 'total' in result.variable_long_name.lower()

    def test_vib_population(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkoh_band(tiegcm_datasets, 'OH_VIB_5', time,
                               selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'cm-3'
        assert 'v=5' in result.variable_long_name

    def test_raw_mode(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkoh_band(tiegcm_datasets, 'OH_8_3', time,
                               selected_lev_ilev=1.0, plot_mode=False)
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)

    def test_waccmx(self, waccmx_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_mkoh_band(waccmx_datasets, 'OH_8_3', time,
                               selected_lev_ilev=1.0, plot_mode=True)
        assert isinstance(result, PlotData)
        assert result.model == 'WACCM-X'

    def test_invalid_band_raises(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        with pytest.raises(ValueError, match="not valid"):
            arr_mkoh_band(tiegcm_datasets, 'OH_9_9', time,
                          selected_lev_ilev=1.0, plot_mode=False)

    def test_invalid_variable_raises(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        with pytest.raises(ValueError, match="Unrecognized"):
            arr_mkoh_band(tiegcm_datasets, 'OH_BOGUS', time,
                          selected_lev_ilev=1.0, plot_mode=False)

    def test_invalid_vib_level_raises(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        with pytest.raises(ValueError, match="must be 0-9"):
            arr_mkoh_band(tiegcm_datasets, 'OH_VIB_15', time,
                          selected_lev_ilev=1.0, plot_mode=False)
