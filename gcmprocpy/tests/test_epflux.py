"""Tests for Eliassen-Palm flux functions in data_epflux module."""
import numpy as np
import pytest
from gcmprocpy.data_epflux import epflux, arr_epflux
from gcmprocpy.containers import PlotData


# ---------------------------------------------------------------------------
# Pure physics function
# ---------------------------------------------------------------------------

class TestEpflux:
    @pytest.fixture
    def synthetic_fields(self):
        """Create synthetic 3-D fields with a simple wave pattern."""
        nlev, nlat, nlon = 8, 12, 24
        lats = np.linspace(-87.5, 87.5, nlat)
        levs = np.linspace(-7, 7, nlev)
        lons = np.linspace(0, 360 - 360 / nlon, nlon)

        np.random.seed(123)
        # Zonal-mean background + wave perturbation
        temp = 300 + 50 * np.random.randn(nlev, nlat, nlon)
        u = 20 * np.sin(np.deg2rad(lats))[np.newaxis, :, np.newaxis] + \
            5 * np.random.randn(nlev, nlat, nlon)
        v = 3 * np.random.randn(nlev, nlat, nlon)
        w = 0.01 * np.random.randn(nlev, nlat, nlon)

        return dict(temp=temp, u=u, v=v, lats=lats, levs=levs, w=w)

    def test_epvy_shape(self, synthetic_fields):
        result = epflux(**synthetic_fields)
        nlev, nlat = synthetic_fields['temp'].shape[:2]
        assert result['EPVY'].shape == (nlev, nlat)

    def test_epvz_shape(self, synthetic_fields):
        result = epflux(**synthetic_fields)
        nlev, nlat = synthetic_fields['temp'].shape[:2]
        assert result['EPVZ'].shape == (nlev, nlat)

    def test_epvdiv_shape(self, synthetic_fields):
        result = epflux(**synthetic_fields)
        nlev, nlat = synthetic_fields['temp'].shape[:2]
        assert result['EPVDIV'].shape == (nlev, nlat)

    def test_without_w_only_epvy(self, synthetic_fields):
        del synthetic_fields['w']
        result = epflux(**synthetic_fields)
        assert result['EPVY'] is not None
        assert result['EPVZ'] is None
        assert result['EPVDIV'] is None

    def test_epvy_finite(self, synthetic_fields):
        result = epflux(**synthetic_fields)
        assert np.all(np.isfinite(result['EPVY']))

    def test_epvz_finite(self, synthetic_fields):
        result = epflux(**synthetic_fields)
        assert np.all(np.isfinite(result['EPVZ']))

    def test_zero_perturbation_gives_small_ep_flux(self):
        """If all fields are zonally uniform, eddy fluxes should be zero."""
        nlev, nlat, nlon = 6, 8, 12
        lats = np.linspace(-80, 80, nlat)
        levs = np.linspace(-6, 6, nlev)

        # Zonally uniform fields (no eddies)
        temp = np.ones((nlev, nlat, nlon)) * 500.0
        u = np.ones((nlev, nlat, nlon)) * 10.0
        v = np.zeros((nlev, nlat, nlon))
        w = np.zeros((nlev, nlat, nlon))

        result = epflux(temp, u, v, lats, levs, w=w)
        # With no eddies, u'v' = 0 and v'T' = 0
        np.testing.assert_allclose(result['EPVY'], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Array / dataset functions
# ---------------------------------------------------------------------------

class TestArrEpflux:
    def test_epvy_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(tiegcm_datasets, 'EPVY', time)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'm² s⁻²'
        assert 'meridional' in result.variable_long_name.lower()
        assert result.values.ndim == 2  # (nlev, nlat)

    def test_epvz_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(tiegcm_datasets, 'EPVZ', time)
        assert isinstance(result, PlotData)
        assert result.variable_unit == 'm² s⁻²'

    def test_epvdiv_tiegcm(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(tiegcm_datasets, 'EPVDIV', time)
        assert isinstance(result, PlotData)
        assert 'day' in result.variable_unit

    def test_waccmx_epvy(self, waccmx_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(waccmx_datasets, 'EPVY', time)
        assert isinstance(result, PlotData)
        assert result.model == 'WACCM-X'

    def test_invalid_component_raises(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        with pytest.raises(ValueError, match="component must be"):
            arr_epflux(tiegcm_datasets, 'FAKE', time)

    def test_missing_time_returns_none(self, tiegcm_datasets):
        result = arr_epflux(tiegcm_datasets, 'EPVY', '2099-01-01T00:00:00')
        assert result is None

    def test_case_insensitive(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(tiegcm_datasets, 'epvy', time)
        assert isinstance(result, PlotData)

    def test_plotdata_has_lats_and_levs(self, tiegcm_datasets):
        time = '2003-03-20T00:00:00'
        result = arr_epflux(tiegcm_datasets, 'EPVY', time)
        assert result.lats is not None
        assert result.levs is not None
        assert len(result.lats) == result.values.shape[1]
