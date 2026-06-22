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

    def test_rho_argument_affects_epvdiv(self, synthetic_fields):
        """Passing an explicit rho must reach EPVDIV (changes vs proxy)."""
        baseline = epflux(**synthetic_fields)
        nlev, nlat, nlon = synthetic_fields['temp'].shape
        np.random.seed(7)
        rho_field = np.random.uniform(1e-7, 1e-3, (nlev, nlat, nlon))
        with_rho = epflux(**synthetic_fields, rho=rho_field)
        # EPVY/EPVZ shouldn't change (rho only enters EPVDIV)
        np.testing.assert_allclose(baseline['EPVY'], with_rho['EPVY'])
        np.testing.assert_allclose(baseline['EPVZ'], with_rho['EPVZ'])
        # EPVDIV should change
        assert not np.allclose(baseline['EPVDIV'], with_rho['EPVDIV'])

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


# ---------------------------------------------------------------------------
# EPVDIV mass density must carry the pkt = p/(k_B·T) vertical falloff.
# Regression for the bug where mixing-ratio species were treated as cm⁻³,
# dropping the pkt factor (so EPVDIV's (1/ρ)∂(ρ·Sz)/∂z was wrong).
# ---------------------------------------------------------------------------

class TestEpvdivMassDensity:
    TIME = '2003-03-20T00:00:00'

    @staticmethod
    def _mmr_tiegcm(time=TIME):
        """A TIE-GCM dataset with MMR major species (N2 = 1 - O2 - O1)."""
        import xarray as xr
        from gcmprocpy.containers import ModelDataset
        t = np.array([time], dtype='datetime64[ns]')
        lat = np.array([-30., 0., 30.])
        lon = np.array([0., 120., 240.])
        lev = np.array([-3., 0., 3., 6.])   # ln(p0/p): increasing → lower pressure
        shp = (1, len(lev), len(lat), len(lon))
        rng = np.random.default_rng(7)
        o1 = rng.uniform(0.1, 0.3, shp)
        o2 = rng.uniform(0.05, 0.2, shp)
        n2 = 1.0 - o1 - o2                  # residual MMR → species sum to 1
        ds = xr.Dataset(
            {
                'TN': (['time', 'lev', 'lat', 'lon'], rng.uniform(200., 1200., shp), {'units': 'K'}),
                'O1': (['time', 'lev', 'lat', 'lon'], o1, {'units': 'mmr'}),
                'O2': (['time', 'lev', 'lat', 'lon'], o2, {'units': 'mmr'}),
                'N2': (['time', 'lev', 'lat', 'lon'], n2, {'units': 'mmr'}),
                'UN': (['time', 'lev', 'lat', 'lon'], rng.uniform(-50, 50, shp), {'units': 'cm/s'}),
                'VN': (['time', 'lev', 'lat', 'lon'], rng.uniform(-30, 30, shp), {'units': 'cm/s'}),
                'W':  (['time', 'lev', 'lat', 'lon'], rng.uniform(-1e-4, 1e-4, shp), {'units': 's-1'}),
                'mtime': (['time', 'mtimedim'], np.array([[80, 0, 0, 0]])),
            },
            coords={'time': t, 'lat': lat, 'lon': lon,
                    'lev': xr.DataArray(lev, dims='lev', attrs={'units': 'ln(p0/p)'})},
        )
        return ModelDataset(ds=ds, filename='mmr.nc', model='TIE-GCM'), time

    def test_mass_density_matches_pkt_barm_amu(self):
        # With N2 the residual (species sum = 1), Σ_species(MMR→GM/CM3) reduces
        # to pkt·barm·m_u; ×1000 → kg m⁻³.  The pre-fix code omitted pkt.
        from gcmprocpy.data_epflux import _total_mass_density_kg_m3
        from gcmprocpy.data_density import compute_pkt, compute_barm, _AMU_G
        mds, time = self._mmr_tiegcm()
        rho = _total_mass_density_kg_m3(mds, time)
        dst = mds.ds.sel(time=time)
        pkt = compute_pkt(mds.ds['lev'].values, dst['TN'].values, model='TIE-GCM')
        barm = compute_barm(dst['O1'].values, dst['O2'].values)
        expected = pkt * barm * _AMU_G * 1.0e3
        np.testing.assert_allclose(rho, expected, rtol=1e-9)

    def test_mass_density_falls_off_with_altitude(self):
        # pkt ∝ exp(-lev) must dominate: density drops orders of magnitude from
        # the lowest to the highest lev.  The pre-fix code produced no falloff.
        from gcmprocpy.data_epflux import _total_mass_density_kg_m3
        mds, time = self._mmr_tiegcm()
        col = _total_mass_density_kg_m3(mds, time).mean(axis=(1, 2))
        assert col[0] > col[-1] * 50

    def test_arr_epvdiv_finite_on_mmr_history(self):
        mds, time = self._mmr_tiegcm()
        result = arr_epflux([mds], 'EPVDIV', time)
        assert result is not None
        assert np.all(np.isfinite(result.values))
