"""Tests for UT-aware solar-local-time (SLT) selection and the grid toggle."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcmprocpy.plot_gen import (
    local_time_to_longitude, longitude_to_local_time, _slice_ut,
    _resolve_localtime_longitude, plt_lev_lat, plt_lev_var, plt_var_lat, plt_lat_lon,
)


def _fig(r):
    return r[0] if isinstance(r, tuple) else r


# --- pure conversion math (UT-aware) -------------------------------------

def test_local_time_to_longitude_ut0():
    assert local_time_to_longitude(0) == 0
    assert local_time_to_longitude(6) == 90
    assert local_time_to_longitude(12) == 180      # exactly 180, no wrap
    assert local_time_to_longitude(18) == -90      # 270 wraps to -90
    assert local_time_to_longitude('mean') == 'mean'


def test_local_time_to_longitude_ut_aware():
    # Noon at UT=12 is the 0-degree meridian, not 180 (the whole thing was the bug).
    assert local_time_to_longitude(12, ut=12) == 0
    assert local_time_to_longitude(12, ut=1) == 165
    assert local_time_to_longitude('mean', ut=5) == 'mean'


def test_longitude_to_local_time_ut_aware():
    assert longitude_to_local_time(0) == 0
    assert longitude_to_local_time(180, ut=0) == 12
    assert longitude_to_local_time(0, ut=1) == 1      # UT shifts SLT
    assert longitude_to_local_time(180, ut=12) == 0


def test_lt_lon_roundtrip():
    for ut in (0, 1, 6.5, 23):
        for lt in (0, 3, 12, 18):
            lon = local_time_to_longitude(lt, ut)
            back = longitude_to_local_time(lon, ut)
            # equal modulo 24 (compare on the circle)
            assert abs(((back - lt + 12) % 24) - 12) < 1e-9


def test_slice_ut():
    assert _slice_ut(np.datetime64('2003-03-20T01:00:00')) == 1.0
    assert _slice_ut(np.datetime64('2003-03-20T00:00:00')) == 0.0
    assert _slice_ut(np.datetime64('2003-03-20T12:30:00')) == 12.5
    assert _slice_ut(None) == 0.0        # NaT-safe fallback
    assert _slice_ut('mean') == 0.0
    assert _slice_ut(np.datetime64('NaT')) == 0.0


# --- grid-snapped, UT-aware longitude resolution -------------------------

def test_resolve_localtime_longitude(tiegcm_datasets):
    t0 = np.datetime64('2003-03-20T00:00:00')   # UT=0
    t1 = np.datetime64('2003-03-20T01:00:00')   # UT=1
    grid = {-150.0, -90.0, -30.0, 30.0, 90.0, 150.0}   # fixture longitudes (step 30)

    assert _resolve_localtime_longitude(tiegcm_datasets, 'mean', t0) == 'mean'
    # lt chosen so (lt - ut)*15 lands exactly on the grid
    assert _resolve_localtime_longitude(tiegcm_datasets, 2, t0) == 30.0    # (2-0)*15
    assert _resolve_localtime_longitude(tiegcm_datasets, 6, t0) == 90.0
    assert _resolve_localtime_longitude(tiegcm_datasets, 3, t1) == 30.0    # (3-1)*15, UT-aware
    assert _resolve_localtime_longitude(tiegcm_datasets, 7, t1) == 90.0
    # every resolution lands on a real grid longitude
    for lt in np.arange(0, 24, 1.0):
        assert _resolve_localtime_longitude(tiegcm_datasets, lt, t0) in grid
    # UT-awareness: a 1-hour UT difference changes the resolved longitude for
    # at least some local times
    changed = [lt for lt in np.arange(0, 24, 0.5)
               if _resolve_localtime_longitude(tiegcm_datasets, lt, t0)
               != _resolve_localtime_longitude(tiegcm_datasets, lt, t1)]
    assert changed, "UT should affect the resolved longitude for some local times"


# --- functional: localtime selection renders on the single-time plots -----

@pytest.mark.parametrize("fn,kw", [
    (plt_lev_lat, {}),
    (plt_lev_var, {'latitude': 30.0}),
    (plt_var_lat, {'level': 5.0}),
])
def test_localtime_selection_renders(tiegcm_datasets, fn, kw):
    fig = _fig(fn(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', localtime=6, **kw))
    assert fig is not None
    plt.close('all')


def test_localtime_mean_renders(tiegcm_datasets):
    fig = _fig(plt_lev_lat(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00', localtime='mean'))
    assert fig is not None
    plt.close('all')


# --- grid toggle ----------------------------------------------------------

def _visible_gridlines(ax):
    return sum(l.get_visible() for l in list(ax.get_xgridlines()) + list(ax.get_ygridlines()))


def test_grid_toggle_plain(tiegcm_datasets):
    t = '2003-03-20T00:00:00'
    on = _fig(plt_var_lat(tiegcm_datasets, 'TN', level=5.0, time=t, longitude=30.0, grid=True))
    assert _visible_gridlines(on.axes[0]) > 0
    plt.close('all')
    off = _fig(plt_var_lat(tiegcm_datasets, 'TN', level=5.0, time=t, longitude=30.0, grid=False))
    assert _visible_gridlines(off.axes[0]) == 0
    plt.close('all')


def _count_gridliners(ax):
    from cartopy.mpl.gridliner import Gridliner
    return sum(isinstance(a, Gridliner) for a in ax.artists)


def test_grid_toggle_cartopy(tiegcm_datasets):
    # default projection 'mercator' -> PlateCarree branch: grid adds cartopy gridlines
    t = '2003-03-20T00:00:00'
    on = _fig(plt_lat_lon(tiegcm_datasets, 'TN', level=5.0, time=t, grid=True))
    off = _fig(plt_lat_lon(tiegcm_datasets, 'TN', level=5.0, time=t, grid=False))
    assert _count_gridliners(on.axes[0]) > _count_gridliners(off.axes[0])
    plt.close('all')
