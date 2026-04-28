"""Tests for the shared data cache and helpers added during perf work."""
import numpy as np
import pytest

from gcmprocpy import containers
from gcmprocpy.containers import (
    cache_data_fn, clear_data_cache, clear_derived_cache,
    _data_cache, _make_cache_key, resolve_derived, register_derived,
    DERIVED_VARIABLES,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_data_cache()
    yield
    clear_data_cache()


class TestCacheKey:
    def test_lists_normalized_to_tuples(self):
        ds = []
        k1 = _make_cache_key('fn', ds, (['a', 'b'],), {'v': [1, 2]})
        k2 = _make_cache_key('fn', ds, (('a', 'b'),), {'v': (1, 2)})
        assert k1 == k2
        hash(k1)

    def test_kwargs_order_independent(self):
        ds = []
        k1 = _make_cache_key('fn', ds, (), {'a': 1, 'b': 2})
        k2 = _make_cache_key('fn', ds, (), {'b': 2, 'a': 1})
        assert k1 == k2

    def test_different_datasets_different_key(self):
        a, b = [1], [2]
        assert _make_cache_key('fn', a, (), {})[1] != _make_cache_key('fn', b, (), {})[1]


class TestCacheDataFn:
    def test_caches_repeated_call(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, x):
            calls['n'] += 1
            return x * 2

        ds = []
        assert fn(ds, 5) == 10
        assert fn(ds, 5) == 10
        assert calls['n'] == 1

    def test_different_args_miss(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, x):
            calls['n'] += 1
            return x

        ds = []
        fn(ds, 1)
        fn(ds, 2)
        assert calls['n'] == 2

    def test_kwargs_included_in_key(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, x, mode='a'):
            calls['n'] += 1
            return (x, mode)

        ds = []
        assert fn(ds, 1, mode='a') == (1, 'a')
        assert fn(ds, 1, mode='b') == (1, 'b')
        assert calls['n'] == 2
        # repeat with mode='a' → cache hit
        fn(ds, 1, mode='a')
        assert calls['n'] == 2

    def test_unhashable_arg_falls_through(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, arr):
            calls['n'] += 1
            return arr.sum()

        ds = []
        arr = np.array([1, 2, 3])
        fn(ds, arr)
        fn(ds, arr)  # unhashable → no cache
        assert calls['n'] == 2

    def test_preserves_name_and_wrapped(self):
        @cache_data_fn
        def my_fn(datasets, x):
            return x

        assert my_fn.__name__ == 'my_fn'
        assert hasattr(my_fn, '__wrapped__')

    def test_different_datasets_separate_cache(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, x):
            calls['n'] += 1
            return x

        ds_a, ds_b = [1], [2]
        fn(ds_a, 5)
        fn(ds_b, 5)
        assert calls['n'] == 2

    def test_clear_data_cache_drops_entries(self):
        calls = {'n': 0}

        @cache_data_fn
        def fn(datasets, x):
            calls['n'] += 1
            return x

        ds = []
        fn(ds, 1)
        assert calls['n'] == 1
        clear_data_cache()
        fn(ds, 1)
        assert calls['n'] == 2

    def test_clear_derived_cache_alias(self):
        # Backwards-compat alias used by the GUI
        assert clear_derived_cache is clear_data_cache


class TestLRUEviction:
    def test_bounded_size(self, monkeypatch):
        monkeypatch.setattr(containers, '_DATA_CACHE_MAX', 3)

        @cache_data_fn
        def fn(datasets, x):
            return x * 10

        ds = []
        for i in range(5):
            fn(ds, i)
        assert len(_data_cache) == 3

    def test_eviction_order_is_lru(self, monkeypatch):
        monkeypatch.setattr(containers, '_DATA_CACHE_MAX', 2)

        @cache_data_fn
        def fn(datasets, x):
            return x

        ds = []
        fn(ds, 1)   # entries: [1]
        fn(ds, 2)   # entries: [1, 2]
        fn(ds, 1)   # touches 1 → [2, 1]
        fn(ds, 3)   # evicts 2 → [1, 3]
        keys = [k[2] for k in _data_cache.keys()]  # args tuple position
        assert (1,) in keys
        assert (3,) in keys
        assert (2,) not in keys


class TestResolveDerivedCaching:
    def test_cached_wrapper_returned(self):
        def _handler(datasets, variable_name, time, **kwargs):
            return ('computed', variable_name, time)

        register_derived('_TEST_CACHE_VAR', _handler)
        try:
            handler, found = resolve_derived('_TEST_CACHE_VAR')
            assert found
            assert handler is not _handler
            assert getattr(handler, '__wrapped__') is _handler
        finally:
            del DERIVED_VARIABLES['_TEST_CACHE_VAR']

    def test_derived_handler_actually_caches(self):
        calls = {'n': 0}

        def _handler(datasets, variable_name, time, **kwargs):
            calls['n'] += 1
            return ('computed', variable_name, time)

        register_derived('_TEST_CACHE_VAR2', _handler)
        try:
            handler, _ = resolve_derived('_TEST_CACHE_VAR2')
            ds = []
            t = np.datetime64('2003-03-20T00:00:00', 'ns')
            handler(ds, '_TEST_CACHE_VAR2', t)
            handler(ds, '_TEST_CACHE_VAR2', t)
            assert calls['n'] == 1
        finally:
            del DERIVED_VARIABLES['_TEST_CACHE_VAR2']


class TestArrLatLonCaching:
    """End-to-end: arr_lat_lon should memoize the same (datasets, var, time, lev)."""

    def test_second_call_is_cached(self, tiegcm_datasets):
        from gcmprocpy.data_parse import arr_lat_lon
        t = np.datetime64('2003-03-20T00:00:00', 'ns')
        r1 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=-3.0, plot_mode=True)
        r2 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=-3.0, plot_mode=True)
        # Same cached PlotData instance
        assert r1 is r2

    def test_different_level_separate_cache(self, tiegcm_datasets):
        from gcmprocpy.data_parse import arr_lat_lon
        t = np.datetime64('2003-03-20T00:00:00', 'ns')
        r1 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=-3.0, plot_mode=True)
        r2 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=1.0, plot_mode=True)
        assert r1 is not r2

    def test_clear_invalidates(self, tiegcm_datasets):
        from gcmprocpy.data_parse import arr_lat_lon
        t = np.datetime64('2003-03-20T00:00:00', 'ns')
        r1 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=-3.0, plot_mode=True)
        clear_data_cache()
        r2 = arr_lat_lon(tiegcm_datasets, 'TN', t, selected_lev_ilev=-3.0, plot_mode=True)
        assert r1 is not r2
        # Values still match
        np.testing.assert_array_equal(r1.values, r2.values)


class TestBatchArrLatLon:
    def test_returns_all_requested_vars(self, tiegcm_datasets):
        from gcmprocpy.data_parse import batch_arr_lat_lon
        t = np.datetime64('2003-03-20T00:00:00', 'ns')
        result = batch_arr_lat_lon(tiegcm_datasets, ['UN', 'VN'], t,
                                   selected_lev_ilev=-3.0, plot_mode=True)
        assert 'UN' in result and 'VN' in result
        assert result['UN'].values.shape == result['VN'].values.shape

    def test_matches_arr_lat_lon_singly(self, tiegcm_datasets):
        from gcmprocpy.data_parse import batch_arr_lat_lon, arr_lat_lon
        t = np.datetime64('2003-03-20T00:00:00', 'ns')
        single = arr_lat_lon(tiegcm_datasets, 'UN', t,
                             selected_lev_ilev=-3.0, plot_mode=True)
        batch = batch_arr_lat_lon(tiegcm_datasets, ['UN', 'VN'], t,
                                  selected_lev_ilev=-3.0, plot_mode=True)
        np.testing.assert_array_equal(single.values, batch['UN'].values)


class TestGeomagEquatorHelper:
    def test_one_value_per_longitude(self):
        from gcmprocpy.plot_gen import _compute_gm_equator_lats
        lons = np.linspace(-180, 180, 37)
        result = _compute_gm_equator_lats(lons)
        assert len(result) == len(lons)
        assert all(isinstance(x, float) for x in result)

    def test_deterministic(self):
        from gcmprocpy.plot_gen import _compute_gm_equator_lats
        lons = np.array([0.0, 90.0, 180.0, -90.0])
        a = _compute_gm_equator_lats(lons)
        b = _compute_gm_equator_lats(lons)
        assert a == b
