"""Tests for polar perimeter ring labels (LT / longitude) on plt_lat_lon."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcmprocpy.plot_gen import plt_lat_lon, _polar_ring_labels


class TestPolarRingLabels:
    def test_lt_labels_default(self, tiegcm_datasets):
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='north_polar')
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # 12 ring labels + footer caption
        assert any('SOLAR LOCAL TIME' in t for t in texts)
        # All hour labels are 2-digit strings 00..23
        hour_labels = [t for t in texts if t.isdigit() and len(t) == 2]
        assert len(hour_labels) == 12
        for h in hour_labels:
            assert 0 <= int(h) <= 23
        plt.close(fig)

    def test_lon_labels(self, tiegcm_datasets):
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='north_polar', polar_label='lon')
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any('GEOGRAPHIC LONGITUDE' in t for t in texts)
        # Longitude labels span -180..150 step 30
        expected = {str(l) for l in range(-180, 180, 30)}
        present = {t for t in texts if t.lstrip('-').isdigit()}
        assert expected.issubset(present)
        plt.close(fig)

    def test_no_ring_when_disabled(self, tiegcm_datasets):
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='north_polar', polar_label=None)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert not any('SOLAR LOCAL TIME' in t for t in texts)
        assert not any('GEOGRAPHIC LONGITUDE' in t for t in texts)
        plt.close(fig)

    def test_lt_at_ut_zero(self, tiegcm_datasets):
        # At UT=00, lon=0 should map to LT=00 and lon=180 to LT=12
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='north_polar', polar_label='lt')
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Expected ring labels for UT=0: 18,20,22,00,02,04,06,08,10,12,14,16
        # (lon -180..150 step 30 → lt = (0 + lon/15) mod 24)
        expected = {f"{int(round((lon / 15.0) % 24)) % 24:02d}"
                    for lon in range(-180, 180, 30)}
        hour_labels = {t for t in texts if t.isdigit() and len(t) == 2}
        assert expected == hour_labels
        plt.close(fig)

    def test_south_polar_lt(self, tiegcm_datasets):
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='south_polar')
        # south panel is the only ax for south_polar
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any('SOLAR LOCAL TIME' in t for t in texts)
        plt.close(fig)

    def test_dual_polar_both_panels(self, tiegcm_datasets):
        fig = plt_lat_lon(tiegcm_datasets, 'TN', time='2003-03-20T00:00:00',
                          level=5.0, projection='polar')
        # Two panels, both should have a footer
        captions = []
        for ax in fig.axes[:2]:
            captions.extend(t.get_text() for t in ax.texts
                            if 'LOCAL TIME' in t.get_text())
        assert len(captions) == 2
        plt.close(fig)
