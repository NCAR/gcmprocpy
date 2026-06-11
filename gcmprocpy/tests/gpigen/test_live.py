"""Opt-in live tests that hit the real GFZ Potsdam endpoints.

Skipped by default. Enable with::

    pytest -m live --run-live

These verify the real wire formats and that the two sources agree on real data.
"""

import numpy as np
import pytest

from gcmprocpy.gpigen import generate_gpi

pytestmark = pytest.mark.live


def test_live_json_small_range():
    ds = generate_gpi(start="2024-01-01", end="2024-04-30", source="json")
    assert int(ds.attrs["yearday_beg"]) == 2024001
    assert int(ds.attrs["yearday_end"]) == 2024081
    assert ds["kp"].shape[1] == 8
    assert np.all(ds["f107a"].values != 0)


def test_live_sources_agree():
    dj = generate_gpi(start="2024-01-01", end="2024-04-30", source="json")
    dt = generate_gpi(start="2024-01-01", end="2024-04-30", source="textfile")
    assert list(dj["year_day"].values) == list(dt["year_day"].values)
    assert np.max(np.abs(dj["f107d"].values - dt["f107d"].values)) < 1e-9
    assert np.max(np.abs(dj["f107a"].values - dt["f107a"].values)) < 1e-9
    assert np.max(np.abs(dj["kp"].values - dt["kp"].values)) < 1e-9
