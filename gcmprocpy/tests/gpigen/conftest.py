"""Shared fixtures and synthetic GFZ data builders for the gcmprocpy.gpigen test suite.

The whole suite runs OFFLINE: ``fake_gfz`` builds in-memory Fobs/Kp dicts in the
exact shape the real sources return, so the pipeline can be exercised without
touching the network. A single opt-in live test lives in ``test_live.py``.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

# Note: the --run-live option and the live/golden skip logic are registered once
# in the top-level tests/conftest.py (pytest only honours pytest_addoption in the
# rootdir conftest), so this subpackage conftest only provides fixtures.


def build_gfz(start="2023-11-22", n_days=160, f107_base=150.0,
              drop_fobs_days=None, kp_extra_days=0, kp_short_days=0,
              f107_sentinels=None):
    """Build synthetic ``(Fobs_data, Kp_data)`` dicts.

    Parameters
    ----------
    start : str
        First calendar day (``YYYY-MM-DD``).
    n_days : int
        Number of Fobs days produced (before any drops).
    drop_fobs_days : list[int] or None
        0-based day offsets to OMIT from Fobs (simulates calendar gaps).
    f107_sentinels : list[int] or None
        0-based day offsets to set to the ``-1`` missing sentinel.
    kp_extra_days : int
        Extra trailing days present in Kp but not Fobs (Kp ends later).
    kp_short_days : int
        Trailing days present in Fobs but not Kp (Fobs ends later).
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    drop = set(drop_fobs_days or [])
    sentinels = set(f107_sentinels or [])

    fobs_dt, fobs_val = [], []
    for i in range(n_days):
        if i in drop:
            continue
        day = start_dt + timedelta(days=i)
        fobs_dt.append(day.strftime("%Y-%m-%dT00:00:00Z"))
        if i in sentinels:
            fobs_val.append(-1.0)
        else:
            # Deterministic, smoothly varying signal.
            fobs_val.append(f107_base + 10.0 * np.sin(i / 12.0))

    n_kp = n_days + kp_extra_days - kp_short_days
    kp_dt, kp_val = [], []
    for i in range(n_kp):
        day = start_dt + timedelta(days=i)
        for hour in range(0, 24, 3):
            stamp = (day + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:00:00Z")
            kp_dt.append(stamp)
            kp_val.append(round((i % 9) * 0.333, 3))

    return (
        {"datetime": fobs_dt, "Fobs": fobs_val},
        {"datetime": kp_dt, "Kp": kp_val},
    )


@pytest.fixture
def fake_gfz():
    """Factory fixture: call with the same kwargs as :func:`build_gfz`."""
    return build_gfz


@pytest.fixture
def patch_fetch(monkeypatch):
    """Patch :func:`gcmprocpy.gpigen.sources.fetch` (and the copy imported into core).

    Returns a setter; pass it the ``(Fobs_data, Kp_data)`` tuple to serve. The
    patched fetch records the call args on ``.calls`` for assertions.
    """
    state = {"data": None, "calls": []}

    def fake_fetch(source, start_dt, end_dt, status="def", cache_dir=None, verbose=False):
        state["calls"].append(
            dict(source=source, start=start_dt, end=end_dt, status=status)
        )
        return state["data"]

    monkeypatch.setattr("gcmprocpy.gpigen.sources.fetch", fake_fetch)
    monkeypatch.setattr("gcmprocpy.gpigen.core.fetch", fake_fetch)

    def setter(data):
        state["data"] = data
        return state

    setter.state = state
    return setter
