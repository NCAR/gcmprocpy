"""Shared fixtures and synthetic data builders for the gcmprocpy.imfgen test suite.

The whole offline suite runs without network: ``fake_omni`` writes tiny
``omni_min<year>.asc`` files in the exact 46-column layout the real parser
expects, ``fake_bcwind`` writes a small HDF5 file, and ``patch_download`` stubs
the FTP downloader. Opt-in ``live`` tests (real FTP) live in ``test_live.py``.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

MINUTES_PER_DAY = 60 * 24
N_COLS = 46

# Column indices used by the parser (mirrors gcmprocpy.imfgen.sources.OMNI_COLS).
COL = {"bx": 14, "by": 17, "bz": 18, "swvel": 21, "swden": 25}
# OMNI fill flags per channel (above the masking thresholds).
FILL = {"bx": 9999.99, "by": 9999.99, "bz": 9999.99, "swvel": 99999.9, "swden": 999.99}


def _channel_value(name, i):
    """Deterministic, smoothly varying, in-range channel value at minute ``i``."""
    base = {"bx": -5.0, "by": 3.0, "bz": -1.0, "swvel": 420.0, "swden": 6.0}[name]
    amp = {"bx": 4.0, "by": 4.0, "bz": 4.0, "swvel": 40.0, "swden": 3.0}[name]
    return round(base + amp * np.sin(i / 30.0), 4)


def build_omni_rows(year, n_days, missing=None, dead_tail=0):
    """Build an ``(n_days*1440, 46)`` OMNI row array for ``year``.

    ``missing`` : iterable of global minute indices whose 5 channels are set to
                  fill flags (masked -> NaN -> interpolated downstream).
    ``dead_tail``: number of trailing minutes set to all-fill across cols [5:]
                   (simulates end-of-data; triggers the edge-trim).
    Non-channel columns are a constant ``2.0`` (``2/9`` is not a fill repunit, so
    valid rows are never mis-flagged as all-fill).
    """
    missing = set(missing or [])
    n = n_days * MINUTES_PER_DAY
    rows = np.full((n, N_COLS), 2.0)
    for i in range(n):
        doy = i // MINUTES_PER_DAY + 1
        minute_of_day = i % MINUTES_PER_DAY
        rows[i, 0] = year
        rows[i, 1] = doy
        rows[i, 2] = minute_of_day // 60
        rows[i, 3] = minute_of_day % 60
        for name, col in COL.items():
            rows[i, col] = FILL[name] if i in missing else _channel_value(name, i)
    if dead_tail:
        rows[n - dead_tail:, 5:] = 9999.99  # all-fill -> row_all_fill True
    return rows


def write_omni_file(cache_dir, year, n_days, missing=None, dead_tail=0):
    path = os.path.join(cache_dir, f"omni_min{year}.asc")
    rows = build_omni_rows(year, n_days, missing=missing, dead_tail=dead_tail)
    np.savetxt(path, rows, fmt="%.4f")
    return path


@pytest.fixture
def fake_omni(tmp_path):
    """Factory: write OMNI files into a temp cache dir; returns ``(cache_dir, writer)``.

    The writer signature is ``writer(year, n_days, missing=None, dead_tail=0)``.
    """
    cache_dir = str(tmp_path / "omni_asc")
    os.makedirs(cache_dir, exist_ok=True)

    def writer(year, n_days, missing=None, dead_tail=0):
        return write_omni_file(cache_dir, year, n_days, missing=missing,
                               dead_tail=dead_tail)

    return cache_dir, writer


@pytest.fixture
def fake_bcwind(tmp_path):
    """Factory: write a small BCWIND HDF5 file; returns a writer.

    ``writer(n=120, start="2024-05-09 18:00:00", va_base=420.0)`` -> path.
    Set ``va_base`` high (>1e4) to exercise the swvel masking path.
    """
    import h5py

    def writer(n=120, start="2024-05-09 18:00:00", va_base=420.0):
        path = str(tmp_path / "bcwind.h5")
        t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        ut = [(t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S").encode()
              for i in range(n)]
        with h5py.File(path, "w") as fh:
            fh["Bx"] = np.array([_channel_value("bx", i) for i in range(n)])
            fh["By"] = np.array([_channel_value("by", i) for i in range(n)])
            fh["Bz"] = np.array([_channel_value("bz", i) for i in range(n)])
            fh["D"] = np.array([_channel_value("swden", i) for i in range(n)])
            fh["Va"] = np.array([va_base + i for i in range(n)], dtype=float)
            fh["UT"] = np.array(ut)
        return path

    return writer


@pytest.fixture
def patch_download(monkeypatch):
    """Stub ``sources.download_omni_files`` so no test ever hits the network."""
    from gcmprocpy.imfgen import sources

    calls = []

    def fake(years, cache_dir, verbose=False):
        calls.append(list(years))
        return []

    monkeypatch.setattr(sources, "download_omni_files", fake)
    return calls


# The --run-live / --run-golden options and the live/golden skip logic are
# registered once in the top-level tests/conftest.py (pytest only honours
# pytest_addoption in the rootdir conftest), so this subpackage conftest only
# provides fixtures.
