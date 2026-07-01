"""Golden regression tests: gpigen output vs the on-disk reference NetCDF files.

Opt-in (the reference files are large / machine-specific and generation hits the
live GFZ service)::

    pytest -m golden --run-golden

Each test skips cleanly if its reference file is absent, or if the live GFZ
fetch fails, so the suite still passes on machines without data or network.
"""

import os

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gcmprocpy.gpigen import generate_gpi
from gcmprocpy.gpigen.indices import kp_to_ap

pytestmark = pytest.mark.golden

TIEGCM_GPI_REF = "/glade/campaign/hao/itmodel/tiegcm3.0/GPI/gpi_1960001-2024047.nc"
WACCMX_GPI_REF = ("/glade/campaign/cesm/development/wawg/joemci/inputdata/atm/"
                  "waccm/solar/wax_solar_3hr_1960-Feb2026_c260504.nc")


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"reference data not present: {path}")


def _generate(**kwargs):
    try:
        return generate_gpi(**kwargs)
    except (RuntimeError, OSError) as exc:      # GFZ unreachable / fetch failure
        pytest.skip(f"live GFZ fetch failed: {exc}")


def test_tiegcm_gpi_matches_reference():
    """Generated TIE-GCM GPI reproduces the reference kp/f107 for overlapping days."""
    _require(TIEGCM_GPI_REF)
    ours = _generate(start="2024-01-01", end="2024-01-20", window=1, centered=False)
    g = xr.open_dataset(TIEGCM_GPI_REF)
    try:
        mask = np.isin(g["year_day"].values, ours["year_day"].values)
        assert int(mask.sum()) == ours.sizes["ndays"]
        assert np.array_equal(ours["kp"].values, g["kp"].values[mask])
        assert np.array_equal(ours["f107d"].values, g["f107d"].values[mask])
    finally:
        g.close()


def test_waccmx_gpi_schema_matches_reference():
    """model='waccmx' output matches joemci's reference WACCM-X GPI schema.

    Variable set, dtypes and long_names all match (with the standard 81-day
    centered f107a window); ``ap`` is the Kp->ap lookup, as in the reference.
    """
    _require(WACCMX_GPI_REF)
    ours = _generate(start="2024-01-01", end="2024-04-15", window=81, centered=True,
                     model="waccmx")
    g = xr.open_dataset(WACCMX_GPI_REF, decode_times=False)
    try:
        assert set(ours.data_vars) == set(g.data_vars)
        for v in ours.data_vars:
            assert str(ours[v].dtype) == str(g[v].dtype), f"dtype {v}"
            assert ours[v].attrs.get("long_name") == g[v].attrs.get("long_name"), \
                f"long_name {v}"
        assert "time" in ours.dims
        # ap is the exact Kp->ap table applied to the file's own kp
        assert np.array_equal(ours["ap"].values,
                              kp_to_ap(ours["kp"].values).astype(float))
    finally:
        g.close()
