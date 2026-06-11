"""Opt-in live tests that hit the real OMNI FTP endpoint (SPDF).

Skipped by default. Enable with::

    pytest -m live --run-live

These download one real ``omni_min<year>.asc`` file into a temp dir and run a
small slice through the pipeline, validating the real wire format end to end.
"""

import numpy as np
import pytest

from gcmprocpy.imfgen import generate_imf
from gcmprocpy.imfgen.sources import download_omni_files, omni_path

pytestmark = pytest.mark.live


def test_live_download_and_generate_small_slice(tmp_path):
    cache_dir = str(tmp_path)
    # start mid-year so only one year file is needed for the 10-min lead-in.
    download_omni_files([2020], cache_dir)
    import os
    assert os.path.exists(omni_path(cache_dir, 2020))

    ds = generate_imf(start="2020-06-01", end="2020-06-02", source="omni",
                      cache_dir=cache_dir, download=False)
    assert ds.sizes["ndata"] == 2 * 1440
    assert int(ds.attrs["yearday_beg"]) == 2020153   # 2020-06-01
    assert int(ds.attrs["yearday_end"]) == 2020154   # 2020-06-02
    # Real solar wind: finite, physically plausible after interpolation.
    assert np.all(np.isfinite(ds["bx"].values))
    assert np.nanmin(ds["swvel"].values) > 100      # plausible flow speed (km/s)
    # Range-mask + interpolation must actually fire somewhere on real data
    # (real OMNI minutes always contain some gaps), exercising the mask path.
    assert (ds["bxMask"].values == 0).any()

