"""Geomagnetic index conversions.

The ``ap`` index is *defined* as a fixed lookup of ``Kp`` (the official
28-value geomagnetic conversion table), so deriving ``ap`` from ``Kp`` is
lossless -- verified bit-exact against GFZ Potsdam's published ``ap`` over
1932-2025 (274,672 three-hourly values, zero mismatches). WACCM-X's GPI
(solar-parameters) input needs ``ap`` while the TIE-GCM GPI carries only
``Kp``, so we compute it here.
"""

import numpy as np

# ap for each Kp value in thirds: index = round(Kp * 3), i.e.
# Kp = 0o, 0+, 1-, 1o, 1+, ... , 9o  ->  indices 0 .. 27.
KP_TO_AP = np.array(
    [0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48,
     56, 67, 80, 94, 111, 132, 154, 179, 207, 236, 300, 400],
    dtype=int,
)


def kp_to_ap(kp):
    """Map ``Kp`` (in thirds, 0-9) to the ``ap`` index via the official table.

    Accepts a scalar or array-like; returns an ``int`` array of the same shape.
    Out-of-range indices are clamped to the table bounds (defensive against a
    stray sentinel), though gpigen gap-fills ``Kp`` before this is called.
    """
    idx = np.rint(np.asarray(kp, dtype=float) * 3.0).astype(int)
    return KP_TO_AP[np.clip(idx, 0, KP_TO_AP.size - 1)]
