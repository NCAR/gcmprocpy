"""Full OH Meinel band vibrational emission model.

Port of tgcmproc ohrad.F (B. Foster, U. B. Makhlouf, SDL/Stewart Radiance Lab).
Solves the coupled steady-state rate equations for 10 vibrational levels
(v=0 through v=9) and computes emission rates for all 39 Meinel bands.
"""
import numpy as np
from .containers import PlotData, get_species_names, register_derived
from .data_parse import batch_arr_lat_lon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NVL = 10  # number of vibrational levels (v=0 ground through v=9)

# Vibrational energy levels (cm⁻¹) — HITRAN
_ENERGY = np.array([
    0.00, 3568.50, 6971.37, 10210.57, 13287.19,
    16201.33, 18951.90, 21536.25, 23949.83, 26185.79,
])

_C2K = 1.438818  # hc/k conversion factor (cm⁻¹ → Kelvin)

# Einstein A coefficients (s⁻¹) — Nelson et al. (1990), updated by Makhlouf (1999)
# _AOH[v, dv_idx] for transition v → v-(dv_idx+1), dv_idx = 0..5 for Δv = 1..6
# Stored column-major in Fortran; here rows = vibrational level, cols = Δv index.
_AOH = np.array([
    #  Δv=1     Δv=2     Δv=3     Δv=4     Δv=5     Δv=6
    [  0.000,   0.000,   0.000,   0.000,   0.000,   0.000],  # v=0
    [ 17.36,    0.000,   0.000,   0.000,   0.000,   0.000],  # v=1
    [ 23.60,   10.32,    0.000,   0.000,   0.000,   0.000],  # v=2
    [ 22.21,   27.57,    1.122,   0.000,   0.000,   0.000],  # v=3
    [ 16.45,   48.65,    4.118,   0.134,   0.000,   0.000],  # v=4
    [  9.360,  70.73,    9.415,   0.625,   0.019,   0.000],  # v=5
    [  3.780,  91.13,   17.16,    1.744,   0.110,   0.003],  # v=6
    [  2.310, 107.3,    27.20,    3.785,   0.364,   0.023],  # v=7
    [  7.250, 116.8,    39.07,    7.033,   0.918,   0.088],  # v=8
    [ 20.43,  117.5,    51.94,   11.74,    1.962,   0.254],  # v=9
])

# O2 quenching rate coefficients (cm³ s⁻¹) — Adler-Golden (1997)
_AAO2 = np.array([0.0, 1.3e-13, 2.7e-13, 5.2e-13, 8.8e-13,
                   1.7e-12, 3.0e-12, 5.42e-12, 9.81e-12, 1.7e-11])

# N2 quenching rate coefficients (cm³ s⁻¹)
_AAN2 = np.array([0.0, 5.757e-15, 1.0e-14, 1.737e-14, 3.017e-14,
                   5.241e-14, 9.103e-14, 1.581e-13, 2.746e-13, 4.77e-13])

# Atomic oxygen quenching — "fast" rates (cm³ s⁻¹)
_AAO = np.array([3.9e-11, 10.5e-11, 2.5e-10, 2.5e-10, 2.5e-10,
                  2.5e-10, 2.5e-10, 2.5e-10, 2.5e-10, 2.5e-10])

# Reverse O2 quenching rate coefficients (detailed balance with AAO2)
_RAAO2 = np.array([1.3e-13, 2.7e-13, 5.2e-13, 8.8e-13, 1.7e-12,
                    3.0e-12, 5.42e-12, 9.81e-12, 1.7e-11, 0.0])

# Reverse N2 quenching rate coefficients
_RAAN2 = np.array([5.757e-15, 1.0e-14, 1.737e-14, 3.017e-14, 5.241e-14,
                    9.103e-14, 1.581e-13, 2.746e-13, 4.77e-13, 0.0])

# Reverse quenching energy differences (Kelvin)
_REE = np.zeros(_NVL)
for _kv in range(_NVL - 1):
    _REE[_kv] = (_ENERGY[_kv + 1] - _ENERGY[_kv]) * _C2K

# Primary reaction branching: H + O3 → OH(v) + O2 — Klenerman & Smith
_FVL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.17, 0.27, 0.48])

# Secondary reaction branching: O + HO2 → OH(v) + O2 — Kaye
_FVLS = np.array([0.52, 0.34, 0.13, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# FAC0: 1 for v=0 only (ground-state recycling flag)
_FAC0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Sum of Einstein A coefficients for each level (total radiative loss rate)
_AOH_SUM = _AOH.sum(axis=1)

# Multi-quantum quenching fractional distribution — precomputed constants.
# _MQ_FRAC[target, source] = fraction of source's quenching that goes to target.
# Uses exponential weighting: FQV(jv) = exp(-jv), normalized. (ohrad.F lines 339-352)
_MQ_FRAC = np.zeros((_NVL, _NVL))
for _s in range(_NVL - 1, 0, -1):
    _kim = min(_s, 6)
    _fqv = np.array([np.exp(-float(_jv)) for _jv in range(_kim)])
    _sqv = _fqv.sum()
    for _jv in range(_kim):
        _target = _s - _jv - 1
        _MQ_FRAC[_target, _s] = _fqv[_jv] / _sqv

# Forward multi-quantum O2 quenching rate matrix (constant, since EEO2=0 → Q1O2=AAO2)
_XMQO2 = _AAO2[np.newaxis, :] * _MQ_FRAC

# Band catalog: (upper_v, lower_v) → metadata
OH_BANDS = {}
for _v in range(1, _NVL):
    for _dv_idx in range(6):
        _lower = _v - (_dv_idx + 1)
        if _lower >= 0 and _AOH[_v, _dv_idx] > 0:
            _delta_e = _ENERGY[_v] - _ENERGY[_lower]
            OH_BANDS[(_v, _lower)] = {
                'name': f'OH({_v}-{_lower})',
                'wavelength_um': round(1.0e4 / _delta_e, 4) if _delta_e > 0 else None,
                'einstein_a': _AOH[_v, _dv_idx],
            }


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def ohrad(temp, o2, o, n2, h, o3, ho2, oh=None):
    """Full OH Meinel band vibrational emission model.

    Port of tgcmproc ``ohrad.F`` (B. Foster, U. B. Makhlouf, SDL).
    Solves the coupled steady-state rate equations for 10 vibrational
    levels (v = 0 … 9) at each grid point independently.

    The steady-state system at each point is ``A x = b`` where *A* is
    a 10×10 rate matrix (radiative decay, collisional quenching,
    multi-quantum O2 redistribution, reverse quenching) and *b* is the
    external production vector (H + O3 primary, O + HO2 secondary).

    References:
        Einstein A coefficients — Nelson et al. (1990), updated Makhlouf (1999)
        Quenching rates — Adler-Golden (1997)
        Production branching — Klenerman & Smith (H+O3), Kaye (O+HO2)

    Args:
        temp: Temperature (K).  Any shape ``(...)``.
        o2:   O2 number density (cm⁻³), same shape.
        o:    Atomic oxygen number density (cm⁻³), same shape.
        n2:   N2 number density (cm⁻³), same shape.
        h:    Atomic hydrogen number density (cm⁻³), same shape.
        o3:   Ozone number density (cm⁻³), same shape.
        ho2:  HO2 number density (cm⁻³), same shape.
        oh:   OH ground-state density (cm⁻³), same shape.
              Optional — if *None*, the v = 0 recycling terms are omitted.

    Returns:
        tuple:
            - **vib_pop** (*ndarray*, shape ``(..., 10)``):
              Vibrational populations [OH(v)] in cm⁻³.
            - **band_emission** (*dict*):
              ``{(upper_v, lower_v): emission_array}`` in
              photons cm⁻³ s⁻¹, each array of shape ``(...)``.
    """
    temp = np.asarray(temp, dtype=float)
    sO2 = np.asarray(o2, dtype=float)
    sO = np.asarray(o, dtype=float)
    sN2 = np.asarray(n2, dtype=float)
    sH = np.asarray(h, dtype=float)
    sO3 = np.asarray(o3, dtype=float)
    sHO2 = np.asarray(ho2, dtype=float)
    sOH0 = np.asarray(oh, dtype=float) if oh is not None else np.zeros_like(temp)

    orig_shape = temp.shape
    N = temp.size

    # Flatten to 1-D for batch matrix construction
    T = temp.ravel()
    sO2 = sO2.ravel()
    sO = sO.ravel()
    sN2 = sN2.ravel()
    sH = sH.ravel()
    sO3 = sO3.ravel()
    sHO2 = sHO2.ravel()
    sOH0 = sOH0.ravel()

    # Temperature-dependent reaction rate constants  (shape N)
    rk1 = 1.4e-10 * np.exp(-470.0 / T)     # H + O3 → OH* + O2
    rk2 = 3.0e-11 * np.exp(200.0 / T)      # O + HO2 → OH* + O2
    rk6 = 1.6e-12 * np.exp(-1140.0 / T)    # OH(v=0) + O3
    rk7 = 4.8e-11                            # OH(v=0) + HO2
    rk8 = 4.2e-12                            # OH(v=0) + OH
    rk5 = 1.1e-14                            # O3 + HO2

    # Production rates  (shape N)
    prohv = rk1 * sH * sO3          # primary H + O3
    prohvs = rk2 * sHO2 * sO        # secondary O + HO2

    # Reverse quenching rates  (shape N × 10)
    rq1o2 = _RAAO2[None, :] * np.exp(-_REE[None, :] / T[:, None])
    rq1n2 = _RAAN2[None, :] * np.exp(-_REE[None, :] / T[:, None])

    # ---- Production vector b  (shape N × 10) ----
    # b[v] = -production[v]  (negative because equation is A*x = b)
    b = -(prohv[:, None] * _FVL[None, :]
          + prohvs[:, None] * _FVLS[None, :]
          + rk5 * sO3[:, None] * sHO2[:, None] * _FAC0[None, :])

    # ---- Rate matrix A  (shape N × 10 × 10) ----
    AA = np.zeros((N, _NVL, _NVL))

    # Diagonal loss terms
    for v in range(_NVL):
        fac0_v = 1.0 if v == 0 else 0.0
        AA[:, v, v] = -(
            _AOH_SUM[v]
            + _AAO2[v] * sO2 + _AAN2[v] * sN2 + _AAO[v] * sO
            + rq1o2[:, v] * sO2 + rq1n2[:, v] * sN2
            + fac0_v * (rk6 * sO3 + rk7 * sHO2 + 2.0 * rk8 * sOH0)
        )

    # Upper triangle: cascade from higher levels (downward transitions)
    # Δv = 1: Einstein A + multi-quantum O2 + single-quantum N2
    for v in range(_NVL - 1):
        s = v + 1
        AA[:, v, s] = _AOH[s, 0] + _XMQO2[v, s] * sO2 + _AAN2[s] * sN2

    # Δv = 2 … 6: Einstein A + multi-quantum O2
    for dv in range(2, 7):
        for v in range(_NVL - dv):
            s = v + dv
            AA[:, v, s] = _AOH[s, dv - 1] + _XMQO2[v, s] * sO2

    # Lower triangle: reverse quenching (upward transitions)
    # Δv = 1: multi-quantum O2 reverse + single-quantum N2 reverse
    for v in range(1, _NVL):
        s = v - 1  # source (lower level)
        AA[:, v, s] = _MQ_FRAC[s, v] * rq1o2[:, v] * sO2 + rq1n2[:, s] * sN2

    # Δv > 1: multi-quantum O2 reverse only
    for v in range(2, _NVL):
        for s in range(v - 1):
            AA[:, v, s] = _MQ_FRAC[s, v] * rq1o2[:, v] * sO2

    # Solve  A x = b  at each grid point
    # Add trailing dim to b so solve treats it as (N, 10, 1) matrix RHS,
    # avoiding ambiguity when N is small.
    x = np.linalg.solve(AA, b[..., np.newaxis])[..., 0]  # shape (N, 10)
    x = np.clip(x, 0.0, None)  # physical constraint: non-negative populations

    vib_pop = x.reshape(orig_shape + (_NVL,))

    # Band emissions: rate = population × Einstein A coefficient  (photons cm⁻³ s⁻¹)
    band_emission = {}
    for v in range(1, _NVL):
        for dv_idx in range(6):
            lower = v - (dv_idx + 1)
            if lower >= 0 and _AOH[v, dv_idx] > 0:
                emission = x[:, v] * _AOH[v, dv_idx]
                band_emission[(v, lower)] = emission.reshape(orig_shape)

    return vib_pop, band_emission


def arr_mkoh_band(datasets, variable_name, time,
                  selected_lev_ilev=None, selected_unit=None, plot_mode=False):
    """Compute OH vibrational emission from datasets using the full model.

    The *variable_name* determines what is returned:

    * ``'OH_<upper>_<lower>'`` — a specific band, e.g. ``'OH_8_3'``
    * ``'OH_VIB_<v>'`` — vibrational level population, e.g. ``'OH_VIB_8'``
    * ``'OH_TOTAL'`` — sum of all 39 band emission rates

    Requires these variables in the datasets: temperature (TN/T),
    O2, O (O1), N2, H, O3, HO2.

    Args:
        datasets (list): List of ModelDataset objects.
        variable_name (str): Encoded output selector (see above).
        time: Time value to extract.
        selected_lev_ilev: Pressure level or ``'mean'``.
        selected_unit (str, optional): Desired unit (unused, kept for API compat).
        plot_mode (bool): If True, return a PlotData object.

    Returns:
        numpy.ndarray or PlotData.

    Raises:
        ValueError: If required species are missing or *variable_name* is invalid.
    """
    names = get_species_names(datasets[0].model)

    # Check which species are available
    available = set()
    for mds in datasets:
        available.update(mds.ds.variables)

    required = [names['temp'], names['o'], names['o2'], names['n2'],
                names['h'], names['o3'], names['ho2']]
    missing = [r for r in required if r not in available]
    if missing:
        raise ValueError(
            f"Full OH model requires {required} but dataset is missing: {missing}"
        )

    # Extract all species at the requested time/level
    var_names = [names['temp'], names['o'], names['o2'], names['n2'],
                 names['h'], names['o3'], names['ho2']]
    results = batch_arr_lat_lon(datasets, var_names, time,
                                selected_lev_ilev, selected_unit, plot_mode)

    r_temp = results[names['temp']]
    r_o = results[names['o']]
    r_o2 = results[names['o2']]
    r_n2 = results[names['n2']]
    r_h = results[names['h']]
    r_o3 = results[names['o3']]
    r_ho2 = results[names['ho2']]

    if plot_mode:
        vals = lambda r: r.values
    else:
        vals = lambda r: r

    vib_pop, band_em = ohrad(
        vals(r_temp), vals(r_o2), vals(r_o), vals(r_n2),
        vals(r_h), vals(r_o3), vals(r_ho2),
    )

    # Parse variable_name to select output
    vn = variable_name.upper()

    if vn == 'OH_TOTAL':
        values = sum(band_em.values())
        long_name = 'OH total emission'
        unit = 'photons cm-3 sec-1'
    elif vn.startswith('OH_VIB_'):
        v = int(vn.split('_')[2])
        if v < 0 or v >= _NVL:
            raise ValueError(f"Vibrational level must be 0-9, got {v}")
        values = vib_pop[..., v]
        long_name = f'OH v={v} population'
        unit = 'cm-3'
    elif vn.startswith('OH_') and vn.count('_') == 2:
        parts = vn.split('_')
        upper_v, lower_v = int(parts[1]), int(parts[2])
        if (upper_v, lower_v) not in OH_BANDS:
            raise ValueError(
                f"OH band ({upper_v},{lower_v}) not valid. "
                f"Available: {sorted(OH_BANDS.keys())}"
            )
        values = band_em[(upper_v, lower_v)]
        long_name = f'OH({upper_v}-{lower_v}) emission'
        unit = 'photons cm-3 sec-1'
    else:
        raise ValueError(
            f"Unrecognized OH variable '{variable_name}'. Use OH_TOTAL, "
            f"OH_VIB_N, or OH_upper_lower (e.g. OH_8_3)."
        )

    if plot_mode:
        return PlotData(
            values=values, variable_unit=unit,
            variable_long_name=long_name, model=r_temp.model,
            filename=r_temp.filename, lats=r_temp.lats,
            lons=r_temp.lons, selected_lev=r_temp.selected_lev,
            mtime=r_temp.mtime,
        )
    return values


register_derived('OH_*', arr_mkoh_band)
