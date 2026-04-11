"""Eliassen-Palm flux calculations.

Port of tgcmproc ``epflux.F`` (B. Foster and Hanli Liu, 2-4/98).
Computes three derived diagnostic fields from 3-D wind and temperature:

- **EPVY** — meridional EP flux component (m² s⁻²)
- **EPVZ** — vertical EP flux component (m² s⁻²)
- **EPVDIV** — EP flux divergence / wave forcing (m s⁻¹ day⁻¹)

Reference:
    Volland, Hans, *Atmospheric Tidal and Planetary Waves*,
    Kluwer Academic Publishers (Norwell, MA), Section 2.7.
"""
import logging
import numpy as np
from .containers import PlotData, MODEL_DEFAULTS, get_species_names, register_derived
from .data_parse import get_mtime, level_log_transform

logger = logging.getLogger(__name__)

# Physical constants (matching tgcmproc epflux.F)
_G = 9.8                             # gravity (m s⁻²)
_CP = 1004.0                         # specific heat at constant pressure (J kg⁻¹ K⁻¹)
_TS = 300.0                          # reference surface temperature (K)
_RE = 6.371e6                        # Earth radius (m)
_OMEGA = 2.0 * np.pi / 86400.0      # Earth rotation rate (rad s⁻¹)
_R = 287.0                           # gas constant for dry air (J kg⁻¹ K⁻¹)
_D2R = np.pi / 180.0                # degrees → radians
_H = _R * _TS / _G                  # reference scale height (m) ≈ 8786


def _interp_ilev_to_lev(w_ilev, ilev_vals, lev_vals):
    """Linearly interpolate a 3-D field from *ilev* to *lev* grid.

    Args:
        w_ilev: Array on interface levels, shape ``(nilev, nlat, nlon)``.
        ilev_vals: Interface level coordinate values, shape ``(nilev,)``.
        lev_vals: Midpoint level coordinate values, shape ``(nlev,)``.

    Returns:
        Array on midpoint levels, shape ``(nlev, nlat, nlon)``.
    """
    nlev = len(lev_vals)
    w_lev = np.empty((nlev,) + w_ilev.shape[1:])
    for k in range(nlev):
        idx = np.searchsorted(ilev_vals, lev_vals[k]) - 1
        idx = max(0, min(idx, len(ilev_vals) - 2))
        denom = ilev_vals[idx + 1] - ilev_vals[idx]
        if abs(denom) < 1e-30:
            alpha = 0.0
        else:
            alpha = (lev_vals[k] - ilev_vals[idx]) / denom
        alpha = max(0.0, min(1.0, alpha))
        w_lev[k] = (1.0 - alpha) * w_ilev[idx] + alpha * w_ilev[idx + 1]
    return w_lev


def epflux(temp, u, v, lats, levs, w=None):
    """Compute Eliassen-Palm flux components.

    All inputs must be in SI units (K, m s⁻¹, degrees, dimensionless
    log-pressure levels).

    The calculation follows the quasi-geostrophic EP flux formulation:

    .. math::

        S_y = -\\overline{u'v'} + \\frac{\\overline{v'T'}}{\\gamma}
              \\frac{\\partial \\bar{u}}{\\partial z}

        S_z = -\\left[\\overline{u'w'} + \\left(\\frac{1}{\\cos\\phi}
              \\frac{\\partial(\\bar{u}\\cos\\phi)}{\\partial y} - f\\right)
              \\frac{\\overline{v'T'}}{\\gamma}\\right]

        D = \\frac{1}{\\cos^2\\phi}
            \\frac{\\partial(\\cos^2\\phi \\, S_y)}{\\partial y}
            + \\frac{1}{\\bar{\\rho}}
            \\frac{\\partial(\\bar{\\rho} \\, S_z)}{\\partial z}

    where primes denote deviations from the zonal mean, overbars are
    zonal means, *γ* is the static stability, and *f* the Coriolis
    parameter.

    Args:
        temp: Temperature (K), shape ``(nlev, nlat, nlon)``.
        u: Zonal wind (m s⁻¹), same shape.
        v: Meridional wind (m s⁻¹), same shape.
        lats: Latitude array (degrees), shape ``(nlat,)``.
        levs: Log-pressure levels (dimensionless), shape ``(nlev,)``.
        w: Vertical wind (m s⁻¹), same shape as *temp*.
           Optional — required for EPVZ and EPVDIV.

    Returns:
        dict: Keys ``'EPVY'``, ``'EPVZ'``, ``'EPVDIV'``, each a
        ``(nlev, nlat)`` ndarray.  EPVZ and EPVDIV are *None* if *w*
        is not provided.
    """
    temp = np.asarray(temp, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    lats = np.asarray(lats, dtype=float)
    levs = np.asarray(levs, dtype=float)

    nlev, nlat, nlon = temp.shape

    # Heights from log-pressure (m)
    zpht = levs * _H

    # Latitude geometry
    lat_rad = lats * _D2R
    cos_lat = np.cos(lat_rad)  # (nlat,)
    dlat = np.abs(lat_rad[1] - lat_rad[0])  # assume uniform spacing
    dy = _RE * dlat  # meridional grid spacing (m)

    # ---- Zonal means (nlev, nlat) ----
    tzm = temp.mean(axis=2)
    uzm = u.mean(axis=2)
    vzm = v.mean(axis=2)

    # ---- Eddy fields ----
    t_prime = temp - tzm[:, :, np.newaxis]
    u_prime = u - uzm[:, :, np.newaxis]
    v_prime = v - vzm[:, :, np.newaxis]

    # ---- Static stability gamma (K m⁻¹) ----
    # gamma = dTzm/dz + g*Tzm/(Ts*cp)
    gamma = np.zeros((nlev, nlat))
    for k in range(1, nlev - 1):
        dz = zpht[k + 1] - zpht[k - 1]
        gamma[k, :] = ((tzm[k + 1, :] - tzm[k - 1, :]) / dz
                        + _G * tzm[k, :] / (_TS * _CP))
    gamma[0, :] = 2.0 * gamma[1, :] - gamma[2, :]
    gamma[-1, :] = 2.0 * gamma[-2, :] - gamma[-3, :]

    # ---- du_zm/dz (s⁻¹) ----
    dudz = np.zeros((nlev, nlat))
    for k in range(1, nlev - 1):
        dz = zpht[k + 1] - zpht[k - 1]
        dudz[k, :] = (uzm[k + 1, :] - uzm[k - 1, :]) / dz
    dudz[0, :] = 2.0 * dudz[1, :] - dudz[2, :]
    dudz[-1, :] = 2.0 * dudz[-2, :] - dudz[-3, :]

    # ---- Zonal-mean eddy covariances (nlev, nlat) ----
    uv_bar = (u_prime * v_prime).mean(axis=2)
    vt_bar = (v_prime * t_prime).mean(axis=2)

    # ---- EPVY: meridional component (m² s⁻²) ----
    epvy = -uv_bar + (vt_bar / gamma) * dudz

    epvz = None
    epvdiv = None

    if w is not None:
        w = np.asarray(w, dtype=float)
        wzm = w.mean(axis=2)
        w_prime = w - wzm[:, :, np.newaxis]
        uw_bar = (u_prime * w_prime).mean(axis=2)

        # ---- EPVZ: vertical component (m² s⁻²) ----
        # Coriolis parameter
        f_cor = 2.0 * _OMEGA * np.sin(lat_rad)  # (nlat,)

        # d(uzm*cos(lat))/dy / cos(lat) — centered differences over lat
        dudy = np.zeros((nlev, nlat))
        for j in range(1, nlat - 1):
            dudy[:, j] = ((uzm[:, j + 1] * cos_lat[j + 1]
                          - uzm[:, j - 1] * cos_lat[j - 1])
                          / (cos_lat[j] * 2.0 * dy))
        # Extrapolate lat boundaries
        dudy[:, 0] = 2.0 * dudy[:, 1] - dudy[:, 2]
        dudy[:, -1] = 2.0 * dudy[:, -2] - dudy[:, -3]

        epvz = -(uw_bar + (dudy - f_cor[np.newaxis, :]) * vt_bar / gamma)

        # ---- EPVDIV: divergence / wave forcing (m s⁻¹ day⁻¹) ----
        # Density profile from hydrostatic approximation: rho ∝ exp(-lev)/T
        rhozm = np.exp(-levs[:, np.newaxis]) / tzm  # (nlev, nlat)

        epvdiv = np.zeros((nlev, nlat))

        # Interior points only (j=1..nlat-2, k=1..nlev-2)
        for j in range(1, nlat - 1):
            # d(cos²(lat)*Sy)/dy / cos²(lat)
            depydy = ((cos_lat[j + 1]**2 * epvy[:, j + 1]
                       - cos_lat[j - 1]**2 * epvy[:, j - 1])
                      / (cos_lat[j]**2 * 2.0 * dy))

            # d(rho*Sz)/dz / rho  — computed at interior levels
            depzdz = np.zeros(nlev)
            for k in range(1, nlev - 1):
                dz = zpht[k + 1] - zpht[k - 1]
                depzdz[k] = ((rhozm[k + 1, j] * epvz[k + 1, j]
                             - rhozm[k - 1, j] * epvz[k - 1, j])
                             / (rhozm[k, j] * dz))
            # Extrapolate vertical boundaries
            depzdz[0] = 2.0 * depzdz[1] - depzdz[2]
            depzdz[-1] = 2.0 * depzdz[-2] - depzdz[-3]

            epvdiv[:, j] = (depydy + depzdz) * 86400.0  # → m/s/day

        # Extrapolate latitude boundaries
        epvdiv[:, 0] = 2.0 * epvdiv[:, 1] - epvdiv[:, 2]
        epvdiv[:, -1] = 2.0 * epvdiv[:, -2] - epvdiv[:, -3]

    return {'EPVY': epvy, 'EPVZ': epvz, 'EPVDIV': epvdiv}


# Variable-name metadata for each component
_COMPONENT_META = {
    'EPVY': ('m² s⁻²', 'EP Flux (meridional)'),
    'EPVZ': ('m² s⁻²', 'EP Flux (vertical)'),
    'EPVDIV': ('m s⁻¹ day⁻¹', 'EP Flux Divergence'),
}


def arr_epflux(datasets, component, time, log_level=True):
    """Compute an EP flux component from model datasets.

    Extracts temperature and wind fields at the requested time, converts
    units to SI, and calls :func:`epflux`.

    Args:
        datasets (list): List of :class:`ModelDataset` objects.
        component (str): ``'EPVY'``, ``'EPVZ'``, or ``'EPVDIV'``.
        time: Timestamp (``numpy.datetime64`` or ISO string).
        log_level (bool): Whether to log-transform level coordinates
            for the output PlotData.

    Returns:
        PlotData: Result with shape ``(nlev, nlat)`` suitable for
        :func:`plt_lev_lat`.

    Raises:
        ValueError: If *component* is invalid or required variables
            are missing from the datasets.
    """
    component = component.upper()
    if component not in _COMPONENT_META:
        raise ValueError(
            f"component must be one of {list(_COMPONENT_META)}, got '{component}'"
        )

    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    for mds in datasets:
        if not mds.has_time(time):
            continue
        ds = mds.ds

        # ---- Model variable names from MODEL_DEFAULTS ----
        defaults = MODEL_DEFAULTS[mds.model]
        sp = get_species_names(mds.model)
        t_name = sp['temp']
        u_name = defaults['wind_u']
        v_name = defaults['wind_v']
        wind_scale = defaults['wind_scale']
        w_name = 'W'  # EP flux uses log-pressure W, not wind_w (WN/OMEGA)

        # Check required variables exist
        required = [t_name, u_name, v_name]
        missing = [r for r in required if r not in ds.variables]
        if missing:
            raise ValueError(
                f"EP flux requires {required} but dataset is missing: {missing}"
            )

        selected_mtime = get_mtime(ds, time)
        ds_t = ds.sel(time=time)

        # ---- Extract 3-D fields on lev grid ----
        temp = ds_t[t_name].values.astype(float)   # (nlev, nlat, nlon)
        u_raw = ds_t[u_name].values.astype(float)
        v_raw = ds_t[v_name].values.astype(float)

        lats = ds_t.lat.values.astype(float)
        try:
            levs = ds_t[t_name].lev.values.astype(float)
        except AttributeError:
            levs = ds_t[t_name].ilev.values.astype(float)

        # Remove all-NaN levels
        not_nan = ~np.isnan(temp).all(axis=(1, 2))
        temp = temp[not_nan]
        u_raw = u_raw[not_nan]
        v_raw = v_raw[not_nan]
        levs_raw = levs[not_nan]

        # Convert horizontal winds to m/s
        u_si = u_raw * wind_scale
        v_si = v_raw * wind_scale

        # ---- Extract W if needed ----
        w_si = None
        need_w = component in ('EPVZ', 'EPVDIV')
        if need_w:
            # TIE-GCM 3.0 renamed W → OMEGA; check both
            if w_name not in ds.variables:
                if 'OMEGA' in ds.variables:
                    w_name = 'OMEGA'
                else:
                    raise ValueError(
                        f"Cannot compute {component}: vertical wind "
                        f"(W or OMEGA) not in dataset. Only EPVY is available."
                    )
            w_data = ds_t[w_name]
            w_raw = w_data.values.astype(float)

            # W may be on ilev (interface levels) — interpolate to lev
            if 'ilev' in w_data.dims:
                ilev_vals = ds_t.ilev.values.astype(float)
                w_raw = _interp_ilev_to_lev(w_raw, ilev_vals, levs)

            # Apply same NaN mask
            w_raw = w_raw[not_nan]

            # TIE-GCM W is d(zp)/dt in s⁻¹, so w_geom ≈ H * W
            # WACCM-X W is already in m/s
            if mds.model == 'TIE-GCM':
                w_si = w_raw * _H
            else:
                w_si = w_raw

        # ---- Compute EP flux ----
        result = epflux(temp, u_si, v_si, lats, levs_raw, w=w_si)

        values = result[component]
        if values is None:
            raise ValueError(
                f"Cannot compute {component}: vertical wind not available"
            )

        # Transform levels for display
        levs_display = level_log_transform(levs_raw.copy(), mds.model, log_level)

        unit, long_name = _COMPONENT_META[component]

        return PlotData(
            values=values,
            lats=lats,
            levs=levs_display,
            variable_unit=unit,
            variable_long_name=long_name,
            mtime=selected_mtime,
            model=mds.model,
            filename=mds.filename,
        )

    logger.warning(f"{time} not found.")
    return None


register_derived('EPVY', arr_epflux, plot_types={'lev_lat'})
register_derived('EPVZ', arr_epflux, plot_types={'lev_lat'})
register_derived('EPVDIV', arr_epflux, plot_types={'lev_lat'})
