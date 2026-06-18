"""Species-aware density unit conversions.

Port of tgcmproc ``denconv.F`` (B. Foster).  Converts atmospheric species
fields among the four density representations used by TIE-GCM post-
processing:

- **MMR**    — mass mixing ratio (dimensionless, mass fraction)
- **CM3**    — number density (molecules cm⁻³)
- **CM3-MR** — volume mixing ratio / mole fraction (dimensionless)
- **GM/CM3** — mass density (g cm⁻³)

The conversion requires the mean air molar mass (``BARM``) and the air
number density at each grid point (``pkt``), both of which depend on the
model's temperature and O / O₂ fields.

Reference formulas (tgcmproc ``denconv.F``)::

    BARM = 1 / (O₂/32 + O/16 + (1 − O₂ − O)/28)
    pkt  = pressure / (k_B · T)                    [CGS]

    MMR → CM3     :   f · pkt · BARM / W
    MMR → CM3-MR  :   f · BARM / W
    MMR → GM/CM3  :   f · pkt · BARM · 1.66e-24

The only model-specific piece is how *pressure* is recovered from the
vertical coordinate (see :func:`compute_pkt`): TIE-GCM uses log-pressure
``p = p₀·exp(−ζ)``; WACCM-X uses the CAM hybrid sigma-pressure
``p = hyam·P0 + hybm·PS``.  Both are supported.
"""
import logging
import numpy as np

from .containers import (
    PlotData,
    MODEL_DEFAULTS,
    get_species_names,
    cache_data_fn,
)
from .data_parse import get_mtime, level_log_transform, derive_aware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (CGS, matching tgcmproc denconv.F)
# ---------------------------------------------------------------------------
_BOLTZ_CGS = 1.3805e-16          # erg K⁻¹
_AMU_G = 1.66e-24                # g per atomic mass unit
_P0_TIEGCM_CGS = 5.0e-4          # dyn cm⁻²  (TIE-GCM log-pressure reference)

SUPPORTED_DENSITY_UNITS = ('MMR', 'CM3', 'CM3-MR', 'GM/CM3')

# Role → molar mass (g/mol).  Role keys match MODEL_DEFAULTS['species'].
_SPECIES_MOLAR_MASS = {
    'o':   16.00,
    'o2':  32.00,
    'n2':  28.00,
    'no':  30.00,
    'co2': 44.00,
    'h':    1.008,
    'o3':  48.00,
    'ho2': 33.00,
}


# Common unit-string aliases seen in model attrs.
_UNIT_ALIASES = {
    'cm-3': 'CM3',
    'cm^-3': 'CM3',
    '1/cm3': 'CM3',
    'molecules/cm3': 'CM3',
    'mmr': 'MMR',
    'kg/kg': 'MMR',
    'g/g': 'MMR',
    'vmr': 'CM3-MR',
    'mol/mol': 'CM3-MR',
    'mole fraction': 'CM3-MR',
    'g/cm3': 'GM/CM3',
    'g cm-3': 'GM/CM3',
}


def _normalize_unit(unit):
    """Canonicalise a density unit string to one of SUPPORTED_DENSITY_UNITS."""
    if unit is None:
        return None
    key = unit.strip().lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    upper = unit.strip().upper()
    if upper in SUPPORTED_DENSITY_UNITS:
        return upper
    return unit  # fall through; caller raises


def get_species_molar_mass(model, variable_name):
    """Molar mass (g/mol) for a species variable name in the given model.

    Resolves through :func:`get_species_names` so that model-specific
    naming (e.g. TIE-GCM ``O1`` vs WACCM-X ``O``) is handled uniformly.

    Raises:
        ValueError: If the variable is not a known species in the model,
            or no molar mass is registered for that species role.
    """
    sp = get_species_names(model)
    inverse = {v: k for k, v in sp.items()}
    role = inverse.get(variable_name)
    if role is None:
        raise ValueError(
            f"'{variable_name}' is not a recognised species for {model}. "
            f"Known: {sorted(sp.values())}"
        )
    if role not in _SPECIES_MOLAR_MASS:
        raise ValueError(
            f"No molar mass registered for species role '{role}' "
            f"({variable_name} in {model}).  Add it to _SPECIES_MOLAR_MASS."
        )
    return _SPECIES_MOLAR_MASS[role]


def compute_barm(o_mmr, o2_mmr):
    """Mean air molar mass (g/mol) from atomic-O and molecular-O₂ MMRs.

    Mirrors ``denconv.F``::

        BARM = 1 / (O₂/32 + O/16 + max(1e-5, 1 − O₂ − O) / 28)

    The residual ``(1 − O₂ − O)`` is treated as N₂.  A 1e-5 floor
    guards against the pathological case where O₂ + O ≈ 1 in the upper
    thermosphere.
    """
    o_mmr = np.asarray(o_mmr, dtype=float)
    o2_mmr = np.asarray(o2_mmr, dtype=float)
    residual = np.maximum(1.0e-5, 1.0 - o2_mmr - o_mmr)
    return 1.0 / (o2_mmr / 32.0 + o_mmr / 16.0 + residual / 28.0)


def compute_pkt(levs, temperature, model='TIE-GCM', *,
                hyam=None, hybm=None, p0=None, ps=None):
    """Air number density (cm⁻³) = pressure / (k_B · T), in CGS.

    The pressure is recovered from the model's vertical coordinate, which
    differs by model:

    - **TIE-GCM** (log-pressure ``lev = ζ = ln(p₀/p)``)::

          p = p₀ · exp(−ζ),   p₀ = 5 × 10⁻⁴ dyn cm⁻²

      ``temperature`` lev axis must be axis 0 when multi-dimensional;
      ``lev`` broadcasts across the remaining axes.

    - **WACCM-X** (CAM hybrid sigma-pressure)::

          p(k,j,i) = hyam(k) · P0 + hybm(k) · PS(j,i)     [Pa]

      converted to dyn cm⁻² (×10).  Requires ``hyam``/``hybm`` (on the
      field's vertical coordinate) and surface pressure ``ps`` (lat, lon);
      ``p0`` defaults to 1 × 10⁵ Pa if not supplied.  Returns a full
      ``(nlev, nlat, nlon)`` field since pressure varies horizontally.

    Args:
        levs: Vertical coordinate values, shape ``(nlev,)``.
        temperature: Temperature in K.
        model: ``'TIE-GCM'`` or ``'WACCM-X'``.
        hyam, hybm: Hybrid A/B coefficients on the field's level coordinate
            (WACCM-X only), shape ``(nlev,)``.
        p0: Reference pressure in Pa (WACCM-X only; default 1e5).
        ps: Surface pressure in Pa, shape ``(nlat, nlon)`` (WACCM-X only).
    """
    t = np.asarray(temperature, dtype=float)
    if model == 'TIE-GCM':
        levs = np.asarray(levs, dtype=float)
        # Broadcast lev (axis 0) across temperature's remaining axes.
        shape = [1] * max(t.ndim, 1)
        if t.ndim >= 1:
            shape[0] = len(levs)
        p = _P0_TIEGCM_CGS * np.exp(-levs.reshape(shape))
    elif model == 'WACCM-X':
        if hyam is None or hybm is None or ps is None:
            raise ValueError(
                "WACCM-X compute_pkt needs hyam, hybm, and ps (surface "
                "pressure); pass them from the dataset."
            )
        hyam = np.asarray(hyam, dtype=float)
        hybm = np.asarray(hybm, dtype=float)
        ps = np.asarray(ps, dtype=float)
        p0 = 1.0e5 if p0 is None else float(p0)
        # p(lev, lat, lon) in Pa, then Pa -> dyn cm⁻² (1 Pa = 10 dyn cm⁻²).
        p_pa = hyam[:, None, None] * p0 + hybm[:, None, None] * ps[None, :, :]
        p = p_pa * 10.0
    else:
        raise NotImplementedError(
            f"compute_pkt supports TIE-GCM and WACCM-X (got '{model}')."
        )
    return p / (_BOLTZ_CGS * t)


def convert_density_units(values, from_unit, to_unit, *,
                          barm, pkt, molar_mass):
    """Convert a density field between MMR / CM3 / CM3-MR / GM/CM3.

    Uses MMR as the pivot.  All formulas mirror ``denconv.F``; the
    reverse (non-MMR → MMR) direction is algebraic inversion.

    Args:
        values: Array-like field to convert.
        from_unit, to_unit: Source / target unit strings.  Accepts
            common aliases (``'cm-3'``, ``'kg/kg'``, ``'g/cm3'``...).
        barm: Mean air molar mass (g/mol) broadcast-compatible with
            *values*.
        pkt: Air number density (cm⁻³), same broadcast shape.
        molar_mass: Species molar mass (g/mol), scalar.

    Returns:
        ndarray with the same shape as *values*, in *to_unit*.
    """
    fu = _normalize_unit(from_unit)
    tu = _normalize_unit(to_unit)
    if fu not in SUPPORTED_DENSITY_UNITS:
        raise ValueError(
            f"Unsupported from_unit '{from_unit}'. "
            f"Supported: {SUPPORTED_DENSITY_UNITS} (aliases accepted)."
        )
    if tu not in SUPPORTED_DENSITY_UNITS:
        raise ValueError(
            f"Unsupported to_unit '{to_unit}'. "
            f"Supported: {SUPPORTED_DENSITY_UNITS} (aliases accepted)."
        )

    values = np.asarray(values, dtype=float)
    if fu == tu:
        return values

    w = float(molar_mass)

    # Step 1: from_unit → MMR
    if fu == 'MMR':
        mmr = values
    elif fu == 'CM3':
        mmr = values * w / (pkt * barm)
    elif fu == 'CM3-MR':
        mmr = values * w / barm
    elif fu == 'GM/CM3':
        rho_air = pkt * barm * _AMU_G
        mmr = values / rho_air

    # Step 2: MMR → to_unit (formulas straight from denconv.F)
    if tu == 'MMR':
        return mmr
    if tu == 'CM3':
        return mmr * pkt * barm / w
    if tu == 'CM3-MR':
        return mmr * barm / w
    if tu == 'GM/CM3':
        return mmr * pkt * barm * _AMU_G


def _select_level(values, coord_vals, selected_lev_ilev):
    """Select a level from a ``(nlev, ...)`` array, mirroring batch_arr_lat_lon.

    ``'mean'`` averages over the level axis; an exact coordinate value selects
    it; an out-of-range value clamps to the nearest end; otherwise the two
    nearest levels are averaged.  Done *after* unit conversion so a level mean
    is the mean of the converted field (not a converted mean).
    """
    if selected_lev_ilev is None:
        return values
    if selected_lev_ilev == 'mean':
        return values.mean(axis=0)
    sel = float(selected_lev_ilev)
    coord_vals = np.asarray(coord_vals, dtype=float)
    exact = np.where(coord_vals == sel)[0]
    if exact.size:
        return values[int(exact[0])]
    if sel >= coord_vals.max():
        return values[int(np.argmax(coord_vals))]
    if sel <= coord_vals.min():
        return values[int(np.argmin(coord_vals))]
    order = np.argsort(np.abs(coord_vals - sel))
    return (values[int(order[0])] + values[int(order[1])]) / 2.0


@derive_aware
@cache_data_fn
def arr_density(datasets, variable_name, time, *,
                to_unit='CM3', from_unit=None, selected_lev_ilev=None,
                log_level=True, **kwargs):
    """Extract a species field and convert it to the requested density unit.

    Reads the species, temperature, and O / O₂ fields from the first
    dataset that has *time*, computes BARM and pkt, and returns the
    converted field.

    Args:
        datasets (list): :class:`ModelDataset` objects.
        variable_name (str): Species variable name (e.g. ``'O2'``,
            ``'NO'``, ``'CO2'``).
        time: ``numpy.datetime64`` or ISO-8601 string.
        to_unit (str): Target unit.  One of :data:`SUPPORTED_DENSITY_UNITS`
            (aliases accepted).  Default ``'CM3'``.
        from_unit (str): Source unit.  If ``None`` (default), reads it
            from the dataset variable's ``units`` attribute.
        selected_lev_ilev (float | str, optional): If given, return only this
            level (or ``'mean'`` over levels), selected *after* conversion.
            ``None`` (default) returns the full vertical grid.
        log_level (bool): Whether to log-transform level coordinates for
            the returned :class:`PlotData`.

    Returns:
        PlotData: full ``(nlev, nlat, nlon)`` field in *to_unit*, or a
        ``(nlat, nlon)`` slice if *selected_lev_ilev* is given.  ``None``
        if no dataset contains *time*.

    Raises:
        ValueError: If required species/temperature/O/O₂ fields are missing,
            the source unit cannot be determined, or (WACCM-X) the hybrid
            pressure coefficients (hyam/hybm/PS) are absent.
    """
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    for mds in datasets:
        if not mds.has_time(time):
            continue
        ds = mds.ds

        sp = get_species_names(mds.model)
        t_name, o_name, o2_name = sp['temp'], sp['o'], sp['o2']
        required = [variable_name, t_name, o_name, o2_name]
        missing = [r for r in required if r not in ds.variables]
        if missing:
            raise ValueError(
                f"Density conversion requires {required} but dataset "
                f"{mds.filename!r} is missing: {missing}"
            )

        molar_mass = get_species_molar_mass(mds.model, variable_name)

        src_unit = from_unit if from_unit is not None else \
            ds[variable_name].attrs.get('units')
        if src_unit is None:
            raise ValueError(
                f"No 'units' attribute on {variable_name!r}; pass from_unit= "
                f"explicitly."
            )

        selected_mtime = get_mtime(ds, time)
        ds_t = ds.sel(time=time)
        field = ds_t[variable_name].values.astype(float)
        temp = ds_t[t_name].values.astype(float)
        o_mmr = ds_t[o_name].values.astype(float)
        o2_mmr = ds_t[o2_name].values.astype(float)

        da = ds_t[variable_name]
        if 'lev' in da.dims:
            levs = ds_t.lev.values.astype(float)
        elif 'ilev' in da.dims:
            levs = ds_t.ilev.values.astype(float)
        else:
            raise ValueError(
                f"'{variable_name}' has no lev/ilev coordinate for density "
                f"conversion."
            )

        # BARM/pkt are only needed when the unit actually changes.  An identity
        # conversion (e.g. an already-cm-3 field requested as CM3) skips them —
        # so it never requires WACCM-X hybrid coefficients for a no-op.
        if _normalize_unit(src_unit) == _normalize_unit(to_unit):
            barm = pkt = None
        else:
            barm = compute_barm(o_mmr, o2_mmr)
            if mds.model == 'WACCM-X':
                # Hybrid sigma-pressure: pressure (hence pkt) varies horizontally.
                vdim = 'lev' if 'lev' in da.dims else 'ilev'
                am, bm = ('hyam', 'hybm') if vdim == 'lev' else ('hyai', 'hybi')
                for req in (am, bm, 'PS'):
                    if req not in ds.variables:
                        raise ValueError(
                            f"WACCM-X density conversion needs '{req}' in "
                            f"{mds.filename!r} (hybrid-pressure coordinate)."
                        )
                p0 = float(ds_t['P0'].values) if 'P0' in ds.variables else None
                pkt = compute_pkt(
                    levs, temp, model='WACCM-X',
                    hyam=ds_t[am].values.astype(float),
                    hybm=ds_t[bm].values.astype(float),
                    p0=p0, ps=ds_t['PS'].values.astype(float),
                )
            else:
                pkt = compute_pkt(levs, temp, model=mds.model)

        converted = convert_density_units(
            field, src_unit, to_unit,
            barm=barm, pkt=pkt, molar_mass=molar_mass,
        )
        long_name = ds[variable_name].attrs.get('long_name', variable_name)

        if selected_lev_ilev is not None:
            # Select the level AFTER conversion (so a 'mean' is the mean of the
            # converted field, not a conversion of the mean).
            sliced = _select_level(converted, levs, selected_lev_ilev)
            return PlotData(
                values=sliced,
                variable_unit=_normalize_unit(to_unit),
                variable_long_name=long_name,
                model=mds.model,
                filename=mds.filename,
                lats=ds_t.lat.values.astype(float),
                lons=ds_t.lon.values.astype(float),
                selected_lev=selected_lev_ilev,
                mtime=selected_mtime,
            )

        levs_display = level_log_transform(levs.copy(), mds.model, log_level)
        return PlotData(
            values=converted,
            variable_unit=_normalize_unit(to_unit),
            variable_long_name=long_name,
            model=mds.model,
            filename=mds.filename,
            levs=levs_display,
            lats=ds_t.lat.values.astype(float),
            lons=ds_t.lon.values.astype(float),
            mtime=selected_mtime,
        )

    logger.warning("%s not found in any dataset.", time)
    return None
