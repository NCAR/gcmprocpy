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
    pkt  = p₀ · exp(−ζ) / (k_B · T)                [CGS]

    MMR → CM3     :   f · pkt · BARM / W
    MMR → CM3-MR  :   f · BARM / W
    MMR → GM/CM3  :   f · pkt · BARM · 1.66e-24

Currently TIE-GCM only.  WACCM-X uses a hybrid-pressure coordinate and
needs a different pressure expression; calling :func:`arr_density` with
a WACCM-X dataset raises :class:`NotImplementedError`.
"""
import logging
import numpy as np

from .containers import (
    PlotData,
    MODEL_DEFAULTS,
    get_species_names,
    cache_data_fn,
)
from .data_parse import get_mtime, level_log_transform

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


def compute_pkt(levs, temperature, model='TIE-GCM'):
    """Air number density (cm⁻³) on TIE-GCM log-pressure levels.

    ``pkt = p₀ · exp(−ζ) / (k_B · T)`` in CGS, where *ζ* is the log-
    pressure coordinate (``lev``) and ``p₀ = 5 × 10⁻⁴`` dyn cm⁻².

    Args:
        levs: Log-pressure level values, shape ``(nlev,)``.
        temperature: Temperature in K.  Lev axis must be axis 0 when
            the array is multi-dimensional.
        model: Currently ``'TIE-GCM'`` only.
    """
    if model != 'TIE-GCM':
        raise NotImplementedError(
            f"compute_pkt supports TIE-GCM only (got '{model}')."
        )
    levs = np.asarray(levs, dtype=float)
    t = np.asarray(temperature, dtype=float)
    # Broadcast lev (axis 0) across temperature's remaining axes.
    shape = [1] * max(t.ndim, 1)
    if t.ndim >= 1:
        shape[0] = len(levs)
    p = _P0_TIEGCM_CGS * np.exp(-levs.reshape(shape))
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


@cache_data_fn
def arr_density(datasets, variable_name, time, *,
                to_unit='CM3', from_unit=None, log_level=True, **kwargs):
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
        log_level (bool): Whether to log-transform level coordinates for
            the returned :class:`PlotData`.

    Returns:
        PlotData: Shape ``(nlev, nlat, nlon)`` in *to_unit*.  ``None``
        if no dataset contains *time*.

    Raises:
        NotImplementedError: For non-TIE-GCM datasets (WACCM-X pending).
        ValueError: If the species / temperature / O / O₂ fields are
            missing, or the source unit cannot be determined.
    """
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    for mds in datasets:
        if not mds.has_time(time):
            continue
        ds = mds.ds

        if mds.model != 'TIE-GCM':
            raise NotImplementedError(
                f"arr_density currently supports TIE-GCM only "
                f"(got '{mds.model}'). WACCM-X pressure-coordinate port "
                f"pending."
            )

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

        barm = compute_barm(o_mmr, o2_mmr)
        pkt = compute_pkt(levs, temp, model=mds.model)

        converted = convert_density_units(
            field, src_unit, to_unit,
            barm=barm, pkt=pkt, molar_mass=molar_mass,
        )

        levs_display = level_log_transform(levs.copy(), mds.model, log_level)
        long_name = ds[variable_name].attrs.get('long_name', variable_name)

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
