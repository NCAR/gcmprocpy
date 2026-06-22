"""Derivable intermediate fields: computed on the full grid when absent from a file.

This module implements the "derive-if-missing" layer. A *derivable* field (see
:data:`gcmprocpy.containers.DERIVABLE_VARIABLES`) is an implicit quantity the
model frequently does not write to its history — e.g. molecular nitrogen
``N2 = 1 - O2 - O1`` (a residual of the major species), or composition ratios.

When an ``arr_*`` extractor or a derived-variable handler asks for a field that
is not in the dataset, :func:`ensure_field` computes it on the FULL native grid
as an ``xarray.DataArray`` (with matching dims/coords) and injects it into the
in-memory dataset. From that point every downstream path — ``check_var_dims``,
``_extract_var_attrs``, the ``.sel(time=..., lev=...)`` slicing — works unchanged,
and the same full-grid array can be written back to NetCDF (:func:`save_derived`).

Computations operate on whatever representation the source fields are stored in.
``N2`` and the composition ratios are mixing-ratio identities (matching tgcmproc's
``fn2 = max(1e-5, 1 - O2 - O1)`` in ``mkderived.F``).  The mass-density (``RHO``),
pressure (``PMB``) and frost-point (``TNFP``) derivables additionally reuse the
density machinery in :mod:`gcmprocpy.data_density` (``barm`` / ``pkt`` / unit
conversion), so they are unit-aware (MMR / VMR / cm⁻³) and model-aware (TIE-GCM
log-pressure, WACCM-X hybrid pressure) — all matching tgcmproc ``mkderived.F``.
"""
import logging

import numpy as np
import xarray as xr

from .containers import (
    get_species_names,
    register_derivable,
    resolve_derivable,
)

logger = logging.getLogger(__name__)

# Mixing-ratio floor matching tgcmproc denconv.F (mkdenparms) and mkderived.F.
_MMR_FLOOR = 1.0e-5


class DerivationError(ValueError):
    """A registered derivable cannot be computed because a required input —
    possibly deep in the dependency chain — is unavailable.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handling
    (e.g. the CLI error wrappers) still catches it.
    """


def compute_derivable_da(mds, name, _chain=None):
    """Compute a derivable field for *mds* as a full-grid ``xarray.DataArray``.

    Recursively resolves inputs (a missing derivable input is computed and
    injected first), so derived-on-derived chains work (e.g. ``O/N2`` -> ``N2``).

    Returns ``None`` ONLY when *name* is not a registered derivable. If *name*
    IS derivable but a required input (or a chain of inputs) is unavailable —
    or the model is unsupported — raises :class:`DerivationError` with a message
    naming the missing field and the full derivation chain.

    Args:
        mds: A :class:`gcmprocpy.containers.ModelDataset`.
        name (str): Field name to derive.
        _chain (list, optional): Internal derivation path (cycle detection +
            diagnostics).

    Raises:
        DerivationError: Cyclic chain, unsupported model, or a missing input.
    """
    entry = resolve_derivable(name)
    if entry is None:
        return None

    _chain = list(_chain or [])
    if name.upper() in (c.upper() for c in _chain):
        raise DerivationError("cyclic derivation: " + " -> ".join(_chain + [name]))
    _chain = _chain + [name]
    root = _chain[0]
    via = f" (via {' -> '.join(_chain)})" if len(_chain) > 1 else ""

    if entry['models'] is not None and mds.model not in entry['models']:
        raise DerivationError(
            f"Cannot derive '{root}'{via}: '{name}' is not available for "
            f"model {mds.model}."
        )

    species = get_species_names(mds.model)
    inputs = {}
    for role in entry['inputs']:
        var_name = species.get(role, role)  # role -> dataset var name, else literal
        if var_name in mds.ds.variables:
            inputs[role] = mds.ds[var_name]
            continue
        # Input absent: derive it, or report exactly what is missing and why.
        if resolve_derivable(var_name) is None:
            chain_str = " -> ".join(_chain + [var_name])
            raise DerivationError(
                f"Cannot derive '{root}': requires '{var_name}', which is not "
                f"in the dataset and is not a derivable quantity "
                f"(chain: {chain_str})."
            )
        sub = compute_derivable_da(mds, var_name, _chain)  # raises on deeper failure
        mds.ds[var_name] = sub  # inject the intermediate so it is reused
        inputs[role] = sub

    da = entry['formula'](inputs, mds)
    da = da.rename(entry['name'])
    # entry['units'] overrides only when set; otherwise the formula's own unit
    # (e.g. N2 inheriting O2's MMR/VMR convention) is preserved.
    if entry['units']:
        da.attrs['units'] = entry['units']
    elif 'units' not in da.attrs:
        da.attrs['units'] = ''
    da.attrs['long_name'] = entry['long_name']
    return da


def ensure_field(mds, name):
    """Ensure *name* is present in ``mds.ds``, computing+injecting it if derivable.

    Returns ``True`` if the field is present (already, or after derivation),
    ``False`` if *name* is not a registered derivable.  Raises
    :class:`DerivationError` if *name* is derivable but its inputs are missing.
    """
    if name in mds.ds.variables:
        return True
    if resolve_derivable(name) is None:
        return False
    mds.ds[name] = compute_derivable_da(mds, name)
    return True


def _ensure_derivable_fields(datasets, names):
    """Inject missing-but-derivable *names* into each dataset.

    A name that is simply not derivable is left untouched (the caller's normal
    not-found handling applies).  A name that IS derivable but whose inputs are
    unavailable raises :class:`DerivationError` with the dependency chain, so
    the user gets a precise reason rather than an opaque downstream failure.
    """
    for mds in datasets:
        for name in names:
            if name is not None and name not in mds.ds.variables:
                ensure_field(mds, name)


# ---------------------------------------------------------------------------
# Registered derivable fields
# ---------------------------------------------------------------------------

def _derive_n2(inp, mds):
    """N2 mixing-ratio residual: ``max(1e-5, 1 - O2 - O1)`` (tgcmproc convention).

    Ported from tgcmproc ``mkderived.F`` (subroutine ``mkderived``): for tgcm
    histories "otherwise (tgcm) n2=1-o2-o", i.e.
    ``fn2 = max(.00001, 1.-flat(:,:,ixo2)-flat(:,:,ixo1))`` (line 112), formed
    in mixing-ratio units before any density conversion.

    Inherits O2's mixing-ratio unit (``kg/kg`` / ``mol/mol``) so the residual
    carries the correct convention for downstream density conversion.
    """
    n2 = (1.0 - inp['o2'] - inp['o']).clip(min=_MMR_FLOOR)
    n2.attrs['units'] = inp['o2'].attrs.get('units', 'MMR')
    return n2


def _ratio(num_role, den_role):
    def _f(inp, mds):
        return inp[num_role] / inp[den_role]
    return _f


def _derive_o_o2n2(inp, mds):
    return inp['o'] / (inp['o2'] + inp['n2'])


# ---------------------------------------------------------------------------
# Mass density / pressure / frost point (tgcmproc mkderived.F)
# ---------------------------------------------------------------------------

def _pressure_cgs_da(mds, ref):
    """Air pressure (dyn cm⁻², CGS) on the model grid, as an ``xarray.DataArray``.

    - **TIE-GCM** log-pressure: ``p = p0·exp(-ζ)`` on the ``lev`` coordinate
      (``p0 = 5e-4`` dyn cm⁻²), matching :func:`gcmprocpy.data_density.compute_pkt`.
    - **WACCM-X** hybrid sigma-pressure: ``p = hyam·P0 + hybm·PS`` (Pa), ×10 to
      dyn cm⁻².  Requires ``hyam``/``hybm``/``PS`` (raises :class:`DerivationError`
      otherwise).

    *ref* supplies the vertical dimension (``lev`` vs ``ilev``); the result
    broadcasts against it.
    """
    from .data_density import _P0_TIEGCM_CGS  # lazy import: avoid an import cycle
    vcoord = 'lev' if 'lev' in ref.dims else ('ilev' if 'ilev' in ref.dims else 'lev')
    if mds.model == 'WACCM-X':
        ds = mds.ds
        am, bm = ('hyam', 'hybm') if vcoord == 'lev' else ('hyai', 'hybi')
        for req in (am, bm, 'PS'):
            if req not in ds.variables:
                raise DerivationError(
                    f"Cannot derive WACCM-X pressure: '{req}' (hybrid-pressure "
                    f"coordinate) is not in {mds.filename!r}."
                )
        p0 = float(ds['P0'].values) if 'P0' in ds.variables else 1.0e5
        return (ds[am] * p0 + ds[bm] * ds['PS']) * 10.0   # Pa -> dyn cm⁻²
    return _P0_TIEGCM_CGS * np.exp(-mds.ds[vcoord])        # TIE-GCM log-pressure


def _derive_pmb(inp, mds):
    """Pressure in millibars (tgcmproc ``mkderived.F`` ``PMB``).

    tgcmproc forms ``PMB = (n_O2 + n_O1 + n_N2)·k_B·T·1e-3``; the summed major
    species' number densities equal the total air number density
    ``pkt = p/(k_B·T)``, so this reduces to ``PMB = p·1e-3`` (dyn cm⁻² → mb) —
    the pressure of the vertical coordinate expressed in mb.
    """
    ref = inp['temp']
    return (_pressure_cgs_da(mds, ref) * 1.0e-3).broadcast_like(ref)


def _derive_rho(inp, mds):
    """Total air mass density in g cm⁻³ (tgcmproc ``mkderived.F`` ``RHO``, iden=3).

    Matches the user-facing ``RHO`` branch (``mkderived.F`` ~L222: ``RHO = O2 + O
    + N2`` in the requested density unit; ``iden=3`` → GM/CM3).  Sums the major
    species (O, O₂, N₂), each converted (MMR / VMR / cm⁻³) → GM/CM3 via
    :func:`gcmprocpy.data_density.convert_density_units` — no ×1000, since the
    result is g cm⁻³ rather than the kg m⁻³ of the internal ``mkrhokg`` helper.
    The conversion uses ``pkt`` (model pressure) and ``barm`` (mean molar mass);
    for VMR / cm⁻³ sources the ``barm`` factor cancels and for MMR it is exact
    (``barm`` assumes MMR-stored O/O₂, as tgcmproc does), so the result is
    correct for both TIE-GCM and WACCM-X.
    """
    from .data_density import (  # lazy import: avoid an import cycle
        compute_barm, convert_density_units, get_species_molar_mass, _BOLTZ_CGS,
    )
    ref = inp['temp']
    dims = ref.dims
    # Align every array to ref's dim order before dropping to numpy, so the
    # element-wise products inside convert_density_units are name-safe.
    pkt = (_pressure_cgs_da(mds, ref) / (_BOLTZ_CGS * ref)
           ).broadcast_like(ref).transpose(*dims).values
    barm = compute_barm(inp['o'].transpose(*dims).values,
                        inp['o2'].transpose(*dims).values)
    rho = None
    for role in ('o', 'o2', 'n2'):
        field = inp[role].transpose(*dims)
        w = get_species_molar_mass(mds.model, field.name)
        src = field.attrs.get('units') or 'MMR'
        contrib = convert_density_units(field.values, src, 'GM/CM3',
                                        barm=barm, pkt=pkt, molar_mass=w)
        rho = contrib if rho is None else rho + contrib
    return xr.DataArray(rho, dims=dims, coords=ref.coords)


def _derive_tnfp(inp, mds):
    """Frost-point temperature in K (tgcmproc ``mkderived.F`` ``TNFP``).

    ``TNFP = T - 6077.4 / (28.548 - ln(p_H2O[mb]))`` where the water-vapour
    partial pressure ``p_H2O = x_H2O · p`` and ``x_H2O`` is the water mole
    fraction.  ``x_H2O`` is taken unit-aware from the ``H2O`` field: a mole
    fraction (VMR) is used directly; a mass mixing ratio is converted with the
    mean molar mass (``x = mmr·barm/18``, the tgcmproc form, which assumes
    MMR-stored O/O₂); a number density is divided by the total (``x = n_H2O/pkt``).
    Requires ``H2O`` in the dataset.  Non-positive ``p_H2O`` is masked to NaN (the
    Fortran ``alog`` is unguarded).
    """
    from .data_density import (  # lazy import: avoid an import cycle
        compute_barm, _normalize_unit, _BOLTZ_CGS,
    )
    ref = inp['temp']
    p_cgs = _pressure_cgs_da(mds, ref)
    press_mb = (p_cgs * 1.0e-3).broadcast_like(ref)
    h2o = inp['H2O']
    src = _normalize_unit(h2o.attrs.get('units'))
    if src == 'CM3-MR':                         # mole fraction (mol/mol)
        x_h2o = h2o
    elif src == 'CM3':                          # number density / total
        pkt = (p_cgs / (_BOLTZ_CGS * ref)).broadcast_like(ref)
        x_h2o = h2o / pkt
    else:                                       # MMR (default): x = mmr·barm/M_H2O
        # barm aligned to H2O's dim order so the numpy product is name-safe.
        barm = compute_barm(inp['o'].transpose(*h2o.dims).values,
                            inp['o2'].transpose(*h2o.dims).values)
        x_h2o = h2o * barm / 18.0
    p_h2o_mb = x_h2o * press_mb
    p_h2o_mb = p_h2o_mb.where(p_h2o_mb > 0)     # mask non-positive (avoid log warnings)
    return ref - 6077.4 / (28.548 - np.log(p_h2o_mb))


def _register_all():
    # N2 = 1 - O2 - O1 (mixing-ratio residual). WACCM-X usually writes N2
    # natively, so this only fires when it is genuinely absent.
    register_derivable(
        'N2', _derive_n2, inputs=['o2', 'o'], units='',  # inherits O2's unit
        long_name='molecular nitrogen (derived: 1 - O2 - O)',
    )
    # Composition ratios (dimensionless; numerator and denominator share the
    # source field's units, so the ratio is representation-independent).
    # Ported from tgcmproc mkderived.F (subroutine mkderived, field-name
    # dispatch, lines 653-667): O/N2 = fo1/fn2, N2/O = fn2/fo1, O/O2 = fo1/fo2,
    # O/(O2+N2) = fo1/(fo2+fn2). tgcmproc converts each operand to the requested
    # density unit first; since both operands share that unit the ratio is
    # invariant, so gcmprocpy forms it directly on the source fields.
    # (key, formula, input roles, long_name)
    ratios = [
        ('O/N2', _ratio('o', 'n2'), ['o', 'n2'], 'atomic oxygen / molecular nitrogen ratio'),
        ('N2/O', _ratio('n2', 'o'), ['n2', 'o'], 'molecular nitrogen / atomic oxygen ratio'),
        ('O/O2', _ratio('o', 'o2'), ['o', 'o2'], 'atomic oxygen / molecular oxygen ratio'),
        ('O/O2+N2', _derive_o_o2n2, ['o', 'o2', 'n2'], 'atomic oxygen / (O2 + N2) ratio'),
    ]
    for key, fn, roles, long_name in ratios:
        register_derivable(key, fn, inputs=roles, units='ratio', long_name=long_name)
        # GUI/CLI-safe alias without the '/' and '+' characters (e.g. O_N2, O_O2pN2).
        alias = key.replace('/', '_').replace('+', 'p')
        register_derivable(alias, fn, inputs=roles, units='ratio', long_name=long_name)

    # Mass density, pressure, frost point (tgcmproc mkderived.F: RHO, PMB, TNFP).
    register_derivable(
        'RHO', _derive_rho, inputs=['temp', 'o', 'o2', 'n2'], units='GM/CM3',
        long_name='total mass density (derived: O2 + O + N2)',
    )
    register_derivable(
        'PMB', _derive_pmb, inputs=['temp'], units='mb',
        long_name='pressure (derived from the model vertical coordinate)',
    )
    register_derivable(
        'TNFP', _derive_tnfp, inputs=['temp', 'o', 'o2', 'H2O'], units='K',
        long_name='frost-point temperature (derived; requires H2O)',
    )


_register_all()
