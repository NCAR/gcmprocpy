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

Computations operate on whatever representation the source fields are stored in:
the residual ``N2`` and the ratios are mixing-ratio identities (matching
tgcmproc's ``fn2 = max(1e-5, 1 - O2 - O1)`` in ``denconv.F`` / ``mkderived.F``),
so they are valid when the major species are mixing ratios.
"""
import logging

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


_register_all()
