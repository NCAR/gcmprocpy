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


def compute_derivable_da(mds, name, _stack=None):
    """Compute a derivable field for *mds* as a full-grid ``xarray.DataArray``.

    Recursively resolves inputs (a missing derivable input is computed and
    injected first), so derived-on-derived chains work (e.g. ``O/N2`` -> ``N2``).
    Returns ``None`` if *name* is not a registered derivable, if it is not
    supported for this model, or if a required input field is unavailable.

    Args:
        mds: A :class:`gcmprocpy.containers.ModelDataset`.
        name (str): Field name to derive.
        _stack (set, optional): Internal recursion guard (cycle detection).

    Raises:
        ValueError: On a cyclic derivation chain.
    """
    entry = resolve_derivable(name)
    if entry is None:
        return None
    if entry['models'] is not None and mds.model not in entry['models']:
        return None

    _stack = set() if _stack is None else _stack
    key = name.upper()
    if key in _stack:
        raise ValueError(
            f"cyclic derivation detected: {' -> '.join(list(_stack) + [key])}"
        )
    _stack = _stack | {key}

    species = get_species_names(mds.model)
    inputs = {}
    for role in entry['inputs']:
        var_name = species.get(role, role)  # role -> dataset var name, else literal
        if var_name not in mds.ds.variables:
            sub = compute_derivable_da(mds, var_name, _stack)
            if sub is None:
                logger.debug("cannot derive %s: input %s unavailable", name, var_name)
                return None
            mds.ds[var_name] = sub  # inject the intermediate so it is reused
        inputs[role] = mds.ds[var_name]

    da = entry['formula'](inputs, mds)
    da = da.rename(entry['name'])
    da.attrs['units'] = entry['units']
    da.attrs['long_name'] = entry['long_name']
    return da


def ensure_field(mds, name):
    """Ensure *name* is present in ``mds.ds``, computing+injecting it if derivable.

    Returns ``True`` if the field is present (already or after derivation),
    ``False`` if it is neither present nor derivable for this dataset.
    """
    if name in mds.ds.variables:
        return True
    da = compute_derivable_da(mds, name)
    if da is None:
        return False
    mds.ds[name] = da
    return True


def _ensure_derivable_fields(datasets, names):
    """Inject any missing-but-derivable *names* into each dataset (best effort)."""
    for mds in datasets:
        for name in names:
            if name is not None and name not in mds.ds.variables:
                try:
                    ensure_field(mds, name)
                except ValueError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("derive %s skipped: %s", name, exc)


# ---------------------------------------------------------------------------
# Registered derivable fields
# ---------------------------------------------------------------------------

def _derive_n2(inp, mds):
    """N2 mixing-ratio residual: ``max(1e-5, 1 - O2 - O1)`` (tgcmproc convention)."""
    n2 = 1.0 - inp['o2'] - inp['o']
    return n2.clip(min=_MMR_FLOOR)


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
        'N2', _derive_n2, inputs=['o2', 'o'], units='MMR',
        long_name='molecular nitrogen (derived: 1 - O2 - O)',
    )
    # Composition ratios (dimensionless; numerator and denominator share the
    # source field's units, so the ratio is representation-independent).
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
