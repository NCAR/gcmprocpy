"""Geographic <-> magnetic (Quasi-Dipole / Apex) coordinate support.

TIE-GCM and WACCM-X write some fields natively on a magnetic (Apex) grid --
dimensions ``mlat`` / ``mlon`` (with ``mlev`` / ``imlev`` vertical) -- and those
can be plotted directly on magnetic axes. Geographic fields (on ``lat`` / ``lon``)
can be reprojected onto a regular **Quasi-Dipole** ``mlat`` / ``mlon`` grid so
they are directly comparable with the model's native magnetic fields.

The geographic<->magnetic transform uses `apexpy <https://apexpy.readthedocs.io>`_
(Richmond-1995 Modified Apex / Quasi-Dipole with IGRF), the same coordinate
family TIE-GCM / WACCM-X use internally. apexpy is an **optional** dependency and
is imported lazily -- it is only needed for the transform, never for plotting a
field that already lives on the magnetic grid.
"""

from datetime import datetime

import numpy as np

# Magnetic-grid dimension names used by TIE-GCM / WACCM-X output.
MAG_LAT_DIM = "mlat"
MAG_LON_DIM = "mlon"
MAG_LEV_DIMS = ("mlev", "imlev")
MAG_DIMS = (MAG_LAT_DIM, MAG_LON_DIM) + MAG_LEV_DIMS

_APEXPY_HELP = (
    "Geographic<->magnetic conversion requires the optional 'apexpy' dependency.\n"
    "Install it with:  pip install 'gcmprocpy[magnetic]'\n"
    "            (or)  conda install -c conda-forge apexpy\n"
    "Fields already stored on a magnetic (mlat/mlon) grid do NOT need apexpy."
)


def require_apexpy():
    """Import and return :mod:`apexpy`, or raise a helpful ImportError."""
    try:
        import apexpy
    except ImportError as exc:            # pragma: no cover - exercised via message
        raise ImportError(_APEXPY_HELP) from exc
    return apexpy


def is_magnetic_var(ds, variable_name):
    """Return True if ``variable_name`` in dataset ``ds`` is on a magnetic grid.

    A variable is considered magnetic if any of its dimensions is ``mlat`` or
    ``mlon`` (TIE-GCM/WACCM-X magnetic-grid fields such as ZMAG or the WACCM-X
    dynamo fields).
    """
    try:
        dims = ds[variable_name].dims
    except (KeyError, AttributeError, TypeError):
        return False
    return MAG_LAT_DIM in dims or MAG_LON_DIM in dims


def extract_mag_lat_lon(datasets, variable_name, time, level=None):
    """Extract a native-magnetic variable as a 2-D ``mlat`` x ``mlon`` field.

    Selects ``time`` (nearest) and a vertical ``level`` on the magnetic vertical
    dimension (``mlev``/``imlev``); if ``level`` is ``None`` or ``'mean'`` the
    field is averaged over that vertical dimension. No coordinate transform is
    performed (the variable is already on the magnetic grid), so this needs no
    apexpy.

    Returns ``(mlat, mlon, values, meta)`` with ``values`` shaped
    ``(len(mlat), len(mlon))`` and ``meta`` carrying units / long_name / model /
    filename.
    """
    for mds in datasets:
        ds = mds.ds
        if variable_name in ds.variables and is_magnetic_var(ds, variable_name):
            da = ds[variable_name]
            if "time" in da.dims:
                da = da.sel(time=time, method="nearest")
            vdim = next((d for d in da.dims if d in MAG_LEV_DIMS), None)
            if vdim is not None:
                if level is None or level == "mean":
                    da = da.mean(dim=vdim)
                else:
                    da = da.sel({vdim: level}, method="nearest")
            # ensure (mlat, mlon) order
            da = da.transpose(MAG_LAT_DIM, MAG_LON_DIM)
            meta = {
                "units": da.attrs.get("units", ""),
                "long_name": da.attrs.get("long_name", variable_name),
                "model": getattr(mds, "model", None),
                "filename": getattr(mds, "filename", None),
            }
            return ds[MAG_LAT_DIM].values, ds[MAG_LON_DIM].values, da.values, meta
    raise ValueError(
        f"'{variable_name}' was not found as a magnetic-grid (mlat/mlon) variable "
        f"in the provided datasets."
    )


def decimal_year(time):
    """Convert a time to a decimal year for ``apexpy.Apex(date=...)``.

    Accepts a decimal-year float (returned unchanged), a ``numpy.datetime64``,
    or anything ``numpy.datetime64`` can parse (ISO string, ``datetime``).
    """
    if (isinstance(time, (int, float, np.integer, np.floating))
            and not isinstance(time, (bool, np.datetime64))):
        return float(time)          # already a decimal year
    dt = np.datetime64(time, "s").astype("datetime64[s]").item()  # -> datetime
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def default_mag_grid(mlat_step=2.5, mlon_step=5.0):
    """Return a regular ``(mlat, mlon)`` target grid (degrees).

    ``mlat`` spans (-90, 90) on half-steps (cell centers, avoiding the poles);
    ``mlon`` spans [-180, 180).
    """
    mlat = np.arange(-90.0 + mlat_step / 2.0, 90.0, mlat_step)
    mlon = np.arange(-180.0, 180.0, mlon_step)
    return mlat, mlon


def geo_to_qd_grid(values, glats, glons, height_km, date,
                   mlat=None, mlon=None):
    """Reproject a geographic 2-D field onto a regular Quasi-Dipole grid.

    Parameters
    ----------
    values : 2-D array, shape ``(nlat, nlon)``
        The field on the geographic grid (rows = latitude, cols = longitude).
    glats, glons : 1-D arrays
        Geographic latitudes and longitudes (degrees) of ``values``.
    height_km : float
        Geometric height of the slice in km (QD latitude depends on altitude).
    date : float or datetime64
        IGRF epoch -- decimal year or a datetime the slice belongs to.
    mlat, mlon : 1-D arrays, optional
        Target magnetic grid; defaults to :func:`default_mag_grid`.

    Returns
    -------
    (mlat, mlon, values_qd)
        ``values_qd`` has shape ``(len(mlat), len(mlon))``; points whose
        magnetic node maps outside the geographic domain are NaN.

    Notes
    -----
    Uses **inverse** mapping (cleaner than scattered forward interpolation): for
    each target ``(mlat, mlon)`` node, ``qd2geo`` gives the geographic
    ``(glat, glon)``, and the geographic field is then bilinearly sampled there
    (longitude treated as periodic).
    """
    apexpy = require_apexpy()
    from scipy.interpolate import RegularGridInterpolator

    if mlat is None or mlon is None:
        _mlat, _mlon = default_mag_grid()
        mlat = _mlat if mlat is None else np.asarray(mlat, float)
        mlon = _mlon if mlon is None else np.asarray(mlon, float)
    mlat = np.asarray(mlat, float)
    mlon = np.asarray(mlon, float)

    apex = apexpy.Apex(date=decimal_year(date))
    mlon_grid, mlat_grid = np.meshgrid(mlon, mlat)          # (nmlat, nmlon)
    glat_t, glon_t, _ = apex.qd2geo(mlat_grid.ravel(), mlon_grid.ravel(),
                                    float(height_km))

    # Build a periodic-in-longitude interpolator over the geographic field.
    glats = np.asarray(glats, float)
    glons = np.asarray(glons, float)
    values = np.asarray(values, float)
    lat_order = np.argsort(glats)
    glats_s = glats[lat_order]
    vals_s = values[lat_order, :]
    lon_order = np.argsort(glons)
    glons_s = glons[lon_order]
    vals_s = vals_s[:, lon_order]
    glon_pad = np.concatenate([glons_s - 360.0, glons_s, glons_s + 360.0])
    vals_pad = np.concatenate([vals_s, vals_s, vals_s], axis=1)
    interp = RegularGridInterpolator((glats_s, glon_pad), vals_pad,
                                     bounds_error=False, fill_value=np.nan)

    glon_t = ((glon_t + 180.0) % 360.0) - 180.0
    sampled = interp(np.column_stack([glat_t, glon_t]))
    return mlat, mlon, sampled.reshape(mlat_grid.shape)
