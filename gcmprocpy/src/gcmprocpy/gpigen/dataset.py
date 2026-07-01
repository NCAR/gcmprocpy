"""Assemble the processed arrays into an xarray Dataset and write NetCDF."""

import os

import numpy as np
import xarray as xr


def build_dataset(year_day, f107d, f107a, kp, window, centered, missing_dates,
                  model="tiegcm"):
    """Build the GPI ``xarray.Dataset`` for the target model.

    ``model="tiegcm"`` (default): the TIE-GCM format -- ``year_day``/``f107d``/
    ``f107a`` on ``ndays`` and ``kp`` on ``(ndays, nkp=8)`` -- with the TIEGCM
    global attributes (unchanged).
    ``model="waccmx"``: the WACCM-X (CESM) 3-hourly solar-parameters format on a
    single (unlimited-on-write) ``time`` dimension of length ``ndays*8``:
    ``date`` (YYYYMMDD), ``datesec`` (0..75600), ``f107`` (= ``f107d`` repeated
    across the 8 daily slots), ``f107a`` (repeated), ``kp`` (flattened) and
    ``ap`` (derived from ``Kp`` via the official lookup table).
    """
    year_day = np.array(year_day)
    kind = "centered" if centered else "trailing"
    if model == "waccmx":
        return _build_waccmx(year_day, f107d, f107a, kp, window, kind, missing_dates)
    avg_label = f"{window}-day {kind} average 10.7 cm solar flux"

    ds = xr.Dataset(
        {
            "year_day": (
                ["ndays"],
                year_day,
                {"long_name": "4-digit year followed by 3-digit day"},
            ),
            "f107d": (
                ["ndays"],
                np.array(f107d),
                {"long_name": "daily 10.7 cm solar flux"},
            ),
            "f107a": (["ndays"], np.array(f107a), {"long_name": avg_label}),
            "kp": (["ndays", "nkp"], np.array(kp), {"long_name": "3-hourly kp index"}),
        },
        coords={"ndays": year_day, "nkp": np.arange(np.array(kp).shape[1])},
    )

    ds.attrs["title"] = "Geophysical Indices, obtained from gfz-potsdam"
    ds.attrs["yearday_beg"] = year_day[0]
    ds.attrs["yearday_end"] = year_day[-1]
    ds.attrs["ncar_mss_path"] = "/TGCM/data/gpi_1960001-2015365.nc"
    ds.attrs["data_source_url"] = "https://kp.gfz.de/"
    ds.attrs["hao_file_write_source"] = "https://github.com/AnonNick/GPI"
    ds.attrs["info"] = (
        "Yearly ascii data files obtained from data_source_url; "
        "nc file written by hao_file_write_source."
    )
    ds.attrs["averaging_window_days"] = window
    ds.attrs["averaging_kind"] = kind
    ds.attrs["F107_missing"] = list(missing_dates)
    return ds


def _build_waccmx(year_day, f107d, f107a, kp, window, kind, missing_dates):
    """WACCM-X 3-hourly GPI Dataset (see :func:`build_dataset`)."""
    from .dates import yearday_to_yyyymmdd
    from .indices import kp_to_ap

    kp = np.asarray(kp, dtype=float)                      # (ndays, nkp)
    ndays, nkp = kp.shape
    date = np.repeat([yearday_to_yyyymmdd(d) for d in year_day],
                     nkp).astype("float64")               # YYYYMMDD, 8x/day
    datesec = np.tile(np.arange(nkp) * 10800.0, ndays)    # 0, 10800, ... 75600
    f107 = np.repeat(np.asarray(f107d, dtype=float), nkp)
    f107a_flat = np.repeat(np.asarray(f107a, dtype=float), nkp)
    kp_flat = kp.reshape(-1)
    ap_flat = kp_to_ap(kp_flat).astype("float64")
    sfu = "10^-22 W m^-2 Hz^-1"

    ds = xr.Dataset(
        {
            "date": ("time", date, {"long_name": "current date (YYYYMMDD)"}),
            "datesec": ("time", datesec,
                        {"long_name": "seconds after midnight UT"}),
            "f107": ("time", f107,
                     {"units": sfu,
                      "long_name": "10.7 cm solar radio flux (F10.7)"}),
            "f107a": ("time", f107a_flat,
                      {"units": sfu,
                       "long_name": f"{window}-day {kind} mean of 10.7 cm "
                                    "solar radio flux (F10.7)"}),
            "kp": ("time", kp_flat, {"long_name": "3 hr planetary K index"}),
            "ap": ("time", ap_flat,
                   {"long_name": "3 hr planetary a index from K index lookup table"}),
        }
    )
    ds.attrs["title"] = ("Geophysical Indices for WACCM-X "
                         "(from GFZ Potsdam via TIE-GCM GPI)")
    ds.attrs["yearday_beg"] = year_day[0]
    ds.attrs["yearday_end"] = year_day[-1]
    ds.attrs["data_source_url"] = "https://kp.gfz.de/"
    ds.attrs["hao_file_write_source"] = "https://github.com/AnonNick/GPI"
    ds.attrs["data_summary"] = (
        "3-hourly Kp and daily F10.7 from GFZ Potsdam, processed for TIE-GCM and "
        "converted to the CAM/WACCM/WACCM-X 3-hourly solar-parameters format; ap "
        "derived from Kp via the official Kp->ap lookup table."
    )
    ds.attrs["averaging_window_days"] = window
    ds.attrs["averaging_kind"] = kind
    ds.attrs["F107_missing"] = list(missing_dates)
    ds.attrs["model"] = "waccmx"
    return ds


def gpi_filename(ds, prefix="gpi"):
    """``<prefix>[_WACCMX]_<begYYYYDDD>-<endYYYYDDD>.nc`` from the dataset's bounds.

    A ``model='waccmx'`` dataset gains a ``WACCMX`` tag (e.g. ``gpi_WACCMX_...``).
    """
    beg = int(ds.attrs["yearday_beg"])
    end = int(ds.attrs["yearday_end"])
    tag = "_WACCMX" if ds.attrs.get("model") == "waccmx" else ""
    return f"{prefix}{tag}_{beg}-{end}.nc"


def save_gpi(ds, output_dir=".", prefix="gpi", path=None):
    """Write ``ds`` to NetCDF and return the path written.

    ``path`` overrides the auto-generated name; otherwise the file is
    ``<output_dir>/<prefix>_<beg>-<end>.nc``.
    """
    if path is None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, gpi_filename(ds, prefix))
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    unlimited = ["time"] if ds.attrs.get("model") == "waccmx" else None
    ds.to_netcdf(path=path, unlimited_dims=unlimited)
    return path
