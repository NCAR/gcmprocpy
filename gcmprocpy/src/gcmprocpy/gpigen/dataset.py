"""Assemble the processed arrays into an xarray Dataset and write NetCDF."""

import os

import numpy as np
import xarray as xr


def build_dataset(year_day, f107d, f107a, kp, window, centered, missing_dates):
    """Build the GPI ``xarray.Dataset`` with the TIEGCM global attributes."""
    year_day = np.array(year_day)
    kind = "centered" if centered else "trailing"
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


def gpi_filename(ds, prefix="gpi"):
    """``<prefix>_<begYYYYDDD>-<endYYYYDDD>.nc`` from the dataset's bounds."""
    beg = int(ds.attrs["yearday_beg"])
    end = int(ds.attrs["yearday_end"])
    return f"{prefix}_{beg}-{end}.nc"


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
    ds.to_netcdf(path=path)
    return path
