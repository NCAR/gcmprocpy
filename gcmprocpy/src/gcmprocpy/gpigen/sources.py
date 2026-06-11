"""Data sources for the GFZ Potsdam geophysical indices.

Every source returns the same two dicts, so the rest of the pipeline is
source-agnostic:

    Fobs_data = {"datetime": [iso, ...], "Fobs": [f107, ...]}
    Kp_data   = {"datetime": [iso, ...], "Kp":   [kp,    ...]}   # 8 per day

Two sources are provided:

- ``"json"``     : two calls to the GFZ JSON API (the original ``gpi_create.py``
                   path). Default. Honours an explicit ``[start, end]`` window.
- ``"textfile"`` : download and parse ``Kp_ap_Ap_SN_F107_since_1932.txt`` locally
                   (the original ``gpi_create_27avg.py`` path). Both indices come
                   from a single parse, filtered to ``[start, end]``.
"""

import os
from datetime import timedelta

import requests

from .dates import ISO_FMT

JSON_API = "https://kp.gfz.de/app/json/"
TEXT_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
TEXT_FILENAME = "Kp_ap_Ap_SN_F107_since_1932.txt"


def fetch(source, start_dt, end_dt, status="def", cache_dir=None, verbose=False):
    """Dispatch to a named source; returns ``(Fobs_data, Kp_data)``."""
    if source == "json":
        return _fetch_json(start_dt, end_dt, status, verbose)
    if source == "textfile":
        return _fetch_textfile(start_dt, end_dt, cache_dir, verbose)
    raise ValueError(f"Unknown source {source!r}; expected 'json' or 'textfile'.")


def _fetch_json(start_dt, end_dt, status, verbose):
    start = start_dt.strftime(ISO_FMT)
    end = end_dt.strftime(ISO_FMT)
    fobs_url = f"{JSON_API}?start={start}&end={end}&index=Fobs&status={status}"
    kp_url = f"{JSON_API}?start={start}&end={end}&index=Kp&status={status}"
    resp_fobs = requests.get(fobs_url)
    resp_kp = requests.get(kp_url)
    if resp_fobs.status_code != 200 or resp_kp.status_code != 200:
        raise RuntimeError(
            f"GFZ JSON API request failed "
            f"(Fobs={resp_fobs.status_code}, Kp={resp_kp.status_code})."
        )
    if verbose:
        print("Fobs and Kp data retrieved from JSON API.")
    fobs = resp_fobs.json()
    kp = resp_kp.json()
    return (
        {"datetime": fobs["datetime"], "Fobs": fobs["Fobs"]},
        {"datetime": kp["datetime"], "Kp": kp["Kp"]},
    )


def _download_textfile(cache_dir, verbose):
    cache_dir = cache_dir or os.getcwd()
    path = os.path.join(cache_dir, TEXT_FILENAME)
    resp = requests.get(TEXT_URL)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download {TEXT_URL} (status {resp.status_code})."
        )
    with open(path, "wb") as fh:
        fh.write(resp.content)
    if verbose:
        print(f"Downloaded source file: {path}")
    return path


def _fetch_textfile(start_dt, end_dt, cache_dir, verbose):
    path = _download_textfile(cache_dir, verbose)
    return parse_textfile(path, start_dt, end_dt)


def parse_textfile(path, start_dt, end_dt):
    """Parse the GFZ combined text file into the two index dicts.

    Column layout (whitespace separated, ``#`` comments skipped):
    ``year month day ... Kp1..Kp8 (cols 7:15) ... F10.7obs (3rd from last)``.
    Rows outside ``[start_dt, end_dt]`` (by calendar day) are dropped.
    ``-1`` sentinels in F10.7 are preserved for downstream interpolation.
    """
    from datetime import datetime

    fobs_dt, fobs_val = [], []
    kp_dt, kp_val = [], []
    start_day = start_dt.date()
    end_day = end_dt.date()
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            date = datetime(year, month, day)
            if date.date() < start_day or date.date() > end_day:
                continue
            fobs_dt.append(date.strftime("%Y-%m-%dT00:00:00Z"))
            fobs_val.append(float(parts[-3]))
            for hour, kpv in zip(range(0, 24, 3), parts[7:15]):
                stamp = (date + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:00:00Z")
                kp_dt.append(stamp)
                kp_val.append(float(kpv))
    return (
        {"datetime": fobs_dt, "Fobs": fobs_val},
        {"datetime": kp_dt, "Kp": kp_val},
    )
