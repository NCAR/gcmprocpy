"""Data sources for imfgen.

Each source returns the same dict so the rest of the pipeline is
source-agnostic::

    {
        "channels":   {name: ndarray, ...},   # raw bx/by/bz/swden/swvel samples
        "timestamps": [datetime, ...],         # one per OUTPUT minute (length n_out)
        "window":     int,                     # trailing-average window
        "interpolate": bool,                   # fill NaN gaps & flag, or pass through
        "source_path": str or None,
    }

``channels`` arrays have ``len >= n_out + window - 1`` (the OMNI arrays carry
``window`` lead-in samples so the trailing average is complete at the first
output minute; see :func:`imfgen.processing.trailing_average`).

Two sources:

- ``"omni"``   : OMNI high-res 1-minute ASCII (``omni_min<year>.asc``), fetched
                 over FTP from SPDF and parsed. A 10-minute trailing average is
                 applied downstream; NaN gaps are interpolated. (Replaces the
                 original ``imf_create.py``.)
- ``"bcwind"`` : a single BCWIND HDF5 file, passed straight through (no average,
                 no interpolation). (Replaces ``bcwind_imf.py``.)
"""

import ftplib
import os
import warnings
from datetime import datetime, timedelta

import numpy as np

# --- OMNI 1-minute ASCII layout (0-based columns) ------------------------
OMNI_COLS = {"bx": 14, "by": 17, "bz": 18, "swvel": 21, "swden": 25}
OMNI_TIME_COLS = {"year": 0, "doy": 1, "hour": 2, "minute": 3}
FILL_CHECK_START = 5  # is_invalid_sequence scans columns [5:]
INVALID_RUN = 19      # >= this many consecutive all-fill minutes ends the data
MINUTES_PER_DAY = 60 * 24

OMNI_FTP_HOST = "spdf.gsfc.nasa.gov"
OMNI_FTP_DIR = "/pub/data/omni/high_res_omni/"
OMNI_FILE_FMT = "omni_min{year}.asc"

# Integers whose decimal digits are all '1' (repunits). A column value is an OMNI
# fill flag iff ``round(value/9)`` is a repunit -- e.g. 9999.99/9 = 1111.11 -> 1111,
# 999.99/9 = 111.11 -> 111, 99999.9/9 = 11111.1 -> 11111. This reproduces the
# original ``check_if_made_of_ones`` (which formatted ``round(x,2)`` as an integer
# string and tested every char == '1'), but vectorised.
_REPUNITS = frozenset(int("1" * k) for k in range(1, 12))

_OMNI_CACHE = {}  # path -> (mtime, ndarray)


# --- OMNI download -------------------------------------------------------

def omni_path(cache_dir, year):
    return os.path.join(cache_dir, OMNI_FILE_FMT.format(year=year))


def _should_refresh(file_mod_time, now):
    """Whether a present file is stale, mirroring the original's monthly cadence."""
    if now.day > 13:
        threshold = now.replace(day=13)
    else:
        last_month = now.replace(day=1) - timedelta(days=1)
        threshold = last_month.replace(day=20)
    return file_mod_time > threshold


def download_omni_files(years, cache_dir, verbose=False):
    """Download missing/stale ``omni_min<year>.asc`` files over FTP-TLS.

    Returns the list of years actually (re)downloaded. Years whose remote file
    does not exist are skipped silently (e.g. a not-yet-published year).
    """
    os.makedirs(cache_dir, exist_ok=True)
    wanted = {int(y) for y in years}
    downloaded = []
    with ftplib.FTP_TLS(OMNI_FTP_HOST) as ftp:
        ftp.login()
        ftp.prot_p()
        ftp.cwd(OMNI_FTP_DIR)
        available = set(ftp.nlst("omni_min*.asc"))
        now = datetime.utcnow()
        for year in sorted(wanted):
            fname = OMNI_FILE_FMT.format(year=year)
            if fname not in available:
                continue
            local = omni_path(cache_dir, year)
            if os.path.exists(local):
                mod = datetime.strptime(ftp.sendcmd(f"MDTM {fname}")[4:].strip(),
                                        "%Y%m%d%H%M%S")
                if not _should_refresh(mod, now):
                    continue
            if verbose:
                print(f"  downloading {fname}")
            with open(local, "wb") as fh:
                ftp.retrbinary(f"RETR {fname}", fh.write)
            downloaded.append(year)
    return downloaded


def load_omni_year(year, cache_dir):
    """``np.loadtxt`` an ``omni_min<year>.asc`` file (cached by path+mtime)."""
    path = omni_path(cache_dir, year)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OMNI file for {year} not found at {path}. "
            f"Pass download=True (default) or place the file there."
        )
    mtime = os.path.getmtime(path)
    cached = _OMNI_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        _OMNI_CACHE[path] = (mtime, np.loadtxt(path))
    return _OMNI_CACHE[path][1]


# --- OMNI fill detection / edge trim -------------------------------------

def _row_all_fill(cols):
    """Per-row boolean: are ALL of ``cols`` (already the [5:] block) fill flags?

    ``cols`` is 2-D ``(n_rows, n_cols)``. Mirrors the original per-minute test
    ``all(check_if_made_of_ones(v/9) for v in row[5:])``.
    """
    rounded = np.rint(np.round(cols / 9.0, 2)).astype(np.int64)
    is_one = np.isin(rounded, list(_REPUNITS))
    return is_one.all(axis=1)


def _first_run_start(flags, run):
    """Index of the first element that begins a run of ``>= run`` True, else -1."""
    count = 0
    for i, f in enumerate(flags):
        count = count + 1 if f else 0
        if count >= run:
            return i - run + 1
    return -1


def last_valid_count(out_rows, run=INVALID_RUN):
    """Number of leading output minutes to KEEP, by the original's day rule.

    The originals truncate at the first *day* that contains a run of ``>= run``
    consecutive all-fill minutes, dropping that day and everything after it. With
    a full, complete record this returns ``len(out_rows)`` (keep everything).
    """
    if len(out_rows) == 0:
        return 0
    fill = _row_all_fill(out_rows[:, FILL_CHECK_START:])
    keys = out_rows[:, 0].astype(np.int64) * 1000 + out_rows[:, 1].astype(np.int64)
    n = len(out_rows)
    i = 0
    while i < n:
        j = i
        while j < n and keys[j] == keys[i]:
            j += 1
        if _first_run_start(fill[i:j], run) != -1:
            return i  # drop this day and everything after
        i = j
    return n


# --- OMNI sample assembly ------------------------------------------------

def _minute_of_year(dt):
    return (dt.timetuple().tm_yday - 1) * MINUTES_PER_DAY + dt.hour * 60 + dt.minute


def omni_samples(start_dt, end_dt, window=10, cache_dir=None, download=True,
                 verbose=False):
    """Assemble OMNI channels + output timestamps for ``[start_dt, end_dt]``.

    ``window`` lead-in minutes are prepended so the trailing average is complete
    at ``start_dt``. The output end is clamped to the last available minute and
    then to the last fully-valid day (fill detection).
    """
    if cache_dir is None:
        cache_dir = os.getcwd()
    lead_start = start_dt - timedelta(minutes=window)
    years = list(range(lead_start.year, end_dt.year + 1))

    if download:
        try:
            download_omni_files(years, cache_dir, verbose=verbose)
        except (ftplib.all_errors) as exc:  # pragma: no cover - network guard
            if verbose:
                print(f"  OMNI download skipped ({exc}); using local files.")

    # Concatenate the needed year files into one contiguous minute stream.
    available_years = [y for y in years if os.path.exists(omni_path(cache_dir, y))]
    if not available_years:
        raise FileNotFoundError(
            f"No OMNI files for years {years} under {cache_dir}."
        )
    # Every output year must be present so the stream stays contiguous: a missing
    # interior year would silently glue non-adjacent data together.
    last_present = max(available_years)
    missing = [y for y in range(start_dt.year, end_dt.year + 1)
               if y <= last_present and not os.path.exists(omni_path(cache_dir, y))]
    if missing:
        raise FileNotFoundError(
            f"Missing OMNI file(s) for year(s) {missing} under {cache_dir}; "
            f"the requested range is not contiguous."
        )
    arrays = {y: load_omni_year(y, cache_dir) for y in available_years}
    lengths = {y: len(arrays[y]) for y in available_years}

    offsets = {}
    running = 0
    for y in available_years:
        offsets[y] = running
        running += lengths[y]
    total = running

    def global_index(dt, clamp=False):
        """Row index of ``dt`` in the concatenated stream, by minute-of-year.

        Returns ``None`` if ``dt``'s year has no file, or if its minute-of-year
        is past that year's available rows (a partial/truncated file) -- so a
        datetime beyond a year's data never silently indexes into the next year.
        ``clamp=True`` instead returns that year's last available row.
        """
        if dt.year not in offsets:
            return None
        moy = _minute_of_year(dt)
        if moy >= lengths[dt.year]:
            return offsets[dt.year] + lengths[dt.year] - 1 if clamp else None
        return offsets[dt.year] + moy

    i_start = global_index(start_dt)
    if i_start is None:
        raise ValueError(
            f"start {start_dt:%Y-%m-%dT%H:%M} is beyond the available OMNI data."
        )
    i_end_req = global_index(end_dt, clamp=True)
    last_avail = total - 1
    i_end = last_avail if i_end_req is None else min(i_end_req, last_avail)
    if i_end < i_start:
        raise ValueError("No OMNI output minutes in the requested range.")

    allrows = np.concatenate([arrays[y] for y in available_years], axis=0)

    out_rows = allrows[i_start:i_end + 1]
    keep = last_valid_count(out_rows)
    if keep == 0:
        raise ValueError("OMNI data ends before the requested start (no valid days).")
    n_out = keep
    i_end = i_start + n_out - 1

    # Lead-in: the `window` rows immediately before `i_start` in the contiguous
    # minute stream. Real OMNI year files are full-year and contiguous, so these
    # are the prior year's last `window` minutes. At the very start of the record
    # fewer are available; left-pad those with NaN so indexing stays uniform.
    lead = min(window, i_start)
    i_lead = i_start - lead
    pad = window - lead
    if pad > 0:
        warnings.warn(
            f"Only {lead} of {window} lead-in minutes available before "
            f"{start_dt:%Y-%m-%dT%H:%M} (prior-year OMNI data missing); the first "
            f"output minutes use a shortened trailing average.",
            stacklevel=2,
        )

    block = allrows[i_lead:i_end + 1]
    channels = {}
    for name, col in OMNI_COLS.items():
        vals = block[:, col].astype(float)
        if pad > 0:
            vals = np.concatenate([np.full(pad, np.nan), vals])
        channels[name] = vals

    out_block = allrows[i_start:i_end + 1]
    timestamps = _omni_timestamps(out_block)
    if verbose:
        print(f"  OMNI: {n_out} output minutes "
              f"{timestamps[0]:%Y-%m-%dT%H:%M} -> {timestamps[-1]:%Y-%m-%dT%H:%M}")
    return {
        "channels": channels,
        "timestamps": timestamps,
        "window": window,
        "interpolate": True,
        "source_path": None,
    }


def _omni_timestamps(rows):
    """Build ``datetime`` timestamps from OMNI year/doy/hour/minute columns."""
    out = []
    yc, dc, hc, mc = (OMNI_TIME_COLS["year"], OMNI_TIME_COLS["doy"],
                      OMNI_TIME_COLS["hour"], OMNI_TIME_COLS["minute"])
    for r in rows:
        out.append(datetime(int(r[yc]), 1, 1)
                   + timedelta(days=int(r[dc]) - 1, hours=int(r[hc]), minutes=int(r[mc])))
    return out


# --- OMNI via CDAWeb HAPI (server-side time subsetting) ------------------
#
# The 'asc' source above downloads whole-year omni_min<year>.asc files even for a
# short request. CDAWeb's HAPI server subsets server-side, returning only the
# requested window. OMNI_HRO_1MIN is the same King & Papitashvili high-res 1-min
# product distributed as the SPDF omni_min ASCII, so the variables, fill values
# and 1-minute UTC grid match -- the downstream processing is identical.

HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
HAPI_DATASET = "OMNI_HRO_1MIN"
# HAPI parameter name for each imfgen channel.
HAPI_PARAMS = {
    "bx": "BX_GSE",        # Bx is identical in GSE/GSM (no separate BX_GSM)
    "by": "BY_GSM",
    "bz": "BZ_GSM",
    "swvel": "flow_speed",
    "swden": "proton_density",
}

try:  # optional dependency; only needed for omni_access='hapi'
    from hapiclient import hapi as _hapi
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _hapi = None


def _hapi_stamp(dt):
    """``datetime`` -> HAPI ISO-8601 UTC string (``YYYY-MM-DDTHH:MM:SSZ``)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_hapi_times(time_col):
    """Parse a HAPI isotime column (bytes or str) into naive UTC ``datetime``s."""
    out = []
    for v in time_col:
        s = (v.decode() if isinstance(v, (bytes, bytearray)) else str(v)).strip()
        out.append(datetime.fromisoformat(s.rstrip("Z")))
    return out


def omni_samples_hapi(start_dt, end_dt, window=10, verbose=False):
    """Assemble OMNI channels via CDAWeb HAPI for ``[start_dt, end_dt]``.

    Queries ``OMNI_HRO_1MIN`` for only the requested window plus ``window``
    lead-in minutes (so the trailing average is complete at ``start_dt``), using
    the server's time subsetting -- no whole-year download. Returns the same dict
    shape as :func:`omni_samples`. HAPI ``time.max`` is exclusive, so the request
    upper bound is ``end_dt`` plus one minute to keep the final output minute.
    """
    if _hapi is None:
        raise ImportError(
            "OMNI access mode 'hapi' requires the 'hapiclient' package "
            "(pip install hapiclient), or use omni_access='asc'."
        )
    params = ",".join(HAPI_PARAMS.values())
    lead_start = start_dt - timedelta(minutes=window)
    if verbose:
        print(f"  HAPI {HAPI_DATASET}: {lead_start:%Y-%m-%dT%H:%M} -> "
              f"{end_dt:%Y-%m-%dT%H:%M} ({params})")
    data, _meta = _hapi(
        HAPI_SERVER, HAPI_DATASET, params,
        _hapi_stamp(lead_start), _hapi_stamp(end_dt + timedelta(minutes=1)),
    )
    if data is None or len(data) == 0:
        raise ValueError(
            f"CDAWeb HAPI returned no OMNI data for "
            f"{start_dt:%Y-%m-%dT%H:%M} -> {end_dt:%Y-%m-%dT%H:%M}."
        )

    times = _parse_hapi_times(data["Time"])
    start_idx = next((i for i, t in enumerate(times) if t >= start_dt), None)
    if start_idx is None:
        raise ValueError("No OMNI output minutes in the requested range.")
    timestamps = times[start_idx:]
    n_out = len(timestamps)

    # Lead-in: the `window` minutes before `start_dt`. Fewer are available only at
    # the very start of the OMNI record; left-pad those with NaN (shortened avg),
    # matching the 'asc' path.
    lead = start_idx
    pad = window - lead
    if pad > 0:
        warnings.warn(
            f"Only {lead} of {window} lead-in minutes available before "
            f"{start_dt:%Y-%m-%dT%H:%M}; the first output minutes use a shortened "
            f"trailing average.",
            stacklevel=2,
        )

    channels = {}
    for name, param in HAPI_PARAMS.items():
        vals = np.asarray(data[param], dtype=float)
        if pad > 0:
            vals = np.concatenate([np.full(pad, np.nan), vals])
        channels[name] = vals

    if verbose:
        print(f"  HAPI: {n_out} output minutes "
              f"{timestamps[0]:%Y-%m-%dT%H:%M} -> {timestamps[-1]:%Y-%m-%dT%H:%M}")
    return {
        "channels": channels,
        "timestamps": timestamps,
        "window": window,
        "interpolate": True,
        "source_path": None,
    }


# --- BCWIND source -------------------------------------------------------

BCWIND_KEYS = {"bx": "Bx", "by": "By", "bz": "Bz", "swden": "D", "swvel": "Va"}
BCWIND_TIME_KEY = "UT"
BCWIND_UT_FMT = "%Y-%m-%d %H:%M:%S"


def read_bcwind(path):
    """Read the BCWIND HDF5 datasets used by imfgen into a dict of arrays."""
    import h5py

    out = {}
    with h5py.File(path, "r") as fh:
        for key in list(BCWIND_KEYS.values()) + [BCWIND_TIME_KEY]:
            out[key] = fh[key][()]
    return out


def _decode_ut(ut_values):
    """Parse BCWIND ``UT`` entries into datetimes.

    The original used ``pandas.to_datetime``; we accept the canonical
    ``YYYY-MM-DD HH:MM:SS`` first and fall back to ISO parsing so ``T``
    separators / fractional seconds still work.
    """
    out = []
    for v in ut_values:
        s = (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)).strip()
        try:
            out.append(datetime.strptime(s, BCWIND_UT_FMT))
        except ValueError:
            out.append(datetime.fromisoformat(s))
    return out


def bcwind_samples(path, start_dt=None, end_dt=None):
    """Assemble BCWIND channels + timestamps from an HDF5 file (raw pass-through).

    Note: the output ``date`` is derived per-timestamp downstream
    (:func:`imfgen.dates.date_value`), so it always reflects each sample's true
    year/day. The original ``bcwind_imf.py`` used a monotonic minute counter that
    produced wrong year-day values across a year boundary; this is an intentional
    fix and matches the original only for the common single-year, gap-free file.
    """
    raw = read_bcwind(path)
    timestamps = _decode_ut(raw[BCWIND_TIME_KEY])
    keep = np.ones(len(timestamps), dtype=bool)
    if start_dt is not None:
        keep &= np.array([t >= start_dt for t in timestamps])
    if end_dt is not None:
        keep &= np.array([t <= end_dt for t in timestamps])
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        raise ValueError("No BCWIND samples in the requested range.")
    timestamps = [timestamps[i] for i in idx]
    channels = {name: np.asarray(raw[key], dtype=float)[idx]
                for name, key in BCWIND_KEYS.items()}
    return {
        "channels": channels,
        "timestamps": timestamps,
        "window": 1,
        "interpolate": False,
        "source_path": path,
    }


def fetch(source, **kwargs):
    """Dispatch to a named source. ``source`` is ``"omni"`` or ``"bcwind"``."""
    if source == "omni":
        return omni_samples(**kwargs)
    if source == "bcwind":
        return bcwind_samples(**kwargs)
    raise ValueError(f"Unknown source {source!r}; expected 'omni' or 'bcwind'.")
