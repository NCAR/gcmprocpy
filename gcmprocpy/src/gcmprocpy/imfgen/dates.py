"""Date helpers shared across imfgen.

The IMF files key everything off two representations:

- ``timestamp`` : an ISO-8601 string *without* a trailing ``Z``
  (``YYYY-MM-DDTHH:MM:SS``), one per output minute.
- ``date``      : a float ``YYYYDDD.frac`` -- 4-digit year, 3-digit day of year,
  plus the fractional day, i.e. ``year*1000 + 1 + minute_of_year/1440`` rounded
  to 8 decimals. This is the exact form the original scripts wrote, so it
  reproduces the existing NetCDF files bit-for-bit.
"""

from datetime import datetime, timedelta

ISO_FMT = "%Y-%m-%dT%H:%M:%S"
MINUTES_PER_DAY = 60 * 24


def minute_of_year(dt):
    """Minutes elapsed since Jan 1 00:00 of ``dt``'s year (``0`` at Jan 1 00:00)."""
    return (dt.timetuple().tm_yday - 1) * MINUTES_PER_DAY + dt.hour * 60 + dt.minute


def date_value(dt):
    """``datetime`` -> ``YYYYDDD.frac`` float (``round(.., 8)``).

    Equals ``year*1000 + 1 + minute_of_year/1440`` -- the original scripts'
    formula -- which simplifies to ``year*1000 + dayofyear + (h*60+m)/1440``.
    """
    return round(dt.year * 1000 + 1 + minute_of_year(dt) / MINUTES_PER_DAY, 8)


def to_yearday(dt):
    """``datetime`` -> ``YYYYDDD`` integer (used for filenames / bounds)."""
    return int(dt.strftime("%Y%j"))


def iso_timestamp(dt):
    """``datetime`` -> ``YYYY-MM-DDTHH:MM:SS`` string."""
    return dt.strftime(ISO_FMT)


def coerce_datetime(value, end_of_day=False):
    """Coerce a user-supplied date into a ``datetime``.

    Accepts ``datetime`` objects, ``YYYY-MM-DD`` / ISO strings (with or without a
    trailing ``Z``), or ``YYYYDDD``. ``end_of_day=True`` snaps to 23:59:00 (the
    last output *minute* of the day); otherwise the time is 00:00:00.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().rstrip("Z")
        for fmt in (ISO_FMT, "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y%j"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Unrecognised date {value!r}; use YYYY-MM-DD, YYYYDDD, or ISO."
            )
    if end_of_day:
        return dt.replace(hour=23, minute=59, second=0, microsecond=0)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def yesterday():
    """End-of-day yesterday (23:59), matching the scripts' implicit default end."""
    return (datetime.now() - timedelta(days=1)).replace(
        hour=23, minute=59, second=0, microsecond=0
    )
