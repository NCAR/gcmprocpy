"""Date helpers shared across gpigen.

GFZ timestamps are ISO-8601 with a trailing ``Z`` (``YYYY-MM-DDTHH:MM:SSZ``).
Internally we key everything off the ``YYYYDDD`` integer (4-digit year + 3-digit
day-of-year) used by the TIEGCM GPI files.
"""

from datetime import datetime, timedelta

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def date_format(timestamp):
    """``YYYY-MM-DDTHH:MM:SSZ`` -> ``YYYYDDD`` string."""
    return datetime.strptime(timestamp, ISO_FMT).strftime("%Y%j")


def yearday_to_date(yearday):
    """``YYYYDDD`` (int or str) -> ``datetime``."""
    return datetime.strptime(str(yearday), "%Y%j")


def coerce_datetime(value, end_of_day=False):
    """Coerce a user-supplied date into a ``datetime``.

    Accepts ``datetime`` objects, ``YYYY-MM-DD`` / ISO strings, or ``YYYYDDD``.
    ``end_of_day=True`` snaps to 23:59:59 (used for the inclusive end bound);
    otherwise the time is 00:00:00.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        for fmt in (ISO_FMT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%j"):
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
        return dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def yesterday():
    """End-of-day yesterday, matching the original scripts' default end."""
    return (datetime.now() - timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
