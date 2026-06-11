"""Tests for gcmprocpy.imfgen.dates."""

from datetime import datetime

import pytest

from gcmprocpy.imfgen.dates import (
    coerce_datetime,
    date_value,
    iso_timestamp,
    minute_of_year,
    to_yearday,
    yesterday,
)


def test_minute_of_year():
    assert minute_of_year(datetime(1982, 1, 1, 0, 0)) == 0
    assert minute_of_year(datetime(1982, 1, 1, 0, 1)) == 1
    assert minute_of_year(datetime(1982, 1, 2, 0, 0)) == 1440
    assert minute_of_year(datetime(1982, 12, 31, 23, 59)) == 525599


def test_date_value_matches_original_formula():
    # round(year*1000 + 1 + minute_of_year/1440, 8)
    assert date_value(datetime(1982, 1, 1, 0, 0)) == 1982001.0
    assert date_value(datetime(1982, 12, 31, 23, 59)) == 1982365.99930556
    # mid-day fractional (matches the bcwind golden's first sample)
    assert date_value(datetime(2024, 5, 9, 18, 0)) == 2024130.75


def test_to_yearday():
    assert to_yearday(datetime(2024, 1, 1)) == 2024001
    assert to_yearday(datetime(2024, 2, 29)) == 2024060  # leap day


def test_iso_timestamp_has_no_trailing_z():
    assert iso_timestamp(datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02T03:04:05"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-01-15", datetime(2024, 1, 15)),
        ("2024015", datetime(2024, 1, 15)),
        ("2024-01-15T12:30:00", datetime(2024, 1, 15)),
        ("2024-01-15T12:30:00Z", datetime(2024, 1, 15)),
        (datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15)),
    ],
)
def test_coerce_datetime_start_snaps_to_midnight(value, expected):
    assert coerce_datetime(value) == expected


def test_coerce_datetime_end_of_day():
    dt = coerce_datetime("2024-01-15", end_of_day=True)
    assert (dt.hour, dt.minute, dt.second) == (23, 59, 0)


def test_coerce_datetime_invalid():
    with pytest.raises(ValueError):
        coerce_datetime("not-a-date")


def test_yesterday_is_end_of_day():
    y = yesterday()
    assert (y.hour, y.minute, y.second) == (23, 59, 0)
