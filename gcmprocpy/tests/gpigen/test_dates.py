"""Tests for gcmprocpy.gpigen.dates."""

from datetime import datetime

import pytest

from gcmprocpy.gpigen.dates import coerce_datetime, date_format, yearday_to_date, yesterday


def test_date_format():
    assert date_format("2024-01-01T00:00:00Z") == "2024001"
    assert date_format("2024-12-31T23:59:59Z") == "2024366"  # leap year


def test_yearday_to_date():
    assert yearday_to_date(2024001) == datetime(2024, 1, 1)
    assert yearday_to_date("2024060") == datetime(2024, 2, 29)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-01-15", datetime(2024, 1, 15)),
        ("2024015", datetime(2024, 1, 15)),
        ("2024-01-15T12:30:00Z", datetime(2024, 1, 15)),
        (datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15)),
    ],
)
def test_coerce_datetime_start(value, expected):
    assert coerce_datetime(value) == expected  # snaps to 00:00:00


def test_coerce_datetime_end_of_day():
    dt = coerce_datetime("2024-01-15", end_of_day=True)
    assert (dt.hour, dt.minute, dt.second) == (23, 59, 59)


def test_coerce_datetime_invalid():
    with pytest.raises(ValueError):
        coerce_datetime("not-a-date")


def test_yesterday_is_end_of_day():
    y = yesterday()
    assert (y.hour, y.minute, y.second) == (23, 59, 59)
