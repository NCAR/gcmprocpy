"""Offline tests for the OMNI FTP staleness logic (no network)."""

from datetime import datetime

from gcmprocpy.imfgen.sources import _should_refresh


# _should_refresh returns True ("re-download") when the remote file is NEWER than
# the cadence threshold: the 13th of the current month (once we're past the 13th)
# or the 20th of the previous month (earlier in the month).

def test_should_refresh_past_the_13th():
    now = datetime(2025, 6, 14)                 # threshold = 2025-06-13
    assert _should_refresh(datetime(2025, 6, 12), now) is False
    assert _should_refresh(datetime(2025, 6, 14), now) is True


def test_should_refresh_on_the_13th_uses_prev_month():
    now = datetime(2025, 6, 13)                 # not > 13 -> threshold = 2025-05-20
    assert _should_refresh(datetime(2025, 5, 19), now) is False
    assert _should_refresh(datetime(2025, 5, 21), now) is True


def test_should_refresh_january_wraps_to_december():
    now = datetime(2025, 1, 5)                  # threshold = 2024-12-20
    assert _should_refresh(datetime(2024, 12, 19), now) is False
    assert _should_refresh(datetime(2024, 12, 21), now) is True
