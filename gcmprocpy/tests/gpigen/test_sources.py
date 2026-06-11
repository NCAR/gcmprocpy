"""Tests for gcmprocpy.gpigen.sources — JSON API (mocked) and text-file parsing."""

from datetime import datetime

import pytest

from gcmprocpy.gpigen import sources
from gcmprocpy.gpigen.sources import fetch, parse_textfile


def test_fetch_unknown_source():
    with pytest.raises(ValueError):
        fetch("nope", datetime(2024, 1, 1), datetime(2024, 1, 2))


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def test_fetch_json_builds_dicts(monkeypatch):
    calls = []

    def fake_get(url):
        calls.append(url)
        if "index=Fobs" in url:
            return _Resp({"datetime": ["2024-01-01T00:00:00Z"], "Fobs": [150.0]})
        return _Resp({"datetime": ["2024-01-01T00:00:00Z"], "Kp": [1.0]})

    monkeypatch.setattr(sources.requests, "get", fake_get)
    fobs, kp = fetch("json", datetime(2024, 1, 1), datetime(2024, 1, 2), status="def")
    assert fobs == {"datetime": ["2024-01-01T00:00:00Z"], "Fobs": [150.0]}
    assert kp == {"datetime": ["2024-01-01T00:00:00Z"], "Kp": [1.0]}
    assert any("index=Fobs" in u for u in calls)
    assert any("index=Kp" in u for u in calls)
    assert all("status=def" in u for u in calls)


def test_fetch_json_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", lambda url: _Resp({}, status=503))
    with pytest.raises(RuntimeError):
        fetch("json", datetime(2024, 1, 1), datetime(2024, 1, 2))


# A couple of rows from the real GFZ combined file format. Columns:
# YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 Ap SN F10.7obs F10.7adj D
SAMPLE_ROWS = """\
# header line to skip
2024 01 01 45291 0 2604 1  0.667 0.333 1.000 1.333 0.667 1.000 1.667 2.000   3  2  4  5  3  4  6  7   4  120  135.7 140.0 0
2024 01 02 45292 0 2604 2  1.000 0.667 0.333 1.000 2.000 2.333 1.667 1.000   4  3  2  4  7  9  6  4   5  118  142.1 145.0 0
2024 01 03 45293 0 2604 3  0.333 0.667 1.000 0.667 1.333 1.000 0.667 0.333   2  3  4  3  5  4  3  2   3  119  -1.0  -1.0  0
"""


def _write_sample(tmp_path):
    path = tmp_path / "kp_sample.txt"
    path.write_text(SAMPLE_ROWS)
    return str(path)


def test_parse_textfile_basic(tmp_path):
    path = _write_sample(tmp_path)
    fobs, kp = parse_textfile(path, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert fobs["datetime"][0] == "2024-01-01T00:00:00Z"
    assert fobs["Fobs"] == [135.7, 142.1, -1.0]  # sentinel preserved
    assert len(kp["Kp"]) == 3 * 8
    assert kp["Kp"][:3] == [0.667, 0.333, 1.000]
    assert kp["datetime"][0] == "2024-01-01T00:00:00Z"
    assert kp["datetime"][1] == "2024-01-01T03:00:00Z"


def test_parse_textfile_date_filter(tmp_path):
    path = _write_sample(tmp_path)
    fobs, kp = parse_textfile(path, datetime(2024, 1, 2), datetime(2024, 1, 2))
    assert fobs["datetime"] == ["2024-01-02T00:00:00Z"]
    assert fobs["Fobs"] == [142.1]
    assert len(kp["Kp"]) == 8


def test_parse_textfile_skips_comments_and_short_rows(tmp_path):
    path = tmp_path / "messy.txt"
    path.write_text("# comment\n\n2024 01 01\n" + SAMPLE_ROWS.splitlines()[1] + "\n")
    fobs, _ = parse_textfile(path, datetime(2024, 1, 1), datetime(2024, 1, 1))
    assert fobs["Fobs"] == [135.7]  # the 3-field row was skipped
