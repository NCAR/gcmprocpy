"""Tests for the gcmprocpy.gpigen CLI (network mocked)."""

import os

import pytest

from gcmprocpy.gpigen.cli import build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.start == "1960-01-01"
    assert args.source == "json"
    assert args.window == 81
    assert args.trailing is False


def test_parser_trailing_and_window():
    args = build_parser().parse_args(["--window", "27", "--trailing"])
    assert args.window == 27 and args.trailing is True


def test_cli_writes_file(patch_fetch, fake_gfz, tmp_path, capsys):
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    rc = main([
        "--start", "2024-01-01", "--end", "2024-04-30",
        "--output-dir", str(tmp_path), "--quiet",
    ])
    assert rc == 0
    files = [f for f in os.listdir(tmp_path) if f.endswith(".nc")]
    assert files == ["gpi_2024001-2024081.nc"]
    assert "NetCDF file written" in capsys.readouterr().out


def test_cli_no_write(patch_fetch, fake_gfz, tmp_path):
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    rc = main([
        "--start", "2024-01-01", "--end", "2024-04-30",
        "--output-dir", str(tmp_path), "--no-write", "--quiet",
    ])
    assert rc == 0
    assert [f for f in os.listdir(tmp_path) if f.endswith(".nc")] == []


def test_cli_trailing_prefix(patch_fetch, fake_gfz, tmp_path):
    patch_fetch(fake_gfz(start="2023-12-05", n_days=148))
    main([
        "--start", "2024-01-01", "--end", "2024-04-30",
        "--window", "27", "--trailing", "--prefix", "gpi_27avg",
        "--output-dir", str(tmp_path), "--quiet",
    ])
    files = os.listdir(tmp_path)
    assert any(f.startswith("gpi_27avg_") for f in files)


def test_cli_plots(patch_fetch, fake_gfz, tmp_path):
    pytest.importorskip("matplotlib")
    patch_fetch(fake_gfz(start="2023-11-22", n_days=161))
    main([
        "--start", "2024-01-01", "--end", "2024-04-30",
        "--output-dir", str(tmp_path), "--plots",
        "--plots-dir", str(tmp_path / "plots"), "--quiet",
    ])
    pngs = os.listdir(tmp_path / "plots")
    assert any(p.endswith(".png") for p in pngs)
