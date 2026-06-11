"""Tests for the gcmprocpy.imfgen CLI (offline, synthetic data)."""

import os

import pytest

from gcmprocpy.imfgen.cli import build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.source == "omni"
    assert args.start is None and args.end is None
    assert args.window == 10
    assert args.split_years is False


def test_cli_omni_writes_file(fake_omni, patch_download, tmp_path, capsys):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 2)
    rc = main([
        "--source", "omni", "--start", "2001-01-01", "--end", "2001-01-02",
        "--cache-dir", cache_dir, "--no-download",
        "--output-dir", str(tmp_path), "--quiet",
    ])
    assert rc == 0
    files = [f for f in os.listdir(tmp_path) if f.endswith(".nc")]
    assert files == ["imf_OMNI_2001001-2001002.nc"]
    assert "NetCDF file written" in capsys.readouterr().out


def test_cli_no_write(fake_omni, patch_download, tmp_path):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 1)
    rc = main([
        "--start", "2001-01-01", "--end", "2001-01-01",
        "--cache-dir", cache_dir, "--no-download",
        "--output-dir", str(tmp_path), "--no-write", "--quiet",
    ])
    assert rc == 0
    assert [f for f in os.listdir(tmp_path) if f.endswith(".nc")] == []


def test_cli_split_years_writes_one_per_year(fake_omni, patch_download, tmp_path):
    cache_dir, writer = fake_omni
    writer(2000, 1)
    writer(2001, 1)
    writer(2002, 1)
    main([
        "--source", "omni", "--start", "2001-01-01", "--end", "2002-12-31",
        "--split-years", "--cache-dir", cache_dir, "--no-download",
        "--output-dir", str(tmp_path), "--quiet",
    ])
    files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".nc"))
    assert files == ["imf_OMNI_2001001-2001001.nc", "imf_OMNI_2002001-2002001.nc"]


def test_cli_split_years_rejects_bcwind(fake_bcwind, tmp_path):
    path = fake_bcwind(n=5)
    with pytest.raises(SystemExit):
        main(["--source", "bcwind", "--bcwind-path", path, "--split-years",
              "--output-dir", str(tmp_path), "--quiet"])


def test_cli_bcwind(fake_bcwind, tmp_path, capsys):
    path = fake_bcwind(n=60)
    rc = main([
        "--source", "bcwind", "--bcwind-path", path,
        "--output-dir", str(tmp_path), "--quiet",
    ])
    assert rc == 0
    files = [f for f in os.listdir(tmp_path) if f.endswith(".nc")]
    assert len(files) == 1 and files[0].startswith("imf_bcwind_")
    assert "NetCDF file written" in capsys.readouterr().out


def test_cli_custom_prefix(fake_bcwind, tmp_path):
    path = fake_bcwind(n=5)
    main(["--source", "bcwind", "--bcwind-path", path, "--prefix", "myimf",
          "--output-dir", str(tmp_path), "--quiet"])
    assert any(f.startswith("myimf_") for f in os.listdir(tmp_path))
