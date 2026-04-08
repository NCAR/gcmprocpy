"""Tests for io module."""
import os
import pytest
import xarray as xr
import numpy as np
from gcmprocpy.io import load_datasets, close_datasets, save_output


class TestLoadDatasets:
    def test_load_single_file(self, tmp_path, tiegcm_dataset):
        """Test loading a single .nc file by path."""
        filepath = tmp_path / "test.nc"
        tiegcm_dataset.to_netcdf(filepath)
        datasets = load_datasets(str(filepath))
        assert len(datasets) == 1
        assert datasets[0][1] == "test.nc"
        assert datasets[0][2] in ('TIE-GCM', 'WACCM-X')
        close_datasets(datasets)

    def test_load_directory(self, tmp_path, tiegcm_dataset):
        """Test loading all .nc files from a directory."""
        (tmp_path / "file1.nc").touch()
        tiegcm_dataset.to_netcdf(tmp_path / "file1.nc")
        tiegcm_dataset.to_netcdf(tmp_path / "file2.nc")
        # Add a non-nc file that should be ignored
        (tmp_path / "readme.txt").write_text("ignore me")

        datasets = load_datasets(str(tmp_path))
        assert len(datasets) == 2
        close_datasets(datasets)

    def test_load_directory_with_filter(self, tmp_path, tiegcm_dataset):
        """Test dataset_filter only loads matching files."""
        tiegcm_dataset.to_netcdf(tmp_path / "run_sech_001.nc")
        tiegcm_dataset.to_netcdf(tmp_path / "run_prim_001.nc")

        datasets = load_datasets(str(tmp_path), dataset_filter='sech')
        assert len(datasets) == 1
        assert 'sech' in datasets[0][1]
        close_datasets(datasets)


class TestCloseDatasets:
    def test_close_without_error(self, tiegcm_datasets):
        """Closing datasets should not raise."""
        close_datasets(tiegcm_datasets)


class TestSaveOutput:
    def test_saves_plot(self, tmp_path):
        """Test that save_output creates a file."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot([1, 2, 3])

        save_output(str(tmp_path), 'test_plot', 'pdf', fig)
        output_path = tmp_path / 'proc' / 'test_plot.pdf'
        assert output_path.exists()
        plt.close(fig)

    def test_creates_proc_directory(self, tmp_path):
        """Test that save_output creates the proc subdirectory."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot([1, 2])

        save_output(str(tmp_path), 'test', 'jpeg', fig)
        assert (tmp_path / 'proc').is_dir()
        plt.close(fig)
