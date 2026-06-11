# GCMPROCPY

GCMprocpy is a post-processing and plot generation tool for [TIE-GCM](https://www.hao.ucar.edu/modeling/tgcm/) and [WACCM-X](https://www.cesm.ucar.edu/models/waccm-x) NetCDF output.

Full documentation: <https://gcmprocpy.readthedocs.io>

## Installation

```bash
pip install gcmprocpy
```

### Requirements

- Python >= 3.8
- cartopy, numpy, matplotlib, xarray, scipy, netcdf4, PySide6, mplcursors, dask, geomag, ipympl, requests, h5py

## Usage

GCMprocpy has three modes:

### 1. GUI

```bash
gcmprocpy
```

Opens an interactive PySide6 window for selecting datasets, plot types, and parameters.
Requires an X11-capable session (`ssh -X` for remote use).

### 2. Python API

```python
import gcmprocpy as gy

datasets = gy.load_datasets('/path/to/output', dataset_filter=None)
fig = gy.plt_lat_lon(datasets, 'TN', time='2022-01-01T12:00:00', level=5.0)
gy.close_datasets(datasets)
```

### 3. Command Line

Each plot type has a console command (`lat_lon`, `lev_var`, `lev_lon`, `lev_lat`, `lev_time`, `lat_time`, `lon_time`, `var_time`, `var_lat`, `var_lon`, `sat_track`):

```bash
lat_lon -dir /path/to/output -var TN -lvl 5.0 -t 2022-01-01T12:00:00 \
        -o_dir ./out -o_file tn_plot -o_format png
```

## Input File Generation

GCMprocpy also builds the geophysical forcing / boundary-condition NetCDF files that
drive a TIE-GCM run, via two console commands and a Python API:

- **`gpigen`** — Geophysical Indices (GPI): daily/averaged 10.7 cm solar flux and the
  3-hourly Kp index, from [GFZ Potsdam](https://kp.gfz-potsdam.de/).
- **`imfgen`** — IMF / solar-wind boundary conditions (Bx/By/Bz, density, velocity),
  from [OMNI](https://omniweb.gsfc.nasa.gov/ow_min.html) 1-minute data or a BCWIND HDF5 file.

```bash
gpigen --start 2024-01-01 --end 2024-06-01            # write a GPI .nc file
imfgen --start 2020-01-01 --end 2020-12-31 --cache-dir ./omni_asc
```

```python
from gcmprocpy import gpigen, imfgen

gpi = gpigen.generate_gpi(start="2024-01-01")          # -> xarray.Dataset
gpigen.save_gpi(gpi, output_dir=".")

imf = imfgen.generate_imf(start="2020-01-01", end="2020-12-31", cache_dir="./omni_asc")
imfgen.save_imf(imf, output_dir=".")
```

## Plot Types

- **Lat vs Lon** maps with mercator, polar, orthographic, mollweide projections, optional coastlines, nightshade, GM equator, wind vector overlays
- **Vertical cross-sections**: Level vs Variable, Level vs Lat, Level vs Lon, Level vs Time
- **Time series**: Lat vs Time, Lon vs Time, Variable vs Time
- **Meridional/zonal cuts**: Variable vs Lat, Variable vs Lon
- **Satellite track interpolation** along arbitrary trajectories

## Features

- **Height interpolation**: Specify levels in km via `level_type='height'` or display vertical axis in km via `y_axis='height'`. Uses ZG (TIE-GCM) or Z3 (WACCM-X) for the conversion.
- **Unit conversion** for temperature, density, and wind variables
- **Derived variables**: emissions (NO 5.3μm, CO2 15μm, OH bands), data density
- **Caching** for fast repeated extraction at the same point/level
