# GCMPROCPY

GCMprocpy is a post-processing and plot generation tool for [TIE-GCM](https://www.hao.ucar.edu/modeling/tgcm/) and [WACCM-X](https://www.cesm.ucar.edu/models/waccm-x) NetCDF output.

Full documentation: <https://gcmprocpy.readthedocs.io>

## Installation

```bash
pip install gcmprocpy
```

### Requirements

- Python >= 3.8
- cartopy, numpy, matplotlib, xarray, scipy, netcdf4, PySide6, mplcursors, dask, geomag, ipympl

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
