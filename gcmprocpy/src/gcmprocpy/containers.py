"""Data classes for gcmprocpy data containers."""
from dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass
class ModelDataset:
    """A loaded NetCDF dataset with its metadata.

    Attributes:
        ds: The opened xarray Dataset.
        filename: The source filename (e.g. 'decsol_smin_2.5x0.25_sech_001.nc').
        model: The model type ('TIE-GCM' or 'WACCM-X').
        _time_set: Cached set of time values for fast lookup.
    """
    ds: xr.Dataset
    filename: str
    model: str
    _time_set: set = None
    _time_values: np.ndarray = None

    def __post_init__(self):
        self._time_values = self.ds['time'].values
        self._time_set = set(self._time_values)

    def has_time(self, time):
        """Fast check whether a timestamp exists in this dataset."""
        return time in self._time_set


@dataclass
class PlotData:
    """Container for data returned by arr_* functions when plot_mode=True.

    Attributes:
        values: The extracted variable values (numpy array).
        variable_unit: The unit string after any conversion.
        variable_long_name: The long descriptive name of the variable.
        model: The model type ('TIE-GCM' or 'WACCM-X').
        filename: The source dataset filename.
        levs: Level/ilevel coordinate array (if applicable).
        lats: Latitude coordinate array (if applicable).
        lons: Longitude coordinate array (if applicable).
        mtime: Single model time as [day, hour, min, sec] (for single-time plots).
        mtime_values: List of model times (for multi-time plots like lev_time, lat_time).
        selected_lat: The latitude value used for selection (if applicable).
        selected_lon: The longitude value used for selection (if applicable).
        selected_lev: The level value used for selection (if applicable).
    """
    values: np.ndarray
    variable_unit: str
    variable_long_name: str
    model: str
    filename: str
    levs: np.ndarray = None
    lats: np.ndarray = None
    lons: np.ndarray = None
    mtime: list = None
    mtime_values: list = None
    selected_lat: float = None
    selected_lon: float = None
    selected_lev: float = None
