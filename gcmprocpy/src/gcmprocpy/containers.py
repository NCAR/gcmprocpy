"""Data classes for gcmprocpy data containers."""
from dataclasses import dataclass
import numpy as np
import xarray as xr


# Model-specific default variable names and configurations.
MODEL_DEFAULTS = {
    'TIE-GCM': {
        'wind_u': 'UN',
        'wind_v': 'VN',
        'wind_w': 'WN',
        'temperature': 'TN',
        'electron_density': 'NE',
        'density': {
            'vars': ['NE', 'DEN', 'O2', 'O1', 'N2', 'NO', 'N4S', 'HE', 'OP',
                     'NMF2', 'TEC'],
            'cmap': 'viridis',
            'line_color': 'white',
        },
        'temperature_type': {
            'vars': ['TN', 'TE', 'TI', 'QJOULE'],
            'cmap': 'inferno',
            'line_color': 'white',
        },
        'wind': {
            'vars': ['UN', 'VN', 'WN', 'UI_ExB', 'VI_ExB', 'WI_ExB'],
            'cmap': 'bwr',
            'line_color': 'black',
        },
        'electric': {
            'vars': ['POTEN'],
            'cmap': 'bwr',
            'line_color': 'black',
        },
        'wind_scale': 0.01,   # cm/s → m/s
        'species': {
            'temp': 'TN', 'o': 'O1', 'o2': 'O2', 'n2': 'N2',
            'no': 'NO', 'co2': 'CO2', 'h': 'H', 'o3': 'O3', 'ho2': 'HO2',
        },
    },
    'WACCM-X': {
        'wind_u': 'U',
        'wind_v': 'V',
        'wind_w': 'OMEGA',
        'temperature': 'T',
        'electron_density': 'EDens',
        'density': {
            'vars': ['EDens', 'OpDens', 'O2p', 'NOp', 'N2p', 'Op',
                     'ElecColDens', 'O3', 'NO', 'NO2', 'N2O', 'CO', 'CO2',
                     'CH4', 'H2O', 'HE', 'O', 'O2', 'N2', 'HNO3', 'NOY',
                     'CLOY', 'BROY'],
            'cmap': 'viridis',
            'line_color': 'white',
        },
        'temperature_type': {
            'vars': ['T', 'TREFHT', 'THETA'],
            'cmap': 'inferno',
            'line_color': 'white',
        },
        'wind': {
            'vars': ['U', 'V', 'OMEGA', 'UTGW_TOTAL', 'VTGW_TOTAL'],
            'cmap': 'bwr',
            'line_color': 'black',
        },
        'electric': {
            'vars': ['ED1', 'ED2', 'POTEN'],
            'cmap': 'bwr',
            'line_color': 'black',
        },
        'radiation': {
            'vars': ['FSDS', 'FSNS', 'FSNT', 'FLDS', 'FLNS', 'FLNT', 'FLUT',
                     'QRL_TOT', 'QRS_TOT', 'QRS_EUV', 'QRS_AUR', 'QTHERMAL',
                     'SWCF', 'LWCF'],
            'cmap': 'plasma',
            'line_color': 'white',
        },
        'wind_scale': 1.0,    # already m/s
        'species': {
            'temp': 'T', 'o': 'O', 'o2': 'O2', 'n2': 'N2',
            'no': 'NO', 'co2': 'CO2', 'h': 'H', 'o3': 'O3', 'ho2': 'HO2',
        },
    },
}


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


def get_species_names(model):
    """Return species name mapping for a model type.

    Uses ``MODEL_DEFAULTS`` as the single source of truth for
    mapping canonical role names to dataset variable names.

    Args:
        model (str): Model type (``'TIE-GCM'`` or ``'WACCM-X'``).

    Returns:
        dict: Mapping from canonical names (e.g. ``'temp'``, ``'o'``,
        ``'o2'``) to dataset variable names (e.g. ``'TN'``, ``'O1'``,
        ``'O2'``).

    Raises:
        ValueError: If *model* is not recognized.
    """
    if model not in MODEL_DEFAULTS:
        raise ValueError(
            f"Unknown model '{model}'. Known: {list(MODEL_DEFAULTS)}"
        )
    return MODEL_DEFAULTS[model]['species']


# ---------------------------------------------------------------------------
# Derived-variable registry
# ---------------------------------------------------------------------------

DERIVED_VARIABLES = {}


def register_derived(name, handler, plot_types=None):
    """Register a derived variable computation handler.

    Args:
        name (str): Variable name (e.g. ``'NO53'``) or a glob-style
            pattern ending with ``*`` (e.g. ``'OH_*'``).
        handler (callable): Function with signature
            ``(datasets, variable_name, time, **kwargs) -> PlotData``.
        plot_types (set, optional): Plot types this variable supports
            (e.g. ``{'lat_lon', 'lev_lat'}``).  *None* means all.
    """
    DERIVED_VARIABLES[name] = {
        'handler': handler,
        'plot_types': plot_types,
    }


def resolve_derived(variable_name):
    """Look up the handler for a derived variable name.

    Checks exact matches first, then pattern matches (keys ending
    with ``*``).

    Args:
        variable_name (str): The variable name to look up.

    Returns:
        tuple: ``(handler, True)`` if found, ``(None, False)`` otherwise.
    """
    # Exact match
    if variable_name in DERIVED_VARIABLES:
        return DERIVED_VARIABLES[variable_name]['handler'], True
    # Also check upper-case form
    vn_upper = variable_name.upper()
    if vn_upper in DERIVED_VARIABLES:
        return DERIVED_VARIABLES[vn_upper]['handler'], True
    # Pattern match (e.g. 'OH_*')
    for key, entry in DERIVED_VARIABLES.items():
        if key.endswith('*') and vn_upper.startswith(key[:-1]):
            return entry['handler'], True
    return None, False
