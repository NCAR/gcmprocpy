import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dask
from .convert_units import convert_units
from .containers import ModelDataset, PlotData, cache_data_fn
import datetime

logger = logging.getLogger(__name__)


def _extract_var_attrs(ds, variable_name, selected_unit=None):
    """Extract common variable attributes from a dataset.

    Args:
        ds: xarray.Dataset
        variable_name: Name of the variable.
        selected_unit: Desired unit override, or None.

    Returns:
        tuple: (variable_unit, variable_long_name, selected_unit)
    """
    variable_unit = ds[variable_name].attrs.get('units', 'N/A')
    if variable_unit == 'cm/s' and selected_unit is None:
        selected_unit = 'm/s'
    variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
    return variable_unit, variable_long_name, selected_unit


def time_list(datasets):
    """
    Compiles and returns a list of all timestamps present in the provided datasets. 
    This function is particularly useful for aggregating time data from multiple sources.

    Args:
        datasets (list of tuples): Each tuple in the list contains an xarray dataset and its corresponding filename. 
            The function will iterate through each dataset to gather timestamps.

    Returns:
        list of np.datetime64: A list containing all the datetime64 timestamps found in the datasets.
    """
    
    # Extract timestamps from each file
    timestamps = []
    for mds in datasets:
        for timestamp in mds._time_values:
            timestamps.append(timestamp)
    return timestamps

def var_list(datasets):
    """
    Reads all the datasets and returns the variables listed in them.
    
    Args:
        datasets (xarray.Dataset): The loaded dataset opened using xarray.
    
    Returns:
        list: A sorted list of variable entries in the datasets.
    """
    
    unique_variables = set()

    for mds in datasets:
        current_variables = set(mds.ds.data_vars)
        unique_variables = unique_variables.union(current_variables)
    variables = sorted(unique_variables)
    return variables

def level_log_transform(array, model, log_level):
    """
    Applies a logarithmic or exponential transformation to the input array based on the model type and log_level flag.

    Args:
        array (numpy.ndarray): The input array to be transformed.
        model (str): The model type, either 'WACCM-X' or 'TIE-GCM'.
        log_level (bool): A flag indicating whether to apply a logarithmic transformation (True) or an exponential transformation (False).

    Returns:
        numpy.ndarray: The transformed array.
    """
    if model == 'WACCM-X' and log_level:
        array = np.log(array)
    elif model == 'TIE-GCM' and not log_level:
        array = np.exp(array)
    return array

def level_list(datasets, log_level=True):
    """
    Reads all the datasets and returns the unique lev and ilev entries in sorted order.
    
    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.

    Returns:
        lev_ilevs (list): A sorted list of unique lev and ilev entries from the datasets.
    """
    
    unique_levels = set()

    for mds in datasets:
        levs = mds.ds.lev.values
        ilevs = mds.ds.ilev.values
        unique_levels.update(levs)
        unique_levels.update(ilevs)
        model = mds.model

    unique_levels_array = np.array(list(unique_levels))
    lev_ilevs = sorted(level_log_transform(unique_levels_array, model, log_level))

    return lev_ilevs

def lon_list(datasets):
    """
    Reads all the datasets and returns the unique longitude (lon) entries in sorted order.
    
    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.

    Returns:
        list: A sorted list of unique longitude entries from the datasets.
    """
    
    unique_lons = set()

    for mds in datasets:
        lons = mds.ds.lon.values
        unique_lons.update(lons)

    # Convert the set to a sorted list
    lons = sorted(unique_lons)
    return lons

def lat_list(datasets):
    """
    Reads all the datasets and returns the unique latitude (lat) entries in sorted order.
    
    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.

    Returns:
        list: A sorted list of unique latitude entries from the datasets.
    """
    
    unique_lats = set()

    for mds in datasets:
        lats = mds.ds.lat.values
        unique_lats.update(lats)

    # Convert the set to a sorted list
    lats = sorted(unique_lats)
    return lats

def dim_list(datasets):
    """
    Retrieves a sorted list of unique dimension names across all datasets.

    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.

    Returns:
        list: A sorted list of unique dimension names across all datasets.
    """
    
    unique_dims = set()

    for mds in datasets:
        unique_dims.update(mds.ds.dims)

    # Convert the set to a sorted list
    dims = sorted(unique_dims)
    return dims

def var_info(datasets, variable_name):
    """
    Retrieves the attributes and dimension information of a specified variable from all datasets.

    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.
        variable_name (str): The name of the variable to retrieve attributes for.

    Returns:
        dict: A dictionary where keys are filenames and values are dictionaries of attributes for the specified variable.
    """
    
    variable_details = {}

    for mds in datasets:
        ds, filename = mds.ds, mds.filename
        if variable_name in ds:
            # Get attributes and dimension information
            attrs = ds[variable_name].attrs
            dims = ds[variable_name].dims
            variable_details[filename] = {
                "attributes": attrs,
                "dimensions": dims
            }
        else:
            variable_details[filename] = None  # If variable does not exist in dataset
    
    return variable_details

def dim_info(datasets, dimension):
    """
    Retrieves information about a specified dimension's size across all datasets.

    Args:
        datasets (list of tuples): A list of tuples, where each tuple contains an xarray dataset and its filename.
        dimension (str): The name of the dimension to retrieve information for.

    Returns:
        dict: A dictionary where keys are filenames and values are the size of the specified dimension.
              If the dimension does not exist in a dataset, the value is None.
    """
    
    dimension_info = {}

    for mds in datasets:
        ds, filename = mds.ds, mds.filename
        if dimension in ds.dims:
            # Gather dimension details
            dim_details = {
                "size": ds.sizes[dimension]
            }
            
            # Check if the dimension is a coordinate and add more details if it is
            if dimension in ds.coords:
                dim_details["values"] = ds.coords[dimension].data  # Coordinate values as array-like (avoiding .tolist())
                dim_details["attributes"] = ds.coords[dimension].attrs  # Additional attributes
            
            dimension_info[filename] = dim_details
        else:
            dimension_info[filename] = None  # If dimension does not exist in the dataset
    
    return dimension_info

@cache_data_fn
def arr_var(datasets, variable_name, time, selected_unit=None, log_level=True, plot_mode=False):
    """
    Extracts and processes data for a given variable at a specific time from multiple datasets. 
    It also handles unit conversion and provides additional information if needed for plotting.

    Args:
        datasets (list[tuple]): Each tuple contains an xarray dataset and its filename. 
            The function will search each dataset for the specified time and variable.
        variable_name (str): The name of the variable to be extracted.
        time (Union[np.datetime64, str]): The specific time for which data is to be extracted.
        selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        plot_mode (bool, optional): If True, the function returns additional data useful for plotting.

    Returns:
        Union[numpy.ndarray, tuple]: If plot_mode is False, returns only the variable values as a numpy array.
        If plot_mode is True, returns a tuple containing:
            numpy.ndarray: The extracted variable values.
            numpy.ndarray: The corresponding level or ilevel values.
            str: The unit of the variable after conversion (if applicable).
            str: The long descriptive name of the variable.
            numpy.ndarray: Model time array corresponding to the specified time.
            str: The name of the dataset file from which data is extracted.
    """
    for mds in datasets:
        if mds.has_time(time):
            variable_unit, variable_long_name, selected_unit = _extract_var_attrs(mds.ds, variable_name, selected_unit)
            selected_mtime = get_mtime(mds.ds, time)
            data = mds.ds[variable_name].sel(time=time)

            not_all_nan_indices = ~np.isnan(data.values).all(axis=(1,2))
            variable_values = data.values[not_all_nan_indices, :, :]

            if selected_unit is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except (KeyError, AttributeError):
                levs_ilevs = data.ilev.values[not_all_nan_indices]

            levs_ilevs = level_log_transform(levs_ilevs, mds.model, log_level)

            if plot_mode:
                return PlotData(values=variable_values, levs=levs_ilevs, variable_unit=variable_unit,
                                variable_long_name=variable_long_name, mtime=selected_mtime,
                                model=mds.model, filename=mds.filename)
            else:
                return variable_values
    logger.warning(f"{time} not found.")
    return None

def check_var_dims(ds, variable_name):
    """
    Checks the dimensions of a given variable in a dataset to determine if it includes specific dimensions ('lev' or 'ilev').

    Args:
        ds (xarray.Dataset): The dataset in which the variable's dimensions are to be checked.
        variable_name (str): The name of the variable for which dimensions are being checked.

    Returns:
        str: Returns 'lev' if the variable includes the 'lev' dimension, 'ilev' if it includes the 'ilev' dimension, 
             'Variable not found in dataset' if the variable does not exist in the dataset, and None if neither 'lev' nor 'ilev' are dimensions of the variable.
    """

    # Check if the variable exists in the dataset
    if variable_name in ds:
        # Get the dimensions of the variable
        var_dims = ds[variable_name].dims

        # Check for 'lev' and 'ilev' in dimensions
        if 'lev' in var_dims:
            return 'lev'
        elif 'ilev' in var_dims:
            return 'ilev'
        else:
            return None
    else:
        return 'Variable not found in dataset'

@cache_data_fn
def arr_lev_lon (datasets, variable_name, time, selected_lat, selected_unit= None, log_level=True, plot_mode = False):
    """
    Extracts and processes data from the dataset based on a specific variable, time, and latitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset opened using xarray.
        variable_name (str): Name of the variable to extract.
        time (Union[str, numpy.datetime64]): Timestamp to filter the data.
        selected_lat (float): Latitude value to filter the data.
        selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        plot_mode (bool, optional): If True, returns additional information for plotting.

    Returns:
        Union[xarray.DataArray, tuple]: 
            If plot_mode is False, returns an xarray object containing the variable values for the specified time and latitude.
            If plot_mode is True, returns a tuple containing:
                xarray.DataArray: Array of variable values for the specified time and latitude.
                xarray.DataArray: Array of longitude values corresponding to the variable values.
                xarray.DataArray: Array of level or ilevel values where data is not NaN.
                float: The latitude value used for data selection.
                str: Unit of the variable after conversion (if applicable).
                str: Long descriptive name of the variable.
                numpy.ndarray: Array containing Day, Hour, Min of the model run.
                str: Name of the dataset file from which data is extracted.
    """
    # Convert time from string to numpy datetime64 format
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    # Iterate over datasets to find the matching time and extract relevant data
    for mds in datasets:
        if mds.has_time(time):
            variable_unit, variable_long_name, selected_unit = _extract_var_attrs(mds.ds, variable_name, selected_unit)
            selected_mtime = get_mtime(mds.ds, time)
            # Data selection based on latitude
            if selected_lat == "mean":
                data = mds.ds[variable_name].sel(time=time).mean(dim='lat')
            else:
                data = mds.ds[variable_name].sel(time=time, lat=selected_lat, method='nearest')
            lons = data.lon.values

            # Filtering non-NaN data
            not_all_nan_indices = ~np.isnan(data.values).all(axis=1)
            variable_values = data.values[not_all_nan_indices, :]

            if selected_unit is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except (KeyError, AttributeError):
                levs_ilevs = data.ilev.values[not_all_nan_indices]

            levs_ilevs = level_log_transform(levs_ilevs, mds.model, log_level)

            if plot_mode:
                return PlotData(values=variable_values, lons=lons, levs=levs_ilevs,
                                selected_lat=selected_lat, variable_unit=variable_unit,
                                variable_long_name=variable_long_name, mtime=selected_mtime,
                                model=mds.model, filename=mds.filename)
            else:
                return variable_values

    logger.warning(f"{time} not found.")
    return None



@cache_data_fn
def batch_arr_lat_lon(datasets, variable_names, time, selected_lev_ilev=None, selected_unit=None, plot_mode=False):
    """Extract multiple variables at once for the same time and level, avoiding redundant lookups.

    This is used by the emissions functions which need 3-4 variables from the same
    time/level slice. Instead of calling arr_lat_lon N times (each re-finding the
    dataset and re-selecting time/level), this selects time once and extracts all
    variables from the same slice.

    Args:
        datasets: List of ModelDataset objects.
        variable_names: List of variable name strings to extract.
        time: Timestamp to filter the data.
        selected_lev_ilev: Level value, 'mean', or None.
        selected_unit: Desired unit override, or None.
        plot_mode: If True, returns PlotData; if False, returns raw arrays.

    Returns:
        dict: Mapping of variable_name -> PlotData (if plot_mode) or numpy array.
    """
    if selected_lev_ilev is not None and selected_lev_ilev != "mean":
        selected_lev_ilev = float(selected_lev_ilev)
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    for mds in datasets:
        if not mds.has_time(time):
            continue
        ds = mds.ds
        lev_ilev = check_var_dims(ds, variable_names[0])
        selected_mtime = get_mtime(ds, time)

        # Select the time slice once
        ds_t = ds.sel(time=time)

        # Determine level selection
        if lev_ilev in ('lev', 'ilev'):
            coord = lev_ilev
            coord_vals = ds[coord].values
            if selected_lev_ilev == "mean":
                select_fn = lambda var: ds_t[var].mean(dim=coord)
            elif selected_lev_ilev in coord_vals:
                select_fn = lambda var: ds_t[var].sel(**{coord: selected_lev_ilev})
            else:
                coord_max = coord_vals.max()
                coord_min = coord_vals.min()
                if selected_lev_ilev > coord_max:
                    logger.warning(f"Using maximum valid {coord} {coord_max}")
                    selected_lev_ilev = coord_max
                    select_fn = lambda var: ds_t[var].sel(**{coord: selected_lev_ilev})
                elif selected_lev_ilev < coord_min:
                    logger.warning(f"Using minimum valid {coord} {coord_min}")
                    selected_lev_ilev = coord_min
                    select_fn = lambda var: ds_t[var].sel(**{coord: selected_lev_ilev})
                else:
                    sorted_levs = sorted(coord_vals, key=lambda x: abs(x - selected_lev_ilev))
                    lev1, lev2 = sorted_levs[0], sorted_levs[1]
                    logger.warning(f"Averaging from the closest valid {coord}s: {lev1} and {lev2}")
                    select_fn = lambda var: (ds_t[var].sel(**{coord: lev1}) + ds_t[var].sel(**{coord: lev2})) / 2
        else:
            select_fn = lambda var: ds_t[var]

        # Extract all variables from the same slice
        results = {}
        for var_name in variable_names:
            data = select_fn(var_name)
            lons = data.lon.values
            lats = data.lat.values
            variable_values = data.values
            variable_unit, variable_long_name, su = _extract_var_attrs(ds, var_name, selected_unit)
            if su is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, su)
            if plot_mode:
                results[var_name] = PlotData(
                    values=variable_values, selected_lev=selected_lev_ilev, lats=lats, lons=lons,
                    variable_unit=variable_unit, variable_long_name=variable_long_name,
                    mtime=selected_mtime, model=mds.model, filename=mds.filename)
            else:
                results[var_name] = variable_values
        return results
    logger.warning(f"{time} not found.")
    return None


@cache_data_fn
def arr_lat_lon(datasets, variable_name, time, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Extracts data from the dataset based on the specified variable, time, and level (lev/ilev).

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): Name of the variable to extract.
        time (Union[str, numpy.datetime64]): Timestamp to filter the data.
        selected_lev_ilev (Union[float, str], optional): Level value to filter the data. If 'mean', calculates the mean over all levels.
        selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
        plot_mode (bool, optional): If True, returns additional information for plotting.

    Returns:
        Union[xarray.DataArray, tuple]:
            If plot_mode is False, returns an xarray object containing the variable values for the specified time and level.
            If plot_mode is True, returns a tuple containing:
                xarray.DataArray: Array of variable values for the specified time and level.
                Union[float, str]: The level value used for data selection.
                xarray.DataArray: Array of latitude values corresponding to the variable values.
                xarray.DataArray: Array of longitude values corresponding to the variable values.
                str: Unit of the variable after conversion (if applicable).
                str: Long descriptive name of the variable.
                numpy.ndarray: Array containing Day, Hour, Min of the model run.
                str: Name of the dataset file from which data is extracted.
    """

    if selected_lev_ilev is not None and selected_lev_ilev != "mean":
        selected_lev_ilev = float(selected_lev_ilev)
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    first_pass = True
    for mds in datasets:
        ds = mds.ds
        if first_pass:
            lev_ilev = check_var_dims(ds, variable_name)
        if lev_ilev == 'lev':
            first_pass = False
            if mds.has_time(time):
                if 'lev' not in ds[variable_name].dims:
                    raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'lev'")

                variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
                selected_mtime = get_mtime(ds, time)

                if selected_lev_ilev == "mean":
                    data = ds[variable_name].sel(time=time).mean(dim='lev')
                    lons = data.lon.values
                    lats = data.lat.values
                    variable_values = data.values
                    if selected_unit is not None:
                        variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                else:
                    if selected_lev_ilev in ds['lev'].values:
                        data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                        lons = data.lon.values
                        lats = data.lat.values
                        variable_values = data.values
                        if selected_unit is not None:
                            variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                    else:
                        logger.warning(f"The lev {selected_lev_ilev} isn't in the listed valid values.")
                        lev_max = ds['lev'].max().values.item()
                        lev_min = ds['lev'].min().values.item()
                        if selected_lev_ilev > lev_max:
                            logger.warning(f"Using maximum valid lev {lev_max}")
                            selected_lev_ilev = lev_max
                            data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)
                        elif selected_lev_ilev < lev_min:
                            logger.warning(f"Using minimum valid lev {lev_min}")
                            selected_lev_ilev = lev_min
                            data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)
                        else:
                            sorted_levs = sorted(ds['lev'].values, key=lambda x: abs(x - selected_lev_ilev))
                            closest_lev1 = sorted_levs[0]
                            closest_lev2 = sorted_levs[1]
                            logger.warning(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                            data1 = ds[variable_name].sel(time=time, lev=closest_lev1)
                            lons = data1.lon.values
                            lats = data1.lat.values
                            variable_values_1 = data1.values

                            data2 = ds[variable_name].sel(time=time, lev=closest_lev2)
                            variable_values_2 = data2.values
                            variable_values = (variable_values_1 + variable_values_2) / 2
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)
                if plot_mode:
                    return PlotData(values=variable_values, selected_lev=selected_lev_ilev, lats=lats, lons=lons,
                                    variable_unit=variable_unit, variable_long_name=variable_long_name,
                                    mtime=selected_mtime, model=mds.model, filename=mds.filename)
                else:
                    return variable_values

        elif lev_ilev == 'ilev':
            first_pass = False
            if mds.has_time(time):
                if 'ilev' not in ds[variable_name].dims:
                    raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'ilev'")

                variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
                selected_mtime = get_mtime(ds, time)

                if selected_lev_ilev == "mean":
                    data = ds[variable_name].sel(time=time).mean(dim='ilev')
                    lons = data.lon.values
                    lats = data.lat.values
                    variable_values = data.values
                    if selected_unit is not None:
                        variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                else:
                    if selected_lev_ilev in ds['ilev'].values:
                        data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                        lons = data.lon.values
                        lats = data.lat.values
                        variable_values = data.values
                        if selected_unit is not None:
                            variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                    else:
                        logger.warning(f"The ilev {selected_lev_ilev} isn't in the listed valid values.")
                        ilev_max = ds['ilev'].max().values.item()
                        ilev_min = ds['ilev'].min().values.item()
                        if selected_lev_ilev > ilev_max:
                            logger.warning(f"Using maximum valid ilev {ilev_max}")
                            selected_lev_ilev = ilev_max
                            data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)
                        elif selected_lev_ilev < ilev_min:
                            logger.warning(f"Using minimum valid ilev {ilev_min}")
                            selected_lev_ilev = ilev_min
                            data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)
                        else:
                            sorted_levs = sorted(ds['ilev'].values, key=lambda x: abs(x - selected_lev_ilev))
                            closest_lev1 = sorted_levs[0]
                            closest_lev2 = sorted_levs[1]
                            logger.warning(f"Averaging from the closest valid ilevs: {closest_lev1} and {closest_lev2}")
                            data1 = ds[variable_name].sel(time=time, ilev=closest_lev1)
                            lons = data1.lon.values
                            lats = data1.lat.values
                            variable_values_1 = data1.values

                            data2 = ds[variable_name].sel(time=time, ilev=closest_lev2)
                            variable_values_2 = data2.values
                            variable_values = (variable_values_1 + variable_values_2) / 2
                            if selected_unit is not None:
                                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                if plot_mode:
                    return PlotData(values=variable_values, selected_lev=selected_lev_ilev, lats=lats, lons=lons,
                                    variable_unit=variable_unit, variable_long_name=variable_long_name,
                                    mtime=selected_mtime, model=mds.model, filename=mds.filename)
                else:
                    return variable_values

        elif lev_ilev is None:
            first_pass = False
            selected_lev_ilev = None
            if mds.has_time(time):
                variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
                selected_mtime = get_mtime(ds, time)

                data = ds[variable_name].sel(time=time)
                lons = data.lon.values
                lats = data.lat.values
                variable_values = data.values
                if selected_unit is not None:
                    variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

                if plot_mode:
                    return PlotData(values=variable_values, selected_lev=selected_lev_ilev, lats=lats, lons=lons,
                                    variable_unit=variable_unit, variable_long_name=variable_long_name,
                                    mtime=selected_mtime, model=mds.model, filename=mds.filename)
                else:
                    return variable_values



    
@cache_data_fn
def arr_lev_var(datasets, variable_name, time, selected_lat, selected_lon, selected_unit= None, log_level=True, plot_mode = False):
    """
    Extracts data from the dataset for a given variable name, latitude, longitude, and time.

    Args:
        datasets (xarray.Dataset): The loaded dataset opened using xarray.
        variable_name (str): Name of the variable to retrieve.
        time (str): Timestamp to filter the data.
        selected_lat (float): Latitude value.
        selected_lon (float): Longitude value.
        selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        plot_mode (bool, optional): If True, returns additional information for plotting.

    Returns:
        Union[xarray.DataArray, tuple]:
            If plot_mode is False, returns an xarray object containing the variable values.
            If plot_mode is True, returns a tuple containing:
                xarray.DataArray: Array of variable values for the specified time and latitude/longitude.
                xarray.DataArray: Array of level or ilevel values where data is not NaN.
                str: Unit of the variable after conversion (if applicable).
                str: Long descriptive name of the variable.
                numpy.ndarray: Array containing Day, Hour, Min of the model run.
                str: Name of the dataset file from which data is extracted.
    """

    
    
    for mds in datasets:
        if mds.has_time(time):
            ds = mds.ds
            if selected_lon == "mean" and selected_lat == "mean":
                data = ds[variable_name].sel(time=time).mean(dim=['lon', 'lat'])
            elif selected_lon == "mean":
                data = ds[variable_name].sel(time=time, lat=selected_lat, method="nearest").mean(dim='lon')
            elif selected_lat == "mean":
                data = ds[variable_name].sel(time=time, lon=selected_lon).mean(dim='lat')
            else:
                data = ds[variable_name].sel(time=time, lat=selected_lat, lon=selected_lon, method="nearest")

            variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
            selected_mtime = get_mtime(ds, time)
            valid_indices = ~np.isnan(data.values)
            variable_values = data.values[valid_indices]
            if selected_unit is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            try:
                levs_ilevs = ds['lev'].values[valid_indices]
            except KeyError:
                levs_ilevs = ds['ilev'].values[valid_indices]

            levs_ilevs = level_log_transform(levs_ilevs, mds.model, log_level)

            if plot_mode:
                return PlotData(values=variable_values, levs=levs_ilevs, variable_unit=variable_unit,
                                variable_long_name=variable_long_name, mtime=selected_mtime,
                                model=mds.model, filename=mds.filename)
            else:
                return variable_values
    logger.warning(f"{time} not found.")
    return None




@cache_data_fn
def arr_lev_lat (datasets, variable_name, time, selected_lon, selected_unit=None, log_level=True, plot_mode = False):
    """
    Extracts data from a dataset based on the specified variable name, timestamp, and longitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset opened using xarray.
        variable_name (str): Name of the variable to extract.
        time (Union[str, numpy.datetime64]): Timestamp to filter the data.
        selected_lon (Union[float, str]): Longitude to filter the data, or 'mean' for averaging over all longitudes.
        selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        plot_mode (bool, optional): If True, returns additional information for plotting.

    Returns:
        Union[xarray.DataArray, tuple]:
            If plot_mode is False, returns an xarray object containing the variable values for the specified time and longitude.
            If plot_mode is True, returns a tuple containing:
                xarray.DataArray: Array of variable values for the specified time and longitude.
                xarray.DataArray: Array of latitude values corresponding to the variable values.
                xarray.DataArray: Array of level or ilevel values where data is not NaN.
                str: Unit of the variable after conversion (if applicable).
                str: Long descriptive name of the variable.
                numpy.ndarray: Array containing Day, Hour, Min of the model run.
                str: Name of the dataset file from which data is extracted.
    """

    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    for mds in datasets:
        if mds.has_time(time):
            ds = mds.ds
            variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
            selected_mtime = get_mtime(ds, time)
            if selected_lon == "mean":
                data = ds[variable_name].sel(time=time).mean(dim='lon')
            else:
                selected_lon = float(selected_lon)
                data = ds[variable_name].sel(time=time, lon=selected_lon, method='nearest')
            lats = data.lat.values

            not_all_nan_indices = ~np.isnan(data.values).all(axis=1)
            variable_values = data.values[not_all_nan_indices, :]
            if selected_unit is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except AttributeError:
                levs_ilevs = data.ilev.values[not_all_nan_indices]

            levs_ilevs = level_log_transform(levs_ilevs, mds.model, log_level)

            if plot_mode:
                return PlotData(values=variable_values, lats=lats, levs=levs_ilevs,
                                selected_lon=selected_lon, variable_unit=variable_unit,
                                variable_long_name=variable_long_name, mtime=selected_mtime,
                                model=mds.model, filename=mds.filename)
            else:
                return variable_values
    logger.warning(f"{time} not found.")
    return None



@cache_data_fn
def arr_lev_time (datasets, variable_name, selected_lat, selected_lon, selected_unit = None, log_level = True, plot_mode = False):
    """
    This function extracts and processes data from multiple datasets based on specified parameters. It focuses on extracting 
    data across different levels and times for a given latitude and longitude.

    Args:
        datasets (list[tuple]): A list of tuples where each tuple contains an xarray dataset and its filename.
        variable_name (str): The name of the variable to be extracted from the dataset.
        selected_lat (Union[float, str]): The latitude value or 'mean' to average over all latitudes.
        selected_lon (Union[float, str]): The longitude value or 'mean' to average over all longitudes.
        selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        plot_mode (bool, optional): If True, the function returns additional data useful for plotting.

    Returns:
        Union[numpy.ndarray, tuple]:
            If plot_mode is False, returns a numpy array of variable values concatenated across datasets.
            If plot_mode is True, returns a tuple containing:
                numpy.ndarray: Concatenated variable values.
                numpy.ndarray: Corresponding level or ilevel values.
                list: List of model times.
                Union[float, str]: The longitude used for data selection.
                str: The unit of the variable after conversion (if applicable).
                str: The long descriptive name of the variable.
    """

    try:
        selected_lon = float(selected_lon)
    except (ValueError, TypeError):
        selected_lon = selected_lon
    if selected_lon == 180:
            selected_lon = -180

    variable_values_all = []
    combined_mtime = []
    levs_ilevs_all = []
    avg_info_print = 0

    for mds in datasets:
        ds = mds.ds
        variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
        try:
            mtime_values = ds['mtime'].values
        except KeyError:
            mtime_values = [get_mtime(ds, ts) for ts in mds._time_values]
        combined_mtime.extend(mtime_values)

        # Spatial selection — stays lazy until .values below
        if selected_lon == "mean" and selected_lat == "mean":
            data = ds[variable_name].mean(dim=['lon', 'lat'])
        elif selected_lon == "mean":
            if selected_lat in ds['lat'].values:
                data = ds[variable_name].sel(lat=selected_lat).mean(dim='lon')
            else:
                sorted_lats = sorted(ds['lat'].values, key=lambda x: abs(x - selected_lat))
                closest_lat1, closest_lat2 = sorted_lats[0], sorted_lats[1]
                if avg_info_print == 0:
                    logger.warning(f"The lat {selected_lat} isn't in the listed valid values.")
                    logger.warning(f"Averaging from the closest valid lats: {closest_lat1} and {closest_lat2}")
                    avg_info_print = 1
                data = (ds[variable_name].sel(lat=closest_lat1, method='nearest').mean(dim='lon') +
                        ds[variable_name].sel(lat=closest_lat2, method='nearest').mean(dim='lon')) / 2
        elif selected_lat == "mean":
            if selected_lon in ds['lon'].values:
                data = ds[variable_name].sel(lon=selected_lon).mean(dim='lat')
            else:
                sorted_lons = sorted(ds['lon'].values, key=lambda x: abs(x - selected_lon))
                closest_lon1, closest_lon2 = sorted_lons[0], sorted_lons[1]
                if avg_info_print == 0:
                    logger.warning(f"The lon {selected_lon} isn't in the listed valid values.")
                    logger.warning(f"Averaging from the closest valid lons: {closest_lon1} and {closest_lon2}")
                    avg_info_print = 1
                data = (ds[variable_name].sel(lon=closest_lon1, method='nearest').mean(dim='lat') +
                        ds[variable_name].sel(lon=closest_lon2, method='nearest').mean(dim='lat')) / 2
        else:
            data = ds[variable_name].sel(lat=selected_lat, lon=selected_lon, method='nearest')

        variable_values_all.append(data)
        try:
            levs_ilevs_all.append(data.lev.values)
        except (KeyError, AttributeError):
            levs_ilevs_all.append(data.ilev.values)
        model = mds.model

    # Batch-compute all lazy selections in one dask graph, then concatenate
    computed = dask.compute(*variable_values_all)
    variable_values_all = np.concatenate([c.values.T for c in computed], axis=1)

    # Mask out levels with all NaN values
    mask = ~np.isnan(variable_values_all).all(axis=1)
    variable_values_all = variable_values_all[mask, :]
    if selected_unit is not None:
        variable_values_all, variable_unit = convert_units(variable_values_all, variable_unit, selected_unit)

    min_lev_size = min([len(lev) for lev in levs_ilevs_all])
    levs_ilevs = levs_ilevs_all[0][:min_lev_size][mask[:min_lev_size]]
    levs_ilevs = level_log_transform(levs_ilevs, model, log_level)

    if plot_mode:
        return PlotData(values=variable_values_all, levs=levs_ilevs, mtime_values=combined_mtime,
                        selected_lon=selected_lon, variable_unit=variable_unit,
                        variable_long_name=variable_long_name, model=model, filename='')
    else:
        return variable_values_all

@cache_data_fn
def arr_lat_time(datasets, variable_name, selected_lon,selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Extracts and processes data from the dataset based on the specified variable name, longitude, and level/ilev.

    Args:
        datasets (list[tuple]): Each tuple contains an xarray dataset and its filename.
        variable_name (str): The name of the variable to extract.
        selected_lon (Union[float, str]): Longitude value or 'mean' to average over all longitudes.
        selected_lev_ilev (Union[float, str, None]): Level or intermediate level value, 'mean' for averaging, or None if not applicable.
        selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
        plot_mode (bool, optional): If True, returns additional data useful for plotting.

    Returns:
        Union[numpy.ndarray, tuple]:
            If plot_mode is False, returns a numpy array of variable values concatenated across datasets.
            If plot_mode is True, returns a tuple containing:
                numpy.ndarray: Concatenated variable values.
                numpy.ndarray: Latitude values corresponding to the variable values.
                list: List of model times.
                Union[float, str]: The longitude used for data selection.
                str: The unit of the variable after conversion (if applicable).
                str: The long descriptive name of the variable.
                str: Name of the dataset file from which data is extracted.
    """

    if selected_lev_ilev != 'mean' and selected_lev_ilev is not None:
        selected_lev_ilev = float(selected_lev_ilev)
    if selected_lon != 'mean':
        selected_lon = float(selected_lon)

    data_arrays = []
    combined_mtime = []
    avg_info_print = 0
    lev_ilev = None

    for mds in datasets:
        ds = mds.ds
        # Determine lev/ilev once from first dataset
        if lev_ilev is None:
            lev_ilev = check_var_dims(ds, variable_name)

        coord = lev_ilev  # 'lev', 'ilev', or None

        if coord is not None and coord not in ds[variable_name].dims:
            raise ValueError(f"The variable {variable_name} doesn't use the dimensions 'lat', 'lon', '{coord}'")

        variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
        try:
            mtime_values = ds['mtime'].values
        except KeyError:
            mtime_values = [get_mtime(ds, ts) for ts in mds._time_values]
        combined_mtime.extend(mtime_values)

        # Level selection — stays lazy
        if coord is not None and selected_lev_ilev == 'mean':
            data = ds[variable_name].mean(dim=coord)
        elif coord is not None and selected_lev_ilev is not None:
            coord_vals = ds[coord].values
            if selected_lev_ilev in coord_vals:
                data = ds[variable_name].sel(**{coord: selected_lev_ilev}, method='nearest')
            else:
                sorted_levs = sorted(coord_vals, key=lambda x: abs(x - selected_lev_ilev))
                closest_lev1, closest_lev2 = sorted_levs[0], sorted_levs[1]
                if avg_info_print == 0:
                    logger.warning(f"The {coord} {selected_lev_ilev} isn't in the listed valid values.")
                    logger.warning(f"Averaging from the closest valid {coord}s: {closest_lev1} and {closest_lev2}")
                    avg_info_print = 1
                data = (ds[variable_name].sel(**{coord: closest_lev1}, method='nearest') +
                        ds[variable_name].sel(**{coord: closest_lev2}, method='nearest')) / 2
        else:
            data = ds[variable_name]

        # Longitude selection — stays lazy
        if selected_lon == 'mean':
            data = data.mean(dim='lon')
        else:
            data = data.sel(lon=selected_lon, method='nearest')

        data_arrays.append(data)

    # Batch-compute all lazy selections in one dask graph, then concatenate
    computed = dask.compute(*data_arrays)
    variable_values_all = np.concatenate([c.values.T for c in computed], axis=1)
    lats = computed[0].lat.values

    if selected_unit is not None:
        variable_values_all, variable_unit = convert_units(variable_values_all, variable_unit, selected_unit)

    if plot_mode:
        return PlotData(values=variable_values_all, lats=lats, mtime_values=combined_mtime,
                        selected_lon=selected_lon, variable_unit=variable_unit,
                        variable_long_name=variable_long_name, model=mds.model, filename=mds.filename)
    else:
        return variable_values_all


@cache_data_fn
def arr_lon_time(datasets, variable_name, selected_lat, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Extracts and processes data from the dataset based on the specified variable name, latitude, and level/ilev.
    Returns a 2D array of longitudes x time.

    Args:
        datasets (list[ModelDataset]): List of ModelDataset objects.
        variable_name (str): The name of the variable to extract.
        selected_lat (Union[float, str]): Latitude value or 'mean' to average over all latitudes.
        selected_lev_ilev (Union[float, str, None]): Level or intermediate level value, 'mean' for averaging, or None if not applicable.
        selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
        plot_mode (bool, optional): If True, returns a PlotData object.

    Returns:
        Union[numpy.ndarray, PlotData]:
            If plot_mode is False, returns a numpy array of variable values concatenated across datasets.
            If plot_mode is True, returns a PlotData object.
    """

    if selected_lev_ilev != 'mean' and selected_lev_ilev is not None:
        selected_lev_ilev = float(selected_lev_ilev)
    if selected_lat != 'mean':
        selected_lat = float(selected_lat)

    data_arrays = []
    combined_mtime = []
    avg_info_print = 0
    lev_ilev = None

    for mds in datasets:
        ds = mds.ds
        if lev_ilev is None:
            lev_ilev = check_var_dims(ds, variable_name)

        coord = lev_ilev

        if coord is not None and coord not in ds[variable_name].dims:
            raise ValueError(f"The variable {variable_name} doesn't use the dimensions 'lat', 'lon', '{coord}'")

        variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
        try:
            mtime_values = ds['mtime'].values
        except KeyError:
            mtime_values = [get_mtime(ds, ts) for ts in mds._time_values]
        combined_mtime.extend(mtime_values)

        # Level selection
        if coord is not None and selected_lev_ilev == 'mean':
            data = ds[variable_name].mean(dim=coord)
        elif coord is not None and selected_lev_ilev is not None:
            coord_vals = ds[coord].values
            if selected_lev_ilev in coord_vals:
                data = ds[variable_name].sel(**{coord: selected_lev_ilev}, method='nearest')
            else:
                sorted_levs = sorted(coord_vals, key=lambda x: abs(x - selected_lev_ilev))
                closest_lev1, closest_lev2 = sorted_levs[0], sorted_levs[1]
                if avg_info_print == 0:
                    logger.warning(f"The {coord} {selected_lev_ilev} isn't in the listed valid values.")
                    logger.warning(f"Averaging from the closest valid {coord}s: {closest_lev1} and {closest_lev2}")
                    avg_info_print = 1
                data = (ds[variable_name].sel(**{coord: closest_lev1}, method='nearest') +
                        ds[variable_name].sel(**{coord: closest_lev2}, method='nearest')) / 2
        else:
            data = ds[variable_name]

        # Latitude selection
        if selected_lat == 'mean':
            data = data.mean(dim='lat')
        else:
            data = data.sel(lat=selected_lat, method='nearest')

        data_arrays.append(data)

    computed = dask.compute(*data_arrays)
    variable_values_all = np.concatenate([c.values.T for c in computed], axis=1)
    lons = computed[0].lon.values

    if selected_unit is not None:
        variable_values_all, variable_unit = convert_units(variable_values_all, variable_unit, selected_unit)

    if plot_mode:
        return PlotData(values=variable_values_all, lons=lons, mtime_values=combined_mtime,
                        selected_lat=selected_lat, variable_unit=variable_unit,
                        variable_long_name=variable_long_name, model=mds.model, filename=mds.filename)
    else:
        return variable_values_all


@cache_data_fn
def arr_var_time(datasets, variable_name, selected_lat, selected_lon, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Extracts a 1D time series of a variable at a specific lat/lon/level location.

    Args:
        datasets (list[ModelDataset]): List of ModelDataset objects.
        variable_name (str): The name of the variable to extract.
        selected_lat (float): Latitude value.
        selected_lon (float): Longitude value.
        selected_lev_ilev (Union[float, str, None]): Level or intermediate level value, 'mean' for averaging, or None if not applicable.
        selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
        plot_mode (bool, optional): If True, returns a PlotData object.

    Returns:
        Union[numpy.ndarray, PlotData]:
            If plot_mode is False, returns a 1D numpy array of variable values over time.
            If plot_mode is True, returns a PlotData object.
    """

    if selected_lev_ilev != 'mean' and selected_lev_ilev is not None:
        selected_lev_ilev = float(selected_lev_ilev)
    selected_lat = float(selected_lat)
    selected_lon = float(selected_lon)

    data_arrays = []
    combined_mtime = []
    avg_info_print = 0
    lev_ilev = None

    for mds in datasets:
        ds = mds.ds
        if lev_ilev is None:
            lev_ilev = check_var_dims(ds, variable_name)

        coord = lev_ilev

        if coord is not None and coord not in ds[variable_name].dims:
            raise ValueError(f"The variable {variable_name} doesn't use the dimensions 'lat', 'lon', '{coord}'")

        variable_unit, variable_long_name, selected_unit = _extract_var_attrs(ds, variable_name, selected_unit)
        try:
            mtime_values = ds['mtime'].values
        except KeyError:
            mtime_values = [get_mtime(ds, ts) for ts in mds._time_values]
        combined_mtime.extend(mtime_values)

        # Level selection
        if coord is not None and selected_lev_ilev == 'mean':
            data = ds[variable_name].mean(dim=coord)
        elif coord is not None and selected_lev_ilev is not None:
            coord_vals = ds[coord].values
            if selected_lev_ilev in coord_vals:
                data = ds[variable_name].sel(**{coord: selected_lev_ilev}, method='nearest')
            else:
                sorted_levs = sorted(coord_vals, key=lambda x: abs(x - selected_lev_ilev))
                closest_lev1, closest_lev2 = sorted_levs[0], sorted_levs[1]
                if avg_info_print == 0:
                    logger.warning(f"The {coord} {selected_lev_ilev} isn't in the listed valid values.")
                    logger.warning(f"Averaging from the closest valid {coord}s: {closest_lev1} and {closest_lev2}")
                    avg_info_print = 1
                data = (ds[variable_name].sel(**{coord: closest_lev1}, method='nearest') +
                        ds[variable_name].sel(**{coord: closest_lev2}, method='nearest')) / 2
        else:
            data = ds[variable_name]

        # Lat/Lon selection
        data = data.sel(lat=selected_lat, method='nearest')
        data = data.sel(lon=selected_lon, method='nearest')

        data_arrays.append(data)

    computed = dask.compute(*data_arrays)
    variable_values_all = np.concatenate([c.values for c in computed])

    if selected_unit is not None:
        variable_values_all, variable_unit = convert_units(variable_values_all, variable_unit, selected_unit)

    if plot_mode:
        return PlotData(values=variable_values_all, mtime_values=combined_mtime,
                        selected_lat=selected_lat, selected_lon=selected_lon,
                        variable_unit=variable_unit, variable_long_name=variable_long_name,
                        model=mds.model, filename=mds.filename)
    else:
        return variable_values_all


def calc_avg_ht(datasets, time, selected_lev_ilev):
    """
    Compute the average Z value for a given set of latitude, longitude, and level from a dataset.

    Args:
        ds (xarray.Dataset): The loaded dataset opened using xarray.
        time (str): Timestamp to filter the data.
        selected_lev_ilev (float): The level for which to retrieve data.

    Returns:
        float: The average ZG value for the given conditions.
    """

    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    #TIEGCM geoportial height variable is 'ZG'
    for mds in datasets:
        ds = mds.ds
        if 'ZG' in ds.variables:
            if mds.has_time(time):
                if selected_lev_ilev in ds['ilev'].values:
                    heights = ds['ZG'].sel(time=time, ilev=selected_lev_ilev).values
                else:
                    sorted_levs = sorted(ds['ilev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]

                    # Extract data for the two closest lev values using .sel()
                    data1 = ds['ZG'].sel(time=time, ilev=closest_lev1).values
                    data2 = ds['ZG'].sel(time=time, ilev=closest_lev2).values
                    
                    # Return the averaged data
                    heights = (data1.mean() + data2.mean()) / 2
                avg_ht= round(heights.mean()/ 100000, 2)
                return avg_ht
        elif 'Z3' in ds.variables:
            if mds.has_time(time):
                if selected_lev_ilev in ds['lev'].values:
                    heights = ds['Z3'].sel(time=time, lev=selected_lev_ilev).values
                else:
                    sorted_levs = sorted(ds['lev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]

                    # Extract data for the two closest lev values using .sel()
                    data1 = ds['Z3'].sel(time=time, lev=closest_lev1).values
                    data2 = ds['Z3'].sel(time=time, lev=closest_lev2).values
                    #print(data1, data2)
                    # Return the averaged data
                    heights = (data1.mean() + data2.mean()) / 2
                avg_ht= round(heights.mean()/ 1000, 2)
                return avg_ht
    return 0


def _get_height_var(ds):
    """Return (height_var_name, lev_dim, scale_to_km) for TIE-GCM or WACCM-X."""
    if 'ZG' in ds.variables:
        return 'ZG', 'ilev', 1e-5     # ZG is in cm → km
    elif 'Z3' in ds.variables:
        return 'Z3', 'lev', 1e-3      # Z3 is in m → km
    return None, None, None


def height_to_pres_level(datasets, time, target_height_km, latitude=None, longitude=None):
    """
    Convert a target height (km) to the nearest pressure level.

    Finds the pressure level whose average geometric height is closest
    to the requested height. Optionally narrows to a specific lat/lon.

    Args:
        datasets: Loaded datasets.
        time: Timestamp for height lookup.
        target_height_km (float): Desired height in km.
        latitude (float, optional): Latitude to evaluate height at.
        longitude (float, optional): Longitude to evaluate height at.

    Returns:
        float: The pressure level (lev or ilev value) closest to target_height_km.
    """
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    for mds in datasets:
        ds = mds.ds
        if not mds.has_time(time):
            continue
        ht_var, lev_dim, scale = _get_height_var(ds)
        if ht_var is None:
            continue

        heights = ds[ht_var].sel(time=time)
        if latitude is not None and 'lat' in heights.dims:
            lat_vals = ds['lat'].values
            closest_lat = lat_vals[np.abs(lat_vals - latitude).argmin()]
            heights = heights.sel(lat=closest_lat)
        if longitude is not None and 'lon' in heights.dims:
            lon_vals = ds['lon'].values
            closest_lon = lon_vals[np.abs(lon_vals - longitude).argmin()]
            heights = heights.sel(lon=closest_lon)

        # Average over remaining spatial dims to get height per level
        avg_dims = [d for d in heights.dims if d != lev_dim]
        if avg_dims:
            avg_heights = heights.mean(dim=avg_dims).values * scale
        else:
            avg_heights = heights.values * scale

        lev_values = ds[lev_dim].values
        closest_idx = np.abs(avg_heights - target_height_km).argmin()
        return float(lev_values[closest_idx])

    raise ValueError(f"Could not find height variable in datasets for time {time}")


def interpolate_to_height(datasets, variable_values, levs, time,
                          target_heights=None, n_heights=50, log_interp=False):
    """
    Interpolate a field from pressure levels to constant height surfaces.

    Args:
        datasets: Loaded datasets (to access ZG/Z3).
        variable_values (np.ndarray): 2D array (nlev, nlat) or (nlev, nlon) on pressure levels.
        levs (np.ndarray): Pressure level coordinate values matching axis 0 of variable_values.
        time: Timestamp for height field lookup.
        target_heights (np.ndarray, optional): Desired height levels in km.
            If None, auto-generates n_heights levels spanning the data range.
        n_heights (int): Number of height levels if target_heights is None.
        log_interp (bool): If True, use exponential interpolation (for densities).

    Returns:
        tuple: (interpolated_values, target_heights_km)
            interpolated_values: 2D array (n_heights, n_spatial)
            target_heights_km: 1D array of height levels in km
    """
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    # Get height field from datasets
    for mds in datasets:
        ds = mds.ds
        if not mds.has_time(time):
            continue
        ht_var, lev_dim, scale = _get_height_var(ds)
        if ht_var is None:
            continue

        height_field = ds[ht_var].sel(time=time).values * scale  # (nlev, nlat, nlon) in km

        # Average over all spatial dims to get a 1D height profile (nlev,)
        if height_field.ndim == 3:
            height_1d = height_field.mean(axis=(1, 2))
        elif height_field.ndim == 2:
            height_1d = height_field.mean(axis=1)
        elif height_field.ndim == 1:
            height_1d = height_field
        else:
            raise ValueError(f"Unexpected height field shape: {height_field.shape}")

        # Match pressure levels: height field may be on ilev while data is on lev
        ht_levs = ds[lev_dim].values
        if len(height_1d) != len(levs):
            from scipy.interpolate import interp1d
            f = interp1d(ht_levs, height_1d, fill_value='extrapolate')
            height_1d = f(levs)

        # Auto-generate target heights if not provided
        if target_heights is None:
            ht_min = np.nanmin(height_1d)
            ht_max = np.nanmax(height_1d)
            target_heights = np.linspace(ht_min, ht_max, n_heights)

        # Interpolate variable_values from pressure to height at each spatial point
        n_spatial = variable_values.shape[1]
        result = np.full((len(target_heights), n_spatial), np.nan)

        # Use the same 1D height profile for all spatial columns
        for j in range(n_spatial):
            col_heights = height_1d.copy()
            col_values = variable_values[:, j]

            # Sort by height (ascending)
            sort_idx = np.argsort(col_heights)
            col_heights = col_heights[sort_idx]
            col_values = col_values[sort_idx]

            # Remove NaN entries
            valid = ~np.isnan(col_heights) & ~np.isnan(col_values)
            col_heights = col_heights[valid]
            col_values = col_values[valid]

            if len(col_heights) < 2:
                continue

            for k, ht in enumerate(target_heights):
                if ht < col_heights[0] or ht > col_heights[-1]:
                    continue  # out of range

                # Find bracketing levels
                idx = np.searchsorted(col_heights, ht) - 1
                idx = max(0, min(idx, len(col_heights) - 2))

                h0, h1 = col_heights[idx], col_heights[idx + 1]
                v0, v1 = col_values[idx], col_values[idx + 1]

                if log_interp and v0 > 0 and v1 > 0:
                    # Exponential interpolation
                    exparg = (np.log(v1 / v0) / (h1 - h0)) * (ht - h0)
                    result[k, j] = v0 * np.exp(exparg)
                else:
                    # Linear interpolation
                    frac = (ht - h0) / (h1 - h0) if h1 != h0 else 0
                    result[k, j] = v0 + frac * (v1 - v0)

        return result, target_heights

    raise ValueError("Could not find height variable in datasets")


def min_max(variable_values):
    """
    Find the minimum and maximum values of varval from the 2D array.

    Args:
        variable_values (xarray.DataArray): A 2D array of variable values.

    Returns:
        tuple:
            float: Minimum value of the variable in the array.
            float: Maximum value of the variable in the array.
    """

    return np.nanmin(variable_values), np.nanmax(variable_values)

def get_time(datasets, mtime):
    """
    Searches for a specific time in a dataset based on the provided model time (mtime) and returns the corresponding 
    np.datetime64 time value. It iterates through multiple datasets to find a match.

    Args:
        datasets (list[tuple]): Each tuple contains an xarray dataset and its filename. The function will search each dataset for the time value.
        mtime (list[int]): Model time represented as a list of integers in the format [day, hour, minute].

    Returns:
        np.datetime64: The corresponding datetime value in the dataset for the given mtime. Returns None if no match is found.
    """

    for mds in datasets:
        # Convert mtime to numpy array for comparison
        mtime_array = np.array(mtime)

        # Find the index where mtime matches in the dataset
        idx = np.where(np.all(mds.ds['mtime'].values == mtime_array, axis=1))[0]

        if len(idx) == 0:
            continue  # Return None if no matching time is found

        # Get the corresponding datetime64 value from the time variable
        time = mds._time_values[idx][0]

        return time

def get_mtime(ds, time):
    """
    Finds and returns the model time (mtime) array that corresponds to a specific time in a dataset. 
    The mtime is an array representing [Day, Hour, Min].

    Args:
        ds (xarray.Dataset): The dataset opened using xarray, containing time and mtime data.
        time (Union[str, numpy.datetime64]): The timestamp for which the corresponding mtime is to be found.

    Returns:
        numpy.ndarray: The mtime array containing [Day, Hour, Min] for the given timestamp. 
                       Returns None if no corresponding mtime is found.
    """

    # Convert it to a datetime object
    date_dt = time.astype('M8[s]').astype(datetime.datetime)
    # Extract day of year, hour, minute, second
    day_of_year = date_dt.timetuple().tm_yday
    hour = date_dt.hour
    minute = date_dt.minute
    second = date_dt.second
    mtime = [day_of_year, hour, minute, second]
    return mtime


def arr_sat_track(datasets, variable_name, sat_time, sat_lat, sat_lon,
                  selected_lev_ilev=None, selected_unit=None, plot_mode=False):
    """
    Interpolates model data along a satellite trajectory.

    Takes arrays of satellite time/lat/lon points and interpolates the model
    field to those locations using xarray's built-in interpolation.

    Args:
        datasets (list[ModelDataset]): Loaded model datasets.
        variable_name (str): The name of the variable to extract.
        sat_time (array-like): Satellite timestamps as numpy datetime64 values.
        sat_lat (array-like): Satellite latitudes in degrees.
        sat_lon (array-like): Satellite longitudes in degrees.
        selected_lev_ilev (Union[float, str, None]): Level value to extract at,
            'mean' to average over all levels, or None to return all levels.
        selected_unit (str, optional): Desired unit for the variable.
        plot_mode (bool, optional): If True, returns a PlotData object.

    Returns:
        Union[numpy.ndarray, PlotData]:
            If selected_lev_ilev is given: 1D array of shape (n_points,).
            If selected_lev_ilev is None: 2D array of shape (n_levels, n_points).
            If plot_mode is True, returns a PlotData object.
    """
    sat_time = np.asarray(sat_time, dtype='datetime64[ns]')
    sat_lat = np.asarray(sat_lat, dtype=float)
    sat_lon = np.asarray(sat_lon, dtype=float)

    if len(sat_time) != len(sat_lat) or len(sat_time) != len(sat_lon):
        raise ValueError("sat_time, sat_lat, and sat_lon must have the same length")

    lev_ilev = None
    variable_unit = None
    variable_long_name = None

    # Collect interpolated results per dataset
    results = []

    for mds in datasets:
        ds = mds.ds

        if lev_ilev is None:
            lev_ilev = check_var_dims(ds, variable_name)
            variable_unit, variable_long_name, selected_unit = _extract_var_attrs(
                ds, variable_name, selected_unit)

        # Find which satellite points fall within this dataset's time range
        ds_times = mds._time_values
        t_min, t_max = ds_times.min(), ds_times.max()
        mask = (sat_time >= t_min) & (sat_time <= t_max)

        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        t_pts = sat_time[idx]
        lat_pts = sat_lat[idx]
        lon_pts = sat_lon[idx]

        coord = lev_ilev if lev_ilev in ('lev', 'ilev') else None

        # Level selection
        if coord is not None and selected_lev_ilev == 'mean':
            data = ds[variable_name].mean(dim=coord)
        elif coord is not None and selected_lev_ilev is not None:
            data = ds[variable_name].sel(**{coord: float(selected_lev_ilev)}, method='nearest')
        else:
            data = ds[variable_name]

        # Compute from dask before interpolation
        data = data.compute()

        # Interpolate each point along the track
        for i, (t, lat, lon) in enumerate(zip(t_pts, lat_pts, lon_pts)):
            interp_kwargs = {'time': t, 'lat': lat, 'lon': lon}
            val = data.interp(**interp_kwargs, method='linear')
            results.append((idx[i], val.values))

    if not results:
        raise ValueError("No satellite points fall within the dataset time range")

    # Sort by original index and assemble
    results.sort(key=lambda x: x[0])
    values = np.array([r[1] for r in results])

    # values shape: (n_points,) if level selected, (n_points, n_levels) if not
    if values.ndim == 2:
        values = values.T  # -> (n_levels, n_points)

    if selected_unit is not None:
        values, variable_unit = convert_units(values, variable_unit, selected_unit)

    if plot_mode:
        levs = None
        if values.ndim == 2 and lev_ilev in ('lev', 'ilev'):
            levs = datasets[0].ds[lev_ilev].values
        # Build mtime for each satellite point
        sorted_idx = [r[0] for r in sorted(results, key=lambda x: x[0])]
        mtime_values = [get_mtime(None, sat_time[i]) for i in sorted_idx]
        return PlotData(
            values=values, levs=levs, variable_unit=variable_unit,
            variable_long_name=variable_long_name, model=mds.model,
            filename=mds.filename, mtime_values=mtime_values,
            lats=sat_lat[sorted_idx], lons=sat_lon[sorted_idx])
    else:
        return values

