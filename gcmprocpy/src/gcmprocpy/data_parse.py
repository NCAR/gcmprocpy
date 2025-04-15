import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .convert_units import convert_units
import datetime


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
    for ds, filename, model in datasets:
        file = str(filename)
        for timestamp in ds['time'].values:
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

    for ds, filename, model in datasets:
        # Convert the current dataset's variables to a set
        current_variables = set(ds.data_vars)
        # Union the current variables with the existing unique variables
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

    for ds, filename, model in datasets:
        model = model
        levs = ds.lev.values
        ilevs = ds.ilev.values
        unique_levels.update(levs)
        unique_levels.update(ilevs)
    
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

    for ds, filename, model in datasets:
        # Get longitude values and add them to the set
        lons = ds.lon.values
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

    for ds, filename, model in datasets:
        # Get latitude values and add them to the set
        lats = ds.lat.values
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

    for ds, _, _ in datasets:
        unique_dims.update(ds.dims)

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

    for ds, filename, model in datasets:
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

    for ds, filename, model in datasets:
        if dimension in ds.dims:
            # Gather dimension details
            dim_details = {
                "size": ds.dims[dimension]
            }
            
            # Check if the dimension is a coordinate and add more details if it is
            if dimension in ds.coords:
                dim_details["values"] = ds.coords[dimension].data  # Coordinate values as array-like (avoiding .tolist())
                dim_details["attributes"] = ds.coords[dimension].attrs  # Additional attributes
            
            dimension_info[filename] = dim_details
        else:
            dimension_info[filename] = None  # If dimension does not exist in the dataset
    
    return dimension_info

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
    for ds, filenames, model in datasets:
        if time in ds['time'].values:
            # Extract variable attributes
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit is None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            selected_mtime = get_mtime(ds, time)
            filename = filenames
            data = ds[variable_name].sel(time=time)

            not_all_nan_indices = ~np.isnan(data.values).all(axis=(1,2))
            variable_values = data.values[not_all_nan_indices, :, :]

            if selected_unit is not None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except:
                levs_ilevs = data.ilev.values[not_all_nan_indices]

            levs_ilevs = level_log_transform(levs_ilevs, model, log_level)

            if plot_mode:
                return (variable_values, levs_ilevs, variable_unit, variable_long_name, selected_mtime, model, filename)
            else:
                return variable_values
    print(f"{time} not found.")
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
    for ds, filenames, model in datasets:
        if time in ds['time'].values:
            # Extracting variable attributes and time information
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            selected_mtime = get_mtime(ds,time)
            filename = filenames
            # Data selection based on latitude
            if selected_lat == "mean":
                # Averaging over all latitudes
                data = ds[variable_name].sel(time=time).mean(dim='lat')
            else:
                # Nearest latitude selection
                data = ds[variable_name].sel(time=time, lat=selected_lat, method='nearest')
            lons = data.lon.values

            # Filtering non-NaN data
            not_all_nan_indices = ~np.isnan(data.values).all(axis=1)
            variable_values = data.values[not_all_nan_indices, :]

            # Unit conversion if a different unit is specified
            if selected_unit != None:
                variable_values, variable_unit = convert_units(variable_values, variable_unit, selected_unit)

            # Extracting level or ilevel values
            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except:
                levs_ilevs = data.ilev.values[not_all_nan_indices]

            levs_ilevs = level_log_transform(levs_ilevs, model, log_level)

            # Conditional return based on plot_mode
            if plot_mode == True:    
                return variable_values, lons, levs_ilevs, selected_lat, variable_unit, variable_long_name, selected_mtime, model, filename
            else:
                return variable_values

    # Handling cases where the specified time is not found in the dataset
    print(f"{time} not found.")
    return None



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

    if selected_lev_ilev != None and selected_lev_ilev != "mean":
        selected_lev_ilev = float(selected_lev_ilev)
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    first_pass = True
    for ds, filenames, model in datasets:
        if first_pass == True:
            lev_ilev = check_var_dims(ds, variable_name)
        if lev_ilev == 'lev':
            first_pass == False
            if time in ds['time'].values:
                if 'lev' not in ds[variable_name].dims:
                    raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'lev'")
                    return 0

                # Extract variable attributes
                variable_unit = ds[variable_name].attrs.get('units', 'N/A')
                if variable_unit == 'cm/s' and selected_unit == None:
                    selected_unit = 'm/s'
                variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
                selected_mtime = get_mtime(ds,time)
                filename = filenames


                if selected_lev_ilev == "mean":
                    # if selected_lon is "mean", then we calculate the mean over all longitudes.
                    data = ds[variable_name].sel(time=time).mean(dim='lev')
                    lons = data.lon.values
                    lats = data.lat.values
                    variable_values = data.values
                    if selected_unit != None:
                        variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        
                else:
                    # Extract the data for the given time and lev
                    if selected_lev_ilev in ds['lev'].values:
                        data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                        lons = data.lon.values
                        lats = data.lat.values
                        variable_values = data.values
                        if selected_unit != None:
                            variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                            
                    else:
                        print(f"The lev {selected_lev_ilev} isn't in the listed valid values.")
                        lev_max = ds['lev'].max().values.item()
                        lev_min = ds['lev'].min().values.item()
                        if selected_lev_ilev > lev_max:
                            print(f"Using maximun valid lev {lev_max}")
                            selected_lev_ilev = lev_max
                            data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit != None:
                                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        elif selected_lev_ilev < lev_min:
                            print(f"Using minimum valid lev {lev_min}")
                            selected_lev_ilev = lev_min
                            data = ds[variable_name].sel(time=time, lev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit != None:
                                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        else:
                            sorted_levs = sorted(ds['lev'].values, key=lambda x: abs(x - selected_lev_ilev))
                            closest_lev1 = sorted_levs[0]
                            closest_lev2 = sorted_levs[1]
                            print(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                            # Extract data for the two closest lev values using .sel()
                            data1 = ds[variable_name].sel(time=time, lev=closest_lev1)
                            lons = data1.lon.values
                            lats = data1.lat.values
                            variable_values_1 = data1.values

                            data2 = ds[variable_name].sel(time=time, lev=closest_lev2)
                            variable_values_2 = data2.values
                            # Return the averaged data
                            variable_values = (variable_values_1 + variable_values_2) / 2
                            if selected_unit != None:
                                variable_values , variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                if plot_mode == True:    
                    return variable_values, selected_lev_ilev, lats, lons, variable_unit, variable_long_name, selected_mtime, model, filename
                else:
                    return variable_values

        elif lev_ilev == 'ilev':
            first_pass == False
            if time in ds['time'].values:
                if 'ilev' not in ds[variable_name].dims:
                    raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'ilev'")
                    return 0
                            
                # Extract variable attributes
                variable_unit = ds[variable_name].attrs.get('units', 'N/A')
                if variable_unit == 'cm/s' and selected_unit == None:
                    selected_unit = 'm/s'
                variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
                selected_mtime=get_mtime(ds,time)
                filename = filenames

                if selected_lev_ilev == "mean":
                    # if selected_lon is "mean", then we calculate the mean over all longitudes.
                    data = ds[variable_name].sel(time=time).mean(dim='lev')
                    lons = data.lon.values
                    lats = data.lat.values
                    variable_values = data.values
                    if selected_unit != None:
                        variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        
                else:
                    # Extract the data for the given time and lev
                    if selected_lev_ilev in ds['ilev'].values:
                        data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                        lons = data.lon.values
                        lats = data.lat.values
                        variable_values = data.values
                        if selected_unit != None:
                            variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                            
                    else:
                        
                        print(f"The ilev {selected_lev_ilev} isn't in the listed valid values.")
                        ilev_max = ds['ilev'].max().values.item()
                        ilev_min = ds['ilev'].min().values.item()
                        if selected_lev_ilev > ilev_max:
                            print(f"Using maximun valid ilev {ilev_max}")
                            selected_lev_ilev = ilev_max
                            data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit != None:
                                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        elif selected_lev_ilev < ilev_min:
                            print(f"Using minimum valid ilev {ilev_min}")
                            selected_lev_ilev = ilev_min
                            data = ds[variable_name].sel(time=time, ilev=selected_lev_ilev)
                            lons = data.lon.values
                            lats = data.lat.values
                            variable_values = data.values
                            if selected_unit != None:
                                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                        else:
                            sorted_levs = sorted(ds['ilev'].values, key=lambda x: abs(x - selected_lev_ilev))
                            closest_lev1 = sorted_levs[0]
                            closest_lev2 = sorted_levs[1]
                            print(f"Averaging from the closest valid ilevs: {closest_lev1} and {closest_lev2}")
                            # Extract data for the two closest lev values using .sel()
                            data1 = ds[variable_name].sel(time=time, ilev=closest_lev1)
                            lons = data1.lon.values
                            lats = data1.lat.values
                            variable_values_1 = data1.values

                            data2 = ds[variable_name].sel(time=time, ilev=closest_lev2)
                            variable_values_2 = data2.values
                            # Return the averaged data
                            variable_values = (variable_values_1 + variable_values_2) / 2
                            if selected_unit != None:
                                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                            
                if plot_mode == True:    
                    return variable_values, selected_lev_ilev, lats, lons, variable_unit, variable_long_name, selected_mtime, model, filename
                else:
                    return variable_values

        elif lev_ilev == None:
            first_pass == False
            selected_lev_ilev = None
            if time in ds['time'].values:

                # Extract variable attributes
                variable_unit = ds[variable_name].attrs.get('units', 'N/A')
                if variable_unit == 'cm/s' and selected_unit == None:
                    selected_unit = 'm/s'
                variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
                selected_mtime = get_mtime(ds,time)
                filename = filenames

                # Extract the data for the given time and lev
                data = ds[variable_name].sel(time=time)
                lons = data.lon.values
                lats = data.lat.values
                variable_values = data.values
                if selected_unit != None:
                    variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                
                
                if plot_mode == True:    
                    return variable_values, selected_lev_ilev, lats, lons, variable_unit, variable_long_name, selected_mtime, model, filename
                else:
                    return variable_values



    
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

    
    
    for ds, filenames, model in datasets:
        if time in ds['time'].values:

            if selected_lon == "mean" and selected_lat == "mean":
                # if selected_lon is "mean", then we calculate the mean over all longitudes.
                data = ds[variable_name].sel(time=time).mean(dim=['lon', 'lat'])
            elif selected_lon == "mean":
                data = ds[variable_name].sel(time=time, lat=selected_lat, method="nearest").mean(dim='lon')  #look into method nearest
            elif selected_lat == "mean":
                data = ds[variable_name].sel(time=time, lon=selected_lon).mean(dim='lat')
            else:
                data = ds[variable_name].sel(time=time, lat=selected_lat, lon=selected_lon, method="nearest")


            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            selected_mtime=get_mtime(ds,time)
            filename = filenames
            valid_indices = ~np.isnan(data.values)
            variable_values = data.values[valid_indices]
            if selected_unit != None:
                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                
            try:
                levs_ilevs = ds['lev'].values[valid_indices]
            except:
                levs_ilevs = ds['ilev'].values[valid_indices]
            
            levs_ilevs = level_log_transform(levs_ilevs, model, log_level)
            
            if plot_mode == True:
                return variable_values , levs_ilevs, variable_unit, variable_long_name, selected_mtime, model, filename
            else:
                return variable_values 
    print(f"{time} not found.")
    return None




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
    for ds, filenames, model in datasets:
        if time in ds['time'].values:
            # Extract variable attributes
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            selected_mtime = get_mtime(ds,time)
            filename = filenames
            if selected_lon == "mean":
                # if selected_lon is "mean", then we calculate the mean over all longitudes.
                data = ds[variable_name].sel(time=time).mean(dim='lon')
            else:
                selected_lon = float(selected_lon)
                data = ds[variable_name].sel(time=time, lon=selected_lon, method='nearest')
            lats = data.lat.values

            not_all_nan_indices = ~np.isnan(data.values).all(axis=1)
            variable_values = data.values[not_all_nan_indices, :]
            if selected_unit != None:
                variable_values ,variable_unit  = convert_units (variable_values, variable_unit, selected_unit)
                
            try:
                levs_ilevs = data.lev.values[not_all_nan_indices]
            except AttributeError:
                levs_ilevs = data.ilev.values[not_all_nan_indices]
            
            lev_ilevs = level_log_transform(levs_ilevs, model, log_level)

            if plot_mode == True:
                return variable_values, lats, levs_ilevs, selected_lon, variable_unit, variable_long_name, selected_mtime, model, filename
            else:
                return variable_values
    print(f"{time} not found.")
    return None



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
    except:
        selected_lon = selected_lon
    if selected_lon == 180:
            selected_lon = -180
    variable_values_all = []
    combined_mtime = []
    levs_ilevs_all = []
    avg_info_print = 0
    for ds, filenames, model in datasets:
        variable_unit = ds[variable_name].attrs.get('units', 'N/A')
        if variable_unit == 'cm/s' and selected_unit == None:
            selected_unit = 'm/s'
        variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
        try:
            mtime_values = ds['mtime'].values
        except:
            mtime_values = []
            for timestamp in ds['time'].values:
                mtime_values.append(get_mtime(ds, timestamp))
        if selected_lon == "mean" and selected_lat == "mean":
            # if selected_lon is "mean", then we calculate the mean over all longitudes. This doesn't work fix
            data = ds[variable_name].mean(dim=['lon', 'lat'])
            variable_values = data.T 
        elif selected_lon == "mean":
            if selected_lat in ds['lat'].values:
                data = ds[variable_name].sel(lat=selected_lat).mean(dim='lon')
                variable_values = data.T 
            else:
                sorted_lats = sorted(ds['lat'].values, key=lambda x: abs(x - selected_lat))
                closest_lat1 = sorted_lats[0]
                closest_lat2 = sorted_lats[1]
                if avg_info_print == 0:
                    print(f"The lat {selected_lat} isn't in the listed valid values.")
                    print(f"Averaging from the closest valid levs: {closest_lat1} and {closest_lat2}")
                    avg_info_print = 1
                data1 = ds[variable_name].sel(lat=closest_lat1, method='nearest').mean(dim='lon')
                variable_values_1 = data1.T 
                data2 = ds[variable_name].sel(lat=closest_lat2, method='nearest').mean(dim='lon')
                variable_values_2 = data2.T 
                variable_values = (variable_values_1 + variable_values_2) / 2
        elif selected_lat == "mean":
            if selected_lon in ds['lon'].values:
                data = ds[variable_name].sel(lon=selected_lon).mean(dim='lat')
                variable_values = data.T 
            else:
                sorted_lons = sorted(ds['lat'].values, key=lambda x: abs(x - selected_lon))
                closest_lon1 = sorted_lons[0]
                closest_lon2 = sorted_lons[1]
                if avg_info_print == 0:
                    print(f"The lon {selected_lon} isn't in the listed valid values.")
                    print(f"Averaging from the closest valid levs: {closest_lon1} and {closest_lon2}")
                    avg_info_print = 1
                data1 = ds[variable_name].sel(lon=closest_lon1, method='nearest').mean(dim='lat')
                variable_values_1 = data1.T 
                data2 = ds[variable_name].sel(lon=closest_lon2, method='nearest').mean(dim='lat')
                variable_values_2 = data2.T 
                variable_values = (variable_values_1 + variable_values_2) / 2
        else:
            #data = ds[variable_name].sel(time=time, lat=selected_lat, lon=selected_lon, method="nearest")    
            data = ds[variable_name].sel(lat=selected_lat, lon=selected_lon, method='nearest')
            variable_values = data.T 
        try:
            levs_ilevs = data.lev.values
        except:
            levs_ilevs = data.ilev.values


        # Adjusting levs_ilevs to match the shape of variable_values
        levs_ilevs = levs_ilevs[:variable_values.shape[0]]


        variable_values_all.append(variable_values)
        combined_mtime.extend(mtime_values)
        levs_ilevs_all.append(levs_ilevs)
    
    # Concatenate data along the time dimension
    variable_values_all = np.concatenate(variable_values_all, axis=1)
    mtime_values = combined_mtime
    
    # Mask out levels with all NaN values
    mask = ~np.isnan(variable_values_all).all(axis=1)
    variable_values_all = variable_values_all[mask, :]
    if selected_unit != None:
        variable_values_all ,variable_unit  = convert_units (variable_values_all, variable_unit, selected_unit)
        
    min_lev_size = min([len(lev) for lev in levs_ilevs_all])
    levs_ilevs = levs_ilevs_all[0][:min_lev_size][mask[:min_lev_size]]

    levs_ilevs = level_log_transform(levs_ilevs, model, log_level)
    if plot_mode == True:
        return variable_values_all, levs_ilevs, mtime_values, selected_lon, variable_unit, variable_long_name, model
    else:
        return variable_values_all

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

    if selected_lev_ilev != 'mean' and selected_lev_ilev != None:
        selected_lev_ilev = float(selected_lev_ilev)
    if selected_lon !='mean':
        selected_lon = float(selected_lon)
    first_pass = True
    variable_values_all = []
    combined_mtime = []
    lats_all = []
    avg_info_print = 0

    for ds, filenames, model in datasets:
        if first_pass == True:
            lev_ilev = check_var_dims(ds, variable_name)
        if lev_ilev == 'lev':
            first_pass == False            
            if 'lev' not in ds[variable_name].dims:
                raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'lev'")
                return 0
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            try:
                mtime_values = ds['mtime'].values
            except:
                mtime_values = []
                for timestamp in ds['time'].values:
                    mtime_values.append(get_mtime(ds, timestamp))
            filename = filenames
            if selected_lon == 'mean' and selected_lev_ilev == 'mean':
                data = ds[variable_name].sel(method='nearest').mean(dim=['lev', 'lon'])
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            if selected_lon =='mean' and selected_lev_ilev != 'mean':
                if selected_lev_ilev in ds['lev'].values:
                    data = ds[variable_name].sel(lev=selected_lev_ilev, method='nearest').mean(dim='lon')
                    variable_values = data.T 
                    lats = data.lat.values    
                    lats_all = lats_all[:variable_values.shape[0]]
                else:
                    sorted_levs = sorted(ds['lev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]
                    if avg_info_print == 0:
                        print(f"The lev {selected_lev_ilev} isn't in the listed valid values.")
                        print(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                        avg_info_print = 1
                    data1 = ds[variable_name].sel(lev=closest_lev1, method='nearest').mean(dim='lon')
                    variable_values_1 = data1.T 
                    lats = data1.lat.values    
                    lats_all = lats_all[:variable_values_1.shape[0]]
                    data2 = ds[variable_name].sel(lev=closest_lev2, method='nearest').mean(dim='lon')
                    variable_values_2 = data2.T 
                    variable_values = (variable_values_1 + variable_values_2) / 2
            if selected_lon !='mean' and selected_lev_ilev == 'mean':
                data = ds[variable_name].sel(lon=selected_lon, method='nearest').mean(dim='lev')
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            if selected_lon !='mean' and selected_lev_ilev != 'mean':
                if selected_lev_ilev in ds['lev'].values:
                    data = ds[variable_name].sel(lev=selected_lev_ilev, lon=selected_lon, method='nearest')
                    variable_values = data.T 
                    lats = data.lat.values    
                    lats_all = lats_all[:variable_values.shape[0]]
                else:
                    sorted_levs = sorted(ds['lev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]
                    if avg_info_print == 0:
                        print(f"The lev {selected_lev_ilev} isn't in the listed valid values.")
                        print(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                        avg_info_print = 1
                    data1 = ds[variable_name].sel(lev=closest_lev1, lon=selected_lon, method='nearest')
                    variable_values_1 = data1.T 
                    lats = data1.lat.values    
                    lats_all = lats_all[:variable_values_1.shape[0]]
                    data2 = ds[variable_name].sel(lev=closest_lev2, lon=selected_lon, method='nearest')
                    variable_values_2 = data2.T 
                    variable_values = (variable_values_1 + variable_values_2) / 2
            variable_values_all.append(variable_values)
            combined_mtime.extend(mtime_values)
            lats_all.append(lats)
        
        elif lev_ilev == 'ilev':
            first_pass == False
            avg_info_print = 0
            if 'ilev' not in ds[variable_name].dims:
                raise ValueError("The variable "+variable_name+" doesn't use the dimensions 'lat', 'lon', 'ilev'")
                return 0
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            try:
                mtime_values = ds['mtime'].values
            except:
                mtime_values = []
                for timestamp in ds['time'].values:
                    mtime_values.append(get_mtime(ds, timestamp))
            filename = filenames
            
            if selected_lon == 'mean' and selected_lev_ilev == 'mean':
                data = ds[variable_name].sel(method='nearest').mean(dim=['ilev', 'lon'])
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            if selected_lon =='mean' and selected_lev_ilev != 'mean':
                if selected_lev_ilev in ds['ilev'].values:
                    data = ds[variable_name].sel(ilev=selected_lev_ilev, method='nearest').mean(dim='lon')
                    variable_values = data.T 
                    lats = data.lat.values    
                    lats_all = lats_all[:variable_values.shape[0]]
                else:
                    sorted_levs = sorted(ds['ilev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]
                    if avg_info_print == 0:
                        print(f"The ilev {selected_lev_ilev} isn't in the listed valid values.")
                        print(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                        avg_info_print = 1
                    data1 = ds[variable_name].sel(ilev=closest_lev1, method='nearest').mean(dim='lon')
                    variable_values_1 = data1.T 
                    lats = data1.lat.values    
                    lats_all = lats_all[:variable_values_1.shape[0]]
                    data2 = ds[variable_name].sel(ilev=closest_lev2, method='nearest').mean(dim='lon')
                    variable_values_2 = data2.T 
                    variable_values = (variable_values_1 + variable_values_2) / 2
            if selected_lon !='mean'  and selected_lev_ilev == 'mean':
                data = ds[variable_name].sel(lon=selected_lon, method='nearest').mean(dim='ilev')
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            elif selected_lon !='mean'  and selected_lev_ilev != 'mean':
                if selected_lev_ilev in ds['ilev'].values:
                    data = ds[variable_name].sel(ilev=selected_lev_ilev, lon=selected_lon, method='nearest')
                    variable_values = data.T 
                    lats = data.lat.values    
                    lats_all = lats_all[:variable_values.shape[0]]
                else:
                    sorted_levs = sorted(ds['ilev'].values, key=lambda x: abs(x - selected_lev_ilev))
                    closest_lev1 = sorted_levs[0]
                    closest_lev2 = sorted_levs[1]
                    if avg_info_print == 0:
                        print(f"The ilev {selected_lev_ilev} isn't in the listed valid values.")
                        print(f"Averaging from the closest valid levs: {closest_lev1} and {closest_lev2}")
                        avg_info_print = 1
                    data1 = ds[variable_name].sel(ilev=closest_lev1, lon=selected_lon, method='nearest')
                    variable_values_1 = data1.T 
                    lats = data1.lat.values    
                    lats_all = lats_all[:variable_values_1.shape[0]]
                    data2 = ds[variable_name].sel(ilev=closest_lev2, lon=selected_lon, method='nearest')
                    variable_values_2 = data2.T 
                    variable_values = (variable_values_1 + variable_values_2) / 2

            variable_values_all.append(variable_values)
            combined_mtime.extend(mtime_values)
            lats_all.append(lats)
            

        elif lev_ilev == None:
            first_pass == False
            selected_lev_ilev = None

            avg_info_print = 0
            variable_unit = ds[variable_name].attrs.get('units', 'N/A')
            if variable_unit == 'cm/s' and selected_unit == None:
                selected_unit = 'm/s'
            variable_long_name = ds[variable_name].attrs.get('long_name', 'N/A')
            try:
                mtime_values = ds['mtime'].values
            except:
                mtime_values = []
                for timestamp in ds['time'].values:
                    mtime_values.append(get_mtime(ds, timestamp))
            filename = filenames

            if selected_lon =='mean':
                data = ds[variable_name].sel(method='nearest').mean(dim='lon')
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            else:
                data = ds[variable_name].sel(lon=selected_lon, method='nearest')
                variable_values = data.T 
                lats = data.lat.values    
                lats_all = lats_all[:variable_values.shape[0]]
            

            variable_values_all.append(variable_values)
            combined_mtime.extend(mtime_values)
            lats_all.append(lats)
            
    # Concatenate data along the time dimension
    variable_values_all = np.concatenate(variable_values_all, axis=1)
    if selected_unit != None:
        variable_values_all ,variable_unit  = convert_units (variable_values_all, variable_unit, selected_unit)
        
    mtime_values = combined_mtime
    if plot_mode == True:    
        return variable_values_all, lats, mtime_values, selected_lon, variable_unit, variable_long_name, model, filename
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
    for ds, filenames, model in datasets:
        if 'ZG' in ds.variables:
            if time in ds['time'].values:
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
            if time in ds['time'].values:
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

    for ds, filenames, model in datasets:
        # Convert mtime to numpy array for comparison
        mtime_array = np.array(mtime)
        
        # Find the index where mtime matches in the dataset
        idx = np.where(np.all(ds['mtime'].values == mtime_array, axis=1))[0]
        
        if len(idx) == 0:
            continue  # Return None if no matching time is found
        
        # Get the corresponding datetime64 value from the time variable
        time = ds['time'].values[idx][0]
        
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

