Data Parsing Functions
=============

tiegcmpy provides a range of functions for data extraction and manipulation. Below are the key plotting routines along with their detailed parameters and usage examples.

Data Exploration
-----------------------------------


Listing Timestamps
~~~~~~~~~~~~~~~~~~~
This function compiles and returns a list of all timestamps present in the provided datasets. 

.. function:: time_list(datasets)
Parameters:
    - datasets (list of tuples): Each tuple in the list contains an xarray dataset and its corresponding filename. 

Returns:
    - timestamps (list of np.datetime64): A list containing all the datetime64 timestamps found in the datasets.


Listing Variables
~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and reutrns the variables listed in there.

.. function:: var_list(datasets)
       
Parameters:
    - datasets (xarray): The loaded dataset opened using xarray.

Returns:
    - variables (array): An array of variable entries in the datasets.



Data Xarrays
-----------------------------------

Selected Time
~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data for a given variable at a specific time from multiple datasets. 
It also handles unit conversion and provides additional information if needed for plotting.

.. function:: arr_var(datasets, variable_name, time, selected_unit=None, plot_mode = False)


Parameters:
    - datasets (list of tuples): Each tuple contains an xarray dataset and its filename. The function will search each dataset for the specified time and variable.
    - variable_name (str): The name of the variable to be extracted.
    - time (np.datetime64/str): The specific time for which data is to be extracted.
    - selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
    - plot_mode (bool, optional): If True, the function returns additional data useful for plotting.

Returns:
    - If plot_mode is False, returns only the variable values as a numpy array.
    - If plot_mode is True, returns a tuple containing:
        - variable_values (numpy array): The extracted variable values.
        - levs_ilevs (numpy array): The corresponding level or ilevel values.
        - variable_unit (str): The unit of the variable after conversion (if applicable).
        - variable_long_name (str): The long descriptive name of the variable.
        - selected_ut (float): Universal Time value in hours for the specified time.
        - selected_mtime (array): Model time array corresponding to the specified time.
        - filename (str): The name of the dataset file from which data is extracted.


Selected Time, Level
~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from the dataset based on the specified variable, time, and level (lev/ilev).

.. function:: arr_lat_lon(datasets, variable_name, time, selected_lev_ilev = None, selected_unit = None, plot_mode = False)

    
Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): Name of the variable to extract.
    - time (str/numpy.datetime64): Timestamp to filter the data.
    - selected_lev_ilev (float/str, optional): Level value to filter the data. If 'mean', calculates the mean over all levels.
    - selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
    - plot_mode (bool, optional): If True, returns additional information for plotting.

Returns:
    - If plot_mode is False: An xarray object containing the variable values for the specified time and level.
    - If plot_mode is True: A tuple containing:
        - variable_values (xarray): Array of variable values for the specified time and level.
        - selected_lev_ilev (float/str): The level value used for data selection.
        - lats (xarray): Array of latitude values corresponding to the variable values.
        - lons (xarray): Array of longitude values corresponding to the variable values.
        - variable_unit (str): Unit of the variable after conversion (if applicable).
        - variable_long_name (str): Long descriptive name of the variable.
        - selected_ut (float): Universal Time value in hours for the specified time.
        - selected_mtime (array): Array containing Day, Hour, Min of the model run.
        - filename (str): Name of the dataset file from which data is extracted.


Selected Time, Latitude, Longitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from the dataset for a given variable name, latitude, longitude, and time.

.. function:: arr_lev_var(datasets, variable_name, time, selected_lat, selected_lon, selected_unit= None, plot_mode = False)

    
    
Parameters:
    - datasets (xarray): The loaded dataset opened using xarray.
    - variable_name (str): Name of the variable to retrieve.
    - time (str): Timestamp to filter the data.
    - selected_lat (float): Latitude value.
    - selected_lon (float): Longitude value.
    - selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
    - plot_mode (bool, optional): If True, returns additional information for plotting.

Returns:
    - If plot_mode is True: A tuple containing:
        - variable_values (xarray): Array of variable values for the specified time and latitude/longitude.
        - levs_ilevs (xarray): Array of level or ilevel values where data is not NaN.
        - variable_unit (str): Unit of the variable after conversion (if applicable).
        - variable_long_name (str): Long descriptive name of the variable.
        - selected_ut (float): Universal Time value in hours for the specified time.
        - selected_mtime (array): Array containing Day, Hour, Min of the model run.
        - filename (str): Name of the dataset file from which data is extracted.
    - If plot_mode is False: An xarray object containing the variable values.



Selected Time Latitude
~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from the dataset based on a specific variable, time, and latitude.

.. function:: arr_lev_lon(datasets, variable_name, time, selected_lat, selected_unit= None, plot_mode = False)


Parameters:
    - datasets (xarray): The loaded dataset opened using xarray.
    - variable_name (str): Name of the variable to extract.
    - time (str/numpy.datetime64): Timestamp to filter the data.
    - selected_lat (float): Latitude value to filter the data.
    - selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
    - plot_mode (bool, optional): If True, returns additional information for plotting.

Returns:
    - If plot_mode is False: An xarray object containing the variable values for the specified time and latitude.
    - If plot_mode is True: A tuple containing:
        - variable_values (xarray): Array of variable values for the specified time and latitude.
        - lons (xarray): Array of longitude values corresponding to the variable values.
        - levs_ilevs (xarray): Array of level or ilevel values where data is not NaN.
        - selected_lat (float): The latitude value used for data selection.
        - variable_unit (str): Unit of the variable after conversion (if applicable).
        - variable_long_name (str): Long descriptive name of the variable.
        - selected_ut (float): Universal Time value in hours for the specified time.
        - selected_mtime (array): Array containing Day, Hour, Min of the model run.
        - filename (str): Name of the dataset file from which data is extracted.


Selected Time, Longitude
~~~~~~~~~~~~~~~~~~~~
This function extracts data from a dataset based on the specified variable name, time, and longitude.

.. function:: arr_lev_lat(datasets, variable_name, time, selected_lon, selected_unit=None, plot_mode = False)

    
Parameters:
    - datasets (xarray): The loaded dataset opened using xarray.
    - variable_name (str): Name of the variable to extract.
    - time (str/numpy.datetime64): Timestamp to filter the data.
    - selected_lon (float/str): Longitude to filter the data, or 'mean' for averaging over all longitudes.
    - selected_unit (str, optional): Desired unit to convert the data to. If None, uses the original unit.
    - plot_mode (bool, optional): If True, returns additional information for plotting.

Returns:
    - If plot_mode is False: An xarray object containing the variable values for the specified time and longitude.
    - If plot_mode is True: A tuple containing:
        - variable_values (xarray): Array of variable values for the specified time and longitude.
        - lats (xarray): Array of latitude values corresponding to the variable values.
        - levs_ilevs (xarray): Array of level or ilevel values where data is not NaN.
        - variable_unit (str): Unit of the variable after conversion (if applicable).
        - variable_long_name (str): Long descriptive name of the variable.
        - selected_ut (float): Universal Time value in hours for the specified time.
        - selected_mtime (array): Array containing Day, Hour, Min of the model run.
        - filename (str): Name of the dataset file from which data is extracted.


Selected Latitude, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from multiple datasets using data across different levels and times for a given latitude and longitude.
.. function:: arr_lev_time(datasets, variable_name, selected_lat, selected_lon, selected_unit = None, plot_mode = False)

Parameters:
    - datasets (list of tuples): A list of tuples where each tuple contains an xarray dataset and its filename.
    - variable_name (str): The name of the variable to be extracted from the dataset.
    - selected_lat (float/str): The latitude value or 'mean' to average over all latitudes.
    - selected_lon (float/str): The longitude value or 'mean' to average over all longitudes.
    - selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
    - plot_mode (bool, optional): If True, the function returns additional data useful for plotting.

Returns:
    - If plot_mode is False, returns a numpy array of variable values concatenated across datasets.
    - If plot_mode is True, returns a tuple containing:
        - variable_values_all (numpy array): Concatenated variable values.
        - levs_ilevs (numpy array): Corresponding level or ilevel values.
        - mtime_values (list): List of model times.
        - selected_lon (float/str): The longitude used for data selection.
        - variable_unit (str): The unit of the variable after conversion (if applicable).
        - variable_long_name (str): The long descriptive name of the variable.


Selected Level, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~

This function extracts and processes data from the dataset based on the specified variable name, longitude, and level/ilev.

.. function:: arr_lat_time(datasets, variable_name, selected_lon,selected_lev_ilev = None, selected_unit = None, plot_mode = False)

        
Parameters:
    - datasets (list of tuples): Each tuple contains an xarray dataset and its filename.
    - variable_name (str): The name of the variable to extract.
    - selected_lon (float/str): Longitude value or 'mean' to average over all longitudes.
    - selected_lev_ilev (float/str/None): Level or intermediate level value or 'mean' for averaging, or None if not applicable.
    - selected_unit (str, optional): The desired unit for the variable. If None, the original unit is used.
    - plot_mode (bool, optional): If True, returns additional data useful for plotting.

Returns:
    - If plot_mode is False, returns a numpy array of variable values concatenated across datasets.
    - If plot_mode is True, returns a tuple containing:
        - variable_values_all (numpy array): Concatenated variable values.
        - lats (numpy array): Latitude values corresponding to the variable values.
        - mtime_values (list): List of model times.
        - selected_lon (float/str): The longitude used for data selection.
        - variable_unit (str): The unit of the variable after conversion (if applicable).
        - variable_long_name (str): The long descriptive name of the variable.
        - filename (str): Name of the dataset file from which data is extracted.




Data manipulation
------------------------------------

mTime to Time 
~~~~~~~~~~~~~~~~~~~~~
This function searches for a specific time in a dataset based on the provided model time (mtime) and returns the corresponding         np.datetime64 time value. It iterates through multiple datasets to find a match.

.. function:: get_time(datasets, mtime)

    
Parameters:
    - datasets (list of tuples): Each tuple contains an xarray dataset and its filename. The function will search each dataset for the time value.
    - mtime (list of int): Model time represented as a list of integers in the format [day, hour, minute].

Returns:
    - np.datetime64: The corresponding datetime value in the dataset for the given mtime. Returns None if no match is found.


Time to mTime 
~~~~~~~~~~~~~~~~~~~~~
This function finds and returns the model time (mtime) array that corresponds to a specific time in a dataset. 
The mtime is an array representing [Day, Hour, Min].
.. function:: get_mtime(ds, time)

Parameters:
    - ds (xarray): The dataset opened using xarray, containing time and mtime data.
    - time (str/numpy.datetime64): The timestamp for which the corresponding mtime is to be found.

Returns:
    - array: The mtime array containing [Day, Hour, Min] for the given timestamp. 
                Returns None if no corresponding mtime is found.


Average Z height
~~~~~~~~~~~~~~~~~~~~~~~~

This function compute the average Z value for a given set of lat, lon, and lev from a dataset.


.. function:: calc_avg_ht(datasets, time, selected_lev_ilev)
        
Parameters:
    - ds (xarray): The loaded dataset opened using xarray.
    - time (str): Timestamp to filter the data.
    - selected_lev_ilev (float): The level for which to retrieve data.

Returns:
    - float: The average ZG value for the given conditions.


Other
---------------------------


Check lev/ilev
~~~~~~~~~~~~~~~~~~~~~~~~
This function checks the dimensions of a given variable in a dataset to determine if it includes specific dimensions ('lev' or 'ilev').

.. function:: check_var_dims(ds, variable_name)
  
Parameters:
    - ds (xarray): The dataset in which the variable's dimensions are to be checked.
    - variable_name (str): The name of the variable for which dimensions are being checked.

Returns:
    - str: Returns 'lev' if the variable includes the 'lev' dimension, 'ilev' if it includes the 'ilev' dimension, 
            'Variable not found in dataset' if the variable does not exist in the dataset, and None if neither 'lev' nor 'ilev' are dimensions of the variable.


Min/Max Array
~~~~~~~~~~~~~~~~~~~~~~
This function finds the minimum and maximum values of varval from the 2D array
.. function:: min_max(variable_values)
        
Parameters:
    - variable_values (xarray): A list of variable values.

Returns:
    - min_val (float): Minimum value of the variable in the array.
    - max_val (float): Maximum value of the variable in the array.



