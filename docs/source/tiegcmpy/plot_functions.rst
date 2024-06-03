Ploting Routines
=============

tiegcmpy provides a range of functions for data visualization. Below are the key plotting routines along with their detailed parameters and usage examples.

API
-----------------------------------

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~

This function loads the netCDF datasets for the plotting routines.

.. function:: load_datasets(directory_or_dataset,dataset_filter = None)

Parameters:
    - directory (str): The location of directory where the files are stored or the location of the file .
    - dataset_filter (str, optional): The string of the NetCDF files to select from eg.('prim','sech').

Returns:
    - list: The array containing datasets loaded via xarray and the corresponding filenames in string.  

Latitude vs Longitude Contour Plots
~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable against latitude and longitude.

.. function:: plt_lat_lon(datasets, variable_name, time= None, mtime=None, level = None,  variable_unit = None, contour_intervals = 20, contour_value = None, symmetric_interval= False, cmap_color = None, line_color = 'white', coastlines=False, nightshade=False, gm_equator=False, latitude_minimum = None, latitude_maximum = None, longitude_minimum = None, longitude_maximum = None, localtime_minimum = None, localtime_maximum = None)


Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
    - mtime (array, optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
    - level (float, optional): The selected lev/ilev value.
    - variable_unit (str, optional): The desired unit of the variable.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
    - contour_value (int, optional): The value between each contour interval.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
    - nightshade (bool, optional): Shows nightshade on the plot. Defaults to False.
    - gm_equator (bool, optional): Shows geomagnetic equator on the plot. Defaults to False.
    - latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to -87.5.
    - latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to 87.5.
    - longitude_minimum (float, optional): Minimum longitude to slice plots. Defaults to -180.
    - longitude_maximum (float, optional): Maximum longitude to slice plots. Defaults to 175.
    - localtime_minimum (float, optional): Minimum local time to slice plots. Defaults to None.
    - localtime_maximum (float, optional): Maximum local time to slice plots. Defaults to None.

Example:
    Load datasets and generate a Latitude vs Longitude contour plot.

    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        value_of_mtime = [360, 0, 0, 0]
        pressure_level = 4.0
        unit_of_variable = 'K'
        intervals = 20
        plot = ty.plt_lat_lon(datasets, variable_name, mtime=value_of_mtime, level=pressure_level, variable_unit=unit_of_variable, contour_intervals=intervals)

Pressure Level vs Variable Line Plot
~~~~~~~~~~~~~~~~~~~~~

This function generates a line plot of a variable at a specific latitude and optional longitude, time, and local time.

.. function:: plt_lev_var(datasets, variable_name, latitude, time= None, mtime=None, longitude = None, localtime = None, variable_unit = None, level_minimum = None, level_maximum = None)

Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - latitude (float): The specific latitude value for the plot.
    - time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
    - mtime (array, optional): The selected time as a list, e.g., [1, 12, 0] for the 1st day, 12 hours, 0 mins.
    - longitude (float, optional): The specific longitude value for the plot.
    - localtime (float, optional): The specific local time value for the plot.
    - variable_unit (str, optional): The desired unit of the variable.
    - level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
    - level_maximum (float, optional): Maximum level value for the plot. Defaults to None.

Example:
    Load datasets and generate a Pressure Level vs Variable Line plot.

    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude = 30.0
        time_value = '2022-01-01T12:00:00'
        longitude_value = 45.0
        unit_of_variable = 'K'
        plot = ty.plt_lev_var(datasets, variable_name, latitude, time=time_value, longitude=longitude_value, variable_unit=unit_of_variable)

# Extracting the details for "Pressure level vs Longitude Contour Plot" and "Pressure Level vs Latitude Contour Plot" 
# from the README.md to create corresponding sections in functionality.rst

Pressure level vs Longitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable at a specific latitude against longitude, with optional time and local time.

.. function:: plt_lev_lon(datasets, variable_name, latitude, time= None, mtime=None, variable_unit = None, contour_intervals = 20, contour_value = None, symmetric_interval= False, cmap_color = None, line_color = 'white', level_minimum = None, level_maximum = None, longitude_minimum = None, longitude_maximum = None, localtime_minimum = None, localtime_maximum = None)
    
Parameters:
    - datasets (xarray): The loaded dataset(s) using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - latitude (float): The specific latitude value for the plot.
    - time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
    - mtime (array, optional): The selected time as a list, e.g., [1, 12, 0] for the 1st day, 12 hours, 0 minutes.
    - variable_unit (str, optional): The desired unit of the variable.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
    - contour_value (int, optional): The value between each contour interval.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
    - level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
    - longitude_minimum (float, optional): Minimum longitude value for the plot. Defaults to -180.
    - longitude_maximum (float, optional): Maximum longitude value for the plot. Defaults to 175.
    - localtime_minimum (float, optional): Minimum local time value for the plot. Defaults to None.
    - localtime_maximum (float, optional): Maximum local time value for the plot. Defaults to None.

Example:
    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude = 30.0
        time_value = '2022-01-01T12:00:00'
        unit_of_variable = 'K'
        contour_intervals = 20
        plot = ty.plt_lev_lon(datasets, variable_name, latitude, time=time_value, variable_unit=unit_of_variable, contour_intervals=contour_intervals)

Pressure Level vs Latitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable against pressure level and latitude.

.. function:: plt_lev_lat(datasets, variable_name, time= None, mtime=None, longitude = None, localtime = None, variable_unit = None, contour_intervals = 20, contour_value = None, symmetric_interval= False, cmap_color = None, line_color = 'white', level_minimum = None, level_maximum = None, latitude_minimum = None, latitude_maximum = None)

Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - longitude (float, optional): The specific longitude value for the plot.
    - time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
    - mtime (array, optional): The selected time as a list, e.g., [1, 12, 0] for the 1st day, 12 hours, 0 mins.
    - localtime (float, optional): The specific local time value for the plot.
    - variable_unit (str, optional): The desired unit of the variable.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
    - contour_value (int, optional): The value between each contour interval.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
    - level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
    - level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
    - latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to None.
    - latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to None.

Example:
    Load datasets and generate a Pressure Level vs Latitude contour plot.

    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        longitude_value = 45.0
        time_value = '2022-01-01T12:00:00'
        unit_of_variable = 'K'
        plot = ty.plt_lev_lat(datasets, variable_name, longitude=longitude_value, time=time_value, variable_unit=unit_of_variable)

Pressure Level vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This function creates a contour plot of a variable against pressure level and time.

.. function:: plt_lev_time(datasets, variable_name, latitude, longitude = None, localtime = None, variable_unit = None, contour_intervals = 20, contour_value = None, symmetric_interval= False, cmap_color = None, line_color = 'white', level_minimum = None, level_maximum = None, time_minimum = None, time_maximum = None)

Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - latitude (float): The specific latitude value for the plot.
    - longitude (float, optional): The specific longitude value for the plot.
    - localtime (float, optional): The specific local time value for the plot.
    - variable_unit (str, optional): The desired unit of the variable.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
    - contour_value (int, optional): The value between each contour interval.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
    - level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
    - level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
    - time_minimum (np.datetime64, optional): Minimum time for the plot. Defaults to None.
    - time_maximum (np.datetime64, optional): Maximum time for the plot. Defaults to None.

Example:
    Load datasets and generate a Pressure Level vs Time contour plot.

    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude_value = 30.0
        time_min = '2022-01-01T00:00:00'
        time_max = '2022-01-02T00:00:00'
        unit_of_variable = 'K'
        plot = ty.plt_lev_time(datasets, variable_name, latitude=latitude_value, time_minimum=time_min, time_maximum=time_max, variable_unit=unit_of_variable)

Latitude vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This function creates a contour plot of a variable against latitude and time.

.. function:: plt_lat_time(datasets, variable_name, level = None, longitude = None, localtime = None, variable_unit = None, contour_intervals = 20, contour_value = None, symmetric_interval= False, cmap_color = None, line_color = 'white', latitude_minimum = None, latitude_maximum = None, time_minimum = None, time_maximum = None)

Parameters:
    - datasets (xarray): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - level (float, optional): The specific pressure level for the plot.
    - longitude (float, optional): The specific longitude value for the plot.
    - localtime (float, optional): The specific local time value for the plot.
    - variable_unit (str, optional): The desired unit of the variable.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
    - contour_value (int, optional): The value between each contour interval.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
    - latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to None.
    - latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to None.
    - time_minimum (np.datetime64, optional): Minimum time for the plot. Defaults to None.
    - time_maximum (np.datetime64, optional): Maximum time for the plot. Defaults to None.

Example:
    Load datasets and generate a Latitude vs Time contour plot.

    .. code-block:: python

        datasets = ty.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        pressure_level = 4.0
        time_min = '2022-01-01T00:00:00'
        time_max = '2022-01-02T00:00:00'
        unit_of_variable = 'K'
        plot = ty.plt_lat_time(datasets, variable_name, level=pressure_level, time_minimum=time_min, time_maximum=time_max, variable_unit=unit_of_variable)




CLI
-----------------------------------

Latitude vs Longitude Contour Plots
~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable against latitude and longitude.

.. code-block:: bash

       tiegcmpy -plt lat_lon -var variable_name -time time -zp level -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot


Arguments
""""""""""
These are the list of arguments for the contour plot of a variable against latitude and longitude.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
``-var VARIABLE_NAME``, ``--variable_name VARIABLE_NAME``
  Name of the variable to be plotted.

``-time TIME``, ``--time TIME``
  Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.

``-mtime [MTIME [MTIME ...]]``, ``--mtime [MTIME [MTIME ...]]``
  Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.

``-zp LEVEL``, ``--level LEVEL``
  Selected lev/ilev for the plot.

Optional
''''''''''
``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-cint CONTOUR_INTERVALS``, ``--contour_intervals CONTOUR_INTERVALS``
  Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cval CONTOUR_VALUE``, ``--contour_value CONTOUR_VALUE``
  Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-ccolor CMAP_COLOR``, ``--cmap_color CMAP_COLOR``
  Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-lcolor LINE_COLOR``, ``--line_color LINE_COLOR``
  Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cstl``, ``--coastlines``
  Add coast lines to the lat_lon plots.

``-ntsh``, ``--nightshade``
  Add nightshade to the lat_lon plots.

``-gmeq``, ``--gm_equator``
  Add geomagnetic equator to the lat_lon plots.

``-latmin LATITUDE_MINIMUM``, ``--latitude_minimum LATITUDE_MINIMUM``
  Minimum latitude to slice plots [lat_lon, lev_lat, lat_time].

``-latmax LATITUDE_MAXIMUM``, ``--latitude_maximum LATITUDE_MAXIMUM``
  Maximum latitude to slice plots [lat_lon, lev_lat, lat_time].

``-lonmin LONGITUDE_MINIMUM``, ``--longitude_minimum LONGITUDE_MINIMUM``
  Minimum longitude to slice plots [lat_lon, lev_lon].

``-lonmax LONGITUDE_MAXIMUM``, ``--longitude_maximum LONGITUDE_MAXIMUM``
  Maximum longitude to slice plots [lat_lon, lev_lon].

``-utmin LOCALTIME_MINIMUM``, ``--localtime_minimum LOCALTIME_MINIMUM``
  Minimum localtime to slice plots [lat_lon, lev_lon].

``-utmax LOCALTIME_MAXIMUM``, ``--localtime_maximum LOCALTIME_MAXIMUM``
  Maximum localtime to slice plots [lat_lon, lev_lon].

Pressure Level vs Variable Line Plot
~~~~~~~~~~~~~~~~~~~~~

This command generates a line plot of a variable at a specific latitude and optional longitude, time, and local time.

.. code-block:: bash

       tiegcmpy -plt lev_var -var variable_name -time time -lat latitude -lon longitude -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot

Arguments
""""""""""
These are the list of arguments for the line plot of a variable at a specific latitude and optional longitude, time, and local time.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
``-var VARIABLE_NAME``, ``--variable_name VARIABLE_NAME``
  Name of the variable to be plotted.

``-time TIME``, ``--time TIME``
  Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.

``-mtime [MTIME [MTIME ...]]``, ``--mtime [MTIME [MTIME ...]]``
  Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.

``-lat LATITUDE``, ``--latitude LATITUDE``
  Selected latitude for the plot.

``-lon LONGITUDE``, ``--longitude LONGITUDE``
  Selected longitude for the plot.

``-ut LOCALTIME``, ``--localtime LOCALTIME``
  Selected localtime / longitude for the plot.

Optional
''''''''''
``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-zpmin LEVEL_MINIMUM``, ``--level_minimum LEVEL_MINIMUM``
  Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-zpmax LEVEL_MAXIMUM``, ``--level_maximum LEVEL_MAXIMUM``
  Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

Pressure level vs Longitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable at a specific latitude against longitude, with optional time and local time.

.. code-block:: bash

       tiegcmpy -plt lev_lon -var variable_name -time time -lat latitude -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot
  
Arguments
""""""""""
These are the list of arguments for the contour plot of a variable at a specific latitude against longitude, with optional time and local time.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
 ``-var VARIABLE_NAME``, ``--variable_name VARIABLE_NAME``
  Name of the variable to be plotted.

``-time TIME``, ``--time TIME``
  Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.

``-mtime [MTIME [MTIME ...]]``, ``--mtime [MTIME [MTIME ...]]``
  Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.

``-lat LATITUDE``, ``--latitude LATITUDE``
  Selected latitude for the plot.

Optional
''''''''''
``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-cint CONTOUR_INTERVALS``, ``--contour_intervals CONTOUR_INTERVALS``
  Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cval CONTOUR_VALUE``, ``--contour_value CONTOUR_VALUE``
  Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-ccolor CMAP_COLOR``, ``--cmap_color CMAP_COLOR``
  Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-lcolor LINE_COLOR``, ``--line_color LINE_COLOR``
  Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-zpmin LEVEL_MINIMUM``, ``--level_minimum LEVEL_MINIMUM``
  Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-zpmax LEVEL_MAXIMUM``, ``--level_maximum LEVEL_MAXIMUM``
  Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-lonmin LONGITUDE_MINIMUM``, ``--longitude_minimum LONGITUDE_MINIMUM``
  Minimum longitude to slice plots [lat_lon, lev_lon].

``-lonmax LONGITUDE_MAXIMUM``, ``--longitude_maximum LONGITUDE_MAXIMUM``
  Maximum longitude to slice plots [lat_lon, lev_lon].

``-utmin LOCALTIME_MINIMUM``, ``--localtime_minimum LOCALTIME_MINIMUM``
  Minimum localtime to slice plots [lat_lon, lev_lon].

``-utmax LOCALTIME_MAXIMUM``, ``--localtime_maximum LOCALTIME_MAXIMUM``
  Maximum localtime to slice plots [lat_lon, lev_lon].

Pressure Level vs Latitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable against pressure level and latitude.

.. code-block:: bash

       tiegcmpy -plt lev_lat -var variable_name -time time -lon longitude -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot

Arguments
""""""""""
These are the list of arguments for the contour plot of a variable against pressure level and latitude.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
 ``-var VARIABLE_NAME``, ``--variable_name VARIABLE_NAME``
  Name of the variable to be plotted.

``-time TIME``, ``--time TIME``
  Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.

``-mtime [MTIME [MTIME ...]]``, ``--mtime [MTIME [MTIME ...]]``
  Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.

``-lon LONGITUDE``, ``--longitude LONGITUDE``
  Selected longitude for the plot.

``-ut LOCALTIME``, ``--localtime LOCALTIME``
  Selected localtime / longitude for the plot.

Optional
''''''''''
``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-cint CONTOUR_INTERVALS``, ``--contour_intervals CONTOUR_INTERVALS``
  Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cval CONTOUR_VALUE``, ``--contour_value CONTOUR_VALUE``
  Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-ccolor CMAP_COLOR``, ``--cmap_color CMAP_COLOR``
  Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-lcolor LINE_COLOR``, ``--line_color LINE_COLOR``
  Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-zpmin LEVEL_MINIMUM``, ``--level_minimum LEVEL_MINIMUM``
  Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-zpmax LEVEL_MAXIMUM``, ``--level_maximum LEVEL_MAXIMUM``
  Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-latmin LATITUDE_MINIMUM``, ``--latitude_minimum LATITUDE_MINIMUM``
  Minimum latitude to slice plots [lat_lon, lev_lat, lat_time].

``-latmax LATITUDE_MAXIMUM``, ``--latitude_maximum LATITUDE_MAXIMUM``
  Maximum latitude to slice plots [lat_lon, lev_lat, lat_time].

Pressure Level vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This command creates a contour plot of a variable against pressure level and time.

.. code-block:: bash

       tiegcmpy -plt lev_time -var variable_name -lat latitude -lon longitude -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot
  
Arguments
""""""""""
These are the list of arguments for the contour plot of a variable against pressure level and time.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
``-lat LATITUDE``, ``--latitude LATITUDE``
  Selected latitude for the plot.

``-lon LONGITUDE``, ``--longitude LONGITUDE``
  Selected longitude for the plot.

``-ut LOCALTIME``, ``--localtime LOCALTIME``
  Selected localtime / longitude for the plot.

Optional
''''''''''

``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-cint CONTOUR_INTERVALS``, ``--contour_intervals CONTOUR_INTERVALS``
  Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cval CONTOUR_VALUE``, ``--contour_value CONTOUR_VALUE``
  Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-ccolor CMAP_COLOR``, ``--cmap_color CMAP_COLOR``
  Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-lcolor LINE_COLOR``, ``--line_color LINE_COLOR``
  Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-zpmin LEVEL_MINIMUM``, ``--level_minimum LEVEL_MINIMUM``
  Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-zpmax LEVEL_MAXIMUM``, ``--level_maximum LEVEL_MAXIMUM``
  Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

**Time min max tbd**

Latitude vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~

This command creates a contour plot of a variable against latitude and time.

.. code-block:: bash

       tiegcmpy -plt lat_time -var variable_name -zp level -lon longitude -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot
  
Arguments
""""""""""
These are the list of arguments for the contour plot of a variable against latitude and time.

Dataset Input/Output
''''''''''
``-ds DATASET``, ``--dataset DATASET``
  Path to the singular dataset.

``-dir DIRECTORY``, ``--directory DIRECTORY``
  Directory path containing the datasets.

``-dsf {prim,sech}``, ``--dataset_filter {prim,sech}``
  Filter to load datasets.

``-outdir OUTPUT_DIRECTORY``, ``--output_directory OUTPUT_DIRECTORY``
  Directory to save the plots. Default: Current working directory.

``-fout {jpeg,pdf}``, ``--output_format {jpeg,pdf}``
  Format to save the plots. Default: jpeg.

``-stdout STANDARD_OUTPUT``, ``--standard_output STANDARD_OUTPUT``
  Custom file name without extension.

Plotting
''''''''''
``-zp LEVEL``, ``--level LEVEL``
  Selected lev/ilev for the plot.

``-lon LONGITUDE``, ``--longitude LONGITUDE``
  Selected longitude for the plot.

``-ut LOCALTIME``, ``--localtime LOCALTIME``
  Selected localtime / longitude for the plot.

Optional
''''''''''
``-unit VARIABLE_UNIT``, ``--variable_unit VARIABLE_UNIT``
  Selected unit of a given variable for the plot.

``-cint CONTOUR_INTERVALS``, ``--contour_intervals CONTOUR_INTERVALS``
  Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-cval CONTOUR_VALUE``, ``--contour_value CONTOUR_VALUE``
  Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-ccolor CMAP_COLOR``, ``--cmap_color CMAP_COLOR``
  Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-lcolor LINE_COLOR``, ``--line_color LINE_COLOR``
  Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time].

``-latmin LATITUDE_MINIMUM``, ``--latitude_minimum LATITUDE_MINIMUM``
  Minimum latitude to slice plots [lat_lon, lev_lat, lat_time].

``-latmax LATITUDE_MAXIMUM``, ``--latitude_maximum LATITUDE_MAXIMUM``
  Maximum latitude to slice plots [lat_lon, lev_lat, lat_time].

**Time min max tbd**


