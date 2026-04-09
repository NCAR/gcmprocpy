Data Parsing Functions
=============================================================

gcmprocpy provides a range of functions for data extraction and manipulation. Below are the key plotting routines along with their detailed parameters and usage examples.

Data Exploration
--------------------------------------------------------------------------------------------------------------------

.. currentmodule:: gcmprocpy.data_parse
Listing Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and returns the unique dimensions present.

.. autofunction:: dim_list
   :noindex:

Example:
      Load datasets and list unique dimensions.

      .. code-block:: python

         datasets = gy.load_datasets(directory, dataset_filter)
         dims = gy.dim_list(datasets)
         print(dims)

Listing Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and reutrns the variables listed in there.

.. autofunction:: var_list
   :noindex:

Example:
      Load datasets and list unique variables.

      .. code-block:: python

         datasets = gy.load_datasets(directory, dataset_filter)
         vars = gy.var_list(datasets)
         print(vars)

Listing Timestamps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function compiles and returns a list of all timestamps present in the provided datasets. 

.. autofunction:: time_list
   :noindex:

Example:
      Load datasets and list unique timestamps.

      .. code-block:: python

         datasets = gy.load_datasets(directory, dataset_filter)
         times = gy.time_list(datasets)
         print(times)

Listing Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and returns the unique lev and ilev entries in sorted order.

.. autofunction:: level_list
   :noindex:

Example:
    Load datasets and list unique lev and ilev entries.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        lev_ilevs = gy.level_list(datasets)
        print(lev_ilevs)

Listing Longitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and returns the unique longitude (lon) entries in sorted order.

.. autofunction:: lon_list
   :noindex:

Example:
    Load datasets and list unique longitude entries.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        lons = gy.lon_list(datasets)
        print(lons)


Listing Latitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function reads all the datasets and returns the unique latitude (lat) entries in sorted order.

.. autofunction:: lat_list
   :noindex:

Example:
    Load datasets and list unique latitude entries.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        lats = gy.lat_list(datasets)
        print(lats)

Variable Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function provides detailed information about a specific variable in the datasets.

.. autofunction:: var_info
   :noindex:

Example:
      Load datasets and get information about a specific variable.

      .. code-block:: python

         datasets = gy.load_datasets(directory, dataset_filter)
         info = gy.var_info(datasets, 'variable_name')
         print(info)


Dimension Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function provides detailed information about a specific dimension in the datasets.

.. autofunction:: dim_info
   :noindex:

Example:
      Load datasets and get information about a specific dimension.

      .. code-block:: python

         datasets = gy.load_datasets(directory, dataset_filter)
         info = gy.dim_info(datasets, 'dimension_name')
         print(info)

Data Xarrays
--------------------------------------------------------------------------------------------------------------------

Selected Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data for a given variable at a specific time from multiple datasets. 
It also handles unit conversion and provides additional information if needed for plotting.

.. autofunction:: arr_var
   :noindex:


Selected Time, Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from the dataset based on the specified variable, time, and level (lev/ilev).

.. autofunction:: arr_lat_lon
   :noindex:

Batch Selected Time, Level (Multiple Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts multiple variables at once for a given time and level, reducing redundant dataset traversal.

.. autofunction:: batch_arr_lat_lon
   :noindex:

Example:
    Load datasets and extract multiple variables in a single pass.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        results = gy.batch_arr_lat_lon(datasets, ['TN', 'O1', 'NO'], time=time_value, selected_lev_ilev=4.0, plot_mode=True)
        for name, result in results.items():
            print(f'{name}: {result.values.shape}')

Selected Time, Latitude, Longitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from the dataset for a given variable name, latitude, longitude, and time.

.. autofunction:: arr_lev_var
   :noindex:


Selected Time Latitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from the dataset based on a specific variable, time, and latitude.

.. autofunction:: arr_lev_lon
   :noindex:

Selected Time, Longitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from a dataset based on the specified variable name, time, and longitude.

.. autofunction:: arr_lev_lat
   :noindex:

Selected Latitude, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from multiple datasets using data across different levels and times for a given latitude and longitude.

.. autofunction:: arr_lev_time
   :noindex:


Selected Level, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts and processes data from the dataset based on the specified variable name, longitude, and level/ilev.

.. autofunction:: arr_lat_time
   :noindex:

Selected Level, Latitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts and processes data from the dataset based on the specified variable name, latitude, and level/ilev.
Returns a 2D array of longitudes x time.

.. autofunction:: arr_lon_time
   :noindex:

Selected Level, Latitude, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts a 1D time series of a variable at a specific latitude, longitude, and level/ilev.

.. autofunction:: arr_var_time
   :noindex:

Satellite Track Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function interpolates model data along a satellite trajectory using trilinear interpolation
(time, latitude, longitude). Input is three arrays of equal length representing the satellite's
position at each point along its orbit.

.. autofunction:: arr_sat_track
   :noindex:

Example:
    Interpolate temperature along a satellite track.

    .. code-block:: python

        import numpy as np
        datasets = gy.load_datasets(directory, dataset_filter)
        times = gy.time_list(datasets)

        sat_time = np.array([times[0] + np.timedelta64(i * 6, 'm') for i in range(20)])
        sat_lat = np.linspace(-60, 60, 20)
        sat_lon = np.linspace(-120, 120, 20)

        # 1D array at a specific level
        values = gy.arr_sat_track(datasets, 'TN', sat_time, sat_lat, sat_lon, selected_lev_ilev=5.0)

        # 2D array (levels x track points)
        values = gy.arr_sat_track(datasets, 'TN', sat_time, sat_lat, sat_lon)

        # PlotData object
        result = gy.arr_sat_track(datasets, 'TN', sat_time, sat_lat, sat_lon, selected_lev_ilev=5.0, plot_mode=True)

Data Utilities
---------------------------------------------------------------------------------------------------------------------

mTime to Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function searches for a specific time in a dataset based on the provided model time (mtime) and returns the corresponding np.datetime64 time value. It iterates through multiple datasets to find a match.

.. autofunction:: get_time
   :noindex:

Time to mTime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function finds and returns the model time (mtime) array that corresponds to a specific time in a dataset.
The mtime is an array representing [Day, Hour, Min].

.. autofunction:: get_mtime
   :noindex:
