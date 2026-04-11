Data Parsing Functions
=============================================================

gcmprocpy provides a range of functions for data extraction and manipulation. Below are the key plotting routines along with their detailed parameters and usage examples.

.. note::
   For live examples with output, see the :doc:`notebooks/01_data_exploration` and :doc:`notebooks/02_data_extraction` notebooks.

.. currentmodule:: gcmprocpy.containers

Data Containers
--------------------------------------------------------------------------------------------------------------------

These dataclasses are used throughout gcmprocpy to hold dataset metadata and extracted plot data.

.. autoclass:: ModelDataset
   :noindex:
   :members:

.. autoclass:: PlotData
   :noindex:
   :members:

Model Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MODEL_DEFAULTS`` is a dictionary containing model-specific default variable names and
color scheme configurations for TIE-GCM and WACCM-X.

.. autodata:: MODEL_DEFAULTS
   :noindex:

Example:
    Access default wind variable names for a model.

    .. code-block:: python

        from gcmprocpy import MODEL_DEFAULTS

        # TIE-GCM wind variables
        print(MODEL_DEFAULTS['TIE-GCM']['wind_u'])  # 'UN'
        print(MODEL_DEFAULTS['TIE-GCM']['wind_v'])  # 'VN'

        # WACCM-X wind variables
        print(MODEL_DEFAULTS['WACCM-X']['wind_u'])  # 'U'
        print(MODEL_DEFAULTS['WACCM-X']['wind_v'])  # 'V'

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

Example:
    Extract all level data for a variable at a specific time.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        time_value = '2022-01-01T12:00:00'

        # Get raw xarray DataArray
        data = gy.arr_var(datasets, 'TN', time=time_value)
        print(data.shape)  # (nlev, nlat, nlon)

        # Get PlotData object with metadata
        result = gy.arr_var(datasets, 'TN', time=time_value, plot_mode=True)
        print(result.variable_unit, result.long_name)

        # Using model time (TIE-GCM)
        data = gy.arr_var(datasets, 'TN', mtime=[360, 0, 0, 0])


Selected Time, Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from the dataset based on the specified variable, time, and level (lev/ilev).

.. autofunction:: arr_lat_lon
   :noindex:

Example:
    Extract a latitude-longitude slice at a specific time and pressure level.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (lat x lon)
        data = gy.arr_lat_lon(datasets, 'TN', time='2022-01-01T12:00:00', selected_lev_ilev=4.0)
        print(data.shape)  # (nlat, nlon)

        # PlotData object for use with custom plotting
        result = gy.arr_lat_lon(datasets, 'TN', time='2022-01-01T12:00:00',
                                selected_lev_ilev=4.0, plot_mode=True)
        print(result.lats, result.lons, result.values.shape)

        # Using model time (TIE-GCM)
        data = gy.arr_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], selected_lev_ilev=4.0)

        # Specify level as height in km
        data = gy.arr_lat_lon(datasets, 'TN', time='2022-01-01T12:00:00',
                              selected_lev_ilev=300.0, level_type='height')

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

Example:
    Extract a vertical profile at a specific latitude, longitude, and time.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (1D array of values at each level)
        data = gy.arr_lev_var(datasets, 'TN', latitude=30.0,
                              time='2022-01-01T12:00:00', longitude=45.0)

        # PlotData object with level information
        result = gy.arr_lev_var(datasets, 'TN', latitude=30.0,
                                time='2022-01-01T12:00:00', longitude=45.0,
                                plot_mode=True)
        print(result.levs, result.values)

        # Using local time instead of longitude
        data = gy.arr_lev_var(datasets, 'TN', latitude=0.0,
                              time='2022-01-01T12:00:00', local_time=12.0)


Selected Time Latitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from the dataset based on a specific variable, time, and latitude.

.. autofunction:: arr_lev_lon
   :noindex:

Example:
    Extract a level-longitude cross section at a specific latitude and time.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (nlev x nlon)
        data = gy.arr_lev_lon(datasets, 'TN', latitude=30.0,
                              time='2022-01-01T12:00:00')
        print(data.shape)

        # PlotData object for custom contour plotting
        result = gy.arr_lev_lon(datasets, 'TN', latitude=30.0,
                                time='2022-01-01T12:00:00', plot_mode=True)
        print(result.levs, result.lons, result.values.shape)

Selected Time, Longitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts data from a dataset based on the specified variable name, time, and longitude.

.. autofunction:: arr_lev_lat
   :noindex:

Example:
    Extract a level-latitude cross section at a specific longitude and time.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (nlev x nlat)
        data = gy.arr_lev_lat(datasets, 'TN', time='2022-01-01T12:00:00',
                              selected_lon=45.0)
        print(data.shape)

        # PlotData object
        result = gy.arr_lev_lat(datasets, 'TN', time='2022-01-01T12:00:00',
                                selected_lon=45.0, plot_mode=True)
        print(result.levs, result.lats, result.values.shape)

Selected Latitude, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function extracts and processes data from multiple datasets using data across different levels and times for a given latitude and longitude.

.. autofunction:: arr_lev_time
   :noindex:

Example:
    Extract a level-time cross section at a specific latitude and longitude.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (nlev x ntime)
        data = gy.arr_lev_time(datasets, 'TN', latitude=30.0, longitude=45.0)
        print(data.shape)

        # With time range filter
        data = gy.arr_lev_time(datasets, 'TN', latitude=30.0, longitude=45.0,
                               time_minimum='2022-01-01T00:00:00',
                               time_maximum='2022-01-02T00:00:00')

        # PlotData object
        result = gy.arr_lev_time(datasets, 'TN', latitude=30.0, longitude=45.0,
                                 plot_mode=True)
        print(result.levs, result.times, result.values.shape)


Selected Level, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts and processes data from the dataset based on the specified variable name, longitude, and level/ilev.

.. autofunction:: arr_lat_time
   :noindex:

Example:
    Extract a latitude-time cross section at a specific level and longitude.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (nlat x ntime)
        data = gy.arr_lat_time(datasets, 'TN', selected_lev_ilev=4.0,
                               longitude=45.0)
        print(data.shape)

        # With time range filter
        data = gy.arr_lat_time(datasets, 'TN', selected_lev_ilev=4.0,
                               longitude=45.0,
                               time_minimum='2022-01-01T00:00:00',
                               time_maximum='2022-01-02T00:00:00')

        # PlotData object
        result = gy.arr_lat_time(datasets, 'TN', selected_lev_ilev=4.0,
                                 longitude=45.0, plot_mode=True)
        print(result.lats, result.times, result.values.shape)

        # Specify level as height in km
        data = gy.arr_lat_time(datasets, 'TN', selected_lev_ilev=300.0,
                               longitude=0.0, level_type='height')

Selected Level, Latitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts and processes data from the dataset based on the specified variable name, latitude, and level/ilev.
Returns a 2D array of longitudes x time.

.. autofunction:: arr_lon_time
   :noindex:

Example:
    Extract a longitude-time cross section at a specific level and latitude.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (nlon x ntime)
        data = gy.arr_lon_time(datasets, 'TN', latitude=0.0,
                               selected_lev_ilev=4.0)
        print(data.shape)

        # PlotData object
        result = gy.arr_lon_time(datasets, 'TN', latitude=0.0,
                                 selected_lev_ilev=4.0, plot_mode=True)
        print(result.lons, result.times, result.values.shape)

        # Specify level as height in km
        data = gy.arr_lon_time(datasets, 'TN', latitude=0.0,
                               selected_lev_ilev=250.0, level_type='height')

Selected Level, Latitude, Longitude Over Time-range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts a 1D time series of a variable at a specific latitude, longitude, and level/ilev.

.. autofunction:: arr_var_time
   :noindex:

Example:
    Extract a time series at a specific latitude, longitude, and level.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Raw xarray DataArray (1D time series)
        data = gy.arr_var_time(datasets, 'TN', latitude=0.0, longitude=45.0,
                               selected_lev_ilev=4.0)
        print(data.shape)

        # PlotData object
        result = gy.arr_var_time(datasets, 'TN', latitude=0.0, longitude=45.0,
                                 selected_lev_ilev=4.0, plot_mode=True)
        print(result.times, result.values)

        # Specify level as height in km
        data = gy.arr_var_time(datasets, 'TN', latitude=0.0, longitude=45.0,
                               selected_lev_ilev=300.0, level_type='height')

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

Example:
    Convert a model time (mtime) to a datetime value.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # TIE-GCM model time: [Day, Hour, Min, Sec]
        mtime = [360, 0, 0, 0]
        time = gy.get_time(datasets, mtime)
        print(time)  # np.datetime64 value

Time to mTime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function finds and returns the model time (mtime) array that corresponds to a specific time in a dataset.
The mtime is an array representing [Day, Hour, Min].

.. autofunction:: get_mtime
   :noindex:

Example:
    Convert a datetime string to model time (mtime).

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Get mtime for a specific datetime
        mtime = gy.get_mtime(datasets, '2022-01-01T12:00:00')
        print(mtime)  # e.g., [1, 12, 0] for Day 1, 12:00

        # Use with time_list to convert all times
        times = gy.time_list(datasets)
        for t in times[:5]:
            mt = gy.get_mtime(datasets, t)
            print(f'{t} -> mtime {mt}')

Height Interpolation
---------------------------------------------------------------------------------------------------------------------

gcmprocpy supports converting between pressure levels and geometric height (km) using the model's
height variable (``ZG`` for TIE-GCM, ``Z3`` for WACCM-X). This enables specifying levels as heights
and plotting vertical axes in km instead of pressure coordinates.

Height to Pressure Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function converts a target height in km to the nearest pressure level by looking up the model's
geometric height field (ZG or Z3).

.. autofunction:: height_to_pres_level
   :noindex:

Example:
    Find the pressure level closest to 300 km altitude.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        time = '2022-01-01T12:00:00'

        # Global average — find the level whose mean height is closest to 300 km
        pres_level = gy.height_to_pres_level(datasets, time, 300.0)
        print(f'300 km ≈ pressure level {pres_level}')

        # At a specific location
        pres_level = gy.height_to_pres_level(datasets, time, 300.0, latitude=0.0, longitude=45.0)
        print(f'300 km at equator, 45°E ≈ pressure level {pres_level}')

Interpolate to Height
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function interpolates a 2D field from pressure levels to constant height surfaces using the
model's geometric height field. Supports both linear and exponential (log) interpolation.

.. autofunction:: interpolate_to_height
   :noindex:

Example:
    Interpolate a latitude-altitude cross section from pressure to height coordinates.

    .. code-block:: python

        import numpy as np
        datasets = gy.load_datasets(directory, dataset_filter)
        time = '2022-01-01T12:00:00'

        # Extract lev vs lat data on pressure levels
        result = gy.arr_lev_lat(datasets, 'TN', time, selected_lon=0.0, plot_mode=True)

        # Interpolate to 40 evenly spaced height levels
        interp_values, heights_km = gy.interpolate_to_height(
            datasets, result.values, result.levs, time, n_heights=40)
        print(f'Height range: {heights_km[0]:.1f} – {heights_km[-1]:.1f} km')
        print(f'Interpolated shape: {interp_values.shape}')

        # Interpolate to specific heights
        target_heights = np.array([100, 200, 300, 400, 500])
        interp_values, _ = gy.interpolate_to_height(
            datasets, result.values, result.levs, time, target_heights=target_heights)

        # Use exponential interpolation for density-like variables
        ne_result = gy.arr_lev_lat(datasets, 'NE', time, selected_lon=0.0, plot_mode=True)
        interp_ne, heights = gy.interpolate_to_height(
            datasets, ne_result.values, ne_result.levs, time, log_interp=True)

Height in Plot Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All plot functions that accept a ``level`` parameter also accept ``level_type`` to specify
whether the level value is a pressure level (default) or a height in km. When ``level_type='height'``,
the height is automatically converted to the nearest pressure level using the model's geometric
height field (``ZG`` for TIE-GCM, ``Z3`` for WACCM-X).

All level-axis plots (``plt_lev_var``, ``plt_lev_lon``, ``plt_lev_lat``, ``plt_lev_time``) also
accept ``y_axis='height'`` to display the vertical axis in km instead of pressure coordinates.

Example:
    Specify a level as height instead of pressure.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Lat-lon plot at 300 km altitude (automatically finds nearest pressure level)
        plot = gy.plt_lat_lon(datasets, 'TN', time='2022-01-01T12:00:00',
                              level=300.0, level_type='height')

        # Latitude vs time at 400 km altitude
        plot = gy.plt_lat_time(datasets, 'TN', level=400.0, level_type='height',
                               longitude=0.0)

        # Longitude vs time at 250 km altitude
        plot = gy.plt_lon_time(datasets, 'TN', latitude=0.0, level=250.0,
                               level_type='height')

        # Variable vs time at 300 km altitude
        plot = gy.plt_var_time(datasets, 'TN', latitude=0.0, longitude=0.0,
                               level=300.0, level_type='height')

Example:
    Plot vertical axis in km instead of pressure.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Vertical profile with height axis
        plot = gy.plt_lev_var(datasets, 'TN', latitude=0.0,
                              time='2022-01-01T12:00:00', longitude=0.0,
                              y_axis='height')

        # Longitude cross-section with height axis
        plot = gy.plt_lev_lon(datasets, 'TN', latitude=0.0,
                              time='2022-01-01T12:00:00', y_axis='height')

        # Latitude cross-section with height axis
        plot = gy.plt_lev_lat(datasets, 'TN', time='2022-01-01T12:00:00',
                              longitude=0.0, y_axis='height')

        # Level vs time with height axis
        plot = gy.plt_lev_time(datasets, 'TN', latitude=0.0, longitude=0.0,
                               y_axis='height')
