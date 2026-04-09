Ploting Routines
====================================================

gcmprocpy provides a range of functions for data visualization. Below are the key plotting routines along with their detailed parameters and usage examples.

Mode: API
--------------------------------------------------------------------------------------------------------------------------------------------


Jupyter notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gcmprocpy can be used in two modes in Jupyter notebooks: Regular plotting and Interactive plotting.

For regular plotting in Jupyter notebooks, use the following code snippet in an executable cell:

.. code-block:: python
    
    %matplotlib inline

.. note::
    Use ``%config InlineBackend.figure_format = 'retina'`` for high quality plots.

For interactive plotting in Jupyter notebooks, use the following code snippet in an executable cell:

.. code-block:: python
    
    %matplotlib widget

.. warning::
    The interactive plotting mode doesn't work in Jupyter notebooks on NCAR JupyterHub.

.. currentmodule:: gcmprocpy.containers

Data Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These dataclasses are used throughout gcmprocpy to hold dataset metadata and extracted plot data.

.. autoclass:: ModelDataset
   :noindex:
   :members:

.. autoclass:: PlotData
   :noindex:
   :members:

.. currentmodule:: gcmprocpy.io

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the netCDF datasets for the plotting routines.

.. autofunction:: load_datasets
   :noindex:

Closing Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function closes the netCDF datasets.

.. warning::

   This function should be called after the plotting routines have been executed.

.. autofunction:: close_datasets
   :noindex:

Saving Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function saves a plot to a file.

.. autofunction:: save_output
   :noindex:

Example:
    Save a plot as a PNG file.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0)
        gy.save_output('/output/path', 'my_plot', 'png', plot)


.. currentmodule:: gcmprocpy.plot_gen


Latitude vs Longitude Contour Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable against latitude and longitude.

.. autofunction:: plt_lat_lon
   :noindex:

Example:
    Load datasets and generate a Latitude vs Longitude contour plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        value_of_mtime = [360, 0, 0, 0]
        pressure_level = 4.0
        unit_of_variable = 'K'
        intervals = 20
        plot = gy.plt_lat_lon(datasets, variable_name, mtime=value_of_mtime, level=pressure_level, variable_unit=unit_of_variable, contour_intervals=intervals)

Polar Projections:
    The ``projection`` parameter supports polar stereographic views in addition to the default Mercator projection.

    .. code-block:: python

        # North polar stereographic
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0, projection='north_polar')

        # South polar stereographic
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0, projection='south_polar')

        # Both hemispheres side by side
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0, projection='polar')

        # Polar with coastlines
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0, projection='polar', coastlines=True)

Pressure Level vs Variable Line Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates a line plot of a variable at a specific latitude and optional longitude, time, and local time.

.. autofunction:: plt_lev_var
   :noindex:

Example:
    Load datasets and generate a Pressure Level vs Variable Line plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude = 30.0
        time_value = '2022-01-01T12:00:00'
        longitude_value = 45.0
        unit_of_variable = 'K'
        plot = gy.plt_lev_var(datasets, variable_name, latitude, time=time_value, longitude=longitude_value, variable_unit=unit_of_variable)

# Extracting the details for "Pressure level vs Longitude Contour Plot" and "Pressure Level vs Latitude Contour Plot" 
# from the README.md to create corresponding sections in functionality.rst

Pressure level vs Longitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable at a specific latitude against longitude, with optional time and local time.

.. autofunction:: plt_lev_lon
   :noindex:

Example:
    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude = 30.0
        time_value = '2022-01-01T12:00:00'
        unit_of_variable = 'K'
        contour_intervals = 20
        plot = gy.plt_lev_lon(datasets, variable_name, latitude, time=time_value, variable_unit=unit_of_variable, contour_intervals=contour_intervals)

Pressure Level vs Latitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates a contour plot of a variable against pressure level and latitude.

.. autofunction:: plt_lev_lat
   :noindex:

Example:
    Load datasets and generate a Pressure Level vs Latitude contour plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        longitude_value = 45.0
        time_value = '2022-01-01T12:00:00'
        unit_of_variable = 'K'
        plot = gy.plt_lev_lat(datasets, variable_name, longitude=longitude_value, time=time_value, variable_unit=unit_of_variable)

Pressure Level vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function creates a contour plot of a variable against pressure level and time.

.. autofunction:: plt_lev_time
   :noindex:

Example:
    Load datasets and generate a Pressure Level vs Time contour plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        latitude_value = 30.0
        time_min = '2022-01-01T00:00:00'
        time_max = '2022-01-02T00:00:00'
        unit_of_variable = 'K'
        plot = gy.plt_lev_time(datasets, variable_name, latitude=latitude_value, time_minimum=time_min, time_maximum=time_max, variable_unit=unit_of_variable)

Latitude vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function creates a contour plot of a variable against latitude and time.

.. autofunction:: plt_lat_time
   :noindex:

Example:
    Load datasets and generate a Latitude vs Time contour plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        variable_name = 'TN'
        pressure_level = 4.0
        time_min = '2022-01-01T00:00:00'
        time_max = '2022-01-02T00:00:00'
        unit_of_variable = 'K'
        plot = gy.plt_lat_time(datasets, variable_name, level=pressure_level, time_minimum=time_min, time_maximum=time_max, variable_unit=unit_of_variable)

Longitude vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function creates a contour plot of a variable against longitude and time.

.. autofunction:: plt_lon_time
   :noindex:

Example:
    Load datasets and generate a Longitude vs Time contour plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        plot = gy.plt_lon_time(datasets, 'TN', latitude=0.0, level=4.0)

Variable vs Time Line Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function creates a line plot of a variable against time at a specific latitude, longitude, and optional level.

.. autofunction:: plt_var_time
   :noindex:

Example:
    Load datasets and generate a Variable vs Time line plot.

    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)
        plot = gy.plt_var_time(datasets, 'TN', latitude=0.0, longitude=45.0, level=4.0)

Satellite Track Interpolation Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function interpolates model data along a satellite trajectory and plots the result.
With a level specified, it produces a 1D line plot along the track. Without a level, it produces
a 2D contour plot of all levels vs along-track points.

.. autofunction:: plt_sat_track
   :noindex:

Example:
    Interpolate and plot model temperature along a simulated satellite pass.

    .. code-block:: python

        import numpy as np
        datasets = gy.load_datasets(directory, dataset_filter)
        times = gy.time_list(datasets)

        # Satellite trajectory: arrays of time, lat, lon (one entry per point)
        sat_time = np.array([times[0] + np.timedelta64(i * 6, 'm') for i in range(20)])
        sat_lat = np.linspace(-60, 60, 20)
        sat_lon = np.linspace(-120, 120, 20)

        # Line plot at a fixed level
        plot = gy.plt_sat_track(datasets, 'TN', sat_time, sat_lat, sat_lon, level=5.0)

        # Contour plot across all levels
        plot = gy.plt_sat_track(datasets, 'TN', sat_time, sat_lat, sat_lon)

Wind Vector Overlays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wind vectors can be overlaid on any ``plt_lat_lon`` plot by setting ``wind=True``.
The wind variable names are automatically selected based on the model type
(TIE-GCM: UN/VN, WACCM-X: U/V) using ``MODEL_DEFAULTS``.

Example:
    .. code-block:: python

        datasets = gy.load_datasets(directory, dataset_filter)

        # Mercator with wind vectors
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0, wind=True)

        # Orthographic with wind vectors and coastlines
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0,
                              projection='orthographic', wind=True, wind_density=3, coastlines=True)

        # Customize wind arrow appearance
        plot = gy.plt_lat_lon(datasets, 'TN', mtime=[360, 0, 0, 0], level=4.0,
                              wind=True, wind_density=5, wind_color='red', wind_scale=500)

.. currentmodule:: gcmprocpy.containers

Model Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. currentmodule:: gcmprocpy.plot_gen

Mode: CLI
--------------------------------------------------------------------------------------------------------------------------------------------

Latitude vs Longitude Contour Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable against latitude and longitude.

.. autoprogram:: gcmprocpy.cmd.cmd_lat_lon:cmd_parser()
   :prog: lat_lon

Pressure Level vs Variable Line Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command generates a line plot of a variable at a specific latitude and optional longitude, time, and local time.

.. autoprogram:: gcmprocpy.cmd.cmd_lev_var:cmd_parser()
   :prog: lev_var

Pressure level vs Longitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable at a specific latitude against longitude, with optional time and local time.

.. autoprogram:: gcmprocpy.cmd.cmd_lev_lon:cmd_parser()
   :prog: lev_lon

Pressure Level vs Latitude Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command generates a contour plot of a variable against pressure level and latitude.

.. autoprogram:: gcmprocpy.cmd.cmd_lev_lat:cmd_parser()
   :prog: lev_lat

Pressure Level vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command creates a contour plot of a variable against pressure level and time.

.. autoprogram:: gcmprocpy.cmd.cmd_lev_time:cmd_parser()
   :prog: lev_time

Latitude vs Time Contour Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command creates a contour plot of a variable against latitude and time.

.. autoprogram:: gcmprocpy.cmd.cmd_lat_time:cmd_parser()
   :prog: lat_time


