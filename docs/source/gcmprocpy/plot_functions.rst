Ploting Routines
====================================================

gcmprocpy provides a range of functions for data visualization. Below are the key plotting routines along with their detailed parameters and usage examples.

Mode: API
--------------------------------------------------------------------------------------------------------------------------------------------

.. currentmodule:: gcmprocpy.io

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the netCDF datasets for the plotting routines.

.. autofunction:: load_datasets
   :noindex:

.. currentmodule:: gcmprocpy.plot_gen

Closing Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function closes the netCDF datasets.

.. autofunction:: close_datasets
   :noindex:

.. currentmodule:: gcmprocpy.close_datasets

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


