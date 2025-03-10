Movie Routines
====================================================

gcmprocpy provides a range of functions for data visualization. Below are the key plotting routines along with their detailed parameters and usage examples.

API
--------------------------------------------------------------------------------------------------------------------------------------------

.. currentmodule:: gcmprocpy.io

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the netCDF datasets for the plotting routines.

.. autofunction:: load_datasets
   :noindex:



Closing Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function closes the netCDF datasets.

.. autofunction:: close_datasets
   :noindex:

.. currentmodule:: gcmprocpy.mov_gen

Latitude vs Longitude Contour Movie
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function generates a sequence of contour plots of a variable against latitude and longitude over time and creates a video animation.

.. autofunction:: mov_lat_lon
   :noindex:


Example:
    Load datasets and generate a video animation of Latitude vs Longitude contour plots over time.

    .. code-block:: python

        from your_module import mov_lat_lon  # Replace 'your_module' with the actual module name

        datasets = load_datasets(directory, dataset_filter)  # Ensure to define your load_datasets function
        variable_name = 'TN'
        level = 4.0
        time_min = '2024-01-01T00:00:00'
        time_max = '2024-01-02T00:00:00'
        fps = 5

        mov_lat_lon(datasets, variable_name, level=level, time_minimum=time_min, time_maximum=time_max, fps=fps)
