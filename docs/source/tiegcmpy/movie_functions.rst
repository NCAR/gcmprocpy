Movie Routines
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

Latitude vs Longitude Contour Movie
~~~~~~~~~~~~~~~~~~~~~
This function generates a sequence of contour plots of a variable against latitude and longitude over time and creates a video animation.

.. function:: mov_lat_lon(datasets, variable_name, level=None, variable_unit=None, contour_intervals=None, contour_value=None, symmetric_interval=False, cmap_color=None, line_color='white', coastlines=False, nightshade=False, gm_equator=False, latitude_minimum=None, latitude_maximum=None, longitude_minimum=None, longitude_maximum=None, localtime_minimum=None, localtime_maximum=None, time_minimum=None, time_maximum=None, fps=None)

Parameters:
    - datasets (xarray.Dataset): The loaded dataset/s using xarray.
    - variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
    - level (float, optional): The selected lev/ilev value. Defaults to None.
    - variable_unit (str, optional): The desired unit of the variable. Defaults to None.
    - contour_intervals (int, optional): The number of contour intervals. Defaults to None.
    - contour_value (int, optional): The value between each contour interval. Defaults to None.
    - symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
    - cmap_color (str, optional): The color map of the contour. Defaults to None.
    - line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
    - coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
    - nightshade (bool, optional): Shows nightshade on the plot. Defaults to False.
    - gm_equator (bool, optional): Shows geomagnetic equator on the plot. Defaults to False.
    - latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to None.
    - latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to None.
    - longitude_minimum (float, optional): Minimum longitude to slice plots. Defaults to None.
    - longitude_maximum (float, optional): Maximum longitude to slice plots. Defaults to None.
    - localtime_minimum (float, optional): Minimum local time to slice plots. Defaults to None.
    - localtime_maximum (float, optional): Maximum local time to slice plots. Defaults to None.
    - time_minimum (np.datetime64 or str, optional): Minimum time for the plot. Defaults to None.
    - time_maximum (np.datetime64 or str, optional): Maximum time for the plot. Defaults to None.
    - fps (int, optional): Frames per second for the video. Defaults to None.

Returns:
    Video file of the contour plot over the specified time range.

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
