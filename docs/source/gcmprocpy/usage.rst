Usage
=====================================================================================

gcmprocpy can be run in two modes: API and Command Line Interface (CLI).

Mode: GUI
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

gcmprocpy can be run in GUI mode by running the following command:

.. code-block:: bash

    gcmprocpy

This will open the GUI window where the user can select the dataset and the plot type.

.. warning:: 
    
    The GUI mode requires an interactive ssh session. If you are using a remote server, you can use the following command to open the GUI window: ``ssh -X user@server``.

Mode: API
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

gcmprocpy can be used in custom Python scripts or Jupyter notebooks.

Importing gcmprocpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    import gcmprocpy as gy

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loading a dataset/datasets:

  .. note::

      For the inbuilt plotting routines only this method can be used to load the NetCDF datasets.


  .. code-block:: python

      gy.load_datasets(directory/file, dataset_filter)

Closing Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function closes the netCDF datasets.

    .. code-block:: python

      gy.close_datasets(datasets)
    
Plot Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following plots can be made with gcmprocpy:

- Latitude vs Longitude plots
- Pressure level / Height vs Variable Value plots
- Pressure level / Height vs Longitude plots
- Pressure level / Height vs Latitude plots
- Pressure level / Height vs Time plots
- Latitude vs Time plots
- Longitude vs Time plots
- Variable vs Time plots
- Satellite Track Interpolation plots

All level-axis plots support ``y_axis='height'`` to display the vertical axis in km.
All level-selection plots support ``level_type='height'`` to specify the level as a height in km
instead of a pressure level. Height conversion uses the model's geometric height field
(``ZG`` for TIE-GCM, ``Z3`` for WACCM-X).

Examples and detailed usage can be found in the Functionality section.

Mode: CLI
-------------------------------------------------------------------------------------------------------

GCMprocpy can also be used directly from the command line. The following plots can be made on the command line:

- Latitude vs Longitude plots
- Pressure level / Height vs Variable Value plots
- Pressure level / Height vs Longitude plots
- Pressure level / Height vs Latitude plots
- Pressure level / Height vs Time plots
- Latitude vs Time plots
- Longitude vs Time plots
- Variable vs Time plots
- Satellite Track Interpolation plots

Use ``-lt height`` to specify the level as height (km) or ``-ya height`` for height y-axis.

Examples and detailed usage can be found in the plotting routines section.
