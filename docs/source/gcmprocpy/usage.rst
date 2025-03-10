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

.. autofunction:: close_datasets
   :noindex:

.. currentmodule:: gcmprocpy.close_datasets
    
Plot Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following plots can be made with gcmprocpy:

- Latitude vs Longitude plots
- Pressure level vs Variable Value plots
- Pressure level vs Longitude plots
- Pressure level vs Latitude plots
- Pressure level vs Time plots
- Latitude vs Time plots

Examples and detailed usage can be found in the Functionality section.

Mode: CLI
-------------------------------------------------------------------------------------------------------

TIEGCMy can also be used directly from the command line. The following plots can be made on the command line:

- Latitude vs Longitude plots
- Pressure level vs Variable Value plots
- Pressure level vs Longitude plots
- Pressure level vs Latitude plots
- Pressure level vs Time plots
- Latitude vs Time plots

Examples and detailed usage can be found in the plotting routines section.
