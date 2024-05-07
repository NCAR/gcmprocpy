Usage
=====

tiegcmpy can be run in two modes: API and Command Line Interface (CLI).

Mode: API
---------

tiegcmpy can be used in custom Python scripts or Jupyter notebooks.

Importing tiegcmpy
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    import tiegcmpy as ty

Loading Datasets
~~~~~~~~~~~~~~~~~~~~~
Loading a dataset/datasets:

  .. note::

      For the inbuilt plotting routines only this method can be used to load the NetCDF datasets.


  .. code-block:: python

      ty.load_datasets(directory/file, dataset_filter)


Plot Generation
~~~~~~~~~~~~~~~~~~~~~

The following plots can be made with tiegcmpy:

- Latitude vs Longitude plots
- Pressure level vs Variable Value plots
- Pressure level vs Longitude plots
- Pressure level vs Latitude plots
- Pressure level vs Time plots
- Latitude vs Time plots

Examples and detailed usage can be found in the Functionality section.

Mode: CLI (Command Line Interface)
----------------------------------

tiegcmpy can also be used directly from the command line.

Below is the usage and description of its command-line arguments:

.. code-block:: bash

  usage: tiegcmpy [-h] [-ds DATASET] [-dir DIRECTORY] [-dsf {prim,sech}] 
                  [-outdir OUTPUT_DIRECTORY] [-fout {jpeg,pdf}] 
                  [-stdout STANDARD_OUTPUT] [-rec] [-multiout MULTIPLE_OUTPUT]
                  [-plt {lat_lon,lev_var,lev_lon,lev_lat,lev_time,lat_time}] 
                  [-var VARIABLE_NAME] [-time TIME] [-mtime [MTIME [MTIME ...]]] 
                  [-zp LEVEL] [-lat LATITUDE] [-lon LONGITUDE] [-ut LOCALTIME]
                  [-unit VARIABLE_UNIT] [-cint CONTOUR_INTERVALS] 
                  [-cval CONTOUR_VALUE] [-ccolor CMAP_COLOR] [-lcolor LINE_COLOR] 
                  [-cstl] [-ntsh] [-gmeq] [-zpmin LEVEL_MINIMUM] 
                  [-zpmax LEVEL_MAXIMUM] [-latmin LATITUDE_MINIMUM] 
                  [-latmax LATITUDE_MAXIMUM] [-lonmin LONGITUDE_MINIMUM] 
                  [-lonmax LONGITUDE_MAXIMUM] [-utmin LOCALTIME_MINIMUM] 
                  [-utmax LOCALTIME_MAXIMUM]

Arguments
~~~~~~~~~~~~~~~~~~~~~
``-h``, ``--help``
  Show this help message and exit.

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

``-rec``, ``--recursive``
  Enable interactive mode until the user inputs "exit".

``-multiout MULTIPLE_OUTPUT``, ``--multiple_output MULTIPLE_OUTPUT``
  Custom file name without extension and enables multiple output in a single PDF.

``-plt {lat_lon,lev_var,lev_lon,lev_lat,lev_time,lat_time}``, ``--plot_type {lat_lon,lev_var,lev_lon,lev_lat,lev_time,lat_time}``
  Type of the plot to be generated.

``-var VARIABLE_NAME``, ``--variable_name VARIABLE_NAME``
  Name of the variable to be plotted.

``-time TIME``, ``--time TIME``
  Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.

``-mtime [MTIME [MTIME ...]]``, ``--mtime [MTIME [MTIME ...]]``
  Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.

``-zp LEVEL``, ``--level LEVEL``
  Selected lev/ilev for the plot.

``-lat LATITUDE``, ``--latitude LATITUDE``
  Selected latitude for the plot.

``-lon LONGITUDE``, ``--longitude LONGITUDE``
  Selected longitude for the plot.

``-ut LOCALTIME``, ``--localtime LOCALTIME``
  Selected localtime / longitude for the plot.

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

``-zpmin LEVEL_MINIMUM``, ``--level_minimum LEVEL_MINIMUM``
  Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

``-zpmax LEVEL_MAXIMUM``, ``--level_maximum LEVEL_MAXIMUM``
  Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time].

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


Useage Types
~~~~~~~~~~~~~~~~~~~~~

**Single Plot**

Example:

.. code-block:: bash

    tiegcmpy --plot_type plot_type -dir directory/of/datasets --dataset_filter prim_or_sech --output_format format_of_output_plot --[Other_optional_arguments_for_specific_plots]

**Multiple Plots**

Multiple plots can be generated from different datasets using tiegcmpy's CLI. Here are some examples:

1. **Interactive Mode for Multiple Plots**

   In this mode, tiegcmpy prompts the user to select datasets and plot types interactively.

   .. code-block:: bash

       tiegcmpy --recursive

   Follow the on-screen prompts to select datasets and specify plot types.

2. **Multiple Plot Generation to Multiple Files**

   This mode allows for the generation of different plots from a single dataset, each saved to a separate file.

   This command loads the datasets.

   .. code-block:: bash

       tiegcmpy -dir /path/to/datasets --dataset_filter prim_or_sech

   
   Wait for the command input request.

   .. code-block:: bash
       Entering Interactive Mode
       Loading datasets globally.
       Enter command or 'exit' to terminate:
    
   This command generates latitude vs longitude plots in png for each dataset in the specified directory.

    .. code-block:: bash
       --plot_type lat_lon --output_format png --[Other_optional_arguments_for_specific_plots]


         

3. **Multiple Plot Generation to a Single PDF File**

   Generate multiple plots from a dataset and compile them into a single PDF file.

   This command loads the datasets and specifies the output pdf file and output format pdf.

   .. code-block:: bash

       tiegcmpy -dir /path/to/datasets--output_format pdf --output_file combined_plots.pdf

   
   Wait for the command input request.

   .. code-block:: bash
       Entering Interactive Mode
       Loading datasets globally.
       Enter command or 'exit' to terminate:
    
   This command generates latitude vs longitude plots into the pdf mentioned above for each dataset in the specified directory.

    .. code-block:: bash
       --plot_type lat_lon --[Other_optional_arguments_for_specific_plots]

   This command generates multiple plots and compiles them into 'combined_plots.pdf'.