File Structure
==============

The gcmprocpy package is structured as follows:

.. code-block:: none

    ├── src                             # Directory for all gcmprocpy source files
    │   ├── gcmprocpy          
    │       ├── __init__.py             # Initialize functions for API
    │       ├── convert_units.py        # Contains unit conversion functions
    │       ├── data_parse.py           # Contains data extraction and parsing functions
    │       ├── plot_gen.py             # Contains plot generation functions
    │       ├── io.py                   # Contains Input Output functions for API
    │       ├── getoptions.py           # Contains argument parser for the Command Line Interface
    │       ├── main.py                 # Main python file to run
    │       └── cmd     
    │           ├── __init__.py         # Initialize functions for CLI
    │           ├── cmd_lat_lon.py      # Latitude vs Longitude plot function
    │           ├── cmd_lat_time.py     # Latitude vs Time plot function
    │           ├── cmd_lev_lat.py      # Level vs Latitude plot function
    │           ├── cmd_lev_lon.py      # Level vs Longitude plot function
    │           ├── cmd_lev_time.py     # Level vs Time plot function
    │           ├── cmd_lev_var.py      # Level vs Variable plot function
    ├── README.md                       # README   
    ├── benchmark_template.py           # Template for running benchmark routines     
    ├── p3postproc.py                   # Testing file    
    ├── requirements.txt                # List of required libraries     
    └── setup.py                        # PIP package builder
