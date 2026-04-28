File Structure
==============

The gcmprocpy package is structured as follows:

.. code-block:: none

    ├── src                             # Directory for all gcmprocpy source files
    │   ├── gcmprocpy
    │       ├── __init__.py             # Initialize functions for API
    │       ├── containers.py           # Data containers, model defaults, derived variable registry
    │       ├── convert_units.py        # Contains unit conversion functions
    │       ├── data_parse.py           # Contains data extraction and parsing functions
    │       ├── data_emissions.py       # Simple emissions (NO53, CO215, OH83)
    │       ├── data_oh.py              # Full OH Meinel band vibrational model (39 bands)
    │       ├── data_epflux.py          # Eliassen-Palm flux (EPVY, EPVZ, EPVDIV)
    │       ├── data_diff.py            # Difference fields (raw and percent)
    │       ├── plot_gen.py             # Contains plot generation functions
    │       ├── mov_gen.py              # Contains movie generation functions
    │       ├── io.py                   # Contains Input Output functions for API
    │       ├── getoptions.py           # Contains argument parser for the Command Line Interface
    │       ├── main.py                 # Main python file to run
    │       ├── gui
    │       │   ├── __init__.py         # Initialize functions for GUI
    │       │   └── gcmprocpy.py        # PySide6 GUI implementation
    │       └── cmd
    │           ├── __init__.py         # Initialize functions for CLI
    │           ├── cmd_lat_lon.py      # Latitude vs Longitude plot function
    │           ├── cmd_lat_time.py     # Latitude vs Time plot function
    │           ├── cmd_lev_lat.py      # Level vs Latitude plot function
    │           ├── cmd_lev_lon.py      # Level vs Longitude plot function
    │           ├── cmd_lev_time.py     # Level vs Time plot function
    │           ├── cmd_lev_var.py      # Level vs Variable plot function
    │           ├── cmd_lon_time.py     # Longitude vs Time plot function
    │           ├── cmd_var_time.py     # Variable vs Time plot function
    │           └── cmd_sat_track.py    # Satellite track interpolation
    ├── README.md                       # README
    ├── benchmark_template.py           # Template for running benchmark routines
    ├── requirements.txt                # List of required libraries
    ├── pyproject.toml                  # Modern Python package configuration
    └── setup.py                        # PIP package builder
