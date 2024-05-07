File Structure
==============

The tiegcmpy package is structured as follows:

.. code-block:: none

    ├── src                         # Directory for all tiegcmpy source files
    │   ├── tiegcmpy          
    │       ├── __init__.py         # Initialize functions for API
    │       ├── convert_units.py    # Contains unit conversion functions
    │       ├── data_parse.py       # Contains data extraction and parsing functions
    │       ├── plot_gen.py         # Contains plot generation functions
    │       ├── io.py               # Contains Input Output functions for API
    │       ├── getoptions.py       # Contains argument parser for the Command Line Interface
    │       └── main.py             # Main python file to run
    ├── README.md                   # README   
    ├── benchmark_template.py       # Template for running benchmark routines     
    ├── p3postproc.py               # Testing file    
    ├── requirements.txt            # List of required libraries     
    └── setup.py                    # PIP package builder