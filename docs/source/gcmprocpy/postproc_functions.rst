Post Processing Functions
=============================================================

gcmprocpy provides a range of functions for post processing the data. Below are the key plotting routines along with their detailed parameters and usage examples.

Emissions Plots 
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_emissions

gcmprocpy provides the ablity for plotting emissions data. Below are the variable names that can be used in the plot functions for emissions calculation.

.. list-table::
   :header-rows: 1

   * - Emissions Long Name
     - Variable Name
     - Requirements
   * - 5.3 micron NO emission
     - NO53
     - Temperature, Atomic Oxygen, and NO data
   * - 15 micron CO2 emission
     - CO215
     - Temperature, Atomic Oxygen, and CO2 data
   * - OH emission for the v(8,3) band
     - OH83
     - Temperature, Atomic Oxygen, Molecular Oxygen and Molecular Nitrogen data

.. note::

   Currently only Latitude vs Longitude contour plots are supported for emissions data.

Example 1: Plotting 5.3 micron NO emission

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    variable_name = 'NO53'
    time = '2022-01-01T12:00:00'
    pressure_level = 4.0
    plot = gy.plt_lat_lon(datasets, variable_name, time=time, level=pressure_level)

Example 2: Plotting 15 micron CO2 emission

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    variable_name = 'CO215'
    time = '2022-01-01T12:00:00'
    pressure_level = 4.0
    plot = gy.plt_lat_lon(datasets, variable_name, time=time, level=pressure_level)


Example 3: Plotting OH emission for the v(8,3) band

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    variable_name = 'OH83'
    time = '2022-01-01T12:00:00'
    pressure_level = 4.0
    plot = gy.plt_lat_lon(datasets, variable_name, time=time, level=pressure_level)

Emissions Xarrays
--------------------------------------------------------------------------------------------------------------------

5.3 micron NO emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function processes the given datasets to generate an array of 5.3-micron NO emissions based on temperature, O1, and NO data.

.. autofunction:: arr_mkeno53
   :noindex:

15 micron CO2 emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function processes the given datasets to generate an array of 15-micron CO2 emissions based on temperature, O1, and CO2 data.

.. autofunction:: arr_mkeco215
   :noindex:

OH emission for the v(8,3) band
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function processes the given datasets to generate an array of OH emissions for the v(8,3) band based on temperature, O1, and OH data.

.. autofunction:: arr_mkeoh83
   :noindex:

Emissions Calculation
--------------------------------------------------------------------------------------------------------------------

5.3 micron NO emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function calcuates 5.3 micron NO emission (from John Wise). 

.. autofunction:: mkeno53
   :noindex:

15 micron CO2 emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function calcuates 15 micron CO2 emission (from John Wise).

.. autofunction:: mkeco215
   :noindex:

OH emission for the v(8,3) band
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function calcuates OH emission for the v(8,3) band.

.. autofunction:: mkeoh83
   :noindex:


