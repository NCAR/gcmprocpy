Post Processing Functions
=============================================================

gcmprocpy provides a range of functions for post processing the data. All derived variables
are registered in a central registry and dispatched automatically when passed to any plot
function — no special handling is required.

.. note::
   For live examples with output, see the :doc:`notebooks/04_emissions` notebook.

Derived Variables
--------------------------------------------------------------------------------------------------------------------

gcmprocpy supports derived variables — quantities computed from multiple dataset fields
rather than read directly. These are used like any other variable name in plot and data
extraction functions.

.. list-table::
   :header-rows: 1

   * - Category
     - Variable Name
     - Description
     - Requirements
   * - Emissions
     - ``NO53``
     - 5.3-micron NO emission
     - Temperature, Atomic Oxygen, NO
   * - Emissions
     - ``CO215``
     - 15-micron CO2 emission
     - Temperature, Atomic Oxygen, CO2
   * - Emissions
     - ``OH83``
     - OH v(8,3) band emission (simple)
     - Temperature, Atomic Oxygen, O2, N2
   * - OH Meinel
     - ``OH_<upper>_<lower>``
     - Specific OH Meinel band (e.g. ``OH_8_3``)
     - Temperature, O, O2, N2, H, O3, HO2
   * - OH Meinel
     - ``OH_TOTAL``
     - Sum of all 39 Meinel band emissions
     - Temperature, O, O2, N2, H, O3, HO2
   * - OH Meinel
     - ``OH_VIB_<v>``
     - Vibrational level population (v=0-9)
     - Temperature, O, O2, N2, H, O3, HO2
   * - EP Flux
     - ``EPVY``
     - Meridional EP flux component
     - Temperature, U, V winds
   * - EP Flux
     - ``EPVZ``
     - Vertical EP flux component
     - Temperature, U, V, W winds
   * - EP Flux
     - ``EPVDIV``
     - EP flux divergence (wave forcing)
     - Temperature, U, V, W winds

Example: Using derived variables in plot functions

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    time = '2022-01-01T12:00:00'

    # Emissions — use with plt_lat_lon
    plot = gy.plt_lat_lon(datasets, 'NO53', time=time, level=4.0)
    plot = gy.plt_lat_lon(datasets, 'CO215', time=time, level=4.0)
    plot = gy.plt_lat_lon(datasets, 'OH_8_3', time=time, level=4.0)

    # EP flux — use with plt_lev_lat
    plot = gy.plt_lev_lat(datasets, 'EPVY', time=time)
    plot = gy.plt_lev_lat(datasets, 'EPVDIV', time=time)

    # Species density — use arr_density for explicit unit control
    cm3 = gy.arr_density(datasets, 'O1', time=time, to_unit='CM3')


Emissions
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_emissions

Simple Emissions Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gcmprocpy provides functions for computing airglow and infrared emissions from model output.

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

Emissions Array Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5.3 micron NO emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function processes the given datasets to generate an array of 5.3-micron NO emissions based on temperature, O1, and NO data.

.. autofunction:: arr_mkeno53
   :noindex:

15 micron CO2 emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function processes the given datasets to generate an array of 15-micron CO2 emissions based on temperature, O1, and CO2 data.

.. autofunction:: arr_mkeco215
   :noindex:

OH emission for the v(8,3) band
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function processes the given datasets to generate an array of OH emissions for the v(8,3) band based on temperature, O1, and OH data.

.. autofunction:: arr_mkeoh83
   :noindex:

Raw Emissions Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the underlying physics functions that compute emission rates from raw arrays.
They can be called directly for custom processing pipelines.

5.3 micron NO emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculates 5.3 micron NO emission (from John Wise).

.. autofunction:: mkeno53
   :noindex:

15 micron CO2 emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculates 15 micron CO2 emission (from John Wise).

.. autofunction:: mkeco215
   :noindex:

OH emission for the v(8,3) band
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculates OH emission for the v(8,3) band.

.. autofunction:: mkeoh83
   :noindex:


OH Meinel Band Model
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_oh

The full OH Meinel band vibrational emission model solves coupled steady-state rate equations
for 10 vibrational levels (v=0 through v=9) and computes emission rates for all 39 Meinel bands.
Ported from tgcmproc ``ohrad.F`` (B. Foster, U. B. Makhlouf, SDL/Stewart Radiance Lab).

Variable names use the pattern ``OH_<upper>_<lower>`` for specific bands (e.g. ``OH_8_3``),
``OH_VIB_<v>`` for vibrational populations, and ``OH_TOTAL`` for the total emission rate.

.. note::

   The full OH model requires seven species in the dataset: temperature, O, O2, N2, H, O3, and HO2.

Example 1: Plot OH(8,3) band emission

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    plot = gy.plt_lat_lon(datasets, 'OH_8_3', time='2022-01-01T12:00:00', level=4.0)

Example 2: Plot total OH emission

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    plot = gy.plt_lat_lon(datasets, 'OH_TOTAL', time='2022-01-01T12:00:00', level=4.0)

Example 3: Extract OH data for custom analysis

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)

    # Get PlotData for a specific band
    result = gy.arr_mkoh_band(datasets, 'OH_8_3', time='2022-01-01T12:00:00',
                              selected_lev_ilev=4.0, plot_mode=True)
    print(result.values.shape, result.variable_unit)

    # Get raw array
    values = gy.arr_mkoh_band(datasets, 'OH_8_3', time='2022-01-01T12:00:00',
                              selected_lev_ilev=4.0)

OH Array Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arr_mkoh_band
   :noindex:

OH Physics Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The underlying steady-state solver for vibrational populations and band emission rates.

.. autofunction:: ohrad
   :noindex:


Eliassen-Palm Flux
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_epflux

Eliassen-Palm (EP) flux diagnostics quantify wave-mean flow interaction in the atmosphere.
gcmprocpy computes three components from 3-D wind and temperature fields, ported from
tgcmproc ``epflux.F`` (B. Foster and Hanli Liu):

- **EPVY** — meridional EP flux component (m\ :sup:`2` s\ :sup:`-2`)
- **EPVZ** — vertical EP flux component (m\ :sup:`2` s\ :sup:`-2`)
- **EPVDIV** — EP flux divergence / wave forcing (m s\ :sup:`-1` day\ :sup:`-1`)

EP flux variables produce level-latitude cross sections and are plotted with ``plt_lev_lat``.
EPVY requires only horizontal winds; EPVZ and EPVDIV additionally require vertical wind (W/OMEGA).

Example 1: Plot EP flux divergence

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    plot = gy.plt_lev_lat(datasets, 'EPVDIV', time='2022-01-01T12:00:00')

Example 2: Plot meridional EP flux

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)
    plot = gy.plt_lev_lat(datasets, 'EPVY', time='2022-01-01T12:00:00')

Example 3: Extract EP flux data for custom analysis

.. code-block:: python

    datasets = gy.load_datasets(directory, dataset_filter)

    result = gy.arr_epflux(datasets, 'EPVDIV', time='2022-01-01T12:00:00')
    print(result.values.shape)   # (nlev, nlat)
    print(result.variable_unit)  # 'm s⁻¹ day⁻¹'

EP Flux Array Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arr_epflux
   :noindex:

EP Flux Physics Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core computation that takes 3-D wind/temperature arrays and returns zonal-mean EP flux
components. Can be called directly with numpy arrays for custom workflows.

.. autofunction:: epflux
   :noindex:


Species-Aware Density Conversions
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_density

gcmprocpy converts atmospheric species fields among the four density representations used by
tgcmproc, ported from ``denconv.F`` (B. Foster):

- **MMR** — mass mixing ratio (dimensionless, mass fraction)
- **CM3** — number density (molecules cm\ :sup:`-3`)
- **CM3-MR** — volume mixing ratio / mole fraction (dimensionless)
- **GM/CM3** — mass density (g cm\ :sup:`-3`)

Cross-unit conversions require the mean air molar mass (``BARM``) and the air number density
(``pkt``), both derived from the model's temperature and O / O\ :sub:`2` fields.  Formulas::

    BARM = 1 / (O₂/32 + O/16 + (1 − O₂ − O)/28)
    pkt  = p / (k_B · T)                           [CGS]

    MMR → CM3     :   f · pkt · BARM / W
    MMR → CM3-MR  :   f · BARM / W
    MMR → GM/CM3  :   f · pkt · BARM · 1.66e-24

.. note::
   Currently supports TIE-GCM only (log-pressure coordinate).  WACCM-X support for
   the hybrid-sigma pressure coordinate is pending; :func:`arr_density` raises
   ``NotImplementedError`` for WACCM-X datasets.

Example: Convert atomic oxygen from MMR to number density

.. code-block:: python

    import gcmprocpy as gy

    datasets = gy.load_datasets(directory, dataset_filter)

    # Reads O1 from the dataset, uses the 'units' attr for the source unit
    # (falls back to passing from_unit=... explicitly)
    result = gy.arr_density(datasets, 'O1', time='2022-01-01T12:00:00',
                             to_unit='CM3')
    print(result.values.shape)      # (nlev, nlat, nlon)
    print(result.variable_unit)     # 'CM3'

Example: Direct conversion with explicit barm / pkt

.. code-block:: python

    from gcmprocpy import (
        convert_density_units, compute_barm, compute_pkt,
        get_species_molar_mass,
    )

    # Using raw numpy arrays — barm/pkt can be scalar or full 3-D fields
    barm = compute_barm(o_mmr=o1_field, o2_mmr=o2_field)
    pkt  = compute_pkt(levs, temperature, model='TIE-GCM')
    w    = get_species_molar_mass('TIE-GCM', 'O2')

    o2_cm3 = convert_density_units(o2_mmr, 'MMR', 'CM3',
                                    barm=barm, pkt=pkt, molar_mass=w)

Density Conversion Array Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arr_density
   :noindex:

Core Conversion and Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pure-math functions that :func:`arr_density` composes.  Use these directly for custom
workflows (e.g. converting in-memory arrays from external sources).

.. autofunction:: convert_density_units
   :noindex:

.. autofunction:: compute_barm
   :noindex:

.. autofunction:: compute_pkt
   :noindex:

.. autofunction:: get_species_molar_mass
   :noindex:

Supported unit aliases include ``'cm-3'`` → ``CM3``, ``'kg/kg'`` → ``MMR``, ``'mol/mol'`` →
``CM3-MR``, and ``'g/cm3'`` → ``GM/CM3``.  See :data:`SUPPORTED_DENSITY_UNITS` for the
canonical tuple.


Difference Fields
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.data_diff

gcmprocpy supports computing difference fields between two datasets (e.g. perturbed vs control runs).
Both raw differences and percent differences are supported.

Example: Compute and plot a difference field

.. code-block:: python

    datasets_pert = gy.load_datasets(pert_dir, dataset_filter)
    datasets_ctrl = gy.load_datasets(ctrl_dir, dataset_filter)
    time = '2022-01-01T12:00:00'

    pert_result = gy.arr_lat_lon(datasets_pert, 'TN', time=time,
                                 selected_lev_ilev=4.0, plot_mode=True)
    ctrl_result = gy.arr_lat_lon(datasets_ctrl, 'TN', time=time,
                                 selected_lev_ilev=4.0, plot_mode=True)

    # Raw difference (perturbation - control)
    diff_result = gy.diff_plotdata(pert_result, ctrl_result, diff_type='RAW')

    # Percent difference
    diff_result = gy.diff_plotdata(pert_result, ctrl_result, diff_type='PERCENT')

.. autofunction:: compute_diff
   :noindex:

.. autofunction:: diff_plotdata
   :noindex:


Derived Variable Registry
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.containers

All derived variables (emissions, OH bands, EP flux) are managed through a central registry.
When a derived variable name is passed to any plot function, the registry automatically dispatches
to the correct handler. You can also register custom derived variables.

.. autofunction:: get_species_names
   :noindex:

.. autofunction:: register_derived
   :noindex:

.. autofunction:: resolve_derived
   :noindex:

Example: Check if a variable is derived

.. code-block:: python

    from gcmprocpy import resolve_derived

    handler, is_derived = resolve_derived('EPVY')
    print(is_derived)   # True

    handler, is_derived = resolve_derived('TN')
    print(is_derived)   # False

Example: Register a custom derived variable

.. code-block:: python

    from gcmprocpy import register_derived

    def my_custom_var(datasets, variable_name, time, **kwargs):
        # Custom computation ...
        return result

    register_derived('MY_VAR', my_custom_var)
