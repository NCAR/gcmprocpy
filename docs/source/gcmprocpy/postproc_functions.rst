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

.. note::
   The emission rate equations operate on **number densities (cm⁻³)**.  Species
   inputs are converted automatically via the density machinery, reading each
   field's ``units`` attribute — so emissions are correct whether a history
   stores species as mass mixing ratio (``kg/kg``), volume mixing ratio
   (``mol/mol``), or number density (``cm-3``).  ``N2`` is derived as
   ``1 − O2 − O`` when absent.  (TIE-GCM thermosphere histories typically lack
   the mesospheric chemistry — H, O3, HO2 — the full OH model needs.)

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
Ported from tgcmproc ``ohrad.F`` (subroutine ``ohrad``; B. Foster, U. B. Makhlouf, SDL/Stewart Radiance Lab).

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
tgcmproc ``epflux.F`` (subroutines ``epfluxy``/``epfluxz``/``epfluxdiv``; B. Foster and Hanli Liu, 1998):

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
tgcmproc, ported from ``denconv.F`` (subroutines ``mkdenparms``/``denconv``; B. Foster):

- **MMR** — mass mixing ratio (dimensionless, mass fraction)
- **CM3** — number density (molecules cm\ :sup:`-3`)
- **CM3-MR** — volume mixing ratio / mole fraction (dimensionless)
- **GM/CM3** — mass density (g cm\ :sup:`-3`)

Cross-unit conversions require the mean air molar mass (``BARM``) and the air number density
(``pkt``), both derived from the model's temperature and O / O\ :sub:`2` fields.  Formulas::

    BARM = 1 / (O₂/32 + O/16 + (1 − O₂ − O)/28)
    pkt  = pressure / (k_B · T)                    [CGS]

    MMR → CM3     :   f · pkt · BARM / W
    MMR → CM3-MR  :   f · BARM / W
    MMR → GM/CM3  :   f · pkt · BARM · 1.66e-24

The source unit is read from each field's ``units`` attribute (``kg/kg`` → MMR,
``mol/mol`` → CM3-MR, ``cm-3`` → CM3, …), so conversion works whether a history stores
species as mass mixing ratio, volume mixing ratio, or number density.

.. note::
   Both **TIE-GCM and WACCM-X** are supported.  The only model-specific step is how pressure
   is recovered from the vertical coordinate: TIE-GCM uses log-pressure ``p = p₀·exp(−ζ)``;
   WACCM-X uses the CAM hybrid sigma-pressure ``p = hyam·P0 + hybm·PS`` (so the file must carry
   ``hyam``/``hybm``/``PS``).

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


Derivable Fields (auto-derive if missing)
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.containers

In addition to the derived *output* registry above, gcmprocpy has a lower-level
**derivable-field** layer for intermediate quantities the model often does not
write to the history file. When a plot or data-extraction function asks for a
field that is absent from the dataset, it is computed on the full native grid
from the fields that *are* present and injected transparently — so it can be
sliced and plotted like any stored variable. A real field in the file always
takes priority over the derived form.

Currently registered derivables:

.. list-table::
   :header-rows: 1

   * - Name
     - Definition
     - Inputs
   * - ``N2``
     - molecular nitrogen mixing-ratio residual ``max(1e-5, 1 - O2 - O)``
     - O2, O
   * - ``O/N2`` (alias ``O_N2``)
     - atomic oxygen / molecular nitrogen ratio
     - O, N2
   * - ``N2/O`` (alias ``N2_O``)
     - molecular nitrogen / atomic oxygen ratio
     - N2, O
   * - ``O/O2`` (alias ``O_O2``)
     - atomic oxygen / molecular oxygen ratio
     - O, O2
   * - ``O/O2+N2`` (alias ``O_O2pN2``)
     - atomic oxygen / (O2 + N2) ratio
     - O, O2, N2
   * - ``RHO``
     - total air mass density ``O2 + O + N2`` (g cm⁻³)
     - TN, O, O2, N2
   * - ``PMB``
     - pressure in millibars (from the model vertical coordinate)
     - TN (+ ``PS`` for WACCM-X)
   * - ``TNFP``
     - frost-point temperature (K) — requires ``H2O``
     - TN, O, O2, H2O
   * - ``OX``
     - odd oxygen ``O + O3`` (additive group, members' unit)
     - O, O3
   * - ``NOZ``
     - odd nitrogen ``NO + NO2``
     - NO, NO2
   * - ``HOX``
     - odd hydrogen ``OH + HO2 + H``
     - OH, HO2, H
   * - ``O/CO2`` (alias ``O_CO2``)
     - atomic oxygen / CO2 ratio
     - O, CO2

This is what lets the full OH model and the ``OH83`` emission run on histories
that omit ``N2`` (it is derived from ``O2``/``O`` automatically). Derivables can
also depend on other derivables (e.g. ``O/N2`` builds on the derived ``N2``).
The ``RHO``/``PMB``/``TNFP`` fields reuse the density machinery, so they are
unit-aware (MMR / VMR / cm⁻³) and work for both TIE-GCM and WACCM-X; ``TNFP``
raises a chain-aware error on histories without ``H2O``.  The composition groups
``OX``/``NOZ``/``HOX`` are summed from their members in the file's shared unit
(exact — densities are additive; tgcmproc writes the group field natively and
gcmprocpy reconstructs it when absent), require the members to share a unit, and
raise a chain-aware error when a member is absent (e.g. ``OH``/``NO2`` are not on
standard TIE-GCM histories).

Register a custom derivable:

.. code-block:: python

    from gcmprocpy import register_derivable

    # formula receives a dict of input DataArrays keyed by canonical role
    register_derivable('O/O', lambda inp, mds: inp['o'] / inp['o2'],
                        inputs=['o', 'o2'], units='ratio',
                        long_name='O / O2 ratio')

.. autofunction:: register_derivable
   :noindex:

.. autofunction:: resolve_derivable
   :noindex:


Persisting Derived Quantities to NetCDF
--------------------------------------------------------------------------------------------------------------------
.. currentmodule:: gcmprocpy.io

Calculated derivable fields can be written back into their source NetCDF file so
that subsequent loads read them directly instead of recomputing. The variable is
computed on the full grid and **appended in place** to the file.

.. code-block:: python

    import gcmprocpy as gy

    datasets = gy.load_datasets('/path/to/output', dataset_filter='sech')
    gy.save_derived(datasets, ['N2', 'O/N2'])   # append once
    # next session: N2 and O/N2 are already in the file, read like any variable

.. note::
   This appends to the original file in place, so the file must be writable —
   read-only archive histories raise ``PermissionError`` (persist into a writable
   copy instead). NetCDF cannot delete a variable in place, so a field already
   present is skipped. Only derivable intermediate fields are persisted this way;
   slice-based derived outputs (emissions, OH bands, EP flux) are not.

.. autofunction:: save_derived
   :noindex:


tgcmproc Provenance & References
--------------------------------------------------------------------------------------------------------------------

The physics routines documented above are ports of NCAR's legacy ``tgcmproc``
Fortran post-processor. Each formula was cross-referenced against its original
source and verified for numerical equivalence; the table records the provenance
— source file, subroutine, and original author/attribution — so every ported
formula can be traced back to the Fortran.

.. list-table::
   :header-rows: 1
   :widths: 24 26 22 28

   * - gcmprocpy function
     - tgcmproc source
     - Attribution
     - Notes
   * - ``mkeno53`` (``NO53``)
     - ``mkemiss.F`` :: ``mkeno53``
     - John Wise
     - 5.3 µm NO emission; cgs number densities → photons cm⁻³ s⁻¹
   * - ``mkeco215`` (``CO215``)
     - ``mkemiss.F`` :: ``mkeco215``
     - John Wise
     - 15 µm CO₂ emission (O–CO₂ collisional term)
   * - ``mkeoh83`` (``OH83``)
     - ``mkemiss.F`` :: ``mkeoh83``
     - tgcmproc (6/95)
     - OH v(8,3) band emission
   * - ``ohrad`` (OH Meinel model)
     - ``ohrad.F`` :: ``ohrad``
     - B. Foster, U. B. Makhlouf (SDL/Stewart Radiance Lab)
     - 10-level steady-state OH(v) model; 39 Meinel bands
   * - ``epflux`` (``EPVY``/``EPVZ``/``EPVDIV``)
     - ``epflux.F`` :: ``epfluxy``/``epfluxz``/``epfluxdiv``/``save_epv``
     - B. Foster & Hanli Liu (1998)
     - Eliassen–Palm flux & divergence; see EPVDIV caveat below
   * - ``compute_barm``
     - ``denconv.F`` :: ``mkdenparms``
     - B. Foster
     - mean air molar mass ``1/(o2/32+o1/16+max(.00001,1-o2-o1)/28)`` (TIE-GCM branch)
   * - ``compute_pkt``
     - ``denconv.F`` :: ``mkdenparms``
     - B. Foster
     - air number density ``p0·exp(-ζ)/(boltz·TN)``; WACCM-X hybrid-pressure branch is a gcmprocpy extension
   * - ``convert_density_units``
     - ``denconv.F`` :: ``denconv``
     - B. Foster
     - MMR ↔ CM3 / CM3-MR / GM-CM3 (reverse paths are algebraic inversions)
   * - ``get_species_molar_mass``
     - ``fset_known.F`` :: ``fset_known``
     - tgcmproc
     - species molar masses (``flds_known(n)%wt``)
   * - ``N2`` derivable
     - ``mkderived.F`` :: ``mkderived``
     - tgcmproc
     - residual ``fn2 = max(.00001, 1-o2-o1)``
   * - ``O/N2``, ``N2/O``, ``O/O2``, ``O/O2+N2`` ratios
     - ``mkderived.F`` :: ``mkderived``
     - tgcmproc
     - composition ratios; unit-invariant, formed on the source fields
   * - ``RHO`` derivable
     - ``mkderived.F`` :: ``mkderived`` (L222) / ``mkrhokg``
     - tgcmproc
     - total mass density ``O2+O+N2`` → GM/CM3 (carries the ``pkt`` falloff)
   * - ``PMB`` derivable
     - ``mkderived.F`` :: ``mkderived`` (L798)
     - tgcmproc
     - ``(n_O2+n_O+n_N2)·k_B·T·1e-3`` = ``p·1e-3`` (mb)
   * - ``TNFP`` derivable
     - ``mkderived.F`` :: ``mkderived`` (L803)
     - tgcmproc
     - frost point ``T − 6077.4/(28.548 − ln(p_H2O[mb]))``; requires ``H2O``
   * - ``OX`` / ``NOZ`` / ``HOX`` groups
     - ``fset_known.F`` (native fields; ``fcomponents`` + group ``%wt`` 16/30/17)
     - tgcmproc
     - odd O/N/H groups (O+O3, NO+NO2, OH+HO2+H); summed from members when absent
   * - ``O/CO2`` ratio
     - ``mkderived.F`` :: ``mkderived`` (L618)
     - tgcmproc
     - atomic oxygen / CO2 ratio (terrestrial branch); unit-invariant

Scientific references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **EP flux** — Volland, H. (1988), *Atmospheric Tidal and Planetary Waves*, Kluwer Academic Publishers, §2.7.
- **OH Meinel model** — Einstein A coefficients: Nelson et al. (1990), updated Makhlouf (1999); quenching rates: Adler-Golden (1997); production branching: Klenerman & Smith (H + O₃), Kaye (O + HO₂).
- **NO 5.3 µm / CO₂ 15 µm emissions** — formulations from John Wise (via tgcmproc ``mkemiss.F``).

.. note::
   **EPVDIV mass density.** Following tgcmproc ``mkrhokg`` (``mkderived.F``),
   ``arr_epflux`` builds the zonal-mean mass density for EPVDIV by summing the
   major species (O, O₂, N₂) converted to ``GM/CM3`` via ``arr_density``
   (×1000 → kg m⁻³), so it carries the ``pkt = p/(k_B·T)`` vertical falloff for
   both TIE-GCM and WACCM-X and for any source unit (MMR / VMR / cm⁻³). If the
   required species — or, for WACCM-X, the hybrid-pressure coefficients
   (``hyam``/``hybm``/``PS``) — are unavailable, it falls back to the ideal-gas
   proxy ``ρ ∝ exp(-lev)/T``.
