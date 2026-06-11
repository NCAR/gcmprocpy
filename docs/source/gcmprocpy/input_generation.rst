Input File Generation
=============================================================

In addition to post-processing model output, gcmprocpy bundles two utilities that
**build the geophysical forcing / boundary-condition NetCDF files that drive a
TIE-GCM run**:

- **gpigen** — Geophysical Indices (GPI): daily 10.7 cm solar flux (``f107d``),
  its running average (``f107a``), and the 3-hourly Kp index, from
  `GFZ Potsdam <https://kp.gfz-potsdam.de/>`_.
- **imfgen** — Interplanetary Magnetic Field / solar-wind boundary conditions:
  ``bx``/``by``/``bz``, solar-wind density and velocity, from
  `OMNI <https://omniweb.gsfc.nasa.gov/ow_min.html>`_ 1-minute data or a BCWIND
  HDF5 file.

Both are available as console commands (``gpigen`` / ``imfgen``) and as a Python
API under ``gcmprocpy.gpigen`` / ``gcmprocpy.imfgen``. Each ``generate_*``
function returns an :class:`xarray.Dataset`, so the data can be inspected or
post-processed before being written to NetCDF.

.. note::
   These tools require network access to fetch the source data (the GFZ API for
   ``gpigen``; CDAWeb / OMNI for ``imfgen``), or a local copy of the source files.
   ``gpigen`` depends on ``requests``; ``imfgen`` depends on ``h5py`` (BCWIND) and
   ``hapiclient`` (OMNI via CDAWeb). All are installed automatically with gcmprocpy.


GPI (gpigen)
-------------------------------------------------------------------------------------------------------

Each output file holds, on a daily grid (``ndays``):

- ``year_day`` — ``YYYYDDD`` integer (4-digit year + 3-digit day of year)
- ``f107d`` — daily 10.7 cm solar flux
- ``f107a`` — running-average 10.7 cm solar flux (default 81-day centered)
- ``kp`` — 3-hourly Kp index, shaped ``(ndays, 8)``

Writes ``<prefix>_<begYYYYDDD>-<endYYYYDDD>.nc`` into ``--output-dir``.

Mode: CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Full series 1960-01-01 -> yesterday, 81-day centered avg, JSON API (defaults)
    gpigen

    # Arbitrary date range
    gpigen --start 2024-01-01 --end 2024-06-01

    # 27-day trailing average
    gpigen --window 27 --trailing --prefix gpi_27avg

    # Parse the raw 1932-onward text file instead of the JSON API, and write plots
    gpigen --source textfile --plots

.. autoprogram:: gcmprocpy.gpigen.cli:build_parser()
   :prog: gpigen

Mode: API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gcmprocpy import gpigen

    ds = gpigen.generate_gpi(
        start="2024-01-01",   # YYYY-MM-DD, YYYYDDD, ISO, or datetime
        end=None,             # default: yesterday
        source="json",        # "json" (GFZ API) or "textfile"
        window=81,            # averaging window in days
        centered=True,        # centered vs trailing
    )

    path = gpigen.save_gpi(ds, output_dir=".")     # write NetCDF
    gpigen.make_plots(ds, output_dir="plots")      # optional per-year PNGs

The top-level entry points ``generate_gpi``, ``save_gpi`` and ``make_plots`` are
also re-exported directly on the ``gcmprocpy`` namespace.

.. currentmodule:: gcmprocpy.gpigen.core

.. autofunction:: generate_gpi
   :noindex:

.. currentmodule:: gcmprocpy.gpigen.dataset

.. autofunction:: save_gpi
   :noindex:

.. autofunction:: build_dataset
   :noindex:

.. autofunction:: gpi_filename
   :noindex:

.. currentmodule:: gcmprocpy.gpigen.plotting

.. autofunction:: make_plots
   :noindex:


IMF / Solar-Wind Boundary Conditions (imfgen)
-------------------------------------------------------------------------------------------------------

Each output file holds, on a per-minute grid (``ndata``):

- ``bx``, ``by``, ``bz`` — IMF components (nT)
- ``swden`` — solar-wind proton density (cm\ :sup:`-3`)
- ``swvel`` — solar-wind flow speed (km/s)
- a 0/1 ``*Mask`` quality flag for each channel (``bxMask``, ``byMask``,
  ``bzMask``, ``denMask``, ``velMask``; **0 = linearly interpolated**)
- ``date`` — ``YYYYDDD.frac`` (year, day-of-year, fractional day)
- ``timestamp`` — ISO string ``YYYY-MM-DDTHH:MM:SS``

OMNI access modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OMNI source can be fetched two ways (``--omni-access`` / ``omni_access=``):

- ``hapi`` (**default**) — query CDAWeb's HAPI server for the ``OMNI_HRO_1MIN``
  dataset, retrieving **only the requested window**. Best for short ranges: a
  few-day request transfers a few days of data instead of whole-year files.
- ``asc`` — download and parse the SPDF ``omni_min<year>.asc`` files (over FTP
  into ``--cache-dir``). This reproduces the legacy per-year output exactly and
  is preferable for bulk regeneration of the full archive.

Both draw on the same underlying product (the same variables, fill values and
1-minute UTC grid), so the processed output matches.

Mode: CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # A specific range — fetched window-only from CDAWeb HAPI (default access).
    imfgen --start 2020-01-01 --end 2020-12-31

    # Full series 1982-01-01 -> yesterday, 10-min trailing average (defaults).
    imfgen

    # Reproduce the legacy per-year files from the SPDF ASCII archive.
    imfgen --split-years --omni-access asc --cache-dir ./omni_asc --output-dir .

    # Convert a BCWIND HDF5 file
    imfgen --source bcwind --bcwind-path bcwind.h5

.. autoprogram:: gcmprocpy.imfgen.cli:build_parser()
   :prog: imfgen

Mode: API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gcmprocpy import imfgen

    # OMNI -> one continuous Dataset for the range
    ds = imfgen.generate_imf(
        start="2020-01-01",   # YYYY-MM-DD, YYYYDDD, ISO, or datetime
        end=None,             # default: yesterday
        source="omni",        # "omni" (default) or "bcwind"
        window=10,            # trailing-average window, minutes
        cache_dir="./omni_asc",
    )
    path = imfgen.save_imf(ds, output_dir=".")          # write NetCDF

    # Per-year files (each interpolated within its own year), like the originals
    for ds_year in imfgen.generate_imf_years(start="1982-01-01", cache_dir="./omni_asc"):
        imfgen.save_imf(ds_year, output_dir=".")

    # BCWIND HDF5 -> Dataset
    ds = imfgen.generate_imf(source="bcwind", bcwind_path="bcwind.h5")

The top-level entry points ``generate_imf``, ``generate_imf_years`` and
``save_imf`` are also re-exported directly on the ``gcmprocpy`` namespace.

.. currentmodule:: gcmprocpy.imfgen.core

.. autofunction:: generate_imf
   :noindex:

.. autofunction:: generate_imf_years
   :noindex:

.. currentmodule:: gcmprocpy.imfgen.dataset

.. autofunction:: save_imf
   :noindex:

.. autofunction:: build_dataset
   :noindex:

.. autofunction:: imf_filename
   :noindex:
