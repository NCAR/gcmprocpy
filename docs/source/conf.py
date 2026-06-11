# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# Only the src root needs to be importable; every autodoc/autoprogram target is
# referenced by its fully-qualified ``gcmprocpy.*`` path. Adding the package and
# cmd directories directly to sys.path (as was done previously) shadows stdlib
# modules of the same name (``cmd``, ``io``) and breaks the Sphinx build.
sys.path.insert(0, os.path.abspath('../../gcmprocpy/src'))

project = 'gcmprocpy'
copyright = '2024, Nikhil Rao'
author = 'Nikhil Rao'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode','sphinxcontrib.autoprogram', 'sphinx.ext.mathjax', 'nbsphinx']

nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['**.ipynb_checkpoints']

autodoc_mock_imports = ["numpy","cartopy","matplotlib","xarray","ipython","geomag","netcdf4","mplcursors","PySide6","dask","scipy","requests","h5py","hapiclient"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
