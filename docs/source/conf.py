# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../gcmprocpy/src'))
sys.path.insert(1, os.path.abspath('../../gcmprocpy/src/gcmprocpy'))
sys.path.insert(2, os.path.abspath('../../gcmprocpy/src/gcmprocpy/cmd'))

project = 'gcmprocpy'
copyright = '2024, Nikhil Rao'
author = 'Nikhil Rao'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode','sphinxcontrib.autoprogram', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["numpy","cartopy","matplotlib","xarray","ipython","geomag","netcdf4","ipympl","mplcursors","PyQt5"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
