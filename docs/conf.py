## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2025 IMEC. All rights reserved.
## ******************************************************************************
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VLSI Design Conference Tutorial 2025'
copyright = '2024, imec vzw'
author = 'imec vzw'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', 'sphinxcontrib.bibtex']

templates_path = ['_templates']
exclude_patterns = []
bibtex_bibfiles = ['references.bib']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

numfig = True
