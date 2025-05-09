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

project = 'Scalable System Simulations Tutorial'
copyright = '2024-2025, imec vzw'
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
html_logo = "512px-LOGO-IMEC_black.svg.png"
html_theme_options = {
    'logo_only': False,
    'display_version': False,
}
html_css_files = ['custom_svg.css']
numfig = True
