# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/pytanksim'))


# -- Project information -----------------------------------------------------

project = 'pytanksim'
copyright = '2024, Muhammad Irfan Maulana Kusdhany'
author = 'Muhammad Irfan Maulana Kusdhany'

# The full version, including alpha/beta/rc tags
release = '1.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'autoapi.extension'
]

autodoc_typehints = "none"

autoapi_dirs = ['../../src/pytanksim']
autoapi_options = ['members', 'private-members', 'show-inheritance',
                   'show-module-summary', 'special-members',
                   'imported-members']
autoapi_keep_files = True
autoapi_python_class_content = 'both'
napoleon_google_docstring = False
autodoc_mock_imports = ["CoolProp", "assimulo"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "pytanksimallwhitelogo.svg"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
