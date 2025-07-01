import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'taihu'
copyright = '2025, Jerome Tze-Hou Hsu, Wenjie Lu, Pinyi Li'
author = 'Jerome Tze-Hou Hsu, Wenjie Lu, Pinyi Li'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Extract docstrings
    'sphinx.ext.napoleon',    # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',    # Add source code links
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']