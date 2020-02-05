# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import pathlib
import toml

ROOT_PATH = pathlib.Path(__file__).parent.parent

sys.path.insert(0, str(ROOT_PATH))


def get_version():
    pyprojecttoml = ROOT_PATH / 'pyproject.toml'
    with pyprojecttoml.open() as fp:
        config = toml.load(fp)
    return config['tool']['poetry']['version']


# -- Project information -----------------------------------------------------

project = 'scikit-mpe'
copyright = '2020, Eugene Prilepin'
author = 'Eugene Prilepin'

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
]

intersphinx_mapping = {
    'scikit-fmm': ('https://scikit-fmm.readthedocs.io/en/latest/', None)
}

autodoc_member_order = 'bysource'
numpydoc_show_class_members = False

plot_apply_rcparams = True
plot_rcparams = {
    'figure.autolayout': 'True',
    'figure.figsize': '5, 3.5',
    'savefig.bbox': 'tight',
    'savefig.facecolor': "None",
}

plot_formats = [("png", 90)]
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'fixed_sidebar': 'true',
    'show_powered_by': 'false',

    'description': 'Minimal path extraction',

    'github_user': 'espdev',
    'github_repo': 'scikit-mpe',
    'github_type': 'star',

    'extra_nav_links': {
        'GitHub repository': 'https://github.com/espdev/scikit-mpe',
        'PyPI': 'https://pypi.org/project/scikit-mpe',
    },
}