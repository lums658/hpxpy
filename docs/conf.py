# Sphinx configuration for hpxpy. Built on the Rostam runner where ``import hpxpy``
# works (the _core extension links HPX; only loading the .so is needed for autodoc,
# not a running HPX runtime). SPDX-License-Identifier: MIT
import hpxpy

project = "hpxpy"
author = "The HPX Project"
release = hpxpy.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # numpydoc-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# furo if available, else the bundled alabaster (keeps the build robust).
try:
    import furo  # noqa: F401
    html_theme = "furo"
except ImportError:
    html_theme = "alabaster"

html_title = f"hpxpy {release}"
