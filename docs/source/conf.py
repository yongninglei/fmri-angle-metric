"""Sphinx configuration for fmri-angle-metric."""

from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "fmri-angle-metric"
copyright = "2026, Yongninglei"
author = "Yongninglei"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "myst_parser",
]

templates_path = ["../_templates"]
exclude_patterns = []

autosummary_generate = True
autodoc_mock_imports = [
    "nibabel",
    "nilearn",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "typer",
    "rich",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
}

suppress_warnings = ["intersphinx.fetch_inventory"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["../_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yongninglei/fmri-angle-metric",
            "icon": "fa-brands fa-github",
        },
    ],
}


# -- linkcode ----------------------------------------------------------------

def linkcode_resolve(domain, info):
    """Map a Python object to its source file on GitHub."""
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
        source, lineno = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None

    if not fn:
        return None

    fn = os.path.relpath(fn, start=os.path.abspath("../../src"))
    linespec = f"#L{lineno}-L{lineno + len(source) - 1}"

    return (
        f"https://github.com/yongninglei/fmri-angle-metric/blob/main/src/{fn}{linespec}"
    )
