"""Sphinx configuration for the ashp documentation."""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())

from ashp import __version__  # noqa: E402

# -- Project information -----------------------------------------------------
project = "ashp"
author = "Mirza Arnaut"
copyright = f"{datetime.datetime.now().year}, {author}"
version = __version__
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # NumPy / Google style docstrings
    "sphinx.ext.intersphinx",        # cross-links to numpy, scipy, ...
    "sphinx_autodoc_typehints",      # type hints in the signature/params
    "autoapi.extension",             # generate API docs from the source
    "myst_parser",                   # Markdown pages
    "sphinx_copybutton",             # copy button on code blocks
    "sphinxcontrib.bibtex",          # references / citations
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- AutoAPI -----------------------------------------------------------------
autoapi_dirs = ["../../src/ashp"]
autoapi_type = "python"
autoapi_add_toctree_entry = True
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = False

# -- Napoleon ----------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_use_ivar = True             # render Attributes inline (no autoapi clash)
napoleon_preprocess_types = False    # let sphinx-autodoc-typehints handle types

# -- sphinx-autodoc-typehints ------------------------------------------------
always_document_param_types = True
typehints_fully_qualified = False
typehints_document_rtype = True
always_use_bars_union = True

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "linkify",
    "smartquotes",
    "tasklist",
]

# -- Bibtex ------------------------------------------------------------------
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_context = {
    "github_user": "arunoruto",
    "github_repo": "ashp",
    "github_version": "main",
    "doc_path": "docs/source/",
}
html_theme_options = {
    "github_url": "https://github.com/arunoruto/ashp",
    "use_edit_page_button": True,
}
pygments_style = "sphinx"

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}
