# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# -- General configuration ------------------------------------------------
from __future__ import annotations

import datetime
import os
import re
import sys
from dataclasses import asdict
from importlib.metadata import version as _version
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from sphinx_gallery.sorting import ExampleTitleSortKey

os.chdir((Path(__file__).resolve()).parent)

for directory_name in ("_ext", "templates"):
    sys.path.append(str(Path(directory_name).resolve()))


from gemseo_templator.blocks import features  # noqa: E402
from gemseo_templator.blocks import main_concepts  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.plantuml",
    "sphinxcontrib.bibtex",
    "autodocsumm",
    "add_toctree_functions",
    "gemseo_pre_processor",
    "default_kwargs_values",
]

################################################################################
# Settings for sphinx_gallery.
if not os.environ.get("DOC_WITHOUT_GALLERY"):
    extensions += ["sphinx_gallery.gen_gallery"]
    current_dir = Path(__file__).parent
    gallery_dir = current_dir / "examples"
    examples_dir = current_dir / "_examples"
    examples_subdirs = [
        subdir.name
        for subdir in examples_dir.iterdir()
        if (examples_dir / subdir).is_dir()
        and (examples_dir / subdir / "README.rst").is_file()
    ]

    examples_dirs = [str(examples_dir / subdir) for subdir in examples_subdirs]
    gallery_dirs = [str(gallery_dir / subdir) for subdir in examples_subdirs]

    sphinx_gallery_conf = {
        # path to your example scripts
        "examples_dirs": examples_dirs,
        # path to where to save gallery generated output
        "gallery_dirs": gallery_dirs,
        "default_thumb_file": str(current_dir / "_static/icon.png"),
        "within_subsection_order": ExampleTitleSortKey,
        "filename_pattern": r"plot_\w+\.py$",
        "ignore_pattern": r"run\.py",
        "only_warn_on_example_error": True,
        "nested_sections": False,
        # directory where function/class granular galleries are stored
        "backreferences_dir": "gen_modules/backreferences",
        # Modules for which function/class level galleries are created. In
        # this case sphinx_gallery and numpy in a tuple of strings.
        "doc_module": "gemseo",
        # objects to exclude from implicit backreferences. The default option
        # is an empty set, i.e. exclude nothing.
        "exclude_implicit_doc": {r"gemseo\.configure_logger"},
    }

################################################################################
# Settings for autodoc.

autodoc_default_options = {"members": True, "inherited-members": True}

autodoc_member_order = "groupwise"

autosummary_generate = True

# Show the typehints in the description instead of the signature.
autodoc_typehints = "description"

# Both the class' and the __init__ method's docstring are concatenated and inserted.
autoclass_content = "both"

# Show arguments default values.
autodoc_kwargs_defaults = True

# Mock missing dependencies.
autodoc_mock_imports = [
    "optimize",
    "matlab",
    "matlabengine",
    "da",
    "pymoo",
    "petsc4py",
    "scilab2py",
    "pyfmi",
    "fmpy",
    "pdfo",
    "jnius",
    "PySide6",
    "pytest",
]

################################################################################
# Settings for napoleon.

# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx's default behavior.
napoleon_include_special_with_doc = False

################################################################################
# Settings for sphinx.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix of source filenames.
source_suffix = ".rst"

nitpick_ignore_regex = []

todo_include_todos = True

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "GEMSEO"

copyright = f"{datetime.datetime.now().year}, IRT Saint Exupéry"  # noqa: A001

pretty_version = release = version = _version("gemseo")
if "dev" in pretty_version:
    pretty_version = "develop"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "scikit-learn-modern"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ["themes"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None
html_logo = "_static/logo-small.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/gemseo.css")
    app.add_css_file("css/all.css")
    app.add_css_file("xdsm/fontello.css")
    app.add_css_file("xdsm/xdsmjs.css")
    app.add_js_file("xdsm/xdsmjs.js")


# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
    "index": "index.html",
    "documentation": "documentation.html",
}

# Output file base name for HTML help builder.
htmlhelp_basename = "GEMSEOdoc"

autosummary_generate = True

if "READTHEDOCS" in os.environ:
    plantuml = "java -Djava.awt.headless=true -jar /usr/share/plantuml/plantuml.jar"
elif "PLANTUML_DIR" in os.environ:
    plantuml = f"java -jar {os.environ['PLANTUML_DIR']}/plantuml.jar"

plantuml_output_format = "png"

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

rst_prolog = """
.. |g| replace:: GEMSEO
"""

################################################################################
# Inter sphinx settings

intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "openturns": ("https://openturns.github.io/openturns/latest", None),
}

################################################################################
# Setup the multiversion display

html_context = {}
html_context["pretty_version"] = pretty_version

__VERSION_REGEX = re.compile(r"^(develop|\d+\.\d+\.\d+\.?\w*)$")


def __filter_versions(
    rtd_versions: Iterable[Mapping[str, str | Mapping[str, str]]],
) -> list[tuple[str, str]]:
    """Select the active versions with a version number.

    A version number follows the semantic versioning: MAJOR.MINOR.PATCH.

    Args:
        rtd_versions: The versions returned by the ReadTheDocs API.

    Returns:
        The active versions with a version number,
        of the form ``(version_name, version_url)``.
    """
    _versions = []
    for rtd_version in rtd_versions:
        slug = rtd_version["slug"]
        if rtd_version["active"] and __VERSION_REGEX.match(slug):
            _versions.append((slug, rtd_version["urls"]["documentation"]))
    return _versions


if os.environ.get("READTHEDOCS") == "True":
    versions = requests.get(
        "https://readthedocs.org/api/v3/projects/gemseo/versions/",
        headers={"Authorization": "token 53f714afc37ec42e882efa094e6e3827202f801d"},
    ).json()["results"]
    html_context["versions"] = __filter_versions(versions)

html_context["features"] = [asdict(feature) for feature in features]
html_context["main_concepts"] = [asdict(main_concept) for main_concept in main_concepts]
html_context["meta_description"] = (
    "GEMSEO: A Generic Engine for Multi-disciplinary Scenarios, "
    "Exploration and Optimization"
)
html_context["meta_og_description"] = (
    "Open source MDO in Python. "
    "Connect your tools. Explore your design space. Find solutions."
)
html_context["meta_og_root_url"] = "https://gemseo.readthedocs.io/en"

html_context["plugins"] = {}
if not os.environ.get("DOC_WITHOUT_PLUGINS"):
    html_context["plugins"] = {
        "gemseo-benchmark": (
            "A GEMSEO-based package to benchmark optimization algorithm.",
            False,
        ),
        "gemseo-calibration": (
            "Capability to calibrate GEMSEO disciplines from data",
            False,
        ),
        "gemseo-fmu": ("GEMSEO plugin for FMU dynamic models", True),
        "gemseo-matlab": ("GEMSEO plugin for MATLAB.", False),
        "gemseo-mlearning": ("Miscellaneous machine learning capabilities", False),
        "gemseo-mma": (
            "GEMSEO plugin for the MMA (Method of Moving Asymptotes) algorithm.",
            False,
        ),
        "gemseo-pdfo": ("GEMSEO plugin for the PDFO library.", False),
        "gemseo-petsc": (
            "PETSc wrapper for :class:`.LinearSolver` and :class:`.MDA`",
            False,
        ),
        "gemseo-pseven": ("GEMSEO plugin for the pSeven library.", False),
        "gemseo-pymoo": ("Pymoo wrapper for optimization algorithms", False),
        "gemseo-scilab": ("Interfacing Scilab functions", False),
        "gemseo-template-editor-gui": (
            "A GUI to create input and output file templates for DiscFromExe.",
            False,
        ),
        "gemseo-umdo": ("Capability for MDO under uncertainty", True),
    }
html_context["js_files"] = ["_static/jquery.js", "_static/xdsm/xdsmjs.js"]

###############################################################################
# Settings for inheritance_diagram

inheritance_edge_attrs = {
    "arrowsize": 1.0,
    "arrowtail": '"empty"',
    "arrowhead": '"none"',
    "dir": '"both"',
    "style": '"setlinewidth(0.5)"',
}

###############################################################################
# bibtex

bibtex_bibfiles = ["references.bib"]
