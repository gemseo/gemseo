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

import collections
import datetime
import os
import re
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Iterable
from typing import Mapping

import requests
import sphinx.ext.autodoc.typehints
from sphinx.util import inspect
from sphinx.util import typing
from sphinx_gallery.sorting import ExampleTitleSortKey

os.chdir((Path(__file__).resolve()).parent)

for directory_name in ("_ext", "templates"):
    sys.path.append(str(Path(directory_name).resolve()))

from gemseo_templator.blocks import features, main_concepts  # noqa: E402

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
    "sphinx_gallery.gen_gallery",
    "autodocsumm",
    "add_toctree_functions",
    "gemseo_pre_processor",
    "default_kwargs_values",
]

################################################################################
# Settings for sphinx_gallery.
current_dir = Path(__file__).parent
gallery_dir = current_dir / "examples"
examples_dir = current_dir / "_examples"
examples_subdirs = [
    subdir.name
    for subdir in examples_dir.iterdir()
    if (examples_dir / subdir).is_dir()
    and (examples_dir / subdir / "README.rst").is_file()
]

examples_dirs = [(examples_dir / subdir) for subdir in examples_subdirs]
gallery_dirs = [(gallery_dir / subdir) for subdir in examples_subdirs]

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": examples_dirs,
    # path to where to save gallery generated output
    "gallery_dirs": gallery_dirs,
    "default_thumb_file": current_dir / "_static/icon.png",
    "within_subsection_order": ExampleTitleSortKey,
    "filename_pattern": r"\.py$",
    "ignore_pattern": r"run\.py",
    "only_warn_on_example_error": True,
}

################################################################################
# Settings for autodoc.

autodoc_default_options = {
    "inherited-members": True,
    "autosummary": True,
}

# Show the typehints in the description instead of the signature.
autodoc_typehints = "description"

# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content = "both"

# Show arguments default values.
autodoc_kwargs_defaults = True

# Mock missing dependencies.
autodoc_mock_imports = [
    "optimize",
    "matlab",
    "da",
    "pymoo",
    "petsc4py",
    "jnius",
    "jnius_config",
    "jep",
    "scilab2py",
]

################################################################################
# Settings for napoleon.

# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
napoleon_include_special_with_doc = False

################################################################################
# Settings for sphinx.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix of source filenames.
source_suffix = ".rst"

nitpick_ignore_regex = [
    ("py:.*", ".*PySide6.*"),
]

todo_include_todos = True

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "GEMSEO"

copyright = f"{datetime.datetime.now().year}, IRT Saint Exupéry"

release = version("gemseo")
version = release

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


# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
    "index": "index.html",
    "documentation": "documentation.html",
}

# Output file base name for HTML help builder.
htmlhelp_basename = "GEMSEOdoc"

autosummary_generate = True

if "PLANTUML_DIR" in os.environ:
    plantuml_dir = os.environ["PLANTUML_DIR"]
else:
    plantuml_dir = "/opt/plantuml/"

plantuml = f"java -jar {plantuml_dir}/plantuml.jar"
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
}


################################################################################
# Setup the multiversion display

html_context = dict()


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
    return [
        (rtd_version["slug"], rtd_version["urls"]["documentation"])
        for rtd_version in rtd_versions
        if rtd_version["active"] and re.match(r"\d+\.\d+\.\d+", rtd_version["slug"])
    ]


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
html_context["plugins"] = {
    "gemseo-java": "Interfacing Java code",
    "gemseo-petsc": "PETSc wrapper for :class:`.LinearSolver` and :class:`.MDA`",
    "gemseo-pymoo": "Pymoo wrapper for optimization algorithms",
    "gemseo-scilab": "Intefacing Scilab functions",
}

###############################################################################
# Sphinx workaround for duplicated args when using typehints
# TODO: remove when it is fixed upstream, see
# https://github.com/sphinx-doc/sphinx/pull/9648

__ANNOTATION_KIND_TO_PARAM_PREFIX = {
    inspect.Parameter.VAR_POSITIONAL: "*",
    inspect.Parameter.VAR_KEYWORD: "**",
}


def record_typehints(
    app,
    objtype,
    name,
    obj,
    options,
    args,
    retann,
):
    """Record type hints to env object."""
    # Fix for type annotation in comments
    if isinstance(obj, type):
        if "__init__" in obj.__dict__:
            obj = obj.__init__
        elif "__new__" in obj.__dict__:
            obj = obj.__new__
    try:
        if callable(obj):
            annotations = app.env.temp_data.setdefault("annotations", {})
            annotation = annotations.setdefault(name, collections.OrderedDict())
            sig = inspect.signature(obj, type_aliases=app.config.autodoc_type_aliases)
            for param in sig.parameters.values():
                if param.annotation is not param.empty:
                    prefix = __ANNOTATION_KIND_TO_PARAM_PREFIX.get(param.kind, "")
                    name = f"{prefix}{param.name}"
                    annotation[name] = typing.stringify(param.annotation)
            if sig.return_annotation is not sig.empty:
                annotation["return"] = typing.stringify(sig.return_annotation)
    except (TypeError, ValueError):
        pass


sphinx.ext.autodoc.typehints.record_typehints = record_typehints
