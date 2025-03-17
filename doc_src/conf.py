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
import json
import os
import re
import sys
from importlib.metadata import version as _version
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from packaging.version import parse
from sphinx_gallery.sorting import ExampleTitleSortKey

os.chdir((Path(__file__).resolve()).parent)

for directory_name in ("_ext", "templates"):
    sys.path.append(str(Path(directory_name).resolve()))


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

extensions = [
    "add_toctree_functions",
    "autodocsumm",
    "default_kwargs_values",
    "gemseo_pre_processor",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.plantuml",
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

################################################################################
# Settings for autodoc.

# generate autosummary even if no references
autosummary_generate = True

# If False, class ClassName instead of class module_path.ClassName
add_module_names = False

autodoc_default_options = {"members": True, "inherited-members": False}

autodoc_member_order = "groupwise"

# Show the typehints in the description instead of the signature.
autodoc_typehints = "description"

# Both the class' and the __init__ method's docstring are concatenated and inserted.
autoclass_content = "both"

# Show arguments default values.
autodoc_kwargs_defaults = True

# Mock missing dependencies.
autodoc_mock_imports = [
    "da",
    "delta",
    "fmpy",
    "hexaly",
    "jax",
    "jnius",
    "matlab",
    "matlabengine",
    "optimize",
    "pdfo",
    "petsc4py",
    "pymoo",
    "pyoptsparse",
    "PySide6",
    "pytest",
    "scilab2py",
    "smt",
]

################################################################################
# Settings for napoleon.

# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx's default behavior.
napoleon_include_special_with_doc = False

################################################################################
# Settings for sphinx.

# If True,
# the Smart Quotes transform will be used
# to convert quotation marks and dashes to typographically correct entities.
# This can be very slow.
smartquotes = False

# If this is True, todo and todolist produce output, else they produce nothing.
todo_include_todos = True

# General information about the project.
project = "GEMSEO"

copyright = f"{datetime.datetime.now().year}, IRT Saint Exupéry"  # noqa: A001

pretty_version = release = version = _version("gemseo")
if "dev" in pretty_version:
    pretty_version = "develop"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    "use_edit_page_button": True,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://gitlab.com/gemseo/dev/gemseo",
            "icon": "fa-brands fa-square-gitlab",
            "type": "fontawesome",
        },
    ],
    # If "prev-next" is included in article_footer_items, then setting show_prev_next
    # to True would repeat prev and next links. See
    # https://github.com/pydata/pydata-sphinx-theme/blob/b731dc230bc26a3d1d1bb039c56c977a9b3d25d8/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/layout.html#L118-L129
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    "switcher": {
        "json_url": "https://gemseo.readthedocs.io/en/stable/_static/versions.json",
        "version_match": release,
    },
    # check_switcher may be set to False if docbuild pipeline fails. See
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html#configure-switcher-json-url
    "check_switcher": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "logo": {
        "alt_text": "gemseo homepage",
        "image_relative": "_static/logo-small.png",
        "image_light": "_static/logo-small.png",
        "image_dark": "_static/logo-small.png",
    },
    "surface_warnings": True,
    # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": ["prev-next"],
    "content_footer_items": [],
    # Use html_sidebars that map page patterns to list of sidebar templates
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    # When specified as a dictionary, the keys should follow glob-style patterns, as in
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # In particular, "**" specifies the default for all pages
    # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    "secondary_sidebar_items": {
        "**": [
            "page-toc",
            "sourcelink",
            # Sphinx-Gallery-specific sidebar components
            # https://sphinx-gallery.github.io/stable/advanced.html#using-sphinx-gallery-sidebar-components
            "sg_download_links",
            "sg_launcher_links",
        ],
    },
    "show_version_warning_banner": True,
    "announcement": None,
}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, maps document names to template names.
# Workaround for removing the left sidebar on pages without TOC
# A better solution would be to follow the merge of:
# https://github.com/pydata/pydata-sphinx-theme/pull/1682
html_sidebars = {
    "user_guide": [],
    "api": [],
    "examples_and_tutorials": [],
    "faq": [],
    "support": [],
    "related_projects": [],
    "roadmap": [],
    "governance": [],
    "about": [],
}


def setup(app) -> None:
    app.add_css_file("gemseo.css")
    app.add_css_file("xdsm/fontello.css")
    app.add_css_file("xdsm/xdsmjs.css")
    app.add_js_file("xdsm/xdsmjs.js")


# Output file base name for HTML help builder.
htmlhelp_basename = "GEMSEOdoc"

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
    json_versions = []
    versions_ = []
    stable_index = 0
    version_index = -1
    stable_version = parse("0.0.0")
    for rtd_version in rtd_versions:
        slug = rtd_version["slug"]
        if rtd_version["active"] and __VERSION_REGEX.match(slug):
            versions_.append((slug, url := rtd_version["urls"]["documentation"]))
            json_versions.append({"name": slug, "version": slug, "url": url})
            version_index += 1
            if slug != "develop":
                parsed_slug = parse(slug)
                if parsed_slug > stable_version:
                    stable_version = parsed_slug
                    stable_index = version_index

    json_versions[stable_index]["preferred"] = True

    with open(Path(__file__).parent / "_static" / "versions.json", "w") as fp:
        json.dump(json_versions, fp)
    return versions_


if os.environ.get("READTHEDOCS") == "True":
    versions = requests.get(
        "https://readthedocs.org/api/v3/projects/gemseo/versions/",
        headers={"Authorization": "token 53f714afc37ec42e882efa094e6e3827202f801d"},
    ).json()["results"]
    html_context["versions"] = __filter_versions(versions)

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
            True,
        ),
        "gemseo-calibration": (
            "Capability to calibrate GEMSEO disciplines from data",
            True,
        ),
        "gemseo-fmu": ("GEMSEO plugin for FMU dynamic models", True),
        "gemseo-hexaly": ("GEMSEO interface to Hexaly solver", True),
        "gemseo-jax": ("GEMSEO plugin for JAX", True),
        "gemseo-matlab": ("GEMSEO plugin for MATLAB.", True),
        "gemseo-mlearning": ("Miscellaneous machine learning capabilities", True),
        "gemseo-mma": (
            "GEMSEO plugin for the MMA (Method of Moving Asymptotes) algorithm.",
            True,
        ),
        "gemseo-pdfo": ("GEMSEO plugin for the PDFO library.", True),
        "gemseo-petsc": (
            "PETSc wrapper for :class:`.LinearSolver` and :class:`.BaseMDA`",
            True,
        ),
        "gemseo-pseven": ("GEMSEO plugin for the pSeven library.", True),
        "gemseo-pymoo": ("Pymoo wrapper for optimization algorithms", True),
        "gemseo-pyoptsparse": ("GEMSEO interface to pyoptsparse algorithms.", True),
        "gemseo-ssh": ("SSH plugin for GEMSEO", True),
        "gemseo-scilab": ("Interfacing Scilab functions", True),
        "gemseo-template-editor-gui": (
            "A GUI to create input and output file templates for DiscFromExe.",
            True,
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

###############################################################################
# autodoc-pydantic

autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_hide_paramlist = False
autodoc_pydantic_model_signature_prefix = "Settings"
autodoc_pydantic_model_erdantic_figure = False
autodoc_pydantic_model_erdantic_figure_collapsed = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_signature_prefix = "Settings"
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_signature_prefix = ""
