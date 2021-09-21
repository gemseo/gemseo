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

# GEMSEO documentation build configuration file, created by
# sphinx-quickstart on Tue Jan 17 11:38:51 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# import sys
# import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use Path.resolve() to make it absolute, like shown here.
# sys.path.insert(0, str(Path('.').resolve()))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import collections
import datetime
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple, Union

import requests
import six
import sphinx
import sphinx.ext.autodoc.typehints
from sphinx.util import inspect, typing
from sphinx_gallery.sorting import ExampleTitleSortKey

import gemseo

# add faked packages for missing deps
try:
    from optimize.snopt7 import SNOPT_solver  # noqa: F401
except ImportError:
    sys.path.append(str(Path("fake_packages/snopt").resolve()))

try:
    import matlab  # noqa: F401
except ImportError:
    sys.path.append(str(Path("fake_packages/matlab").resolve()))

try:
    import da  # noqa: F401
except ImportError:
    sys.path.append(str(Path("fake_packages/pseven").resolve()))

os.chdir((Path(__file__).resolve()).parent)

sys.path.append(str(Path("_ext").resolve()))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinxcontrib.plantuml",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.apidoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "autodocsumm",
    "add_toctree_functions",
    "gemseo_pre_processor",
]

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

################################################################################
# Settings for napoleon.

# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
napoleon_include_special_with_doc = False

################################################################################
# Settings for apidoc.

apidoc_module_dir = "../../src/gemseo"
apidoc_excluded_paths = [
    "utils/n2d3/js",
    "utils/n2d3/css",
    "core/grammar.py",
    "core/json_grammar.py",
    "third_party/fastjsonschema/version.py",
]
apidoc_output_dir = "_modules"
apidoc_separate_modules = True
apidoc_module_first = True

################################################################################
# Settings for sphinx_gallery.

examples_dir = Path("examples")
examples_path = Path(".." / examples_dir)
examples_subdirs = [
    subdir
    for subdir in examples_path.iterdir()
    if (examples_path / subdir).is_dir()
    and (examples_path / subdir / "README.rst").is_file()
]

examples_dirs = [(examples_path / subdir) for subdir in examples_subdirs]
gallery_dirs = [(examples_dir / subdir) for subdir in examples_subdirs]

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": examples_dirs,
    # path to where to save gallery generated output
    "gallery_dirs": gallery_dirs,
    "default_thumb_file": Path(__file__).parent / "_static/icon.png",
    "within_subsection_order": ExampleTitleSortKey,
    "ignore_pattern": r"run\.py",
    "only_warn_on_example_error": True,
}

################################################################################
# Settings for sphinx.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix of source filenames.
source_suffix = ".rst"

nitpicky = True
nitpick_ignore = []

for line in open("nitpick-exceptions"):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, six.u(target)))

todo_include_todos = True
# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "GEMSEO"

#
copyright = "{}, IRT Saint Exupéry".format(datetime.datetime.now().year)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = gemseo.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "python"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# exclude_patterns = ["_autosummary", "*tests*.rst", "_modules"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "scikit-learn-modern"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ["themes"]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

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
    app.add_js_file("js/copybutton.js")


# def setup(app):
#     if html_theme == 'sphinx_rtd_theme':
#         app.add_stylesheet('custom.css')
#     app.add_stylesheet('slideshow.css')
#     app.add_javascript('slideshow.js')

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
    "index": "index.html",
    "documentation": "documentation.html",
}  # redirects to index

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None


# Output file base name for HTML help builder.
htmlhelp_basename = "GEMSEOdoc"


# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': ''
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "GEMSEO.tex", "GEMSEO Documentation", "IRT Saint Exupéry", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = "_images/logo.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "gemseo", "GEMSEO Documentation", ["IRT Saint Exupéry"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "GEMSEO",
        "GEMSEO Documentation",
        "IRT Saint Exupéry ",
        "GEMSEO",
        "GEMSEO (Generic Engine for Multi-disciplinary Scenarios, "
        "Exploration and Optimization)"
        "is developed in order to "
        "facilitate Multi-Disciplinary Design Optimization (MDO) based design.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False
# The sphinx.ext.autosummary can also create .rst files with autodoc directives, but,
# unlike sphinx-apidoc, those files have autousummary tables for classes,
# functions and exceptions.
# That is #sphinx.ext.autosummary can be used instead of sphinx-apidoc.
# To enable this feature you need to include autosummary_generate = True
# parameter into your conf.py file.

autosummary_generate = True

# Additional stuff for the LaTeX preamble.
latex_elements["preamble"] = "\\usepackage{amsmath}\n\\usepackage{amssymb}\n"

#####################################################
# add LaTeX macros

f_html = open("macros/latex_macros_html.sty")
f_latex = open("macros/latex_macros_latex.sty")

imgmath_latex_preamble = ""

for macro in f_latex:
    # used when building latex and pdf versions
    latex_elements["preamble"] += macro + "\n"

for macro in f_html:
    # used when building html version
    imgmath_latex_preamble += macro + "\n"

if "PLANTUML_DIR" in os.environ:
    plantuml_dir = os.environ["PLANTUML_DIR"]
else:
    plantuml_dir = "/opt/plantuml/"

plantuml = "java -jar {}/plantuml.jar".format(plantuml_dir)
plantuml_output_format = "png"

autoclass_content = "both"

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

rst_prolog = """
.. |g| replace:: GEMSEO
"""

################################################################################
# Settings for readthedocs.

# Setup the multiversion display
html_context = dict()


def __filter_versions(
    rtd_versions,  # type: Iterable[Mapping[str,Union[str,Mapping[str,str]]]]
):  # type: (...) -> List[Tuple[str,str]]
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


################################################################################
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
