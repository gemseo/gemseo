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
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import datetime
import os
import sys
from importlib.metadata import version
from pathlib import Path

import six
from sphinx_gallery.sorting import ExampleTitleSortKey

import gemseo

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath("_ext"))

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
    "add_toctree_functions",
    "gemseo_pre_processor",
]

apidoc_module_dir = "../../src/gemseo"
apidoc_output_dir = "_modules"
apidoc_separate_modules = True
apidoc_module_first = True

examples_dir = "examples"
examples_path = os.path.join("..", examples_dir)
examples_subdirs = [
    subdir
    for subdir in os.listdir(examples_path)
    if os.path.isdir(os.path.join(examples_path, subdir))
    and os.path.isfile(os.path.join(examples_path, subdir, "README.rst"))
]
tutorials_dir = "tutorials"
tutorials_path = os.path.join("..", tutorials_dir)
tutorials_subdirs = [
    subdir
    for subdir in os.listdir(tutorials_path)
    if os.path.isdir(os.path.join(tutorials_path, subdir))
    and os.path.isfile(os.path.join(tutorials_path, subdir, "README.rst"))
]
tmp1 = [os.path.join(examples_path, subdir) for subdir in examples_subdirs]
tmp2 = [os.path.join(tutorials_path, subdir) for subdir in tutorials_subdirs]
examples_dirs = tmp1 + tmp2
tmp1 = [os.path.join(examples_dir, subdir) for subdir in examples_subdirs]
tmp2 = [os.path.join(tutorials_dir + "_sg", subdir) for subdir in tutorials_subdirs]
gallery_dirs = tmp1 + tmp2
sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": examples_dirs,
    # path to where to save gallery generated output
    "gallery_dirs": gallery_dirs,
    "default_thumb_file": Path(__file__).parent / "_static/icon.png",
    "within_subsection_order": ExampleTitleSortKey,
    "ignore_pattern": r"run\.py",
}

napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

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
# See https://github.com/pypa/setuptools_scm#usage-from-sphinx
release = version("gemseo")
version = release

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
# html_theme = 'default'
# html_theme = 'sphinx_rtd_theme'
html_theme = "scikit-learn-modern"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}
if html_theme == "sphinx_rtd_theme":
    html_theme_options = {
        "canonical_url": "",
        "logo_only": False,
        "display_version": True,
        "prev_next_buttons_location": "bottom",
        # Toc options
        "collapse_navigation": False,
        "sticky_navigation": True,
        "navigation_depth": 4,
    }

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {'oldversion': False, 'collapsiblesidebar': True,
#                      'surveybanner': False, 'sprintbanner': True}

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
