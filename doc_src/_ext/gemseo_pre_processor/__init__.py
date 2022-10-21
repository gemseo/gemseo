# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
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
"""A sphinx extension to pre-process the docs."""
from __future__ import annotations

from pathlib import Path

from sphinx.ext.autodoc.mock import mock

from . import authors_template
from . import generate_algos_doc
from . import module_template
from . import plugins_template


def setup(app):
    app.connect("builder-inited", builder_inited)
    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }


def builder_inited(app):
    gen_opts_path = Path(app.srcdir) / "algorithms" / "gen_opts"
    with mock(app.config.autodoc_mock_imports):
        # Mock the dependencies that are optional and the ones from the plugins.
        generate_algos_doc.main(gen_opts_path)
    root_path = Path(app.srcdir)
    _modules_path = root_path / "_modules"
    (_modules_path / "modules.rst").unlink(missing_ok=True)

    plugins = app.config.html_context["plugins"]
    names = ["gemseo"] + list(plugins)

    with open(_modules_path / "projects.rst", "w") as f:
        f.write(".. _projects:\n\n")
        f.write("Packages\n")
        f.write("--------\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for name in names:
            f.write(f"   {name.replace('-', '_')}\n")

    for name in names:
        module_template.main(_modules_path, name.replace("-", "_"))

    plugins_template.create_plugins_page(plugins, root_path)
    authors_template.create_authors_page(root_path)
