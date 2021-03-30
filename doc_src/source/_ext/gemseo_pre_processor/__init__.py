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

import os
from pathlib import Path

from . import generate_algos_doc, module_template


def setup(app):
    app.connect("builder-inited", builder_inited)
    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }


def builder_inited(app):
    gen_opts_path = Path(app.srcdir) / "algorithms" / "gen_opts"
    generate_algos_doc.main(gen_opts_path)
    _modules_path = Path(app.srcdir) / "_modules"
    # shutil.rmtree(_modules_path, ignore_errors=True)
    os.remove(_modules_path / "modules.rst")
    module_template.main(_modules_path)
