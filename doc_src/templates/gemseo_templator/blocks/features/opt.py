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
from __future__ import annotations

from gemseo_templator.blocks.template import Block
from gemseo_templator.blocks.template import WebLink

block = Block(
    title="Optimization",
    description=(
        "Define, solve and post-process an optimization problem "
        "from an optimization algorithm."
    ),
    url="algorithms/opt_algos.html",
    dependencies=[
        WebLink("GCMMA-MMA", url="https://github.com/arjendeetman/GCMMA-MMA-Python"),
        WebLink(
            "NLopt",
            url="https://nlopt.readthedocs.io/en/latest/",
        ),
        WebLink("PDFO", url="https://www.pdfo.net/"),
        WebLink("pSeven", url="https://www.pseven.io/product/pseven/"),
        WebLink("pymoo", url="https://pymoo.org/"),
        WebLink("SciPy", url="https://scipy.org/"),
    ],
    button_info_url="optimization.html",
    button_examples_url="examples/optimization_problem/index.html",
    button_types_url="algorithms/opt_algos.html",
)
