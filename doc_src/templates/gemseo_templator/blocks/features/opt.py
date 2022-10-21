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
        WebLink(
            "nlopt",
            url="https://nlopt.readthedocs.io/en/latest/",
        ),
        WebLink("scipy", url="https://www.scipy.org/"),
        WebLink("snopt", url="https://github.com/snopt/snopt-python"),
        WebLink(
            "pdfo",
            url="https://www.pdfo.net",
        ),
    ],
    examples="examples/optimization_problem/index.html",
    info="optimization.html",
    options="algorithms/opt_algos.html",
)
