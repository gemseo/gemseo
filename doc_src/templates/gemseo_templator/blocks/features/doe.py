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
    title="DOE &#x26; Trade-off",
    description=(
        "Define, solve and post-process a trade-off problem "
        "from a DOE (design of experiments) algorithm."
    ),
    url="algorithms/doe_algos.html",
    dependencies=[
        WebLink("OpenTURNS", url="http://www.openturns.org/"),
        WebLink("pyDOE", url="https://pythonhosted.org/pyDOE/"),
    ],
    examples="examples/doe/index.html",
    info="doe.html",
    options="algorithms/doe_algos.html",
)
