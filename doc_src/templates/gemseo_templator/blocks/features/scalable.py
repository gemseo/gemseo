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
    title="Scalable models",
    description=(
        "Use scalable data-driven models to compare MDO formulations and algorithms"
        " for different problem dimensions."
    ),
    features=[
        WebLink(
            "scalability study",
            url=(
                "scalable_models/index.html"
                "#benchmark-mdo-formulations-based-on-scalable-disciplines"
            ),
        ),
        WebLink(
            "scalable problem",
            url="scalable_models/index.html#scalable-mdo-problem",
        ),
        WebLink(
            "scalable discipline",
            url="scalable_models/index.html#scalable-discipline",
        ),
        WebLink(
            "diagonal-based",
            url="scalable_models/index.html#scalable-diagonal-model",
        ),
    ],
    examples="examples/scalable/index.html",
    info="scalable.html",
)
