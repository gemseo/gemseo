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

block = Block(
    title="Study analysis",
    description=(
        "An intuitive tool to discover MDO without writing any code, "
        "and define the right MDO problem and process. "
        "From an Excel workbook, "
        '"specify your disciplines, design space, objective and constraints, '
        '"select an MDO formulation and plot both coupling structure '
        '(<a href="mdo/coupling.html#n2-chart">N2 chart</a>) '
        '"and MDO process '
        '(<a href="mdo/mdo_formulations.html#xdsm-visualization">XDSM</a>), '
        "even before wrapping any software."
    ),
    info="interface/study_analysis.html",
    examples="examples/study_analysis/index.html",
)
