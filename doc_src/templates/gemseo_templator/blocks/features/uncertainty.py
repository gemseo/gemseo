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
    title="Uncertainty",
    description="Define, propagate, analyze and manage uncertainties.",
    features=[
        WebLink("distribution", url="uncertainty/distribution.html"),
        WebLink("uncertain space", url="uncertainty/parameter_space.html"),
        WebLink(
            "empirical and parametric statistics",
            url="uncertainty/statistics.html",
        ),
        WebLink(
            "distribution fitting",
            url="_modules/gemseo.uncertainty.distributions.openturns.fitting.html",
        ),
        WebLink("sensitivity analysis", url="uncertainty/sensitivity.html"),
    ],
    dependencies=[WebLink("OpenTURNS", url="http://www.openturns.org/")],
    examples="examples/uncertainty/index.html",
    info="uncertainty.html",
    options="algorithms/uncertainty_algos.html",
)
