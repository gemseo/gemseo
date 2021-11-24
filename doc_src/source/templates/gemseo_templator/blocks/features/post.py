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
from gemseo_templator.blocks.template import Block, WebLink

block = Block(
    title="Visualization",
    description="Generate graphical post-processings of optimization histories.",
    url="algorithms/post_algos.html",
    algorithms=[
        WebLink("basic history", anchor="basichistory"),
        WebLink("constraints history", anchor="constraintshistory"),
        WebLink("correlations", anchor="correlations"),
        WebLink("gradient sensitivity", anchor="gradientsensitivity"),
        WebLink("k-means", anchor="kmeans"),
        WebLink("objective and constraint history", anchor="objconstrhist"),
        WebLink("optimization history view", anchor="opthistoryview"),
        WebLink("optimization history view", anchor="parallelcoordinates"),
        WebLink("quadratic approximation", anchor="quadapprox"),
        WebLink("radar chart", anchor="radarchart"),
        WebLink("robustness", anchor="robustness"),
        WebLink("scatter matrix", anchor="scatterplotmatrix"),
        WebLink("self organizing map", anchor="som"),
        WebLink("variable influence", anchor="variableinfluence"),
    ],
    examples="examples/post_process/index.html",
    info="postprocessing/index.html",
    options="algorithms/post_algos.html",
)
