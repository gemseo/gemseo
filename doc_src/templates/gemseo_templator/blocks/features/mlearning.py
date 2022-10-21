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
    title="Machine learning",
    description=(
        "Apply clustering, classification and regression methods "
        "from the machine learning community."
    ),
    features=[
        WebLink("clustering", url="machine_learning/clustering/clustering_models.html"),
        WebLink(
            "classification",
            url="machine_learning/classification/classification_models.html",
        ),
        WebLink("regression", url="machine_learning/regression/regression_models.html"),
        WebLink(
            "quality measures",
            url="machine_learning/qual_measure/quality_measures.html",
        ),
        WebLink(
            "data transformation", url="machine_learning/transform/transformer.html"
        ),
    ],
    dependencies=[
        WebLink("OpenTURNS", url="http://www.openturns.org/"),
        WebLink("scikit-learn", url="https://scikit-learn.org/stable/"),
    ],
    examples="examples/mlearning/index.html",
    info="machine_learning.html",
    options="algorithms/mlearning_algos.html",
)
