# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Iris dataset
============

This is one of the best known :class:`.Dataset`
to be found in the machine learning literature.

It was introduced by the statistician Ronald Fisher
in his 1936 paper "The use of multiple measurements in taxonomic problems",
Annals of Eugenics. 7 (2): 179–188.

It contains 150 instances of iris plants:

- 50 Iris Setosa,
- 50 Iris Versicolour,
- 50 Iris Virginica.

Each instance is characterized by:

- its sepal length in cm,
- its sepal width in cm,
- its petal length in cm,
- its petal width in cm.

This :class:`.Dataset` can be used for either clustering purposes
or classification ones.

`More information about the Iris dataset
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_

"""
from __future__ import annotations

from pathlib import Path

from gemseo.core.dataset import Dataset


class IrisDataset(Dataset):
    """Iris dataset parametrization."""

    def __init__(self, name="Iris", by_group=True, as_io=False):
        """Constructor."""
        super().__init__(name, by_group)
        file_path = Path(__file__).parent / "iris.data"
        variables = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "specy",
        ]
        sizes = {
            "sepal_length": 1,
            "sepal_width": 1,
            "petal_length": 1,
            "petal_width": 1,
            "specy": 1,
        }
        if as_io:
            groups = {
                "sepal_length": "inputs",
                "sepal_width": "inputs",
                "petal_length": "inputs",
                "petal_width": "inputs",
                "specy": "outputs",
            }
        else:
            groups = {"specy": "labels"}

        self.set_from_file(file_path, variables, sizes, groups, ",", False)
