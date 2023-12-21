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
"""Iris dataset.

This is one of the best known :class:`.Dataset`
to be found in the machine learning literature.

It was introduced by the statistician Ronald Fisher
in his 1936 paper "The use of multiple measurements in taxonomic problems",
Annals of Eugenics. 7 (2): 179-188.

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

from numpy import int64 as np_int64
from pandas import factorize

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset


def create_iris_dataset(
    as_io: bool = False,
    as_numeric: bool = True,
) -> Dataset:
    """Iris dataset parametrization.

    Args:
        as_io: Whether to use Input/Output group names.
        as_numeric: Whether to consider a string label or a numeric one.

    Returns:
        The Iris dataset.
    """
    file_path = Path(__file__).parent / "iris.data"
    cls = IODataset if as_io else Dataset
    dataset = cls.from_csv(file_path)
    dataset.name = "Iris"

    if as_numeric:
        numeric_data, numeric_meaning = factorize(
            dataset.get_view(variable_names="specy").to_numpy().T[0]
        )
        dataset.update_data(numeric_data, variable_names="specy")
        dataset = dataset.astype({("labels", "specy", 0): np_int64})
        dataset.misc["labels"] = {"specy": numeric_meaning}

    if as_io:
        groups = {
            "parameters": IODataset.INPUT_GROUP,
            "labels": IODataset.OUTPUT_GROUP,
        }
        for group, new_group in groups.items():
            dataset.rename_group(group_name=group, new_group_name=new_group)

    return dataset
