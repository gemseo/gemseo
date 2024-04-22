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
#                         documentation
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import hstack
from numpy import ndarray

from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """The Rosenbrock dataset."""
    return create_rosenbrock_dataset(opt_naming=False)


@pytest.fixture(scope="module")
def dataset_2(dataset) -> IODataset:
    """The Rosenbrock dataset with 2d-output."""
    data = IODataset()
    data.add_variable(
        "x",
        dataset.get_view(variable_names="x").to_numpy(),
        group_name=data.INPUT_GROUP,
    )
    data.add_variable(
        "rosen",
        dataset.get_view(variable_names="rosen").to_numpy(),
        group_name=data.OUTPUT_GROUP,
    )
    data.add_variable(
        "rosen2",
        hstack((
            dataset.get_view(variable_names="rosen").to_numpy(),
            dataset.get_view(variable_names="rosen").to_numpy(),
        )),
        group_name=data.OUTPUT_GROUP,
    )
    return data


@pytest.fixture(scope="module")
def input_data(dataset: IODataset) -> ndarray:
    """The learning input data."""
    return dataset.get_view(group_names=dataset.INPUT_GROUP).to_numpy()


@pytest.fixture(scope="module")
def output_data(dataset: IODataset) -> ndarray:
    """The learning output data."""
    return dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
