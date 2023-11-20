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
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test machine learning classification algorithm module."""

from __future__ import annotations

import pytest
from numpy import arange

from gemseo.datasets.dataset import Dataset


@pytest.fixture()
def dataset() -> Dataset:
    """A dataset used to train the classification algorithms."""
    data = arange(60).reshape(10, 6)
    variables = ["x_1", "x_2", "y_1"]
    variable_names_to_n_components = {"x_1": 1, "x_2": 2, "y_1": 3}
    variable_names_to_group_names = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    io_dataset = Dataset.from_array(
        data, variables, variable_names_to_n_components, variable_names_to_group_names
    )
    io_dataset.name = "dataset_name"
    return io_dataset
