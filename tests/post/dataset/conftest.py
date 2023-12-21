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
from __future__ import annotations

import pytest
from numpy import array
from numpy import linspace
from numpy import newaxis

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Dataset: A dataset containing 4 samples of variables x, y and z and cluster c."""
    return Dataset.from_array(
        array([
            [0.0, 0.0, 0.0, 1],
            [1.0, 1.0, -1.0, 2],
            [2.0, 2.0, -2.0, 2],
            [3.0, 3.0, -3.0, 1],
        ]),
        ["x", "y", "z", "c"],
    )


@pytest.fixture(scope="module")
def quadratic_dataset() -> IODataset:
    """A dataset containing 10 equispaced evaluations of f(x)=x**2 over [0,1]."""
    x = linspace(0, 1, 10)[:, newaxis]
    # Mix the components to test the robustness of the tested features
    x = x[[2, 0, 4, 1, 9, 7, 6, 5, 3, 8]]
    dataset = IODataset()
    dataset.add_input_group(x, "x")
    dataset.add_output_group(x**2, "y")
    return dataset
