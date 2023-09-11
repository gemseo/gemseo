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
from gemseo.datasets.dataset import Dataset
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 4 samples of variables x, y and z and cluster c."""
    sample1 = [0.0, 0.0, 0.0, 1]
    sample2 = [1.0, 1.0, -1.0, 2]
    sample3 = [2.0, 2.0, -2.0, 2]
    sample4 = [3.0, 3.0, -3.0, 1]
    dataset = Dataset.from_array(
        array([sample1, sample2, sample3, sample4]), ["x", "y", "z", "c"]
    )
    return dataset
