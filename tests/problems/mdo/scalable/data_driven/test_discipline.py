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
from __future__ import annotations

import pytest
from numpy import array
from numpy import newaxis

from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.mdo.scalable.data_driven.discipline import (
    DataDrivenScalableDiscipline,
)


@pytest.fixture
def dataset():
    data = IODataset()
    val = array([0.0, 0.25, 0.5, 0.75, 1.0])
    data.add_variable("x", (val * 2)[:, newaxis], data.INPUT_GROUP)
    data.add_variable("y", val[:, newaxis], data.INPUT_GROUP)
    data.add_variable("z", val[:, newaxis], data.OUTPUT_GROUP, False)
    return data


def test_constructor(dataset) -> None:
    DataDrivenScalableDiscipline("ScalableDiagonalModel", dataset)
