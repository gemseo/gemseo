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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.problems.scalable.data_driven.model import ScalableModel
from numpy import array
from numpy import newaxis


def test_notimplementederror():
    dataset = Dataset()
    val = array([0.0, 0.25, 0.5, 0.75, 1.0])
    dataset.add_variable("x", (val * 2)[:, newaxis], dataset.INPUT_GROUP)
    dataset.add_variable("y", val[:, newaxis], dataset.INPUT_GROUP)
    dataset.add_variable("z", val[:, newaxis], dataset.OUTPUT_GROUP, False)
    with pytest.raises(NotImplementedError):
        ScalableModel(dataset)

    class NewScalableModel(ScalableModel):
        def build_model(self):
            return None

    model = NewScalableModel(dataset)
    with pytest.raises(NotImplementedError):
        model.scalable_function()
    with pytest.raises(NotImplementedError):
        model.scalable_derivatives()
