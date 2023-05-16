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

from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.scalable.data_driven.factory import ScalableModelFactory
from numpy import array
from numpy import newaxis


def test_constructor():
    ScalableModelFactory()


def test_create():
    factory = ScalableModelFactory()
    dataset = IODataset()
    val = array([0.0, 0.25, 0.5, 0.75, 1.0])
    dataset.add_variable("x", (val * 2)[:, newaxis], dataset.INPUT_GROUP)
    dataset.add_variable("y", val[:, newaxis], dataset.INPUT_GROUP)
    dataset.add_variable("z", val[:, newaxis], dataset.OUTPUT_GROUP, False)
    factory.create("ScalableDiagonalModel", data=dataset)


def test_list_available():
    factory = ScalableModelFactory()
    assert "ScalableDiagonalModel" in factory.scalable_models


def test_is_available():
    factory = ScalableModelFactory()
    assert factory.is_available("ScalableDiagonalModel")
