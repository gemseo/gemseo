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

from gemseo.problems.dataset.factory import DatasetFactory


def test_instantiate_factory():
    DatasetFactory()


def test_datasets():
    factory = DatasetFactory()
    datasets = factory.datasets
    assert "IrisDataset" in datasets
    assert "RosenbrockDataset" in datasets


def test_is_available():
    factory = DatasetFactory()
    assert factory.is_available("IrisDataset")
    assert not factory.is_available("DummyDataset")


def test_create():
    factory = DatasetFactory()
    dataset = factory.create("IrisDataset")
    assert dataset.name == "Iris"
    dataset = factory.create("RosenbrockDataset")
    assert dataset.name == "Rosenbrock"
