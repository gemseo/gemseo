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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.problems.dataset.burgers import BurgersDataset
from numpy import allclose


def test_constructor():
    dataset = BurgersDataset()
    assert dataset.name == "Burgers"
    assert len(dataset) == 30
    assert "inputs" in dataset.groups
    assert "outputs" in dataset.groups


def test_constructor_categorize():
    dataset = BurgersDataset(categorize=False)
    assert dataset.name == "Burgers"
    assert len(dataset) == 30
    assert "inputs" not in dataset.groups
    assert "outputs" not in dataset.groups


def test_constructor_parameters():
    nu = 0.03
    dataset = BurgersDataset(n_samples=50, n_x=100, fluid_viscosity=nu)
    assert dataset.name == "Burgers"
    assert len(dataset) == 50
    assert dataset.n_variables == 2
    assert dataset.get_data_by_group(dataset.OUTPUT_GROUP).shape[1] == 100
    assert allclose(dataset.metadata["nu"], nu)
