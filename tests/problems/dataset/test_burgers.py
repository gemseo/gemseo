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

from numpy import allclose

from gemseo.problems.dataset.burgers import create_burgers_dataset


def test_constructor():
    dataset = create_burgers_dataset()
    assert dataset.name == "Burgers"
    assert len(dataset) == 30
    assert "inputs" in dataset.group_names
    assert "outputs" in dataset.group_names


def test_constructor_categorize():
    dataset = create_burgers_dataset(categorize=False)
    assert dataset.name == "Burgers"
    assert len(dataset) == 30
    assert "inputs" not in dataset.group_names
    assert "outputs" not in dataset.group_names


def test_constructor_parameters():
    nu = 0.03
    dataset = create_burgers_dataset(n_samples=50, n_x=100, fluid_viscosity=nu)
    assert dataset.name == "Burgers"
    assert len(dataset) == 50
    assert len(dataset.variable_names) == 2
    assert dataset.get_view(group_names=dataset.OUTPUT_GROUP).shape == (50, 100)
    assert allclose(dataset.misc["nu"], nu)
