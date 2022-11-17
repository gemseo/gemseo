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

from gemseo.problems.dataset.rosenbrock import RosenbrockDataset


def test_constructor():
    dataset = RosenbrockDataset()
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" not in dataset.groups
    assert "design_parameters" in dataset.groups


def test_constructor_categorize():
    dataset = RosenbrockDataset(categorize=False)
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" not in dataset.groups
    assert "design_parameters" not in dataset.groups


def test_constructor_optnaming():
    dataset = RosenbrockDataset(opt_naming=False)
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" in dataset.groups
    assert "design_parameters" not in dataset.groups
