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

from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset


def test_constructor():
    dataset = create_rosenbrock_dataset()
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" not in dataset.group_names
    assert "designs" in dataset.group_names


def test_constructor_categorize():
    dataset = create_rosenbrock_dataset(categorize=False)
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" not in dataset.group_names
    assert "designs" not in dataset.group_names


def test_constructor_optnaming():
    dataset = create_rosenbrock_dataset(opt_naming=False)
    assert dataset.name == "Rosenbrock"
    assert len(dataset) == 100
    assert "inputs" in dataset.group_names
    assert "designs" not in dataset.group_names
