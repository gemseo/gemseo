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

from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary
from gemseo.utils.seeder import SEED
from gemseo.utils.seeder import Seeder


@pytest.mark.parametrize(
    ("kwargs", "initial_seed"), [({}, SEED), ({"default_seed": 12}, 12)]
)
def test_seeder(kwargs, initial_seed):
    """Check Seeder."""
    seeder = Seeder(**kwargs)
    assert seeder.default_seed == initial_seed
    assert seeder.get_seed() == seeder.default_seed == initial_seed + 1
    assert seeder.get_seed() == seeder.default_seed == initial_seed + 2
    assert seeder.get_seed(134) == 134
    assert seeder.default_seed == initial_seed + 3
    assert seeder.get_seed() == seeder.default_seed == initial_seed + 4


def test_setter():
    """Check the default_seed setter."""
    default_seed = 123
    seeder = Seeder()
    seeder.default_seed = default_seed
    assert seeder.default_seed == default_seed
    doe_library = PyDOELibrary("PYDOE_LHS")
    doe_library.seed = default_seed
    assert doe_library._seeder.default_seed == default_seed
