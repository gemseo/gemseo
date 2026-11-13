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

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)


def _create_random_space() -> RandomSpace:
    """Create a random space with two independent standard normal variables.

    Returns:
        The random space with the variables `x1` and `x2`.
    """
    random_space = RandomSpace()
    random_space.add_variable("x1", OTNormalDistribution_Settings())
    random_space.add_variable("x2", OTNormalDistribution_Settings())
    return random_space


@pytest.fixture(scope="module")
def random_space() -> RandomSpace:
    """A random space with two independent standard normal variables `x1` and `x2`."""
    return _create_random_space()


@pytest.fixture(scope="module")
def discipline_with_constant_output_and_space() -> tuple[
    AnalyticDiscipline, RandomSpace
]:
    """A discipline with a constant output and its random space.

    The random space is created here rather than taken from the fixture
    `random_space`, which some test modules override.
    """
    discipline = AnalyticDiscipline({"varying": "x1+x2", "constant": "1"})
    return discipline, _create_random_space()
