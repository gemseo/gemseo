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

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline


@pytest.fixture(scope="module")
def discipline_with_constant_output_and_space() -> tuple[
    AnalyticDiscipline, ParameterSpace
]:
    """A discipline with a constant output and its uncertain space."""
    discipline = AnalyticDiscipline({"varying": "x1+x2", "constant": "1"})
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("x1", "OTNormalDistribution")
    uncertain_space.add_random_variable("x2", "OTNormalDistribution")
    return discipline, uncertain_space
