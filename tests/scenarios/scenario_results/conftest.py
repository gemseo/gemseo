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
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.doe_scenario import DOEScenario


@pytest.fixture
def scenario() -> DOEScenario:
    """A bi-level DOE Scenario.

    For (x, y), we use successively:

    - x = 0
      - y = 0 => z = 0
      - y = 1 => z = 1
    - x = 1
      - y = 0 => z = 1
      - y = 1 => z = 2
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    sub_scenario = DOEScenario(
        [AnalyticDiscipline({"z": "x+y"})],
        "z",
        design_space.filter(["y"], copy=True),
        formulation_name="DisciplinaryOpt",
        name="FooScenario",
    )
    sub_scenario.set_algorithm(algo_name="CustomDOE", samples=array([[0.0], [1.0]]))
    scenario = DOEScenario(
        [sub_scenario], "z", design_space.filter(["x"]), formulation_name="BiLevel"
    )
    scenario.set_algorithm(algo_name="CustomDOE", samples=array([[0.0], [1.0]]))
    return scenario
