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
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline


@pytest.fixture()
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
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y", l_b=0.0, u_b=1.0, value=0.5)
    sub_scenario = DOEScenario(
        [AnalyticDiscipline({"z": "x+y"})],
        "DisciplinaryOpt",
        "z",
        design_space.filter(["y"], copy=True),
        name="FooScenario",
    )
    input_data = {
        "algo": "CustomDOE",
        "algo_options": {"samples": array([[0.0], [1.0]])},
    }
    sub_scenario.default_inputs = input_data
    scenario = DOEScenario([sub_scenario], "BiLevel", "z", design_space.filter(["x"]))
    scenario.default_inputs = input_data
    return scenario
