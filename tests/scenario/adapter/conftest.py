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
"""Fixtures shared by the tests of the scenario adapters."""

from __future__ import annotations

import pytest
from numpy import array

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.space.design import DesignSpace


@pytest.fixture
def evaluation_scenario(request) -> EvaluationScenario:
    """An evaluation scenario sampling y=x**2+z over x, with z as default input.

    The number of processes used by the DOE can be set by indirect parametrization;
    it defaults to 1.

    Args:
        request: The pytest request, whose optional `param` indirectly sets
            the number of processes used by the DOE.

    Returns:
        The evaluation scenario.
    """
    discipline = AnalyticDiscipline({"y": "x**2 + z"}, name="d")
    discipline.io.input_grammar.defaults["z"] = array([1.0])
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = EvaluationScenario([discipline], design_space)
    scenario.add_observable("y")
    scenario.set_algorithm(
        PYDOE_FULLFACT_Settings(n_samples=3, n_processes=getattr(request, "param", 1))
    )
    return scenario
