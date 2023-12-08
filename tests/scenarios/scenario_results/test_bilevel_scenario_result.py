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
"""Tests for the class BiLevelScenarioResult."""

from __future__ import annotations

import re

import pytest
from numpy import array

from gemseo.scenarios.scenario_results.bilevel_scenario_result import (
    BiLevelScenarioResult,
)


def test_bilevel_scenario_result_after_execution(scenario):
    """Check BiLevelScenarioResult after execution."""
    scenario.execute()
    # The optimal objective is z*=0 and is achieved in (x*, y*) = (0, 0).

    # We can get x* and z* from the main optimization problem:
    f_opt, x_opt, _, _, _ = scenario.formulation.opt_problem.get_optimum()
    assert x_opt == 0.0
    assert f_opt == 0.0

    # But we cannot get z* from the sub-optimization problem
    # whose optimum corresponds to the last iteration of the main optimization loop;
    # in other words, this is not the z*(x=0) but z*(x=1) with f*(x=1) equal to 1.
    sub_problem = scenario.formulation.disciplines[0].formulation.opt_problem
    f_opt, y_opt, _, _, _ = sub_problem.get_optimum()
    assert y_opt == 0.0
    assert f_opt == 1.0

    # Use the BiLevelScenarioResult to retrieve the optimum (x*, z*).
    scenario_result = BiLevelScenarioResult(scenario)
    optimization_results = scenario_result.optimization_problems_to_results
    optimum_design = scenario_result.design_variable_names_to_values
    assert len(optimization_results) == 2
    label = BiLevelScenarioResult._MAIN_PROBLEM_LABEL
    assert optimization_results[label].x_opt == array([0.0])
    assert optimization_results["sub_0"].x_opt == array([0.0])
    assert optimum_design == {"x": array([0.0]), "y": array([0.0])}

    # We check that the database of the sub-optimization problem
    # corresponds to the last iteration as optimal_design_values handled it.
    f_opt, y_opt, _, _, _ = sub_problem.get_optimum()
    assert y_opt == 0.0
    assert f_opt == 1.0


def test_get_sub_optimization_result(scenario):
    """Check get_sub_optimization_result."""
    scenario.execute()
    scenario_result = BiLevelScenarioResult(scenario)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The index (1) of a sub-optimization result must be between 0 and 0."
        ),
    ):
        scenario_result.get_sub_optimization_result(1)

    assert (
        scenario_result.get_sub_optimization_result(0)
        == scenario_result.optimization_problems_to_results["sub_0"]
    )


def test_get_top_optimization_result(scenario):
    """Check get_top_optimization_result."""
    scenario.execute()
    scenario_result = BiLevelScenarioResult(scenario)
    assert (
        scenario_result.get_top_optimization_result()
        == scenario_result.optimization_result
    )
