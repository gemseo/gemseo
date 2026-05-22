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
"""Tests for the class ScenarioResult."""

from __future__ import annotations

from pathlib import Path

from numpy import array

from gemseo.post import BasicHistory_Settings
from gemseo.post.basic_history import BasicHistory
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult
from gemseo.utils.testing.helpers import assert_exception


def test_scenario_result_before_execution(scenario, snapshot) -> None:
    """Check ScenarioResult before execution."""
    with assert_exception(ValueError, snapshot):
        ScenarioResult(scenario)


def test_scenario_result(scenario) -> None:
    """Check ScenarioResult."""
    scenario.execute()
    scenario_result = ScenarioResult(scenario)

    optimization_results = scenario_result.optimization_problem_to_result
    optimum_design = scenario_result.design_variable_name_to_value
    first_optimization_result = optimization_results[
        scenario_result._MAIN_PROBLEM_LABEL
    ]
    assert len(optimization_results) == 1
    assert scenario_result.optimization_result is first_optimization_result
    assert first_optimization_result.x_opt == array([0.0])
    assert optimum_design == {"x": array([0.0])}


def test_hdf_file() -> None:
    """Check ScenarioResult with an HDF file instead of a Scenario."""
    scenario_result = ScenarioResult(Path(__file__).parent / "scenario.hdf5")
    assert scenario_result.design_variable_name_to_value == {"x": array([0.0])}


def test_plot(scenario) -> None:
    """Check ScenarioResult.plot."""
    scenario.execute()
    assert isinstance(
        ScenarioResult(scenario).plot(
            BasicHistory_Settings(variable_names=["x"], save=False, show=False)
        ),
        BasicHistory,
    )
