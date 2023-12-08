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

import re
from pathlib import Path

import pytest
from numpy import array

from gemseo.post.basic_history import BasicHistory
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult


def test_scenario_result_before_execution(scenario):
    """Check ScenarioResult before execution."""
    with pytest.raises(
        ValueError,
        match=re.escape("A ScenarioResult requires a scenario that has been executed."),
    ):
        ScenarioResult(scenario)


def test_scenario_result(scenario):
    """Check ScenarioResult."""
    scenario.execute()
    scenario_result = ScenarioResult(scenario)

    optimization_results = scenario_result.optimization_problems_to_results
    optimum_design = scenario_result.design_variable_names_to_values
    first_optimization_result = optimization_results[
        scenario_result._MAIN_PROBLEM_LABEL
    ]
    assert len(optimization_results) == 1
    assert scenario_result.optimization_result is first_optimization_result
    assert first_optimization_result.x_opt == array([0.0])
    assert optimum_design == {"x": array([0.0])}


def test_hdf_file():
    """Check ScenarioResult with an HDF file instead of a Scenario."""
    scenario_result = ScenarioResult(Path(__file__).parent / "scenario.hdf5")
    assert scenario_result.design_variable_names_to_values == {"x": array([0.0])}


def test_plot(scenario):
    """Check ScenarioResult.plot."""
    scenario.execute()
    assert isinstance(
        ScenarioResult(scenario).plot(
            "BasicHistory", variable_names=["x"], save=False, show=False
        ),
        BasicHistory,
    )
