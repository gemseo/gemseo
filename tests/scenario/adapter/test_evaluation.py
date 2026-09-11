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
"""Tests for the adapter of an evaluation scenario."""

from __future__ import annotations

import pytest
from numpy import allclose
from numpy import array

from gemseo.scenario.adapter.evaluation import EvaluationScenarioAdapter
from gemseo.util.testing.helper import assert_exception


@pytest.mark.parametrize("evaluation_scenario", [1, 2], indirect=True)
def test_evaluation_scenario(evaluation_scenario) -> None:
    """Check that an evaluation scenario adapter can wrap an evaluation scenario.

    Args:
        evaluation_scenario: Fixture that returns an evaluation scenario.
    """
    adapter = EvaluationScenarioAdapter(
        evaluation_scenario, input_names=["z"], output_names=["y", "x"]
    )
    # The outputs are those of the last design point evaluated by the DOE,
    # i.e. x=1 here, and not those of an optimum.
    output_data = adapter.execute({"z": array([2.0])})
    assert allclose(output_data["x"], array([1.0]))
    assert allclose(output_data["y"], array([3.0]))


def test_evaluation_scenario_adapter_has_no_multipliers(evaluation_scenario) -> None:
    """Check that an evaluation scenario adapter cannot output multipliers.

    Args:
        evaluation_scenario: Fixture that returns an evaluation scenario.
    """
    with pytest.raises(TypeError, match="output_multipliers"):
        EvaluationScenarioAdapter(
            evaluation_scenario,
            input_names=["z"],
            output_names=["y"],
            output_multipliers=True,
        )


def test_evaluation_scenario_linearization(evaluation_scenario, snapshot) -> None:
    """Check the error raised when linearizing an evaluation scenario adapter.

    Args:
        evaluation_scenario: Fixture that returns an evaluation scenario.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    adapter = EvaluationScenarioAdapter(
        evaluation_scenario, input_names=["z"], output_names=["y"]
    )
    with assert_exception(NotImplementedError, snapshot):
        adapter.linearize({"z": array([2.0])}, compute_all_jacobians=True)
