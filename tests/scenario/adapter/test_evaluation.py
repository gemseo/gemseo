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

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy.testing import assert_allclose

from gemseo.core.discipline import Discipline
from gemseo.doe import CustomDOE_Settings
from gemseo.scenario import EvaluationScenario
from gemseo.scenario.adapter.evaluation import EvaluationScenarioAdapter
from gemseo.space import DesignSpace
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


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


class DisciplineWithStringInput(Discipline):
    """A discipline computing y=x from x and a string input."""

    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__(name="d")
        self.io.input_grammar.update_from_names(["x"])
        self.io.input_grammar.update_from_types({"label": str})
        self.io.output_grammar.update_from_names(["y"])
        self.io.input_grammar.defaults["label"] = "foo"

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"y": input_data["x"]}


def test_string_input_data_history() -> None:
    """Check that the input values of a string input are recorded."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = EvaluationScenario([DisciplineWithStringInput()], design_space)
    scenario.add_observable("y")
    scenario.set_algorithm(CustomDOE_Settings(samples=array([[0.0], [1.0]])))
    adapter = EvaluationScenarioAdapter(
        scenario, input_names=["label"], output_names=["y"], keep_databases=True
    )

    adapter.execute({"label": "bar"})

    assert adapter.input_data_history == [{"label": "bar"}]


def test_database_file_mtimes(tmp_wd) -> None:
    """Check that the modification times of the exported files are recorded."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = EvaluationScenario([DisciplineWithStringInput()], design_space)
    scenario.add_observable("y")
    scenario.set_algorithm(CustomDOE_Settings(samples=array([[0.0], [1.0]])))
    adapter = EvaluationScenarioAdapter(
        scenario, input_names=["label"], output_names=["y"], save_databases=True
    )

    adapter.execute({"label": "bar"})
    adapter.execute({"label": "baz"})

    assert len(adapter.database_file_paths) == 2
    assert adapter.database_file_mtimes == [
        path.stat().st_mtime_ns for path in adapter.database_file_paths
    ]


class DisciplineWithMaterial(Discipline):
    """A discipline computing y=x*density from a material label and x."""

    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__(name="d")
        self.io.input_grammar.update_from_names(["x"])
        self.io.input_grammar.update_from_types({"material": str})
        self.io.output_grammar.update_from_names(["y"])
        self.io.input_grammar.defaults["material"] = "steel"
        self.io.input_grammar.defaults["x"] = array([1.0])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        density = {"steel": 7.8, "aluminium": 2.7}[input_data["material"]]
        return {"y": density * input_data["x"]}


def test_set_x0_and_bounds_with_a_categorical_variable() -> None:
    """Check that the adapter handles a categorical design variable."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("material", ("steel", "aluminium"))
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = EvaluationScenario([DisciplineWithMaterial()], design_space)
    scenario.add_observable("y")
    scenario.set_algorithm(CustomDOE_Settings(samples=array([[0.0, 0.5], [1.0, 0.5]])))
    adapter = EvaluationScenarioAdapter(
        scenario,
        input_names=["x", "material"],
        output_names=["y"],
        set_x0_before_exec=True,
        set_bounds_before_exec=True,
    )

    adapter.execute({
        "x": array([0.25]),
        "x_lower_bnd": array([0.0]),
        "x_upper_bnd": array([0.5]),
        "material": "aluminium",
    })

    assert_allclose(design_space.get_lower_bounds(["x"]), array([0.0]))
    assert_allclose(design_space.get_upper_bounds(["x"]), array([0.5]))


def test_categorical_design_variable_as_output() -> None:
    """Check that the adapter outputs the label of a categorical design variable."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("material", ("steel", "aluminium"))
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = EvaluationScenario([DisciplineWithMaterial()], design_space)
    scenario.add_observable("y")
    scenario.set_algorithm(CustomDOE_Settings(samples=array([[1.0, 0.5]])))
    adapter = EvaluationScenarioAdapter(scenario, (), ["y", "material"])

    adapter.execute()

    assert adapter.io.output_data["material"] == "aluminium"
