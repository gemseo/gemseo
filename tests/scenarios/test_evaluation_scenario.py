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
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.chain import MDAChain
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.scenarios.evaluation import EvaluationScenario


@pytest.fixture
def discipline_a() -> AnalyticDiscipline:
    """Discipline A."""
    return AnalyticDiscipline({"y": "x+1"}, name="A")


@pytest.fixture
def discipline_b() -> AnalyticDiscipline:
    """Discipline B."""
    return AnalyticDiscipline({"z": "y*2"}, name="B")


@pytest.fixture
def discipline_c() -> AnalyticDiscipline:
    """Discipline C."""
    return AnalyticDiscipline({"w": "x*2"}, name="C")


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    return space


def test_two_coupled_disciplines_default(discipline_a, discipline_b, design_space):
    """Check the evaluation scenario with two disciplines coupled using default MDA."""
    scenario = EvaluationScenario([discipline_b, discipline_a], design_space)
    assert isinstance(scenario.formulation, MDF)
    assert isinstance(scenario.formulation.mda, MDAChain)
    scenario.add_observable("y")
    scenario.add_observable("z")
    result = scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    assert result is None
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="y"), array([[3.0], [4.0]]))
    assert_equal(dataset.get_view(variable_names="z"), array([[6.0], [8.0]]))
    problem = scenario.formulation.problem
    assert isinstance(problem, EvaluationProblem)
    assert problem.function_names == ["y", "z"]
    assert len(problem.database) == 2


def test_two_coupled_disciplines(discipline_a, discipline_b, design_space):
    """Check the evaluation scenario with two disciplines coupled using Jacobi."""
    scenario = EvaluationScenario(
        [discipline_b, discipline_a],
        design_space,
        formulation_settings=MDF_Settings(main_mda_settings=MDAJacobi_Settings()),
    )
    assert isinstance(scenario.formulation, MDF)
    assert isinstance(scenario.formulation.mda, MDAJacobi)
    scenario.add_observable("y")
    scenario.add_observable("z")
    scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="y"), array([[3.0], [4.0]]))
    assert_equal(dataset.get_view(variable_names="z"), array([[6.0], [8.0]]))


def test_observable_name(discipline_a, design_space):
    """Check the use of a custom observable name."""
    scenario = EvaluationScenario([discipline_a], design_space)
    scenario.add_observable("y", observable_name="foo")
    scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="foo"), array([[3.0], [4.0]]))


@pytest.mark.parametrize("add_y", [False, True])
def test_observe_all_outputs(
    discipline_a, discipline_b, discipline_c, design_space, add_y
):
    """Test the observe_all_outputs method."""
    scenario = EvaluationScenario(
        [discipline_b, discipline_a, discipline_c],
        design_space,
    )
    if add_y:
        scenario.add_observable("y")
        assert scenario.formulation.problem.function_names == ["y"]

    scenario.observe_all_outputs()
    assert scenario.formulation.problem.function_names == (
        ["y", "w", "z"]
        if add_y
        else [
            "w",
            "y",
            "z",
        ]
    )


def test_raise_exception_when_missing_algo_settings(discipline_a, design_space):
    """Check that a ValueError is raised when the algo_settings are missing."""
    scenario = EvaluationScenario([discipline_a], design_space)
    msg = (
        "Algorithm settings are necessary for executing a scenario. "
        "Pass the settings in the execute method "
        "or use the set_algorithm method."
    )
    with pytest.raises(ValueError, match=msg):
        scenario.execute()
