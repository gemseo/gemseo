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

from logging import WARNING

import pytest
from numpy import allclose
from numpy import array

from gemseo import create_discipline
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.linear_solvers.scipy_linalg.settings.lgmres import LGMRES_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.bilevel_settings import BiLevel_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.mdo import MDOScenario
from gemseo.scenarios.scenario_results.bilevel_scenario_result import (
    BiLevelScenarioResult,
)
from gemseo.utils.testing.helpers import assert_exception


def test_bilevel_scenario_result_after_execution(scenario) -> None:
    """Check BiLevelScenarioResult after execution."""
    scenario.execute()
    # The optimal objective is z*=0 and is achieved in (x*, y*) = (0, 0).

    # We can get x* and z* from the main optimization problem:
    f_opt, x_opt, _, _, _ = scenario.formulation.problem.optimum
    assert x_opt == 0.0
    assert f_opt == 0.0

    # But we cannot get z* from the sub-optimization problem
    # whose optimum corresponds to the last iteration of the main optimization loop;
    # in other words, this is not the z*(x=0) but z*(x=1) with f*(x=1) equal to 1.
    sub_problem = scenario.formulation.disciplines[0].formulation.problem
    f_opt, y_opt, _, _, _ = sub_problem.optimum
    assert y_opt == 0.0
    assert f_opt == 1.0

    # Use the BiLevelScenarioResult to retrieve the optimum (x*, z*).
    scenario_result = BiLevelScenarioResult(scenario)
    optimization_results = scenario_result.optimization_problem_to_result
    optimum_design = scenario_result.design_variable_name_to_value
    assert len(optimization_results) == 2
    label = BiLevelScenarioResult._MAIN_PROBLEM_LABEL
    assert optimization_results[label].x_opt == array([0.0])
    assert optimization_results["sub_0"].x_opt == array([0.0])
    assert optimum_design == {"x": array([0.0]), "y": array([0.0])}

    # We check that the database of the sub-optimization problem
    # corresponds to the last iteration as optimal_design_values handled it.
    f_opt, y_opt, _, _, _ = sub_problem.optimum
    assert y_opt == 0.0
    assert f_opt == 1.0


def test_get_sub_optimization_result(scenario, snapshot) -> None:
    """Check get_sub_optimization_result."""
    scenario.execute()
    scenario_result = BiLevelScenarioResult(scenario)
    with assert_exception(ValueError, snapshot):
        scenario_result.get_sub_optimization_result(1)

    assert (
        scenario_result.get_sub_optimization_result(0)
        == scenario_result.optimization_problem_to_result["sub_0"]
    )


def test_get_top_optimization_result(scenario) -> None:
    """Check get_top_optimization_result."""
    scenario.execute()
    scenario_result = BiLevelScenarioResult(scenario)
    assert (
        scenario_result.get_top_optimization_result()
        == scenario_result.optimization_result
    )


@pytest.mark.slow
def test_get_results():
    """Test `get_result()` method."""
    propulsion, aerodynamics, mission, structure = create_discipline([
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ])
    design_space = SobieskiDesignSpace()
    slsqp_settings = SLSQP_Settings(
        max_iter=30,
        xtol_rel=1e-7,
        xtol_abs=1e-7,
        ftol_rel=1e-7,
        ftol_abs=1e-7,
        ineq_tolerance=1e-4,
    )
    sc_prop = MDOScenario(
        (propulsion,), design_space.filter("x_3", copy=True), name="PropulsionScenario"
    )
    sc_prop.add_objective("y_34")
    sc_prop.set_algorithm(slsqp_settings)
    sc_prop.add_constraint("g_3", constraint_type="ineq")

    sc_aero = MDOScenario(
        (aerodynamics,),
        design_space.filter("x_2", copy=True),
        name="AerodynamicsScenario",
    )
    sc_aero.add_objective("y_24", minimize=False)
    sc_aero.set_algorithm(slsqp_settings)
    sc_aero.add_constraint("g_2", constraint_type="ineq")

    sc_str = MDOScenario(
        (structure,),
        design_space.filter("x_1", copy=True),
        name="StructureScenario",
    )
    sc_str.add_objective("y_11", minimize=False)
    sc_str.add_constraint("g_1", constraint_type="ineq")
    sc_str.set_algorithm(slsqp_settings)

    system_scenario = MDOScenario(
        (sc_prop, sc_aero, sc_str, mission),
        design_space.filter("x_shared", copy=True),
        formulation_settings=BiLevel_Settings(
            apply_constraints_to_sub_scenarios=False,
            parallel_scenarios=False,
            multithread_scenarios=True,
            main_mda_settings=MDAGaussSeidel_Settings(
                tolerance=1e-14,
                max_mda_iter=50,
                warm_start=True,
                linear_solver_settings=LGMRES_Settings(rtol=1e-14),
            ),
            sub_scenarios_log_level=WARNING,
        ),
    )
    system_scenario.add_objective("y_4", minimize=False)
    system_scenario.formulation.problem.objective *= 1e-4
    system_scenario.add_constraint(["g_1", "g_2", "g_3"], constraint_type="ineq")

    system_scenario.execute(
        NLOPT_COBYLA_Settings(
            max_iter=140,
            xtol_rel=1e-7,
            xtol_abs=1e-7,
            ftol_rel=1e-7,
            ftol_abs=1e-7,
            ineq_tolerance=1e-4,
        )
    )

    bilevel_result = system_scenario.get_result()
    assert isinstance(bilevel_result, BiLevelScenarioResult)

    for i in range(3):
        sub = bilevel_result.get_sub_optimization_result(i)
        assert sub is not None

    optimization_result = bilevel_result.get_top_optimization_result()
    assert isinstance(optimization_result, OptimizationResult)
    assert allclose(optimization_result.f_opt, array([-3963.4]) * 1e-4, rtol=1e-4)


def test_no_databases():
    """Test that it works properly when keep_opt_history is False."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)

    sub_scenario = MDOScenario(
        [AnalyticDiscipline({"z": "x+y"})],
        design_space.filter(["y"], copy=True),
        name="FooScenario",
    )
    sub_scenario.add_objective("z")
    sub_scenario.set_algorithm(CustomDOE_Settings(samples=array([[0.0], [1.0]])))

    scenario = MDOScenario(
        [sub_scenario],
        design_space.filter(["x"]),
        formulation_settings=BiLevel_Settings(keep_opt_history=False),
    )
    scenario.add_objective("z")
    scenario.execute(CustomDOE_Settings(samples=array([[0.0], [1.0]])))

    result = BiLevelScenarioResult(scenario)
    assert len(result.optimization_problem_to_result) == 1
    assert result.get_sub_optimization_result(0) is None
