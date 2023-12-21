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
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.scalable.parametric.disciplines.main_discipline import (
    MainDiscipline,
)
from gemseo.problems.scalable.parametric.disciplines.scalable_discipline import (
    ScalableDiscipline,
)
from gemseo.problems.scalable.parametric.scalable_problem import ScalableProblem


@pytest.fixture()
def scalable_problem() -> ScalableProblem:
    """The scalable problem."""
    return ScalableProblem()


def test_main_discipline(scalable_problem):
    """Check the property main_discipline."""
    assert isinstance(scalable_problem.main_discipline, MainDiscipline)


def test_scalable_disciplines(scalable_problem):
    """Check the property scalable_disciplines."""
    for discipline in scalable_problem.scalable_disciplines:
        assert isinstance(discipline, ScalableDiscipline)


@pytest.mark.parametrize("use_optimizer", [False, True])
@pytest.mark.parametrize(
    ("formulation_name", "options"),
    [
        ("MDF", {"inner_mda_name": "MDAGaussSeidel"}),
        ("IDF", {"start_at_equilibrium": True}),
    ],
)
def test_create_scenario(scalable_problem, use_optimizer, formulation_name, options):
    """Check the creation of a scenario."""
    scenario = scalable_problem.create_scenario(
        formulation_name=formulation_name, use_optimizer=use_optimizer, **options
    )

    # Check the type of scenario.
    if use_optimizer:
        assert isinstance(scenario, MDOScenario)
    else:
        assert isinstance(scenario, DOEScenario)

    # Check the disciplines.
    discipline_names = [discipline.name for discipline in scenario.disciplines]
    assert discipline_names == [
        "MainDiscipline",
        "ScalableDiscipline[1]",
        "ScalableDiscipline[2]",
    ]

    # Check the formulation.
    formulation = scenario.formulation
    assert formulation.__class__.__name__ == formulation_name
    if formulation_name == "MDF":
        assert formulation.mda.inner_mdas[0].__class__.__name__ == "MDAGaussSeidel"
    else:
        assert scenario.design_space.get_current_value(["y_2"]) != array([0.5])

    # Check the optimization functions.
    assert scenario.formulation.opt_problem.objective.name == "f"
    constraint_names = [
        constraint.name for constraint in formulation.opt_problem.constraints
    ]
    if formulation_name == "MDF":
        assert constraint_names == ["c_1", "c_2"]
    else:
        assert constraint_names == ["y_1", "y_2", "c_1", "c_2"]

    assert [
        constraint.name for constraint in formulation.opt_problem.get_ineq_constraints()
    ] == ["c_1", "c_2"]

    assert "x_0" in scenario.design_space
    assert "x_1" in scenario.design_space
    assert "x_2" in scenario.design_space


def test_create_quadratic_optimization_problem(scalable_problem):
    """Check create_quadratic_optimization_problem."""
    qp_problem = scalable_problem.create_quadratic_programming_problem()
    assert len(qp_problem.design_space) == 1
    assert_equal(
        qp_problem.design_space["x"],
        DesignSpace.DesignVariable(
            size=3,
            l_b=array([0.0] * 3),
            u_b=array([1.0] * 3),
            value=array([0.5] * 3),
            var_type=array(["float"] * 3),
        ),
    )
    x = array([1.0, 2.0, 3.0])
    assert_almost_equal(qp_problem.objective(x), array(15.784), decimal=3)
    assert_almost_equal(qp_problem.constraints[0](x), array([0.79, 3.097]), decimal=3)

    scenario = scalable_problem.create_scenario()
    scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})

    assert_almost_equal(
        execute_algo(qp_problem, algo_name="NLOPT_SLSQP", max_iter=100).x_opt,
        scenario.optimization_result.x_opt,
    )


def test_create_quadratic_optimization_problem_with_coupling(scalable_problem):
    """Check create_quadratic_optimization_problem with coupling as observable."""
    qp_problem = scalable_problem.create_quadratic_programming_problem(
        add_coupling=True
    )
    execute_algo(qp_problem, algo_name="lhs", n_samples=3, algo_type="doe")
    assert qp_problem.database.get_function_history("coupling").shape == (3, 2)


def test_create_quadratic_optimization_problem_uncertainty_default(scalable_problem):
    """Check quadratic problem with uncertainties."""
    qp_problem = scalable_problem.create_quadratic_programming_problem(
        covariance_matrices=(array([[1]]), array([[1.25]]))
    )
    x = array([1.0, 2.0, 3.0])
    assert_almost_equal(qp_problem.objective(x), array(18.539), decimal=3)
    assert_almost_equal(qp_problem.constraints[0](x), array([2.843, 5.706]), decimal=3)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, [2.843, 5.706]),
        ({"margin_factor": 3.0}, [3.869, 7.01]),
        ({"use_margin": False}, [3.178, 6.132]),
        ({"use_margin": False, "tolerance": 0.1}, [2.105, 4.769]),
    ],
)
def test_robust_quadratic_optimization(scalable_problem, options, expected):
    """Check quadratic optimization problem with uncertainties."""
    qp_problem = scalable_problem.create_quadratic_programming_problem(
        covariance_matrices=(array([[1]]), array([[1.25]])), **options
    )
    x = array([1.0, 2.0, 3.0])
    assert_almost_equal(qp_problem.objective(x), array(18.539), decimal=3)
    assert_almost_equal(qp_problem.constraints[0](x), array(expected), decimal=3)


def test_compute_y(scalable_problem):
    """Check the method compute_y."""
    assert_almost_equal(
        scalable_problem.compute_y(array([1, 2, 3])), array([-1.3081518, -3.6156121])
    )
    assert_almost_equal(
        scalable_problem.compute_y(array([1, 2, 3]), array([0.1, 0.2])),
        array([-1.1971993, -3.3483043]),
    )
