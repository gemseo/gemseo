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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.lib_scipy import ScipyOpt
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.optimization.power_2 import Power2

OPT_LIB_NAME = "ScipyOpt"


@pytest.fixture()
def power() -> Power2:
    """The power-2 optimization problem with inequality and equality constraints."""
    return Power2()


@pytest.mark.parametrize(
    ("name", "handle_eq", "handle_ineq"),
    [("L-BFGS-B", False, False), ("SLSQP", True, True)],
)
def test_algorithm_handles_constraints(name, handle_eq, handle_ineq) -> None:
    assert ScipyOpt.ALGORITHM_INFOS[name].handle_equality_constraints is handle_eq
    assert ScipyOpt.ALGORITHM_INFOS[name].handle_inequality_constraints is handle_ineq


def test_is_algorithm_suited() -> None:
    """Check is_algorithm_suited when True."""
    description = OptimizationAlgorithmDescription("foo", "bar")
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    assert BaseOptimizationLibrary.is_algorithm_suited(description, problem)


def test_is_algorithm_suited_design_space() -> None:
    """Check is_algorithm_suited with unhandled empty design space."""
    description = OptimizationAlgorithmDescription("foo", "bar")
    problem = OptimizationProblem(DesignSpace())
    assert not BaseOptimizationLibrary.is_algorithm_suited(description, problem)
    assert (
        BaseOptimizationLibrary._get_unsuitability_reason(description, problem)
        == _UnsuitabilityReason.EMPTY_DESIGN_SPACE
    )


def test_is_algorithm_suited_has_eq_constraints() -> None:
    """Check is_algorithm_suited with unhandled equality constraints."""
    description = OptimizationAlgorithmDescription(
        "foo", "bar", handle_equality_constraints=False
    )
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.has_eq_constraints = lambda: True
    assert not BaseOptimizationLibrary.is_algorithm_suited(description, problem)
    assert (
        BaseOptimizationLibrary._get_unsuitability_reason(description, problem)
        == _UnsuitabilityReason.EQUALITY_CONSTRAINTS
    )


def test_is_algorithm_suited_has_ineq_constraints() -> None:
    """Check is_algorithm_suited with unhandled inequality constraints."""
    description = OptimizationAlgorithmDescription(
        "foo", "bar", handle_inequality_constraints=False
    )
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.has_ineq_constraints = lambda: True
    assert not BaseOptimizationLibrary.is_algorithm_suited(description, problem)
    assert (
        BaseOptimizationLibrary._get_unsuitability_reason(description, problem)
        == _UnsuitabilityReason.INEQUALITY_CONSTRAINTS
    )


def test_is_algorithm_suited_pbm_type() -> None:
    """Check is_algorithm_suited with unhandled problem type."""
    description = OptimizationAlgorithmDescription(
        "foo", "bar", problem_type=OptimizationProblem.ProblemType.LINEAR
    )
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.pb_type = problem.ProblemType.NON_LINEAR
    assert not BaseOptimizationLibrary.is_algorithm_suited(description, problem)
    assert (
        BaseOptimizationLibrary._get_unsuitability_reason(description, problem)
        == _UnsuitabilityReason.NON_LINEAR_PROBLEM
    )


def test_pre_run_fail(power) -> None:
    """Check that pre_run raises an exception if maxiter cannot be determined."""
    slsqp = ScipyOpt("SLSQP")
    with pytest.raises(
        ValueError, match="Could not determine the maximum number of iterations."
    ):
        slsqp._pre_run(power)


def test_check_constraints_handling_fail(power) -> None:
    """Test that check_constraints_handling can raise an exception."""
    lbfgsb = ScipyOpt("L-BFGS-B")
    with pytest.raises(
        ValueError,
        match=(
            "Requested optimization algorithm L-BFGS-B "
            "can not handle equality constraints."
        ),
    ):
        lbfgsb._check_constraints_handling(power)


def test_optimization_algorithm() -> None:
    """Check the default settings of OptimizationAlgorithmDescription."""
    description = OptimizationAlgorithmDescription(
        algorithm_name="bar", internal_algorithm_name="foo"
    )
    assert not description.handle_inequality_constraints
    assert not description.handle_equality_constraints
    assert not description.handle_integer_variables
    assert not description.require_gradient
    assert not description.positive_constraints
    assert not description.handle_multiobjective
    assert description.description == ""
    assert description.website == ""
    assert description.library_name == ""


def test_execute_without_current_value() -> None:
    """Check that the driver can be executed when a current design value is missing."""
    design_space = DesignSpace()
    design_space.add_variable("x")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: (x - 1) ** 2, "obj")
    driver = OptimizationLibraryFactory().create("NLOPT_COBYLA")
    driver.execute(problem, max_iter=1)
    assert design_space["x"].value == 0.0


@pytest.mark.parametrize(
    ("scaling_threshold", "pow2", "ineq1", "ineq2", "eq"),
    [(None, 3, -0.5, -0.5, -0.1), (0.1, 1.0, -1.0, -1.0, -0.1)],
)
def test_function_scaling(power, scaling_threshold, pow2, ineq1, ineq2, eq) -> None:
    """Check the scaling of functions."""
    library = ScipyOpt("SLSQP")
    library.problem = power
    library.problem.preprocess_functions()
    library._pre_run(power, max_iter=2, scaling_threshold=scaling_threshold)
    current_value = power.design_space.get_current_value()
    assert library.problem.objective(current_value) == pow2
    assert library.problem.constraints[0](current_value) == ineq1
    assert library.problem.constraints[1](current_value) == ineq2
    assert library.problem.constraints[2](current_value) == pytest.approx(eq, 0, 1e-16)
