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
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2


OPT_LIB_NAME = "ScipyOpt"


@pytest.fixture()
def power() -> Power2:
    """The power-2 optimization problem with inequality and equality constraints."""
    return Power2()


@pytest.fixture(scope="module")
def lib() -> OptimizersFactory:
    """The factory of optimizers."""
    factory = OptimizersFactory()
    if factory.is_available(OPT_LIB_NAME):
        return factory.create(OPT_LIB_NAME)

    raise ImportError("SciPy is not available.")


@pytest.mark.parametrize(
    "name,handle_eq,handle_ineq", [("L-BFGS-B", False, False), ("SLSQP", True, True)]
)
def test_algorithm_handles_constraints(lib, name, handle_eq, handle_ineq):
    """Check algorithm_handles_eqcstr() and algorithm_handles_ineqcstr()."""
    assert lib.algorithm_handles_eqcstr(name) is handle_eq
    assert lib.algorithm_handles_ineqcstr(name) is handle_ineq


@pytest.mark.parametrize(
    "name,remove_eq_constraint,suited",
    [("L-BFGS-B", False, False), ("SLSQP", True, True), ("L-BFGS-B", True, False)],
)
def test_is_algorithm_suited(lib, power, name, remove_eq_constraint, suited):
    """Check is_algorithm_suited."""
    if remove_eq_constraint:
        power.constraints = power.constraints[0:1]
    assert lib.is_algorithm_suited(lib.descriptions[name], power) is suited


def test_pre_run_fail(lib, power):
    """Check that pre_run raises an exception if maxiter cannot be determined."""
    with pytest.raises(
        ValueError, match="Could not determine the maximum number of iterations."
    ):
        lib._pre_run(power, "SLSQP")


def test_check_constraints_handling_fail(lib, power):
    """Test that check_constraints_handling can raise an exception."""
    with pytest.raises(
        ValueError,
        match=(
            "Requested optimization algorithm L-BFGS-B "
            "can not handle equality constraints."
        ),
    ):
        lib._check_constraints_handling("L-BFGS-B", power)


def test_algorithm_handles_eqcstr_fail(lib, power):
    """Test that algorithm_handles_eqcstr can raise an exception."""
    with pytest.raises(KeyError, match="Algorithm TOTO not in library ScipyOpt."):
        lib.algorithm_handles_eqcstr("TOTO")


def test_optimization_algorithm():
    """Check the default settings of OptimizationAlgorithmDescription."""
    lib = OptimizationLibrary()
    lib.descriptions["new_algo"] = OptimizationAlgorithmDescription(
        algorithm_name="bar", internal_algorithm_name="foo"
    )
    algo = lib.descriptions["new_algo"]
    assert not lib.algorithm_handles_ineqcstr("new_algo")
    assert not lib.algorithm_handles_eqcstr("new_algo")
    assert not algo.handle_inequality_constraints
    assert not algo.handle_equality_constraints
    assert not algo.handle_integer_variables
    assert not algo.require_gradient
    assert not algo.positive_constraints
    assert not algo.handle_multiobjective
    assert algo.description == ""
    assert algo.website == ""
    assert algo.library_name == ""


def test_execute_without_current_value():
    """Check that the driver can be executed when a current design value is missing."""
    design_space = DesignSpace()
    design_space.add_variable("x")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: (x - 1) ** 2, "obj")
    driver = OptimizersFactory().create("NLOPT_COBYLA")
    driver.execute(problem, "NLOPT_COBYLA", max_iter=1)
    assert design_space["x"].value == 0.0
