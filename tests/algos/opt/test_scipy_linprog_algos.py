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
#        :author: Benoit Pauwels
from __future__ import annotations

import re

import pytest
from numpy import allclose
from numpy import array
from scipy.sparse import csr_array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.lib_scipy_linprog import ScipyLinprog
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock


@pytest.fixture(scope="module")
def library() -> ScipyLinprog:
    """The SciPyLinprog library."""
    return OptimizersFactory().create("ScipyLinprog")


def test_library_name():
    """Tests the library name."""
    assert ScipyLinprog.LIBRARY_NAME == "SciPy"


def test_factory(library):
    """Tests creation of library from factory."""
    assert isinstance(library, ScipyLinprog)


def test_nonlinear_optimization_problem(library):
    """Tests that library does not support non-linear problems."""
    problem = Rosenbrock()
    assert not library.filter_adapted_algorithms(problem)
    for algo_name in library.algorithms:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The algorithm {algo_name} is not adapted to the problem because it "
                "does not handle non-linear problems."
            ),
        ):
            library.execute(problem, algo_name)


def get_opt_problem(sparse_jacobian: bool = False) -> OptimizationProblem:
    """Construct a linear optimization problem.

    Args:
        sparse_jacobian: Whether the objective and constraints Jacobians are sparse.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y", l_b=0.0, u_b=1.0, value=0.5)

    input_names = ["x", "y"]
    array_ = csr_array if sparse_jacobian else array

    problem = OptimizationProblem(design_space, OptimizationProblem.ProblemType.LINEAR)
    problem.objective = MDOLinearFunction(
        array_([1.0, 1.0]),
        "f",
        MDOFunction.FunctionType.OBJ,
        input_names,
        array([-1.0]),
    )
    problem.add_ineq_constraint(
        MDOLinearFunction(array_([1.0, 1.0]), "g", input_names=input_names), 1.0
    )
    problem.add_eq_constraint(
        MDOLinearFunction(array_([-2.0, 1.0]), "h", input_names=input_names)
    )
    return problem


@pytest.mark.parametrize(
    ("minimization", "x_opt", "f_opt"),
    [(True, array([0.0, 0.0]), -1.0), (False, array([1 / 3, 2 / 3]), 0.0)],
)
def test_linprog_algorithms(minimization, x_opt, f_opt, library):
    """Tests algorithms on linear optimization problems."""
    for algo_name in library.algorithms:
        linprog_problem = get_opt_problem()
        if not minimization:
            linprog_problem.change_objective_sign()

        optimization_result = library.execute(linprog_problem, algo_name)

        assert allclose(optimization_result.x_opt, x_opt)
        assert allclose(optimization_result.f_opt, f_opt)


@pytest.mark.parametrize(
    ("minimization", "x_opt", "f_opt"),
    [(True, array([0.0, 0.0]), -1.0), (False, array([1 / 3, 2 / 3]), 0.0)],
)
@pytest.mark.parametrize(
    "algo_name", ["HIGHS", "HIGHS_DUAL_SIMPLEX", "HIGHS_INTERIOR_POINT"]
)
def test_sparse_linprog_algorithms(minimization, x_opt, f_opt, algo_name, library):
    """Tests algorithms on linear optimization problems with sparse Jacobians."""
    linprog_problem = get_opt_problem(sparse_jacobian=True)
    if not minimization:
        linprog_problem.change_objective_sign()

    optimization_result = library.execute(linprog_problem, algo_name)
    assert allclose(optimization_result.x_opt, x_opt)
    assert allclose(optimization_result.f_opt, f_opt)
