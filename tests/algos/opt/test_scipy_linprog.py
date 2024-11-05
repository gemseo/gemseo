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
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.scipy_linprog.scipy_linprog import ScipyLinprog
from gemseo.algos.opt.scipy_linprog.settings.highs_dual_simplex import (
    DUAL_SIMPLEX_Settings,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock


@pytest.fixture(scope="module")
def library_cls() -> type[ScipyLinprog]:
    """The SciPyLinprog library."""
    return OptimizationLibraryFactory().get_class("ScipyLinprog")


def test_factory(library_cls) -> None:
    """Tests creation of library from factory."""
    assert library_cls == ScipyLinprog


def test_nonlinear_optimization_problem(library_cls) -> None:
    """Tests that library does not support non-linear problems."""
    problem = Rosenbrock()
    assert not library_cls.filter_adapted_algorithms(problem)
    for algo_name in library_cls.ALGORITHM_INFOS:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The algorithm {algo_name} is not adapted to the problem because it "
                "does not handle non-linear problems."
            ),
        ):
            library_cls(algo_name).execute(problem)


def get_opt_problem(sparse_jacobian: bool = False) -> OptimizationProblem:
    """Construct a linear optimization problem.

    Args:
        sparse_jacobian: Whether the objective and constraints Jacobians are sparse.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)

    input_names = ["x", "y"]
    array_ = csr_array if sparse_jacobian else array

    problem = OptimizationProblem(design_space)
    problem.objective = MDOLinearFunction(
        array_([[1.0, 1.0]]),
        "f",
        MDOFunction.FunctionType.OBJ,
        input_names,
        array([-1.0]),
    )
    problem.add_constraint(
        MDOLinearFunction(
            array_([[1.0, 1.0]]),
            "g",
            input_names=input_names,
            f_type=MDOLinearFunction.ConstraintType.INEQ,
        ),
        value=1.0,
    )
    problem.add_constraint(
        MDOLinearFunction(
            array_([[-2.0, 1.0]]),
            "h",
            input_names=input_names,
            f_type=MDOLinearFunction.ConstraintType.EQ,
        )
    )
    return problem


@pytest.mark.parametrize(
    ("minimization", "x_opt", "f_opt"),
    [(True, array([0.0, 0.0]), -1.0), (False, array([1 / 3, 2 / 3]), 0.0)],
)
def test_linprog_algorithms(minimization, x_opt, f_opt, library_cls) -> None:
    """Tests algorithms on linear optimization problems."""
    for algo_name in library_cls.ALGORITHM_INFOS:
        linprog_problem = get_opt_problem()
        if not minimization:
            linprog_problem.minimize_objective = False

        optimization_result = library_cls(algo_name).execute(linprog_problem)

        assert allclose(optimization_result.x_opt, x_opt)
        assert allclose(optimization_result.f_opt, f_opt)


@pytest.mark.parametrize(
    ("minimization", "x_opt", "f_opt"),
    [(True, array([0.0, 0.0]), -1.0), (False, array([1 / 3, 2 / 3]), 0.0)],
)
@pytest.mark.parametrize("algo_name", ScipyLinprog.ALGORITHM_INFOS)
def test_sparse_linprog_algorithms(
    minimization, x_opt, f_opt, algo_name, library_cls
) -> None:
    """Tests algorithms on linear optimization problems with sparse Jacobians."""
    linprog_problem = get_opt_problem(sparse_jacobian=True)
    if not minimization:
        linprog_problem.minimize_objective = False

    optimization_result = library_cls(algo_name).execute(linprog_problem)
    assert allclose(optimization_result.x_opt, x_opt)
    assert allclose(optimization_result.f_opt, f_opt)


@pytest.mark.parametrize("scaling_threshold", [None, 0.1])
def test_autoscale_setting(scaling_threshold):
    """Check that the ``scale_threshold`` setting enables the ``autoscale`` setting."""
    settings = DUAL_SIMPLEX_Settings(scaling_threshold=scaling_threshold)
    assert settings.autoscale if scaling_threshold else not settings.autoscale
