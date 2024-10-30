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
#    INITIAL AUTHORS - API and implementation and/or documentation
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the NLopt library wrapper."""

from __future__ import annotations

from unittest import TestCase

import pytest
from numpy import array
from numpy import inf
from scipy.optimize.optimize import rosen
from scipy.optimize.optimize import rosen_der

from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary as OptLib
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.nlopt.nlopt import Nlopt
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.opt_lib_test_base import OptLibraryTestBase

from .problems.x2 import X2


class TestNLOPT(TestCase):
    OPT_LIB_NAME = "Nlopt"

    def _test_init(self) -> None:
        """Tests the initialization of the problem."""
        factory = OptimizationLibraryFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create(self.OPT_LIB_NAME)

    def test_failed(self) -> None:
        """"""
        algo_name = "NLOPT_SLSQP"
        self.assertRaises(
            Exception,
            OptLibraryTestBase.generate_error_test,
            "NloptAlgorithms",
            algo_name=algo_name,
            max_iter=10,
        )

    def test_roundoff(self) -> None:
        problem = Power2()

        def obj_grad(x):
            return array([-1.0, 1.0, 1.0e300])

        problem.objective.jac = obj_grad
        problem.constraints.clear()
        opt_library = OptimizationLibraryFactory().create("NLOPT_BFGS")
        opt_library.execute(problem, max_iter=10)

    def test_normalization(self) -> None:
        """Runs a problem with one variable to be normalized and three not to be."""
        design_space = DesignSpace()
        design_space.add_variable(
            "x1", 1, DesignSpace.DesignVariableType.FLOAT, -1.0, 1.0, 0.0
        )
        design_space.add_variable(
            "x2", 1, DesignSpace.DesignVariableType.FLOAT, -inf, 1.0, 0.0
        )
        design_space.add_variable(
            "x3", 1, DesignSpace.DesignVariableType.FLOAT, -1.0, inf, 0.0
        )
        design_space.add_variable(
            "x4", 1, DesignSpace.DesignVariableType.FLOAT, -inf, inf, 0.0
        )
        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
        OptimizationLibraryFactory().execute(problem, algo_name="NLOPT_COBYLA")

    def test_tolerance_activation(self) -> None:
        def run_pb(algo_options):
            design_space = DesignSpace()
            design_space.add_variable(
                "x1", 2, DesignSpace.DesignVariableType.FLOAT, -1.0, 1.0, 0.0
            )
            problem = OptimizationProblem(design_space)
            problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
            res = OptimizationLibraryFactory().execute(
                problem, algo_name="NLOPT_SLSQP", **algo_options
            )
            return res, problem

        for tol_name in (
            OptLib._F_TOL_ABS,
            OptLib._F_TOL_REL,
            OptLib._X_TOL_ABS,
            OptLib._X_TOL_REL,
        ):
            res, pb = run_pb({tol_name: 1e10})
            assert tol_name in res.message
            # Check that the criterion is activated asap
            assert len(pb.database) == 3
        for tol_name in (
            OptLib._KKT_TOL_ABS,
            OptLib._KKT_TOL_REL,
        ):
            res, pb = run_pb({tol_name: 1e10})
            assert tol_name in res.message
            # Check that the KKT criterion is activated asap
            assert len(pb.database) == 1


def test_cast_to_float() -> None:
    """Test that the NLopt library handles functions that return an `ndarray`."""
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(
        lambda x: x, "my_function", jac=lambda x: array([[1.0]])
    )
    res = OptimizationLibraryFactory().execute(
        problem, algo_name="NLOPT_SLSQP", max_iter=100
    )
    assert res.x_opt == array([0.0])
    assert res.f_opt == 0.0


def get_options(algo_name):
    """

    :param algo_name:

    """

    if algo_name == "NLOPT_SLSQP":
        return {
            Nlopt._X_TOL_REL: 1e-5,
            Nlopt._F_TOL_REL: 1e-5,
            "max_iter": 100,
            Nlopt._KKT_TOL_REL: 1e-5,
            Nlopt._KKT_TOL_ABS: 1e-5,
        }
    if algo_name == "NLOPT_MMA":
        return {
            "max_iter": 2700,
            Nlopt._X_TOL_REL: 1e-8,
            Nlopt._F_TOL_REL: 1e-8,
            Nlopt._INNER_MAXEVAL: 10,
            Nlopt._KKT_TOL_REL: 1e-8,
            Nlopt._KKT_TOL_ABS: 1e-8,
        }
    if algo_name == "NLOPT_COBYLA":
        return {"max_iter": 10000, Nlopt._X_TOL_REL: 1e-8, Nlopt._F_TOL_REL: 1e-8}
    if algo_name == "NLOPT_BOBYQA":
        return {"max_iter": 2200}
    return {
        "max_iter": 100,
        Nlopt._EQ_TOLERANCE: 1e-10,
        Nlopt._INEQ_TOLERANCE: 1e-10,
        Nlopt._STOPVAL: 0.0,
    }


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("Nlopt", get_options):
    setattr(TestNLOPT, test_method.__name__, test_method)


@pytest.fixture
def x2_problem() -> X2:
    """Instantiate a :class:`.X2` test problem.

    Returns:
         A X_2 problem.
    """
    x2_problem = X2()
    x2_problem.preprocess_functions()
    return x2_problem


@pytest.mark.parametrize("algo_name", ["NLOPT_COBYLA", "NLOPT_BOBYQA"])
def test_no_stop_during_doe_phase(x2_problem: X2, algo_name: str) -> None:
    """Test that COBYLA and BOBYQA does not trigger a stop criterion during the DoE
    phase.

    In this test, stop_crit_n_x is automatically set by :method:`.Nlopt._pre_run`.
    Also, n_stop_crit_x is automatically set,
    with a large enough value which lead to a algorithm stop
    due to the max iteration stop criterion.

    Args:
        x2_problem: An instanciated :class:`.X_2` optimization problem.
        algo_name: The optimization algorithm used.
    """
    res = execute_algo(x2_problem, algo_name=algo_name, max_iter=10)
    assert res.n_obj_call == 12


def test_cobyla_stopped_due_to_small_crit_n_x(x2_problem: X2) -> None:
    """Test that COBYLA does not trigger a stop criterion during the doe phase.

    In this test, stop_crit_n_x is set by the user. An insufficient value is given,
    which lead to a premature stop of the algorithm.

    Args:
        x2_problem: An instantiated :class:`.X_2` optimization problem.
    """
    res = execute_algo(
        x2_problem, algo_name="NLOPT_COBYLA", max_iter=100, stop_crit_n_x=3
    )
    assert res.n_obj_call == 5


def test_bobyqa_stopped_due_to_small_crit_n_x(x2_problem: X2) -> None:
    """Test that BOBYQA does not trigger a stop criterion during its DoE phase.

    In this test, stop_crit_n_x is set by the user. An insufficient value is given,
    which lead to a premature stop of the algorithm.

    Args:
        x2_problem: An instantiated :class:`.X_2` optimization problem.
    """
    res = execute_algo(
        x2_problem, algo_name="NLOPT_BOBYQA", max_iter=100, stop_crit_n_x=3
    )
    assert res.n_obj_call == 6
