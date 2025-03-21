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
from __future__ import annotations

import re
from unittest import TestCase
from warnings import warn

import pytest
from numpy import allclose
from numpy import array
from numpy import inf
from pydantic import ValidationError
from scipy.optimize.optimize import rosen
from scipy.optimize.optimize import rosen_der
from scipy.sparse import csr_array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary as OptLib
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.scipy_local.scipy_local import ScipyOpt
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.compatibility.scipy import SCIPY_GREATER_THAN_1_14
from gemseo.utils.testing.opt_lib_test_base import OptLibraryTestBase


class TestScipy(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyOpt"

    def test_display(self) -> None:
        """"""
        algo_name = "SLSQP"
        OptLibraryTestBase.generate_one_test(algo_name, max_iter=10, disp=True)

    def test_handles_cstr(self) -> None:
        """"""
        algo_name = "TNC"
        self.assertRaises(
            Exception,
            OptLibraryTestBase.generate_one_test,
            algo_name,
            max_iter=10,
        )

    def test_algorithm_suited(self) -> None:
        """"""
        algo_name = "SLSQP"
        opt_library, problem = OptLibraryTestBase.generate_one_test(
            algo_name, max_iter=10
        )

        assert not opt_library.is_algorithm_suited(
            opt_library.ALGORITHM_INFOS["TNC"], problem
        )

        problem._OptimizationProblem__has_linear_functions = False
        opt_library.ALGORITHM_INFOS["SLSQP"].for_linear_problems = True
        assert not opt_library.is_algorithm_suited(
            opt_library.ALGORITHM_INFOS["SLSQP"], problem
        )
        opt_library.ALGORITHM_INFOS["SLSQP"].for_linear_problems = False

    def test_positive_constraints(self) -> None:
        """"""
        algo_name = "SLSQP"
        opt_library, _ = OptLibraryTestBase.generate_one_test(algo_name, max_iter=10)
        assert opt_library.ALGORITHM_INFOS[algo_name].positive_constraints
        assert not opt_library.ALGORITHM_INFOS["TNC"].positive_constraints

    def test_fail_opt(self) -> None:
        """"""
        algo_name = "SLSQP"
        problem = Rosenbrock()

        def i_fail(x):
            if rosen(x) < 1e-3:
                raise ValueError(x)
            return rosen(x)

        problem.objective = MDOFunction(i_fail, "rosen")
        self.assertRaises(
            Exception, OptimizationLibraryFactory().execute, problem, algo_name
        )

    def test_tnc_options(self) -> None:
        """"""
        algo_name = "TNC"
        OptLibraryTestBase.generate_one_test_unconstrained(
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter=100,
            disp=True,
            maxCGit=178,
            gtol=1e-8,
            eta=-1.0,
            ftol_rel=1e-10,
            xtol_rel=1e-10,
            minfev=4,
        )

    def test_lbfgsb_options(self) -> None:
        """"""
        algo_name = "L-BFGS-B"
        OptLibraryTestBase.generate_one_test_unconstrained(
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter=100,
            disp=True,
            maxcor=12,
            gtol=1e-8,
        )
        self.assertRaises(
            ValidationError,
            OptLibraryTestBase.generate_one_test_unconstrained,
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter="100",
            disp=True,
            maxcor=12,
            gtol=1e-8,
            unknown_option="foo",
        )

        problem = OptLibraryTestBase.generate_one_test_unconstrained(
            self.OPT_LIB_NAME, algo_name=algo_name, max_iter=100, max_time=0.0000000001
        )
        assert problem.solution.message.startswith("Maximum time reached")

    def test_slsqp_options(self) -> None:
        """"""
        algo_name = "SLSQP"
        OptLibraryTestBase.generate_one_test(
            algo_name,
            max_iter=100,
            disp=True,
            ftol_rel=1e-10,
        )

    def test_normalization(self) -> None:
        """Runs a problem with one variable to be normalized and three not to be
        normalized."""
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
        OptimizationLibraryFactory().execute(
            problem, algo_name="L-BFGS-B", normalize_design_space=True
        )
        OptimizationLibraryFactory().execute(
            problem, algo_name="L-BFGS-B", normalize_design_space=False
        )

    def test_xtol_ftol_activation(self) -> None:
        def run_pb(algo_options):
            design_space = DesignSpace()
            design_space.add_variable(
                "x1", 2, DesignSpace.DesignVariableType.FLOAT, -1.0, 1.0, 0.0
            )
            problem = OptimizationProblem(design_space)
            problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
            res = OptimizationLibraryFactory().execute(
                problem, algo_name="L-BFGS-B", **algo_options
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
            # Check that the criteria is activated as ap
            assert len(pb.database) == 3


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("SCIPY"):
    setattr(TestScipy, test_method.__name__, test_method)


@pytest.fixture(params=[True, False])
def jacobians_are_sparse(request) -> bool:
    """Whether the Jacobians of MDO Functions are sparse or not."""
    return request.param


@pytest.fixture
def opt_problem(jacobians_are_sparse: bool) -> OptimizationProblem:
    """A linear optimization problem.

    Args:
        jacobians_are_sparse: Whether the objective and constraints Jacobians are
            sparse.

    Returns:
        The linear optimization problem.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=1.0)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=5.0, value=5)
    design_space.add_variable("z", lower_bound=0.0, upper_bound=5.0, value=0)

    problem = OptimizationProblem(design_space)

    array_ = csr_array if jacobians_are_sparse else array
    input_names = ["x", "y", "z"]

    problem.objective = MDOLinearFunction(
        array_([[1.0, 1.0, -1]]), "f", MDOFunction.FunctionType.OBJ, input_names, -1.0
    )
    problem.add_constraint(
        MDOLinearFunction(
            array_([[0, 0.5, -0.25]]),
            "g",
            input_names=input_names,
            f_type=MDOLinearFunction.ConstraintType.INEQ,
        ),
        value=0.333,
        positive=True,
    )
    problem.add_constraint(
        MDOLinearFunction(
            array_([[-2.0, 1.0, 1.0]]),
            "h",
            input_names=input_names,
            f_type=MDOLinearFunction.ConstraintType.EQ,
        )
    )

    return problem


def test_recasting_sparse_jacobians(opt_problem) -> None:
    """Test that sparse Jacobians are recasted as dense arrays.

    The SLSQP algorithm from SciPy does not support sparse Jacobians. The fact that the
    optimizer can be executed and converges implies that the mdo_functions' Jacobians
    are indeed recast as dense NumPy arrays before being sent to SciPy.
    """
    optimization_result = OptimizationLibraryFactory().execute(
        opt_problem,
        algo_name="SLSQP",
        ftol_abs=1e-10,
    )
    assert allclose(optimization_result.f_opt, -0.001, atol=1e-10)


@pytest.mark.parametrize(
    "initial_simplex", [None, [[0.6, 0.6], [0.625, 0.6], [0.6, 0.625]]]
)
def test_nelder_mead(initial_simplex) -> None:
    """Test the Nelder-Mead algorithm on the Rosenbrock problem."""
    problem = Rosenbrock()
    opt = OptimizationLibraryFactory().execute(
        problem, algo_name="NELDER-MEAD", max_iter=800, initial_simplex=initial_simplex
    )
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=1.0e-3)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)


def test_tnc_maxiter(caplog):
    """Check that TNC no longer receives the unknown maxiter option."""
    problem = Rosenbrock()
    with pytest.warns(UserWarning, match="foo") as record:  # noqa: B028, PT031
        OptimizationLibraryFactory().execute(problem, algo_name="TNC", max_iter=2)
        warn("foo", UserWarning, stacklevel=2)

    assert len(record) == 1


@pytest.mark.parametrize("algorithm_name", ["SLSQP", "L-BFGS-B", "TNC", "NELDER-MEAD"])
def test_stop_crit_n_x(algorithm_name) -> None:
    """Check that option stop_crit_n_x is supported."""
    library = ScipyOpt(algorithm_name)
    library._problem = Rosenbrock()
    assert library._validate_settings(stop_crit_n_x=5)["stop_crit_n_x"] == 5


@pytest.mark.skipif(
    not SCIPY_GREATER_THAN_1_14, reason="Algo COBYQA is only available in scipy>=1.14."
)
def test_cobyqa() -> None:
    """Test the COBYQA algorithm on the Rosenbrock problem."""
    problem = Rosenbrock()
    opt = OptimizationLibraryFactory().execute(
        problem, algo_name="COBYQA", max_iter=100
    )
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=1.0e-3)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)


@pytest.mark.skipif(
    not SCIPY_GREATER_THAN_1_14, reason="Algo COBYQA is only available in scipy>=1.14."
)
def test_initial_tr_radius_cobyqa() -> None:
    """Check that option initial_tr_radius is supported."""
    library = ScipyOpt("COBYQA")
    library._problem = Rosenbrock()
    assert library._validate_settings(initial_tr_radius=1)["initial_tr_radius"] == 1


def test_cannot_handle_inequality_constraints():
    """Check the error raised when an algo does not handle inequality constraints."""
    problem = Rosenbrock()
    problem.add_constraint(
        MDOFunction(sum, "sum"),
        value=1.0,
        constraint_type=MDOFunction.ConstraintType.INEQ,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The algorithm TNC is not adapted to the problem "
            "because it does not handle inequality constraints."
        ),
    ):
        OptimizationLibraryFactory().execute(problem, algo_name="TNC")
