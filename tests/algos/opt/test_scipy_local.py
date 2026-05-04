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

from unittest import TestCase
from warnings import warn

import pytest
from numpy import allclose
from numpy import array
from numpy import inf
from pydantic import ValidationError
from scipy.optimize import rosen
from scipy.optimize import rosen_der
from scipy.sparse import csr_array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.scipy_local.scipy_local import ScipyOpt
from gemseo.algos.opt.scipy_local.settings.cobyla import COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.cobyqa import COBYQA_Settings
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.opt.scipy_local.settings.nelder_mead import NELDER_MEAD_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.opt.scipy_local.settings.tnc import TNC_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.core.functions.linear_function import LinearFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.compatibility.scipy import SCIPY_GREATER_THAN_1_14
from gemseo.utils.compatibility.scipy import SCIPY_GREATER_THAN_1_16
from gemseo.utils.pydantic import create_model
from gemseo.utils.testing.helpers import assert_exception
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
            ValueError,
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

        problem.objective = ArrayFunction(i_fail, name="rosen")
        self.assertRaises(
            AttributeError, OPTIMIZATION_LIBRARY_FACTORY.execute, problem, algo_name
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
        algo_name = "L_BFGS_B"
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
        problem.objective = ArrayFunction(
            rosen, name="Rosenbrock", f_type="obj", jac=rosen_der
        )
        OPTIMIZATION_LIBRARY_FACTORY.execute(
            problem, settings=L_BFGS_B_Settings(normalize_design_space=True)
        )
        OPTIMIZATION_LIBRARY_FACTORY.execute(
            problem, settings=L_BFGS_B_Settings(normalize_design_space=False)
        )

    def test_xtol_ftol_activation(self) -> None:
        def run_pb(algo_options):
            design_space = DesignSpace()
            design_space.add_variable(
                "x1", 2, DesignSpace.DesignVariableType.FLOAT, -1.0, 1.0, 0.0
            )
            problem = OptimizationProblem(design_space)
            problem.objective = ArrayFunction(
                rosen, name="Rosenbrock", f_type="obj", jac=rosen_der
            )
            res = OPTIMIZATION_LIBRARY_FACTORY.execute(
                problem, settings=L_BFGS_B_Settings(**algo_options)
            )
            return res, problem

        for tol_name in (
            "ftol_abs",
            "ftol_rel",
            "xtol_abs",
            "xtol_rel",
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
    """Whether the Jacobians of array functions are sparse or not."""
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

    problem.objective = LinearFunction(
        array_([[1.0, 1.0, -1]]), "f", ArrayFunction.FunctionType.OBJ, input_names, -1.0
    )
    problem.add_constraint(
        LinearFunction(
            array_([[0, 0.5, -0.25]]),
            "g",
            input_names=input_names,
        ),
        value=0.333,
        positive=True,
        constraint_type=LinearFunction.ConstraintType.INEQ,
    )
    problem.add_constraint(
        LinearFunction(
            array_([[-2.0, 1.0, 1.0]]),
            "h",
            input_names=input_names,
            f_type=LinearFunction.ConstraintType.EQ,
        )
    )

    return problem


def test_recasting_sparse_jacobians(opt_problem) -> None:
    """Test that sparse Jacobians are recasted as dense arrays.

    The SLSQP algorithm from SciPy does not support sparse Jacobians. The fact that the
    optimizer can be executed and converges implies that the mdo_functions' Jacobians
    are indeed recast as dense NumPy arrays before being sent to SciPy.
    """
    optimization_result = OPTIMIZATION_LIBRARY_FACTORY.execute(
        opt_problem, settings=SLSQP_Settings(ftol_abs=1e-10)
    )
    assert allclose(optimization_result.f_opt, -0.001, atol=1e-10)


@pytest.mark.parametrize(
    "initial_simplex", [None, [[0.6, 0.6], [0.625, 0.6], [0.6, 0.625]]]
)
def test_nelder_mead(initial_simplex) -> None:
    """Test the Nelder-Mead algorithm on the Rosenbrock problem."""
    problem = Rosenbrock()
    opt = OPTIMIZATION_LIBRARY_FACTORY.execute(
        problem,
        settings=NELDER_MEAD_Settings(max_iter=800, initial_simplex=initial_simplex),
    )
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=1.0e-3)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)


def test_tnc_maxiter(caplog):
    """Check that TNC no longer receives the unknown maxiter option."""
    problem = Rosenbrock()
    with pytest.warns(UserWarning, match="foo") as record:  # noqa: B028, PT031
        OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=TNC_Settings(max_iter=2))
        warn("foo", UserWarning, stacklevel=2)

    assert len(record) == 1


@pytest.mark.parametrize("algorithm_name", ["SLSQP", "L_BFGS_B", "TNC", "NELDER_MEAD"])
def test_stop_crit_n_x(algorithm_name) -> None:
    """Check that option stop_crit_n_x is supported."""
    library = ScipyOpt(algorithm_name)
    library._problem = Rosenbrock()
    library._settings = create_model(
        library.ALGORITHM_INFOS[library.algo_name].settings_class, stop_crit_n_x=5
    )
    assert library._settings.stop_crit_n_x == 5


@pytest.mark.skipif(
    not SCIPY_GREATER_THAN_1_14, reason="Algo COBYQA is only available in scipy>=1.14."
)
def test_cobyqa() -> None:
    """Test the COBYQA algorithm on the Rosenbrock problem."""
    problem = Rosenbrock()
    opt = OPTIMIZATION_LIBRARY_FACTORY.execute(
        problem, settings=COBYQA_Settings(max_iter=100)
    )
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=2.0e-2)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)


@pytest.mark.skipif(
    not SCIPY_GREATER_THAN_1_14, reason="Algo COBYQA is only available in scipy>=1.14."
)
def test_initial_tr_radius_cobyqa() -> None:
    """Check that option initial_tr_radius is supported."""
    reference = ScipyOpt("COBYQA").execute(
        Rosenbrock(), settings=COBYQA_Settings(max_iter=10)
    )
    result = ScipyOpt("COBYQA").execute(
        Rosenbrock(),
        settings=COBYQA_Settings(max_iter=10, initial_tr_radius=0.05),
    )
    assert reference.f_opt != result.f_opt


def test_cobyla() -> None:
    """Test the COBYLA algorithm on the Rosenbrock problem."""
    problem = Rosenbrock()
    opt = ScipyOpt("COBYLA").execute(
        problem, settings=COBYLA_Settings(max_iter=500, enable_progress_bar=False)
    )
    x_opt, f_opt = problem.get_solution()
    xtol = 2.0e-1 if SCIPY_GREATER_THAN_1_16 else 6.0e-1
    ftol = 1.0e-2 if SCIPY_GREATER_THAN_1_16 else 1.1e-1
    assert opt.x_opt == pytest.approx(x_opt, abs=xtol)
    assert opt.f_opt == pytest.approx(f_opt, abs=ftol)


def test_rhobeg_cobyla() -> None:
    """Check that option rhobeg is supported."""
    reference = ScipyOpt("COBYLA").execute(
        Rosenbrock(),
        settings=COBYLA_Settings(max_iter=10, enable_progress_bar=False),
    )
    result = ScipyOpt("COBYLA").execute(
        Rosenbrock(),
        settings=COBYLA_Settings(max_iter=10, rhobeg=0.1, enable_progress_bar=False),
    )
    assert reference.f_opt != result.f_opt


def test_catol_cobyla() -> None:
    """Check that option catol is supported."""
    problem = Rosenbrock()
    problem.add_constraint(
        ArrayFunction(lambda x: x[0] + x[1], name="constr"),
        value=1.0,
        constraint_type=ArrayFunction.ConstraintType.INEQ,
    )
    reference = ScipyOpt("COBYLA").execute(
        problem, settings=COBYLA_Settings(max_iter=10, enable_progress_bar=False)
    )
    result = ScipyOpt("COBYLA").execute(
        problem,
        settings=COBYLA_Settings(max_iter=10, catol=1e-5, enable_progress_bar=False),
    )
    assert reference.f_opt != result.f_opt


def test_cannot_handle_inequality_constraints(snapshot):
    """Check the error raised when an algo does not handle inequality constraints."""
    problem = Rosenbrock()
    problem.add_constraint(
        ArrayFunction(sum, name="sum"),
        value=1.0,
        constraint_type=ArrayFunction.ConstraintType.INEQ,
    )
    with assert_exception(ValueError, snapshot):
        OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=TNC_Settings())
