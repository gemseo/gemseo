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

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import vstack
from numpy.testing import assert_almost_equal

from gemseo.core.function.array_function import ArrayFunction
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.multi_start.multi_start import MultiStart
from gemseo.optimization.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.optimization.nlopt.settings.nlopt_cobyla_settings import (
    NLOPT_COBYLA_Settings,
)
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.space.design import DesignSpace
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


@pytest.mark.parametrize(
    ("max_iter", "options", "expected_length"),
    [
        (10, {"doe_algo_settings": LHS_Settings(n_samples=5)}, 10),
        (5, {"doe_algo_settings": LHS_Settings(n_samples=4)}, 5),
        (
            15,
            {
                "opt_algo_settings": SLSQP_Settings(max_iter=2),
                "doe_algo_settings": LHS_Settings(n_samples=5),
            },
            11,
        ),
    ],
)
def test_database_length(
    max_iter, options, expected_length, enable_function_statistics
):
    """Check the database length and the number of calls to the objective.

    Flaky test, convergence is dependent on the SLSQP build.
    """
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, settings=MultiStart_Settings(max_iter=max_iter, **options))
    assert len(problem.database) == expected_length
    assert problem.objective.n_calls == 1


@pytest.mark.parametrize("max_iter", [4, 5])
def test_max_iter_error_1(max_iter, snapshot):
    """Check that max_iter <= n_start raises an error."""
    problem = Power2()
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=max_iter, doe_algo_settings=LHS_Settings(n_samples=5)
            ),
        )


def test_max_iter_error_2(snapshot):
    """Check that opt_algo_max_iter * n_start > max_iter raises an error."""
    problem = Power2()
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=10,
                opt_algo_settings=SLSQP_Settings(max_iter=10),
                doe_algo_settings=LHS_Settings(n_samples=5),
            ),
        )


@pytest.mark.parametrize("n_processes", [1, 2])
def test_database(n_processes):
    """Check that the initial points are in the database."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
            n_processes=n_processes,
        ),
    )
    x_history = vstack(problem.database.get_x_vect_history())
    # The first iteration is the evaluation of the functions
    # at the initial design value.
    assert_almost_equal(
        x_history[[0, 1, 6]], array([[1.0, 1.0, 1.0], [0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
    )


def test_normalize_design_space():
    """Check that normalizing the design space changes nothing but the coordinates.

    This algorithm builds sub-problems pairing a copy of the input space
    with the original functions,
    so it must be handed the problem the user declared;
    iterating on a working one paired the normalized space
    with the functions of the original one
    and returned a wrong optimum.
    """
    x_opts = []
    for normalize_design_space in (False, True):
        design_space = DesignSpace()
        # A range straddling neither 0 nor 1,
        # so that a point of the normalized space is outside the declared bounds.
        design_space.add_variable("x", lower_bound=10.0, upper_bound=20.0, value=12.0)
        problem = OptimizationProblem(design_space)
        problem.objective = ArrayFunction(
            lambda x: array([(x[0] - 16.0) ** 2]),
            name="f",
            jac=lambda x: array([[2.0 * (x[0] - 16.0)]]),
        )
        result = MultiStart().execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=20,
                normalize_design_space=normalize_design_space,
                opt_algo_settings=SLSQP_Settings(max_iter=5),
                doe_algo_settings=CustomDOE_Settings(samples=array([[11.0], [19.0]])),
            ),
        )
        x_opts.append(result.x_opt)
        for x_vect in problem.database.get_x_vect_history():
            assert 10.0 <= x_vect[0] <= 20.0

    assert_almost_equal(x_opts[1], x_opts[0])


def test_factory():
    """Check that the factory of optimization algorithms knows this algorithm."""
    assert optimization_library_factory.is_available("MultiStart")


def test_relaxed_integer_variable():
    """Check that a relaxed integer variable is projected back and flagged.

    `MultiStart` iterates on no working problem of its own,
    so its own `_transformation` is an empty composition;
    the point it writes back into the design space, and `to_dataset()`,
    must nonetheless account for the relaxation the sub-optimizations did.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", type_="integer", lower_bound=0, upper_bound=10)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([(x[0] - 3.4) ** 2]),
        name="f",
        jac=lambda x: array([[2.0 * (x[0] - 3.4)]]),
    )
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=20,
            opt_algo_settings=SLSQP_Settings(relax_integer_variables=True, max_iter=5),
            doe_algo_settings=CustomDOE_Settings(samples=array([[1.0], [5.0]])),
        ),
    )
    assert design_space.get_current_value() == array([3])
    assert "x" in problem.database.relaxed_variable_names
    # `to_dataset` casts an integer input column to a `pandas.Int64Dtype`,
    # which fails on a relaxed, non-integral, value unless the column is
    # exported as a float one, decided from `relaxed_variable_names`.
    dataset = problem.to_dataset()
    assert dataset.get_view(variable_names="x").to_numpy().dtype == float


@pytest.fixture(scope="module")
def x_history() -> RealArray:
    """The reference samples."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10, doe_algo_settings=LHS_Settings(n_samples=5)
        ),
    )
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "settings",
    [
        {"opt_algo_settings": NLOPT_COBYLA_Settings()},
        {"doe_algo_settings": MC_Settings(n_samples=5)},
    ],
)
def test_algo_settings(x_history, settings):
    """Check that the algorithm settings can be changed."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, settings=MultiStart_Settings(max_iter=10, **settings))
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history)


@pytest.fixture(scope="module")
def x_history_cobyla() -> RealArray:
    """The reference samples with COBYLA algorithm."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=NLOPT_COBYLA_Settings(),
            doe_algo_settings=LHS_Settings(n_samples=5),
        ),
    )
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "options",
    [
        {
            "opt_algo_settings": NLOPT_COBYLA_Settings(init_step=0.5),
            "doe_algo_settings": LHS_Settings(n_samples=5),
        },
        {"doe_algo_settings": LHS_Settings(n_samples=5, scramble=False)},
    ],
)
def test_algo_settings_(x_history_cobyla, options):
    """Check that the algorithm options can be changed."""
    problem = Power2()
    algo = MultiStart()
    kwargs = {"opt_algo_settings": NLOPT_COBYLA_Settings()}
    kwargs.update(options)
    algo.execute(problem, settings=MultiStart_Settings(max_iter=10, **kwargs))
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history_cobyla)


def test_multistart_file_path(tmp_wd):
    """Check the multistart_file_path option."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=LHS_Settings(n_samples=5),
            multistart_file_path="local_optima.hdf5",
        ),
    )
    problem = problem.__class__.from_hdf("local_optima.hdf5")
    assert len(problem.database) == 5
