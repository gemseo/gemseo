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

from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.multi_start.multi_start import MultiStart
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.helpers import assert_exception

if TYPE_CHECKING:
    from gemseo.typing import RealArray


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


def test_factory():
    """Check that the factory of optimization algorithms knows this algorithm."""
    assert OPTIMIZATION_LIBRARY_FACTORY.is_available("MultiStart")


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
