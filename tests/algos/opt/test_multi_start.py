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

import re
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import vstack
from numpy.testing import assert_almost_equal

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.multi_start.multi_start import MultiStart
from gemseo.problems.optimization.power_2 import Power2

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.mark.parametrize(
    ("max_iter", "options", "expected_length"),
    [
        (10, {}, 10),
        (10, {"n_start": 4}, 9),
        (15, {"opt_algo_max_iter": 2}, 11),
    ],
)
def test_database_length(max_iter, options, expected_length):
    """Check the database length and the number of calls to the objective."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=max_iter, **options)
    assert len(problem.database) == expected_length
    assert problem.objective.n_calls == 1


@pytest.mark.parametrize("max_iter", [4, 5])
def test_max_iter_error_1(max_iter):
    """Check that max_iter <= n_start raises an error."""
    problem = Power2()
    algo = MultiStart()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Multi-start optimization: "
            f"the maximum number of iterations ({max_iter}) must be "
            "greater than the number of initial points (5)."
        ),
    ):
        algo.execute(problem, max_iter=max_iter)


def test_max_iter_error_2():
    """Check that opt_algo_max_iter * n_start > max_iter raises an error."""
    problem = Power2()
    algo = MultiStart()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Multi-start optimization: "
            "the sum of the maximum number of iterations (50) "
            "related to the sub-optimizations is greater than the limit (10-1=9)."
        ),
    ):
        algo.execute(problem, max_iter=10, n_start=5, opt_algo_max_iter=10)


@pytest.mark.parametrize("n_processes", [1, 2])
def test_database(n_processes):
    """Check that the initial points are in the database."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        max_iter=10,
        n_start=2,
        doe_algo_name="CustomDOE",
        doe_algo_settings={"samples": array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])},
        n_processes=n_processes,
    )
    x_history = vstack(problem.database.get_x_vect_history())
    # The first iteration is the evaluation of the functions
    # at the initial design value.
    assert_almost_equal(
        x_history[[0, 1, 6]], array([[1.0, 1.0, 1.0], [0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
    )


def test_factory():
    """Check that the factory of optimization algorithms knows this algorithm."""
    assert OptimizationLibraryFactory().is_available("MultiStart")


@pytest.fixture(scope="module")
def x_history() -> RealArray:
    """The reference samples."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=10)
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "settings",
    [{"opt_algo_name": "NLOPT_COBYLA"}, {"doe_algo_name": "MC"}],
)
def test_algo_name(x_history, settings):
    """Check that the algorithm name can be changed."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=10, **settings)
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history)


@pytest.fixture(scope="module")
def x_history_cobyla() -> RealArray:
    """The reference samples with COBYLA algorithm."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=10, opt_algo_name="NLOPT_COBYLA")
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "options",
    [
        {"opt_algo_settings": {"init_step": 0.5}},
        {"doe_algo_settings": {"scramble": False}},
    ],
)
def test_algo_settings(x_history_cobyla, options):
    """Check that the algorithm options can be changed."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=10, opt_algo_name="NLOPT_COBYLA", **options)
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history_cobyla)


def test_multistart_file_path(tmp_wd):
    """Check the multistart_file_path option."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, max_iter=10, multistart_file_path="local_optima.hdf5")
    problem = problem.__class__.from_hdf("local_optima.hdf5")
    assert len(problem.database) == 5
