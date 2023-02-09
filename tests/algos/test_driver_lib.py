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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Driver library tests."""
from __future__ import annotations

from unittest import mock

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.driver_lib import DriverDescription
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.problems.analytical.power_2 import Power2
from numpy import array


class MyDriver(DriverLib):
    def __init__(self):
        super().__init__()
        self.descriptions = {"algo_name": None}


@pytest.fixture(scope="module")
def optimization_problem():
    """A mock optimization problem."""
    design_space = mock.Mock()
    design_space.dimension = 2
    problem = mock.Mock()
    problem.dimension = 2
    problem.design_space = design_space
    return problem


def test_empty_design_space():
    """Check that a driver cannot be executed with an empty design space."""
    driver = MyDriver()
    driver.algo_name = "algo_name"
    with pytest.raises(
        ValueError,
        match=(
            "The algorithm algo_name is not adapted to the problem "
            "because the design space is empty."
        ),
    ):
        driver._check_algorithm("algo_name", OptimizationProblem(DesignSpace()))


def test_max_iter_fail(optimization_problem):
    """Check that a ValueError is raised for an invalid `max_iter` input."""
    MyDriver()._pre_run(optimization_problem, None)
    with pytest.raises(ValueError, match="max_iter must be >=1, got -1"):
        MyDriver().init_iter_observer(max_iter=-1)


def test_no_algo_fail(optimization_problem):
    """Check that a ValueError is raised when no algorithm name is set."""
    with pytest.raises(
        ValueError,
        match="Algorithm name must be either passed as "
        "argument or set by the attribute 'algo_name'.",
    ):
        MyDriver().execute(optimization_problem)


def test_grammar_fail():
    """Check that a ValueError is raised when the grammar file is not found."""
    with pytest.raises(
        ValueError,
        match=(
            "Neither the options grammar file .+ for the algorithm 'unknown' "
            "nor the options grammar file .+ for the library 'DriverLib' "
            "has been found."
        ),
    ):
        DriverLib().init_options_grammar("unknown")


def test_require_grad():
    """Check that an error is raised when a particular gradient method is not given."""

    class MyDriver(DriverLib):
        def __init__(self):
            super().__init__()
            self.descriptions = {
                "SLSQP": DriverDescription(
                    algorithm_name="SLSQP",
                    internal_algorithm_name="SLSQP",
                    require_gradient=True,
                )
            }

    with pytest.raises(ValueError, match="Algorithm toto is not available."):
        MyDriver().is_algo_requires_grad("toto")

    assert MyDriver().is_algo_requires_grad("SLSQP")


def test_new_iteration_callback_xvect(caplog):
    """Test the new iteration callback when no x_vect is given.

    Args:
        caplog: Fixture to access and control log capturing.
    """
    problem = Power2()
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )

    test_driver = DriverLib()
    test_driver.problem = problem
    test_driver._max_time = 0
    test_driver.init_iter_observer(max_iter=2)
    test_driver.new_iteration_callback()
    assert "...   0%|" in caplog.text


def test_init_iter_observer_message(caplog):
    """Check the iteration prefix in init_iter_observer."""
    test_driver = DriverLib()
    test_driver.problem = Power2()
    test_driver.init_iter_observer(max_iter=2)
    assert "...   0%|" in caplog.text
    test_driver.init_iter_observer(max_iter=2, message="foo")
    assert "foo   0%|" in caplog.text


@pytest.mark.parametrize("activate_progress_bar", [False, True])
def test_progress_bar(activate_progress_bar):
    """Check the activation of the progress bar from the options of a DriverLib."""
    driver = OptimizersFactory().create("SLSQP")
    driver.execute(Power2(), activate_progress_bar=activate_progress_bar)
    assert (driver._DriverLib__progress_bar is None) is not activate_progress_bar


def test_common_options():
    """Check that the options common to all the drivers are in the option grammar."""
    driver = MyDriver()
    driver.init_options_grammar("AlgoName")
    assert driver.opt_grammar.names == {
        DriverLib.ROUND_INTS_OPTION,
        DriverLib.NORMALIZE_DESIGN_SPACE_OPTION,
        DriverLib.USE_DATABASE_OPTION,
        DriverLib._DriverLib__RESET_ITERATION_COUNTERS_OPTION,
    }
    assert not driver.opt_grammar.required_names
