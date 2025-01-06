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

import logging
import re
from typing import ClassVar
from unittest import mock

import pytest
from numpy import array
from numpy import full

from gemseo.algos._progress_bars.progress_bar import ProgressBar
from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.base_driver_library import DriverDescription
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.scipy_local.scipy_local import ScipyOpt
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture(scope="module")
def power_2() -> Power2:
    """The power-2 problem."""
    problem = Power2()
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )
    return problem


class MyDriver(BaseDriverLibrary):
    ALGORITHM_INFOS: ClassVar[dict[str, DriverDescription]] = {"algo_name": None}

    def __init__(self, algo_name: str = "algo_name") -> None:
        super().__init__(algo_name)


@pytest.fixture(scope="module")
def optimization_problem():
    """A mock optimization problem."""
    design_space = mock.Mock()
    design_space.dimension = 2
    problem = mock.Mock()
    problem.dimension = 2
    problem.design_space = design_space
    return problem


def test_empty_design_space() -> None:
    """Check that a driver cannot be executed with an empty design space."""
    with concretize_classes(MyDriver):
        driver = MyDriver()
    driver._algo_name = "algo_name"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The algorithm algo_name is not adapted to the problem "
            "because the design space is empty."
        ),
    ):
        driver._check_algorithm(OptimizationProblem(DesignSpace()))


@pytest.mark.parametrize(
    ("kwargs", "expected"), [({}, "    50%|"), ({"message": "foo"}, "foo  50%|")]
)
def test_new_iteration_callback_xvect(caplog, power_2, kwargs, expected) -> None:
    """Test the new iteration callback."""
    test_driver = ScipyOpt("SLSQP")
    test_driver._problem = power_2
    test_driver._max_time = 0
    test_driver._init_iter_observer(power_2, max_iter=2, **kwargs)
    test_driver._new_iteration_callback(array([0, 0]))
    test_driver._new_iteration_callback(array([0, 0]))
    assert expected in caplog.text


@pytest.mark.parametrize("enable_progress_bar", [False, True])
def test_progress_bar(enable_progress_bar, caplog) -> None:
    """Check the activation of the progress bar from the options of a
    BaseDriverLibrary."""
    driver = OptimizationLibraryFactory().create("SLSQP")
    driver.execute(Power2(), enable_progress_bar=enable_progress_bar)
    assert (
        isinstance(driver._BaseDriverLibrary__progress_bar, ProgressBar)
        is enable_progress_bar
    )
    assert (
        "Solving optimization problem with algorithm SLSQP" in caplog.text
    ) is enable_progress_bar


@pytest.fixture
def driver_library() -> BaseDriverLibrary:
    """A driver library."""
    driver_library = ScipyOpt("SLSQP")
    design_space = DesignSpace()
    design_space.add_variable("x", 1, "float", -2, 3, 1)
    driver_library._problem = OptimizationProblem(design_space)
    return driver_library


@pytest.mark.parametrize(
    ("as_dict", "x0", "lower_bounds", "upper_bounds"),
    [(False, 0.6, 0, 1), (True, {"x": 0.6}, {"x": 0}, {"x": 1})],
)
def test_get_value_and_bounds_vects_normalized_as_ndarrays(
    driver_library, as_dict, x0, lower_bounds, upper_bounds
) -> None:
    """Check the getting of the normalized initial values and bounds."""
    assert get_value_and_bounds(
        driver_library._problem.design_space, True, as_dict=as_dict
    ) == (
        pytest.approx(x0),
        lower_bounds,
        upper_bounds,
    )


@pytest.mark.parametrize(
    ("as_dict", "x0", "lower_bounds", "upper_bounds"),
    [(False, 1, -2, 3), (True, {"x": 1}, {"x": -2}, {"x": 3})],
)
def test_get_value_and_bounds_vects_non_normalized(
    driver_library, as_dict, x0, lower_bounds, upper_bounds
) -> None:
    """Check the getting of the non-normalized initial values and bounds."""
    assert get_value_and_bounds(
        driver_library._problem.design_space, False, as_dict=as_dict
    ) == (
        x0,
        lower_bounds,
        upper_bounds,
    )


@pytest.mark.parametrize("name", ["new_iter_listener", "store_listener"])
def test_clear_listeners(name):
    """Check clear_listeners."""
    problem = Power2()
    getattr(problem.database, f"add_{name}")(sum)
    driver = CustomDOE()
    driver.execute(problem, samples=array([[-0.5, 0.0, 0.5]]))
    assert getattr(problem.database, f"_Database__{name}s") == [sum]


@pytest.mark.parametrize("max_dimension", [1, 3])
def test_max_design_space_dimension_to_log(max_dimension, caplog):
    """Check the cap on the dimension of a design space to log."""
    problem = Power2()
    initial_space_string = problem.design_space._get_string_representation(
        False, "   over the design space"
    ).replace("\n", "\n      ")
    CustomDOE().execute(
        problem,
        samples=full((1, 3), pow(0.9, 1.0 / 3.0)),
        max_design_space_dimension_to_log=max_dimension,
    )

    # Check the logging of the initial design space
    assert (max_dimension >= 3) == (
        ("gemseo.algos.base_driver_library", logging.INFO, initial_space_string)
        in caplog.record_tuples
    )

    # Check the logging of the final design space
    assert (max_dimension >= 3) == (
        (
            "gemseo.algos.base_driver_library",
            logging.INFO,
            problem.design_space._get_string_representation(False)
            .replace("Design space", "      Design space")
            .replace("\n", "\n         "),
        )
        in caplog.record_tuples
    )
