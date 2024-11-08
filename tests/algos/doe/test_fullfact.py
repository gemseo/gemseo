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

import logging

import pytest
from numpy import allclose
from numpy import array
from numpy import array_equal
from numpy import atleast_2d
from pydantic import ValidationError

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.base_full_factorial_doe import BaseFullFactorialDOE
from gemseo.algos.doe.openturns.openturns import OpenTURNS
from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture
def doe_problem_dim_2():
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, lower_bound=-2.0, upper_bound=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(sum, "func")
    return problem


@pytest.mark.parametrize(
    ("doe_library_class", "algo_name"),
    [(PyDOELibrary, "PYDOE_FULLFACT"), (OpenTURNS, "OT_FULLFACT")],
)
@pytest.mark.parametrize(
    "expected",
    [
        array([[1.0]]),
        array([[0.0], [2.0]]),
        array([[0.0], [1.0], [2.0]]),
        array([[1.0, 1.0]]),
        array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
    ],
)
def test_fullfact_values(doe_library_class, algo_name, expected) -> None:
    """Check fullfactorial DOEs in terms of exact values."""
    n_samples, size = atleast_2d(expected).shape
    n_samples = int(n_samples)
    design_space = DesignSpace()
    design_space.add_variable("x", size=size, lower_bound=0.0, upper_bound=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(sum, "func")
    doe_library_class(algo_name).execute(problem, n_samples=n_samples)
    assert array_equal(
        problem.to_dataset("data").get_view(variable_names="x").to_numpy(),
        expected,
    )


@pytest.mark.parametrize(
    ("doe_library_class", "algo_name"),
    [(PyDOELibrary, "PYDOE_FULLFACT"), (OpenTURNS, "OT_FULLFACT")],
)
@pytest.mark.parametrize("n_samples", [1, 100])
@pytest.mark.parametrize("size", [2, 5])
def test_fullfact_properties(doe_library_class, algo_name, n_samples, size) -> None:
    """Check fullfactorial DOEs in terms of properties (bounds and dimensions)."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=size, lower_bound=0.0, upper_bound=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(sum, "func")
    doe_library_class(algo_name).execute(problem, n_samples=n_samples)
    data = problem.to_dataset().get_view(variable_names="x").to_numpy()
    if n_samples < 2**size:
        expected_min = expected_max = 1.0
        expected_shape = (1, size)
    else:
        expected_min = 0.0
        expected_max = 2.0
        expected_shape = (int(n_samples ** (1.0 / size)) ** size, size)

    for feature in data.T:
        assert feature.min() == expected_min
        assert feature.max() == expected_max

    assert data.shape == expected_shape


@pytest.mark.parametrize(
    ("doe_library_class", "algo_name"),
    [(PyDOELibrary, "PYDOE_FULLFACT"), (OpenTURNS, "OT_FULLFACT")],
)
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {"levels": [2, 3]},
            array([[-2, -2], [2, -2], [-2, 0], [2, 0], [-2, 2], [2, 2]]),
        ),
        (
            {"levels": [2, 1]},
            array([[-2, 0], [2, 0]]),
        ),
        ({"levels": 1}, array([[0, 0]])),
    ],
)
def test_fullfact_levels(
    doe_problem_dim_2, doe_library_class, algo_name, options, expected
) -> None:
    """Check that ``levels`` option in full-factorial is correctly taken into
    account."""

    doe_library_class(algo_name).execute(doe_problem_dim_2, **options)
    assert allclose(doe_problem_dim_2.database.get_x_vect_history(), expected)


@pytest.mark.parametrize(
    ("doe_library_class", "algo_name"),
    [(PyDOELibrary, "PYDOE_FULLFACT"), (OpenTURNS, "OT_FULLFACT")],
)
@pytest.mark.parametrize(
    ("options", "exception", "error_msg"),
    [
        (
            {},
            ValueError,
            "Either 'n_samples' or 'levels' is required as an input parameter "
            "for the full-factorial DOE.",
        ),
        (
            {"n_samples": 6, "levels": [2, 2]},
            ValueError,
            "Only one input parameter among 'n_samples' and 'levels' must be given "
            "for the full-factorial DOE.",
        ),
        (
            {"levels": -1},
            ValidationError,
            None,
        ),  # Raised by grammar, do not check the message
    ],
)
def test_fullfact_error(
    doe_problem_dim_2, doe_library_class, algo_name, options, exception, error_msg
) -> None:
    """Check that an error is raised if both levels and n_sample are provided, or if
    none of them are provided.

    Also check negative levels
    """

    with pytest.raises(exception, match=error_msg):
        doe_library_class(algo_name).execute(doe_problem_dim_2, **options)


def test__compute_fullfact_levels(caplog) -> None:
    """Check the WARNING logged when the number of samples is less than expected."""
    with concretize_classes(BaseFullFactorialDOE):
        BaseFullFactorialDOE()._compute_fullfact_levels(10, 3)
    message = (
        "A full-factorial DOE of 10 samples in dimension 3 does not exist; "
        "use 8 samples instead, i.e. the largest 3-th integer power less than 10."
    )
    _, log_level, log_message = caplog.record_tuples[0]
    assert log_level == logging.WARNING
    assert message in log_message


def test_numerical_precision_issue() -> None:
    """Check that the number of samples is robust to numerical precision."""
    with concretize_classes(BaseFullFactorialDOE):
        levels = BaseFullFactorialDOE()._compute_fullfact_levels(1000, 3)

    # In the issue #1028, the result was wrong: [9, 9, 9].
    assert levels == [10, 10, 10]
