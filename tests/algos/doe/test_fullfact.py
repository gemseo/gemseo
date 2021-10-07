# -*- coding: utf-8 -*-
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
import pytest
from numpy import allclose, array, array_equal, atleast_2d

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction


@pytest.fixture()
def doe_problem_dim_2():
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, l_b=-2.0, u_b=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: sum(x), "func")
    return problem


@pytest.mark.parametrize(
    "doe_library_class,algo_name", [(PyDOE, "fullfact"), (OpenTURNS, "OT_FULLFACT")]
)
@pytest.mark.parametrize(
    "expected",
    [
        array([1.0]),
        array([[0.0], [2.0]]),
        array([[0.0], [1.0], [2.0]]),
        array([1.0, 1.0]),
        array([1.0, 1.0]),
        array([1.0, 1.0]),
        array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
    ],
)
def test_fullfact_values(doe_library_class, algo_name, expected):
    """Check fullfactorial DOEs in terms of exact values."""
    n_samples, size = atleast_2d(expected).shape
    n_samples = int(n_samples)
    design_space = DesignSpace()
    design_space.add_variable("x", size=size, l_b=0.0, u_b=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: sum(x), "func")
    doe_library_class().execute(problem, algo_name, n_samples=n_samples)
    assert array_equal(problem.export_to_dataset("data")["x"]["x"], expected)


@pytest.mark.parametrize(
    "doe_library_class,algo_name", [(PyDOE, "fullfact"), (OpenTURNS, "OT_FULLFACT")]
)
@pytest.mark.parametrize("n_samples", [1, 100])
@pytest.mark.parametrize("size", [2, 5])
def test_fullfact_properties(doe_library_class, algo_name, n_samples, size):
    """Check fullfactorial DOEs in terms of properties (bounds and dimensions)."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=size, l_b=0.0, u_b=2.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: sum(x), "func")
    doe_library_class().execute(problem, algo_name, n_samples=n_samples)
    data = problem.export_to_dataset("data")["x"]["x"]
    if n_samples < 2 ** size:
        expected_min = expected_max = 1.0
        expected_ndim = 1
        expected_shape_0 = size
    else:
        expected_min = 0.0
        expected_max = 2.0
        expected_ndim = 2
        expected_shape_0 = int(n_samples ** (1.0 / size)) ** size

    for feature in data.T:
        assert feature.min() == expected_min
        assert feature.max() == expected_max

    assert data.ndim == expected_ndim
    assert data.shape[0] == expected_shape_0


@pytest.mark.parametrize(
    "doe_library_class, algo_name", [(PyDOE, "fullfact"), (OpenTURNS, "OT_FULLFACT")]
)
@pytest.mark.parametrize(
    "options, expected",
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
):
    """Check that ``levels`` option in full-factorial is correctly taken into
    account."""

    doe_library_class().execute(doe_problem_dim_2, algo_name, **options)
    assert allclose(doe_problem_dim_2.database.get_x_history(), expected)


@pytest.mark.parametrize(
    "doe_library_class, algo_name", [(PyDOE, "fullfact"), (OpenTURNS, "OT_FULLFACT")]
)
@pytest.mark.parametrize(
    "options, error_msg",
    [
        (
            {},
            "Either 'n_samples' or 'levels' is required as an input parameter "
            "for the full-factorial DOE.",
        ),
        (
            {"n_samples": 6, "levels": [2, 2]},
            "Only one input parameter among 'n_samples' and 'levels' must be given "
            "for the full-factorial DOE.",
        ),
        ({"levels": -1}, None),  # Raised by grammar, do not check the message
    ],
)
def test_fullfact_error(
    doe_problem_dim_2, doe_library_class, algo_name, options, error_msg
):
    """Check that an error is raised if both levels and n_sample are provided, or if
    none of them are provided.

    Also check negative levels
    """

    with pytest.raises(ValueError, match=error_msg):
        doe_library_class().execute(doe_problem_dim_2, algo_name, **options)
