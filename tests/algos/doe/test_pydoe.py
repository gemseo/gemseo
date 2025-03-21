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
#      :author: Damien Guenot - 20 avr. 2016
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from typing import Any

import pytest
from numpy.linalg import norm
from pydantic import ValidationError

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.problems.optimization.rosenbrock import Rosenbrock

from .utils import execute_problem
from .utils import generate_test_functions

DOE_LIB_NAME = "PyDOELibrary"


def test_invalid_algo() -> None:
    """Check that an invalid algorithm name."""
    with pytest.raises(
        ValueError,
        match=(
            r"No algorithm named unknown_algo is available; available algorithms are .+"
        ),
    ):
        execute_problem(
            "unknown_algo",
            dim=2,
            n_samples=3,
        )


def test_lhs_maximin() -> None:
    """Check that an optimal LHS and LHS are different."""
    dim = 3
    algo_name = "PYDOE_LHS"
    n_samples = 100
    doe_library = execute_problem(
        algo_name,
        dim=dim,
        n_samples=n_samples,
        criterion="maximin",
    )
    samples_maximin = doe_library.unit_samples
    doe_library = execute_problem(algo_name, dim=dim, n_samples=n_samples)
    samples = doe_library.unit_samples
    assert norm(samples - samples_maximin) / norm(samples_maximin) >= 1e-8


@pytest.mark.parametrize(
    ("algo_name", "options", "error"),
    [
        (
            "PYDOE_CCDESIGN",
            {"alpha": "unknown_value"},
            "Input should be 'orthogonal', 'o', 'rotatable' or 'r'",
        ),
        (
            "PYDOE_CCDESIGN",
            {"face": "unknown_value"},
            "Input should be 'circumscribed', 'ccc', 'inscribed', 'cci', 'faced' or"
            " 'ccf'",
        ),
        (
            "PYDOE_CCDESIGN",
            {"center": 1},
            "Input should be a valid tuple",
        ),
        (
            "PYDOE_BBDESIGN",
            {"center": (4, 4)},
            "Input should be a valid integer",
        ),
        (
            "PYDOE_LHS",
            {"criterion": "corr", "iterations": "unknown_value", "n_samples": 2},
            "Input should be a valid integer",
        ),
        (
            "PYDOE_LHS",
            {"criterion": "unknown_value", "n_samples": 2},
            "Input should be 'center', 'c', 'maximin', 'm', 'centermaximin', 'cm',"
            " 'correlation', 'corr' or 'lhsmu'",
        ),
    ],
)
def test_algo_with_unknown_options(algo_name, options, error) -> None:
    """Check that exceptions are raised when unknown options are passed to an algo."""
    match = f"{error}"
    with pytest.raises(ValidationError, match=re.escape(match)):
        execute_problem(algo_name, dim=3, **options)


@pytest.mark.parametrize(
    ("algo_name", "dim", "n_samples", "options"),
    [
        ("PYDOE_CCDESIGN", 5, 62, {"center": [10, 10]}),
        ("PYDOE_BBDESIGN", 5, 41, {"center": 1}),
        ("PYDOE_LHS", 2, 3, {"n_samples": 3, "criterion": "corr"}),
        ("PYDOE_LHS", 2, 3, {"n_samples": 3, "criterion": "centermaximin"}),
        ("PYDOE_LHS", 2, 3, {"n_samples": 3, "criterion": "center"}),
        ("PYDOE_LHS", 2, 3, {"n_samples": 3, "criterion": "maximin"}),
    ],
)
def test_algos(algo_name, dim, n_samples, options) -> None:
    """Check that the PyDOE library returns samples correctly shaped."""
    doe_library = execute_problem(algo_name, dim=dim, **options)
    assert doe_library.unit_samples.shape == (n_samples, dim)


def test_integer_lhs() -> None:
    """Check that a DOE with integer variables stores integer values in the Database."""
    problem = Rosenbrock()
    problem.design_space.add_variable(
        "y", type_="integer", lower_bound=10.0, upper_bound=15.0
    )
    DOELibraryFactory().execute(problem, algo_name="PYDOE_LHS", n_samples=10)

    for sample in problem.database.get_x_vect_history():
        assert int(sample[-1]) == sample[-1]


def get_expected_nsamples(
    algo: str,
    dim: int,
    n_samples: int | None = None,
) -> int | None:
    """Returns the expected number of samples.

    This number depends on the dimension of the problem.

    Args:
       algo: The name of the DOE algorithm.
       dim: The dimension of the variables space.
       n_samples: The number of samples.
           If None, deduce it from the dimension of the variables space.

    Returns:
        The expected number of samples.
    """
    if algo == "PYDOE_FF2N":
        return 2**dim
    if algo == "PYDOE_BBDESIGN" and dim == 5:
        return 46
    if algo == "PYDOE_PBDESIGN":
        if dim == 1:
            return 4
        if dim == 5:
            return 8
    if algo == "PYDOE_CCDESIGN" and dim == 5:
        return 50
    if algo == "PYDOE_FULLFACT":
        return None
    return n_samples


def get_options(
    algo_name: str,
    dim: int,
) -> dict[str, Any]:
    """Returns the options of the algorithms.

    Args:
        algo_name: The name of the DOE algorithm.
        dim: The dimension of the variables spaces.:param algo_name: param dim:

    Returns:
        The options of the DOE algorithm.
    """
    if algo_name in ["PYDOE_LHS", "PYDOE_FULLFACT"]:
        return {"n_samples": 13}
    return {}


@pytest.mark.parametrize(
    "test_method",
    generate_test_functions(DOE_LIB_NAME, get_expected_nsamples, get_options),
)
def test_methods(test_method) -> None:
    """Apply the tests generated by the."""
    test_method()
