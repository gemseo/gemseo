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
#      :author: Damien Guenot - 28 avr. 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from logging import ERROR
from pathlib import Path
from typing import Any

import pytest
from numpy import array
from numpy.testing import assert_equal
from pandas.errors import ParserError
from pydantic import ValidationError

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.factory import DOELibraryFactory

from .utils import execute_problem
from .utils import generate_test_functions
from .utils import get_problem

DOE_LIB_NAME = "CustomDOE"
DOE_FILE_PATH = str(Path(__file__).parent / "dim_3_semicolon.csv")


def test_library_from_factory():
    """Check that the DOELibraryFactory can create the CustomDOE library."""
    factory = DOELibraryFactory()
    if factory.is_available(DOE_LIB_NAME):
        factory.create(DOE_LIB_NAME)


def test_check_dimension_inconsistency():
    """Check that an error is raised if the dimensions are inconsistent."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Dimension mismatch between the variables space (4) and the samples (3)."
        ),
    ):
        execute_problem(
            DOE_LIB_NAME,
            dim=4,
            delimiter=";",
            doe_file=DOE_FILE_PATH,
        )


def test_read_file_error():
    """Check that an error is raised when the separator is wrong."""
    with pytest.raises(ValueError):
        execute_problem(
            DOE_LIB_NAME,
            dim=4,
            doe_file=DOE_FILE_PATH,
        )


@pytest.mark.parametrize(
    ("n_samples", "options"),
    [
        (2, {"samples": array([[1.0, 2.0, 1.0], [1.0, 2.0, 0.0]])}),
        (2, {"samples": {"x": array([[1.0, 2.0, 1.0], [1.0, 2.0, 0.0]])}}),
        (
            2,
            {"samples": [{"x": array([1.0, 2.0, 1.0])}, {"x": array([1.0, 2.0, 0.0])}]},
        ),
        (
            30,
            {
                "delimiter": ";",
                "doe_file": DOE_FILE_PATH,
            },
        ),
        (
            30,
            {
                "delimiter": ";",
                "doe_file": Path(DOE_FILE_PATH),
            },
        ),
    ],
)
def test_samples_shape(n_samples, options):
    """Check that the samples shape is correct."""
    doe_library = execute_problem(DOE_LIB_NAME, dim=3, **options)
    assert doe_library.unit_samples.shape == (n_samples, 3)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "doe_file": "foo.txt",
            "samples": array([[1.0, 2, 3.0], [1.0, 2.0, 3.0]]),
        },
    ],
)
def test_wrong_arguments(options):
    """Check that an error is raised when arguments are wrong."""
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "The algorithm CustomDOE requires "
            "either a 'doe_file' or the input 'samples' as settings."
        ),
    ):
        execute_problem(DOE_LIB_NAME, dim=3, **options)


def get_expected_nsamples(
    algo: str,
    dim: int,
    n_samples: int | None = None,
) -> int:
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
    if dim == 1:
        return 9
    if dim == 5:
        return 2
    return None


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
    return {
        "doe_file": str(Path(__file__).parent / f"dim_{dim}.csv"),
    }


@pytest.mark.parametrize(
    "test_method",
    generate_test_functions("CustomDOE", get_expected_nsamples, get_options),
)
def test_methods(test_method):
    """Apply the tests generated by the."""
    test_method()


def test_use_custom_doe_directly():
    """Check the use of CustomDOE without setting algo_name."""
    problem = get_problem(2)
    CustomDOE().execute(problem, samples=array([[0.0, 0.0]]))
    assert len(problem.database) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": array([[1.0, 2.0]])},
        {"settings_model": CustomDOE_Settings(samples=array([[1.0, 2.0]]))},
    ],
)
def test_compute_doe(kwargs):
    """Check that CustomDOE.compute_doe works."""
    variables_space = DesignSpace()
    variables_space.add_variable("x", size=2)
    assert_equal(
        CustomDOE().compute_doe(variables_space, **kwargs), array([[1.0, 2.0]])
    )


def test_malformed_file(caplog):
    """Check that an error message is logged when reading a file raises an exception."""
    with pytest.raises(ParserError):
        CustomDOE().compute_doe(3, doe_file=Path(__file__).parent / "malformed_doe.csv")

    _, level, message = caplog.record_tuples[0]
    assert level == ERROR
    assert re.match(r"Failed to load the DOE file .+malformed_doe\.csv", message)
