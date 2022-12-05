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

from typing import Any

import pytest
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import array_equal
from numpy import loadtxt
from numpy import ones
from numpy.linalg import norm

from .utils import execute_problem
from .utils import generate_test_functions

DOE_LIB_NAME = "PyDOE"


def test_library_from_factory():
    """Check that the DOEFactory can create the PyDOE library."""
    factory = DOEFactory()
    if factory.is_available(DOE_LIB_NAME):
        factory.create(DOE_LIB_NAME)


def test_export_samples(tmp_wd):
    """Check that samples can be correctly exported."""
    algo_name = "lhs"
    n_samples = 3
    dim = 2
    doe_library = execute_problem(
        DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples
    )
    doe_file_name = f"test_{algo_name}.csv"
    doe_library.export_samples(doe_output_file=doe_file_name)
    file_samples = loadtxt(doe_file_name, delimiter=",")
    assert array_equal(doe_library.unit_samples, file_samples)


def test_invalid_algo():
    """Check that an invalid algorithm name."""
    with pytest.raises(
        KeyError,
        match=("The algorithm unknown_algo is unknown; available ones are: *."),
    ):
        execute_problem(
            DOE_LIB_NAME,
            algo_name="unknown_algo",
            dim=2,
            n_samples=3,
        )


def test_lhs_maximin():
    """Check that an optimal LHS and LHS are different."""
    dim = 3
    algo_name = "lhs"
    n_samples = 100
    doe_library = execute_problem(
        DOE_LIB_NAME,
        algo_name=algo_name,
        dim=dim,
        n_samples=n_samples,
        criterion="maximin",
    )
    samples_maximin = doe_library.unit_samples
    doe_library = execute_problem(
        DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples
    )
    samples = doe_library.unit_samples
    assert norm(samples - samples_maximin) / norm(samples_maximin) >= 1e-8


@pytest.mark.parametrize(
    "algo_name, options",
    [
        ("ccdesign", {"alpha": "unknown_value"}),
        ("ccdesign", {"face": "unknown_value"}),
        ("ccdesign", {"center_cc": 1}),
        ("bbdesign", {"center_bb": (4, 4)}),
        ("lhs", {"criterion": "corr", "iterations": "unknown_value", "n_samples": 2}),
        ("lhs", {"criterion": "unknown_value", "n_samples": 2}),
    ],
)
def test_algo_with_unknown_options(algo_name, options):
    """Check that exceptions are raised when unknown options are passed to an algo."""
    with pytest.raises(ValueError, match=f"Invalid options for algorithm {algo_name}"):
        execute_problem(DOE_LIB_NAME, algo_name=algo_name, dim=3, **options)


def test_missing_algo_name():
    """Check that a DOELibrary cannot produce samples without an algorithm name."""
    with pytest.raises(
        Exception,
        match=(
            "Algorithm name must be either passed as argument "
            "or set by the attribute 'algo_name'."
        ),
    ):
        execute_problem(
            DOE_LIB_NAME,
            dim=2,
            n_samples=3,
        )


def test_export_error():
    """Check that a DOELibrary.export_samples raises an error if there is no samples."""
    doe_library = DOEFactory().create(DOE_LIB_NAME)
    with pytest.raises(
        Exception, match="Samples are missing, execute method before export."
    ):
        doe_library.export_samples("test.csv")


def test_rescale_samples():
    """"""
    algo_name = "lhs"
    dim = 3
    n_samples = 100
    doe_library = execute_problem(
        DOE_LIB_NAME,
        algo_name=algo_name,
        dim=dim,
        n_samples=n_samples,
        criterion="corr",
    )
    samples = ones((10,))
    samples[0] += 1e-15
    doe_library._rescale_samples(samples)


@pytest.mark.parametrize(
    "algo_name,dim,n_samples,options",
    [
        ("ccdesign", 5, 62, {"center_cc": [10, 10]}),
        ("bbdesign", 5, 46, {"center": 1}),
        ("lhs", 2, 3, {"n_samples": 3, "criterion": "corr"}),
        ("lhs", 2, 3, {"n_samples": 3, "criterion": "centermaximin"}),
        ("lhs", 2, 3, {"n_samples": 3, "criterion": "center"}),
        ("lhs", 2, 3, {"n_samples": 3, "criterion": "maximin"}),
    ],
)
def test_algos(algo_name, dim, n_samples, options):
    """Check that the PyDOE library returns samples correctly shaped."""
    doe_library = execute_problem(DOE_LIB_NAME, algo_name=algo_name, dim=dim, **options)
    assert doe_library.unit_samples.shape == (n_samples, dim)


def test_integer_lhs():
    """Check that a DOE with integer variables stores integer values in the Database."""
    problem = Rosenbrock()
    problem.design_space.add_variable("y", var_type="integer", l_b=10.0, u_b=15.0)
    DOEFactory().execute(problem, "lhs", n_samples=10)

    for sample in problem.database.get_x_history():
        assert int(sample[-1]) == sample[-1]


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
    if algo == "ff2n":
        return 2**dim
    if algo == "bbdesign":
        if dim == 5:
            return 46
    if algo == "pbdesign":
        if dim == 1:
            return 4
        if dim == 5:
            return 8
    if algo == "ccdesign":
        if dim == 5:
            return 50
    if algo == "fullfact":
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
    options = {"n_samples": 13}
    return options


@pytest.mark.parametrize(
    "test_method",
    generate_test_functions(DOE_LIB_NAME, get_expected_nsamples, get_options),
)
def test_methods(test_method):
    """Apply the tests generated by the."""
    test_method()


def test_library_name():
    """Check the library name."""
    assert PyDOE.LIBRARY_NAME == "PyDOE"
