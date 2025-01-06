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
#      :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from unittest import mock

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.algos.doe.factory import DOELibraryFactory

from .utils import check_problem_execution
from .utils import execute_problem

DOE_LIB_NAME = "DiagonalDOE"


def test_init() -> None:
    """Check the creation of the library."""
    factory = DOELibraryFactory()
    if factory.is_available(DOE_LIB_NAME):
        factory.create(DOE_LIB_NAME)


def test_invalid_algo() -> None:
    """Check the request of an invalid algorithm."""
    with pytest.raises(
        ValueError,
        match=(
            r"No algorithm named invalid_algo is available; available algorithms are .+"
        ),
    ):
        execute_problem("invalid_algo", n_samples=100)


def test_diagonal_doe() -> None:
    """Check the computation of a diagonal DOE."""
    dim = 3
    n_samples = 10
    doe_library = execute_problem("DiagonalDOE", dim=dim, n_samples=n_samples)
    samples = doe_library.unit_samples
    assert samples.shape == (n_samples, dim)
    assert samples[4, 0] == pytest.approx(0.4, rel=0.0, abs=0.1)


@pytest.mark.parametrize("dimension", [1, 5])
def test_diagonal_doe_on_rosenbrock(dimension) -> None:
    """Check the diagonal DOE on the Rosenbrock problem."""
    assert (
        check_problem_execution(
            dimension,
            DOE_LIB_NAME,
            lambda algo, dim, n_samples: n_samples,
            {"n_samples": 13},
        )
        is None
    )


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = mock.MagicMock()
    design_space.dimension = 2
    design_space.variable_names = ["x", "y"]
    design_space.variable_sizes = {"x": 1, "y": 1}
    design_space.__iter__.return_value = ["x", "y"]

    def side_effect(name: str) -> int:
        return {"x": 1, "y": 1}[name]

    design_space.get_size = mock.Mock(side_effect=side_effect)
    return design_space


def test_compute_doe(variables_space) -> None:
    """Check the computation of a DOE out of a variables space."""
    library = DOELibraryFactory().create(DOE_LIB_NAME)
    doe = library.compute_doe(variables_space, n_samples=3, unit_sampling=True)
    assert_equal(doe, array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))


@pytest.mark.parametrize(
    ("reverse", "samples"),
    [
        (["x"], array([[1.0, 0.0], array([0.5, 0.5]), array([0.0, 1.0])])),
        (["1"], array([[0.0, 1.0], array([0.5, 0.5]), array([1.0, 0.0])])),
    ],
)
def test_reverse(variables_space, reverse, samples) -> None:
    """Check the sampling of variables in reverse order."""
    library = DOELibraryFactory().create(DOE_LIB_NAME)
    doe = library.compute_doe(
        variables_space, n_samples=3, unit_sampling=True, reverse=reverse
    )
    assert_equal(doe, samples)
