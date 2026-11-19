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
"""Tests for the Normalizer collaborator."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import dtype
from numpy import float64
from numpy import int64
from numpy.testing import assert_array_equal

from gemseo.space._design.bounds import Bounds
from gemseo.space._design.integer_rounder import IntegerRounder
from gemseo.space._design.normalizer import Normalizer
from gemseo.space._design.variables import DesignVariables
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import IntegerVariable
from gemseo.space.variable import RealVariable

if TYPE_CHECKING:
    from gemseo.space.variable.deterministic import BaseDeterministicVariable


@pytest.fixture
def normalizer() -> Normalizer:
    """A normalizer over a single real variable with bounds [0, 2]."""
    variables = DesignVariables()
    variables["x"] = RealVariable(size=2, lower_bound=0.0, upper_bound=2.0)
    return Normalizer(
        variables,
        Bounds(variables),
        IntegerRounder(variables),
    )


@pytest.fixture
def integer_normalizer() -> Normalizer:
    """A normalizer over a single normalized integer variable with bounds [0, 10]."""
    variables = DesignVariables()
    variables["n"] = IntegerVariable(size=2, lower_bound=0, upper_bound=10)
    variables.enable_integer_variables_normalization = True
    return Normalizer(
        variables,
        Bounds(variables),
        IntegerRounder(variables),
    )


def test_normalize_integer_common_dtype(normalizer) -> None:
    """Check that an integer common dtype is promoted to float."""
    normalized = normalizer.normalize(array([1, 2]), dtype("int64"))
    assert normalized.dtype == float64
    assert_array_equal(normalized, [0.5, 1.0])


def test_denormalize_with_integer_dtype(integer_normalizer) -> None:
    """Check denormalization with an integer common dtype.

    The normalized value is scaled in an intermediate float array
    before being rounded,
    so that the integer dtype does not truncate it beforehand.
    """
    denormalized = integer_normalizer.denormalize(array([0.4, 0.6]), dtype("int64"))
    assert denormalized.dtype == int64
    assert_array_equal(denormalized, [4, 6])


def test_denormalize_keeps_imaginary_part(normalizer) -> None:
    """Check that denormalization keeps the imaginary part of a complex full value.

    The imaginary part carries the perturbation
    of the complex-step differentiation,
    so dropping it would zero out the approximated derivatives.
    """
    denormalized = normalizer.denormalize(
        array([0.5 + 1e-8j, 1.0 + 1e-8j]), dtype("complex128")
    )
    assert_array_equal(denormalized.imag, [2e-8, 2e-8])


def test_denormalize_keeps_imaginary_part_with_a_real_common_dtype(normalizer) -> None:
    """Check that denormalization keeps the imaginary part with a real common dtype.

    The common dtype is taken from the current value of the design space,
    which is real even when a complex-step perturbation is being
    denormalized, so casting to that common dtype must not drop the
    imaginary part.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        denormalized = normalizer.denormalize(
            array([0.5 + 1e-8j, 1.0 + 1e-8j]), dtype("float64")
        )

    assert_array_equal(denormalized.imag, [2e-8, 2e-8])


@pytest.fixture
def discrete_normalizer() -> Normalizer:
    """A normalizer over a normalized discrete variable with choices 10, 20, 30."""
    return create_normalizer(d=DiscreteVariable(choices=[10, 20, 30]))


def create_normalizer(**variables_: BaseDeterministicVariable) -> Normalizer:
    """Create a normalizer over variables normalized like the integer ones.

    Args:
        **variables_: The variables.

    Returns:
        The normalizer.
    """
    variables = DesignVariables()
    for name, variable in variables_.items():
        variables[name] = variable

    variables.enable_integer_variables_normalization = True
    return Normalizer(variables, Bounds(variables), IntegerRounder(variables))


@pytest.mark.parametrize(
    ("unit_value", "choice"),
    [
        (0.0, 10.0),
        (1 / 3 - 1e-9, 10.0),
        (1 / 3, 20.0),
        (0.5, 20.0),
        (2 / 3, 30.0),
        (1.0, 30.0),
    ],
    ids=[
        "lower-end",
        "before-cell-border",
        "cell-border",
        "middle",
        "last-cell-border",
        "upper-end",
    ],
)
def test_discrete_cells(discrete_normalizer, unit_value, choice) -> None:
    """Check that a unit value is mapped to the choice owning its cell."""
    denormalized = discrete_normalizer.denormalize(
        array([unit_value]), dtype("float64")
    )
    assert denormalized.dtype == float64
    assert_array_equal(denormalized, [choice])


@pytest.mark.parametrize(
    ("choices", "choice"),
    [([1, 2, 100], 1), ([1, 2, 100], 2), ([1, 2, 100], 100), ([3], 3)],
    ids=["uneven-first", "uneven-middle", "uneven-last", "single"],
)
def test_discrete_round_trip(choices, choice) -> None:
    """Check that a choice is recovered by normalizing then denormalizing it."""
    normalizer = create_normalizer(d=DiscreteVariable(choices=choices))
    normalized = normalizer.normalize(array([float(choice)]), dtype("float64"))
    assert 0.0 <= normalized[0] <= 1.0
    assert_array_equal(normalizer.denormalize(normalized, dtype("float64")), [choice])


@pytest.mark.parametrize(
    ("choices", "chances"),
    [
        ([10, 20, 30], [1 / 3, 1 / 3, 1 / 3]),
        ([1, 2, 100], [1 / 3, 1 / 3, 1 / 3]),
    ],
    ids=["even", "uneven"],
)
def test_discrete_chances(choices, chances) -> None:
    """Check that the choices have the same chance whatever their spacing."""
    normalizer = create_normalizer(d=DiscreteVariable(choices=choices))
    n = 300_000
    samples = normalizer.denormalize(
        ((arange(n) + 0.5) / n).reshape(-1, 1), dtype("float64")
    )
    frequencies = [(samples == choice).mean() for choice in choices]
    assert frequencies == pytest.approx(chances, abs=1e-5)


@pytest.mark.parametrize(
    ("method_name", "keyword"),
    [("normalize", "subtract_lower_bound"), ("denormalize", "add_lower_bound")],
)
def test_discrete_gradient_is_left_as_it_is(
    discrete_normalizer, method_name, keyword
) -> None:
    """Check that the discrete components of a gradient are left as they are."""
    gradient = getattr(discrete_normalizer, method_name)(
        array([0.3]), dtype("float64"), **{keyword: False}
    )
    assert_array_equal(gradient, [0.3])


@pytest.mark.parametrize(
    ("choice", "unit_value"),
    [(10.0, 1 / 6), (20.0, 0.5), (30.0, 5 / 6), (24.0, 0.5), (26.0, 5 / 6)],
    ids=["first", "middle", "last", "nearest-lower", "nearest-upper"],
)
def test_discrete_normalization(discrete_normalizer, choice, unit_value) -> None:
    """Check that a value is mapped to the centre of the cell of its nearest choice."""
    normalized = discrete_normalizer.normalize(array([choice]), dtype("float64"))
    assert_array_equal(normalized, [unit_value])


def test_discrete_with_real_stays_float() -> None:
    """Check that a space with real and discrete variables only stays float."""
    normalizer = create_normalizer(
        x=RealVariable(lower_bound=0.0, upper_bound=2.0),
        d=DiscreteVariable(choices=[10, 20, 30]),
    )
    denormalized = normalizer.denormalize(array([0.25, 0.5]), dtype("int64"))
    assert denormalized.dtype == float64
    assert_array_equal(denormalized, [0.5, 20.0])
