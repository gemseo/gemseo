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

import pytest
from numpy import array
from numpy import dtype
from numpy import float64
from numpy import int64
from numpy.testing import assert_array_equal

from gemseo.space._design.bounds import Bounds
from gemseo.space._design.integer_rounder import IntegerRounder
from gemseo.space._design.normalizer import Normalizer
from gemseo.space._design.variables import DesignVariables
from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import IntegerVariable


@pytest.fixture
def normalizer() -> Normalizer:
    """A normalizer over a single float variable with bounds [0, 2]."""
    variables = DesignVariables()
    variables["x"] = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=2.0)
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
