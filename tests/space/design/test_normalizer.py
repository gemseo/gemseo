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
from numpy import complex128
from numpy import dtype
from numpy import float64
from numpy import int64
from numpy import zeros
from numpy.testing import assert_array_equal

from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
from gemseo.space.design._bounds import Bounds
from gemseo.space.design._integer_rounder import IntegerRounder
from gemseo.space.design._normalizer import Normalizer
from gemseo.space.design._variables import Variables
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def normalizer() -> Normalizer:
    """A normalizer over a single float variable with bounds [0, 2]."""
    variables = Variables()
    variables["x"] = Variable(
        size=2, type=DataType.FLOAT, lower_bound=0.0, upper_bound=2.0
    )
    return Normalizer(
        variables,
        Bounds(variables),
        IntegerRounder(variables),
    )


@pytest.fixture
def integer_normalizer() -> Normalizer:
    """A normalizer over a single normalized integer variable with bounds [0, 10]."""
    variables = Variables()
    variables["n"] = Variable(
        size=2, type=DataType.INTEGER, lower_bound=0, upper_bound=10
    )
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


def test_denormalize_out_with_integer_dtype(integer_normalizer) -> None:
    """Check denormalization into an integer out array.

    The normalized value is scaled in an intermediate float array
    before being rounded and stored,
    so that the integer array does not truncate it beforehand.
    """
    out = zeros(2, dtype=int64)
    denormalized = integer_normalizer.denormalize(
        array([0.4, 0.6]), dtype("int64"), out=out
    )
    assert denormalized is out
    assert_array_equal(out, [4, 6])


def test_denormalize_out_and_allocated_agree(integer_normalizer) -> None:
    """Check that denormalizing into an out array and allocating one agree."""
    full_value = array([0.4, 0.6])
    allocated = integer_normalizer.denormalize(full_value.copy(), dtype("int64"))
    out = zeros(2, dtype=allocated.dtype)
    integer_normalizer.denormalize(full_value.copy(), dtype("int64"), out=out)
    assert_array_equal(out, allocated)


def test_normalize_out_with_different_dtype(normalizer, snapshot) -> None:
    """Check the error raised when the out array cannot store the normalized value."""
    with assert_exception(ValueError, snapshot):
        normalizer.normalize(
            array([1.0 + 0j, 2.0 + 0j]), dtype("complex128"), out=array([0.0, 0.0])
        )


def test_denormalize_out_with_different_dtype(normalizer, snapshot) -> None:
    """Check the error raised when the out array cannot store the denormalized value."""
    with assert_exception(ValueError, snapshot):
        normalizer.denormalize(
            array([0.5 + 0j, 1.0 + 0j]), dtype("complex128"), out=array([0.0, 0.0])
        )


def test_normalize_out_with_common_dtype(normalizer) -> None:
    """Check normalization into an out array of the dtype of the result."""
    out = zeros(2, dtype=complex128)
    normalized = normalizer.normalize(
        array([1.0 + 0j, 2.0 + 0j]), dtype("complex128"), out=out
    )
    assert normalized is out
    assert_array_equal(out, [0.5 + 0j, 1.0 + 0j])


def test_denormalize_out_with_common_dtype(normalizer) -> None:
    """Check denormalization into an out array of the dtype of the result."""
    out = zeros(2, dtype=complex128)
    denormalized = normalizer.denormalize(
        array([0.5 + 0j, 1.0 + 0j]), dtype("complex128"), out=out
    )
    assert denormalized is out
    assert_array_equal(out, [1.0 + 0j, 2.0 + 0j])


@pytest.mark.parametrize("use_out", [False, True])
def test_denormalize_keeps_imaginary_part(normalizer, use_out) -> None:
    """Check that denormalization keeps the imaginary part of a complex full value.

    The imaginary part carries the perturbation
    of the complex-step differentiation,
    so dropping it would zero out the approximated derivatives.
    """
    out = zeros(2, dtype=complex128) if use_out else None
    denormalized = normalizer.denormalize(
        array([0.5 + 1e-8j, 1.0 + 1e-8j]), dtype("complex128"), out=out
    )
    assert_array_equal(denormalized.imag, [2e-8, 2e-8])


def test_normalize_out_with_wrong_shape(normalizer, snapshot) -> None:
    """Check the error raised when the out array has not the shape of the result."""
    with assert_exception(ValueError, snapshot):
        normalizer.normalize(array([1.0, 2.0]), dtype("float64"), out=zeros((3, 2)))
