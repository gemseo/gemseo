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
"""Tests for the codec module."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import complex128
from numpy import float64
from numpy.testing import assert_array_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space.design._codec import concatenate_values
from gemseo.space.design._codec import split_full_value
from gemseo.space.design._variables import Variables


@pytest.fixture
def variables() -> Variables:
    """A variables with a float variable of size 2 and one of size 3."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=2)
    variables["y"] = ContinuousVariable(size=3)
    return variables


def test_split_full_value(variables: Variables) -> None:
    """Check that a full value is split by variable name."""
    name_to_value = split_full_value(array([1.0, 2.0, 3.0, 4.0, 5.0]), variables)
    assert_array_equal(name_to_value["x"], [1.0, 2.0])
    assert_array_equal(name_to_value["y"], [3.0, 4.0, 5.0])


def test_split_full_value_empty_variables() -> None:
    """Check that splitting against an empty variables returns an empty dict."""
    assert split_full_value(array([]), Variables()) == {}


def test_concatenate_values_empty_names() -> None:
    """Check that concatenating with no names returns an empty array."""
    assert_array_equal(concatenate_values({}, ()), array([]))


def test_concatenate_values_order() -> None:
    """Check that the values are concatenated in the order of `names`."""
    name_to_value = {"x": array([1.0, 2.0]), "y": array([3.0, 4.0, 5.0])}
    full_value = concatenate_values(name_to_value, ("y", "x"))
    assert_array_equal(full_value, [3.0, 4.0, 5.0, 1.0, 2.0])


def test_concatenate_values_dtype_promotion_int_float() -> None:
    """Check that mixing int and float values promotes the result to float64."""
    name_to_value = {"x": array([1, 2]), "y": array([3.0, 4.0, 5.0])}
    full_value = concatenate_values(name_to_value, ("x", "y"))
    assert full_value.dtype == float64


def test_concatenate_values_dtype_promotion_complex() -> None:
    """Check that a complex value promotes the result to complex128."""
    name_to_value = {"x": array([1.0 + 0j, 2.0 + 0j]), "y": array([3.0, 4.0, 5.0])}
    full_value = concatenate_values(name_to_value, ("x", "y"))
    assert full_value.dtype == complex128


def test_split_concatenate_round_trip(variables: Variables) -> None:
    """Check that concatenating then splitting recovers the original values."""
    name_to_value = {"x": array([1.0, 2.0]), "y": array([3.0, 4.0, 5.0])}
    full_value = concatenate_values(name_to_value, variables)
    split_value = split_full_value(full_value, variables)
    assert_array_equal(split_value["x"], name_to_value["x"])
    assert_array_equal(split_value["y"], name_to_value["y"])
