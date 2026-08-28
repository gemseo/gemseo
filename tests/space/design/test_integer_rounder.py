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
"""Tests for the IntegerRounder collaborator."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space._variable import IntegerVariable
from gemseo.space.design._integer_rounder import IntegerRounder
from gemseo.space.design._variables import Variables


@pytest.fixture
def variables() -> Variables:
    """A variables with one float variable and one integer variable."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=1, lower_bound=0.0, upper_bound=2.0)
    variables["y"] = IntegerVariable(size=2, lower_bound=0, upper_bound=10)
    return variables


def test_has_integer_true(variables) -> None:
    """Check that has_integer is True when the registry has an integer variable."""
    assert IntegerRounder(variables).has_integer


def test_has_integer_false() -> None:
    """Check that has_integer is False when there is no integer variable."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=1, lower_bound=0.0, upper_bound=2.0)
    assert not IntegerRounder(variables).has_integer


def test_round(variables) -> None:
    """Check that round only rounds the integer components."""
    integer_rounder = IntegerRounder(variables)
    full_value = array([0.6, 1.2, 1.9])
    rounded = integer_rounder.round(full_value)
    assert_array_equal(rounded, [0.6, 1.0, 2.0])
    assert rounded is not full_value


def test_round_no_copy(variables) -> None:
    """Check that round can mutate the input in place."""
    integer_rounder = IntegerRounder(variables)
    full_value = array([0.6, 1.2, 1.9])
    rounded = integer_rounder.round(full_value, copy=False)
    assert rounded is full_value
    assert_array_equal(full_value, [0.6, 1.0, 2.0])


def test_round_no_integer() -> None:
    """Check that round is a no-op when there is no integer variable."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=1, lower_bound=0.0, upper_bound=2.0)
    integer_rounder = IntegerRounder(variables)
    full_value = array([0.6])
    rounded = integer_rounder.round(full_value)
    assert rounded is full_value
