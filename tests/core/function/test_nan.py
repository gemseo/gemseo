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

"""Tests for the check stopping an evaluation on a NaN."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import nan
from scipy.sparse import csr_array

from gemseo.core.function._nan import check_for_nan
from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.util.testing.helper import assert_exception


def test_nan_input(snapshot) -> None:
    """Check the error raised for an input value holding a NaN."""
    with assert_exception(DesvarIsNan, snapshot):
        check_for_nan(array([nan]))


def test_nan_output(snapshot) -> None:
    """Check the error raised for an output value holding a NaN."""
    with assert_exception(FunctionIsNan, snapshot):
        check_for_nan(array([nan]), True, "f", array([1.0]))


def test_nan_output_tolerated() -> None:
    """Check that a NaN is accepted when the evaluation does not stop on one."""
    check_for_nan(array([nan]), False, "f", array([1.0]))


@pytest.mark.parametrize(
    "value", [array(["some_string"]), array("some_string")], ids=["1d", "0d"]
)
@pytest.mark.parametrize("function_name", ["", "f"])
def test_string_array(value, function_name) -> None:
    """Check that a string array is ignored.

    A string-valued observable reaches this check like any other function,
    and `isnan` raises a `TypeError` on such an array.
    """
    check_for_nan(value, True, function_name, array([1.0]))


def test_sparse_jacobian() -> None:
    """Check that a sparse matrix is ignored.

    A driver declaring it supports a sparse Jacobian reads one itself,
    and `isnan` raises a `TypeError` on it.
    """
    check_for_nan(csr_array(array([[1.0, nan]])), True, "f", array([1.0]))
