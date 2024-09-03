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
"""Tests for Functions."""

from __future__ import annotations

import re
from collections.abc import MutableSequence

import pytest
from numpy import array

from gemseo.core.mdo_functions.collections.functions import Functions
from gemseo.core.mdo_functions.mdo_function import MDOFunction


@pytest.fixture(scope="module")
def mdo_functions() -> list[MDOFunction]:
    """Some MDOFunction objects."""
    return [MDOFunction(lambda x: x, f"f{i}") for i in range(3)]


@pytest.fixture
def functions(problem) -> Functions:
    """A set of functions."""
    return Functions()


def test_is_mutable_sequence(functions):
    """Check that Functions is a mutable sequence."""
    assert isinstance(functions, MutableSequence)


def test_is_initially_empty(functions):
    """Check that Functions is initially empty."""
    assert not functions


def test_set_insert_get_del(functions, problem, mdo_functions):
    """Check the methods __setitem__, __getitem__, __delitem__, __len__ and insert."""
    function_0 = mdo_functions[0]
    functions.append(function_0)
    assert functions[0] == function_0
    assert len(functions) == 1
    function_1 = mdo_functions[1]
    functions.insert(0, function_1)
    assert len(functions) == 2
    assert functions[0] == function_1
    del functions[0]
    assert functions[0] == function_0
    assert len(functions) == 1


def test_format(functions):
    """Check that the method format does nothing."""
    assert functions.format("a") == "a"


def test_names(functions, mdo_functions):
    """Check the property names."""
    functions.extend(mdo_functions)
    assert functions.get_names() == [
        mdo_function.name for mdo_function in mdo_functions
    ]


def test_original_reset(functions, mdo_functions):
    """Check the property original and the method reset."""
    functions.extend(mdo_functions)
    assert list(functions.get_originals()) == mdo_functions
    original_mdo_functions = [
        MDOFunction(mdo_function, f"g{i}")
        for i, mdo_function in enumerate(mdo_functions)
    ]
    for mdo_function, original_mdo_function in zip(
        mdo_functions, original_mdo_functions
    ):
        mdo_function.original = original_mdo_function

    assert list(functions) == mdo_functions
    assert list(functions.get_originals()) == original_mdo_functions

    functions.reset()
    assert list(functions) == original_mdo_functions
    assert list(functions.get_originals()) == original_mdo_functions


def test_dimension(functions, mdo_functions):
    """Check the property dimension."""
    assert functions.dimension == 0
    functions.extend(mdo_functions)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The function output dimension is not available yet, "
            "please call function f0 once."
        ),
    ):
        assert functions.dimension == 3

    for mdo_function in mdo_functions:
        mdo_function.evaluate(array([1]))

    assert functions.dimension == 3


def test_f_types(functions):
    """Check _F_TYPES."""
    f_types = functions._F_TYPES = ("foo",)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The function type 'bar' is not one of those authorized (foo)."
        ),
    ):
        functions.append(MDOFunction(lambda x: x, "f", f_type="bar"))

    functions._F_TYPES = f_types
