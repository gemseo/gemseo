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
"""Tests for the Variables collaborator."""

from __future__ import annotations

from collections.abc import MutableMapping

import pytest
from numpy.testing import assert_array_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space._variable import DataType
from gemseo.space._variable import IntegerVariable
from gemseo.space.design._variables import Variables


@pytest.fixture
def variables() -> Variables:
    """A variables with a float and an integer variable."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=1.0)
    variables["n"] = IntegerVariable(size=1, lower_bound=0, upper_bound=10)
    return variables


def test_enable_integer_variables_normalization(variables) -> None:
    """Check the toggling of the integer variables normalization."""
    version = variables.version
    assert not variables.name_to_normalization_mask["n"].any()

    variables.enable_integer_variables_normalization = False
    assert variables.version == version

    variables.enable_integer_variables_normalization = True
    assert variables.version == version + 1
    assert variables.name_to_normalization_mask["n"].all()

    variables.enable_integer_variables_normalization = True
    assert variables.version == version + 1


@pytest.mark.parametrize(
    ("names", "expected"),
    [((), False), (("x",), False), (("n",), True), (("x", "n"), True)],
)
def test_has_integer(names, expected) -> None:
    """Check the detection of integer variables."""
    variables = Variables()
    classes = {"x": ContinuousVariable, "n": IntegerVariable}
    for name in names:
        variables[name] = classes[name](size=1, lower_bound=0, upper_bound=1)
    assert variables.has_integer_variables is expected


def test_get_integer_components(variables) -> None:
    """Check the integer-component mask."""
    assert_array_equal(variables.get_integer_mask(), [False, False, True])


def test_mapping_interface(variables) -> None:
    """Check that the variables reads as a name-to-variable mapping."""
    assert isinstance(variables, MutableMapping)
    assert list(variables) == ["x", "n"]
    assert len(variables) == 2
    assert "x" in variables
    assert "missing" not in variables
    assert list(variables.keys()) == ["x", "n"]
    assert [variable.size for variable in variables.values()] == [2, 1]
    assert dict(variables.items()).keys() == {"x", "n"}
    assert variables["x"].type == DataType.FLOAT
    assert variables.get("missing") is None


def test_getitem_unknown_variable(variables) -> None:
    """Check that indexing an unknown variable raises."""
    with pytest.raises(KeyError):
        variables["missing"]


def test_setitem_insert(variables) -> None:
    """Check that setting a new name appends it and allocates its indices."""
    version = variables.version
    variables["z"] = ContinuousVariable(size=3, lower_bound=0.0, upper_bound=1.0)
    assert list(variables) == ["x", "n", "z"]
    assert variables.size == 6
    assert variables.name_to_indices["z"] == range(3, 6)
    assert variables.version == version + 1


def test_setitem_replace_same_size(variables) -> None:
    """Check that replacing keeps position, size and index ranges."""
    variables["x"] = ContinuousVariable(size=2, lower_bound=-1.0, upper_bound=2.0)
    assert list(variables) == ["x", "n"]
    assert variables.size == 3
    assert variables.name_to_indices["x"] == range(2)
    assert variables.name_to_indices["n"] == range(2, 3)
    assert_array_equal(variables["x"].lower_bound, [-1.0, -1.0])


def test_setitem_replace_resize(variables) -> None:
    """Check that replacing with a different size rebuilds indices and size."""
    variables["x"] = ContinuousVariable(size=4, lower_bound=0.0, upper_bound=1.0)
    assert variables.size == 5
    assert variables.name_to_indices["x"] == range(4)
    assert variables.name_to_indices["n"] == range(4, 5)


def test_delitem(variables) -> None:
    """Check that deleting a variable removes it and rebuilds indices."""
    version = variables.version
    del variables["x"]
    assert list(variables) == ["n"]
    assert variables.size == 1
    assert variables.name_to_indices["n"] == range(1)
    assert "x" not in variables.name_to_normalization_mask
    assert variables.version == version + 1


def test_delitem_unknown_variable(variables) -> None:
    """Check that deleting an unknown variable raises."""
    with pytest.raises(KeyError):
        del variables["missing"]


def test_filter_components(variables) -> None:
    """Check that filtering the components preserves the kind of the variable."""
    version = variables.version
    variables["x"] = ContinuousVariable(
        size=3, lower_bound=[0.0, 1.0, 2.0], upper_bound=[3.0, 4.0, 5.0]
    )
    variables.filter_components("x", [0, 2])
    variable = variables["x"]
    assert isinstance(variable, ContinuousVariable)
    assert variable.size == 2
    assert_array_equal(variable.lower_bound, [0.0, 2.0])
    assert_array_equal(variable.upper_bound, [3.0, 5.0])
    assert variables.name_to_indices["n"] == range(2, 3)
    assert variables.version > version


def test_filter_components_custom_variable(variables) -> None:
    """Check that filtering the components works for a variable class of one's own."""

    class MyVariable(ContinuousVariable):
        """A variable class that the factory cannot discover."""

    variables["x"] = MyVariable(size=2, lower_bound=0.0, upper_bound=1.0)
    variables.filter_components("x", [1])
    assert isinstance(variables["x"], MyVariable)
    assert variables["x"].size == 1
