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
"""Tests for the DesignVariables collaborator."""

from __future__ import annotations

from copy import deepcopy
from pickle import dumps
from pickle import loads

import pytest
from numpy import bool_
from numpy import int8
from numpy import ndarray
from numpy.testing import assert_array_equal

from gemseo.space._design.variables import DesignVariables
from gemseo.space.variable import IntegerVariable
from gemseo.space.variable import RealVariable
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def variables() -> DesignVariables:
    """A variables with a real and an integer variable."""
    variables = DesignVariables()
    variables["x"] = RealVariable(size=2, lower_bound=0.0, upper_bound=1.0)
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


@pytest.mark.parametrize("name", ["x", "n"])
def test_normalization_mask_is_read_only(variables, name, snapshot) -> None:
    """Check that a normalization mask cannot be mutated in place."""
    mask = variables.name_to_normalization_mask[name]
    assert not mask.flags.writeable
    with assert_exception(ValueError, snapshot):
        mask[0] = True

    with assert_exception(ValueError, snapshot):
        mask.setflags(write=True)


@pytest.mark.parametrize("name", ["x", "n"])
def test_normalization_mask_cannot_be_thawed_through_its_base(variables, name) -> None:
    """Check that no array reachable from a normalization mask is writeable."""
    mask = variables.name_to_normalization_mask[name]

    array_ = mask
    while isinstance(array_, ndarray):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            array_.setflags(write=True)

        array_ = array_.base

    assert isinstance(array_, bytes)
    assert_array_equal(variables.name_to_normalization_mask[name], mask)


@pytest.mark.parametrize("name", ["x", "n"])
@pytest.mark.parametrize(
    ("attribute_name", "value"),
    [("shape", (1, 1)), ("strides", (0,)), ("dtype", int8)],
)
def test_reassigning_mask_attributes_leaves_the_registry_alone(
    variables, name, attribute_name, value
) -> None:
    """Check that a normalization mask is handed out as a view of the mask stored.

    NumPy lets a caller reassign the shape, the strides and the data type
    of a read-only array, so the mapping returns a view of the frozen mask
    and such a reassignment reaches the view only.
    """
    mask = variables.name_to_normalization_mask[name]
    if attribute_name == "shape":
        value = (mask.size, 1)

    setattr(mask, attribute_name, value)

    stored_mask = variables.name_to_normalization_mask[name]
    assert stored_mask.shape == (variables[name].size,)
    assert stored_mask.strides == (1,)
    assert stored_mask.dtype == bool_


@pytest.mark.parametrize("copy_", [deepcopy, lambda obj: loads(dumps(obj))])
@pytest.mark.parametrize("name", ["x", "n"])
def test_normalization_mask_is_read_only_after_copy(
    variables, copy_, name, snapshot
) -> None:
    """Check that a copy of the registry hands out read-only normalization masks.

    Neither pickling nor copying preserves the writeable flag of an array
    nor the immutable buffer that a frozen array is a view of,
    so the registry freezes the masks again when its state is restored.
    """
    copied_variables = copy_(variables)
    mask = copied_variables.name_to_normalization_mask[name]

    assert not mask.flags.writeable
    with assert_exception(ValueError, snapshot):
        mask[0] = True

    assert_array_equal(mask, variables.name_to_normalization_mask[name])


def test_setitem_normalization_mask(variables) -> None:
    """Check that setting a variable computes its normalization mask."""
    version = variables.version
    variables["z"] = RealVariable(size=2, lower_bound=0.0, upper_bound=1.0)
    assert variables.name_to_normalization_mask["z"].all()
    assert variables.version == version + 1


def test_delitem_normalization_mask(variables) -> None:
    """Check that deleting a variable drops its normalization mask."""
    version = variables.version
    del variables["x"]
    assert "x" not in variables.name_to_normalization_mask
    assert variables.version == version + 1


def test_rename_normalization_mask(variables) -> None:
    """Check that renaming a variable renames its normalization mask."""
    version = variables.version
    variables.rename("x", "y")
    assert "x" not in variables.name_to_normalization_mask
    assert variables.name_to_normalization_mask["y"].all()
    assert variables.version == version + 1


def test_rename_collision(variables, snapshot) -> None:
    """Check that renaming to an existing name raises and keeps the mask intact."""
    version = variables.version
    with assert_exception(ValueError, snapshot):
        variables.rename("x", "n")
    assert list(variables) == ["x", "n"]
    assert "x" in variables.name_to_normalization_mask
    assert "n" in variables.name_to_normalization_mask
    assert variables.version == version


def test_rename_same_name_is_noop(variables) -> None:
    """Check that renaming a variable to its own name is a no-op."""
    version = variables.version
    mask = variables.name_to_normalization_mask["x"]
    variables.rename("x", "x")
    assert list(variables) == ["x", "n"]
    assert_array_equal(variables.name_to_normalization_mask["x"], mask)
    assert variables.version == version + 1


def test_filter_components_normalization_mask(variables) -> None:
    """Check that filtering the components updates the normalization mask."""
    version = variables.version
    variables.filter_components("x", [1])
    assert len(variables.name_to_normalization_mask["x"]) == 1
    assert variables.version == version + 1
