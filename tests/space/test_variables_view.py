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
"""Tests for the read-only view over the variables of a space."""

from __future__ import annotations

import pickle
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import inf
from numpy.testing import assert_array_equal

from gemseo.space.design import DesignSpace
from gemseo.space.design._variables import UnknownVariableError
from gemseo.space.variables_view import VariablesView
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def space() -> DesignSpace:
    """A design space with a single scalar variable."""
    space = DesignSpace()
    space.add_variable("x")
    return space


def test_variables_view(space) -> None:
    """Check that the view gives access to the variables of the space."""
    view = space.variables
    assert isinstance(view, VariablesView)
    assert isinstance(view, ReadOnlyMapping)
    assert view is space.variables
    assert list(view) == ["x"]
    assert len(view) == 1
    assert "x" in view
    assert view["x"] is space._variables["x"]
    assert list(view.items()) == [("x", space._variables["x"])]
    assert repr(view).startswith("VariablesView(")


def test_variables_view_name_to_indices(space) -> None:
    """Check that the view gives access to the indices of the variables."""
    space.add_variable("y")
    name_to_indices = space.variables.name_to_indices
    assert name_to_indices is space._variables.name_to_indices
    assert name_to_indices is space.name_to_indices
    assert dict(name_to_indices) == {"x": range(1), "y": range(1, 2)}
    with pytest.raises(TypeError, match="does not support item assignment"):
        name_to_indices["x"] = range(2)


def test_variables_view_has_integer_variables(space) -> None:
    """Check that the view tells whether a variable is of integer type."""
    assert not space.variables.has_integer_variables
    assert not space.has_integer_variables

    space.add_variable("n", type_=DesignSpace.DesignVariableType.INTEGER)
    assert space.variables.has_integer_variables
    assert space.has_integer_variables


def test_variables_view_is_live(space) -> None:
    """Check that the view reflects the mutations of the space."""
    space.add_variable("y")
    assert list(space.variables) == ["x", "y"]
    space.rename_variable("y", "z")
    assert list(space.variables) == ["x", "z"]
    space.remove_variable("x")
    assert list(space.variables) == ["z"]


@pytest.mark.parametrize(
    ("bound_name", "stale_bound"), [("lower_bound", -inf), ("upper_bound", inf)]
)
def test_variables_view_replaces_a_changed_variable(
    space, bound_name, stale_bound: float
) -> None:
    """Check that changing a variable replaces it instead of mutating it."""
    variable = space.variables["x"]

    getattr(space, f"set_{bound_name}")("x", 1.0)

    assert space.variables["x"] is not variable
    assert_array_equal(getattr(space.variables["x"], bound_name), array([1.0]))
    assert_array_equal(getattr(variable, bound_name), array([stale_bound]))


def test_variables_view_unknown_name(space, snapshot) -> None:
    """Check the error raised when reading a variable that does not exist."""
    with assert_exception(UnknownVariableError, snapshot):
        space.variables["y"]


def test_variables_view_forbids_item_assignment(space) -> None:
    """Check that a variable cannot be replaced through the view."""
    with pytest.raises(TypeError, match="does not support item assignment"):
        space.variables["x"] = space._variables["x"]


def test_variables_view_forbids_item_deletion(space) -> None:
    """Check that a variable cannot be deleted through the view."""
    with pytest.raises(TypeError, match="does not support item deletion"):
        del space.variables["x"]


@pytest.mark.parametrize(
    "method_name",
    [
        "pop",
        "popitem",
        "clear",
        "update",
        "setdefault",
        "rename",
        "filter_components",
        "bump_version",
    ],
)
def test_variables_view_has_no_mutator(space, method_name) -> None:
    """Check that the view exposes no method mutating the registry."""
    assert not hasattr(space.variables, method_name)


def test_variables_view_cannot_be_replaced(space) -> None:
    """Check that the view of a space cannot be rebound."""
    with pytest.raises(AttributeError):
        space.variables = None


def copy_with_pickle(space: DesignSpace) -> DesignSpace:
    """Copy a space of variables through a pickle round trip.

    Args:
        space: The space of variables.

    Returns:
        The copy of the space of variables.
    """
    return pickle.loads(pickle.dumps(space))


@pytest.mark.parametrize("copy_space", [deepcopy, copy_with_pickle])
def test_variables_view_after_copy(
    space, copy_space: Callable[[DesignSpace], DesignSpace]
) -> None:
    """Check that the view of a copied space views the registry of this copy."""
    other_space = copy_space(space)
    assert other_space.variables is not space.variables
    other_space.add_variable("y")
    assert list(other_space.variables) == ["x", "y"]
    assert list(space.variables) == ["x"]


@pytest.mark.parametrize("bound_name", ["lower_bound", "upper_bound"])
def test_variables_view_gives_immutable_bounds(space, bound_name, snapshot) -> None:
    """Check that a bound of a variable read through the view cannot be unfrozen.

    A variable stores read-only views of its bounds,
    which do not own their data,
    so NumPy refuses to make them writeable again;
    freezing the arrays alone would not be enough,
    since they own their data.
    """
    bound = getattr(space.variables["x"], bound_name)

    assert not bound.flags.writeable

    with assert_exception(ValueError, snapshot):
        bound.setflags(write=True)


@pytest.mark.parametrize("copy_space", [deepcopy, copy_with_pickle])
@pytest.mark.parametrize("bound_name", ["lower_bound", "upper_bound"])
def test_variables_view_gives_immutable_bounds_after_copy(
    space, bound_name, copy_space: Callable[[DesignSpace], DesignSpace], snapshot
) -> None:
    """Check that the bounds of a copied space are immutable through its view."""
    bound = getattr(copy_space(space).variables["x"], bound_name)

    assert not bound.flags.writeable

    with assert_exception(ValueError, snapshot):
        bound.setflags(write=True)
