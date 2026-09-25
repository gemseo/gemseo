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
"""Tests for the Bounds collaborator."""

from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import float64
from numpy import int32
from numpy import ndarray
from numpy.testing import assert_array_equal

from gemseo.space._design.bounds import Bounds
from gemseo.space._design.variables import DesignVariables
from gemseo.space.variable import RealVariable
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def variables() -> DesignVariables:
    """A variables with a single real variable of size 2, bounds [0, 10]."""
    variables = DesignVariables()
    variables["x"] = RealVariable(size=2, lower_bound=0.0, upper_bound=10.0)
    return variables


def test_full_bounds(variables) -> None:
    """Check the aggregate bound arrays."""
    bounds = Bounds(variables)
    assert_array_equal(bounds.full_lower_bound, [0.0, 0.0])
    assert_array_equal(bounds.full_upper_bound, [10.0, 10.0])


def test_set_lower_bound_invalidates_full_bounds(variables) -> None:
    """Check that setting a lower bound refreshes the full lower bound."""
    bounds = Bounds(variables)
    # Populate the value.
    assert_array_equal(bounds.full_lower_bound, [0.0, 0.0])

    bounds.set_lower_bound("x", array([1.0, 2.0]))

    assert_array_equal(bounds.full_lower_bound, [1.0, 2.0])
    assert_array_equal(bounds.get_lower_bound("x"), [1.0, 2.0])


def test_set_upper_bound_invalidates_full_bounds(variables) -> None:
    """Check that setting an upper bound refreshes the full upper bound."""
    bounds = Bounds(variables)
    # Populate the value.
    assert_array_equal(bounds.full_upper_bound, [10.0, 10.0])

    bounds.set_upper_bound("x", array([5.0, 6.0]))

    assert_array_equal(bounds.full_upper_bound, [5.0, 6.0])
    assert_array_equal(bounds.get_upper_bound("x"), [5.0, 6.0])


def test_set_bound_bumps_version(variables) -> None:
    """Check that setting a bound bumps the variable-registry version."""
    bounds = Bounds(variables)
    version = variables.version
    bounds.set_lower_bound("x", array([1.0, 1.0]))
    assert variables.version > version


BOUND_ACCESSORS = [
    (lambda bounds: bounds.full_lower_bound, None),
    (lambda bounds: bounds.full_upper_bound, None),
    (lambda bounds: bounds.get_lower_bound("x"), None),
    (lambda bounds: bounds.get_upper_bound("x"), None),
    (lambda bounds: bounds.get_lower_bounds(), None),
    (lambda bounds: bounds.get_upper_bounds(), None),
    (lambda bounds: bounds.get_lower_bounds(["x"]), None),
    (lambda bounds: bounds.get_upper_bounds(["x"]), None),
    (lambda bounds: bounds.get_lower_bounds(["x"], as_dict=True), "x"),
    (lambda bounds: bounds.get_upper_bounds(["x"], as_dict=True), "x"),
]
"""The bound accessors of Bounds, with the dictionary key to read, if any."""

BOUND_ACCESSOR_IDS = [
    "full_lower_bound",
    "full_upper_bound",
    "get_lower_bound",
    "get_upper_bound",
    "get_lower_bounds",
    "get_upper_bounds",
    "get_lower_bounds[names]",
    "get_upper_bounds[names]",
    "get_lower_bounds[as_dict]",
    "get_upper_bounds[as_dict]",
]
"""The identifiers of the bound accessors."""


@pytest.mark.parametrize(
    ("get_result", "dict_key"), BOUND_ACCESSORS, ids=BOUND_ACCESSOR_IDS
)
def test_read_only_bounds(variables, get_result, dict_key, snapshot) -> None:
    """Check that every bound accessor of Bounds returns a read-only array."""
    bounds = Bounds(variables)
    bounds.set_lower_bound("x", array([1.0, 1.0]))
    bounds.set_upper_bound("x", array([1.0, 1.0]))

    result = get_result(bounds)
    if dict_key is not None:
        result = result[dict_key]

    with assert_exception(ValueError, snapshot):
        result[0] = 2.0


@pytest.mark.parametrize(
    ("get_result", "dict_key"), BOUND_ACCESSORS, ids=BOUND_ACCESSOR_IDS
)
def test_read_only_bounds_cannot_be_unfrozen(
    variables, get_result, dict_key, snapshot
) -> None:
    """Check that the writeable flag of a bound cannot be re-enabled."""
    bounds = Bounds(variables)

    result = get_result(bounds)
    if dict_key is not None:
        result = result[dict_key]

    with assert_exception(ValueError, snapshot):
        result.setflags(write=True)


@pytest.mark.parametrize(
    ("get_result", "dict_key"), BOUND_ACCESSORS, ids=BOUND_ACCESSOR_IDS
)
def test_bounds_cannot_be_thawed_through_their_base(
    variables, get_result, dict_key
) -> None:
    """Check that no array reachable from a bound can be made writeable."""
    bounds = Bounds(variables)

    result = get_result(bounds)
    if dict_key is not None:
        result = result[dict_key]

    array_ = result
    while isinstance(array_, ndarray):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            array_.setflags(write=True)

        array_ = array_.base

    assert isinstance(array_, bytes)
    assert_array_equal(bounds.get_lower_bounds(), [0.0, 0.0])
    assert_array_equal(bounds.get_upper_bounds(), [10.0, 10.0])
    assert_array_equal(variables["x"].lower_bound, [0.0, 0.0])
    assert_array_equal(variables["x"].upper_bound, [10.0, 10.0])


@pytest.mark.parametrize(
    ("get_result", "dict_key"), BOUND_ACCESSORS, ids=BOUND_ACCESSOR_IDS
)
@pytest.mark.parametrize(
    ("attribute_name", "value"),
    [("shape", (2, 1)), ("strides", (0,)), ("dtype", int32)],
)
def test_reassigning_bound_attributes_leaves_the_bounds_alone(
    variables, get_result, dict_key, attribute_name, value
) -> None:
    """Check that a bound is handed out as a view of what the bounds store.

    NumPy lets a caller reassign the shape, the strides and the data type
    of a read-only array, so a bound accessor returns a view of the frozen bound
    and such a reassignment reaches the view only.
    """
    bounds = Bounds(variables)

    result = get_result(bounds)
    if dict_key is not None:
        result = result[dict_key]

    setattr(result, attribute_name, value)

    for bound in (
        bounds.full_lower_bound,
        bounds.full_upper_bound,
        bounds.get_lower_bound("x"),
        bounds.get_upper_bound("x"),
        variables["x"].lower_bound,
        variables["x"].upper_bound,
    ):
        assert bound.shape == (2,)
        assert bound.strides == (8,)
        assert bound.dtype == float64


def test_reassigning_variable_bound_attributes_leaves_the_bounds_alone(
    variables,
) -> None:
    """Check that a bound read from a variable is a view of what the variable stores.

    The full bounds are rebuilt from the bounds of the variables,
    so a reassignment reaching a variable would corrupt every bound of the registry.
    """
    bounds = Bounds(variables)
    variables["x"].lower_bound.shape = (2, 1)
    variables["y"] = RealVariable(size=1, lower_bound=1.0, upper_bound=2.0)

    assert_array_equal(bounds.full_lower_bound, [0.0, 0.0, 1.0])
    assert_array_equal(bounds.get_lower_bound("x"), [0.0, 0.0])


def test_bounds_are_read_only_after_unpickling(variables) -> None:
    """Check that the full bounds are still read-only after a pickle round-trip.

    Pickling does not preserve the writeable flag,
    so the restored cache must be rebuilt before being handed out.
    """
    bounds = Bounds(variables)
    # Warm the cache so that a writeable copy of it is pickled.
    assert_array_equal(bounds.full_lower_bound, [0.0, 0.0])

    restored = pickle.loads(pickle.dumps(bounds))

    assert not restored.full_lower_bound.flags.writeable
    assert not restored.full_upper_bound.flags.writeable
    assert not restored.get_lower_bounds().flags.writeable
    assert not restored.get_upper_bounds().flags.writeable
    assert_array_equal(restored.full_lower_bound, [0.0, 0.0])
    assert_array_equal(restored.full_upper_bound, [10.0, 10.0])

    # The guard was reset, not disabled: a later mutation still refreshes the cache.
    restored.set_lower_bound("x", array([1.0, 2.0]))
    assert_array_equal(restored.full_lower_bound, [1.0, 2.0])
