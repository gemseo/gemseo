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
from numpy.testing import assert_array_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space.design._bounds import Bounds
from gemseo.space.design._variables import Variables
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def variables() -> Variables:
    """A variables with a single float variable of size 2, bounds [0, 10]."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=10.0)
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


@pytest.mark.parametrize(
    ("get_result", "dict_key"),
    [
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
    ],
    ids=[
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
    ],
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
    "get_result",
    [
        lambda bounds: bounds.full_lower_bound,
        lambda bounds: bounds.full_upper_bound,
        lambda bounds: bounds.get_lower_bound("x"),
        lambda bounds: bounds.get_upper_bound("x"),
        lambda bounds: bounds.get_lower_bounds(),
        lambda bounds: bounds.get_upper_bounds(),
    ],
    ids=[
        "full_lower_bound",
        "full_upper_bound",
        "get_lower_bound",
        "get_upper_bound",
        "get_lower_bounds",
        "get_upper_bounds",
    ],
)
def test_read_only_bounds_cannot_be_unfrozen(variables, get_result, snapshot) -> None:
    """Check that the writeable flag of a bound cannot be re-enabled.

    The accessors hand out views,
    which do not own their data,
    so NumPy refuses to make them writeable again;
    freezing the arrays alone would not be enough,
    since they own their data.
    """
    bounds = Bounds(variables)

    with assert_exception(ValueError, snapshot):
        get_result(bounds).setflags(write=True)


def test_full_bounds_cache_is_not_handed_out(variables) -> None:
    """Check that the cached full bounds are never handed out by identity.

    Otherwise a caller could reach the cache
    and corrupt it for every later reader.
    """
    bounds = Bounds(variables)
    cached_lower_bound = bounds._Bounds__full_lower_bound
    cached_upper_bound = bounds._Bounds__full_upper_bound

    for result in (
        bounds.full_lower_bound,
        bounds.get_lower_bounds(),
        bounds.full_upper_bound,
        bounds.get_upper_bounds(),
    ):
        assert result is not cached_lower_bound
        assert result is not cached_upper_bound

    assert_array_equal(bounds.get_lower_bounds(), [0.0, 0.0])
    assert_array_equal(bounds.get_upper_bounds(), [10.0, 10.0])


def test_bounds_are_read_only_after_unpickling(variables) -> None:
    """Check that the full bounds are still read-only after a pickle round-trip.

    NumPy does not preserve the writeable flag,
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
