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
"""Tests for the factory of variables."""

from __future__ import annotations

import pytest
from numpy.testing import assert_array_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space._variable import DataType
from gemseo.space._variable import IntegerVariable
from gemseo.space._variable import VariableFactory
from gemseo.util.pydantic import BaseSettings
from gemseo.util.testing.helper import assert_exception

pytestmark = pytest.mark.usefixtures("reset_factory")


@pytest.fixture
def factory() -> VariableFactory:
    """The factory of variables."""
    return VariableFactory()


def test_create_from_settings_not_implemented(factory) -> None:
    """Check `create_from_settings` raises error."""
    with pytest.raises(NotImplementedError):
        factory.create_from_settings(BaseSettings())


def test_class_names(factory) -> None:
    """Check that the concrete kinds are discovered and the abstract base is not."""
    assert factory.class_names == ["ContinuousVariable", "IntegerVariable"]


def test_singleton(factory) -> None:
    """Check that the factory is a singleton."""
    assert VariableFactory() is factory


def test_create(factory) -> None:
    """Check the creation of a variable from its class name."""
    variable = factory.create("integer", size=2, lower_bound=0, upper_bound=10)
    assert isinstance(variable, IntegerVariable)
    assert variable.size == 2
    assert_array_equal(variable.lower_bound, [0, 0])
    assert_array_equal(variable.upper_bound, [10, 10])


@pytest.mark.parametrize(
    ("data_type", "cls"),
    [
        (DataType.FLOAT, ContinuousVariable),
        (DataType.INTEGER, IntegerVariable),
        ("float", ContinuousVariable),
        ("integer", IntegerVariable),
        (b"float", ContinuousVariable),
        (b"integer", IntegerVariable),
    ],
)
def test_create_kind(factory, data_type, cls) -> None:
    """Check the resolution of a data type to the kind pinning it."""
    variable = factory.create(data_type, size=2)
    assert isinstance(variable, cls)
    assert variable.size == 2


def test_create_for_unknown_variable_type(factory, snapshot) -> None:
    """Check that an unknown data type raises."""
    with assert_exception(ValueError, snapshot):
        factory.create("complex")


def test_update_invalidates_the_data_types(factory, monkeypatch) -> None:
    """Check that rediscovering the classes rebuilds the data-type map."""

    class OtherIntegerVariable(IntegerVariable):
        """Another variable class pinning the integer data type."""

    # Build the map from the classes discovered so far.
    assert isinstance(factory.create("integer"), IntegerVariable)

    # Rediscover the classes with the integer data type pinned by another class.
    class_names = ["ContinuousVariable", "OtherIntegerVariable"]
    # IntegerVariable is still resolvable by name, but no longer discovered,
    # so that a stale map fails the assertion below instead of raising.
    name_to_class = {
        "ContinuousVariable": ContinuousVariable,
        "IntegerVariable": IntegerVariable,
        "OtherIntegerVariable": OtherIntegerVariable,
    }
    monkeypatch.setattr(
        VariableFactory, "class_names", property(lambda self: class_names)
    )
    monkeypatch.setattr(
        VariableFactory, "get_class", lambda self, name: name_to_class[name]
    )
    factory.update()

    # Without the invalidation, the stale map would still resolve to IntegerVariable.
    variable = factory.create("integer")
    assert isinstance(variable, OtherIntegerVariable)


def test_duplicate_data_type(factory, monkeypatch, snapshot) -> None:
    """Check that two classes pinning the same data type raise."""

    class OtherContinuousVariable(ContinuousVariable):
        """Another variable class pinning the float data type."""

    name_to_class = {
        "ContinuousVariable": ContinuousVariable,
        "OtherContinuousVariable": OtherContinuousVariable,
    }
    monkeypatch.setattr(
        VariableFactory, "class_names", property(lambda self: list(name_to_class))
    )
    monkeypatch.setattr(
        VariableFactory, "get_class", lambda self, name: name_to_class[name]
    )
    with assert_exception(ValueError, snapshot):
        factory.create("float")
