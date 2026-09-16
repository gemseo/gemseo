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
"""Tests for the descriptor reading a field holding a frozen array as a view."""

from __future__ import annotations

from typing import Any  # noqa: TC003

import pytest
from numpy import array
from numpy.testing import assert_array_equal
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError

from gemseo.space.variable._view_field import expose_fields_as_views
from gemseo.util._numpy import freeze_array
from gemseo.util.testing.helper import assert_exception


class Model(BaseModel):
    """A model with a field holding an array and a field holding an integer."""

    model_config = ConfigDict(frozen=True)

    values: Any = freeze_array(array([1.0, 2.0]))
    """The values."""

    count: Any = 2
    """The number of values."""


expose_fields_as_views(Model, "values", "count")


@pytest.fixture
def model() -> Model:
    """A model."""
    return Model()


def test_get_from_class(snapshot) -> None:
    """Check that reading the field from the class raises an AttributeError."""
    with assert_exception(AttributeError, snapshot):
        Model.values


def test_get_view(model) -> None:
    """Check that the field is read as a view of the array."""
    stored = model.__dict__["values"]
    values = model.values
    assert values is not stored
    assert values.base is stored
    assert_array_equal(values, stored)


def test_get_view_is_not_shared(model) -> None:
    """Check that reshaping the array read from the field does not reach the model."""
    values = model.values
    values.shape = (2, 1)
    assert model.values.shape == (2,)


def test_get_non_array(model) -> None:
    """Check that a field not holding an array is read as is."""
    assert model.count == 2


def test_get_unset_field(model, snapshot) -> None:
    """Check that reading a field that pydantic has not set raises an AttributeError."""
    del model.__dict__["values"]
    with assert_exception(AttributeError, snapshot):
        model.values


def test_assignment(model, snapshot) -> None:
    """Check that the frozen model refuses the assignment of the field."""
    with assert_exception(ValidationError, snapshot):
        model.values = array([3.0])


def test_set(model, snapshot) -> None:
    """Check that the descriptor is a data one refusing the assignment."""
    with assert_exception(AttributeError, snapshot):
        vars(Model)["values"].__set__(model, array([3.0]))


def test_expose_unknown_field(snapshot) -> None:
    """Check that only a field can be read as a view."""
    with assert_exception(ValueError, snapshot):
        expose_fields_as_views(Model, "foo")
