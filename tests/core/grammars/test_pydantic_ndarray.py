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
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Union

import pytest
from numpy import array
from pydantic import ValidationError
from pydantic import create_model

from gemseo.core.grammars.pydantic_ndarray import NDArrayPydantic
from gemseo.core.grammars.pydantic_ndarray import _NDArrayPydantic


class Data(NamedTuple):
    """Helper for naming data."""

    valid: Any
    invalid: tuple


# Test data with type annotations bound to valid and invalid data.
TYPES_TO_VALUES = {
    _NDArrayPydantic: Data(array([0]), (0, 0.0, "0", False)),
    NDArrayPydantic: Data(array([0]), (0, 0.0, "0", False)),
    NDArrayPydantic[Any]: Data(array([0]), (0, 0.0, "0", False)),
    NDArrayPydantic[int]: Data(
        array([0]),
        (
            array([0.0]),
            array([False]),
        ),
    ),
    NDArrayPydantic[float]: Data(
        array([0.0]),
        (
            array([0]),
            array([False]),
        ),
    ),
    # The following verifies Unions, Callable is just not a type of the data item.
    Union[_NDArrayPydantic, Callable]: Data(array([0]), (0, 0.0, "0", False)),
    Union[NDArrayPydantic, Callable]: Data(array([0]), (0, 0.0, "0", False)),
    Union[NDArrayPydantic[Any], Callable]: Data(array([0]), (0, 0.0, "0", False)),
    Union[NDArrayPydantic[int], Callable]: Data(
        array([0]),
        (
            array([0.0]),
            array([False]),
        ),
    ),
    Union[NDArrayPydantic[float], Callable]: Data(
        array([0.0]),
        (
            array([0]),
            array([False]),
        ),
    ),
}


@pytest.mark.parametrize(("type_", "data"), TYPES_TO_VALUES.items())
def test_valid_data(type_, data):
    """Verify valid models built from annotate or the base class."""
    model = create_model("Model", name=(type_, ...))
    model(name=data.valid)


@pytest.mark.parametrize(("type_", "data"), TYPES_TO_VALUES.items())
def test_invalid_data(type_, data):
    """Verify invalid models built from annotate or the base class."""
    model = create_model("Model", name=(type_, ...))
    for invalid_data in data.invalid:
        with pytest.raises(ValidationError):
            model(name=invalid_data)
