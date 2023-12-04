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

import re
from typing import Any
from typing import NamedTuple

import pytest
from numpy import array
from numpy import ndarray
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import create_model

from gemseo.core.grammars.pydantic_ndarray import BaseModelWithNDArray
from gemseo.core.grammars.pydantic_ndarray import annotate


class Data(NamedTuple):
    """Helper for naming data."""

    valid: Any
    invalid: tuple


# Test data with type annotations bound to valid and invalid data.
TYPES_TO_VALUES = {
    ndarray: Data(array([0]), (0, 0.0, "0", False)),
    NDArray: Data(array([0]), (0, 0.0, "0", False)),
    NDArray[Any]: Data(array([0]), (0, 0.0, "0", False)),
    NDArray[int]: Data(
        array([0]),
        (
            array([0.0]),
            array([False]),
        ),
    ),
    NDArray[float]: Data(
        array([0.0]),
        (
            array([0]),
            array([False]),
        ),
    ),
}


def test_annotate_error():
    """Verify annotate error."""
    match = (
        "Unable to generate a schema for <class 'int'>. "
        "It shall be a NDArray based type"
    )

    with pytest.raises(TypeError, match=re.escape(match)):
        annotate(int)


@pytest.mark.parametrize(("type_", "data"), TYPES_TO_VALUES.items())
@pytest.mark.parametrize(
    ("base", "annotate"), [(BaseModel, annotate), (BaseModelWithNDArray, lambda x: x)]
)
def test_valid_data(base, annotate, type_, data):
    """Verify valid models built from annotate or the base class."""
    model = create_model(
        "Model",
        __base__=base,
        name=(annotate(type_), ...),
    )

    model(name=data.valid)


@pytest.mark.parametrize(("type_", "data"), TYPES_TO_VALUES.items())
@pytest.mark.parametrize(
    ("base", "annotate"), [(BaseModel, annotate), (BaseModelWithNDArray, lambda x: x)]
)
def test_invalid_data(base, annotate, type_, data):
    """Verify invalid models built from annotate or the base class."""
    model = create_model(
        "Model",
        __base__=base,
        name=(annotate(type_), ...),
    )

    for invalid_data in data.invalid:
        with pytest.raises(ValidationError):
            model(name=invalid_data)
