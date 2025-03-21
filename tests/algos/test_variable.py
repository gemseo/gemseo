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

import re

import pytest
from numpy import array
from numpy import atleast_1d
from numpy import inf
from pydantic import ValidationError

from gemseo.algos._variable import DataType
from gemseo.algos._variable import Variable


def test_init_defaults():
    """Test the default values of __init__."""
    v = Variable()
    assert v.type == DataType.FLOAT
    assert v.size == 1
    assert v.lower_bound == -array([inf])
    assert v.upper_bound == array([inf])


@pytest.mark.parametrize(
    ("size", "type_", "lower_bound", "upper_bound"),
    [
        (1, DataType.FLOAT, 0, 0.0),
        (10, "integer", -5.0, 0.0),
        (1, "float", -5.0, inf),
        (1, "float", -inf, inf),
        (2, "float", (-4, 4), inf),
    ],
)
def test_initialization(
    size: int,
    type_: DataType,
    lower_bound: float | tuple[float, float],
    upper_bound: float,
) -> None:
    """Test the instantiation."""
    my_variable = Variable(
        size=size, type=type_, lower_bound=lower_bound, upper_bound=upper_bound
    )
    assert my_variable.size == size
    assert my_variable.type == type_
    assert (my_variable.lower_bound == atleast_1d(lower_bound)).all()
    assert (my_variable.upper_bound == atleast_1d(upper_bound)).all()


@pytest.mark.parametrize("size", [-1, 0])
def test_non_positive_size(size) -> None:
    """Check non-positive variables size."""
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Variable(size=size)


def test_invalid_type() -> None:
    """Check invalid variable type."""
    with pytest.raises(ValidationError, match="Input should be 'float' or 'integer'"):
        Variable(type="complex")


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_size(side) -> None:
    """Check invalid bound size."""
    with pytest.raises(
        ValidationError, match=re.escape(f"The {side} bound should be of size 1.")
    ):
        Variable(**{f"{side}_bound": [0, 0]})


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_value_scalar(side) -> None:
    """Check invalid bound value type."""
    with pytest.raises(
        ValidationError,
        match="validation errors for Variable",
    ):
        Variable(**{f"{side}_bound": 1j})


@pytest.mark.parametrize(
    ("size", "type_", "lower_bound", "upper_bound"),
    [(1, DataType.FLOAT, 0, -1.0), (1, "integer", 0, -1)],
)
def test_wrong_boundaries(
    size: int, type_: str, lower_bound: float, upper_bound: float
) -> None:
    """Test the instantiation with ``upper_bound`` lower than ``lower_bound``."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The upper bounds must be greater than or equal to the lower bounds."
        ),
    ):
        Variable(
            size=size, type=type_, lower_bound=lower_bound, upper_bound=upper_bound
        )


@pytest.fixture
def variable() -> Variable:
    """A variable."""
    return Variable(size=1, type="float", lower_bound=0, upper_bound=1)


def test_invalid_lower_bound_assignment(variable) -> None:
    """Check the handling of an invalid lower bound assignment."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The upper bounds must be greater than or equal to the lower bounds."
        ),
    ):
        variable.lower_bound = 5


def test_invalid_upper_bound_assignment(variable) -> None:
    """Check the handling of an invalid upper bound assignment."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The upper bounds must be greater than or equal to the lower bounds."
        ),
    ):
        variable.upper_bound = -1
