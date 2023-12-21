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
"""Test models creators."""

from typing import Union

from numpy import array
from pydantic import BaseModel
from pydantic import Field

from gemseo.core.grammars.pydantic_grammar import ModelType
from gemseo.core.grammars.pydantic_ndarray import NDArrayPydantic


def get_model1() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        name1: int
        name2: NDArrayPydantic[int] = Field(default_factory=lambda: array([0]))

    return Model


def get_model2() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        name1: int
        name2: Union[int, str] = 0

    return Model


def get_model3() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        an_int: int
        a_float: float
        a_bool: bool
        an_int_ndarray: NDArrayPydantic[int]
        a_float_ndarray: NDArrayPydantic[float]
        a_bool_ndarray: NDArrayPydantic[bool]
        an_int_list: list[int]
        a_float_list: list[float]
        a_bool_list: list[bool]

    return Model


def get_model4() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        name1: NDArrayPydantic[int]
        name2: NDArrayPydantic
        name3: NDArrayPydantic

    return Model
