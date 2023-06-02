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
from typing import List
from typing import Union

from numpy import array
from numpy import ndarray
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import Field

from gemseo.core.grammars.pydantic_grammar import ModelType


def get_model1() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        name1: int
        name2: NDArray[int] = Field(default_factory=lambda: array([0]))

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
        name1: int
        name2: NDArray[int]
        name3: List[int]

    return Model


def get_model4() -> ModelType:
    """Return a pydantic model."""

    class Model(BaseModel):
        name1: NDArray[int]
        name2: NDArray
        name3: ndarray

    return Model
