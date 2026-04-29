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
"""Data values to NumPy arrays and vice versa from a [PydanticGrammar][gemseo.core.grammars.pydantic.PydanticGrammar]."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import get_args
from typing import get_origin

from numpy import inexact
from numpy import integer as np_integer
from numpy import issubdtype
from numpy import ndarray

from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic
from gemseo.utils.pydantic_ndarray import _ScalarType_co

if TYPE_CHECKING:
    from gemseo.core.grammars.pydantic import PydanticGrammar  # noqa: F401


class PydanticGrammarDataConverter(BaseDataConverter["PydanticGrammar"]):
    """Data values to NumPy arrays and vice versa from a [PydanticGrammar][gemseo.core.grammars.pydantic.PydanticGrammar]."""  # noqa: E501

    def _has_type(  # noqa:D102
        self,
        name: str,
        types: tuple[type, ...],
    ) -> bool:
        annotation = self._grammar[name].annotation

        if annotation in types or annotation is _NDArrayPydantic:
            return True

        type_origin = get_origin(annotation)

        if type_origin is not _NDArrayPydantic:
            return False

        if ndarray not in types:
            return False

        # annotation is _NDArrayPydantic[Any, dtype[X]], extract X.
        dtype = get_args(get_args(annotation)[1])[0]

        if dtype is _ScalarType_co:
            # Unparameterized ndarray: no dtype constraint, matches any ndarray type.
            return True

        if issubdtype(dtype, inexact):
            # Float or complex dtype: matches any continuous scalar type.
            return any(
                t in types for t in self._IS_CONTINUOUS_TYPES if t is not ndarray
            )

        if issubdtype(dtype, np_integer):
            # Integer dtype: matches any numeric-but-not-continuous scalar type.
            return any(
                t in types
                for t in self._IS_NUMERIC_TYPES
                if t not in self._IS_CONTINUOUS_TYPES
            )

        return False
