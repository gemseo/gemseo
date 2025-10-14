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
"""Data values to NumPy arrays and vice versa from a :class:`.PydanticGrammar`."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING
from typing import get_args
from typing import get_origin

from numpy import ndarray

from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic
from gemseo.utils.pydantic_ndarray import _ScalarType_co

if TYPE_CHECKING:
    from gemseo.core.grammars.pydantic_grammar import PydanticGrammar  # noqa: F401


class PydanticGrammarDataConverter(BaseDataConverter["PydanticGrammar"]):
    """Data values to NumPy arrays and vice versa from a :class:`.PydanticGrammar`."""

    @staticmethod
    @cache
    def __convert_types(types: tuple[type, ...]) -> tuple[type, ...]:
        """Convert from python types to json types.

        This method is cached for performance.

        Args:
            types: The types to be converted.

        Returns:
            The converted types.
        """
        if ndarray in types:
            return (*types, _ScalarType_co)
        return types

    def _has_type(  # noqa:D102
        self,
        name: str,
        types: tuple[type, ...],
    ) -> bool:
        types = self.__convert_types(types)
        annotation = self._grammar[name].annotation

        if annotation in types or annotation is _NDArrayPydantic:
            return True

        type_origin = get_origin(annotation)

        if type_origin is not _NDArrayPydantic:
            return False

        # This is X in NDArray[X].
        dtype = get_args(get_args(annotation)[1])[0]

        return dtype in types
