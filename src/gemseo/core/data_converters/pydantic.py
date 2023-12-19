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

from typing import TYPE_CHECKING
from typing import Any

from typing_extensions import get_args
from typing_extensions import get_origin

from gemseo.core.data_converters.base import _NUMERIC_TYPES
from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.core.grammars.pydantic_ndarray import _NDArrayPydantic

if TYPE_CHECKING:
    from numpy import ndarray


class PydanticGrammarDataConverter(BaseDataConverter):
    """Data values to NumPy arrays and vice versa from a :class:`.PydanticGrammar`."""

    def is_numeric(  # noqa:D102
        self,
        name: str,
    ) -> bool:
        annotation = self._grammar[name].annotation

        if annotation in _NUMERIC_TYPES or annotation is _NDArrayPydantic:
            return True

        type_origin = get_origin(annotation)

        if type_origin is not _NDArrayPydantic:
            return False

        # This is X in NDArray[X].
        dtype = get_args(get_args(annotation)[1])[0]

        return dtype in _NUMERIC_TYPES

    # @classmethod
    # def __is_collection_of_numbers(cls, type_: type) -> bool:
    #     """Whether the array (which can be nested) contains numeric values at the end.
    #
    #     This method is recursive to be able to take into account nested arrays.
    #
    #     Args:
    #         type_: The type arguments to check.
    #
    #     Returns:
    #         Whether the type contains numbers at the end.
    #     """
    #     type_args = get_args(type_)
    #
    #     if len(type_args) != 1:
    #         # Currently this does not handle mappings and types defined
    #         # with ndarray instead of NDArray.
    #         return False
    #
    #     type_arg = type_args[0]
    #
    #     if issubclass(type_arg, Collection):
    #         return cls.__is_collection_of_numbers(type_arg)
    #
    #     return type_arg in _NUMERIC_TYPES

    def _convert_array_to_value(self, name: str, array: ndarray) -> Any:  # noqa: D102
        if self._grammar[name].annotation in _NUMERIC_TYPES:
            return array[0]
        return array
