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
from typing import ClassVar

from typing_extensions import get_args
from typing_extensions import get_origin

from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic
from gemseo.utils.pydantic_ndarray import _ScalarType_co

if TYPE_CHECKING:
    from gemseo.core.grammars.pydantic_grammar import PydanticGrammar  # noqa: F401
    from gemseo.typing import NumberArray


class PydanticGrammarDataConverter(BaseDataConverter["PydanticGrammar"]):
    """Data values to NumPy arrays and vice versa from a :class:`.PydanticGrammar`."""

    # For consistency with the other grammars, a NumPy array that has no dtype info is
    # considered to be continuous, hence _ScalarType_co here.
    _IS_CONTINUOUS_TYPES: ClassVar[tuple[type, type, object]] = (
        float,
        complex,
        _ScalarType_co,
    )
    _IS_NUMERIC_TYPES: ClassVar[tuple[type, type, type, object]] = (
        int,
        *_IS_CONTINUOUS_TYPES,
    )

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

        # This is X in NDArray[X].
        dtype = get_args(get_args(annotation)[1])[0]

        return dtype in types

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
    #     return type_arg in _NON_ARRAY_TYPES

    def _convert_array_to_value(self, name: str, array: NumberArray) -> Any:  # noqa: D102
        if self._grammar[name].annotation in self._NON_ARRAY_TYPES:
            return array[0]
        return array
