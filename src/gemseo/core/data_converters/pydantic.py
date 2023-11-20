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

from typing import Any
from typing import Final

from numpy import ndarray

from gemseo.core.data_converters.base import _NUMERIC_TYPES
from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.utils.compatibility.python import PYTHON_VERSION
from gemseo.utils.compatibility.python import get_args
from gemseo.utils.compatibility.python import get_origin


class PydanticGrammarDataConverter(BaseDataConverter):
    """Data values to NumPy arrays and vice versa from a :class:`.PydanticGrammar`."""

    __NUMERIC_TYPES: Final[set[type]] = set(_NUMERIC_TYPES)

    if PYTHON_VERSION < (3, 9):  # pragma: >=3.9 no cover

        def is_numeric(  # noqa:D102
            self,
            name: str,
        ) -> bool:
            outer_type_ = self._grammar[name].outer_type_

            if outer_type_ in _NUMERIC_TYPES:
                return True

            type_origin = get_origin(outer_type_)

            if type_origin is None:
                # Workaround for ndarray in particular,
                # see get_origin of Python 3.9+.
                type_origin = getattr(outer_type_, "__origin__", None)

            if type_origin is not ndarray:
                return False

            # This is X in NDArray[X].
            dtype = outer_type_.__args__[1].__args__[0]

            # This is for instance Y,Z when dtype=Y|Z,
            # this is empty when there is no union.
            try:
                dtype_types = dtype.__args__
            except AttributeError:
                dtype_types = get_args(dtype)

            if not dtype_types:
                dtype_types = {dtype}

            return self.__NUMERIC_TYPES.issuperset(dtype_types)

    else:  # pragma: <3.9 no cover

        def is_numeric(  # noqa:D102
            self,
            name: str,
        ) -> bool:
            outer_type_ = self._grammar[name].outer_type_

            if outer_type_ in _NUMERIC_TYPES:
                return True

            type_origin = get_origin(outer_type_)

            if type_origin is not ndarray:
                return False

            # This is X in NDArray[X].
            dtype = get_args(get_args(outer_type_)[1])[0]

            # This is for instance Y,Z when dtype=Y|Z,
            # this is empty when there is no union.
            dtype_types = get_args(dtype)

            if not dtype_types:
                dtype_types = {dtype}

            return self.__NUMERIC_TYPES.issuperset(dtype_types)

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
        if self._grammar[name].outer_type_ in _NUMERIC_TYPES:
            return array[0]
        return array
