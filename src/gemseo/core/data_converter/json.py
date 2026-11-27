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
"""Data values to NumPy arrays and vice versa from a [JSONGrammar][gemseo.core.grammar.json.JSONGrammar]."""  # noqa: E501

from __future__ import annotations

from functools import cache
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.data_converter.base import BaseDataConverter
from gemseo.core.grammar._python_to_json import python_to_json_types

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.grammar.json import JSONGrammar  # noqa: F401


class JSONGrammarDataConverter(BaseDataConverter["JSONGrammar"]):
    """Data values to NumPy arrays and vice versa from a [JSONGrammar][gemseo.core.grammar.json.JSONGrammar]."""  # noqa: E501

    @staticmethod
    @cache
    def __convert_types(types: tuple[type, ...]) -> tuple[str | type, ...]:
        """Convert from python types to json types.

        The path types are left out, because a JSONGrammar does not support them.

        This method is cached for performance.

        Args:
            types: The types to be converted.

        Returns:
            The converted types.
        """
        return tuple(
            python_to_json_types.get(type_, type_)
            for type_ in types
            # JSONGrammar does not support path types.
            if not (isinstance(type_, type) and issubclass(type_, PurePath))
        )

    def _has_type(self, name: str, types: tuple[type, ...]) -> bool:
        return self.__has_property_type(name, types, False)

    def is_continuous(self, name: str) -> bool:  # noqa: D102
        return self.__has_property_type(name, self._is_continuous_types, True)

    def __has_property_type(
        self, name: str, types: tuple[type, ...], ignore_null: bool
    ) -> bool:
        """Return whether the type of a grammar item is one of the expected types.

        Args:
            name: The name of the grammar item.
            types: The expected types.
            ignore_null: Whether to ignore a `"null"` alternative of the type,
                e.g. `["number", "null"]` is then read as `"number"`.

        Returns:
            Whether the type of the grammar item is one of the expected types.
        """
        json_types = self.__convert_types(types)
        prop = self._grammar.schema["properties"][name]
        type_ = _get_type(prop, ignore_null)
        if type_ not in json_types:
            return False
        if type_ != "array":
            return True
        sub_prop = prop.get("items")
        if sub_prop is None:
            # If the sub_prob is not defined, we assume that it is a numeric value
            # TODO: Keep that behavior?
            return True
        return _get_type(sub_prop, ignore_null) in json_types


def _get_type(prop: Mapping[str, Any], ignore_null: bool) -> str | list[str] | None:
    """Return the type of a JSON schema property.

    Args:
        prop: The JSON schema property.
        ignore_null: Whether to ignore a `"null"` alternative of the type,
            e.g. `["number", "null"]` is then read as `"number"`.

    Returns:
        The type of the property.
    """
    type_: str | list[str] | None = prop.get("type")
    if ignore_null and isinstance(type_, list):
        types = [item for item in type_ if item != "null"]
        if len(types) == 1:
            return types[0]

    return type_
