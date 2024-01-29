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
"""Data values to NumPy arrays and vice versa from a :class:`.JSONGrammar`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.data_converters.base import BaseDataConverter

if TYPE_CHECKING:
    from gemseo.core.grammars.json_grammar import JSONGrammar  # noqa: F401
    from gemseo.core.grammars.json_schema import Property
    from gemseo.typing import NumberArray


class JSONGrammarDataConverter(BaseDataConverter["JSONGrammar"]):
    """Data values to NumPy arrays and vice versa from a :class:`.JSONGrammar`."""

    _IS_CONTINUOUS_TYPES: ClassVar[tuple[str, ...]] = ("number",)
    _IS_NUMERIC_TYPES: ClassVar[tuple[str, ...]] = (
        *_IS_CONTINUOUS_TYPES,
        "integer",
    )

    def _has_type(self, name: str, types: tuple[str, ...]) -> bool:
        prop = self.__get_property(name)
        type_ = prop.get("type")
        if type_ == "array":
            return self.__is_collection_of_numbers(prop, types)
        return type_ in types

    @classmethod
    def __is_collection_of_numbers(cls, prop: Any, types: tuple[str, ...]) -> bool:
        """Whether the property contains numeric values.

        This method is recursive in order to be able to take into account nested arrays.

        Args:
            prop: The grammar property.
            types: The names of the expected number type.

        Returns:
            Whether the property contains numeric values at the end.
        """
        sub_prop = prop.get("items")
        if sub_prop is None:
            # If the sub_prob is not defined, we assume that it is a numeric value
            # TODO: Keep that behavior?
            return True
        sub_prop_type = sub_prop.get("type")
        if sub_prop_type == "array":
            return cls.__is_collection_of_numbers(sub_prop, types)
        return sub_prop.get("type") in types

    def _convert_array_to_value(self, name: str, array: NumberArray) -> Any:  # noqa: D102
        if self.__get_property(name).get("type") == "array":
            return array
        return array[0]

    def __get_property(self, name: str) -> Property:
        """Return a property of a schema given its name."""
        return self._grammar.schema["properties"][name]
