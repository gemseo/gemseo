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

from functools import cache
from typing import TYPE_CHECKING

from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.core.grammars._python_to_json import PYTHON_TO_JSON_TYPES

if TYPE_CHECKING:
    from gemseo.core.grammars.json_grammar import JSONGrammar  # noqa: F401


class JSONGrammarDataConverter(BaseDataConverter["JSONGrammar"]):
    """Data values to NumPy arrays and vice versa from a :class:`.JSONGrammar`."""

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
        return tuple(PYTHON_TO_JSON_TYPES.get(type_, type_) for type_ in types)

    def _has_type(self, name: str, types: tuple[str, ...]) -> bool:
        types = self.__convert_types(types)
        prop = self._grammar.schema["properties"][name]
        type_ = prop.get("type")
        if type_ not in types:
            return False
        if type_ != "array":
            return True
        sub_prop = prop.get("items")
        if sub_prop is None:
            # If the sub_prob is not defined, we assume that it is a numeric value
            # TODO: Keep that behavior?
            return True
        return sub_prop.get("type") in types
