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
#
# Copyright 2024 Capgemini Engineering
# Created on 10/09/2024, 14:25
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A grammar that accepts dictionaries, and the associated data converter.

Used for tests on data converters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.data_converters.simple import SimpleGrammarDataConverter
from gemseo.core.grammars.simple_grammar import SimpleGrammar

if TYPE_CHECKING:
    from gemseo.core.data_converters.base import ValueType
    from gemseo.typing import NumberArray


class DictGrammar(SimpleGrammar):
    """A simple grammar that accepts dictionaries."""

    DATA_CONVERTER_CLASS = "DictDataConverter"


class DictDataConverter(SimpleGrammarDataConverter):
    """A converter that accepts dictionaries."""

    _MAPPING_TYPES: ClassVar[tuple[type, ...]] = (dict, Mapping)

    _IS_CONTINUOUS_TYPES: ClassVar[tuple[type, ...]] = (float, complex, *_MAPPING_TYPES)

    _IS_NUMERIC_TYPES: ClassVar[tuple[type, ...]] = (int, *_IS_CONTINUOUS_TYPES)

    def convert_value_to_array(  # noqa: D102
        self,
        name: str,
        value: ValueType,
    ) -> NumberArray:
        if isinstance(value, Mapping):
            return super().convert_data_to_array(list(value), value)
        return super().convert_value_to_array(name, value)

    def convert_array_to_value(self, name: str, array: NumberArray) -> ValueType:  # noqa: D102
        if self._grammar[name] is Mapping:
            return {name: array}
        return super().convert_array_to_value(name, array)

    def get_value_size(self, name: str, value: ValueType):  # noqa: D102
        if self._grammar[name] in self._MAPPING_TYPES:
            return len(value.items())
        return super().get_value_size(name, value)
