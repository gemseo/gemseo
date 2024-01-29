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
"""Data values to NumPy arrays and vice versa from a :class:`.SimpleGrammar`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import ndarray

from gemseo.core.data_converters.base import BaseDataConverter

if TYPE_CHECKING:
    from gemseo.core.grammars.simple_grammar import SimpleGrammar  # noqa: F401
    from gemseo.typing import NumberArray


class SimpleGrammarDataConverter(BaseDataConverter["SimpleGrammar"]):
    """Data values to NumPy arrays and vice versa from a :class:`.SimpleGrammar`.

    .. warning::

        Since :class:`.SimpleGrammar` cannot make a distinction between the types of
        data in a NumPy array, it is assumed that those types are numeric and can
        differentiate. You may use another type of grammar if the distinction is needed.
    """

    _IS_CONTINUOUS_TYPES: ClassVar[tuple[type, ...]] = (float, complex)
    _IS_NUMERIC_TYPES: ClassVar[tuple[type, ...]] = (int, *_IS_CONTINUOUS_TYPES)

    def _has_type(self, name: str, types: tuple[type, ...]) -> bool:  # noqa: D102
        element_type = self._grammar[name]
        return element_type is not None and (
            issubclass(element_type, ndarray) or element_type in types
        )

    def _convert_array_to_value(self, name: str, array: NumberArray) -> Any:  # noqa: D102
        if self._grammar[name] in self._NUMERIC_TYPES:
            return array[0]
        return array
