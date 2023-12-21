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

from typing import Any

from numpy import ndarray

from gemseo.core.data_converters.base import _NUMERIC_TYPES
from gemseo.core.data_converters.base import BaseDataConverter


class SimpleGrammarDataConverter(BaseDataConverter):
    """Data values to NumPy arrays and vice versa from a :class:`.SimpleGrammar`."""

    def is_numeric(self, name: str) -> bool:  # noqa: D102
        element_type = self._grammar[name]
        return element_type is not None and (
            issubclass(element_type, ndarray) or element_type in _NUMERIC_TYPES
        )

    def _convert_array_to_value(self, name: str, array: ndarray) -> Any:  # noqa: D102
        if self._grammar[name] in _NUMERIC_TYPES:
            return array[0]
        return array
