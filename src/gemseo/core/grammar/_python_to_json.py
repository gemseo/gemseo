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

"""Provide type conversions from python to json."""

from __future__ import annotations

from numbers import Complex
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Mapping

python_to_json_types: Final[Mapping[type, str]] = MappingProxyType({
    ndarray: "array",
    list: "array",
    tuple: "array",
    str: "string",
    int: "integer",
    bool: "boolean",
    complex: "number",
    Complex: "number",
    float: "number",
})
