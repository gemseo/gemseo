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
"""Constants shared across the design space package."""

from __future__ import annotations

from typing import Final

from gemseo.space._variable._base import _LOWER_BOUND
from gemseo.space._variable._base import _UPPER_BOUND
from gemseo.util.constant import EPSILON

BOUND_ATOL: Final[float] = 100.0 * EPSILON
"""The absolute tolerance for a deviation from a bound."""

_DESIGN_SPACE_GROUP: Final[str] = "design_space"
"""The name of the HDF group storing a design space."""

_NAMES_GROUP: Final[str] = "names"
"""The name of the HDF dataset storing the variable names."""

_LB_GROUP: Final[str] = "l_b"
"""The name of the HDF dataset storing a variable lower bound."""

_UB_GROUP: Final[str] = "u_b"
"""The name of the HDF dataset storing a variable upper bound."""

_VAR_TYPE_GROUP: Final[str] = "var_type"
"""The name of the HDF dataset storing a variable type."""

_VALUE_GROUP: Final[str] = "value"
"""The name of the HDF dataset storing a variable value."""

_SIZE_GROUP: Final[str] = "size"
"""The name of the HDF dataset storing a variable size."""

_TABLE_NAMES: Final[list[str]] = [
    "name",
    _LOWER_BOUND,
    "value",
    _UPPER_BOUND,
    "type",
]
"""The fields of the tabular view of a design space."""
