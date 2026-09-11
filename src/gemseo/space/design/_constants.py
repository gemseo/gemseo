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

from gemseo.space.variable.base import _lower_bound
from gemseo.space.variable.base import _upper_bound
from gemseo.space.variable.discrete import _choices
from gemseo.util.constant import epsilon

bound_atol: Final[float] = 100.0 * epsilon
"""The absolute tolerance for a deviation from a bound."""

_design_space_group: Final[str] = "design_space"
"""The name of the HDF group storing a design space."""

_names_group: Final[str] = "names"
"""The name of the HDF dataset storing the variable names."""

_lb_group: Final[str] = "l_b"
"""The name of the HDF dataset storing a variable lower bound."""

_ub_group: Final[str] = "u_b"
"""The name of the HDF dataset storing a variable upper bound."""

_var_type_group: Final[str] = "var_type"
"""The name of the HDF dataset storing a variable type."""

_value_group: Final[str] = "value"
"""The name of the HDF dataset storing a variable value."""

_size_group: Final[str] = "size"
"""The name of the HDF dataset storing a variable size."""

_choices_group: Final[str] = _choices
"""The name of the HDF dataset storing the choices of a variable.

This is also the name of the CSV column.
"""

_choices_separator: Final[str] = "|"
"""The string separating the choices within a CSV cell.

A CSV file is exported with a space delimiter by default,
so the separator shall not be a whitespace.
"""

_table_names: Final[tuple[str, ...]] = (
    "name",
    _lower_bound,
    "value",
    _upper_bound,
    "type",
)
"""The fields of the tabular view of a design space."""
